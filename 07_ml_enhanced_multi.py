#!/usr/bin/env python3
"""Enhanced ML for multispectral data: indices + polynomial features + MixUp + stacking."""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.feature_selection import pca_reduce, rf_importance_select
from analysis.ml_pipeline import (
    compare_models, stacking_ensemble, multi_output_gpr,
)
from analysis.preprocessing import mixup_augment, noise_augment
from analysis.explainability import plot_prediction_scatter
from sklearn.preprocessing import PolynomialFeatures


def build_multi_features(bands_df, cfg):
    """
    Строит наборы признаков из 5 мультиспектральных каналов.
    Возвращает dict: name -> (X array, feature_names list)
    """
    fs = {}
    B  = bands_df.values.astype(float)

    # 1. Сырые каналы (5 признаков)
    fs["bands"] = (B, list(bands_df.columns))

    # 2. Вегетативные индексы (44 признака)
    idx_df  = calculate_indices(bands_df)
    idx_arr = np.nan_to_num(idx_df.values.astype(float),
                            nan=0.0, posinf=0.0, neginf=0.0)
    fs["indices"] = (idx_arr, list(idx_df.columns))

    # 3. Каналы + индексы (49 признаков) — то же что 03_ml_multi
    fs["bands_idx"] = (
        np.hstack([B, idx_arr]),
        list(bands_df.columns) + list(idx_df.columns),
    )

    # 4. PCA индексов (до 10 компонент)
    try:
        nc   = min(10, idx_arr.shape[0] - 2, idx_arr.shape[1])
        Xpca, _ = pca_reduce(idx_arr, n_components=nc)
        fs["pca"] = (Xpca, [f"PC{i+1}" for i in range(Xpca.shape[1])])
    except Exception as e:
        print(f"    pca skip: {e}")

    # 5. Полиномиальные взаимодействия каналов degree=2
    # 5 каналов -> 15 признаков (сами + попарные произведения)
    try:
        poly   = PolynomialFeatures(degree=2, include_bias=False)
        Xpoly  = poly.fit_transform(B)
        pnames = [f"poly_{n}" for n in
                  poly.get_feature_names_out(list(bands_df.columns))]
        fs["poly"] = (Xpoly, pnames)
    except Exception as e:
        print(f"    poly skip: {e}")

    # 6. Комбинированный: индексы + poly + PCA
    parts, names = [], []
    for k in ("indices", "poly", "pca"):
        if k in fs:
            parts.append(fs[k][0])
            names.extend(fs[k][1])
    if parts:
        Xc = np.nan_to_num(np.hstack(parts).astype(float),
                           nan=0.0, posinf=0.0, neginf=0.0)
        fs["combined"] = (Xc, names)

    return fs


def run(args):
    cfg   = load_config(args.config)
    out   = args.output or "results/07_enhanced_multi"
    os.makedirs(out, exist_ok=True)

    cols   = cfg["chemistry"]["columns"]
    elems  = args.elements or cfg["chemistry"]["target_elements"]
    mdls   = args.models   or cfg["ml"]["models"]
    nf     = cfg["ml"]["cv_folds"]
    dsmap  = cfg["chemistry"]["date_sampling_map"]
    bmap   = cfg["camera"]["band_map"]
    dpi    = cfg["plots"]["dpi"]

    # PLSR исключён из MixUp — нестабилен на аугментированных данных
    mixup_mdls = [m for m in mdls if m != "PLSR"]

    res = []

    for dk in (args.date or ["date1", "date2"]):
        pattern = cfg["paths"]["multi"].get(dk)
        if not pattern:
            print(f"[{dk}] no multi path in config"); continue

        try:
            bands_df = load_multi(pattern, bmap)
        except FileNotFoundError as e:
            print(f"[{dk}] {e}"); continue

        sk = dsmap.get(dk, "sampling1")
        cp = cfg["paths"]["chemistry"].get(sk)
        if not cp or not Path(cp).exists():
            print(f"[{dk}] chemistry not found: {cp}"); continue

        off  = cfg["chemistry"]["id_offsets"].get(sk, 0)
        chem = load_chemistry(cp, cols, off)

        ids = bands_df.index.intersection(chem.index)
        if len(ids) < 15:
            print(f"[{dk}] too few common points: {len(ids)}"); continue

        bands_c = bands_df.loc[ids]
        chem_c  = chem.loc[ids]
        print(f"\n[{dk}] {len(ids)} точек, каналы: {list(bands_c.columns)}")

        fsets = build_multi_features(bands_c, cfg)
        print(f"  feature sets: "
              f"{', '.join(f'{k}({v[0].shape[1]})' for k, v in fsets.items())}")

        for el in elems:
            if el not in chem_c.columns:
                continue
            y    = chem_c[el].values.astype(float)
            nv   = int(np.isfinite(y).sum())
            if nv < 15:
                continue
            mask_y = np.isfinite(y)
            sn   = short_name(cfg, el)
            print(f"\n  {sn} (n={nv})")

            # RF-отбор топ-20 признаков из bands_idx под конкретный нутриент
            fsets_el = dict(fsets)
            try:
                X49  = fsets["bands_idx"][0]
                sel  = rf_importance_select(X49[mask_y], y[mask_y], n_select=20)
                fsets_el["rf_selected"] = (
                    X49[:, sel],
                    [fsets["bands_idx"][1][i] for i in sel],
                )
            except Exception:
                pass

            for fn, (X, feat_names) in fsets_el.items():
                X       = np.nan_to_num(X.astype(float),
                                        nan=0.0, posinf=0.0, neginf=0.0)
                nf_feat = X.shape[1]

                # A: стандартная кросс-валидация без аугментации
                rdf, pred = compare_models(X, y, mdls, n_splits=nf)
                for _, r in rdf.iterrows():
                    d = r.to_dict()
                    d.update(element=el, date=dk, feature_set=fn,
                             n_features=nf_feat, augmented="none")
                    res.append(d)

                top_r2  = rdf.iloc[0]["R2"]
                top_mdl = rdf.iloc[0]["model"]
                print(f"    {fn:>12}({nf_feat:>3}f): "
                      f"{top_mdl:<12} R2={top_r2:.3f}")

                # scatter для лучшей модели
                if top_mdl in pred:
                    fig = plot_prediction_scatter(
                        pred[top_mdl]["y_true"],
                        pred[top_mdl]["y_pred"],
                        top_mdl, sn,
                    )
                    fig.savefig(
                        f"{out}/scatter_{sn}_{fn}_{dk}.png",
                        dpi=dpi, bbox_inches="tight",
                    )
                    plt.close(fig)

                # B: MixUp (только для информативных наборов, без PLSR)
                if fn in ("indices", "bands_idx", "rf_selected",
                          "combined") and nv >= 20:
                    Xm, ym = mixup_augment(X[mask_y], y[mask_y], n_aug=nv)
                    rdf2, _ = compare_models(Xm, ym, mixup_mdls, n_splits=nf)
                    for _, r in rdf2.iterrows():
                        d = r.to_dict()
                        d.update(element=el, date=dk, feature_set=fn,
                                 n_features=nf_feat, augmented="mixup")
                        res.append(d)
                    aug_r2  = rdf2.iloc[0]["R2"]
                    aug_mdl = rdf2.iloc[0]["model"]
                    delta   = aug_r2 - top_r2
                    if abs(delta) > 0.01:
                        print(f"    {'':>12} +mixup: "
                              f"{aug_mdl:<12} R2={aug_r2:.3f} "
                              f"({delta:+.3f})")

                # C: noise augmentation для combined
                if fn == "combined" and nv >= 20:
                    Xn, yn = noise_augment(X[mask_y], y[mask_y], n_aug=nv)
                    rdf3, _ = compare_models(Xn, yn, mixup_mdls, n_splits=nf)
                    for _, r in rdf3.iterrows():
                        d = r.to_dict()
                        d.update(element=el, date=dk, feature_set=fn,
                                 n_features=nf_feat, augmented="noise")
                        res.append(d)

            # D: Stacking на combined
            if "combined" in fsets_el:
                Xc = np.nan_to_num(
                    fsets_el["combined"][0].astype(float),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                print(f"    {'stacking':>12}({Xc.shape[1]:>3}f)...", end=" ")
                sm = stacking_ensemble(Xc, y, n_splits=nf)
                if sm and np.isfinite(sm.get("R2", np.nan)):
                    print(f"R2={sm['R2']:.3f}")
                    res.append({
                        "model": "Stacking", "element": el, "date": dk,
                        "feature_set": "combined",
                        "n_features": Xc.shape[1],
                        "R2": sm["R2"], "RMSE": sm.get("RMSE"),
                        "RPD": sm.get("RPD"), "augmented": "none",
                    })
                else:
                    print("skip")

        # E: Multi-output GPR на PCA (все нутриенты сразу)
        if "pca" in fsets:
            Xpca  = fsets["pca"][0]
            avail = [e for e in elems if e in chem_c.columns]
            Y     = np.column_stack([chem_c[e].values for e in avail])
            ok    = np.all(np.isfinite(Y), axis=1)
            if ok.sum() >= 15:
                print(f"\n  Multi-output GPR "
                      f"({Xpca.shape[1]}f, {len(avail)} targets, "
                      f"n={ok.sum()})...", end=" ")
                try:
                    mo = multi_output_gpr(Xpca, Y, n_splits=nf)
                    print("done")
                    for j, e in enumerate(avail):
                        if j in mo:
                            m  = mo[j]
                            sn = short_name(cfg, e)
                            print(f"    {sn:>5}: R2={m['R2']:.3f}")
                            res.append({
                                "model": "MO-GPR", "element": e,
                                "date": dk, "feature_set": "pca",
                                "n_features": Xpca.shape[1],
                                "R2": m["R2"], "RMSE": m.get("RMSE"),
                                "RPD": m.get("RPD"), "augmented": "none",
                            })
                except Exception as e:
                    print(f"error: {e}")

    if not res:
        print("no results"); return

    df = pd.DataFrame(res)
    df.to_csv(f"{out}/enhanced_multi.csv", index=False, float_format="%.4f")

    # Итоговая сводка
    print(f"\n{'='*72}")
    print(f"{'elem':>6} | {'model':<14} | {'features':<13} | "
          f"{'aug':<6} | {'R2':>6} | {'RMSE':>8} | {'RPD':>5}")
    print("-" * 72)
    for el in elems:
        sub = df[df["element"] == el]
        if sub.empty:
            continue
        b  = sub.loc[sub["R2"].idxmax()]
        sn = short_name(cfg, el)
        print(f"{sn:>6} | {b['model']:<14} | {b['feature_set']:<13} | "
              f"{str(b.get('augmented','')):<6} | {b['R2']:>+6.3f} | "
              f"{b.get('RMSE', float('nan')):>8.4f} | "
              f"{b.get('RPD', float('nan')):>5.2f}")

    # MixUp delta
    print(f"\n{'='*52}")
    print("MixUp delta (лучший mixup − лучший none):")
    print("-" * 52)
    for el in elems:
        sub = df[df["element"] == el]
        if sub.empty:
            continue
        r2_none  = sub[sub["augmented"] == "none"]["R2"].max()
        sub_mx   = sub[sub["augmented"] == "mixup"]
        r2_mixup = sub_mx["R2"].max() if not sub_mx.empty else np.nan
        delta    = r2_mixup - r2_none if np.isfinite(r2_mixup) else np.nan
        mark     = ("↑" if delta > 0.05
                    else ("↓" if delta < -0.05 else "≈"))
        sn = short_name(cfg, el)
        print(f"  {sn:>6}: none={r2_none:+.3f}  "
              f"mixup={r2_mixup:+.3f}  Δ={delta:+.3f} {mark}")

    print(f"\ndone -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Enhanced ML pipeline for multispectral data")
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--date",     nargs="+",
                    help="date keys: date1 date2 (default: both)")
    ap.add_argument("--elements", nargs="+",
                    help="element column names")
    ap.add_argument("--models",   nargs="+",
                    help="model names")
    ap.add_argument("--output",   default=None)
    run(ap.parse_args())
