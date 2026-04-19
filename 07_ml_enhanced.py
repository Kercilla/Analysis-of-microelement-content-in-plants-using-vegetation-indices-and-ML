#!/usr/bin/env python3
"""Enhanced ML: CWT wavelets + shape features + augmentation + stacking + multi-output GPR."""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_hyper_from_xlsx, load_hyper_date, load_hyper_wavelength_map, load_chemistry
from analysis.preprocessing import (
    preprocess_pipeline, cwt_features, spectral_shape_features,
    mixup_augment, noise_augment,
)
from analysis.feature_selection import combined_selection, pca_reduce
from analysis.hyper_indices import calculate_hyper_indices
from analysis.ml_pipeline import (
    compare_models, evaluate_model, stacking_ensemble, multi_output_gpr,
    regression_metrics, _registry,
)
from analysis.explainability import plot_prediction_scatter


def _load_hyper(cfg, dk):
    hp = cfg["paths"]["hyper"]
    hn = cfg["hyper_naming"]
    xp = hp.get(f"{dk}_xlsx")
    if xp and Path(xp).exists():
        return load_hyper_from_xlsx(xp)
    fld = hp.get(f"{dk}_folder")
    if fld and Path(fld).exists():
        wm = load_hyper_wavelength_map(hp["wavelength_map"])
        return load_hyper_date(fld, wm, hn["prefix_length"],
                               hn["value_column"], hn["id_column"],
                               hn["min_valid_bands"])
    return None, None


def build_enhanced_features(spec, wl, cfg):
    """Build feature sets: raw+CWT+shape, CARS-selected, hyper indices, combined."""
    fs = {}

    # 1. Smoothed raw
    Xsm, _ = preprocess_pipeline(spec, wl, ["smooth"])
    wn = [f"{w:.1f}" for w in wl]
    fs["smooth_300"] = (Xsm, wn)

    # 2. deriv1+snv
    Xd, _ = preprocess_pipeline(spec, wl, ["smooth", "deriv1", "snv"])
    fs["d1snv_300"] = (Xd, [f"d1_{n}" for n in wn])

    # 3. CWT wavelets
    try:
        Xcwt, cwt_names = cwt_features(Xsm, wl)
        fs["cwt"] = (Xcwt, cwt_names)
    except Exception as e:
        print(f"    cwt skip: {e}")

    # 4. Spectral shape
    try:
        Xshape, shape_names = spectral_shape_features(Xsm, wl)
        fs["shape"] = (Xshape, shape_names)
    except Exception as e:
        print(f"    shape skip: {e}")

    # 5. Hyper indices
    hi = calculate_hyper_indices(spec, wl)
    if hi.shape[1] > 0:
        fs["hyper_idx"] = (hi.values, list(hi.columns))

    # 6. PCA
    try:
        nc = min(cfg["ml"]["hyper_pca_components"], spec.shape[0] - 2)
        Xpca, _ = pca_reduce(Xsm, n_components=nc)
        fs["pca"] = (Xpca, [f"PC{i+1}" for i in range(Xpca.shape[1])])
    except Exception:
        pass

    # 7. Combined: PCA + CWT + shape + hyper_idx
    parts, names = [], []
    for k in ("pca", "cwt", "shape", "hyper_idx"):
        if k in fs:
            parts.append(fs[k][0])
            names.extend(fs[k][1])
    if parts:
        fs["combined"] = (np.hstack(parts), names)

    return fs


def run(args):
    cfg = load_config(args.config)
    out = args.output or "results/07_enhanced"
    os.makedirs(out, exist_ok=True)

    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    mdls = args.models or cfg["ml"]["models"]
    nf = cfg["ml"]["cv_folds"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dpi = cfg["plots"]["dpi"]

    res = []
    for dk in (args.date or ["date1", "date2"]):
        hdf, wl = _load_hyper(cfg, dk)
        if hdf is None:
            print(f"[{dk}] no hyper data"); continue

        sk = dsmap.get(dk, "sampling1")
        for k, v in dsmap.items():
            if k in dk or dk in k:
                sk = v; break
        cp = cfg["paths"]["chemistry"].get(sk)
        if not cp or not Path(cp).exists(): continue
        off = cfg["chemistry"]["id_offsets"].get(sk, 0)
        chem = load_chemistry(cp, cols, off)

        ids = hdf.index.intersection(chem.index).values
        spec = hdf.loc[ids].values
        print(f"\n[{dk}] {len(ids)} pts, {spec.shape[1]} bands")

        fsets = build_enhanced_features(spec, wl, cfg)
        print(f"  feature sets: {', '.join(f'{k}({v[0].shape[1]})' for k,v in fsets.items())}")

        for el in elems:
            if el not in chem.columns: continue
            y = chem.loc[ids, el].values
            nv = np.isfinite(y).sum()
            if nv < 15: continue
            sn = short_name(cfg, el)
            print(f"\n  {sn} (n={nv})")

            for fn, (X, names) in fsets.items():
                nf_feat = X.shape[1]

                # --- A: standard CV (no augmentation) ---
                rdf, pred = compare_models(X, y, mdls, n_splits=nf)
                for _, r in rdf.iterrows():
                    d = r.to_dict()
                    d.update(element=el, date=dk, feature_set=fn,
                             n_features=nf_feat, augmented="none")
                    res.append(d)

                top_r2 = rdf.iloc[0]["R2"]
                top_mdl = rdf.iloc[0]["model"]
                print(f"    {fn:>12}({nf_feat:>3}f): {top_mdl:<12} R2={top_r2:.3f}")

                # --- B: MixUp augmentation (only for combined/cwt/shape) ---
                if fn in ("combined", "cwt", "shape", "hyper_idx") and nv >= 20:
                    mask = np.isfinite(y)
                    Xa, ya = mixup_augment(X[mask], y[mask], n_aug=nv)
                    rdf2, _ = compare_models(Xa, ya, mdls, n_splits=nf)
                    for _, r in rdf2.iterrows():
                        d = r.to_dict()
                        d.update(element=el, date=dk, feature_set=fn,
                                 n_features=nf_feat, augmented="mixup")
                        res.append(d)
                    aug_r2 = rdf2.iloc[0]["R2"]
                    aug_mdl = rdf2.iloc[0]["model"]
                    delta = aug_r2 - top_r2
                    if abs(delta) > 0.01:
                        print(f"    {'':>12} +mixup: {aug_mdl:<12} R2={aug_r2:.3f} ({delta:+.3f})")

                # scatter for best
                if top_mdl in pred:
                    fig = plot_prediction_scatter(
                        pred[top_mdl]["y_true"], pred[top_mdl]["y_pred"], top_mdl, sn)
                    fig.savefig(f"{out}/scatter_{sn}_{fn}_{dk}.png",
                                dpi=dpi, bbox_inches="tight")
                    plt.close(fig)

            # --- C: Stacking ensemble on combined features ---
            if "combined" in fsets:
                Xc = fsets["combined"][0]
                print(f"    stacking ({Xc.shape[1]}f)...", end=" ")
                sm = stacking_ensemble(Xc, y, n_splits=nf)
                if sm:
                    sr2 = sm.get("R2", np.nan)
                    print(f"R2={sr2:.3f}")
                    res.append({
                        "model": "Stacking", "element": el, "date": dk,
                        "feature_set": "combined", "n_features": Xc.shape[1],
                        "R2": sr2, "RMSE": sm.get("RMSE"), "RPD": sm.get("RPD"),
                        "augmented": "none",
                    })
                else:
                    print("skip")

        # --- D: Multi-output GPR (all nutrients at once) ---
        if "pca" in fsets:
            Xpca = fsets["pca"][0]
            avail = [e for e in elems if e in chem.columns]
            Y = np.column_stack([chem.loc[ids, e].values for e in avail])
            mask_all = np.all(np.isfinite(Y), axis=1)
            if mask_all.sum() >= 15:
                print(f"\n  Multi-output GPR ({Xpca.shape[1]}f, {len(avail)} targets, "
                      f"n={mask_all.sum()})...", end=" ")
                try:
                    mo_res = multi_output_gpr(Xpca, Y, n_splits=nf)
                    print("done")
                    for j, e in enumerate(avail):
                        if j in mo_res:
                            m = mo_res[j]
                            sn = short_name(cfg, e)
                            print(f"    {sn:>5}: R2={m['R2']:.3f}")
                            res.append({
                                "model": "MO-GPR", "element": e, "date": dk,
                                "feature_set": "pca", "n_features": Xpca.shape[1],
                                "R2": m["R2"], "RMSE": m["RMSE"], "RPD": m["RPD"],
                                "augmented": "none",
                            })
                except Exception as e:
                    print(f"error: {e}")

    if not res:
        print("no results"); return

    df = pd.DataFrame(res)
    df.to_csv(f"{out}/enhanced.csv", index=False, float_format="%.4f")

    # summary: best per element (including augmentation tag)
    print(f"\n{'='*65}")
    print(f"{'elem':>5} | {'model':<15} | {'features':<12} | {'aug':<6} | {'R2':>6}")
    print("-" * 65)
    for el in elems:
        sub = df[df["element"] == el]
        if sub.empty: continue
        b = sub.loc[sub["R2"].idxmax()]
        sn = short_name(cfg, el)
        print(f"{sn:>5} | {b['model']:<15} | {b['feature_set']:<12} | "
              f"{b.get('augmented',''):<6} | {b['R2']:+.3f}")

    print(f"\ndone -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--output")
    run(ap.parse_args())
