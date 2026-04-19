#!/usr/bin/env python3
"""ML from hyperspectral: preprocessing + feature selection + models."""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_hyper_from_xlsx, load_hyper_date, load_hyper_wavelength_map, load_chemistry
from analysis.preprocessing import preprocess_pipeline
from analysis.feature_selection import combined_selection, pca_reduce, select_by_vip
from analysis.hyper_indices import calculate_hyper_indices
from analysis.ml_pipeline import compare_models, evaluate_model
from analysis.explainability import plot_prediction_scatter


def _load_hyper(cfg, dk):
    hp = cfg["paths"]["hyper"]
    hn = cfg["hyper_naming"]
    for key, fn in [(f"{dk}_xlsx", load_hyper_from_xlsx)]:
        p = hp.get(key)
        if p and Path(p).exists():
            return fn(p)
    fld = hp.get(f"{dk}_folder")
    if fld and Path(fld).exists():
        wm = load_hyper_wavelength_map(hp["wavelength_map"])
        return load_hyper_date(fld, wm, hn["prefix_length"],
                               hn["value_column"], hn["id_column"],
                               hn["min_valid_bands"])
    return None, None


def _fsets(spec, wl, y, cfg):
    m = np.isfinite(y)
    wn = [f"{w:.1f}" for w in wl]
    fs = {}
    want = cfg["ml"]["hyper_feature_sets"]

    if "raw_smooth" in want:
        X, _ = preprocess_pipeline(spec, wl, ["smooth"])
        fs["raw_smooth"] = (X, wn)
    if "deriv1_snv" in want:
        X, _ = preprocess_pipeline(spec, wl, ["smooth", "deriv1", "snv"])
        fs["deriv1_snv"] = (X, [f"d1_{n}" for n in wn])
    if "pca_10" in want:
        try:
            Xs, _ = preprocess_pipeline(spec, wl, ["smooth"])
            nc = min(cfg["ml"]["hyper_pca_components"], spec.shape[0] - 2)
            Xp, _ = pca_reduce(Xs, n_components=nc)
            fs["pca_10"] = (Xp, [f"PC{i+1}" for i in range(Xp.shape[1])])
        except Exception:
            pass
    if "cars_spa_25" in want:
        try:
            Xs = fs.get("raw_smooth", (spec, wn))[0]
            idx = combined_selection(Xs, y, "cars_spa", n_select=cfg["ml"]["hyper_selected_bands"])
            if len(idx) >= 5:
                fs["cars_spa"] = (Xs[:, idx], [wn[i] for i in idx])
        except Exception:
            pass
    if "vip" in want:
        try:
            Xs = fs.get("raw_smooth", (spec, wn))[0]
            idx = select_by_vip(Xs[m], y[m])
            if len(idx) >= 3:
                fs["vip"] = (Xs[:, idx], [wn[i] for i in idx])
        except Exception:
            pass

    hi = calculate_hyper_indices(spec, wl)
    if hi.shape[1] > 0:
        fs["hyper_idx"] = (hi.values, list(hi.columns))
    return fs


def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg["paths"]["output"]["ml_hyper"]
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
            print(f"[{dk}] no data"); continue

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
        print(f"[{dk}] {len(ids)} pts, {spec.shape[1]} bands")

        for el in elems:
            if el not in chem.columns: continue
            y = chem.loc[ids, el].values
            nv = np.isfinite(y).sum()
            if nv < 15: continue
            sn = short_name(cfg, el)
            print(f"\n  {sn} n={nv}")

            for fn, (X, names) in _fsets(spec, wl, y, cfg).items():
                print(f"    {fn} ({X.shape[1]}f)", end=" ")
                rdf, pred = compare_models(X, y, mdls, n_splits=nf)
                for _, r in rdf.iterrows():
                    d = r.to_dict()
                    d.update(element=el, date=dk, feature_set=fn, n_features=X.shape[1])
                    res.append(d)

                top = rdf.iloc[0]["model"]
                if top in pred:
                    fig = plot_prediction_scatter(
                        pred[top]["y_true"], pred[top]["y_pred"], top, sn)
                    fig.savefig(f"{out}/scatter_{sn}_{fn}_{dk}.png",
                                dpi=dpi, bbox_inches="tight")
                    plt.close(fig)

    if not args.no_dl:
        try:
            from analysis.dl_models import CNN1DRegressor
            print("\n--- 1D-CNN ---")
            Xsm, _ = preprocess_pipeline(spec, wl, ["smooth"])
            for el in elems[:4]:
                if el not in chem.columns: continue
                y = chem.loc[ids, el].values
                if np.isfinite(y).sum() < 20: continue
                sn = short_name(cfg, el)
                cnn = CNN1DRegressor(n_bands=Xsm.shape[1], use_attention=True,
                                     epochs=150, batch_size=16, patience=25)
                mt = evaluate_model(cnn, Xsm, y, n_splits=nf, scale=False)
                print(f"  {sn}: R2={mt.get('R2', np.nan):.4f}")
                res.append({"model": "1D-CNN-Attn", "element": el, "date": dk,
                            "feature_set": "raw_300", "n_features": Xsm.shape[1],
                            "R2": mt.get("R2"), "RMSE": mt.get("RMSE"),
                            "MAE": mt.get("MAE"), "RPD": mt.get("RPD")})
        except ImportError:
            print("  torch not installed")

    if res:
        df = pd.DataFrame(res).sort_values(["element", "R2"], ascending=[True, False])
        df.to_csv(f"{out}/ml_hyper.csv", index=False, float_format="%.4f")
        best = df.loc[df.groupby("element")["R2"].idxmax()]
        print("\n--- best ---")
        for _, r in best.iterrows():
            print(f"  {short_name(cfg, r['element']):>5}: {r['model']:<15} "
                  f"R2={r['R2']:.3f} [{r.get('feature_set', '')}]")
    print(f"done -> {out}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--output")
    ap.add_argument("--no-dl", action="store_true")
    run(ap.parse_args())
