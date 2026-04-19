#!/usr/bin/env python3
"""ML prediction from multispectral vegetation indices."""
import argparse, os, sys, warnings
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.ml_pipeline import compare_models
from analysis.explainability import plot_prediction_scatter

def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg["paths"]["output"]["ml_multi"]
    os.makedirs(out, exist_ok=True)

    bmap = cfg["camera"]["band_map"]
    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    mdls = args.models or cfg["ml"]["models"]
    nfold = cfg["ml"]["cv_folds"]
    tiers = cfg["statistics"]["index_tiers"]
    dsmap = cfg["chemistry"]["date_sampling_map"]

    dates = args.date or list(cfg["paths"]["multi"].keys())
    res = []

    for dk in dates:
        pat = cfg["paths"]["multi"].get(dk)
        if not pat or not glob(pat):
            continue
        df_bands = load_multi(pat, bmap)
        df_idx = calculate_indices(df_bands, tiers=tiers)
        X_full = pd.concat([df_bands, df_idx], axis=1)

        sk = dsmap.get(dk, "sampling1")
        for k, v in dsmap.items():
            if k in pat:
                sk = v
                break
        cp = cfg["paths"]["chemistry"].get(sk)
        if not cp or not Path(cp).exists():
            continue
        off = cfg["chemistry"]["id_offsets"].get(sk, 0)
        df_chem = load_chemistry(cp, cols, off)

        ids = X_full.index.intersection(df_chem.index)
        Xmat = X_full.loc[ids].values
        print(f"[{dk}] {Xmat.shape[1]} feat, n={len(ids)}")

        for el in elems:
            if el not in df_chem.columns:
                continue
            y = df_chem.loc[ids, el].values
            nvalid = np.isfinite(y).sum()
            if nvalid < 15:
                continue
            sn = short_name(cfg, el)
            print(f"  {sn} n={nvalid}")

            rdf, pred = compare_models(Xmat, y, mdls, n_splits=nfold)
            for _, r in rdf.iterrows():
                d = r.to_dict()
                d["element"] = el
                d["date"] = dk
                d["n_features"] = Xmat.shape[1]
                res.append(d)

            top = rdf.iloc[0]["model"]
            if top in pred:
                fig = plot_prediction_scatter(
                    pred[top]["y_true"], pred[top]["y_pred"], top, sn)
                fig.savefig(f"{out}/scatter_{sn}_{dk}.png",
                            dpi=cfg["plots"]["dpi"], bbox_inches="tight")
                plt.close(fig)

    if res:
        df = pd.DataFrame(res).sort_values(["element", "R2"], ascending=[True, False])
        df.to_csv(f"{out}/ml_multi.csv", index=False, float_format="%.4f")
        best = df.loc[df.groupby("element")["R2"].idxmax()]
        print("\n--- best ---")
        for _, r in best.iterrows():
            print(f"  {short_name(cfg, r['element']):>5}: {r['model']:<12} R2={r['R2']:.3f}")
    print(f"done -> {out}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--output")
    run(ap.parse_args())
