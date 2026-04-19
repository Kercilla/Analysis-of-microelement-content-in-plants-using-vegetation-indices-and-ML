#!/usr/bin/env python3
"""Fusion: multispectral indices + hyperspectral PCA + hyper indices."""
import argparse, os, sys, warnings
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import (
    load_multi, load_hyper_from_xlsx, load_hyper_date,
    load_hyper_wavelength_map, load_chemistry,
)
from analysis.indices import calculate_indices
from analysis.hyper_indices import calculate_hyper_indices
from analysis.feature_selection import pca_reduce
from analysis.ml_pipeline import compare_models


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


def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg["paths"]["output"]["ml_fusion"]
    os.makedirs(out, exist_ok=True)

    bmap = cfg["camera"]["band_map"]
    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    mdls = args.models or cfg["ml"]["models"]
    nf = cfg["ml"]["cv_folds"]
    tiers = cfg["statistics"]["index_tiers"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    npca = cfg["ml"]["hyper_pca_components"]

    res = []
    for dk in (args.date or ["date1"]):
        pat = cfg["paths"]["multi"].get(dk)
        if not pat or not glob(pat):
            continue
        df_b = load_multi(pat, bmap)
        df_i = pd.concat([df_b, calculate_indices(df_b, tiers=tiers)], axis=1)

        hdf, wl = _load_hyper(cfg, dk)
        if hdf is None:
            print(f"[{dk}] no hyper"); continue

        sk = dsmap.get(dk, "sampling1")
        for k, v in dsmap.items():
            if k in dk or dk in k:
                sk = v; break
        cp = cfg["paths"]["chemistry"].get(sk)
        if not cp or not Path(cp).exists(): continue
        off = cfg["chemistry"]["id_offsets"].get(sk, 0)
        chem = load_chemistry(cp, cols, off)

        ids = df_i.index.intersection(hdf.index).intersection(chem.index)
        print(f"[{dk}] {len(ids)} common pts")
        if len(ids) < 15: continue

        spec = hdf.loc[ids].values
        nc = min(npca, len(ids) - 2, spec.shape[1])
        Xpca, _ = pca_reduce(spec, n_components=nc)
        hi = calculate_hyper_indices(spec, wl)

        Xa = df_i.loc[ids].values
        Xb = np.hstack([Xpca, hi.values])
        Xc = np.hstack([Xa, Xb])
        na = list(df_i.columns)
        nb = [f"HPC{i+1}" for i in range(Xpca.shape[1])] + list(hi.columns)

        tracks = {"A_multi": (Xa, na), "B_hyper": (Xb, nb), "C_fusion": (Xc, na + nb)}

        for el in elems:
            if el not in chem.columns: continue
            y = chem.loc[ids, el].values
            if np.isfinite(y).sum() < 15: continue
            sn = short_name(cfg, el)
            print(f"  {sn}:")
            for tr, (X, _fn) in tracks.items():
                rdf, _ = compare_models(X, y, mdls, n_splits=nf)
                for _, r in rdf.iterrows():
                    d = r.to_dict()
                    d.update(element=el, date=dk, track=tr, n_features=X.shape[1])
                    res.append(d)
                b = rdf.iloc[0]
                print(f"    {tr[0]}: {b['model']:<10} R2={b['R2']:.3f} ({X.shape[1]}f)")

    if not res:
        print("no results"); return

    df = pd.DataFrame(res).sort_values(["element", "R2"], ascending=[True, False])
    df.to_csv(f"{out}/fusion.csv", index=False, float_format="%.4f")

    print(f"\n{'':>7} | {'multi':>12} | {'hyper':>12} | {'fusion':>12}")
    print("  " + "-" * 45)
    for el in elems:
        sub = df[df["element"] == el]
        if sub.empty: continue
        sn = short_name(cfg, el)
        parts = []
        for tr in ["A_multi", "B_hyper", "C_fusion"]:
            ts = sub[sub["track"] == tr]
            if not ts.empty:
                b = ts.iloc[0]
                parts.append(f"{b['R2']:+.3f}")
            else:
                parts.append("  -")
        print(f"  {sn:>5} | {parts[0]:>12} | {parts[1]:>12} | {parts[2]:>12}")
    print(f"done -> {out}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--output")
    run(ap.parse_args())
