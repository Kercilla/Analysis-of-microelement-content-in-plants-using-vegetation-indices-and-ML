#!/usr/bin/env python3
"""Statistical correlation: multispectral indices vs chemistry."""
import argparse, os, sys, warnings
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.correlation import AnalysisConfig, run_correlation, get_top_correlations
from analysis.visualization import plot_heatmap, plot_scatter_top, plot_method_comparison

def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg["paths"]["output"]["stat_multi"]
    os.makedirs(out, exist_ok=True)

    bmap = cfg["camera"]["band_map"]
    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    st = cfg["statistics"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dpi = cfg["plots"]["dpi"]

    acfg = AnalysisConfig(methods=st["methods"], alpha=st["alpha"],
                          index_tiers=st["index_tiers"])

    dates = args.date or list(cfg["paths"]["multi"].keys())
    all_corr = []

    for dk in dates:
        pat = cfg["paths"]["multi"].get(dk)
        if not pat or not glob(pat):
            continue
        files = sorted(glob(pat))
        print(f"[{dk}] {len(files)} files")

        df_bands = load_multi(pat, bmap)
        df_idx = calculate_indices(df_bands, tiers=st["index_tiers"])

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

        ids = df_idx.index.intersection(df_chem.index)
        if len(ids) < st["min_samples"]:
            print(f"  skip: {len(ids)} pts < {st['min_samples']}")
            continue

        avail = [e for e in elems if e in df_chem.columns]
        corr = run_correlation(df_idx.loc[ids], df_chem.loc[ids], avail, acfg)
        corr["date"] = dk
        all_corr.append(corr)

        for m in st["methods"]:
            fig = plot_heatmap(corr, m, title=f"{m} {dk}")
            fig.savefig(f"{out}/heatmap_{m}_{dk}.png", dpi=dpi, bbox_inches="tight")

        fig = plot_scatter_top(corr, df_idx.loc[ids], df_chem.loc[ids],
                               st["scatter_top_n"], title=f"top {dk}")
        fig.savefig(f"{out}/scatter_{dk}.png", dpi=dpi, bbox_inches="tight")

        fig = plot_method_comparison(corr)
        fig.savefig(f"{out}/methods_{dk}.png", dpi=dpi, bbox_inches="tight")

        nsig = (corr["pearson_p"] < st["alpha"]).sum() if "pearson_p" in corr.columns else 0
        print(f"  {len(corr)} pairs, {nsig} significant")
        for el in avail:
            sub = corr[(corr["element"] == el) & corr["pearson_r"].notna()]
            if sub.empty: continue
            b = sub.loc[sub["pearson_r"].abs().idxmax()]
            print(f"    {short_name(cfg, el):>5}: {b['index']:<12} r={b['pearson_r']:+.3f}")

    if all_corr:
        full = pd.concat(all_corr, ignore_index=True)
        full.to_csv(f"{out}/correlations.csv", index=False, float_format="%.6f")
        top3 = get_top_correlations(full, top_n=3, per_element=True)
        top3.to_csv(f"{out}/top3.csv", index=False, float_format="%.4f")
    print(f"done -> {out}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output")
    run(ap.parse_args())
