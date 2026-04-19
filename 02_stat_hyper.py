#!/usr/bin/env python3
"""Hyperspectral correlogram analysis + narrowband indices."""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import (
    load_hyper_date, load_hyper_from_xlsx,
    load_hyper_wavelength_map, load_chemistry,
)
from analysis.preprocessing import preprocess_pipeline
from analysis.hyper_indices import calculate_hyper_indices, HYPER_INDEX_REGISTRY
from analysis.correlation import AnalysisConfig, run_correlation


def _corr(spec, y, wl):
    m = np.isfinite(y)
    X, yy = spec[m], y[m]
    out = []
    for i, w in enumerate(wl):
        b = X[:, i]
        v = np.isfinite(b)
        if v.sum() < 10: continue
        r, p = stats.pearsonr(b[v], yy[v])
        rho, _ = stats.spearmanr(b[v], yy[v])
        out.append({"wl": w, "idx": i, "r": r, "p": p, "rho": rho, "n": v.sum()})
    return pd.DataFrame(out)


def _plot_cg(cgs, elem, sname, path):
    fig, ax = plt.subplots(figsize=(14, 4))
    clr = ["#2E75B6", "#E06666", "#70AD47", "#9B59B6", "#F39C12"]
    for i, (nm, cg) in enumerate(cgs.items()):
        ax.plot(cg["wl"], cg["r"], lw=1.1, color=clr[i % 5], label=nm, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.4)
    for t in (0.3, -0.3):
        ax.axhline(t, color="gray", lw=0.4, ls=":")
    for b, c in [(475, "blue"), (560, "green"), (668, "red"),
                 (717, "orange"), (840, "brown")]:
        ax.axvline(b, color=c, lw=0.6, ls="--", alpha=0.35)
    ax.set(xlabel="nm", ylabel="r", ylim=(-0.6, 0.6), title=sname)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_hyper(cfg, dk):
    hp = cfg["paths"]["hyper"]
    hn = cfg["hyper_naming"]
    for key, fn in [(f"{dk}_xlsx", load_hyper_from_xlsx)]:
        p = hp.get(key)
        if p and Path(p).exists():
            return fn(p)
    folder = hp.get(f"{dk}_folder")
    if folder and Path(folder).exists():
        wm = load_hyper_wavelength_map(hp["wavelength_map"])
        return load_hyper_date(folder, wm, hn["prefix_length"],
                               hn["value_column"], hn["id_column"],
                               hn["min_valid_bands"])
    return None, None


def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg["paths"]["output"]["stat_hyper"]
    os.makedirs(out, exist_ok=True)

    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    pvars = cfg["hyper_preprocessing"]["variants"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dpi = cfg["plots"]["dpi"]

    for dk in (args.date or ["date1", "date2"]):
        print(f"\n[{dk}]")
        hdf, wl = _load_hyper(cfg, dk)
        if hdf is None:
            print("  no data"); continue

        sk = dsmap.get(dk, "sampling1")
        for k, v in dsmap.items():
            if k in dk or dk in k:
                sk = v; break
        cp = cfg["paths"]["chemistry"].get(sk)
        if not cp or not Path(cp).exists():
            continue
        off = cfg["chemistry"]["id_offsets"].get(sk, 0)
        chem = load_chemistry(cp, cols, off)
        common = hdf.index.intersection(chem.index).values
        spec = hdf.loc[common].values
        print(f"  {len(common)} pts, {spec.shape[1]} bands")
        if len(common) < 10:
            continue

        rows = []
        for pname, steps in pvars.items():
            Xp, _ = preprocess_pipeline(spec, wl, steps=steps)
            print(f"  [{pname}]")
            for el in elems:
                if el not in chem.columns: continue
                y = chem.loc[common, el].values
                cg = _corr(Xp, y, wl)
                if cg.empty: continue
                bst = cg.loc[cg["r"].abs().idxmax()]
                print(f"    {short_name(cfg, el):>5}: r={bst['r']:+.3f} @{bst['wl']:.0f}nm")
                for _, rw in cg.iterrows():
                    d = rw.to_dict()
                    d["element"] = el
                    d["prep"] = pname
                    rows.append(d)

        # correlograms per element
        for el in elems:
            if el not in chem.columns: continue
            sn = short_name(cfg, el)
            pcg = {}
            for pn, st in pvars.items():
                Xp, _ = preprocess_pipeline(spec, wl, steps=st)
                pcg[pn] = _corr(Xp, chem.loc[common, el].values, wl)
            _plot_cg(pcg, el, sn, f"{out}/correlogram_{sn}_{dk}.png")

        # heatmap
        Xr, _ = preprocess_pipeline(spec, wl, steps=["smooth"])
        raw_cg = {}
        for el in elems:
            if el not in chem.columns: continue
            raw_cg[el] = _corr(Xr, chem.loc[common, el].values, wl)

        mat = np.zeros((len(raw_cg), len(wl)))
        elist = list(raw_cg.keys())
        for i, el in enumerate(elist):
            for _, rw in raw_cg[el].iterrows():
                mat[i, int(rw["idx"])] = rw["r"]
        step = max(1, len(wl) // 50)
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=[f"{w:.0f}" if j % step == 0 else "" for j, w in enumerate(wl)],
                    yticklabels=[short_name(cfg, e) for e in elist], ax=ax)
        ax.set_title(f"r(lambda) {dk}")
        plt.tight_layout()
        fig.savefig(f"{out}/heatmap_wl_{dk}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # hyper indices
        print(f"  hyper indices ({len(HYPER_INDEX_REGISTRY)})")
        hi = calculate_hyper_indices(spec, wl)
        hi.index = common
        avail = [e for e in elems if e in chem.columns]
        acfg = AnalysisConfig(methods=cfg["statistics"]["methods"],
                              alpha=cfg["statistics"]["alpha"])
        hcorr = run_correlation(hi, chem.loc[common], avail, acfg)
        hcorr["date"] = dk
        for el in avail:
            sub = hcorr[(hcorr["element"] == el) & hcorr["pearson_r"].notna()]
            if sub.empty: continue
            b = sub.loc[sub["pearson_r"].abs().idxmax()]
            print(f"    {short_name(cfg, el):>5}: {b['index']:<20} r={b['pearson_r']:+.3f}")
        hcorr.to_csv(f"{out}/hyper_idx_{dk}.csv", index=False, float_format="%.4f")

        if rows:
            pd.DataFrame(rows).to_csv(f"{out}/correlogram_{dk}.csv",
                                       index=False, float_format="%.6f")
    print(f"done -> {out}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output")
    run(ap.parse_args())
