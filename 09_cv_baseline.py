#!/usr/bin/env python3
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.cv_pipeline import (
    compute_band_stats, extract_point_features,
    load_point_coords, world_to_pixel,
    spatial_cv_splits, compute_metrics,
    apply_gbr_tiled,
)


def train_models(X, Y, target_names):
    models, scalers = [], []
    for j, name in enumerate(target_names):
        y    = Y[:, j]; mask = np.isfinite(y)
        if mask.sum() < 10:
            models.append(None); scalers.append(None); continue
        sc = StandardScaler()
        m  = GradientBoostingRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8, random_state=42)
        m.fit(sc.fit_transform(X[mask]), y[mask])
        models.append(m); scalers.append(sc)
        print(f"    {name}: n={mask.sum()}")
    return models, scalers


def plot_scatter(all_true, all_pred, names, out_dir, cfg, dpi):
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        yt = np.array(all_true[name]); yp = np.array(all_pred[name])
        msk= np.isfinite(yt) & np.isfinite(yp)
        m  = compute_metrics(yt[msk], yp[msk])
        ax.scatter(yt[msk], yp[msk], alpha=0.6, s=25, c="#2E75B6", edgecolors="none")
        lims = [min(yt[msk].min(), yp[msk].min()), max(yt[msk].max(), yp[msk].max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        xl = np.linspace(*lims, 50)
        ax.plot(xl, np.poly1d(np.polyfit(yt[msk], yp[msk], 1))(xl), color="#C00000", lw=1.5)
        ax.set_title(short_name(cfg, name), fontsize=9)
        ax.text(0.05, 0.95,
                f"R²={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}\nRPD={m['RPD']:.2f}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
        ax.set_xlabel("Observed", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.suptitle("GBR Baseline — Spatial CV", fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "baseline_scatter_cv.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_maps(pred_map, names, out_dir, dpi):
    cmaps = ["RdYlGn","YlOrRd","Blues","Greens","PuRd",
             "Oranges","BuGn","RdPu","YlGn","BuPu","GnBu","OrRd"]
    for i, name in enumerate(names):
        band  = pred_map[i]
        valid = band[np.isfinite(band) & (band != 0)]
        if len(valid) == 0: continue
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(band, cmap=cmaps[i % len(cmaps)], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=name, shrink=0.6)
        ax.set_title(name, fontsize=12); ax.axis("off"); plt.tight_layout()
        fn = name.replace("%","pct").replace("/","_").replace(" ","_")
        fig.savefig(Path(out_dir) / f"map_{fn}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def run(args):
    cfg   = load_config(args.config)
    out   = args.output or "results/09_cv_baseline"
    os.makedirs(out, exist_ok=True)

    elems = args.elements or cfg["chemistry"]["target_elements"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dk    = args.date or "date2"
    sk    = dsmap.get(dk, "sampling2")
    chem  = load_chemistry(cfg["paths"]["chemistry"][sk],
                           cfg["chemistry"]["columns"],
                           cfg["chemistry"]["id_offsets"].get(sk, 0))
    avail = [e for e in elems if e in chem.columns]
    print(f"Targets: {avail}")

    import rasterio
    print(f"Computing band stats (downsampled)...")
    band_stats = compute_band_stats(args.ortho)
    print(f"  stats: {[(round(lo,3), round(hi,3)) for lo,hi in band_stats]}")

    with rasterio.open(args.ortho) as src:
        meta = src.meta.copy()

    gpkg_files = sorted(glob(cfg["paths"]["multi"].get(dk)))
    if not gpkg_files: print("No gpkg found"); return
    ids, coords   = load_point_coords(gpkg_files[0], target_crs=meta["crs"])
    rows_px, cpx  = world_to_pixel(coords, meta["transform"])
    H, W          = meta["height"], meta["width"]
    ok            = (rows_px >= 0) & (rows_px < H) & (cpx >= 0) & (cpx < W)
    ids, coords   = ids[ok], coords[ok]
    rows_px, cpx  = rows_px[ok], cpx[ok]

    common  = np.intersect1d(ids, chem.index.values)
    msk     = np.isin(ids, common)
    rows_px, cpx  = rows_px[msk], cpx[msk]
    chem_sub      = chem.loc[ids[msk]]
    coords_sub    = coords[msk]
    Y             = chem_sub[avail].values.astype(float)

    print(f"Extracting features at {len(common)} points (windowed reads)...")
    X = extract_point_features(args.ortho, rows_px, cpx, band_stats)
    print(f"  feature matrix: {X.shape}")

    splits   = spatial_cv_splits(coords_sub, n_folds=min(5, args.folds))
    all_true = {n: [] for n in avail}
    all_pred = {n: [] for n in avail}

    print(f"\nSpatial {len(splits)}-fold CV:")
    for fold, (tr, val) in enumerate(splits):
        mdls, scs = train_models(X[tr], Y[tr], avail)
        for j, name in enumerate(avail):
            if mdls[j] is None: continue
            yp = mdls[j].predict(scs[j].transform(X[val]))
            all_true[name].extend(Y[val, j].tolist())
            all_pred[name].extend(yp.tolist())
        r2s = [f"{compute_metrics(np.array(all_true[n]), np.array(all_pred[n]))['R2']:+.3f}" for n in avail[:4]]
        print(f"  Fold {fold+1}: [{', '.join(r2s)}...]")

    print(f"\n{'Nutrient':<20} {'R2':>6} {'RMSE':>8} {'RPD':>6}")
    print("-"*44)
    summary = []
    for name in avail:
        m  = compute_metrics(np.array(all_true[name]), np.array(all_pred[name]))
        sn = short_name(cfg, name)
        print(f"{sn:<20} {m['R2']:>+6.3f} {m['RMSE']:>8.4f} {m['RPD']:>6.2f}")
        summary.append({"element": name, **m})

    pd.DataFrame(summary).to_csv(f"{out}/baseline_cv_metrics.csv",
                                  index=False, float_format="%.4f")
    plot_scatter(all_true, all_pred, avail, out, cfg, cfg["plots"]["dpi"])

    if args.predict_map:
        print("\nTraining final models on all data...")
        mdls_f, scs_f = train_models(X, Y, avail)
        print("Running tiled map prediction...")
        apply_gbr_tiled(args.ortho, mdls_f, scs_f, avail,
                         band_stats, meta,
                         out_path=f"{out}/baseline_nutrients.tif",
                         tile_size=args.tile_size)
        # plot maps from saved file
        import rasterio as _rio
        with _rio.open(f"{out}/baseline_nutrients.tif") as _src:
            pred_map = _src.read()
        plot_maps(pred_map, avail, out, cfg["plots"]["dpi"])

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="config.yaml")
    ap.add_argument("--ortho",       required=True)
    ap.add_argument("--date",        default="date2")
    ap.add_argument("--elements",    nargs="+")
    ap.add_argument("--folds",       type=int, default=5)
    ap.add_argument("--tile-size",   type=int, default=512)
    ap.add_argument("--predict-map", action="store_true")
    ap.add_argument("--output",      default=None)
    run(ap.parse_args())
