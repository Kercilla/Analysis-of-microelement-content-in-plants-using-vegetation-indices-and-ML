#!/usr/bin/env python3
"""
11_test_unet.py — evaluate trained ResUNet on labeled points.

Loads model_final.pt, extracts predictions at labeled point locations,
computes R²/RMSE/RPD vs ground truth chemistry, and generates scatter plots
+ side-by-side comparison with GBR baseline.

    python 11_test_unet.py \
        --model results/10_cv_local/model_final.pt \
        --pred-map results/10_cv_local/nutrients_date2.tif \
        --baseline results/09_cv_baseline/baseline_cv_metrics.csv
"""
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.cv_pipeline import (
    LOSS_WEIGHTS, N_CHANNELS,
    compute_band_stats, extract_patches,
    load_point_coords, world_to_pixel,
    build_model, eval_r2_per_target,
    NutrientPatchDataset, LabelScaler,
    spatial_cv_splits, compute_metrics,
)
from torch.utils.data import DataLoader


def load_pred_at_points(pred_tif, rows_px, cols_px):
    """
    Extract predicted values at each labeled point from GeoTIFF.
    Returns (N, T) float32 array.
    """
    with rasterio.open(pred_tif) as src:
        H, W  = src.height, src.width
        n_bands = src.count
        preds = np.full((len(rows_px), n_bands), np.nan, dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows_px, cols_px)):
            r = int(np.clip(r, 0, H-1))
            c = int(np.clip(c, 0, W-1))
            win = rasterio.windows.Window(c, r, 1, 1)
            data = src.read(window=win)   # (T, 1, 1)
            preds[i] = data[:, 0, 0]
    return preds


def plot_scatter_comparison(y_true, y_pred_unet, y_pred_gbr,
                            target_names, cfg, out_dir, dpi=150):
    """Scatter plots: ResUNet vs GBR for each nutrient."""
    n = len(target_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    for tag, y_pred in [("unet", y_pred_unet), ("gbr", y_pred_gbr)]:
        if y_pred is None:
            continue
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
        axes = axes.flatten()
        for i, name in enumerate(target_names):
            ax = axes[i]
            yt = y_true[:, i]
            yp = y_pred[:, i] if y_pred.ndim == 2 else y_pred[:, i]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() < 3:
                ax.set_visible(False); continue
            m = compute_metrics(yt[mask], yp[mask])
            ax.scatter(yt[mask], yp[mask], alpha=0.6, s=30,
                       c="#2E75B6" if tag == "unet" else "#E06666",
                       edgecolors="none")
            lims = [min(yt[mask].min(), yp[mask].min()),
                    max(yt[mask].max(), yp[mask].max())]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
            if mask.sum() > 2:
                xl = np.linspace(*lims, 50)
                ax.plot(xl, np.poly1d(np.polyfit(yt[mask], yp[mask], 1))(xl),
                        color="#C00000", lw=1.5)
            sn = short_name(cfg, name)
            ax.set_title(sn, fontsize=9)
            ax.text(0.05, 0.95,
                    f"R²={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}\nRPD={m['RPD']:.2f}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
            ax.set_xlabel("Observed", fontsize=8)
            ax.set_ylabel("Predicted", fontsize=8)
            ax.tick_params(labelsize=7)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        label = "ResUNet" if tag == "unet" else "GBR Baseline"
        plt.suptitle(f"{label} — Predicted vs Observed at 100 points", fontsize=12)
        plt.tight_layout()
        fig.savefig(Path(out_dir) / f"scatter_{tag}.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: scatter_{tag}.png")


def plot_side_by_side(pred_tif_unet, pred_tif_gbr, target_names, out_dir, dpi=120):
    """Side-by-side map comparison for each nutrient."""
    if not pred_tif_gbr or not Path(pred_tif_gbr).exists():
        print("  GBR tif not found, skipping side-by-side")
        return

    cmaps = ["RdYlGn","YlOrRd","Blues","Greens","PuRd",
             "Oranges","BuGn","RdPu","YlGn","BuPu","GnBu","OrRd"]

    with rasterio.open(pred_tif_unet) as u, rasterio.open(pred_tif_gbr) as g:
        for i, name in enumerate(target_names):
            bu = u.read(i+1).astype(np.float32)
            bg = g.read(i+1).astype(np.float32) if g.count > i else None

            vu = bu[np.isfinite(bu) & (bu != 0)]
            if len(vu) == 0:
                continue
            vmin = np.percentile(vu, 2)
            vmax = np.percentile(vu, 98)

            ncols = 2 if bg is not None else 1
            fig, axes = plt.subplots(1, ncols, figsize=(10*ncols, 8))
            if ncols == 1:
                axes = [axes]

            axes[0].imshow(bu, cmap=cmaps[i % len(cmaps)],
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[0].set_title("ResUNet", fontsize=11)
            axes[0].axis("off")

            if bg is not None:
                im = axes[1].imshow(bg, cmap=cmaps[i % len(cmaps)],
                                    vmin=vmin, vmax=vmax, interpolation="nearest")
                axes[1].set_title("GBR Baseline", fontsize=11)
                axes[1].axis("off")
                plt.colorbar(im, ax=axes, label=name, shrink=0.6)

            plt.suptitle(name, fontsize=13, fontweight="bold")
            plt.tight_layout()
            fn = name.replace("%","pct").replace("/","_").replace(" ","_")
            fig.savefig(Path(out_dir) / f"compare_{fn}.png",
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    print(f"  saved {len(target_names)} comparison maps")


def run(args):
    cfg   = load_config(args.config)
    out   = args.output or "results/11_test_unet"
    os.makedirs(out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    elems = args.elements or cfg["chemistry"]["target_elements"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dk    = args.date or "date2"
    sk    = dsmap.get(dk, "sampling2")
    chem  = load_chemistry(cfg["paths"]["chemistry"][sk],
                           cfg["chemistry"]["columns"],
                           cfg["chemistry"]["id_offsets"].get(sk, 0))
    avail     = [e for e in elems if e in chem.columns]
    n_targets = len(avail)
    print(f"Targets ({n_targets}): {avail}")

    # ── Band stats + meta ──
    print("Computing band stats...")
    band_stats = compute_band_stats(args.ortho)
    with rasterio.open(args.ortho) as src:
        meta = src.meta.copy()

    # ── Point coordinates ──
    gpkg_files = sorted(glob(cfg["paths"]["multi"].get(dk)))
    if not gpkg_files:
        print("No gpkg found"); return
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
    Y             = chem_sub[avail].values.astype(np.float32)
    print(f"Labeled points: {len(common)}")

    # ── Load patches + model ──
    print("Extracting patches...")
    patches = extract_patches(args.ortho, rows_px, cpx, args.patch_size, band_stats)
    patches = np.nan_to_num(patches, nan=0.0, posinf=1.0, neginf=-1.0)

    label_scaler = LabelScaler().fit(Y)
    Y_scaled     = label_scaler.transform(Y)

    print(f"Loading model: {args.model}")
    model = build_model(n_targets, pretrained=False).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Evaluate on all points ──
    ds     = NutrientPatchDataset(patches, Y_scaled, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_pred, all_true = [], []
    with torch.no_grad():
        for p, l in loader:
            p = p.to(device)
            out_t = model(p)
            cp    = out_t[:, :, out_t.shape[2]//2, out_t.shape[3]//2]
            all_pred.append(cp.cpu().numpy())
            all_true.append(l.numpy())

    y_pred_scaled = np.concatenate(all_pred)   # (N, T)
    y_true_scaled = np.concatenate(all_true)   # (N, T)

    # Denormalise
    y_pred = label_scaler.inverse_transform(y_pred_scaled)
    y_true = label_scaler.inverse_transform(
        np.where(np.isfinite(y_true_scaled), y_true_scaled, 0.0))
    # Restore NaN in y_true
    y_true = np.where(np.isfinite(Y), y_true, np.nan)

    # ── Metrics ──
    print(f"\n{'='*60}")
    print(f"{'Nutrient':<20} {'R2':>6} {'RMSE':>8} {'RPD':>6} {'n':>5}")
    print("-"*44)
    results = []
    for j, name in enumerate(avail):
        yt = y_true[:, j]; yp = y_pred[:, j]
        m  = compute_metrics(yt[np.isfinite(yt)], yp[np.isfinite(yt)])
        sn = short_name(cfg, name)
        print(f"{sn:<20} {m['R2']:>+6.3f} {m['RMSE']:>8.4f} "
              f"{m['RPD']:>6.2f} {np.isfinite(yt).sum():>5}")
        results.append({"element": name, **m})

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{out}/unet_point_metrics.csv",
                      index=False, float_format="%.4f")

    # ── Load GBR baseline metrics for comparison ──
    df_gbr = None
    if args.baseline and Path(args.baseline).exists():
        df_gbr = pd.read_csv(args.baseline).set_index("element")
        df_unet = df_results.set_index("element")
        print(f"\n{'='*60}")
        print(f"{'Nutrient':<20} {'GBR R2':>8} {'UNet R2':>8} {'Δ':>6}")
        print("-"*46)
        for name in avail:
            sn   = short_name(cfg, name)
            r2g  = df_gbr.loc[name, "R2"] if name in df_gbr.index else np.nan
            r2u  = df_unet.loc[name, "R2"] if name in df_unet.index else np.nan
            d    = r2u - r2g if (np.isfinite(r2g) and np.isfinite(r2u)) else np.nan
            mark = "↑" if d > 0.02 else ("↓" if d < -0.02 else "≈")
            print(f"{sn:<20} {r2g:>+8.3f} {r2u:>+8.3f} {d:>+6.3f} {mark}")

    # ── GBR predictions at points (from tif) ──
    y_pred_gbr = None
    if args.baseline_tif and Path(args.baseline_tif).exists():
        print("\nLoading GBR predictions at points from tif...")
        y_pred_gbr = load_pred_at_points(args.baseline_tif, rows_px, cpx)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_scatter_comparison(y_true, y_pred, y_pred_gbr, avail, cfg, out)
    plot_side_by_side(args.pred_map, args.baseline_tif, avail, out)

    # ── Prediction range stats from full map ──
    if args.pred_map and Path(args.pred_map).exists():
        print(f"\nPrediction map stats ({args.pred_map}):")
        with rasterio.open(args.pred_map) as src:
            for j, name in enumerate(avail):
                band  = src.read(j+1).astype(np.float32)
                valid = band[np.isfinite(band) & (band != 0)]
                if len(valid) == 0:
                    continue
                sn = short_name(cfg, name)
                ref_mean = np.nanmean(Y[:, j])
                print(f"  {sn:<6}: map=[{valid.min():.3f}, {valid.max():.3f}] "
                      f"mean={valid.mean():.3f}  ref_mean={ref_mean:.3f}")

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       default="config.yaml")
    ap.add_argument("--ortho",        required=True,
                    help="path to orthomosaic GeoTIFF")
    ap.add_argument("--model",        required=True,
                    help="path to model_final.pt")
    ap.add_argument("--pred-map",     required=True,
                    help="path to nutrients_date2.tif (ResUNet output)")
    ap.add_argument("--baseline",     default=None,
                    help="path to baseline_cv_metrics.csv")
    ap.add_argument("--baseline-tif", default=None,
                    help="path to baseline_nutrients.tif for map comparison")
    ap.add_argument("--date",         default="date2")
    ap.add_argument("--elements",     nargs="+")
    ap.add_argument("--patch-size",   type=int, default=128)
    ap.add_argument("--output",       default=None)
    run(ap.parse_args())
