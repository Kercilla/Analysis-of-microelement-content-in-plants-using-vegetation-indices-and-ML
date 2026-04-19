#!/usr/bin/env python3
"""
08. Pixel-wise nutrient mapping via U-Net regression.

Requires: orthomosaic GeoTIFF + labeled point coordinates + chemistry.

    python 08_unet_map.py --ortho data/ortho_multi_0529.tif --date date1
    python 08_unet_map.py --ortho data/ortho_hyper_0529.tif --date date1 --bands 1-50
"""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry


def _load_point_coords(gpkg_path, target_crs=None):
    """Extract point centroids, reproject to target CRS if needed."""
    import geopandas as gpd
    gdf = gpd.read_file(gpkg_path)
    if target_crs and str(gdf.crs) != str(target_crs):
        print(f"  reprojecting points: {gdf.crs} -> {target_crs}")
        gdf = gdf.to_crs(target_crs)
    coords = np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry])
    ids = gdf["id"].values.astype(int)
    return ids, coords


def run(args):
    from analysis.unet_pipeline import train_unet, predict_map, save_geotiff

    cfg = load_config(args.config)
    out = args.output or "results/08_unet"
    os.makedirs(out, exist_ok=True)

    cols = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dk = args.date or "date1"

    # Chemistry
    sk = dsmap.get(dk, "sampling1")
    for k, v in dsmap.items():
        if k in dk or dk in k:
            sk = v; break
    cp = cfg["paths"]["chemistry"].get(sk)
    off = cfg["chemistry"]["id_offsets"].get(sk, 0)
    chem = load_chemistry(cp, cols, off)

    # Point coordinates from gpkg (reprojected to ortho CRS)
    import rasterio as rio
    with rio.open(args.ortho) as src:
        ortho_crs = src.crs

    gpkg_pattern = cfg["paths"]["multi"].get(dk)
    if gpkg_pattern:
        from glob import glob
        gpkg_files = sorted(glob(gpkg_pattern))
        if gpkg_files:
            ids, coords = _load_point_coords(gpkg_files[0], target_crs=ortho_crs)
        else:
            print("no gpkg files for coordinates"); return
    else:
        print("no multi path in config for coordinates"); return

    # Match IDs
    common = np.intersect1d(ids, chem.index.values)
    idx_mask = np.isin(ids, common)
    pts_xy = coords[idx_mask]
    chem_common = chem.loc[ids[idx_mask]]

    avail = [e for e in elems if e in chem_common.columns]
    labels = chem_common[avail].values
    target_names = [short_name(cfg, e) for e in avail]

    # Handle NaN in labels: replace with column mean
    for j in range(labels.shape[1]):
        col_mean = np.nanmean(labels[:, j])
        labels[np.isnan(labels[:, j]), j] = col_mean

    print(f"[{dk}] {len(common)} points, {len(avail)} targets")
    print(f"  targets: {target_names}")
    print(f"  ortho: {args.ortho}")

    # Parse band selection
    band_idx = None
    if args.bands:
        parts = args.bands.split("-")
        if len(parts) == 2:
            band_idx = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            band_idx = [int(b) for b in args.bands.split(",")]
        print(f"  bands: {len(band_idx)} selected")

    # Train
    model, meta = train_unet(
        args.ortho, pts_xy, labels,
        target_names=target_names,
        patch_size=args.patch_size,
        n_augments=args.n_aug,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        band_indices=band_idx,
    )

    # Predict full map
    print("\nInference...")
    pred, unc = predict_map(
        model, args.ortho, meta,
        tile_size=args.patch_size * 2,
        overlap=args.patch_size // 2,
        batch_size=args.batch,
        mc_dropout=args.mc_dropout,
    )

    # Save
    out_tif = Path(out) / f"nutrients_{dk}.tif"
    save_geotiff(pred, meta, str(out_tif), uncertainty=unc)

    # Quick stats
    print(f"\nPrediction stats:")
    for i, name in enumerate(target_names):
        band = pred[i]
        valid = band[band != 0]
        if len(valid) > 0:
            print(f"  {name:>5}: mean={valid.mean():.3f}, "
                  f"std={valid.std():.3f}, "
                  f"range=[{valid.min():.3f}, {valid.max():.3f}]")

    print(f"\ndone -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ortho", required=True, help="path to orthomosaic GeoTIFF")
    ap.add_argument("--date", default="date1")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--bands", default=None, help="band range: '1-5' or '1,2,3,4,5'")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--n-aug", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mc-dropout", type=int, default=0, help="MC Dropout passes (0=off)")
    ap.add_argument("--output")
    run(ap.parse_args())
