#!/usr/bin/env python3
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.cv_pipeline import (
    LOSS_WEIGHTS, N_CHANNELS,
    compute_band_stats, extract_patches,
    load_point_coords, world_to_pixel,
    NutrientPatchDataset, build_model,
    weighted_huber_loss, train_epoch, eval_epoch, eval_r2_per_target,
    spatial_cv_splits, predict_full_map_tiled, save_nutrient_geotiff,
)


def _loader(patches, labels, patch_size, batch, augment):
    ds = NutrientPatchDataset(patches, labels, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=augment,
                      num_workers=0, pin_memory=False)


def _train_loop(model, loader, opt, sch, weights, device, epochs, patience, fp16, tag=""):
    scaler = GradScaler() if fp16 else None
    best   = np.inf; pat = 0; best_state = None

    for ep in range(1, epochs+1):
        if fp16:
            model.train(); tloss = 0.0
            for patches, labels in loader:
                patches, labels = patches.to(device), labels.to(device)
                opt.zero_grad()
                with autocast():
                    out  = model(patches)
                    cp   = out[:, :, out.shape[2]//2, out.shape[3]//2]
                    loss = weighted_huber_loss(cp.float(), labels, weights)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                tloss += loss.item()
            tloss /= max(len(loader), 1)
        else:
            tloss = train_epoch(model, loader, opt, weights, device)

        sch.step()
        if tloss < best:
            best = tloss; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f"  {tag} early stop @ ep {ep}"); break
        if ep % 10 == 0 or ep == 1:
            print(f"  {tag} ep {ep:>3}  loss={tloss:.4f}  best={best:.4f}")

    if best_state: model.load_state_dict(best_state)
    return model


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
    import rasterio
    cfg    = load_config(args.config)
    out    = args.output or "results/10_cv_local"
    os.makedirs(out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16   = device.type == "cuda" and args.fp16
    print(f"Device: {device}  fp16={fp16}")

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

    w_arr   = np.array([LOSS_WEIGHTS.get(e, 0.5) for e in avail], dtype=np.float32)
    w_arr   = w_arr / w_arr.sum() * n_targets
    weights = torch.tensor(w_arr).to(device)

    print("Computing band stats...")
    band_stats = compute_band_stats(args.ortho)

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
    Y             = chem_sub[avail].values.astype(np.float32)

    print(f"Extracting {len(common)} patches {args.patch_size}×{args.patch_size} (windowed reads)...")
    all_patches = extract_patches(args.ortho, rows_px, cpx,
                                   args.patch_size, band_stats)
    print(f"  patches: {all_patches.shape}  ({all_patches.nbytes/1e6:.0f} MB)")

    splits = spatial_cv_splits(coords_sub, n_folds=args.folds)
    cv_res = []

    print(f"\nSpatial {args.folds}-fold CV  epochs={args.epochs}  patch={args.patch_size}")
    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(f"\n[Fold {fold+1}]  train={len(tr_idx)}  val={len(val_idx)}")
        tr_loader  = _loader(all_patches[tr_idx],  Y[tr_idx],  args.patch_size, args.batch, True)
        val_loader = _loader(all_patches[val_idx], Y[val_idx], args.patch_size, args.batch, False)
        if len(val_loader.dataset) == 0: print("  empty val, skip"); continue

        model = build_model(n_targets, not args.no_pretrain).to(device)
        if fp16: model = model.half()
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=max(10, args.epochs//5), T_mult=2)

        model = _train_loop(model, tr_loader, opt, sch, weights, device,
                            args.epochs, args.patience, fp16, tag=f"F{fold+1}")

        r2s = eval_r2_per_target(model, val_loader, device, n_targets)
        row = {"fold": fold+1}
        for i, e in enumerate(avail):
            row[e] = r2s[i]; print(f"  {short_name(cfg,e):<6}: R²={r2s[i]:+.3f}")
        cv_res.append(row)
        torch.save(model.state_dict(), f"{out}/ckpt_fold{fold+1}.pt")

    if cv_res:
        cv_df = pd.DataFrame(cv_res)
        cv_df.to_csv(f"{out}/cv_r2_folds.csv", index=False, float_format="%.4f")
        print(f"\nCV mean R²:")
        for e in avail:
            vals = cv_df[e].dropna().values
            print(f"  {short_name(cfg,e):<6}: {np.mean(vals):+.3f} ± {np.std(vals):.3f}")

    print("\nFinal model on all data...")
    full_loader = _loader(all_patches, Y, args.patch_size, args.batch, True)
    final = build_model(n_targets, not args.no_pretrain).to(device)
    if fp16: final = final.half()
    opt_f = torch.optim.AdamW(final.parameters(), lr=args.lr, weight_decay=1e-4)
    sch_f = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_f, T_0=max(10, args.epochs//5), T_mult=2)
    final = _train_loop(final, full_loader, opt_f, sch_f, weights, device,
                        args.epochs, args.patience, fp16, tag="final")
    torch.save(final.state_dict(), f"{out}/model_final.pt")

    if args.predict_map:
        print("\nPredicting full map (tiled)...")
        pred = predict_full_map_tiled(
            final, args.ortho, band_stats, meta, device,
            tile=args.tile_size, overlap=args.tile_size//4,
            batch_size=args.batch, fp16=fp16)
        save_nutrient_geotiff(pred, meta, f"{out}/nutrients_{dk}.tif", avail)
        plot_maps(pred, avail, out, cfg["plots"]["dpi"])

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="config.yaml")
    ap.add_argument("--ortho",       required=True)
    ap.add_argument("--date",        default="date2")
    ap.add_argument("--elements",    nargs="+")
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--batch",       type=int,   default=8)
    ap.add_argument("--patch-size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--patience",    type=int,   default=20)
    ap.add_argument("--folds",       type=int,   default=5)
    ap.add_argument("--tile-size",   type=int,   default=512)
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--fp16",        action="store_true")
    ap.add_argument("--predict-map", action="store_true")
    ap.add_argument("--output",      default=None)
    run(ap.parse_args())
