"""cv_pipeline — tiled windowed approach, never loads full raster into memory."""
import numpy as np
import pandas as pd
from pathlib import Path

import rasterio
import rasterio.transform as rtf
import rasterio.windows as riow
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import segmentation_models_pytorch as smp

BAND_ORDER = ("Blue", "Green", "Red", "RedEdge", "NIR")

SELECTED_INDICES = [
    "NDRE", "CCCI", "MSAVI2", "MCARI", "TCARI_OSAVI",
    "SRPI", "ARI", "MCARI_OSAVI", "NPCI", "Datt", "CRI700", "CIre",
]

LOSS_WEIGHTS = {
    "Ca_%":          0.763, "Co_мг_кг":      0.751,
    "Cu_мг_кг":      0.615, "Fe_мг_кг":      0.565,
    "K_%":           0.757, "Mg_%":          0.679,
    "Mn_мг_кг":      0.768, "N_%":           0.725,
    "P_%":           0.753, "S_%":           0.709,
    "Zn_мг_кг":      0.753, "Нитраты_мг_кг": 0.586,
}

N_CHANNELS = len(BAND_ORDER) + len(SELECTED_INDICES)   # 17


# ── Label normalisation (per-nutrient z-score, NaN-safe) ─────────────────────

class LabelScaler:
    """
    Per-column z-score normalisation that handles NaN values.
    Fit on raw Y matrix, transform before training, inverse_transform after inference.
    """
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, Y):
        self.mean_ = np.nanmean(Y, axis=0)
        self.std_  = np.nanstd(Y,  axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, Y):
        out = (Y - self.mean_) / self.std_
        return out.astype(np.float32)

    def inverse_transform(self, Y_scaled):
        return Y_scaled * self.std_ + self.mean_

    def inverse_transform_map(self, pred_map):
        """pred_map: (T, H, W) → denormalised (T, H, W)."""
        out = np.empty_like(pred_map)
        for t in range(pred_map.shape[0]):
            out[t] = pred_map[t] * self.std_[t] + self.mean_[t]
        return out



# ── Band statistics ───────────────────────────────────────────────────────────

def compute_band_stats(tif_path, n_strips=40):
    """Per-band (p2, p98) by sampling n_strips horizontal strips."""
    with rasterio.open(tif_path) as src:
        H, W  = src.height, src.width
        step  = max(H // n_strips, 1)
        accum = [[] for _ in range(src.count)]
        for r in range(0, H, step):
            win  = riow.Window(0, r, W, 1)
            data = src.read(window=win)
            for i in range(src.count):
                v = data[i, 0, :].astype(np.float32)
                v = v[v > 0]
                if len(v) > 0:
                    accum[i].append(v)
    stats = []
    for i in range(len(accum)):
        if not accum[i]:
            stats.append((0.0, 1.0)); continue
        vals = np.concatenate(accum[i])
        stats.append((float(np.percentile(vals, 2)),
                      float(np.percentile(vals, 98))))
    return stats


# ── Tile I/O ──────────────────────────────────────────────────────────────────

def _read_window(src, r0, c0, h, w):
    win = riow.Window(col_off=c0, row_off=r0, width=w, height=h)
    return src.read(window=win)   # (C, h, w) uint16


def _normalise(raw, band_stats):
    out = {}
    for i, name in enumerate(BAND_ORDER):
        lo, hi = band_stats[i]
        arr = np.clip((raw[i].astype(np.float32) - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        arr[raw[i] == 0] = 0.0
        out[name] = arr
    return out


# ── Index computation (works on any-size arrays) ──────────────────────────────

def _safe(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan).astype(np.float32)


def _recip(a):
    out = np.empty_like(a, dtype=np.float32)
    nz  = a != 0
    out[nz]  = np.float32(1.0) / a[nz]
    out[~nz] = np.nan
    return out


def compute_index_maps(bands):
    B  = bands["Blue"];  G  = bands["Green"]
    R  = bands["Red"];   RE = bands["RedEdge"]; N = bands["NIR"]
    f  = np.float32

    ndvi  = _safe(N-R,  N+R)
    ndre  = _safe(N-RE, N+RE)
    osavi = _safe(f(1.16)*(N-R),  N+R+f(0.16))
    tcari = f(3)*((RE-R) - f(0.2)*(RE-G)*_safe(RE, R))
    mcari = ((RE-R) - f(0.2)*(RE-G)) * _safe(RE, R)
    d     = (f(2)*N+f(1))**2 - f(8)*(N-R)
    ari    = np.where((G!=0)&(RE!=0), _recip(G)-_recip(RE), np.nan).astype(np.float32)
    cri700 = np.where((B!=0)&(RE!=0), _recip(B)-_recip(RE), np.nan).astype(np.float32)

    return {
        "NDRE":        ndre,
        "CCCI":        _safe(ndre, ndvi),
        "MSAVI2":      np.where(d>=0, (f(2)*N+f(1)-np.sqrt(np.maximum(d,f(0))))/f(2), np.nan).astype(np.float32),
        "MCARI":       mcari,
        "TCARI_OSAVI": _safe(tcari, osavi),
        "SRPI":        _safe(B, R),
        "ARI":         ari,
        "MCARI_OSAVI": _safe(mcari, osavi),
        "NPCI":        _safe(R-B, R+B),
        "Datt":        _safe(N-RE, N-R),
        "CRI700":      cri700,
        "CIre":        (_safe(N, RE)-f(1)),
    }


def _bands_to_tensor(bands):
    """dict of (h,w) → (17, h, w) float32."""
    idx   = compute_index_maps(bands)
    chans = [bands[n] for n in BAND_ORDER]
    for name in SELECTED_INDICES:
        arr = idx.get(name, np.zeros_like(bands["Blue"]))
        chans.append(np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -5, 5).astype(np.float32))
    return np.stack(chans, axis=0)


def _read_tile_tensor(src, r0, c0, h, w, band_stats):
    return _bands_to_tensor(_normalise(_read_window(src, r0, c0, h, w), band_stats))


# ── Point / patch extraction ──────────────────────────────────────────────────

def load_point_coords(gpkg_path, target_crs=None):
    gdf = gpd.read_file(gpkg_path)
    if target_crs and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return (gdf["id"].values.astype(int),
            np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry]))


def world_to_pixel(coords_xy, transform):
    rows, cols = rtf.rowcol(transform, coords_xy[:, 0], coords_xy[:, 1])
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def extract_point_features(tif_path, rows_px, cols_px, band_stats):
    """1×1 windowed reads at each point → (N, 17) float32. Tiny memory."""
    feats = []
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        for r, c in zip(rows_px, cols_px):
            r = int(np.clip(r, 0, H-1)); c = int(np.clip(c, 0, W-1))
            t = _read_tile_tensor(src, r, c, 1, 1, band_stats)
            feats.append(t[:, 0, 0])
    return np.array(feats, dtype=np.float32)


def extract_patches(tif_path, rows_px, cols_px, patch_size, band_stats):
    """Windowed reads → (N, 17, P, P) float32. Memory O(N × P²)."""
    p2 = patch_size // 2
    patches = []
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        for r, c in zip(rows_px, cols_px):
            r0 = int(np.clip(r-p2, 0, H-patch_size))
            c0 = int(np.clip(c-p2, 0, W-patch_size))
            patches.append(_read_tile_tensor(src, r0, c0, patch_size, patch_size, band_stats))
    return np.array(patches, dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NutrientPatchDataset:
    def __init__(self, patches, labels, augment=True, mixup_alpha=0.4, seed=42):
        self.rng    = np.random.default_rng(seed)
        self.alpha  = mixup_alpha
        self.augment= augment
        self._build(patches, labels)

    def _aug(self, p):
        p = np.rot90(p, k=self.rng.integers(0, 4), axes=(1, 2))
        if self.rng.random() > 0.5: p = np.flip(p, axis=2).copy()
        if self.rng.random() > 0.5: p = np.flip(p, axis=1).copy()
        return p

    def _build(self, patches, labels):
        pts, lbls = [], []
        for i in range(len(patches)):
            base = patches[i]; lbl = labels[i]
            if self.augment:
                for _ in range(8):
                    pts.append(self._aug(base.copy())); lbls.append(lbl)
                for _ in range(4):
                    pts.append(base + self.rng.normal(0, 0.01, base.shape).astype(np.float32))
                    lbls.append(lbl)
            else:
                pts.append(base); lbls.append(lbl)

        self.patches   = np.array(pts,  dtype=np.float32)
        self.label_arr = np.array(lbls, dtype=np.float32)

        if self.augment and self.alpha > 0:
            n   = len(self.patches); nm = n // 2
            ia  = self.rng.integers(0, n, nm); ib = self.rng.integers(0, n, nm)
            lam = self.rng.beta(self.alpha, self.alpha, nm).astype(np.float32)
            mp  = lam[:,None,None,None]*self.patches[ia] + (1-lam[:,None,None,None])*self.patches[ib]
            # MixUp labels: if either parent is NaN for a target, result is NaN
            la  = self.label_arr[ia]; lb = self.label_arr[ib]
            la0 = np.where(np.isfinite(la), la, 0.0)
            lb0 = np.where(np.isfinite(lb), lb, 0.0)
            ml  = lam[:,None]*la0 + (1-lam[:,None])*lb0
            # Restore NaN where either parent was NaN
            ml  = np.where(np.isfinite(la) & np.isfinite(lb), ml, np.nan).astype(np.float32)
            self.patches   = np.concatenate([self.patches,   mp])
            self.label_arr = np.concatenate([self.label_arr, ml])

    def __len__(self):        return len(self.patches)
    def __getitem__(self, i): return self.patches[i], self.label_arr[i]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_targets=12, pretrained=True):
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet" if pretrained else None,
        in_channels     = N_CHANNELS,
        classes         = n_targets,
        activation      = None,
    )


# ── Loss & training ───────────────────────────────────────────────────────────

def weighted_huber_loss(pred, target, weights, delta=1.0):
    # Replace NaN in target with 0 before huber (avoids nan*anything=nan in loss)
    target_clean = torch.nan_to_num(target, nan=0.0)
    loss  = F.huber_loss(pred, target_clean, delta=delta, reduction="none")
    # Mask: valid only where BOTH pred and target are finite
    valid = torch.isfinite(target) & torch.isfinite(pred)
    denom = valid.float().sum()
    if denom == 0:
        return pred.sum() * 0.0   # zero loss, zero grad, no NaN
    return (loss * valid.float() * weights.unsqueeze(0)).sum() / denom


def _centre(out):
    return out[:, :, out.shape[2]//2, out.shape[3]//2]


def train_epoch(model, loader, optimiser, weights, device):
    model.train()
    total, n, skipped = 0.0, 0, 0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device)
        # Clean inputs — eliminate any residual NaN/Inf from patch extraction
        patches = torch.nan_to_num(patches, nan=0.0, posinf=1.0, neginf=-1.0)
        optimiser.zero_grad()
        loss = weighted_huber_loss(_centre(model(patches)), labels, weights)
        if not torch.isfinite(loss):
            skipped += 1; continue
        loss.backward()
        # Aggressive clipping to prevent gradient explosion on adapted 17-ch input
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()
        total += loss.item(); n += 1
    if skipped > 0:
        print(f"    [warn] {skipped} NaN-loss batches skipped", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, weights, device):
    model.eval()
    total, n = 0.0, 0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device)
        patches = torch.nan_to_num(patches, nan=0.0, posinf=1.0, neginf=-1.0)
        loss = weighted_huber_loss(_centre(model(patches)), labels, weights)
        if torch.isfinite(loss):
            total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_r2_per_target(model, loader, device, n_targets):
    model.eval()
    pred_all = [[] for _ in range(n_targets)]
    true_all = [[] for _ in range(n_targets)]
    for patches, labels in loader:
        cp  = _centre(model(patches.to(device))).cpu().numpy()
        lbl = labels.numpy()
        for t in range(n_targets):
            m = np.isfinite(lbl[:, t])
            pred_all[t].extend(cp[m, t]); true_all[t].extend(lbl[m, t])
    return np.array([
        r2_score(true_all[t], pred_all[t]) if len(true_all[t]) > 2 else np.nan
        for t in range(n_targets)
    ])


# ── Tiled inference ───────────────────────────────────────────────────────────

def predict_full_map_tiled(model, tif_path, band_stats, meta, device,
                            tile=512, overlap=64, batch_size=4, fp16=False):
    H, W  = meta["height"], meta["width"]
    step  = tile - overlap
    win2d = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)

    model.eval()
    ctx = torch.cuda.amp.autocast() if fp16 else torch.no_grad()
    with torch.no_grad(), ctx:
        dummy = torch.zeros(1, N_CHANNELS, tile, tile, device=device)
        n_targets = model(dummy).shape[1]

    accum  = np.zeros((n_targets, H, W), dtype=np.float32)
    weight = np.zeros((H, W),            dtype=np.float32)
    rs = sorted({max(r,0) for r in range(0, H-tile+1, step)} | {max(H-tile,0)})
    cs = sorted({max(c,0) for c in range(0, W-tile+1, step)} | {max(W-tile,0)})

    batch_t, batch_p = [], []

    def _flush():
        if not batch_t: return
        xb = torch.from_numpy(np.stack(batch_t)).to(device)
        ctx = torch.cuda.amp.autocast() if fp16 else torch.no_grad()
        with ctx:
            pb = model(xb).float().cpu().numpy()
        for (r0, c0), p in zip(batch_p, pb):
            accum[:, r0:r0+tile, c0:c0+tile] += p * win2d
            weight[r0:r0+tile, c0:c0+tile]   += win2d
        batch_t.clear(); batch_p.clear()

    total = len(rs)*len(cs); done = 0
    with rasterio.open(tif_path) as src:
        for r0 in rs:
            for c0 in cs:
                h = min(tile, H-r0); w = min(tile, W-c0)
                t = np.zeros((N_CHANNELS, tile, tile), dtype=np.float32)
                t[:, :h, :w] = _read_tile_tensor(src, r0, c0, h, w, band_stats)
                batch_t.append(t); batch_p.append((r0, c0))
                if len(batch_t) >= batch_size: _flush()
                done += 1
                if done % 50 == 0: print(f"  {done}/{total} tiles", end="\r", flush=True)
    _flush()
    print(f"  {total}/{total} tiles done    ")
    return accum / np.where(weight > 0, weight, 1.0)


def apply_gbr_tiled(tif_path, models, scalers, target_names, band_stats, meta,
                    out_path, tile_size=512):
    """
    Predict band by band to avoid rasterio windowed-write + LZW conflict.
    Each band is accumulated row-by-row in a (H,) strip buffer and written once per row.
    Peak memory: O(W * tile_size * n_models) — well under 1 GB.
    """
    H, W      = meta["height"], meta["width"]
    n_targets = len(target_names)

    # No compression — windowed writes + LZW cause GDAL dirty-block errors
    out_meta  = {**meta, "count": n_targets, "dtype": "float32"}
    out_meta.pop("compress", None)

    print(f"  Writing {n_targets} bands to {out_path}")
    with rasterio.open(out_path, "w", **out_meta) as dst:
        for i, name in enumerate(target_names):
            dst.update_tags(i+1, nutrient=name)

    # Process and write one row-strip at a time, one band at a time
    total = ((H + tile_size - 1) // tile_size); done = 0
    with rasterio.open(out_path, "r+") as dst:
        with rasterio.open(tif_path) as src:
            for r0 in range(0, H, tile_size):
                h   = min(tile_size, H - r0)
                # Read all columns for this strip
                raw = _read_window(src, r0, 0, h, W)
                bands_strip = _normalise(raw, band_stats)
                t   = _bands_to_tensor(bands_strip)        # (17, h, W)
                X   = t.reshape(N_CHANNELS, -1).T          # (h*W, 17)
                valid = np.isfinite(X).all(axis=1) & (X[:, :5] != 0).all(axis=1)

                for j, (m, sc) in enumerate(zip(models, scalers)):
                    if m is None: continue
                    row = np.full(X.shape[0], np.nan, dtype=np.float32)
                    if valid.sum() > 0:
                        row[valid] = m.predict(sc.transform(X[valid])).astype(np.float32)
                    win = riow.Window(col_off=0, row_off=r0, width=W, height=h)
                    dst.write(row.reshape(1, h, W), indexes=[j+1], window=win)

                done += 1
                print(f"  strip {done}/{total}", end="\r", flush=True)

    print(f"  {total}/{total} strips done    ")
    print(f"  saved: {out_path}")


# ── Save / utils ──────────────────────────────────────────────────────────────

def save_nutrient_geotiff(pred_map, meta, out_path, target_names):
    out_meta = {**meta, "count": pred_map.shape[0], "dtype": "float32", "compress": "lzw"}
    with rasterio.open(out_path, "w", **out_meta) as dst:
        for i, name in enumerate(target_names):
            dst.write(pred_map[i], i+1)
            dst.update_tags(i+1, nutrient=name)
    print(f"  saved: {out_path}")


def spatial_cv_splits(coords_xy, n_folds=5, seed=42):
    fids = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(coords_xy)
    return [(np.where(fids != f)[0], np.where(fids == f)[0]) for f in range(n_folds)]


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "RPD": np.nan}
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return {
        "R2":   round(r2_score(yt, yp), 4),
        "RMSE": round(rmse, 4),
        "MAE":  round(mean_absolute_error(yt, yp), 4),
        "RPD":  round(np.std(yt) / (rmse + 1e-12), 2),
    }
