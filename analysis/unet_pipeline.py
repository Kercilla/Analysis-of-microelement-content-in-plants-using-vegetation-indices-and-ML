"""
U-Net pixel-wise regression for nutrient mapping from spectral orthomosaics.

Pipeline:
  1. Load orthomosaic GeoTIFF (5 or 300 bands)
  2. Extract NxN patches around labeled points
  3. Train U-Net with masked Huber loss (only at labeled pixels)
  4. Tile-based inference with Gaussian blending
  5. Output georeferenced GeoTIFF nutrient map
"""
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ── Dataset ──────────────────────────────────────────────

class PatchDataset(Dataset):
    """Extracts NxN patches from orthomosaic around labeled points."""

    def __init__(self, ortho_path, points_xy, labels, patch_size=64,
                 n_augments=20, band_indices=None):
        self.ps = patch_size
        self.half = patch_size // 2
        self.labels = labels
        self.n_targets = labels.shape[1] if labels.ndim > 1 else 1
        self.band_indices = band_indices
        self.ortho_path = ortho_path

        with rasterio.open(ortho_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.h, self.w = src.height, src.width
            self.n_bands = src.count if band_indices is None else len(band_indices)
            self._bands = band_indices or list(range(1, src.count + 1))

        # Compute normalization from sampled pixels (not full raster)
        self._norm_params = self._compute_norm(ortho_path, n_sample=5000)

        # Convert CRS coords -> pixel coords
        inv = ~self.transform
        self.pix_rc = []
        n_inside = 0
        for x, y in points_xy:
            col, row = inv * (x, y)
            r, c = int(row), int(col)
            if 0 <= r < self.h and 0 <= c < self.w:
                n_inside += 1
            self.pix_rc.append((r, c))

        if n_inside == 0:
            raise ValueError(f"0 points inside raster ({self.w}x{self.h}). "
                           f"Check CRS match: points should be in {self.crs}")
        if n_inside < len(points_xy):
            print(f"  WARNING: {len(points_xy) - n_inside} points outside raster")

        # Pre-extract patches (much less memory than full raster)
        self._extract_patches(n_augments)
        print(f"  patches: {len(self.patches)}, bands={self.n_bands}, "
              f"{n_inside}/{len(points_xy)} pts inside")

    def _compute_norm(self, path, n_sample=5000):
        params = []
        with rasterio.open(path) as src:
            rng = np.random.default_rng(42)
            rows = rng.integers(0, self.h, n_sample)
            cols = rng.integers(0, self.w, n_sample)
            for b in self._bands:
                vals = np.array([src.read(b, window=Window(int(c), int(r), 1, 1))[0, 0]
                                 for r, c in zip(rows[:500], cols[:500])], dtype=np.float32)
                valid = vals[vals > 0]
                if len(valid) > 10:
                    p2, p98 = np.percentile(valid, [2, 98])
                else:
                    p2, p98 = 0.0, 1.0
                params.append((float(p2), float(p98)))
        return params

    def _read_patch(self, r0, c0):
        with rasterio.open(self.ortho_path) as src:
            window = Window(c0, r0, self.ps, self.ps)
            data = src.read(self._bands, window=window).astype(np.float32)
        if data.shape[1] < self.ps or data.shape[2] < self.ps:
            padded = np.zeros((len(self._bands), self.ps, self.ps), dtype=np.float32)
            padded[:, :data.shape[1], :data.shape[2]] = data
            data = padded
        for b in range(data.shape[0]):
            p2, p98 = self._norm_params[b]
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8), 0, 1)
        return data

    def _extract_patches(self, n_augments):
        rng = np.random.default_rng(42)
        self.patches = []
        self._patch_data = []
        max_off = self.ps // 4
        for i, (r, c) in enumerate(self.pix_rc):
            if not (0 <= r < self.h and 0 <= c < self.w):
                continue
            for _ in range(n_augments):
                dr = rng.integers(-max_off, max_off + 1)
                dc = rng.integers(-max_off, max_off + 1)
                r0 = int(np.clip(r + dr - self.half, 0, self.h - self.ps))
                c0 = int(np.clip(c + dc - self.half, 0, self.w - self.ps))
                lr = r - r0
                lc = c - c0
                if 0 <= lr < self.ps and 0 <= lc < self.ps:
                    data = self._read_patch(r0, c0)
                    self.patches.append((lr, lc, i))
                    self._patch_data.append(data)

        print(f"  patches: {len(self.patches)} from {len(points_xy)} points "
              f"({n_augments} aug/pt), bands={self.n_bands}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        lr, lc, pt_idx = self.patches[idx]
        patch = self._patch_data[idx].copy()

        # Random flip/rotate
        if np.random.random() > 0.5:
            patch = patch[:, ::-1, :]
            lr = self.ps - 1 - lr
        if np.random.random() > 0.5:
            patch = patch[:, :, ::-1]
            lc = self.ps - 1 - lc
        k = np.random.randint(4)
        if k > 0:
            patch = np.rot90(patch, k, axes=(1, 2)).copy()
            for _ in range(k):
                lr, lc = lc, self.ps - 1 - lr

        # Mask: 1 at label position, 0 elsewhere
        mask = np.zeros((self.ps, self.ps), dtype=np.float32)
        # 3x3 region around label for smoother gradient
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = lr + dr, lc + dc
                if 0 <= rr < self.ps and 0 <= cc < self.ps:
                    mask[rr, cc] = 1.0

        label_map = np.zeros((self.n_targets, self.ps, self.ps), dtype=np.float32)
        lab = self.labels[pt_idx]
        if self.n_targets == 1:
            label_map[0, lr, lc] = lab
        else:
            for t in range(self.n_targets):
                label_map[t, lr, lc] = lab[t]

        return (torch.from_numpy(patch.copy()),
                torch.from_numpy(label_map),
                torch.from_numpy(mask))


# ── Model ────────────────────────────────────────────────

class SpectralReducer(nn.Module):
    """1x1 conv to reduce spectral bands: 300 -> 64 -> 32."""
    def __init__(self, in_ch, mid=64, out=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, out, 1), nn.BatchNorm2d(out), nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class _DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    """Lightweight U-Net for spectral regression. Handles 5-300 input bands."""

    def __init__(self, in_channels, n_targets, base_ch=32, reduce_to=32):
        super().__init__()
        self.reducer = SpectralReducer(in_channels, 64, reduce_to) if in_channels > 32 else nn.Identity()
        ch0 = reduce_to if in_channels > 32 else in_channels

        self.enc1 = _DoubleConv(ch0, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck = _DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _DoubleConv(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, n_targets, 1)
        self.drop = nn.Dropout2d(0.15)

    def forward(self, x):
        x = self.reducer(x)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        b = self.drop(b)

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ── Training ─────────────────────────────────────────────

def masked_huber_loss(pred, target, mask, delta=1.0):
    """Huber loss computed only where mask > 0."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear

    # Apply mask: (B, T, H, W) * (B, 1, H, W)
    m = mask.unsqueeze(1)
    masked_loss = (loss * m).sum()
    n_valid = m.sum() * pred.shape[1]
    return masked_loss / (n_valid + 1e-8)


def train_unet(ortho_path, points_xy, labels, target_names=None,
               patch_size=64, n_augments=30, epochs=100, batch_size=4,
               lr=1e-3, band_indices=None, device=None):
    """
    Train U-Net for pixel-wise nutrient regression.

    Returns trained model and dataset metadata.
    """
    if not HAS_TORCH:
        raise ImportError("pip install torch")
    if not HAS_RASTERIO:
        raise ImportError("pip install rasterio")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PatchDataset(ortho_path, points_xy, labels, patch_size,
                      n_augments, band_indices)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=True)

    n_targets = labels.shape[1] if labels.ndim > 1 else 1
    model = MiniUNet(ds.n_bands, n_targets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  training: {len(ds)} patches, {epochs} epochs, device={device}")

    best_loss = float("inf")
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for batch_x, batch_y, batch_m in dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_m = batch_m.to(device)

            pred = model(batch_x)
            loss = masked_huber_loss(pred, batch_y, batch_m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        sched.step()
        avg_loss = ep_loss / len(dl)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    ep {ep+1:>3}/{epochs}: loss={avg_loss:.4f} (best={best_loss:.4f})")

    model.load_state_dict(best_state)
    model.eval()

    meta = {
        "transform": ds.transform, "crs": ds.crs,
        "h": ds.h, "w": ds.w, "n_bands": ds.n_bands,
        "n_targets": n_targets, "target_names": target_names,
        "band_indices": band_indices,
        "norm_params": getattr(ds, "_norm_params", None),
    }
    return model, meta


# ── Inference ────────────────────────────────────────────

def predict_map(model, ortho_path, meta, tile_size=256, overlap=64,
                batch_size=4, device=None, mc_dropout=0):
    """
    Tile-based inference with Gaussian blending.

    mc_dropout > 0: run mc_dropout forward passes for uncertainty estimation.
    Returns (prediction, uncertainty) arrays.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    band_indices = meta.get("band_indices")

    with rasterio.open(ortho_path) as src:
        H, W = src.height, src.width
        bands = band_indices or list(range(1, src.count + 1))

    n_targets = meta["n_targets"]
    accum = np.zeros((n_targets, H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    # Gaussian weight kernel
    g1d = np.exp(-0.5 * np.linspace(-2, 2, tile_size) ** 2)
    gauss = np.outer(g1d, g1d).astype(np.float32)

    step = tile_size - overlap
    tiles = []
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            r0 = min(r0, max(0, H - tile_size))
            c0 = min(c0, max(0, W - tile_size))
            tiles.append((r0, c0))

    n_passes = max(1, mc_dropout)
    if mc_dropout > 0:
        model.train()  # enable dropout
    else:
        model.eval()

    all_preds = np.zeros((n_passes, n_targets, H, W), dtype=np.float32) if mc_dropout > 0 else None

    norm_params = meta.get("norm_params")

    print(f"  inference: {len(tiles)} tiles, {n_passes} passes")

    for pass_i in range(n_passes):
        with rasterio.open(ortho_path) as src:
            for ti in range(0, len(tiles), batch_size):
                batch_tiles = tiles[ti:ti + batch_size]
                batch = []
                for r0, c0 in batch_tiles:
                    window = Window(c0, r0, tile_size, tile_size)
                    data = src.read(bands, window=window).astype(np.float32)
                    # Pad if at edge
                    if data.shape[1] < tile_size or data.shape[2] < tile_size:
                        padded = np.zeros((len(bands), tile_size, tile_size), dtype=np.float32)
                        padded[:, :data.shape[1], :data.shape[2]] = data
                        data = padded
                    # Same normalization as training
                    if norm_params:
                        for b in range(data.shape[0]):
                            p2, p98 = norm_params[b]
                            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8), 0, 1)
                    batch.append(data)

                batch_t = torch.from_numpy(np.stack(batch)).to(device)
                with torch.no_grad():
                    pred = model(batch_t).cpu().numpy()

                for j, (r0, c0) in enumerate(batch_tiles):
                    rh = min(tile_size, H - r0)
                    rw = min(tile_size, W - c0)
                    p = pred[j, :, :rh, :rw]
                    g = gauss[:rh, :rw]

                    if mc_dropout > 0:
                        all_preds[pass_i, :, r0:r0+rh, c0:c0+rw] += p
                    else:
                        accum[:, r0:r0+rh, c0:c0+rw] += p * g[np.newaxis]
                        if pass_i == 0:
                            weight[r0:r0+rh, c0:c0+rw] += g

    if mc_dropout > 0:
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)
        return mean_pred, std_pred
    else:
        weight[weight == 0] = 1
        prediction = accum / weight[np.newaxis]
        return prediction, None


def save_geotiff(prediction, meta, out_path, uncertainty=None):
    """Save prediction as georeferenced GeoTIFF."""
    n_targets = prediction.shape[0]
    names = meta.get("target_names") or [f"target_{i}" for i in range(n_targets)]

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": meta["w"], "height": meta["h"],
        "count": n_targets, "crs": meta["crs"],
        "transform": meta["transform"],
        "compress": "lzw", "tiled": True,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(n_targets):
            dst.write(prediction[i].astype(np.float32), i + 1)
            dst.set_band_description(i + 1, names[i])

    print(f"  saved: {out_path} ({n_targets} bands)")

    if uncertainty is not None:
        unc_path = str(out_path).replace(".tif", "_uncertainty.tif")
        with rasterio.open(unc_path, "w", **profile) as dst:
            for i in range(n_targets):
                dst.write(uncertainty[i].astype(np.float32), i + 1)
                dst.set_band_description(i + 1, f"{names[i]}_std")
        print(f"  saved: {unc_path}")
