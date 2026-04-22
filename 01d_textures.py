#!/usr/bin/env python3
"""
01d_textures.py — текстурные признаки из ортомозаики.

GLCM (Gray-Level Co-occurrence Matrix): contrast, dissimilarity,
homogeneity, energy, correlation, ASM.
Gabor filters: 4 ориентации × 3 частоты = 12 признаков.
LBP (Local Binary Patterns): гистограмма 8 бинов.

    python 01d_textures.py --ortho data/map_tif/20230619_F14_Micasense.tif
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
import rasterio.windows as riow
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.cv_pipeline import load_point_coords, world_to_pixel, compute_band_stats

BAND_ORDER = ("Blue", "Green", "Red", "RedEdge", "NIR")
PATCH_SIZE = 32   # 32px = 1.6м при GSD 5см — достаточно для текстуры


def _read_patch(src, r, c, size, band_stats):
    """Читает нормализованный патч size×size вокруг (r,c)."""
    H, W = src.height, src.width
    half = size // 2
    r0 = max(0, min(r - half, H - size))
    c0 = max(0, min(c - half, W - size))
    win = riow.Window(col_off=c0, row_off=r0, width=size, height=size)
    raw = src.read(window=win).astype(np.float32)
    out = {}
    for i, name in enumerate(BAND_ORDER):
        lo, hi  = band_stats[i]
        band    = np.clip((raw[i] - lo) / (hi - lo + 1e-8), 0, 1)
        band[raw[i] == 0] = 0.0
        out[name] = band
    return out


def glcm_features(band, levels=64, distances=(1, 2), angles=None):
    """
    GLCM-признаки для одного канала.
    Квантизует в levels уровней серого, вычисляет матрицу совместной встречаемости.
    """
    from skimage.feature import graycomatrix, graycoprops
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    img = (band * (levels - 1)).astype(np.uint8)
    feats = {}
    try:
        glcm = graycomatrix(img, distances=list(distances),
                             angles=angles, levels=levels,
                             symmetric=True, normed=True)
        for prop in ("contrast", "dissimilarity", "homogeneity",
                     "energy", "correlation", "ASM"):
            vals = graycoprops(glcm, prop).flatten()
            feats[f"glcm_{prop}_mean"] = float(vals.mean())
            feats[f"glcm_{prop}_std"]  = float(vals.std())
    except Exception:
        pass
    return feats


def gabor_features(band, frequencies=(0.1, 0.25, 0.4),
                    thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    Gabor filter bank: 3 частоты × 4 ориентации.
    Для каждого фильтра: mean и std энергии ответа.
    """
    from skimage.filters import gabor
    feats = {}
    for freq in frequencies:
        for theta in thetas:
            try:
                real, imag = gabor(band, frequency=freq, theta=theta,
                                   sigma_x=2, sigma_y=2)
                energy = np.sqrt(real**2 + imag**2)
                key    = f"gabor_f{freq:.2f}_t{theta:.2f}"
                feats[f"{key}_mean"] = float(energy.mean())
                feats[f"{key}_std"]  = float(energy.std())
            except Exception:
                pass
    return feats


def lbp_features(band, radius=2, n_points=16, n_bins=8):
    """
    Local Binary Patterns гистограмма.
    Возвращает нормализованные бины как признаки.
    """
    from skimage.feature import local_binary_pattern
    try:
        lbp  = local_binary_pattern(band, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(),
                                bins=n_bins,
                                range=(0, n_points + 2),
                                density=True)
        return {f"lbp_bin{i}": float(v) for i, v in enumerate(hist)}
    except Exception:
        return {}


def compute_texture_features(bands_patch, selected_bands=("Red", "NIR", "RedEdge")):
    """Вычисляет все текстурные признаки для патча."""
    feats = {}
    for bname in selected_bands:
        band = bands_patch.get(bname)
        if band is None: continue

        # GLCM
        glcm = glcm_features(band)
        feats.update({f"{bname}_{k}": v for k, v in glcm.items()})

        # Gabor (только для ключевых каналов — вычислительно дорого)
        if bname in ("NIR", "RedEdge"):
            gab = gabor_features(band)
            feats.update({f"{bname}_{k}": v for k, v in gab.items()})

        # LBP
        lbp = lbp_features(band)
        feats.update({f"{bname}_{k}": v for k, v in lbp.items()})

    return feats


def _analyze_for_date(args, cfg, dk, map_stem, out_root, band_stats, meta, dpi):
    """Текстурный анализ для одной даты хим.анализа."""
    elems  = args.elements or cfg["chemistry"]["target_elements"]
    dsmap  = cfg["chemistry"]["date_sampling_map"]
    sk     = dsmap.get(dk)
    if sk is None:
        print(f"[{dk}] нет в date_sampling_map, пропуск")
        return False

    cp = cfg["paths"]["chemistry"].get(sk)
    if not cp or not Path(cp).exists():
        print(f"[{dk}] chemistry не найден ({cp}), пропуск")
        return False

    chem   = load_chemistry(cp, cfg["chemistry"]["columns"],
                            cfg["chemistry"]["id_offsets"].get(sk, 0))
    avail  = [e for e in elems if e in chem.columns]

    gpkg_pattern = cfg["paths"]["multi"].get(dk)
    if not gpkg_pattern:
        print(f"[{dk}] нет gpkg-паттерна, пропуск")
        return False
    gpkg_files = sorted(glob(gpkg_pattern))
    if not gpkg_files:
        print(f"[{dk}] gpkg не найдены, пропуск")
        return False

    ids, coords  = load_point_coords(gpkg_files[0], target_crs=meta["crs"])
    rows_px, cpx = world_to_pixel(coords, meta["transform"])
    H, W         = meta["height"], meta["width"]
    ok           = (rows_px >= 0) & (rows_px < H) & (cpx >= 0) & (cpx < W)
    ids, rows_px, cpx = ids[ok], rows_px[ok], cpx[ok]

    common  = np.intersect1d(ids, chem.index.values)
    msk     = np.isin(ids, common)
    rows_px, cpx = rows_px[msk], cpx[msk]
    chem_sub     = chem.loc[ids[msk], avail]

    print(f"\n{'='*70}")
    print(f"[{dk} / {sk}]  Points: {len(common)}  patch: {PATCH_SIZE}px  ortho: {map_stem}")
    print(f"{'='*70}")

    prefix = f"{map_stem}__{dk}"
    out    = out_root

    print(f"Извлечение текстурных признаков...")
    all_feats = []
    with rasterio.open(args.ortho) as src:
        for i, (r, c) in enumerate(zip(rows_px, cpx)):
            bands = _read_patch(src, int(r), int(c), PATCH_SIZE, band_stats)
            feats = compute_texture_features(bands)
            all_feats.append(feats)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(rows_px)}", end="\r", flush=True)
    print(f"  {len(rows_px)}/{len(rows_px)} done")

    feat_df = pd.DataFrame(all_feats, index=common)
    feat_df = feat_df.loc[:, feat_df.std() > 1e-8]
    feat_df.to_csv(f"{out}/{prefix}_texture_features.csv", float_format="%.6f")
    print(f"Текстурных признаков: {feat_df.shape[1]}")

    print(f"\nКорреляционный анализ...")
    rows = []
    for feat_col in feat_df.columns:
        x = feat_df[feat_col].values.astype(float)
        for el in avail:
            y = chem_sub[el].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10: continue
            r, p = stats.pearsonr(x[mask], y[mask])
            rows.append({"feature": feat_col, "element": el,
                         "pearson_r": round(r, 4), "p_value": round(p, 4),
                         "abs_r": round(abs(r), 4)})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(f"{out}/{prefix}_texture_correlations.csv",
                   index=False, float_format="%.4f")

    print(f"\nТОП-3 ТЕКСТУРНЫХ ПРИЗНАКА ПО НУТРИЕНТАМ")
    print("-"*60)
    for el in avail:
        sub = corr_df[corr_df["element"] == el].nlargest(3, "abs_r")
        if sub.empty: continue
        sn = short_name(cfg, el)
        top_str = "  |  ".join([f"{r['feature']}: {r['pearson_r']:+.3f}"
                                  for _, r in sub.iterrows()])
        print(f"{sn:<6}: {top_str}")

    corr_best = corr_df.groupby("element")["abs_r"].max()
    print(f"\nМаксимальная |r| от текстур по нутриентам:")
    for el in avail:
        sn = short_name(cfg, el)
        tx = corr_best.get(el, np.nan)
        print(f"  {sn:<6}: texture_max_|r|={tx:.3f}")

    if not corr_df.empty:
        top_global = corr_df.nlargest(20, "abs_r")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_global)),
                top_global["abs_r"].values,
                color=["#C00000" if r > 0 else "#2E75B6"
                       for r in top_global["pearson_r"].values])
        ax.set_yticks(range(len(top_global)))
        ax.set_yticklabels(
            [f"{row['feature']} → {short_name(cfg, row['element'])}"
             for _, row in top_global.iterrows()],
            fontsize=7)
        ax.set_xlabel("|Pearson r|", fontsize=10)
        ax.set_title(f"Топ-20 текстурных признаков [{prefix}]", fontsize=11)
        ax.axvline(0.2, color="gray", lw=0.8, ls="--")
        plt.tight_layout()
        fname = f"{prefix}_top_texture_features.png"
        fig.savefig(Path(out) / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  saved: {fname}")
    return True


def run(args):
    cfg      = load_config(args.config)
    map_stem = Path(args.ortho).stem
    out      = args.output or f"results/01d_textures/{map_stem}"
    os.makedirs(out, exist_ok=True)
    dpi      = cfg["plots"]["dpi"]

    print("Computing band stats...")
    band_stats = compute_band_stats(args.ortho)

    with rasterio.open(args.ortho) as src:
        meta = src.meta.copy()

    if args.date:
        dates = args.date if isinstance(args.date, list) else [args.date]
    else:
        dates = list(cfg["chemistry"]["date_sampling_map"].keys())

    print(f"Даты химии для анализа: {dates}\n")

    n_done = 0
    for dk in dates:
        if _analyze_for_date(args, cfg, dk, map_stem, out, band_stats, meta, dpi):
            n_done += 1

    print(f"\n{'='*50}")
    print(f"Обработано дат: {n_done}/{len(dates)}")
    print(f"done → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--ortho",    required=True)
    ap.add_argument("--date",     nargs="+",
                    help="ключи дат (date1 date2). По умолчанию — все из config")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output",   default=None)
    run(ap.parse_args())
