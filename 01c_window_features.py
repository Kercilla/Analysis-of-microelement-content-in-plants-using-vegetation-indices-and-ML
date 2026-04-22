#!/usr/bin/env python3
"""
01c_window_features.py — извлечение window-based признаков из ортомозаики.

Для каждой из 100 точек извлекает окна 1×1, 3×3, 5×5, 7×7 пикселей
вокруг координаты и вычисляет статистики (mean, std, median, cv, p25, p75).

Затем проводит корреляционный анализ: какой размер окна и какая статистика
даёт максимальную корреляцию с нутриентами.

    python 01c_window_features.py --ortho data/map_tif/20230619_F14_Micasense.tif
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.indices import calculate_indices
from scipy import stats

# Импортируем coord-loading из cv_pipeline
sys.path.insert(0, str(Path(__file__).parent))
from analysis.cv_pipeline import load_point_coords, world_to_pixel, compute_band_stats


BAND_ORDER = ("Blue", "Green", "Red", "RedEdge", "NIR")
WINDOWS    = [1, 3, 5, 7]   # размеры окон в пикселях


def extract_window_features(src, r, c, win_size, band_stats):
    """
    Извлекает win_size×win_size окно вокруг пикселя (r,c).
    Возвращает dict: {band_stat: value}.
    """
    H, W = src.height, src.width
    half = win_size // 2
    r0 = max(0, r - half); r1 = min(H, r + half + 1)
    c0 = max(0, c - half); c1 = min(W, c + half + 1)
    h, w = r1 - r0, c1 - c0

    win  = riow.Window(col_off=c0, row_off=r0, width=w, height=h)
    raw  = src.read(window=win).astype(np.float32)  # (5, h, w)

    feats = {}
    for i, bname in enumerate(BAND_ORDER):
        lo, hi = band_stats[i]
        band = np.clip((raw[i] - lo) / (hi - lo + 1e-8), 0, 1)
        vals = band[raw[i] > 0].flatten()
        if len(vals) == 0:
            vals = np.array([0.0])

        prefix = f"{bname}_w{win_size}"
        feats[f"{prefix}_mean"]   = float(np.mean(vals))
        feats[f"{prefix}_std"]    = float(np.std(vals))
        feats[f"{prefix}_median"] = float(np.median(vals))
        feats[f"{prefix}_p25"]    = float(np.percentile(vals, 25))
        feats[f"{prefix}_p75"]    = float(np.percentile(vals, 75))
        feats[f"{prefix}_cv"]     = float(np.std(vals) / (np.mean(vals) + 1e-8))
        feats[f"{prefix}_range"]  = float(vals.max() - vals.min())

    return feats


def correlation_by_window(feat_df, chem_sub, avail, cfg):
    """
    Для каждого признака вычисляет корреляцию Пирсона с каждым нутриентом.
    Возвращает сводную таблицу.
    """
    rows = []
    for feat_col in feat_df.columns:
        x = feat_df[feat_col].values.astype(float)
        for el in avail:
            y = chem_sub[el].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            rows.append({
                "feature":  feat_col,
                "element":  el,
                "pearson_r": round(r, 4),
                "p_value":   round(p, 4),
                "abs_r":     round(abs(r), 4),
            })
    return pd.DataFrame(rows)


def plot_window_comparison(corr_df, avail, cfg, out_dir, dpi=150, prefix=""):
    """
    Для каждого нутриента показывает max |r| по размеру окна.
    """
    # Извлекаем размер окна из имени признака
    corr_df["window"] = corr_df["feature"].str.extract(r"_w(\d+)_").astype(float)
    corr_df["band"]   = corr_df["feature"].str.split("_w").str[0]
    corr_df["stat"]   = corr_df["feature"].str.split("_").str[-1]

    n = len(avail)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()

    for i, el in enumerate(avail):
        ax  = axes[i]
        sub = corr_df[corr_df["element"] == el]
        if sub.empty: ax.set_visible(False); continue

        by_win = sub.groupby("window")["abs_r"].max().reset_index()
        ax.bar(by_win["window"].astype(str), by_win["abs_r"],
               color="#2E75B6", alpha=0.8, edgecolor="white")
        ax.axhline(0.2, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(short_name(cfg, el), fontsize=9)
        ax.set_xlabel("размер окна (пкс)", fontsize=8)
        ax.set_ylabel("max |r|", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 0.6)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Влияние размера окна на корреляцию с нутриентами\n"
                 "(красная линия = |r|=0.2)", fontsize=11)
    plt.tight_layout()
    fname = f"{prefix}_window_size_comparison.png" if prefix else "window_size_comparison.png"
    fig.savefig(Path(out_dir) / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


def plot_best_features_heatmap(corr_df, avail, cfg, out_dir, dpi=150, prefix=""):
    """Тепловая карта: топ-20 window-признаков по нутриентам."""
    import seaborn as sns

    # Топ-20 признаков по среднему |r|
    top_feats = (corr_df.groupby("feature")["abs_r"]
                 .mean().nlargest(20).index.tolist())
    sub = corr_df[corr_df["feature"].isin(top_feats)]
    pivot = sub.pivot(index="feature", columns="element", values="pearson_r")
    pivot.columns = [short_name(cfg, e) for e in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                vmin=-0.5, vmax=0.5, linewidths=0.3,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Топ-20 window-признаков (Pearson r)", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fname = f"{prefix}_window_features_heatmap.png" if prefix else "window_features_heatmap.png"
    fig.savefig(Path(out_dir) / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


def _analyze_for_date(args, cfg, dk, map_stem, out_root, band_stats, meta, dpi):
    """Полный анализ для одной даты хим.анализа. Возвращает True если данные нашлись."""
    elems  = args.elements or cfg["chemistry"]["target_elements"]
    dsmap  = cfg["chemistry"]["date_sampling_map"]
    sk     = dsmap.get(dk)
    if sk is None:
        print(f"[{dk}] нет маппинга в date_sampling_map, пропуск")
        return False

    cp     = cfg["paths"]["chemistry"].get(sk)
    if not cp or not Path(cp).exists():
        print(f"[{dk}] chemistry file не найден ({cp}), пропуск")
        return False

    off    = cfg["chemistry"]["id_offsets"].get(sk, 0)
    chem   = load_chemistry(cp, cfg["chemistry"]["columns"], off)
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
    ids, coords, rows_px, cpx = ids[ok], coords[ok], rows_px[ok], cpx[ok]

    common  = np.intersect1d(ids, chem.index.values)
    msk     = np.isin(ids, common)
    rows_px, cpx = rows_px[msk], cpx[msk]
    chem_sub     = chem.loc[ids[msk], avail]

    print(f"\n{'='*70}")
    print(f"[{dk} / {sk}]  Labeled points: {len(common)}  ortho: {map_stem}")
    print(f"{'='*70}")

    # Префикс для всех файлов: <ortho>__<date_key>
    prefix = f"{map_stem}__{dk}"
    out    = out_root

    print(f"Извлечение window-признаков {WINDOWS} пикселей...")
    all_feats = []
    with rasterio.open(args.ortho) as src:
        for i, (r, c) in enumerate(zip(rows_px, cpx)):
            row_feats = {}
            for ws in WINDOWS:
                wf = extract_window_features(src, int(r), int(c), ws, band_stats)
                row_feats.update(wf)
            all_feats.append(row_feats)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(rows_px)}", end="\r", flush=True)
    print(f"  {len(rows_px)}/{len(rows_px)} done")

    feat_df = pd.DataFrame(all_feats, index=common)
    feat_df.to_csv(f"{out}/{prefix}_window_features.csv", float_format="%.6f")
    print(f"Признаков: {feat_df.shape[1]}  →  {prefix}_window_features.csv")

    print(f"\nКорреляционный анализ {feat_df.shape[1]} признаков × {len(avail)} нутриентов...")
    corr_df = correlation_by_window(feat_df, chem_sub, avail, cfg)
    corr_df.to_csv(f"{out}/{prefix}_window_correlations.csv",
                   index=False, float_format="%.4f")

    corr_df["window"] = corr_df["feature"].str.extract(r"_w(\d+)_").astype(float)
    print(f"\n{'Нутриент':<8}  {'Лучший признак':<25} {'|r|':>5}  {'Окно':>6}")
    print("-"*50)
    for el in avail:
        sub = corr_df[corr_df["element"] == el]
        if sub.empty: continue
        b  = sub.nlargest(1, "abs_r").iloc[0]
        sn = short_name(cfg, el)
        print(f"{sn:<8}  {b['feature']:<25} {b['abs_r']:>5.3f}  "
              f"{int(b['window']) if pd.notna(b['window']) else '?':>6}px")

    w1    = corr_df[corr_df["window"] == 1].groupby("element")["abs_r"].max()
    wbest = corr_df.groupby("element")["abs_r"].max()
    print(f"\nПрирост от контекста (max window vs 1px):")
    for el in avail:
        if el not in w1 or el not in wbest: continue
        delta = wbest[el] - w1[el]
        sn    = short_name(cfg, el)
        mark  = "↑" if delta > 0.02 else "≈"
        print(f"  {sn:<6}: 1px={w1[el]:.3f}  best={wbest[el]:.3f}  "
              f"Δ={delta:+.3f} {mark}")

    # Plots — сохраняются с префиксом
    plot_window_comparison(corr_df, avail, cfg, out, dpi, prefix=prefix)
    plot_best_features_heatmap(corr_df, avail, cfg, out, dpi, prefix=prefix)
    return True


def run(args):
    cfg      = load_config(args.config)
    map_stem = Path(args.ortho).stem
    out      = args.output or f"results/01c_window/{map_stem}"
    os.makedirs(out, exist_ok=True)
    dpi      = cfg["plots"]["dpi"]

    print(f"Computing band stats (strip sampling)...")
    band_stats = compute_band_stats(args.ortho)
    print(f"  {[(round(lo), round(hi)) for lo,hi in band_stats]}")

    with rasterio.open(args.ortho) as src:
        meta = src.meta.copy()

    # Определяем какие даты прогонять
    if args.date:
        # Пользователь явно указал даты
        dates = args.date if isinstance(args.date, list) else [args.date]
    else:
        # По умолчанию — все даты из date_sampling_map
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
