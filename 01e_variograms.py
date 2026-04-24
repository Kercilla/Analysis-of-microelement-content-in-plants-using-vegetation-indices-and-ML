#!/usr/bin/env python3
"""
01e_variograms.py — пространственная автокорреляция нутриентов.

Строит эмпирические вариограммы для каждого нутриента,
оценивает радиус автокорреляции (range) — входной параметр для buffered-LOO.
Также вычисляет Moran's I.

    python 01e_variograms.py                      # обе даты
    python 01e_variograms.py --date date1         # только дата 1
"""
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_chemistry
from analysis.cv_pipeline import load_point_coords
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


# ── Вариограммные модели (параметризация через partial_sill) ──────────────────
#
# ИСПРАВЛЕНИЕ бага: используем nugget + partial_sill вместо nugget + sill.
# total_sill = nugget + partial_sill.
# Это гарантирует что total_sill >= nugget всегда, так как partial_sill >= 0.
# В старой версии nugget и sill были независимы → nugget > sill возможен
# (физически бессмысленно: вариограмма убывала с расстоянием).

def spherical_model(h, nugget, partial_sill, range_):
    h = np.asarray(h, dtype=float)
    return np.where(
        h <= range_,
        nugget + partial_sill * (1.5*(h/range_) - 0.5*(h/range_)**3),
        nugget + partial_sill
    )


def exponential_model(h, nugget, partial_sill, range_):
    return nugget + partial_sill * (1 - np.exp(-3*h/range_))


def gaussian_model(h, nugget, partial_sill, range_):
    return nugget + partial_sill * (1 - np.exp(-3*(h/range_)**2))


VARIOGRAM_MODELS = {
    "spherical":   spherical_model,
    "exponential": exponential_model,
    "gaussian":    gaussian_model,
}


def empirical_variogram(coords, values, n_lags=12, max_dist_pct=0.5):
    """
    Эмпирическая полувариограмма.
    γ(h) = 1/(2·N(h)) · Σ(z_i - z_j)²

    ИСПРАВЛЕНИЕ: перед вычислением удаляем выбросы (IQR × 3),
    чтобы пара точек с экстремальными значениями не искажала γ(h).
    """
    mask = np.isfinite(values)
    c, v = coords[mask], values[mask]

    # Удаляем выбросы по IQR
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    iqr = q3 - q1
    if iqr > 0:
        outlier_mask = (v >= q1 - 3*iqr) & (v <= q3 + 3*iqr)
        c, v = c[outlier_mask], v[outlier_mask]

    if len(v) < 10:
        return np.array([]), np.array([]), np.array([])

    dists = cdist(c, c)
    diffs = (v[:, None] - v[None, :])**2

    max_dist = np.percentile(dists[dists > 0], max_dist_pct * 100)
    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centers, gammas, counts = [], [], []

    for i in range(n_lags):
        lo, hi  = lag_edges[i], lag_edges[i+1]
        msk     = (dists > lo) & (dists <= hi)
        n_pairs = msk.sum()
        if n_pairs < 4:
            continue
        lag_centers.append((lo + hi) / 2)
        gammas.append(diffs[msk].mean() / 2)
        counts.append(n_pairs)

    return np.array(lag_centers), np.array(gammas), np.array(counts)


def fit_variogram(lag_h, gamma, max_range_m=None):
    """
    Подбирает параметры вариограммной модели методом МНК.

    ИСПРАВЛЕНИЯ:
    1. Параметризация через partial_sill (nugget + partial_sill = total_sill)
       → nugget всегда <= total_sill, физически корректно.
    2. Верхняя граница range ограничена реальным max_dist (не *2).
    3. Проверка на pure nugget: если gamma почти не растёт → структуры нет.
    """
    if len(lag_h) < 3:
        return None

    # Проверка на pure nugget (нет пространственной структуры)
    gamma_range = gamma.max() - gamma.min()
    gamma_mean  = gamma.mean()
    if gamma_mean > 0 and gamma_range / gamma_mean < 0.15:
        # Вариограмма плоская — чистый nugget-эффект
        return {
            "model": "pure_nugget",
            "nugget": float(gamma_mean),
            "partial_sill": 0.0,
            "total_sill": float(gamma_mean),
            "range": np.nan,
            "rmse": 0.0,
            "note": "pure nugget",
        }

    variance  = gamma.max()
    range_max = max_range_m if max_range_m else lag_h.max()

    best = None
    for name, fn in VARIOGRAM_MODELS.items():
        try:
            p0     = [variance * 0.2, variance * 0.8, lag_h[len(lag_h)//2]]
            bounds = (
                [0.0,  0.0,  lag_h.min() * 0.1],
                [variance, variance * 2, range_max],
            )
            popt, _ = curve_fit(fn, lag_h, gamma, p0=p0,
                                bounds=bounds, maxfev=5000)
            nugget, partial_sill, range_ = popt
            fitted = fn(lag_h, *popt)
            rmse   = np.sqrt(((gamma - fitted)**2).mean())

            # Дополнительная проверка: range не должен быть > 95% max_dist
            if range_ > range_max * 0.95:
                rmse *= 2  # штрафуем за выход к границе

            if best is None or rmse < best["rmse"]:
                best = {
                    "model":        name,
                    "nugget":       float(nugget),
                    "partial_sill": float(partial_sill),
                    "total_sill":   float(nugget + partial_sill),
                    "range":        float(range_),
                    "rmse":         float(rmse),
                    "note":         "",
                }
        except Exception:
            continue

    return best


def morans_i(coords, values, k_neighbors=8):
    """
    Moran's I — мера пространственной автокорреляции.

    ИСПРАВЛЕНИЕ: вместо медианы всех попарных расстояний в качестве порога
    используем k-ближайших соседей (k=8).
    Старый способ (медиана) давал порог ~300 м и делал 50% всех пар
    «соседями», что полностью убивало смысл локальной автокорреляции.
    """
    mask = np.isfinite(values)
    c, v = coords[mask], values[mask]
    n    = len(v)
    if n < 10:
        return {"I": np.nan, "z_score": np.nan, "p_value": np.nan}

    dists = cdist(c, c)

    # Матрица весов: 1 если точка среди k ближайших соседей, 0 иначе
    k = min(k_neighbors, n - 1)
    W = np.zeros((n, n))
    for i in range(n):
        row = dists[i].copy()
        row[i] = np.inf
        nn_idx = np.argpartition(row, k)[:k]
        W[i, nn_idx] = 1.0
    # Симметризуем
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)

    W_sum = W.sum()
    if W_sum == 0:
        return {"I": np.nan, "z_score": np.nan, "p_value": np.nan}

    v_dev = v - v.mean()
    num   = (W * np.outer(v_dev, v_dev)).sum()
    den   = (v_dev**2).sum()
    I     = (n / W_sum) * (num / den) if den > 0 else np.nan

    # Дисперсия Moran's I (аппроксимация при нормальности)
    S0 = W_sum
    S1 = 0.5 * ((W + W.T)**2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0))**2).sum()
    E_I   = -1.0 / (n - 1)
    n2    = n * n
    Var_I = ((n * (n2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2) /
             ((n-1) * (n2 - n) * S0**2) -
             (n * (n - 1) * S1 - 2*n*S2 + 6*S0**2) /
             ((n-1) * (n-2) * (n-3) * S0**2) * 0 - E_I**2)
    # Упрощённая формула (нормализованная)
    Var_I_simple = (n**2 * S1 - n * S2 + 3 * S0**2) / (S0**2 * (n**2 - 1)) - E_I**2

    z = (I - E_I) / np.sqrt(abs(Var_I_simple)) if Var_I_simple > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    return {
        "I":       round(float(I), 4),
        "z_score": round(float(z), 3),
        "p_value": round(float(p), 4),
    }


# ── Основная логика по одной дате ─────────────────────────────────────────────

def process_date(cfg, dk, out, dpi, args):
    """Вариограммный анализ для одной даты химии."""
    elems  = args.elements or cfg["chemistry"]["target_elements"]
    dsmap  = cfg["chemistry"]["date_sampling_map"]
    sk     = dsmap.get(dk)
    if sk is None:
        print(f"[{dk}] нет в date_sampling_map, пропуск"); return None

    cp = cfg["paths"]["chemistry"].get(sk)
    if not cp or not Path(cp).exists():
        print(f"[{dk}] chemistry не найден ({cp}), пропуск"); return None

    chem  = load_chemistry(cp, cfg["chemistry"]["columns"],
                           cfg["chemistry"]["id_offsets"].get(sk, 0))
    avail = [e for e in elems if e in chem.columns]

    gpkg_pattern = cfg["paths"]["multi"].get(dk)
    if not gpkg_pattern:
        print(f"[{dk}] нет gpkg-паттерна, пропуск"); return None
    gpkg_files = sorted(glob(gpkg_pattern))
    if not gpkg_files:
        print(f"[{dk}] gpkg не найдены, пропуск"); return None

    import geopandas as gpd
    gdf = gpd.read_file(gpkg_files[0])
    try:
        gdf_m = gdf.to_crs(epsg=32636)
    except Exception:
        gdf_m = gdf

    ids_all = gdf_m["id"].values.astype(int)
    xy_m    = np.array([(g.centroid.x, g.centroid.y) for g in gdf_m.geometry])

    common   = np.intersect1d(ids_all, chem.index.values)
    msk      = np.isin(ids_all, common)
    xy_c     = xy_m[msk]
    chem_sub = chem.loc[ids_all[msk], avail]
    print(f"\n[{dk}/{sk}]  {len(common)} точек")

    # Максимальное расстояние для ограничения range
    all_dists = cdist(xy_c, xy_c)
    max_dist  = np.percentile(all_dists[all_dists > 0], 50)  # медиана = реалистичный потолок

    results = []
    n_cols  = 4
    n_rows  = int(np.ceil(len(avail) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    print(f"{'Нутриент':<8} {'Модель':<12} {'Nugget':>8} {'P.Sill':>8} "
          f"{'T.Sill':>8} {'Range_m':>9} {'NugRatio':>9} {'Moran_I':>8} {'p':>7}")
    print("-"*82)

    for i, el in enumerate(avail):
        y  = chem_sub[el].values.astype(float)
        sn = short_name(cfg, el)
        ax = axes[i] if i < len(axes) else None

        mi         = morans_i(xy_c, y)
        lag_h, gamma, counts = empirical_variogram(xy_c, y)
        fit        = fit_variogram(lag_h, gamma, max_range_m=max_dist) if len(lag_h) >= 3 else None

        nug_ratio = np.nan
        row = {"element": el, "date": dk, "model": "failed",
               "nugget": np.nan, "partial_sill": np.nan, "total_sill": np.nan,
               "range_m": np.nan, "nugget_ratio": np.nan,
               "fit_rmse": np.nan, "note": "",
               "moran_I": mi["I"], "moran_z": mi["z_score"],
               "moran_p": mi["p_value"]}

        if fit:
            nug_ratio = fit["nugget"] / fit["total_sill"] if fit["total_sill"] > 0 else np.nan
            row.update({
                "model":        fit["model"],
                "nugget":       round(fit["nugget"], 6),
                "partial_sill": round(fit["partial_sill"], 6),
                "total_sill":   round(fit["total_sill"], 6),
                "range_m":      round(fit["range"], 1) if not np.isnan(fit.get("range", np.nan)) else np.nan,
                "nugget_ratio": round(nug_ratio, 3) if np.isfinite(nug_ratio) else np.nan,
                "fit_rmse":     round(fit["rmse"], 6),
                "note":         fit.get("note", ""),
            })

        results.append(row)

        range_str = f"{row['range_m']:>9.1f}" if pd.notna(row['range_m']) else f"{'—':>9}"
        nug_str   = f"{row['nugget_ratio']:>9.3f}" if pd.notna(row.get('nugget_ratio')) else f"{'—':>9}"
        print(f"{sn:<8} {row['model']:<12} "
              f"{row['nugget'] if pd.notna(row['nugget']) else 0:>8.4f} "
              f"{row['partial_sill'] if pd.notna(row['partial_sill']) else 0:>8.4f} "
              f"{row['total_sill'] if pd.notna(row['total_sill']) else 0:>8.4f} "
              f"{range_str} {nug_str} "
              f"{mi['I'] if pd.notna(mi['I']) else 0:>8.4f} "
              f"{mi['p_value'] if pd.notna(mi['p_value']) else 1:>7.4f}")

        if ax is not None and len(lag_h) >= 3:
            sc = counts / counts.max() * 100 + 10
            ax.scatter(lag_h, gamma, s=sc, c="#2E75B6", alpha=0.8, zorder=3, label="empirical")
            if fit and fit["model"] != "pure_nugget" and pd.notna(row["range_m"]):
                h_fine = np.linspace(0, lag_h.max(), 100)
                fn = VARIOGRAM_MODELS[fit["model"]]
                ax.plot(h_fine,
                        fn(h_fine, fit["nugget"], fit["partial_sill"], fit["range"]),
                        "r-", lw=2,
                        label=f"{fit['model']}\nrange={fit['range']:.0f}m\nnugRatio={nug_ratio:.2f}")
                ax.axvline(fit["range"], color="red", lw=0.8, ls="--", alpha=0.5)
            elif fit and fit["model"] == "pure_nugget":
                ax.axhline(fit["nugget"], color="red", lw=2, ls="--", label="pure nugget")
            mi_str = f"{mi['I']:.3f}" if pd.notna(mi["I"]) else "n/a"
            ax.set_title(f"{sn} [{dk}]  I={mi_str}", fontsize=8)
            ax.set_xlabel("расстояние (м)", fontsize=7)
            ax.set_ylabel("γ(h)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
            ax.grid(alpha=0.2)

    for j in range(len(avail), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Вариограммы нутриентов [{dk}/{sk}]", fontsize=11, y=1.01)
    plt.tight_layout()
    fname = f"variograms_{dk}.png"
    fig.savefig(Path(out) / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved: {fname}")

    return pd.DataFrame(results)


# ── run ───────────────────────────────────────────────────────────────────────

def run(args):
    cfg  = load_config(args.config)
    out  = args.output or "results/01e_variograms"
    os.makedirs(out, exist_ok=True)
    dpi  = cfg["plots"]["dpi"]

    dates = args.date if args.date else list(cfg["chemistry"]["date_sampling_map"].keys())
    print(f"Даты для анализа: {dates}")

    all_results = []
    for dk in dates:
        df = process_date(cfg, dk, out, dpi, args)
        if df is not None:
            all_results.append(df)

    if not all_results:
        print("Нет результатов"); return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{out}/spatial_range_all.csv",
                    index=False, float_format="%.4f")

    # Сводка рекомендуемых буферов
    print(f"\n{'='*65}")
    print("РЕКОМЕНДУЕМЫЕ БУФЕРЫ для buffered-LOO (по нутриентам):")
    print(f"{'Нутриент':<10} {'date1_range':>12} {'date2_range':>12} {'Буфер':>8} {'Предупреждение'}")
    print("-"*65)

    buffer_map = {}
    for el in combined["element"].unique():
        sub = combined[combined["element"] == el]
        r_vals = sub.set_index("date")["range_m"]

        r1 = r_vals.get("date1", np.nan)
        r2 = r_vals.get("date2", np.nan)
        valid = [v for v in [r1, r2] if pd.notna(v) and v > 0]

        note = ""
        if not valid:
            buf = 50  # дефолт
            note = "нет данных → дефолт 50м"
        else:
            buf_raw = np.median(valid)
            # Ограничиваем: не меньше 20м, не больше 100м
            buf = int(np.clip(buf_raw, 20, 100))
            if buf_raw > 200:
                note = f"range={buf_raw:.0f}м подозрителен → ограничен до {buf}м"
            # Если nugget_ratio > 0.9 для всех дат → структуры нет
            nug_ratios = sub["nugget_ratio"].dropna().values
            if len(nug_ratios) > 0 and (nug_ratios > 0.9).all():
                note = "pure nugget (нет пространственной структуры)"

        buffer_map[el] = buf
        sn = short_name(cfg, el) if hasattr(cfg, '__getitem__') else el
        r1s = f"{r1:.0f}м" if pd.notna(r1) else "—"
        r2s = f"{r2:.0f}м" if pd.notna(r2) else "—"
        print(f"{el:<10} {r1s:>12} {r2s:>12} {buf:>7}м  {note}")

    # Единый глобальный буфер
    global_buf = int(np.median(list(buffer_map.values())))
    print(f"\nГлобальный медианный буфер (для простоты): {global_buf} м")
    print(f"  python 02_ml_honest.py --buffer {global_buf}")

    # Сохраняем сводку
    buf_df = pd.DataFrame([
        {"element": k, "recommended_buffer_m": v}
        for k, v in buffer_map.items()
    ])
    buf_df["global_buffer_m"] = global_buf
    buf_df.to_csv(f"{out}/recommended_buffers.csv", index=False)

    # Сравнительный график по датам
    if len(all_results) > 1:
        elems_plot = combined["element"].unique()
        x = np.arange(len(elems_plot))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_i, col, title in [(0, "range_m", "Радиус автокорреляции (м)"),
                                  (1, "moran_I", "Moran's I")]:
            ax = axes[ax_i]
            for j, (dk, color) in enumerate(zip(dates, ["#2E75B6", "#C00000"])):
                sub = combined[combined["date"] == dk].set_index("element")
                vals = [sub.loc[e, col] if e in sub.index and pd.notna(sub.loc[e, col]) else 0
                        for e in elems_plot]
                ax.bar(x + j*width, vals, width, label=dk, color=color, alpha=0.8)
            ax.set_xticks(x + width/2)
            ax.set_xticklabels([short_name(cfg, e) if hasattr(cfg, '__getitem__') else e
                                 for e in elems_plot], rotation=45, ha="right", fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            if ax_i == 0:
                ax.axhline(global_buf, color="gray", ls="--", lw=1, label=f"буфер={global_buf}м")

        plt.tight_layout()
        fig.savefig(f"{out}/spatial_range_comparison.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: spatial_range_comparison.png")

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--date",     nargs="+",
                    help="ключи дат (date1 date2). По умолчанию — все")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output",   default=None)
    run(ap.parse_args())
