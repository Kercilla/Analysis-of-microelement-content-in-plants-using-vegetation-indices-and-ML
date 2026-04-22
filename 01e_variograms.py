#!/usr/bin/env python3
"""
01e_variograms.py — пространственная автокорреляция нутриентов.

Строит эмпирические вариограммы для каждого нутриента,
оценивает радиус автокорреляции (range) — входной параметр для buffered-LOO.
Также вычисляет Moran's I.

Обрабатывает обе даты химического анализа.

    python 01e_variograms.py
    python 01e_variograms.py --date date1 date2
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
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


# ── Вариограммные модели ──────────────────────────────────────────────────────

def spherical_model(h, nugget, sill, range_):
    h = np.asarray(h, dtype=float)
    return np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5*(h/range_) - 0.5*(h/range_)**3),
        sill
    )

def exponential_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-3*h/range_))

def gaussian_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-3*(h/range_)**2))

VARIOGRAM_MODELS = {
    "spherical":   spherical_model,
    "exponential": exponential_model,
    "gaussian":    gaussian_model,
}


def empirical_variogram(coords, values, n_lags=12, max_dist_pct=0.5):
    """γ(h) = 1/(2·N(h)) · Σ(z_i - z_j)²"""
    mask = np.isfinite(values)
    c, v = coords[mask], values[mask]
    dists = cdist(c, c)
    diffs = (v[:, None] - v[None, :])**2
    max_dist  = np.percentile(dists[dists > 0], max_dist_pct * 100)
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


def fit_variogram(lag_h, gamma):
    best = None
    for name, fn in VARIOGRAM_MODELS.items():
        try:
            sill_0  = gamma.max()
            range_0 = lag_h[len(lag_h)//2]
            p0      = [0.0, sill_0, range_0]
            bounds  = ([0, 0, lag_h.min()*0.1],
                       [sill_0, sill_0*2, lag_h.max()*2])
            popt, _ = curve_fit(fn, lag_h, gamma, p0=p0,
                                bounds=bounds, maxfev=5000)
            rmse = np.sqrt(((gamma - fn(lag_h, *popt))**2).mean())
            if best is None or rmse < best["rmse"]:
                best = {"model": name, "nugget": popt[0],
                        "sill": popt[1], "range": popt[2], "rmse": rmse}
        except Exception:
            continue
    return best


def morans_i(coords, values):
    mask = np.isfinite(values)
    c, v = coords[mask], values[mask]
    n    = len(v)
    if n < 10:
        return {"I": np.nan, "z_score": np.nan, "p_value": np.nan}
    dists  = cdist(c, c)
    thresh = np.median(dists[dists > 0])
    W      = (dists < thresh).astype(float)
    np.fill_diagonal(W, 0)
    W_sum  = W.sum()
    if W_sum == 0:
        return {"I": np.nan, "z_score": np.nan, "p_value": np.nan}
    v_dev = v - v.mean()
    num   = (W * np.outer(v_dev, v_dev)).sum()
    den   = (v_dev**2).sum()
    I     = (n / W_sum) * (num / den) if den > 0 else np.nan
    E_I   = -1 / (n - 1)
    Var_I = (n**2*W_sum + 3*W_sum**2 - n*(W**2).sum()) / \
            ((W_sum**2)*(n**2 - 1)) - E_I**2
    z = (I - E_I) / np.sqrt(abs(Var_I)) if Var_I > 0 else np.nan
    p = 2*(1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return {"I": round(float(I), 4), "z_score": round(float(z), 3),
            "p_value": round(float(p), 4)}


# ── Обработка одной даты ──────────────────────────────────────────────────────

def process_date(dk, cfg, elems, out, dpi):
    import geopandas as gpd

    dsmap = cfg["chemistry"]["date_sampling_map"]
    sk    = dsmap.get(dk, "sampling1")
    cp    = cfg["paths"]["chemistry"].get(sk)
    if not cp:
        print(f"  [{dk}] нет пути к химии в config"); return None

    chem  = load_chemistry(cp,
                           cfg["chemistry"]["columns"],
                           cfg["chemistry"]["id_offsets"].get(sk, 0))
    avail = [e for e in elems if e in chem.columns]

    gpkg_files = sorted(glob(cfg["paths"]["multi"].get(dk, "")))
    if not gpkg_files:
        print(f"  [{dk}] gpkg не найдены"); return None

    gdf = gpd.read_file(gpkg_files[0])
    try:
        gdf_m = gdf.to_crs(epsg=32636)   # UTM Zone 36N
    except Exception:
        gdf_m = gdf

    ids_all   = gdf_m["id"].values.astype(int)
    xy_m      = np.array([(g.centroid.x, g.centroid.y) for g in gdf_m.geometry])
    common    = np.intersect1d(ids_all, chem.index.values)
    msk       = np.isin(ids_all, common)
    xy_common = xy_m[msk]
    chem_sub  = chem.loc[ids_all[msk], avail]
    print(f"  n={len(common)}  химия: {sk}")

    results = []
    n_cols  = 4
    n_rows  = int(np.ceil(len(avail) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    print(f"\n  {'Нутриент':<8} {'Модель':<12} {'Nugget':>8} {'Sill':>8} "
          f"{'Range_m':>9} {'Moran_I':>9} {'p':>7}")
    print("  " + "-"*63)

    for i, el in enumerate(avail):
        y  = chem_sub[el].values.astype(float)
        sn = short_name(cfg, el)
        ax = axes[i] if i < len(axes) else None

        mi                   = morans_i(xy_common, y)
        lag_h, gamma, counts = empirical_variogram(xy_common, y)

        if len(lag_h) < 3:
            results.append({"element": el, "date": dk,
                             "range_m": np.nan, "moran_I": mi["I"]})
            continue

        fit = fit_variogram(lag_h, gamma)
        row = {
            "element":  el,
            "date":     dk,
            "model":    fit["model"]         if fit else "failed",
            "nugget":   round(fit["nugget"], 6) if fit else np.nan,
            "sill":     round(fit["sill"],   6) if fit else np.nan,
            "range_m":  round(fit["range"],  1) if fit else np.nan,
            "fit_rmse": round(fit["rmse"],   6) if fit else np.nan,
            "moran_I":  mi["I"],
            "moran_z":  mi["z_score"],
            "moran_p":  mi["p_value"],
        }
        results.append(row)

        print(f"  {sn:<8} {row['model']:<12} {row['nugget']:>8.4f} "
              f"{row['sill']:>8.4f} {row['range_m']:>9.1f} "
              f"{row['moran_I']:>9.4f} {row['moran_p']:>7.4f}")

        if ax is not None:
            ax.scatter(lag_h, gamma, s=counts/counts.max()*100+10,
                       c="#2E75B6", alpha=0.8, zorder=3, label="empirical")
            if fit:
                h_fine = np.linspace(0, lag_h.max(), 100)
                fn_m   = VARIOGRAM_MODELS[fit["model"]]
                ax.plot(h_fine, fn_m(h_fine, fit["nugget"],
                                     fit["sill"], fit["range"]),
                        "r-", lw=2,
                        label=f"{fit['model']}\nrange={fit['range']:.0f}m")
                ax.axvline(fit["range"], color="red", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{sn}  I={mi['I']:.3f}", fontsize=9)
            ax.set_xlabel("расстояние (м)", fontsize=8)
            ax.set_ylabel("γ(h)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Вариограммы нутриентов — {dk}  (химия: {sk})\n"
                 "Размер точки ∝ числу пар в лаге", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(Path(out) / f"variograms_{dk}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(f"{out}/spatial_range_{dk}.csv", index=False, float_format="%.4f")

    valid = df["range_m"].dropna()
    if len(valid) > 0:
        print(f"\n  Рекомендуемый буфер [{dk}]: {valid.median():.0f} м")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    cfg   = load_config(args.config)
    out   = args.output or "results/01e_variograms"
    os.makedirs(out, exist_ok=True)

    elems  = args.elements or cfg["chemistry"]["target_elements"]
    dates  = args.date or list(cfg["paths"]["multi"].keys())
    dpi    = cfg["plots"]["dpi"]

    all_dfs = []
    for dk in dates:
        print(f"\n{'='*65}")
        print(f"ДАТА: {dk}")
        print(f"{'='*65}")
        df = process_date(dk, cfg, elems, out, dpi)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("Нет результатов"); return

    # Сводный файл
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(f"{out}/spatial_range_all.csv", index=False, float_format="%.4f")

    valid_all = df_all["range_m"].dropna()
    if len(valid_all) > 0:
        buf = round(valid_all.median(), 0)
        print(f"\n{'='*50}")
        print(f"ИТОГОВЫЙ РЕКОМЕНДУЕМЫЙ БУФЕР (по всем датам):")
        print(f"  median range = {buf:.0f} м")
        print(f"  Используй: --buffer-radius {buf:.0f}")
        print(f"  (запиши в config.yaml → cv.buffer_radius)")

    # Сводный график сравнения дат
    fig, ax = plt.subplots(figsize=(11, 5))
    for dk, grp in df_all.groupby("date"):
        grp_sorted = grp.sort_values("element")
        ax.plot(grp_sorted["element"], grp_sorted["range_m"],
                "o-", label=dk, linewidth=1.5, markersize=6)
    ax.axhline(valid_all.median(), color="red", lw=1.2, ls="--",
               label=f"median = {valid_all.median():.0f} м")
    ax.set_xlabel("Нутриент", fontsize=10)
    ax.set_ylabel("Range (м)", fontsize=10)
    ax.set_title("Радиус пространственной автокорреляции по нутриентам и датам",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/spatial_range_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: spatial_range_comparison.png")

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--date",     nargs="+", default=None,
                    help="даты для обработки (по умолч. все из config)")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output",   default=None)
    run(ap.parse_args())
