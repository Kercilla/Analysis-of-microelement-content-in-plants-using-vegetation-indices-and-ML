#!/usr/bin/env python3
"""
05_pixel_mapping.py — попиксельное картирование 12 нутриентов.

Загружает модели из 02_ml_honest.py (--multi-output),
применяет к ортомозаике тайлами 512×512, сохраняет 12-канальный GeoTIFF.

Поддерживает два режима моделей:
  multi_output  — один pkl → все 12 нутриентов за один predict()
  per_nutrient  — 12 pkl  → обратная совместимость

Вывод:
  nutrient_map.tif          — 12-канальный GeoTIFF, float32, georeferenced
  uncertainty_map.tif       — std предсказаний деревьев (ET/RF)
  nutrient_maps_preview.png — превью всех карт
  rgb_composite.png         — RGB: R=N, G=P, B=K
  stats.csv                 — min/mean/max/std для каждого нутриента

Использование:
  python 05_pixel_mapping.py \\
      --models results/02_honest_kaggle/models \\
      --ortho  data/map_tif/20230608_F14_Pollux.tif \\
      --output results/05_maps
"""
import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import rasterio.windows as riow

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Общий модуль извлечения признаков — тот же что и при обучении
sys.path.insert(0, str(Path(__file__).parent))
from analysis.pixel_features import features_from_array, compute_band_stats

TILE_SIZE     = 512
UNCERTAINTY   = True
RGB_NUTRIENTS = ("N", "P", "K")   # короткие имена для RGB-превью


# ════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════════════

def load_models(models_dir: str) -> dict:
    """
    Загружает модель(и) и meta.json.

    Возвращает унифицированный dict — независимо от режима,
    дальнейший код работает одинаково.
    """
    import joblib
    p = Path(models_dir)
    if not p.exists():
        raise FileNotFoundError(f"Папка моделей не найдена: {p}")

    meta_path = p / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json не найден в {p}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # ── Multi-output (рекомендуемый) ──────────────────────────────────────────
    if meta.get("mode") == "multi_output":
        mp = p / "model_multioutput.pkl"
        if not mp.exists():
            raise FileNotFoundError(f"model_multioutput.pkl не найден в {p}")
        model = joblib.load(mp)
        print(f"  Режим:      multi_output")
        print(f"  Модель:     {meta.get('model','?')}")
        print(f"  Нутриентов: {len(meta['nutrients'])}")
        print(f"  Признаков:  {meta['n_features']}")
        print(f"  R² (CV):")
        for sn, r2 in meta.get("R2_per_nutrient", {}).items():
            mark = "✓" if r2 > 0 else "✗"
            print(f"    {mark} {sn:<6}: {r2:+.3f}")
        return {
            "_mode":        "multi_output",
            "_model":       model,
            "_feat_idx":    np.array(meta["feat_indices"]),
            "_col_median":  np.array(meta["col_median"]),
            "_short_names": meta["short_names"],
            "_nutrients":   meta["nutrients"],
            "_n_out":       len(meta["nutrients"]),
        }

    # ── Per-nutrient (обратная совместимость) ─────────────────────────────────
    loaded = {}
    n_ok = 0
    for el, info in meta.items():
        if not isinstance(info, dict) or "safe_name" not in info:
            continue
        mp = p / f"model_{info['safe_name']}.pkl"
        if not mp.exists():
            print(f"  [!] не найден: {mp.name}")
            continue
        m = joblib.load(mp)
        sc = None
        sp = p / f"scaler_{info['safe_name']}.pkl"
        if sp.exists():
            sc = joblib.load(sp)
        loaded[el] = {
            "model":       m,
            "scaler":      sc,
            "short_name":  info["short_name"],
            "feat_indices":np.array(info["feat_indices"]),
            "col_median":  np.array(info["col_median"]),
            "R2":          info.get("R2_cv", np.nan),
        }
        n_ok += 1
        mark = "✓" if info.get("R2_cv", 0) > 0 else "✗"
        print(f"  {mark} {info['short_name']:<6} R²={info.get('R2_cv',np.nan):+.3f}")

    print(f"\n  Загружено: {n_ok}/{len(meta)} моделей")
    loaded["_mode"]       = "per_nutrient"
    loaded["_n_out"]      = n_ok
    loaded["_short_names"]= [v["short_name"] for k,v in loaded.items()
                               if not k.startswith("_")]
    loaded["_nutrients"]  = [k for k in loaded if not k.startswith("_")]
    return loaded


# ════════════════════════════════════════════════════════════════════════════
#  ПРЕДСКАЗАНИЕ ДЛЯ ТАЙЛА
# ════════════════════════════════════════════════════════════════════════════

def predict_tile(X_tile: np.ndarray,
                  loaded: dict) -> tuple:
    """
    Предсказывает концентрации нутриентов для тайла.

    X_tile: (N, n_all_features) — все возможные признаки
    Возвращает:
      preds   : (N, n_nutrients) float32
      uncert  : (N, n_nutrients) float32  — std деревьев
    """
    N   = X_tile.shape[0]
    n_out = loaded["_n_out"]
    preds  = np.full((N, n_out), np.nan, dtype=np.float32)
    uncert = np.full((N, n_out), np.nan, dtype=np.float32)

    # ── Multi-output ──────────────────────────────────────────────────────────
    if loaded["_mode"] == "multi_output":
        feat_idx   = loaded["_feat_idx"]
        col_median = loaded["_col_median"]
        model      = loaded["_model"]

        # Защита: feat_idx не выходит за границы X_tile
        valid_idx  = feat_idx[feat_idx < X_tile.shape[1]]
        if len(valid_idx) < len(feat_idx):
            print(f"  [!] feat_idx выходит за границы: "
                  f"{len(feat_idx)} → {len(valid_idx)}")
        if len(valid_idx) == 0:
            return preds, uncert

        Xf = X_tile[:, valid_idx].copy()
        # NaN → медиана обучающей выборки
        nan_mask = np.isnan(Xf)
        if nan_mask.any():
            Xf = np.where(nan_mask,
                           col_median[:len(valid_idx)], Xf)
        Xf = np.asarray(Xf, dtype=np.float32)

        try:
            pred_raw = model.predict(Xf)
            preds = np.asarray(pred_raw, dtype=np.float32)
            if preds.ndim == 1:
                preds = preds.reshape(N, 1)

            # Неопределённость из деревьев (ET/RF)
            if UNCERTAINTY and hasattr(model, "estimators_"):
                try:
                    tree_p = np.array(
                        [t.predict(Xf) for t in model.estimators_],
                        dtype=np.float32)
                    if tree_p.ndim == 3:        # (n_trees, N, n_out)
                        uncert = tree_p.std(axis=0)
                    elif tree_p.ndim == 2:       # (n_trees, N)
                        uncert[:, 0] = tree_p.std(axis=0)
                except Exception:
                    pass
        except Exception as e:
            print(f"\n  [predict] ошибка: {e}")

        return preds, uncert

    # ── Per-nutrient ──────────────────────────────────────────────────────────
    for ch, el in enumerate(loaded["_nutrients"]):
        entry    = loaded[el]
        feat_idx = entry["feat_indices"]
        valid_fi = feat_idx[feat_idx < X_tile.shape[1]]
        if len(valid_fi) == 0:
            continue
        Xf = X_tile[:, valid_fi].copy()
        nan_m = np.isnan(Xf)
        if nan_m.any():
            Xf = np.where(nan_m, entry["col_median"][:len(valid_fi)], Xf)
        sc = entry.get("scaler")
        if sc is not None:
            Xf = sc.transform(Xf.astype(np.float64)).astype(np.float32)
        Xf = np.asarray(Xf, dtype=np.float32)
        try:
            preds[:, ch] = entry["model"].predict(Xf).astype(np.float32)
            if UNCERTAINTY and hasattr(entry["model"], "estimators_"):
                tp = np.array([t.predict(Xf)
                                for t in entry["model"].estimators_],
                               dtype=np.float32)
                uncert[:, ch] = tp.std(axis=0)
        except Exception:
            pass

    return preds, uncert


# ════════════════════════════════════════════════════════════════════════════
#  ТАЙЛОВАЯ ОБРАБОТКА ОРТОМОЗАИКИ
# ════════════════════════════════════════════════════════════════════════════

def process_ortho(ortho_path: str,
                   loaded: dict,
                   out_dir: str,
                   win_size: int = 7,
                   tile_size: int = TILE_SIZE) -> tuple:
    """
    Читает ортомозаику тайлами, предсказывает, записывает GeoTIFF.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    short_names = loaded["_short_names"]
    n_nut       = len(short_names)

    print(f"\nОртомозаика: {Path(ortho_path).name}")

    # Band stats (p2/p98) для нормировки
    print("Вычисление band stats...", end=" ", flush=True)
    band_stats = compute_band_stats(ortho_path)
    print(f"OK  ({len(band_stats)} каналов)")
    for i, (lo, hi) in enumerate(band_stats):
        bname = ("Blue","Green","Red","RedEdge","NIR")[i] \
                if i < 5 else f"band{i+1}"
        print(f"  {bname}: [{lo:.3f}, {hi:.3f}]")

    with rasterio.open(ortho_path) as src:
        H, W    = src.height, src.width
        profile = src.profile.copy()

    print(f"Размер: {W}×{H} px")
    n_r = (H + tile_size - 1) // tile_size
    n_c = (W + tile_size - 1) // tile_size
    print(f"Тайлов: {n_r}×{n_c} = {n_r*n_c}")

    # Выходные профили
    out_profile = profile.copy()
    out_profile.update(count=n_nut, dtype="float32",
                        compress="lzw", nodata=np.nan,
                        bigtiff="IF_SAFER")
    unc_profile = out_profile.copy()

    out_tif = str(out / "nutrient_map.tif")
    unc_tif = str(out / "uncertainty_map.tif")

    with rasterio.open(ortho_path) as src, \
         rasterio.open(out_tif, "w", **out_profile) as dst, \
         rasterio.open(unc_tif, "w", **unc_profile) as dst_unc:

        total = n_r * n_c
        done  = 0

        for r0 in range(0, H, tile_size):
            for c0 in range(0, W, tile_size):
                h = min(tile_size, H - r0)
                w = min(tile_size, W - c0)
                win = riow.Window(c0, r0, w, h)

                try:
                    tile = src.read(window=win).astype(np.float32)
                except Exception:
                    done += 1
                    continue

                # Маска nodata (все каналы = 0)
                nodata_mask = (tile == 0).all(axis=0)

                # Извлечение признаков (та же функция что и при обучении)
                X_tile, _ = features_from_array(tile, band_stats, win_size)

                # Предсказание
                pred_flat, unc_flat = predict_tile(X_tile, loaded)

                # Reshape (N=h*w, n_nut) → (n_nut, h, w)
                pred_3d = pred_flat.reshape(h, w, n_nut).transpose(2, 0, 1)
                unc_3d  = unc_flat.reshape(h, w, n_nut).transpose(2, 0, 1)

                # Маскируем nodata
                pred_3d[:, nodata_mask] = np.nan
                unc_3d[:, nodata_mask]  = np.nan

                dst.write(pred_3d,     window=win)
                dst_unc.write(unc_3d,  window=win)

                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  {done}/{total} ({done/total*100:.0f}%)",
                          end="\r", flush=True)

    print(f"\n\nСохранено: {out_tif}")
    print(f"Сохранено: {unc_tif}")

    # Записываем имена каналов в метаданные TIF
    with rasterio.open(out_tif, "r+") as dst:
        for i, sn in enumerate(short_names, 1):
            dst.update_tags(i, name=sn)

    return out_tif, unc_tif, short_names


# ════════════════════════════════════════════════════════════════════════════
#  СТАТИСТИКА И ВИЗУАЛИЗАЦИЯ
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(tif_path: str, short_names: list, out_dir: str):
    """Считает min/mean/max/std для каждого канала, сохраняет CSV."""
    rows = []
    with rasterio.open(tif_path) as src:
        for i, sn in enumerate(short_names, 1):
            data  = src.read(i).astype(np.float32)
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                rows.append({"nutrient": sn, "n_pixels": 0,
                              "min": np.nan, "mean": np.nan,
                              "max": np.nan, "std": np.nan})
                continue
            rows.append({
                "nutrient": sn,
                "n_pixels": int(len(valid)),
                "min":  round(float(valid.min()),  4),
                "mean": round(float(valid.mean()),  4),
                "max":  round(float(valid.max()),  4),
                "std":  round(float(valid.std()),   4),
                "p5":   round(float(np.percentile(valid, 5)),  4),
                "p95":  round(float(np.percentile(valid, 95)), 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "stats.csv", index=False, float_format="%.4f")
    print("\nСтатистика по картам:")
    print(df[["nutrient","mean","std","min","max"]].to_string(index=False))
    return df


def make_preview(tif_path: str, short_names: list,
                  out_dir: str, dpi: int = 150):
    """Превью всех 12 нутриентов + RGB-композит."""
    out = Path(out_dir)
    n   = len(short_names)
    nc  = 4
    nr  = int(np.ceil(n / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = axes.flatten()

    with rasterio.open(tif_path) as src:
        for i, sn in enumerate(short_names):
            data  = src.read(i+1)
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                axes[i].set_visible(False); continue
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            im = axes[i].imshow(data, cmap="RdYlGn",
                                 vmin=p2, vmax=p98,
                                 interpolation="nearest")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].set_title(sn, fontsize=10)
            axes[i].axis("off")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Карты концентраций нутриентов (buffered-LOO prediction)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out / "nutrient_maps_preview.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved: nutrient_maps_preview.png")

    # RGB-композит
    with rasterio.open(tif_path) as src:
        rgb_bands = []
        for target in RGB_NUTRIENTS:
            for i, sn in enumerate(short_names):
                if sn.upper() == target.upper():
                    band = src.read(i+1).astype(np.float32)
                    valid = band[np.isfinite(band)]
                    if len(valid) > 0:
                        lo = np.percentile(valid, 2)
                        hi = np.percentile(valid, 98)
                        band = np.clip((band-lo)/(hi-lo+1e-8), 0, 1)
                        band[~np.isfinite(src.read(i+1))] = 0
                    rgb_bands.append(band)
                    break

        if len(rgb_bands) == 3:
            rgb = np.stack(rgb_bands, axis=-1)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(f"RGB-композит: R={RGB_NUTRIENTS[0]}, "
                         f"G={RGB_NUTRIENTS[1]}, B={RGB_NUTRIENTS[2]}",
                         fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(out / "rgb_composite.png",
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print("saved: rgb_composite.png")


def make_uncertainty_preview(unc_tif: str, short_names: list,
                               out_dir: str, dpi: int = 150):
    """Превью карт неопределённости."""
    out = Path(out_dir)
    n   = len(short_names)
    nc  = 4
    nr  = int(np.ceil(n / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = axes.flatten()

    with rasterio.open(unc_tif) as src:
        for i, sn in enumerate(short_names):
            data  = src.read(i+1)
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                axes[i].set_visible(False); continue
            p98 = np.percentile(valid, 98)
            im  = axes[i].imshow(data, cmap="Reds",
                                  vmin=0, vmax=p98,
                                  interpolation="nearest")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].set_title(f"{sn} σ", fontsize=10)
            axes[i].axis("off")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Неопределённость предсказания (std деревьев)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out / "uncertainty_preview.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved: uncertainty_preview.png")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run(args):
    print("=" * 60)
    print("PIXEL-WISE NUTRIENT MAPPING  (05_pixel_mapping.py)")
    print("=" * 60)

    print(f"\nМодели: {args.models}")
    loaded = load_models(args.models)

    out_tif, unc_tif, short_names = process_ortho(
        ortho_path=args.ortho,
        loaded=loaded,
        out_dir=args.output,
        win_size=args.win_size,
        tile_size=args.tile_size,
    )

    print("\nСтатистика и превью...")
    compute_stats(out_tif, short_names, args.output)
    make_preview(out_tif, short_names, args.output, dpi=args.dpi)
    make_uncertainty_preview(unc_tif, short_names, args.output, dpi=args.dpi)

    print(f"\n{'='*60}")
    print(f"Готово!  →  {args.output}/")
    print(f"  nutrient_map.tif          12-канальный GeoTIFF")
    print(f"  uncertainty_map.tif       карта неопределённости (σ)")
    print(f"  nutrient_maps_preview.png превью всех 12 карт")
    print(f"  rgb_composite.png         RGB: R=N, G=P, B=K")
    print(f"  stats.csv                 статистика по каждому нутриенту")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Pixel-wise mapping of 12 nutrients from UAV ortho")
    ap.add_argument("--models",    required=True,
                    help="папка с pkl-моделями (results/02_honest_kaggle/models)")
    ap.add_argument("--ortho",     required=True,
                    help="ортомозаика .tif (та же что при --single-ortho)")
    ap.add_argument("--output",    default="results/05_maps")
    ap.add_argument("--win-size",  type=int, default=7,
                    help="размер окна (должен совпадать с обучением)")
    ap.add_argument("--tile-size", type=int, default=512,
                    help="размер тайла (больше = быстрее, но больше RAM)")
    ap.add_argument("--dpi",       type=int, default=150)
    run(ap.parse_args())
