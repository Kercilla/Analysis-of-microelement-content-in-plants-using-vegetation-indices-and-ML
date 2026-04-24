"""
analysis/pixel_features.py — единая функция извлечения признаков из tif.

"""
import numpy as np
import rasterio
import rasterio.windows as riow
from pathlib import Path

BAND_NAMES = ("Blue", "Green", "Red", "RedEdge", "NIR")


# ── Band stats ────────────────────────────────────────────────────────────────

def compute_band_stats(tif_path: str,
                        n_samples: int = 40,
                        tile_size: int = 256) -> list:
    """
    Оценивает p2/p98 для нормировки каналов по случайным тайлам.
    Не читает весь растр — избегает OOM на больших файлах.

    Возвращает list[(lo, hi)] длиной src.count.
    """
    with rasterio.open(tif_path) as src:
        H, W   = src.height, src.width
        n_cols = max(W // tile_size, 1)
        n_rows = max(H // tile_size, 1)
        step_r = max(n_rows // max(int(np.sqrt(n_samples)), 1), 1)
        step_c = max(n_cols // max(int(np.sqrt(n_samples)), 1), 1)
        accum  = [[] for _ in range(src.count)]

        for ri in range(0, n_rows, step_r):
            for ci in range(0, n_cols, step_c):
                r0 = ri * tile_size
                c0 = ci * tile_size
                h  = min(tile_size, H - r0)
                w  = min(tile_size, W - c0)
                if h <= 0 or w <= 0:
                    continue
                try:
                    win  = riow.Window(col_off=c0, row_off=r0,
                                        width=w, height=h)
                    data = src.read(window=win)
                    for i in range(src.count):
                        v = data[i].astype(np.float32).ravel()
                        v = v[v > 0]
                        if len(v) > 0:
                            step = max(len(v) // 500, 1)
                            accum[i].append(v[::step])
                except Exception:
                    continue

    stats = []
    for vals_list in accum:
        if not vals_list:
            stats.append((0.0, 1.0))
        else:
            vals = np.concatenate(vals_list)
            stats.append((float(np.percentile(vals, 2)),
                           float(np.percentile(vals, 98))))
    return stats


# ── VI ────────────────────────────────────────────────────────────────────────

def compute_vi(bands: dict) -> dict:
    """
    44 вегетационных индекса из нормированных каналов.
    bands: {'Blue': arr, 'Green': arr, 'Red': arr, 'RedEdge': arr, 'NIR': arr}
    Все массивы одного shape.
    """
    eps = 1e-8
    B   = bands.get("Blue",    np.zeros(1, np.float32))
    G   = bands.get("Green",   np.zeros(1, np.float32))
    R   = bands.get("Red",     np.zeros(1, np.float32))
    RE  = bands.get("RedEdge", np.zeros(1, np.float32))
    NIR = bands.get("NIR",     np.zeros(1, np.float32))

    def nd(a, b):  return (a - b) / (a + b + eps)
    def rat(a, b): return a / (b + eps)

    NDVI = nd(NIR, R)
    vi = {
        "NDVI":          NDVI,
        "GNDVI":         nd(NIR, G),
        "NDRE":          nd(NIR, RE),
        "BNDVI":         nd(NIR, B),
        "SAVI":          1.5*(NIR-R)/(NIR+R+0.5+eps),
        "EVI":           2.5*(NIR-R)/(NIR+6*R-7.5*B+1+eps),
        "EVI2":          2.5*(NIR-R)/(NIR+2.4*R+1+eps),
        "OSAVI":         1.16*(NIR-R)/(NIR+R+0.16+eps),
        "MSR":           (rat(NIR,R)-1)/(np.sqrt(rat(NIR,R)+eps)+eps),
        "CIgreen":       rat(NIR,G) - 1,
        "CIre":          rat(NIR,RE) - 1,
        "CCCI":          nd(NIR,RE)/(nd(NIR,R)+eps),
        "PSRI":          (R-B)/(RE+eps),
        "MCARI":         ((RE-R)-0.2*(RE-G))*rat(RE,R),
        "TCARI":         3*((RE-R)-0.2*(RE-G)*rat(RE,R)),
        "MCARI_OSAVI":   ((RE-R)-0.2*(RE-G))*rat(RE,R) /
                         (1.16*(NIR-R)/(NIR+R+0.16+eps)+eps),
        "RDVI":          (NIR-R)/(np.sqrt(NIR+R+eps)),
        "WDRVI":         (0.1*NIR-R)/(0.1*NIR+R+eps),
        "GRNDVI":        nd(NIR, G+R),
        "TGI":           G - 0.39*R - 0.61*B,
        "GLI":           (2*G-R-B)/(2*G+R+B+eps),
        "VARI":          (G-R)/(G+R-B+eps),
        "NGRDI":         nd(G, R),
        "SR":            rat(NIR, R),
        "SRre":          rat(NIR, RE),
        "DVI":           NIR - R,
        "NDVI_RE":       nd(RE, R),
        "LSWI":          nd(NIR, RE),
        "MTCI":          (NIR-RE)/(RE-R+eps),
        "IRECI":         (NIR-R)/(rat(RE,G)+eps),
        "NLI":           (NIR**2-R)/(NIR**2+R+eps),
        "MSRre":         (rat(NIR,RE)-1)/(np.sqrt(rat(NIR,RE)+eps)+eps),
        "RTVIcore":      100*(NIR-RE) - 10*(NIR-G),
        "CRI700":        rat(1.0, G+eps) - rat(1.0, RE+eps),
        "ARI":           rat(1.0, G+eps) - rat(1.0, RE+eps),
        "SRPI":          rat(B, R),
        "NPCI":          (R-B)/(R+B+eps),
        "BGI":           rat(B, G),
        "GI":            rat(G, R),
        "Datt":          nd(NIR,RE)/(nd(NIR,R)+eps),
        "RGRI":          rat(R, G),
        "NDVI_blue":     nd(NIR, B),
        "NIRv":          NDVI * NIR,
    }
    for k in vi:
        vi[k] = np.nan_to_num(vi[k], nan=0.0, posinf=1.0, neginf=-1.0)
        vi[k] = np.clip(vi[k], -10, 10)
    return vi


# ── Главная функция: признаки из тайла ───────────────────────────────────────

def features_from_array(tile: np.ndarray,
                          band_stats: list,
                          win_size: int = 7) -> tuple:
    """
    Извлекает признаки из numpy-массива (n_bands, H, W).

    Возвращает:
      X          : np.ndarray (H*W, n_features)  float32
      feat_names : list[str]
    """
    n_bands, H, W = tile.shape
    eps  = 1e-8
    half = win_size // 2

    # ── Нормировка ────────────────────────────────────────────────────────────
    bands_raw = {}
    for i, name in enumerate(BAND_NAMES[:n_bands]):
        lo, hi = band_stats[i]
        b = tile[i].astype(np.float32)
        bands_raw[name] = np.clip((b - lo) / (hi - lo + eps), 0, 1)

    feat_list  = []
    feat_names = []

    # ── 5 сырых каналов ───────────────────────────────────────────────────────
    for name, arr in bands_raw.items():
        feat_list.append(arr.ravel())
        feat_names.append(f"{name}_raw")

    # ── Window-статистики + средние для VI ────────────────────────────────────
    bands_win_mean = {}

    for name, arr in bands_raw.items():
        arr_pad = np.pad(arr, half, mode="reflect")
        slid    = np.lib.stride_tricks.sliding_window_view(
                    arr_pad, (win_size, win_size)).reshape(H, W, -1)

        wmean   = slid.mean(axis=-1)
        wstd    = slid.std(axis=-1)
        wmed    = np.median(slid, axis=-1)
        wp25    = np.percentile(slid, 25, axis=-1)
        wp75    = np.percentile(slid, 75, axis=-1)
        wcv     = wstd / (wmean + eps)

        bands_win_mean[name] = wmean

        for stat, sarr in [("mean",wmean),("std",wstd),("median",wmed),
                             ("p25",wp25),("p75",wp75),("cv",wcv)]:
            feat_list.append(sarr.ravel())
            feat_names.append(f"{name}_w{win_size}_{stat}")

    # ── VI из window-средних ──────────────────────────────────────────────────
    vi = compute_vi(bands_win_mean)
    for vi_name, vi_arr in vi.items():
        feat_list.append(vi_arr.ravel())
        feat_names.append(vi_name)

    X = np.column_stack(feat_list).astype(np.float32)
    return X, feat_names


# ── Извлечение признаков в заданных координатах ───────────────────────────────

def extract_features_at_points(tif_path: str,
                                 coords_xy: np.ndarray,
                                 win_size: int = 7) -> tuple:
    """
    Извлекает признаки в заданных координатах из tif.

    Параметры
    ----------
    coords_xy : np.ndarray (n, 2) — координаты в СК растра
    win_size  : размер окна в пикселях

    Возвращает
    ----------
    X          : np.ndarray (n_valid, n_features)
    feat_names : list[str]
    valid_mask : np.ndarray (n,) bool
    """
    band_stats = compute_band_stats(tif_path)
    half       = win_size // 2

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        tf   = src.transform

        rows_arr, cols_arr = rasterio.transform.rowcol(
            tf, coords_xy[:, 0], coords_xy[:, 1])
        rows_arr = np.asarray(rows_arr, dtype=int)
        cols_arr = np.asarray(cols_arr, dtype=int)

        valid = ((rows_arr >= half) & (rows_arr < H - half) &
                 (cols_arr >= half) & (cols_arr < W - half))

        feat_list_pts  = []
        feat_names_out = None

        for i in range(len(coords_xy)):
            if not valid[i]:
                continue
            r, c = rows_arr[i], cols_arr[i]
            r0   = max(0, r - half)
            c0   = max(0, c - half)
            h_   = min(win_size, H - r0)
            w_   = min(win_size, W - c0)
            try:
                win  = riow.Window(col_off=c0, row_off=r0,
                                    width=w_, height=h_)
                tile = src.read(window=win).astype(np.float32)

                # Паддинг краевых патчей
                if tile.shape[1] < win_size or tile.shape[2] < win_size:
                    ph = win_size - tile.shape[1]
                    pw = win_size - tile.shape[2]
                    tile = np.pad(tile, ((0,0),(0,ph),(0,pw)), mode="reflect")

                X_tile, names = features_from_array(tile, band_stats, win_size)
                # Центральный пиксель
                center = half * win_size + half
                feat_list_pts.append(X_tile[center])
                if feat_names_out is None:
                    feat_names_out = names

            except Exception:
                valid[i] = False

    if not feat_list_pts:
        return np.empty((0, 0), dtype=np.float32), [], valid

    X = np.vstack(feat_list_pts).astype(np.float32)
    return X, feat_names_out or [], valid
