"""
analysis/spatial_cv.py — пространственная кросс-валидация и аугментация.

Реализует:
  - Buffered Leave-One-Out (buffered-LOO)
  - BCa bootstrap для доверительных интервалов R²
  - Permutation test
  - C-Mixup (regression-aware аугментация)

Буферы нутриентов получены из 01e_variograms.py (Moran's I + range_m).
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ── Рекомендуемые буферы (из 01e_variograms, обе даты) ─────────────────────

BUFFER_PER_NUTRIENT = {
    "N_%":            30,
    "P_%":           100,
    "K_%":            86,
    "Ca_%":           30,
    "Mg_%":           30,
    "S_%":            55,
    "Нитраты_мг_кг":  30,
    "Fe_мг_кг":      100,
    "Mn_мг_кг":      100,
    "Cu_мг_кг":       30,
    "Zn_мг_кг":       31,
    "Co_мг_кг":       76,
}

GLOBAL_BUFFER_M = 50   # единый буфер если нутриент не задан


# ── Метрики ─────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                "n": len(yt), "RPD": np.nan}
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = float(np.mean(np.abs(yt - yp)))
    rpd  = float(np.std(yt) / (rmse + 1e-12))
    return {"R2": round(r2,4), "RMSE": round(rmse,4),
            "MAE": round(mae,4), "n": len(yt), "RPD": round(rpd,4)}


# ── Buffered Leave-One-Out CV ─────────────────────────────────────────────────

def buffered_loo_cv(model, X, y, coords, buffer_radius,
                    scale=True, min_train=10):
    """
    Buffered Leave-One-Out кросс-валидация.

    Для каждой тестовой точки i:
      - Исключаем из обучения все точки в радиусе buffer_radius вокруг i
      - Обучаем модель на оставшихся точках
      - Предсказываем для точки i

    Параметры
    ----------
    model        : sklearn-совместимый регрессор
    X            : np.ndarray (n, p)
    y            : np.ndarray (n,)
    coords       : np.ndarray (n, 2) в метрах
    buffer_radius: float, метры
    scale        : bool, применять ли StandardScaler внутри fold
    min_train    : int, минимальное число обучающих точек

    Возвращает
    ----------
    dict с y_true, y_pred, metrics
    """
    mask = np.isfinite(y)
    Xc, yc, cc = X[mask], y[mask], coords[mask]
    n = len(yc)

    tree_dists = cdist(cc, cc)   # (n, n) попарные расстояния
    y_pred = np.full(n, np.nan)

    for i in range(n):
        # Точки за пределами буфера → обучение
        excluded = tree_dists[i] < buffer_radius  # включает саму точку
        train_idx = np.where(~excluded)[0]
        if len(train_idx) < min_train:
            continue

        Xtr, ytr = Xc[train_idx], yc[train_idx]
        Xte       = Xc[[i]]

        if scale:
            sc  = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        try:
            m = clone(model)
            m.fit(Xtr, ytr)
            y_pred[i] = float(m.predict(Xte)[0])
        except Exception:
            pass

    valid = np.isfinite(y_pred)
    metrics = regression_metrics(yc[valid], y_pred[valid])
    return {
        "y_true":  yc,
        "y_pred":  y_pred,
        "metrics": metrics,
        "n_predicted": valid.sum(),
        "n_skipped":   (~valid).sum(),
    }


def sweep_buffer_radii(model, X, y, coords,
                       radii=(0, 20, 30, 50, 75, 100),
                       scale=True):
    """
    Прогон buffered-LOO по нескольким радиусам буфера.
    Возвращает DataFrame: radius → R², RMSE, n_predicted.
    Полезно для диагностики — кривая R²(radius) показывает
    насколько результат зависит от пространственной автокорреляции.
    """
    rows = []
    for r in radii:
        res = buffered_loo_cv(model, X, y, coords, r, scale)
        rows.append({
            "buffer_m":    r,
            "R2":          res["metrics"]["R2"],
            "RMSE":        res["metrics"]["RMSE"],
            "n_predicted": res["n_predicted"],
        })
    return pd.DataFrame(rows)


# ── BCa Bootstrap для доверительных интервалов ───────────────────────────────

def bca_bootstrap_r2(y_true, y_pred, B=2000, alpha=0.05, seed=42):
    """
    BCa (Bias-Corrected and Accelerated) Bootstrap для R².

    Возвращает (r2_point, ci_lower, ci_upper).

    BCa предпочтительнее обычного перцентильного bootstrap
    при асимметричных распределениях (типично для R² при малых n).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        r2 = r2_score(yt, yp)
        return r2, np.nan, np.nan

    rng   = np.random.default_rng(seed)
    theta = r2_score(yt, yp)

    # Bootstrap replications
    boots = np.empty(B)
    for b in range(B):
        idx      = rng.integers(0, n, n)
        boots[b] = r2_score(yt[idx], yp[idx])

    # Bias correction: z0
    z0 = stats.norm.ppf(np.mean(boots < theta) + 1e-10)

    # Acceleration: jackknife
    jk = np.array([r2_score(np.delete(yt, i), np.delete(yp, i))
                   for i in range(n)])
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk)**3)
    den = 6 * (np.sum((jk_mean - jk)**2))**1.5 + 1e-12
    a   = num / den

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a*(z0 + z)))

    p_lo = adj(z_lo)
    p_hi = adj(z_hi)
    ci_lo, ci_hi = np.quantile(boots, [p_lo, p_hi])
    return round(float(theta), 4), round(float(ci_lo), 4), round(float(ci_hi), 4)


# ── Permutation test ──────────────────────────────────────────────────────────

def permutation_test_r2(model, X, y, coords, buffer_radius,
                        n_perm=199, scale=True, seed=42):
    """
    Пространственный permutation test.
    H0: R² наблюдаемый не превышает R² случайной модели.

    Перемешиваем y (не X!) и прогоняем buffered-LOO.
    p-value = доля permuted R² >= observed R².
    """
    rng = np.random.default_rng(seed)
    obs = buffered_loo_cv(model, X, y, coords, buffer_radius, scale)
    obs_r2 = obs["metrics"]["R2"]
    if np.isnan(obs_r2):
        return {"observed_R2": np.nan, "p_value": np.nan,
                "perm_mean_R2": np.nan, "significant": False}

    mask = np.isfinite(y)
    yc   = y[mask]
    perm_r2s = []

    for _ in range(n_perm):
        y_perm      = y.copy()
        y_perm[mask] = rng.permutation(yc)
        res = buffered_loo_cv(model, X, y_perm, coords, buffer_radius, scale)
        r2  = res["metrics"]["R2"]
        if np.isfinite(r2):
            perm_r2s.append(r2)

    if not perm_r2s:
        return {"observed_R2": obs_r2, "p_value": np.nan,
                "perm_mean_R2": np.nan, "significant": False}

    perm_arr = np.array(perm_r2s)
    p_val    = float((perm_arr >= obs_r2).mean())

    return {
        "observed_R2":  round(obs_r2, 4),
        "perm_mean_R2": round(float(perm_arr.mean()), 4),
        "perm_std_R2":  round(float(perm_arr.std()), 4),
        "p_value":      round(p_val, 4),
        "significant":  p_val < 0.05,
        "n_perm":       len(perm_arr),
    }


# ── C-Mixup аугментация (regression-aware) ───────────────────────────────────

def c_mixup(X, y, alpha=2.0, bandwidth=None, n_aug=None, seed=42):
    """
    C-Mixup (Yao et al., 2022) — regression-aware вариант MixUp.

    В отличие от стандартного MixUp, пары формируются с вероятностью
    пропорциональной близости в пространстве меток y:
        w_ij ∝ exp(-(y_i - y_j)² / (2 * bw²))

    Это гарантирует что интерполированные образцы лежат в «физически
    реалистичной» области и не создают синтетические пары из
    противоположных концов распределения.

    Параметры
    ----------
    X         : np.ndarray (n, p)
    y         : np.ndarray (n,)
    alpha     : float, параметр Beta-распределения (2.0 рекомендован)
    bandwidth : float или None. Если None — медиана |y_i - y_j| / 2
    n_aug     : int или None. Если None — удваиваем выборку (n_aug = n)
    seed      : int

    Возвращает
    ----------
    X_aug, y_aug — объединение оригинальных и синтетических точек
    """
    mask = np.isfinite(y)
    Xc, yc = X[mask].astype(float), y[mask].astype(float)
    n = len(yc)

    # NaN в X заменяем медианой (уже должны быть импутированы до вызова,
    # но на всякий случай — защитная импутация)
    if np.isnan(Xc).any():
        col_med = np.nanmedian(Xc, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xc = np.where(np.isnan(Xc), col_med, Xc)

    if n_aug is None:
        n_aug = n

    if bandwidth is None:
        diffs = np.abs(yc[:, None] - yc[None, :])
        bandwidth = np.median(diffs[diffs > 0]) / 2 if diffs.max() > 0 else 1.0

    rng = np.random.default_rng(seed)

    X_syn = np.empty((n_aug, Xc.shape[1]))
    y_syn = np.empty(n_aug)

    for k in range(n_aug):
        # Выбираем первую точку случайно
        i = rng.integers(0, n)

        # Вероятности выбора второй точки ~ близость в y
        dy2  = (yc - yc[i])**2
        w    = np.exp(-dy2 / (2 * bandwidth**2 + 1e-12))
        w[i] = 0.0
        w   /= w.sum() + 1e-12
        j    = rng.choice(n, p=w)

        lam     = float(rng.beta(alpha, alpha))
        X_syn[k] = lam * Xc[i] + (1 - lam) * Xc[j]
        y_syn[k] = lam * yc[i] + (1 - lam) * yc[j]

    X_out = np.vstack([Xc, X_syn])
    y_out = np.concatenate([yc, y_syn])
    return X_out, y_out


def c_mixup_with_coords(X, y, coords, alpha=2.0, bandwidth=None,
                        n_aug=None, seed=42):
    """
    C-Mixup с возвратом синтетических координат (для spatial CV).
    Координаты синтетической точки — среднее координат родителей.
    """
    mask = np.isfinite(y)
    Xc, yc, cc = X[mask].astype(float), y[mask].astype(float), coords[mask]
    n = len(yc)
    if n_aug is None:
        n_aug = n

    if bandwidth is None:
        diffs = np.abs(yc[:, None] - yc[None, :])
        bandwidth = np.median(diffs[diffs > 0]) / 2 if diffs.max() > 0 else 1.0

    rng = np.random.default_rng(seed)
    X_syn = np.empty((n_aug, Xc.shape[1]))
    y_syn = np.empty(n_aug)
    c_syn = np.empty((n_aug, 2))

    for k in range(n_aug):
        i   = rng.integers(0, n)
        dy2 = (yc - yc[i])**2
        w   = np.exp(-dy2 / (2 * bandwidth**2 + 1e-12))
        w[i] = 0.0
        w   /= w.sum() + 1e-12
        j    = rng.choice(n, p=w)

        lam       = float(rng.beta(alpha, alpha))
        X_syn[k]  = lam * Xc[i] + (1 - lam) * Xc[j]
        y_syn[k]  = lam * yc[i] + (1 - lam) * yc[j]
        c_syn[k]  = lam * cc[i] + (1 - lam) * cc[j]

    return (np.vstack([Xc, X_syn]),
            np.concatenate([yc, y_syn]),
            np.vstack([cc, c_syn]))


# ── Вспомогательная: загрузка буфера из файла или дефолт ─────────────────────

def load_buffer_map(csv_path=None):
    """
    Загружает recommended_buffers.csv из 01e_variograms,
    возвращает dict {element: buffer_m}.
    Если файл не найден — возвращает встроенные константы.
    """
    if csv_path:
        try:
            df  = pd.read_csv(csv_path)
            out = dict(zip(df["element"], df["recommended_buffer_m"]))
            print(f"  Буферы загружены из {csv_path}")
            return out
        except Exception as e:
            print(f"  Не удалось загрузить буферы: {e}. Используем дефолт.")
    return BUFFER_PER_NUTRIENT.copy()
