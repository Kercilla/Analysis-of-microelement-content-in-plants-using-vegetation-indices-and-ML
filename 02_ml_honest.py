#!/usr/bin/env python3
"""
02_ml_honest.py — честный ML-пайплайн (Этап 3).

Запуск:
    python 02_ml_honest.py
    python 02_ml_honest.py --ortho data/map_tif/20230608_F14_Pollux.tif
    python 02_ml_honest.py --multi-output --pca 0.95
    python 02_ml_honest.py --skip-perm --bca-b 500   # быстрый тест
"""
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.spatial_cv import (
    bca_bootstrap_r2, permutation_test_r2,
    sweep_buffer_radii, c_mixup, load_buffer_map,
    GLOBAL_BUFFER_M, regression_metrics,
)
from analysis.cv_pipeline import world_to_pixel, compute_band_stats

import rasterio
import rasterio.windows as riow
import geopandas as gpd
from scipy.spatial.distance import cdist

from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestRegressor,
                               ExtraTreesRegressor,
                               StackingRegressor)
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV, BayesianRidge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  xgboost не установлен; XGB пропущен")

try:
    import lightgbm as lgb

    class _LGBMNumpy(lgb.LGBMRegressor):
        """Принудительная конвертация X в numpy перед fit/predict."""
        def fit(self, X, y, **kw):
            return super().fit(np.asarray(X), y, **kw)
        def predict(self, X, **kw):
            return super().predict(np.asarray(X), **kw)

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  lightgbm не установлен; LGB пропущен")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  optuna не установлен; байесовская оптимизация пропущена")


# ════════════════════════════════════════════════════════════════════════════
#  PLSR с CV-тюнингом n_components (Fix #5)
# ════════════════════════════════════════════════════════════════════════════

class PlsrCV(BaseEstimator, RegressorMixin):
    
    def __init__(self, max_components=15, cv=5, scale=True, random_state=42):
        self.max_components = max_components
        self.cv             = cv
        self.scale          = scale
        self.random_state   = random_state

    def fit(self, X, y):
        n_samp, n_feat = np.asarray(X).shape
        max_k = min(self.max_components, n_feat, n_samp - 2)
        max_k = max(1, max_k)

        best_k, best_r2 = 1, -np.inf
        cv = KFold(n_splits=min(self.cv, n_samp), shuffle=True,
                   random_state=self.random_state)
        for k in range(1, max_k + 1):
            try:
                m = PLSRegression(n_components=k, scale=self.scale)
                scores = cross_val_score(m, X, y, cv=cv,
                                          scoring="r2", n_jobs=1)
                mean_r2 = float(np.mean(scores))
                if np.isfinite(mean_r2) and mean_r2 > best_r2:
                    best_k, best_r2 = k, mean_r2
            except Exception:
                continue

        self.n_components_ = best_k
        self.estimator_    = PLSRegression(n_components=best_k,
                                            scale=self.scale)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(X)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred


# ════════════════════════════════════════════════════════════════════════════
#  МОДЕЛИ
# ════════════════════════════════════════════════════════════════════════════

def get_base_models():
    
    from sklearn.svm import SVR

    models = {
        "Ridge":        (Ridge(alpha=1.0),                               True),
        "ElasticNet":   (ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000), True),
        "BayesianRidge":(BayesianRidge(),                                True),
        "SVR":          (SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1), True),
        "PLSR":         (PlsrCV(max_components=15, cv=5, scale=True),    False),  # fix #1,#5
        "RF":           (RandomForestRegressor(n_estimators=200,
                                                max_features="sqrt",
                                                min_samples_leaf=3,
                                                random_state=42, n_jobs=-1), False),
        "ET":           (ExtraTreesRegressor(n_estimators=200,
                                              min_samples_leaf=3,
                                              random_state=42, n_jobs=-1),  False),
        "GBR":          (GradientBoostingRegressor(n_estimators=200,
                                                    max_depth=4,
                                                    learning_rate=0.05,
                                                    subsample=0.8,
                                                    random_state=42),        False),
    }
    if HAS_XGB:
        models["XGB"] = (XGBRegressor(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, random_state=42,
                                       verbosity=0, n_jobs=-1),             False)
    if HAS_LGB:
        models["LGB"] = (_LGBMNumpy(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42, verbose=-1,
                                     n_jobs=-1),                            False)
    return models


def get_multioutput_models(n_outputs: int = 12):
    
    from sklearn.neural_network import MLPRegressor

    models = {
        # Native multi-output (один fit — все выходы)
        "ET":    (ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                       random_state=42, n_jobs=-1),     "native"),
        "RF":    (RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                         random_state=42, n_jobs=-1),   "native"),
        "PLSR":  (PlsrCV(max_components=15, cv=5, scale=True),           "native"),
        "MLP":   (MLPRegressor(hidden_layer_sizes=(128, 64),
                                activation="relu", max_iter=500,
                                learning_rate="adaptive",
                                random_state=42),                        "native"),
        # Wrapper (per-output, корректно обрабатывает NaN)
        "MO_GBR":(MultiOutputRegressor(
                     GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                learning_rate=0.05,
                                                random_state=42),
                     n_jobs=-1),                                          "wrapper"),
    }
    if HAS_XGB:
        models["MO_XGB"] = (MultiOutputRegressor(
                                XGBRegressor(n_estimators=200, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              random_state=42, verbosity=0),
                                n_jobs=-1),                              "wrapper")
    if HAS_LGB:
        models["MO_LGB"] = (MultiOutputRegressor(
                                _LGBMNumpy(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, subsample=0.8,
                                            random_state=42, verbose=-1),
                                n_jobs=-1),                              "wrapper")
    return models


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-OUTPUT Y-SCALER (Fix #7)
# ════════════════════════════════════════════════════════════════════════════

class MultiOutputYScaler:
    
    def fit(self, y_mat):
        self.mean_ = np.nanmean(y_mat, axis=0)
        self.std_  = np.nanstd(y_mat, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, y_mat):
        return (y_mat - self.mean_) / self.std_

    def fit_transform(self, y_mat):
        self.fit(y_mat)
        return self.transform(y_mat)

    def inverse_transform(self, y_scaled):
        return y_scaled * self.std_ + self.mean_


def c_mixup_mo(X, y_mat, alpha=2.0, n_aug=None, sigma=1.0, rng=None):
    
    rng = rng if rng is not None else np.random.default_rng(42)
    if n_aug is None:
        n_aug = len(X)

    # NaN-aware: считаем расстояния только по валидным парным координатам
    y_fill = np.where(np.isnan(y_mat), 0.0, y_mat)
    d = cdist(y_fill, y_fill)
    n = len(X)

    w = np.exp(-d**2 / (2 * sigma**2))
    np.fill_diagonal(w, 0.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1.0
    w = w / w_sum

    idx_a = rng.integers(0, n, size=n_aug)
    idx_b = np.array([rng.choice(n, p=w[i]) for i in idx_a])
    lam   = rng.beta(alpha, alpha, size=n_aug).reshape(-1, 1)

    X_aug = lam * X[idx_a] + (1 - lam) * X[idx_b]
    # Для y: NaN сохраняются если в обоих источниках NaN, иначе берём валидное
    y_a, y_b = y_mat[idx_a], y_mat[idx_b]
    # Если в одной из пар NaN — используем другое значение (без смешивания)
    y_aug = lam * np.where(np.isnan(y_a), y_b, y_a) + \
            (1 - lam) * np.where(np.isnan(y_b), y_a, y_b)
    # Если в обеих NaN — результат NaN (так и должно быть)
    both_nan = np.isnan(y_a) & np.isnan(y_b)
    y_aug[both_nan] = np.nan

    X_out = np.vstack([X, X_aug])
    y_out = np.vstack([y_mat, y_aug])
    return X_out, y_out


# ════════════════════════════════════════════════════════════════════════════
#  PCA helper (Fix #2)
# ════════════════════════════════════════════════════════════════════════════

def _apply_pca_in_fold(Xtr, Xte, pca_components):
    """
    pca_components может быть:
      - None или 0 → PCA не применяется
      - int > 0    → фиксированное число компонент
      - float 0<p<1→ % дисперсии (напр. 0.95)
      - -1         → 95% дисперсии (legacy)
    """
    if pca_components is None or pca_components == 0:
        return Xtr, Xte

    if isinstance(pca_components, float) and 0 < pca_components < 1:
        n_comp_spec = pca_components
    elif pca_components == -1:
        n_comp_spec = 0.95
    elif isinstance(pca_components, (int, np.integer)) and pca_components > 0:
        max_k = min(pca_components, Xtr.shape[1], Xtr.shape[0] - 1)
        if max_k < 1:
            return Xtr, Xte
        n_comp_spec = int(max_k)
    else:
        return Xtr, Xte

    try:
        pca = PCA(n_components=n_comp_spec, random_state=42)
        Xtr_p = pca.fit_transform(Xtr)
        Xte_p = pca.transform(Xte)
        return Xtr_p, Xte_p
    except Exception:
        return Xtr, Xte


# ════════════════════════════════════════════════════════════════════════════
#  OPTUNA TUNING (без изменений, только cleanup)
# ════════════════════════════════════════════════════════════════════════════

def tune_xgb_optuna(X_train, y_train, n_trials=40, cv_inner=5, seed=42):
    if not HAS_OPTUNA or not HAS_XGB:
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, random_state=42, verbosity=0)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 400),
            "max_depth":       trial.suggest_int("max_depth", 2, 6),
            "learning_rate":   trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("col", 0.5, 1.0),
            "min_child_weight":trial.suggest_int("mcw", 1, 10),
            "reg_alpha":       trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda":      trial.suggest_float("lambda", 1e-4, 10, log=True),
            "random_state": seed, "verbosity": 0, "n_jobs": -1,
        }
        model  = XGBRegressor(**params)
        cv     = KFold(n_splits=cv_inner, shuffle=True, random_state=seed)
        sc     = StandardScaler()
        Xs     = sc.fit_transform(X_train)
        preds  = cross_val_predict(model, Xs, y_train, cv=cv)
        return -r2_score(y_train, preds)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    b = study.best_params
    return XGBRegressor(
        n_estimators=b["n_estimators"], max_depth=b["max_depth"],
        learning_rate=b["lr"], subsample=b["subsample"],
        colsample_bytree=b["col"], min_child_weight=b["mcw"],
        reg_alpha=b["alpha"], reg_lambda=b["lambda"],
        random_state=seed, verbosity=0, n_jobs=-1,
    )


def tune_lgb_optuna(X_train, y_train, n_trials=40, cv_inner=5, seed=42):
    if not HAS_OPTUNA or not HAS_LGB:
        return _LGBMNumpy(n_estimators=200, random_state=seed,
                          verbose=-1) if HAS_LGB else None

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 400),
            "max_depth":       trial.suggest_int("max_depth", 2, 7),
            "learning_rate":   trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("col", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("mcs", 5, 30),
            "reg_alpha":       trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda":      trial.suggest_float("lambda", 1e-4, 10, log=True),
            "random_state": seed, "verbose": -1, "n_jobs": -1,
        }
        model  = _LGBMNumpy(**params)
        cv     = KFold(n_splits=cv_inner, shuffle=True, random_state=seed)
        sc     = StandardScaler()
        Xs     = sc.fit_transform(X_train)
        preds  = cross_val_predict(model, Xs, y_train, cv=cv)
        return -r2_score(y_train, preds)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    b = study.best_params
    return _LGBMNumpy(
        n_estimators=b["n_estimators"], max_depth=b["max_depth"],
        learning_rate=b["lr"], subsample=b["subsample"],
        colsample_bytree=b["col"], min_child_samples=b["mcs"],
        reg_alpha=b["alpha"], reg_lambda=b["lambda"],
        random_state=seed, verbose=-1, n_jobs=-1,
    )


def build_stacking(X_train, y_train, cv_inner=5):
    estimators = [("ET", ExtraTreesRegressor(n_estimators=200,
                                              min_samples_leaf=3,
                                              random_state=42, n_jobs=-1)),
                  ("RF", RandomForestRegressor(n_estimators=200,
                                               min_samples_leaf=3,
                                               random_state=42, n_jobs=-1)),
                  ("EN", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000))]
    if HAS_XGB:
        estimators.append(("XGB", XGBRegressor(n_estimators=200, max_depth=4,
                                                 learning_rate=0.05,
                                                 subsample=0.8, random_state=42,
                                                 verbosity=0, n_jobs=-1)))
    if HAS_LGB:
        estimators.append(("LGB", _LGBMNumpy(n_estimators=200,
                                              learning_rate=0.05,
                                              random_state=42,
                                              verbose=-1, n_jobs=-1)))
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=KFold(n_splits=cv_inner, shuffle=True, random_state=42),
        n_jobs=-1,
    )


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ════════════════════════════════════════════════════════════════════════════

def select_features_mi(X, y, top_k=50):
    """Отбор top-k признаков через Mutual Information. Для per-nutrient."""
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    col_med = np.nanmedian(Xc, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    Xc_imp  = np.where(np.isnan(Xc), col_med, Xc)
    mi = mutual_info_regression(Xc_imp, yc, n_neighbors=5, random_state=42)
    if top_k >= X.shape[1]:
        return np.ones(X.shape[1], dtype=bool)
    idx = np.argsort(mi)[::-1][:top_k]
    feat_mask = np.zeros(X.shape[1], dtype=bool)
    feat_mask[idx] = True
    return feat_mask


def select_features_mi_mo(X, y_mat, top_k=50):
    
    n_feat    = X.shape[1]
    mi_ranked = np.zeros(n_feat)
    n_used    = 0

    for j in range(y_mat.shape[1]):
        y_j  = y_mat[:, j].astype(float)
        mask = np.isfinite(y_j)
        if mask.sum() < 10:
            continue
        Xc  = X[mask].copy()
        col_med = np.nanmedian(Xc, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xc = np.where(np.isnan(Xc), col_med, Xc)
        mi = mutual_info_regression(Xc, y_j[mask],
                                     n_neighbors=5, random_state=42)
        # Нормализация к [0, 1]
        mi_max = mi.max() if mi.max() > 0 else 1.0
        mi_ranked += mi / mi_max
        n_used += 1

    if n_used == 0 or top_k >= n_feat:
        return np.ones(n_feat, dtype=bool)

    mi_ranked /= n_used
    top_idx = np.argsort(mi_ranked)[::-1][:top_k]
    mask = np.zeros(n_feat, dtype=bool)
    mask[top_idx] = True
    return mask


def select_features_boruta(X, y, max_iter=50, seed=42):
    try:
        from BorutaShap import BorutaShap
        mask_fin = np.isfinite(y)
        Xc, yc   = X[mask_fin], y[mask_fin]
        col_med  = np.nanmedian(Xc, axis=0)
        Xc_imp   = np.where(np.isnan(Xc), col_med, Xc)
        rf = RandomForestRegressor(n_estimators=100, random_state=seed,
                                    n_jobs=-1)
        selector = BorutaShap(model=rf, importance_measure="shap",
                               classification=False)
        selector.fit(X=pd.DataFrame(Xc_imp), y=yc,
                     n_trials=max_iter, sample=False,
                     train_or_test="train", normalize=True, verbose=False)
        accepted = selector.Subset().columns.tolist()
        col_names = [str(i) for i in range(X.shape[1])]
        feat_mask = np.array([c in accepted for c in col_names])
        print(f"    BorutaShap: {feat_mask.sum()} из {X.shape[1]} признаков")
        return feat_mask
    except Exception:
        return select_features_mi(X, y, top_k=min(50, X.shape[1]))


# ════════════════════════════════════════════════════════════════════════════
#  PER-NUTRIENT BUFFERED LOO (Fix #2 в _loo_iter)
# ════════════════════════════════════════════════════════════════════════════

def _loo_iter(i, Xc, yc, dists, buffer_radius, model,
              scale, use_cmixup, cmixup_alpha, min_train,
              pca_components=0):
    
    excluded  = dists[i] < buffer_radius
    train_idx = np.where(~excluded)[0]
    if len(train_idx) < min_train:
        return i, np.nan

    Xtr, ytr = Xc[train_idx].copy(), yc[train_idx].copy()
    Xte       = Xc[[i]].copy()

    # NaN-импутация (медиана train → test)
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        col_med = np.nanmedian(Xtr, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xtr = np.where(np.isnan(Xtr), col_med, Xtr)
        Xte = np.where(np.isnan(Xte), col_med, Xte)

    # C-Mixup (per-target, внутри fold)
    if use_cmixup and len(Xtr) >= 10:
        try:
            Xtr, ytr = c_mixup(Xtr, ytr, alpha=cmixup_alpha, n_aug=len(Xtr))
        except Exception:
            pass

    if scale:
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    Xtr, Xte = _apply_pca_in_fold(Xtr, Xte, pca_components)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    m = clone(model)
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=1)
    try:
        m.fit(Xtr, ytr)
        return i, float(np.ravel(m.predict(Xte))[0])
    except Exception:
        return i, np.nan


def buffered_loo_nested(model, X, y, coords, buffer_radius,
                         scale=True, use_cmixup=True,
                         cmixup_alpha=2.0, min_train=10,
                         tune_fn=None, n_jobs=-1,
                         pca_components=0):
    """Nested buffered-LOO с параллельным внешним циклом."""
    from joblib import Parallel, delayed as jdelayed

    mask_fin = np.isfinite(y)
    Xc, yc, cc = X[mask_fin], y[mask_fin], coords[mask_fin]
    n     = len(yc)
    dists = cdist(cc, cc)

    # Путь с tune_fn: последовательно (lambda не picklable)
    if tune_fn is not None:
        y_pred = np.full(n, np.nan)
        for i in range(n):
            excluded  = dists[i] < buffer_radius
            train_idx = np.where(~excluded)[0]
            if len(train_idx) < min_train:
                continue
            Xtr, ytr = Xc[train_idx].copy(), yc[train_idx].copy()
            Xte       = Xc[[i]].copy()
            if np.isnan(Xtr).any() or np.isnan(Xte).any():
                col_med = np.nanmedian(Xtr, axis=0)
                col_med = np.where(np.isnan(col_med), 0.0, col_med)
                Xtr = np.where(np.isnan(Xtr), col_med, Xtr)
                Xte = np.where(np.isnan(Xte), col_med, Xte)
            try:
                m = tune_fn(Xtr, ytr)
            except Exception:
                m = clone(model) if model is not None else \
                    GradientBoostingRegressor(n_estimators=100, random_state=42)
            if use_cmixup and len(Xtr) >= 10:
                try:
                    Xtr, ytr = c_mixup(Xtr, ytr, alpha=cmixup_alpha, n_aug=len(Xtr))
                except Exception:
                    pass
            if scale:
                sc  = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
                Xte = sc.transform(Xte)
            Xtr, Xte = _apply_pca_in_fold(Xtr, Xte, pca_components)
            Xtr = np.asarray(Xtr, dtype=np.float32)
            Xte = np.asarray(Xte, dtype=np.float32)
            try:
                m.fit(Xtr, ytr)
                y_pred[i] = float(np.ravel(m.predict(Xte))[0])
            except Exception:
                pass
        valid   = np.isfinite(y_pred)
        metrics = regression_metrics(yc[valid], y_pred[valid])
        return {"y_true": yc, "y_pred": y_pred, "metrics": metrics,
                "n_predicted": int(valid.sum())}

    if model is None:
        raise ValueError("model=None при tune_fn=None")

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        jdelayed(_loo_iter)(
            i, Xc, yc, dists, buffer_radius, model,
            scale, use_cmixup, cmixup_alpha, min_train,
            pca_components
        )
        for i in range(n)
    )

    y_pred = np.full(n, np.nan)
    for idx, val in results:
        y_pred[idx] = val

    valid   = np.isfinite(y_pred)
    metrics = regression_metrics(yc[valid], y_pred[valid])
    return {"y_true": yc, "y_pred": y_pred, "metrics": metrics,
            "n_predicted": int(valid.sum())}


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-OUTPUT BUFFERED LOO (Fixes #4, #7, #8, #9 — полностью переписано)
# ════════════════════════════════════════════════════════════════════════════

def _mo_loo_iter(i, Xc, yc, cc_dists, buffer_radius, model, model_kind,
                  scale, use_cmixup, cmixup_alpha, min_train,
                  pca_components, native_nan_frac_max):
    """
    Одна итерация multi-output LOO (module-level для joblib).

    Фундаментальные исправления:
      #7  y-scaling (z-score per column, NaN-aware) ВНУТРИ fold.
          Без этого MO-модель минимизирует дисперсию NO₃, игнорируя
          все остальные нутриенты.
      #4  native MO тренируется только на "чистых" таргетах
          (NaN_frac < native_nan_frac_max). Wrapper тренируется per-output.
      #8  C-Mixup применяется к (X, y_scaled) внутри fold.
      #9  Эта функция теперь вызывается Parallel.
    """
    excluded  = cc_dists < buffer_radius
    train_idx = np.where(~excluded)[0]
    if len(train_idx) < min_train:
        return i, None

    Xtr = Xc[train_idx].copy()
    ytr = yc[train_idx].copy()
    Xte = Xc[[i]].copy()

    # NaN-импутация X
    col_med_X = np.nanmedian(Xtr, axis=0)
    col_med_X = np.where(np.isnan(col_med_X), 0.0, col_med_X)
    Xtr = np.where(np.isnan(Xtr), col_med_X, Xtr)
    Xte = np.where(np.isnan(Xte), col_med_X, Xte)

    # --- Y-scaling (критическое исправление) ---
    y_scaler = MultiOutputYScaler()
    y_scaler.fit(ytr)
    ytr_s = y_scaler.transform(ytr)     # NaN остаются NaN после деления

    # C-Mixup на scaled y (до дальнейшей обработки)
    if use_cmixup and len(Xtr) >= 10:
        try:
            Xtr, ytr_s = c_mixup_mo(Xtr, ytr_s,
                                     alpha=cmixup_alpha,
                                     n_aug=len(Xtr))
        except Exception:
            pass

    if scale:
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    Xtr, Xte = _apply_pca_in_fold(Xtr, Xte, pca_components)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    n_out = ytr.shape[1]
    pred_i_scaled = np.full(n_out, np.nan)

    if model_kind == "native":
        # Native MO: обучаем только на таргетах с NaN_frac < threshold
        nan_frac  = np.mean(np.isnan(ytr_s), axis=0)
        clean_tgt = nan_frac < native_nan_frac_max
        if clean_tgt.sum() == 0:
            return i, pred_i_scaled  # all-NaN

        # Заполняем оставшиеся NaN медианой (на scaled пространстве)
        ytr_fit = ytr_s[:, clean_tgt].copy()
        for j in range(ytr_fit.shape[1]):
            nan_j = np.isnan(ytr_fit[:, j])
            if nan_j.any():
                med = np.nanmedian(ytr_fit[:, j])
                ytr_fit[nan_j, j] = med if np.isfinite(med) else 0.0

        try:
            m = clone(model)
            if hasattr(m, "n_jobs"):
                m.set_params(n_jobs=1)
            m.fit(Xtr, ytr_fit)
            pred_clean = np.ravel(m.predict(Xte))
            if pred_clean.shape[0] == clean_tgt.sum():
                pred_i_scaled[clean_tgt] = pred_clean
            else:
                # PLSR иногда возвращает (1, k)
                pred_reshape = np.asarray(m.predict(Xte)).reshape(-1)
                if pred_reshape.shape[0] == clean_tgt.sum():
                    pred_i_scaled[clean_tgt] = pred_reshape
        except Exception:
            pass

    else:  # wrapper
        inner = model.estimator if hasattr(model, "estimator") else model
        for j in range(n_out):
            valid_j = ~np.isnan(ytr_s[:, j])
            if valid_j.sum() < min_train:
                continue
            try:
                m_j = clone(inner)
                if hasattr(m_j, "n_jobs"):
                    m_j.set_params(n_jobs=1)
                m_j.fit(Xtr[valid_j], ytr_s[valid_j, j])
                pred_i_scaled[j] = float(np.ravel(m_j.predict(Xte))[0])
            except Exception:
                pass

    # Инвертируем scaling → исходные единицы нутриентов
    pred_i = pred_i_scaled * y_scaler.std_ + y_scaler.mean_
    return i, pred_i


def buffered_loo_multioutput(model, model_kind, X, y_mat, coords,
                              buffer_radius, scale=False,
                              use_cmixup=False, cmixup_alpha=2.0,
                              min_train=10, pca_components=0,
                              native_nan_frac_max=0.25,
                              n_jobs=-1):
    """
    Параллельный multi-output buffered-LOO.

    model_kind: 'native' (ET/RF/PLSR/MLP) или 'wrapper' (MultiOutputRegressor).
    native_nan_frac_max: таргеты с NaN_frac выше порога не участвуют в native MO.
    """
    from joblib import Parallel, delayed as jdelayed

    any_valid = np.isfinite(y_mat).any(axis=1)
    Xc = X[any_valid].astype(float)
    yc = y_mat[any_valid].astype(float)
    cc = coords[any_valid]
    n  = len(Xc)
    dists = cdist(cc, cc)

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        jdelayed(_mo_loo_iter)(
            i, Xc, yc, dists[i], buffer_radius, model, model_kind,
            scale, use_cmixup, cmixup_alpha, min_train,
            pca_components, native_nan_frac_max
        )
        for i in range(n)
    )

    y_pred_local = np.full_like(yc, np.nan, dtype=float)
    for idx, pred_vec in results:
        if pred_vec is not None:
            y_pred_local[idx] = pred_vec

    full_pred = np.full(y_mat.shape, np.nan, dtype=float)
    full_pred[any_valid] = y_pred_local
    return full_pred


# ════════════════════════════════════════════════════════════════════════════
#  PERMUTATION TEST для MULTI-OUTPUT (Fix #12)
# ════════════════════════════════════════════════════════════════════════════

def permutation_test_r2_mo(model, model_kind, X, y_mat, coords,
                            buffer_radius, n_perm=99, scale=False,
                            pca_components=0, native_nan_frac_max=0.25,
                            n_jobs=-1, rng_seed=42):
    """
    Permutation test для multi-output модели.
    Метрика: mean R² по всем нутриентам (где достаточно точек).
    Возвращает: p-value, observed_mean_R², perm_mean_R²_array.
    """
    # Observed
    y_pred_obs = buffered_loo_multioutput(
        model, model_kind, X, y_mat, coords, buffer_radius,
        scale=scale, use_cmixup=False,
        pca_components=pca_components,
        native_nan_frac_max=native_nan_frac_max, n_jobs=n_jobs)
    r2_list_obs = []
    for j in range(y_mat.shape[1]):
        yt = y_mat[:, j].astype(float)
        yp = y_pred_obs[:, j]
        vm = np.isfinite(yt) & np.isfinite(yp)
        if vm.sum() >= 5:
            r2_list_obs.append(r2_score(yt[vm], yp[vm]))
    obs_mean = float(np.mean(r2_list_obs)) if r2_list_obs else np.nan

    rng = np.random.default_rng(rng_seed)
    perm_means = []
    for k in range(n_perm):
        # Перемешиваем y ВНУТРИ каждого столбца независимо
        y_perm = y_mat.copy()
        for j in range(y_perm.shape[1]):
            col = y_perm[:, j]
            valid = np.isfinite(col)
            col_v = col[valid].copy()
            rng.shuffle(col_v)
            y_perm[valid, j] = col_v

        y_pred_p = buffered_loo_multioutput(
            model, model_kind, X, y_perm, coords, buffer_radius,
            scale=scale, use_cmixup=False,
            pca_components=pca_components,
            native_nan_frac_max=native_nan_frac_max, n_jobs=n_jobs)
        r2_list_p = []
        for j in range(y_perm.shape[1]):
            yt = y_perm[:, j].astype(float)
            yp = y_pred_p[:, j]
            vm = np.isfinite(yt) & np.isfinite(yp)
            if vm.sum() >= 5:
                r2_list_p.append(r2_score(yt[vm], yp[vm]))
        if r2_list_p:
            perm_means.append(float(np.mean(r2_list_p)))

    if not perm_means or np.isnan(obs_mean):
        return {"p_value": np.nan, "significant": False,
                "obs_mean_R2": obs_mean,
                "perm_mean_R2": np.nan}

    perm_means = np.array(perm_means)
    # p-value: доля permutation-прогонов с R² >= observed
    p_value = float((np.sum(perm_means >= obs_mean) + 1) / (len(perm_means) + 1))
    return {"p_value":      p_value,
            "significant":  p_value < 0.05,
            "obs_mean_R2":  obs_mean,
            "perm_mean_R2": float(np.mean(perm_means))}


# ════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ДАННЫХ (без существенных изменений)
# ════════════════════════════════════════════════════════════════════════════

def load_combined_dataset(cfg, args):
    """Загружает признаки и химию для ОБЕИХ дат, объединяет (~200 точек)."""
    bmap  = cfg["camera"]["band_map"]
    cols  = cfg["chemistry"]["columns"]
    elems = cfg["chemistry"]["target_elements"]
    tiers = cfg["statistics"]["index_tiers"]
    dsmap = cfg["chemistry"]["date_sampling_map"]

    all_X, all_y, all_coords, all_meta = [], [], [], []

    hyper_cache = {}
    if args.hyper:
        print("  Загрузка гиперспектра...")
        try:
            from analysis.hyper_features import load_hyper_for_all_dates
            hyper_cache = load_hyper_for_all_dates(
                cfg, band_step=args.hyper_step)
            if hyper_cache:
                total_h = sum(v.shape[1] for v in hyper_cache.values())
                print(f"  Гиперспектр загружен: {list(hyper_cache.keys())}, "
                      f"~{total_h//len(hyper_cache)} признаков/дата")
            else:
                print("  Гиперспектр: не загружен (нет данных)")
        except Exception as e:
            print(f"  Гиперспектр: ошибка {e}")

    for dk in dsmap.keys():
        pat = cfg["paths"]["multi"].get(dk)
        if not pat or not glob(pat):
            print(f"  [{dk}] нет данных"); continue

        df_bands = load_multi(pat, bmap)
        df_idx   = calculate_indices(df_bands, tiers=tiers)

        sk   = dsmap[dk]
        chem = load_chemistry(cfg["paths"]["chemistry"][sk],
                               cols,
                               cfg["chemistry"]["id_offsets"].get(sk, 0))

        ids   = df_idx.index.intersection(chem.index)
        avail = [e for e in elems if e in chem.columns]

        gpkg_files = sorted(glob(pat))
        gdf = gpd.read_file(gpkg_files[0])
        try:
            gdf_m = gdf.to_crs(epsg=32636)
        except Exception:
            gdf_m = gdf

        gdf_ids = gdf_m["id"].values.astype(int)
        xy_m    = np.array([(g.centroid.x, g.centroid.y)
                             for g in gdf_m.geometry])
        msk_g   = np.isin(gdf_ids, ids)
        id2xy   = dict(zip(gdf_ids[msk_g], xy_m[msk_g]))

        _ortho_src = (args.single_ortho
                      if args.single_ortho and Path(args.single_ortho).is_file()
                      else None)

        if _ortho_src:
            from analysis.pixel_features import extract_features_at_points
            win_size = cfg.get("cv", {}).get("window_size", 7)

            with rasterio.open(_ortho_src) as _src:
                _crs = _src.crs
            _gdf_r  = gdf_m.to_crs(_crs)
            _xy     = np.array([(g.centroid.x, g.centroid.y)
                                  for g in _gdf_r.geometry])
            _gids   = gdf_m["id"].values.astype(int)
            _id_msk = np.isin(_gids, ids)
            _xy_sub = _xy[_id_msk]
            _ids_sub= _gids[_id_msk]

            X_tif, feat_names, valid_mask = extract_features_at_points(
                _ortho_src, _xy_sub, win_size)

            ids_valid = _ids_sub[valid_mask]
            ids       = np.intersect1d(ids_valid, chem.index)
            keep      = np.isin(ids_valid, ids)
            X_base    = X_tif[keep]
            print(f"    Tif-mode [{dk}]: {X_base.shape[1]} признаков "
                  f"(w={win_size}px, n={len(ids)})")

        else:
            X_base     = df_idx.loc[ids].values.astype(float)
            feat_names = list(df_idx.columns)

            _ortho_list = _collect_all_orthos(args.ortho, cfg)
            for ortho_path in _ortho_list:
                stem = Path(ortho_path).stem
                Xw, fw = _extract_window_features(ortho_path, ids, gdf_m)
                if Xw is not None:
                    X_base     = np.hstack([X_base, Xw])
                    feat_names = feat_names + [f"{stem}__{f}" for f in fw]
            if _ortho_list:
                print(f"    gpkg+window [{dk}]: {X_base.shape[1]} признаков")

            if args.textures:
                Xt, ft = _load_all_textures(args.textures, ids)
                if Xt is not None:
                    X_base     = np.hstack([X_base, Xt])
                    feat_names = feat_names + ft
                    print(f"    Textures: +{len(ft)} признаков")

        if args.hyper and dk in hyper_cache:
            hf       = hyper_cache[dk]
            common_h = np.intersect1d(ids, hf.index)
            if len(common_h) > 0:
                aligned_h  = hf.reindex(ids).fillna(hf.median()).fillna(0)
                X_base     = np.hstack([X_base, aligned_h.values])
                feat_names = feat_names + [f"H_{c}" for c in aligned_h.columns]
                print(f"    Hyper [{dk}]: +{aligned_h.shape[1]} признаков")

        y_mat  = chem.loc[ids, avail].values.astype(float)
        coords = np.array([id2xy.get(i, [np.nan, np.nan]) for i in ids])

        for k, pid in enumerate(ids):
            if not np.all(np.isfinite(coords[k])):
                continue
            all_X.append(X_base[k])
            all_y.append(y_mat[k])
            all_coords.append(coords[k])
            all_meta.append({"date": dk, "point_id": int(pid)})

        print(f"  [{dk}] {len(ids)} точек  признаков={X_base.shape[1]}")

    if not all_X:
        return None, None, None, None, None, None

    # Возвращаем ещё и feat_names для сохранения в meta
    return (np.array(all_X), np.array(all_y),
            np.array(all_coords), pd.DataFrame(all_meta), avail,
            feat_names)


def _collect_all_orthos(ortho_arg, cfg):
    result = []
    if ortho_arg:
        p = Path(ortho_arg)
        if p.is_dir():
            result = [str(f) for f in sorted(p.glob("*.tif"))]
        elif p.is_file():
            result = [str(p)]
    if not result:
        map_tif = cfg.get("paths", {}).get("map_tif", {})
        for key, path in map_tif.items():
            if key == "primary" or not path:
                continue
            if Path(path).is_file():
                result.append(path)
    existing = [p for p in result if Path(p).is_file()]
    if existing:
        print(f"    Ортомозаики: {[Path(p).name for p in existing]}")
    return existing


def _load_all_textures(tex_dir, ids):
    p = Path(tex_dir)
    all_csvs = sorted(p.rglob("*texture_features.csv"))
    if not all_csvs:
        return None, None

    all_dfs   = []
    all_names = []
    for csv_path in all_csvs:
        stem = csv_path.stem.replace("_texture_features", "")
        try:
            df     = pd.read_csv(csv_path, index_col=0)
            common = np.intersect1d(ids, df.index)
            if len(common) == 0:
                continue
            aligned = df.loc[common].reindex(ids)
            aligned = aligned.fillna(aligned.median()).fillna(0)
            prefix = stem[:30]
            aligned.columns = [f"{prefix}__{c}" for c in aligned.columns]
            all_dfs.append(aligned.values)
            all_names.extend(list(aligned.columns))
        except Exception as e:
            print(f"    Texture {csv_path.name}: ошибка {e}")
            continue

    if not all_dfs:
        return None, None
    print(f"    Textures: {len(all_csvs)} файлов")
    return np.hstack(all_dfs), all_names


def _extract_window_features(ortho_path, ids, gdf_m, win_size=7):
    if not Path(ortho_path).exists():
        return None, None
    try:
        band_stats = compute_band_stats(ortho_path)
        BANDS      = ("Blue","Green","Red","RedEdge","NIR")
        STATS      = ["mean","std","median","p25","p75","cv"]
        feat_names = [f"{b}_w{win_size}_{s}" for b in BANDS for s in STATS]
        all_feats  = []

        with rasterio.open(ortho_path) as src:
            meta    = src.meta
            gdf_rep = gdf_m.to_crs(meta["crs"])
            xy_rep  = np.array([(g.centroid.x, g.centroid.y)
                                  for g in gdf_rep.geometry])
            rp, cp  = world_to_pixel(xy_rep, meta["transform"])
            H, W    = src.height, src.width
            half    = win_size // 2
            gdf_ids = gdf_m["id"].values.astype(int)

            for pid in ids:
                idx = np.where(gdf_ids == pid)[0]
                if len(idx) == 0:
                    all_feats.append(np.zeros(len(feat_names)))
                    continue
                r, c = int(rp[idx[0]]), int(cp[idx[0]])
                r0   = max(0, min(r - half, H - win_size))
                c0   = max(0, min(c - half, W - win_size))
                win  = riow.Window(c0, r0, win_size, win_size)
                try:
                    raw = src.read(window=win).astype(np.float32)
                except Exception:
                    all_feats.append(np.zeros(len(feat_names)))
                    continue
                row_f = []
                for i, _ in enumerate(BANDS):
                    lo, hi = band_stats[i]
                    band   = np.clip((raw[i]-lo)/(hi-lo+1e-8), 0, 1)
                    vals   = band[raw[i] > 0].flatten()
                    if len(vals) == 0:
                        vals = np.zeros(1)
                    row_f += [np.mean(vals), np.std(vals), np.median(vals),
                               np.percentile(vals,25), np.percentile(vals,75),
                               np.std(vals)/(np.mean(vals)+1e-8)]
                all_feats.append(row_f)

        return np.array(all_feats), feat_names
    except Exception as e:
        import traceback
        print(f"  Ошибка window-признаков: {e}")
        traceback.print_exc()
        return None, None


# ════════════════════════════════════════════════════════════════════════════
#  SPATIAL HOLDOUT (Fix #3: inner CV для model selection)
# ════════════════════════════════════════════════════════════════════════════

def _spatial_holdout(X, y_mat, coords, avail, cfg, models,
                      buffer_map, args, pca_comps,
                      use_cmixup, cmixup_alpha, bca_b, out_dir, dpi):
    
    from sklearn.cluster import KMeans

    holdout_n = args.holdout_n
    print(f"\n{'='*55}")
    print(f"ПРОСТРАНСТВЕННЫЙ HOLDOUT (n={holdout_n} точек)")
    print(f"{'='*55}")

    # Кластеризация по координатам
    n_clusters = max(2, len(coords) // holdout_n)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords)
    labels = km.labels_

    center  = coords.mean(axis=0)
    cl_dists = np.array([
        np.linalg.norm(coords[labels == c].mean(axis=0) - center)
        for c in range(n_clusters)
    ])
    test_cluster  = int(np.argmax(cl_dists))
    test_mask     = labels == test_cluster
    train_mask    = ~test_mask

    n_test  = test_mask.sum()
    n_train = train_mask.sum()
    print(f"  Кластер {test_cluster}: {n_test} тест / {n_train} трейн")
    print(f"  Удалённость кластера от центра: "
          f"{cl_dists[test_cluster]:.0f} м")

    X_train, X_test   = X[train_mask],   X[test_mask]
    y_train, y_test   = y_mat[train_mask], y_mat[test_mask]

    rows = []
    for el_idx, el in enumerate(avail):
        sn   = short_name(cfg, el)
        yt   = y_test[:, el_idx].astype(float)
        ytr_ = y_train[:, el_idx].astype(float)
        valid_test  = np.isfinite(yt)
        valid_train = np.isfinite(ytr_)

        if valid_test.sum() < 3 or valid_train.sum() < 10:
            continue

        # Feature selection на train
        fs_mask = (select_features_mi(X_train[valid_train],
                                       ytr_[valid_train], top_k=50)
                    if args.feature_selection
                    else np.ones(X_train.shape[1], bool))
        Xtr_fs = X_train[:, fs_mask][valid_train]
        Xte_fs = X_test[:,  fs_mask][valid_test]

        # NaN-импутация (медиана train → test)
        col_med = np.nanmedian(Xtr_fs, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xtr_fs  = np.where(np.isnan(Xtr_fs), col_med, Xtr_fs)
        Xte_fs  = np.where(np.isnan(Xte_fs), col_med, Xte_fs)
        ytr_v   = ytr_[valid_train]
        yte_v   = yt[valid_test]

        # ── Inner CV на train для выбора модели (Fix #3) ───────────────────
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_cv_r2, best_model_name = -np.inf, None
        best_needs_scale = False
        for mname, (mobj, needs_scale) in models.items():
            try:
                sc = StandardScaler() if needs_scale else None
                Xs = sc.fit_transform(Xtr_fs) if sc else Xtr_fs.copy()
                if pca_comps:
                    pca = PCA(
                        n_components=min(pca_comps, Xs.shape[1], Xs.shape[0]-1)
                        if isinstance(pca_comps, int) and pca_comps > 0
                        else (0.95 if pca_comps == -1 else pca_comps),
                        random_state=42)
                    Xs_cv = pca.fit_transform(Xs)
                else:
                    Xs_cv = Xs
                scores = cross_val_score(clone(mobj), Xs_cv, ytr_v,
                                          cv=cv_inner, scoring="r2",
                                          n_jobs=-1)
                cv_r2 = float(np.mean(scores))
                if np.isfinite(cv_r2) and cv_r2 > best_cv_r2:
                    best_cv_r2       = cv_r2
                    best_model_name  = mname
                    best_needs_scale = needs_scale
            except Exception:
                continue

        if best_model_name is None:
            continue

        # ── Финальная оценка на holdout (один раз) ──────────────────────────
        sc = StandardScaler() if best_needs_scale else None
        Xtr_final = sc.fit_transform(Xtr_fs) if sc else Xtr_fs.copy()
        Xte_final = sc.transform(Xte_fs)     if sc else Xte_fs.copy()
        Xtr_final, Xte_final = _apply_pca_in_fold(
            Xtr_final, Xte_final, pca_comps)

        best_obj = models[best_model_name][0]
        m_final  = clone(best_obj)
        try:
            m_final.fit(np.asarray(Xtr_final, np.float32), ytr_v)
            yp     = m_final.predict(np.asarray(Xte_final, np.float32))
            r2_ho  = float(r2_score(yte_v, yp))
        except Exception:
            r2_ho = np.nan

        print(f"  {sn:<6}: CV_R²={best_cv_r2:+.4f}  "
              f"holdout_R²={r2_ho:+.4f}  [{best_model_name}]  "
              f"n_test={valid_test.sum()}")
        rows.append({
            "element": el, "short_name": sn,
            "R2_cv_train": best_cv_r2,
            "R2_holdout":  r2_ho,
            "best_model":  best_model_name,
            "n_test":      int(valid_test.sum()),
            "n_train":     int(valid_train.sum()),
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{out_dir}/spatial_holdout_results.csv",
                  index=False, float_format="%.4f")
        print(f"\n  saved: spatial_holdout_results.csv")
        print(f"  Среднее holdout R²: {df['R2_holdout'].mean():+.4f}")

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-OUTPUT ВЫПОЛНЕНИЕ (Fixes #7, #8, #11, #12)
# ════════════════════════════════════════════════════════════════════════════

def _run_multi_output(X, y_mat, coords, avail, cfg,
                       buffer_map, args, use_cmixup, cmixup_alpha,
                       bca_b, out_dir, dpi, pca_comps=0,
                       feat_names=None, global_feat_mask=None):
    """Multi-output: y-scaling, C-Mixup, normalized MI, permutation test."""
    import json

    buf = float(np.median(list(buffer_map.values()))) \
          if buffer_map else GLOBAL_BUFFER_M
    print(f"  Буфер (медиана): {buf:.0f} м")

    # Feature selection — нормализованная MI (Fix #11)
    if args.feature_selection:
        print(f"  Feature selection (normalized MI)...", end=" ", flush=True)
        fs_mask = select_features_mi_mo(X, y_mat, top_k=50)
        Xf      = X[:, fs_mask]
        print(f"{fs_mask.sum()} признаков")
    else:
        fs_mask = np.ones(X.shape[1], dtype=bool)
        Xf      = X

    any_valid = np.isfinite(y_mat).any(axis=1)
    print(f"  Точек с хотя бы одним нутриентом: {any_valid.sum()}")

    mo_models     = get_multioutput_models(n_outputs=len(avail))
    best_mean_r2  = -np.inf
    best_mo_name  = None
    best_mo_kind  = None
    best_y_pred   = None
    all_model_r2  = {}

    for mo_name, (mo_model, mo_kind) in mo_models.items():
        print(f"\n  [{mo_name}] ({mo_kind}) buffered-LOO (buf={buf:.0f}м)...",
              flush=True)
        try:
            # Линейные модели PLSR не требуют внешнего scale
            needs_scale = (mo_kind == "wrapper" and mo_name in ("MO_SVR",)) \
                          or (mo_kind == "native" and mo_name == "MLP")
            y_pred_mat = buffered_loo_multioutput(
                mo_model, mo_kind, Xf, y_mat, coords,
                buffer_radius=buf,
                scale=needs_scale,
                use_cmixup=use_cmixup,
                cmixup_alpha=cmixup_alpha,
                pca_components=pca_comps,
            )
            r2s    = []
            r2_per = {}
            for j, el_j in enumerate(avail):
                yt = y_mat[:, j].astype(float)
                yp = y_pred_mat[:, j]
                vm = np.isfinite(yt) & np.isfinite(yp)
                if vm.sum() >= 5:
                    r2_j = float(r2_score(yt[vm], yp[vm]))
                    r2s.append(r2_j)
                    r2_per[short_name(cfg, el_j)] = r2_j

            mean_r2 = float(np.mean(r2s)) if r2s else -np.inf
            per_str = "  ".join(f"{sn}={r2:+.3f}" for sn, r2 in r2_per.items())
            print(f"mean R²={mean_r2:+.4f}")
            print(f"    {per_str}")

            if mean_r2 > best_mean_r2:
                best_mean_r2 = mean_r2
                best_mo_name = mo_name
                best_mo_kind = mo_kind
                best_y_pred  = y_pred_mat
            all_model_r2[mo_name] = r2_per
        except Exception as e:
            import traceback
            print(f"ошибка: {e}")
            traceback.print_exc()
            continue

    if best_y_pred is None:
        print("Нет результатов"); return []

    print(f"\n  ★ Лучшая multi-output модель: {best_mo_name}  "
          f"mean R²={best_mean_r2:+.4f}")

    # ── Permutation test для лучшей модели (Fix #12) ────────────────────────
    perm_result = {"p_value": np.nan, "significant": np.nan,
                   "perm_mean_R2": np.nan}
    if not args.skip_perm:
        print(f"\n  Permutation test ({args.n_perm} перест.)...",
              end=" ", flush=True)
        best_mo_obj = mo_models[best_mo_name][0]
        perm_result = permutation_test_r2_mo(
            best_mo_obj, best_mo_kind, Xf, y_mat, coords,
            buffer_radius=buf, n_perm=args.n_perm,
            pca_components=pca_comps,
        )
        sig = "✓" if perm_result["significant"] else "✗"
        print(f"p={perm_result['p_value']:.3f} {sig}  "
              f"(obs={perm_result['obs_mean_R2']:+.4f} / "
              f"perm_mean={perm_result['perm_mean_R2']:+.4f})")

    # ── Метрики + BCa CI для каждого нутриента ───────────────────────────────
    all_results = []
    for el_idx, el in enumerate(avail):
        sn      = short_name(cfg, el)
        y_true  = y_mat[:, el_idx].astype(float)
        y_pred  = best_y_pred[:, el_idx]
        valid   = np.isfinite(y_true) & np.isfinite(y_pred)

        if valid.sum() < 5:
            print(f"  {sn:<6}: недостаточно точек")
            continue

        metrics = regression_metrics(y_true[valid], y_pred[valid])
        r2_pt, ci_lo, ci_hi = bca_bootstrap_r2(
            y_true[valid], y_pred[valid], B=bca_b)

        print(f"  {sn:<6}: R²={r2_pt:+.4f}  "
              f"CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"RMSE={metrics['RMSE']:.4f}  n={valid.sum()}")

        _plot_scatter(y_true, y_pred, sn, best_mo_name,
                      r2_pt, ci_lo, ci_hi, out_dir, dpi)

        all_results.append({
            "element":     el,
            "short_name":  sn,
            "n_valid":     int(np.isfinite(y_true).sum()),
            "buffer_m":    buf,
            "best_model":  best_mo_name,
            "R2":          r2_pt,
            "R2_ci_lo":    ci_lo,
            "R2_ci_hi":    ci_hi,
            "RMSE":        metrics["RMSE"],
            "MAE":         metrics["MAE"],
            "RPD":         metrics["RPD"],
            "p_perm":      perm_result["p_value"],
            "significant": perm_result.get("significant", np.nan),
            "n_predicted": int(valid.sum()),
        })

    # ── Сохраняем финальную модель + meta (Fix #10) ──────────────────────────
    import joblib
    models_dir = Path(out_dir) / "models"
    models_dir.mkdir(exist_ok=True)

    print("\n  Обучение финальной модели на всех данных...")
    col_med = np.nanmedian(Xf, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    Xfit    = np.where(np.isnan(Xf), col_med, Xf).astype(np.float32)

    # y-scaling для финальной модели тоже обязателен
    y_scaler = MultiOutputYScaler()
    y_fit_s  = y_scaler.fit_transform(y_mat.astype(float))
    # NaN в y → медиана столбца (уже на scaled пространстве = 0)
    y_fit_s  = np.where(np.isnan(y_fit_s), 0.0, y_fit_s)

    row_mask = ~np.isnan(Xfit).any(axis=1)

    _best_mo = mo_models.get(best_mo_name)
    if _best_mo:
        final_model = clone(_best_mo[0])
    else:
        final_model = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if hasattr(final_model, "n_jobs"):
        final_model.set_params(n_jobs=-1)

    try:
        final_model.fit(Xfit[row_mask], y_fit_s[row_mask])
        joblib.dump(final_model, models_dir / "model_multioutput.pkl")
        joblib.dump(y_scaler,    models_dir / "y_scaler.pkl")
        print(f"  Сохранено: model_multioutput.pkl + y_scaler.pkl")
    except Exception as e:
        print(f"  Ошибка финальной модели: {e}")

    # Fix #10: сохраняем global_feat_mask для корректного pixel-inference
    meta = {
        "mode":              "multi_output",
        "model":             best_mo_name,
        "model_kind":        best_mo_kind,
        "nutrients":         avail,
        "short_names":       [short_name(cfg, e) for e in avail],
        "fs_mask_indices":   np.where(fs_mask)[0].tolist(),  # в X_fs space
        "global_feat_mask":  (global_feat_mask.tolist()
                               if global_feat_mask is not None else None),
        "feature_names":     feat_names if feat_names else None,
        "col_median":        col_med.tolist(),
        "y_mean":            y_scaler.mean_.tolist(),
        "y_std":             y_scaler.std_.tolist(),
        "n_features":        int(fs_mask.sum()),
        "buffer_m":          buf,
        "permutation_p":     perm_result["p_value"],
        "permutation_significant": bool(perm_result["significant"])
                                    if isinstance(perm_result["significant"], (bool, np.bool_))
                                    else False,
        "R2_per_nutrient": {
            r["short_name"]: round(r["R2"], 4)
            for r in all_results
        }
    }
    with open(models_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: meta.json ({len(avail)} нутриентов)")

    # ── Визуализация ──────────────────────────────────────────────────────────
    print("\n  Построение графиков...")
    _plot_multioutput_results(
        all_results=all_results,
        all_model_r2=all_model_r2,
        best_mo_name=best_mo_name,
        y_mat=y_mat,
        y_pred_mat=best_y_pred,
        avail=avail,
        cfg=cfg,
        out_dir=out_dir,
        dpi=dpi,
    )

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  ПЛОТТИНГ (упрощённый, ключевые графики сохранены)
# ════════════════════════════════════════════════════════════════════════════

def _plot_scatter(y_true, y_pred, sn, model_name, r2, ci_lo, ci_hi,
                  out_dir, dpi):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return
    yt, yp = y_true[valid], y_pred[valid]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=22, alpha=0.7, color="#2E75B6",
               edgecolors="white", lw=0.3)
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, alpha=0.7, label="1:1")
    z = np.polyfit(yt, yp, 1)
    xf = np.linspace(mn, mx, 50)
    ax.plot(xf, np.poly1d(z)(xf), "g-", lw=1, alpha=0.6, label="trend")

    ax.set_xlabel(f"Измерено ({sn})", fontsize=10)
    ax.set_ylabel(f"Предсказано ({sn})", fontsize=10)
    ax.set_title(f"{sn}  [{model_name}]\n"
                 f"R²={r2:+.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    safe_sn = sn.replace("/", "_").replace(" ", "_")
    fig.savefig(Path(out_dir)/f"scatter_{safe_sn}.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _print_summary(df_res):
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ — честная пространственная CV")
    print("="*70)
    print(f"\n{'Нутриент':<7} {'Модель':<13} {'R²':>6} "
          f"{'CI_lo':>7} {'CI_hi':>7} {'RMSE':>7} "
          f"{'RPD':>5} {'p_perm':>7} {'Зн.':>4}")
    print("-"*68)
    for _, r in df_res.iterrows():
        sig = "+" if r.get("significant") else "-"
        p   = f"{r['p_perm']:.3f}" if pd.notna(r.get("p_perm")) else "  n/a"
        print(f"{r['short_name']:<7} {r['best_model']:<13} "
              f"{r['R2']:>+6.3f} {r['R2_ci_lo']:>7.3f} {r['R2_ci_hi']:>7.3f} "
              f"{r['RMSE']:>7.4f} {r['RPD']:>5.2f} {p:>7} {sig:>4}")
    pos = df_res[df_res["R2"] > 0]
    if len(pos):
        print(f"\nR² > 0: {len(pos)}/{len(df_res)}  "
              f"(средний R² среди положительных: {pos['R2'].mean():+.3f})")


def _plot_r2_ranking(df_res, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_s = df_res.sort_values("R2", ascending=True)
    colors = ["#2E75B6" if r > 0 else "#C00000" for r in df_s["R2"]]
    ax.barh(range(len(df_s)), df_s["R2"].values, color=colors, alpha=0.85)
    for j, (_, row) in enumerate(df_s.iterrows()):
        if pd.notna(row["R2_ci_lo"]) and pd.notna(row["R2_ci_hi"]):
            ax.plot([row["R2_ci_lo"], row["R2_ci_hi"]], [j, j],
                    "k-", lw=2.5, alpha=0.6)
            ax.plot([row["R2_ci_lo"], row["R2_ci_hi"]], [j, j],
                    "|k", ms=9, mew=2)
    ax.set_yticks(range(len(df_s)))
    ax.set_yticklabels(df_s["short_name"].values, fontsize=9)
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("R² (buffered-LOO)", fontsize=10)
    ax.set_title("Честные R² по нутриентам + 95% BCa CI\n"
                 "(синий = R²>0, красный = модель хуже среднего)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(out_dir)/"r2_ranking.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: r2_ranking.png")


def _plot_multioutput_results(all_results, all_model_r2, best_mo_name,
                               y_mat, y_pred_mat, avail, cfg,
                               out_dir, dpi=150):
    """Визуализация multi-output: heatmap, ranking, scatter grid, residuals."""
    import seaborn as sns
    out = Path(out_dir)
    short_names = [short_name(cfg, e) for e in avail]

    # 1. Heatmap моделей × нутриентов
    if len(all_model_r2) > 1:
        df_heat = pd.DataFrame(all_model_r2).T
        fig, ax = plt.subplots(figsize=(max(10, len(avail)*0.9),
                                         max(4, len(all_model_r2)*0.7)))
        mask = df_heat.isna()
        sns.heatmap(df_heat.astype(float), ax=ax,
                    cmap="RdYlGn", center=0,
                    vmin=-0.3, vmax=0.6,
                    annot=True, fmt=".2f", annot_kws={"size": 8},
                    linewidths=0.4, linecolor="white",
                    mask=mask,
                    cbar_kws={"shrink": 0.6, "label": "R²"})
        if best_mo_name in all_model_r2:
            best_idx = list(all_model_r2.keys()).index(best_mo_name)
            ax.add_patch(plt.Rectangle((0, best_idx), len(avail), 1,
                                        fill=False, edgecolor="gold",
                                        lw=3, clip_on=False))
        ax.set_title("Сравнение multi-output моделей: R² (buffered-LOO)\n"
                     "(золотая рамка = лучшая модель)", fontsize=11)
        ax.set_xlabel("Нутриент", fontsize=10)
        ax.set_ylabel("Модель", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        fig.savefig(out/"mo_model_comparison_heatmap.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved: mo_model_comparison_heatmap.png")

    # 2. Ranking bar chart с CI
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values("R2", ascending=True)
        colors = ["#C00000" if r < 0 else
                  "#F4A460" if r < 0.2 else
                  "#2E75B6" if r < 0.4 else "#1A4D8C"
                  for r in df_res["R2"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(df_res)), df_res["R2"].values,
                color=colors, alpha=0.85, height=0.6)
        for j, (_, row) in enumerate(df_res.iterrows()):
            if pd.notna(row.get("R2_ci_lo")) and pd.notna(row.get("R2_ci_hi")):
                ax.plot([row["R2_ci_lo"], row["R2_ci_hi"]], [j, j],
                        "k-", lw=2.5, alpha=0.7)
                ax.plot([row["R2_ci_lo"], row["R2_ci_hi"]], [j, j],
                        "|k", ms=8, mew=2)
        ax.set_yticks(range(len(df_res)))
        ax.set_yticklabels(df_res["short_name"].values, fontsize=10)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.axvline(0.2, color="orange", lw=0.8, ls=":", alpha=0.5,
                   label="R²=0.2 (удовл.)")
        ax.axvline(0.4, color="green", lw=0.8, ls=":", alpha=0.5,
                   label="R²=0.4 (хорошо)")
        ax.set_xlabel(f"R² — buffered-LOO  [{best_mo_name}]", fontsize=10)
        ax.set_title(f"Предсказуемость нутриентов: {best_mo_name}\n"
                     "95% BCa CI | синий ≥0.4 | оранжевый 0.2–0.4 | красный <0",
                     fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out/"mo_r2_ranking.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved: mo_r2_ranking.png")

    # 3. Scatter 4x3
    nc = 4
    nr = int(np.ceil(len(avail) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 3.5*nr))
    axes = axes.flatten()
    for i, (el, sn) in enumerate(zip(avail, short_names)):
        ax    = axes[i]
        y_t   = y_mat[:, i].astype(float)
        y_p   = y_pred_mat[:, i] if y_pred_mat is not None else np.full_like(y_t, np.nan)
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        if valid.sum() < 5:
            ax.set_visible(False); continue
        yt, yp = y_t[valid], y_p[valid]
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, s=18, alpha=0.65, color="#2E75B6",
                   edgecolors="white", lw=0.3)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, alpha=0.7)
        z = np.polyfit(yt, yp, 1)
        xf = np.linspace(mn, mx, 50)
        ax.plot(xf, np.poly1d(z)(xf), "g-", lw=1, alpha=0.6)
        rmse = float(np.sqrt(np.mean((yt - yp)**2)))
        col = "#1A4D8C" if r2 >= 0.4 else \
              "#2E75B6" if r2 >= 0.2 else \
              "#C00000" if r2 < 0 else "#F4A460"
        ax.set_title(f"{sn}\nR²={r2:+.3f}  RMSE={rmse:.3f}",
                     fontsize=8, color=col)
        ax.set_xlabel("Измерено", fontsize=7)
        ax.set_ylabel("Предсказано", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Predicted vs Observed — {best_mo_name} (buffered-LOO)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out/"mo_scatter_all.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: mo_scatter_all.png")


def _compare_random_vs_buffered(X, y_mat, coords, avail, models,
                                  buffer_map, cfg, out_dir, dpi):
    """Random KFold vs buffered-LOO: ключевой методологический график."""
    print("\nСравнение: random KFold vs buffered-LOO...")
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                     learning_rate=0.1, random_state=42)
    rows = []
    for el_idx, el in enumerate(avail):
        y   = y_mat[:, el_idx].astype(float)
        sn  = short_name(cfg, el)
        buf = buffer_map.get(el, GLOBAL_BUFFER_M)
        if np.isfinite(y).sum() < 15:
            continue

        res_buf = buffered_loo_nested(
            gbr, X, y, coords, buf, scale=False, use_cmixup=False)
        r2_buf  = res_buf["metrics"]["R2"]

        mask   = np.isfinite(y)
        cv     = KFold(n_splits=5, shuffle=True, random_state=42)
        # Простой NaN-impute для random CV
        Xm = X[mask]
        col_med = np.nanmedian(Xm, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xm = np.where(np.isnan(Xm), col_med, Xm)
        yp = cross_val_predict(clone(gbr), Xm, y[mask], cv=cv)
        r2_rnd = r2_score(y[mask], yp)

        inflation = r2_rnd - r2_buf
        rows.append({"nutrient": sn, "R2_random": r2_rnd,
                      "R2_buffered": r2_buf, "inflation": inflation})
        print(f"  {sn:<6}: rnd={r2_rnd:+.3f}  "
              f"buf={r2_buf:+.3f}  Δ={inflation:+.3f}")

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/comparison_random_vs_buffered.csv",
              index=False, float_format="%.4f")
    print(f"\n  Среднее завышение random CV: {df['inflation'].mean():+.3f}")

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-0.2, df["R2_random"],   0.35, label="Random 5-fold CV",
           color="#F4A460", alpha=0.85)
    ax.bar(x+0.2, df["R2_buffered"], 0.35, label="Buffered-LOO (честный)",
           color="#2E75B6", alpha=0.85)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["nutrient"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("Оптимистическое смещение random CV vs buffered-LOO\n"
                 "(разница = завышение из-за пространственной автокорреляции)",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/random_vs_buffered.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: random_vs_buffered.png")


# ════════════════════════════════════════════════════════════════════════════
#  FINAL PER-NUTRIENT MODELS (Fix #10)
# ════════════════════════════════════════════════════════════════════════════

def _save_final_models(X, y_mat, avail, df_res, models,
                        out_dir, elems, args,
                        feat_names=None, global_feat_mask=None):
    """
    Переобучает лучшую per-nutrient модель на всех данных.
    """
    import joblib, json
    models_dir = Path(out_dir) / "models"
    models_dir.mkdir(exist_ok=True)

    meta = {"mode": "per_nutrient", "nutrients": {}}
    # Глобальные метаданные — один раз
    meta["global_feat_mask"] = (global_feat_mask.tolist()
                                  if global_feat_mask is not None else None)
    meta["feature_names"]     = feat_names if feat_names else None

    for _, row in df_res.iterrows():
        el      = row["element"]
        sn      = row["short_name"]
        el_idx  = list(avail).index(el) if el in avail else None
        if el_idx is None:
            continue

        y   = y_mat[:, el_idx].astype(float)
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue

        feat_mask = np.nanstd(X, axis=0) > 1e-8
        Xc = X[:, feat_mask]

        if args.feature_selection:
            fs_mask   = select_features_mi(Xc, y, top_k=50)
            Xc        = Xc[:, fs_mask]
            feat_indices = np.where(feat_mask)[0][fs_mask]
        else:
            feat_indices = np.where(feat_mask)[0]

        col_med = np.nanmedian(Xc[mask], axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        Xfit    = np.where(np.isnan(Xc[mask]), col_med, Xc[mask])
        yfit    = y[mask]

        best_name = row["best_model"].replace("_Optuna", "")
        model_entry = models.get(best_name)
        if model_entry is None:
            model_entry = list(models.values())[4]
        model_obj, needs_scale = model_entry

        scaler = None
        if needs_scale:
            scaler = StandardScaler()
            Xfit   = scaler.fit_transform(Xfit)

        m = clone(model_obj)
        if hasattr(m, "n_jobs"):
            m.set_params(n_jobs=-1)
        Xfit = np.asarray(Xfit, dtype=np.float32)
        m.fit(Xfit, yfit)

        safe_el = el.replace("%","pct").replace(" ","_").replace("/","_")
        joblib.dump(m, models_dir / f"model_{safe_el}.pkl")
        if scaler is not None:
            joblib.dump(scaler, models_dir / f"scaler_{safe_el}.pkl")

        meta["nutrients"][el] = {
            "short_name":    sn,
            "safe_name":     safe_el,
            "best_model":    best_name,
            "needs_scale":   needs_scale,
            "R2_cv":         float(row["R2"]),
            "R2_ci":         [float(row["R2_ci_lo"]), float(row["R2_ci_hi"])],
            "n_train":       int(mask.sum()),
            "feat_indices":  feat_indices.tolist(),
            "col_median":    col_med.tolist(),
        }
        print(f"  Сохранено: {sn}  ({best_name}, "
              f"n={mask.sum()}, R²={row['R2']:+.3f})")

    with open(models_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n  Модели сохранены → {models_dir}/")
    print(f"  meta.json: {len(meta['nutrients'])} нутриентов")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def run(args):
    cfg = load_config(args.config)
    out = args.output or cfg.get("paths", {}).get("output", {}).get(
        "honest_ml", "results/02_honest")
    os.makedirs(out, exist_ok=True)
    dpi = cfg["plots"]["dpi"]

    cv_cfg = cfg.get("cv", {})
    elems = cfg["chemistry"]["target_elements"]

    # Buffer map: CLI > config.cv.buffer_per_nutrient > csv > hardcoded
    if args.buffer:
        buffer_map = {e: args.buffer for e in elems}
        print(f"  Буфер (CLI): единый {args.buffer} м")
    elif cv_cfg.get("buffer_per_nutrient"):
        buffer_map = cv_cfg["buffer_per_nutrient"]
        print(f"  Буферы из config.yaml (cv.buffer_per_nutrient)")
    else:
        csv_path = args.buffer_csv or cv_cfg.get(
            "buffer_csv", "results/01e_variograms/recommended_buffers.csv")
        buffer_map = load_buffer_map(csv_path)

    bca_b         = args.bca_b         or cv_cfg.get("bca_b",         2000)
    n_perm        = args.n_perm        or cv_cfg.get("n_perm",         199)
    optuna_trials = args.optuna_trials or cv_cfg.get("optuna_trials",   40)
    cmixup_alpha  = cv_cfg.get("cmixup_alpha", 2.0)
    use_cmixup    = args.cmixup

    ortho = args.ortho
    if not ortho:
        primary_key = cv_cfg.get("primary_ortho_key", "")
        ortho = cfg.get("paths", {}).get("map_tif", {}).get(primary_key) or \
                cfg.get("paths", {}).get("map_tif", {}).get("primary")
        if ortho:
            print(f"  Ортомозаика из config: {Path(ortho).name}")

    
    if args.pca is None:
        pca_comps = 0
    elif isinstance(args.pca, bool):   # True from const
        pca_comps = -1
    else:
        pca_comps = args.pca   # int или float — оба валидны для _apply_pca_in_fold
    if args.pca_components is not None:
        pca_comps = args.pca_components

    args.bca_b         = bca_b
    args.n_perm        = n_perm
    args.optuna_trials = optuna_trials
    args.ortho         = ortho

    pca_str = "выкл." if pca_comps == 0 else (
        f"{pca_comps*100:.0f}% дисперсии" if isinstance(pca_comps, float) and 0 < pca_comps < 1
        else f"{pca_comps} компонент" if isinstance(pca_comps, int) and pca_comps > 0
        else "95% дисперсии (legacy -1)"
    )
    print(f"\n  Параметры CV: buf=per-nutrient  bca_b={bca_b}  "
          f"n_perm={n_perm}  cmixup={use_cmixup}  "
          f"cmixup_alpha={cmixup_alpha}  PCA={pca_str}")

    print("\n" + "="*65)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*65)
    X, y_mat, coords, meta, avail, feat_names = load_combined_dataset(cfg, args)
    if X is None:
        print("Нет данных"); return

    print(f"\n  Итого точек:  {len(X)}  "
          f"({meta['date'].value_counts().to_dict()})")
    print(f"  Признаков:   {X.shape[1]}")
    print(f"  Нутриентов:  {len(avail)}")

    # Удаляем признаки с нулевой дисперсией (Fix #10: сохраняем маску!)
    global_feat_mask = np.nanstd(X, axis=0) > 1e-8
    X = X[:, global_feat_mask]
    if feat_names:
        feat_names = [f for f, m in zip(feat_names, global_feat_mask) if m]
    print(f"  Признаков (после удаления константных): {X.shape[1]}")

    models = get_base_models()
    all_results     = []
    all_results_raw = {}

    print("\n" + "="*65)
    print("NESTED BUFFERED-LOO CV + C-MIXUP + BCa + PERMUTATION TEST")
    print("="*65)

    # MULTI-OUTPUT MODE
    if args.multi_output:
        print("\n[MULTI-OUTPUT MODE] Одна модель для всех нутриентов\n")
        all_results = _run_multi_output(
            X, y_mat, coords, avail, cfg,
            buffer_map, args, use_cmixup, cmixup_alpha,
            bca_b, out, dpi, pca_comps=pca_comps,
            feat_names=feat_names,
            global_feat_mask=global_feat_mask)
        if all_results:
            df_res = pd.DataFrame(all_results).sort_values("R2", ascending=False)
            df_res.to_csv(f"{out}/honest_metrics.csv",
                          index=False, float_format="%.4f")
            _print_summary(df_res)
            _plot_r2_ranking(df_res, out, dpi)
        if args.spatial_holdout:
            _spatial_holdout(X, y_mat, coords, avail, cfg, models,
                              buffer_map, args, pca_comps,
                              use_cmixup, cmixup_alpha, bca_b, out, dpi)
        print(f"\ndone → {out}/")
        return

    # PER-NUTRIENT MODE
    for el_idx, el in enumerate(avail):
        y   = y_mat[:, el_idx].astype(float)
        sn  = short_name(cfg, el)
        buf = buffer_map.get(el, GLOBAL_BUFFER_M)
        nv  = np.isfinite(y).sum()

        print(f"\n── {sn}  (n={nv}, buf={buf}м) " + "─"*30)
        if nv < 15:
            print(f"  Пропускаем: мало точек"); continue

        if args.feature_selection:
            print(f"  Feature selection...", end=" ", flush=True)
            fs_mask = select_features_boruta(X, y) if args.boruta \
                      else select_features_mi(X, y, top_k=50)
            Xf = X[:, fs_mask]
            print(f"{fs_mask.sum()} признаков выбрано")
        else:
            Xf = X

        best_r2, best_name, best_res = -np.inf, None, None

        for mname, (model, needs_scale) in models.items():
            res = buffered_loo_nested(
                model=model, X=Xf, y=y, coords=coords,
                buffer_radius=buf, scale=needs_scale,
                use_cmixup=use_cmixup, cmixup_alpha=cmixup_alpha,
                pca_components=pca_comps,
            )
            r2 = res["metrics"]["R2"]
            if np.isnan(r2):
                continue
            print(f"  {mname:<13}: R²={r2:+.4f}  "
                  f"RMSE={res['metrics']['RMSE']:.4f}  "
                  f"n={res['n_predicted']}")
            if r2 > best_r2:
                best_r2, best_name, best_res = r2, mname, res

        if args.optuna and HAS_OPTUNA:
            print(f"  Optuna XGB ({args.optuna_trials} trials)...",
                  end=" ", flush=True)
            res_xgb = buffered_loo_nested(
                model=None, X=Xf, y=y, coords=coords,
                buffer_radius=buf, scale=True,
                use_cmixup=use_cmixup,
                tune_fn=lambda Xtr, ytr: tune_xgb_optuna(
                    Xtr, ytr, n_trials=args.optuna_trials),
            )
            r2_xgb = res_xgb["metrics"]["R2"]
            print(f"R²={r2_xgb:+.4f}")
            if np.isfinite(r2_xgb) and r2_xgb > best_r2:
                best_r2, best_name, best_res = r2_xgb, "XGB_Optuna", res_xgb

            if HAS_LGB:
                print(f"  Optuna LGB ({args.optuna_trials} trials)...",
                      end=" ", flush=True)
                res_lgb = buffered_loo_nested(
                    model=None, X=Xf, y=y, coords=coords,
                    buffer_radius=buf, scale=True,
                    use_cmixup=use_cmixup,
                    tune_fn=lambda Xtr, ytr: tune_lgb_optuna(
                        Xtr, ytr, n_trials=args.optuna_trials),
                )
                r2_lgb = res_lgb["metrics"]["R2"]
                print(f"R²={r2_lgb:+.4f}")
                if np.isfinite(r2_lgb) and r2_lgb > best_r2:
                    best_r2, best_name, best_res = r2_lgb, "LGB_Optuna", res_lgb

        if args.stacking:
            print(f"  Stacking...", end=" ", flush=True)
            stack = build_stacking(Xf, y)
            res_st = buffered_loo_nested(
                model=stack, X=Xf, y=y, coords=coords,
                buffer_radius=buf, scale=False,
                use_cmixup=False,
            )
            r2_st = res_st["metrics"]["R2"]
            print(f"R²={r2_st:+.4f}")
            if np.isfinite(r2_st) and r2_st > best_r2:
                best_r2, best_name, best_res = r2_st, "Stacking", res_st

        if best_res is None:
            print(f"  Нет результатов для {sn}"); continue

        print(f"\n  ★ Лучшая модель: {best_name}  R²={best_r2:+.4f}")

        valid = np.isfinite(best_res["y_pred"])
        r2_pt, ci_lo, ci_hi = bca_bootstrap_r2(
            best_res["y_true"][valid], best_res["y_pred"][valid],
            B=args.bca_b)
        print(f"  BCa 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

        if not args.skip_perm:
            print(f"  Permutation test ({args.n_perm} перест.)...",
                  end=" ", flush=True)
            _bm_entry = models.get(best_name.replace("_Optuna", ""))
            bm_obj = _bm_entry[0] if _bm_entry is not None else None
            if bm_obj is None:
                for _fallback in ("GBR", "XGB", "RF", "ET"):
                    if _fallback in models:
                        bm_obj = models[_fallback][0]
                        break
            perm = permutation_test_r2(
                model=bm_obj, X=Xf, y=y, coords=coords,
                buffer_radius=buf, n_perm=args.n_perm,
            )
            sig = "+" if perm["significant"] else "-"
            print(f"p={perm['p_value']:.3f} {sig}")
        else:
            perm = {"p_value": np.nan, "significant": np.nan,
                    "perm_mean_R2": np.nan}

        if args.sweep:
            bm_sweep = list(models.values())[5][0]   # GBR
            sw = sweep_buffer_radii(bm_sweep, Xf, y, coords,
                                     radii=[0, 20, 30, 50, 75, 100])
            sw.to_csv(f"{out}/sweep_{sn}.csv", index=False)
            print(f"  Sweep: {sw[['buffer_m','R2']].to_string(index=False)}")

        _plot_scatter(best_res["y_true"], best_res["y_pred"],
                      sn, best_name, r2_pt, ci_lo, ci_hi, out, dpi)

        all_results_raw[sn] = {
            "y_true": best_res["y_true"],
            "y_pred": best_res["y_pred"],
            "best_model": best_name,
        }
        all_results.append({
            "element":     el,
            "short_name":  sn,
            "n_valid":     nv,
            "buffer_m":    buf,
            "best_model":  best_name,
            "R2":          r2_pt,
            "R2_ci_lo":    ci_lo,
            "R2_ci_hi":    ci_hi,
            "RMSE":        best_res["metrics"]["RMSE"],
            "MAE":         best_res["metrics"]["MAE"],
            "RPD":         best_res["metrics"]["RPD"],
            "p_perm":      perm["p_value"],
            "significant": perm.get("significant", np.nan),
            "perm_mean_R2":perm.get("perm_mean_R2", np.nan),
            "n_predicted": best_res["n_predicted"],
        })

    if not all_results:
        print("\nНет результатов"); return

    df_res = pd.DataFrame(all_results).sort_values("R2", ascending=False)
    df_res.to_csv(f"{out}/honest_metrics.csv", index=False, float_format="%.4f")

    _print_summary(df_res)
    _plot_r2_ranking(df_res, out, dpi)
    _compare_random_vs_buffered(X, y_mat, coords, avail, models,
                                 buffer_map, cfg, out, dpi)

    print("\nОбучение финальных моделей на всех данных...")
    _save_final_models(X, y_mat, avail, df_res, models,
                        out, elems, args,
                        feat_names=feat_names,
                        global_feat_mask=global_feat_mask)

    # Spatial holdout (опционально)
    if args.spatial_holdout:
        _spatial_holdout(X, y_mat, coords, avail, cfg, models,
                          buffer_map, args, pca_comps,
                          use_cmixup, cmixup_alpha, bca_b, out, dpi)

    print(f"\ndone → {out}/")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _parse_pca(value):
    """Парсит --pca: может быть bool-флагом ('--pca'), float (0.95) или int (10)."""
    if value is None:
        return None
    try:
        v = float(value)
        if 0 < v < 1:
            return v
        return int(v)
    except ValueError:
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Этап 3: честный ML с buffered-LOO, nested CV, C-Mixup")
    ap.add_argument("--config",      default="config.yaml")
    ap.add_argument("--output",      default=None)
    ap.add_argument("--ortho",       default=None,
                    help="путь к ортомозаике или папке (window w7)")
    ap.add_argument("--textures",    default=None,
                    help="папка с результатами 01d_textures")
    ap.add_argument("--buffer",      type=float, default=None)
    ap.add_argument("--buffer-csv",  default="results/01e_variograms/recommended_buffers.csv")
    ap.add_argument("--cmixup",      dest="cmixup", action="store_true",  default=True)
    ap.add_argument("--no-cmixup",   dest="cmixup", action="store_false")
    ap.add_argument("--multi-output",  action="store_true", default=False)
    # PCA теперь принимает значение: 0.95 (% дисп.) или 10 (компонент)
    ap.add_argument("--pca",           type=_parse_pca, default=None, nargs="?",
                    const=True,
                    help="PCA: float 0<p<1 (%% дисперсии) или int (число компонент). "
                         "Без аргумента = 95%% дисперсии")
    ap.add_argument("--pca-components",type=int, default=None)
    ap.add_argument("--spatial-holdout",action="store_true", default=False)
    ap.add_argument("--holdout-n",     type=int, default=20)
    ap.add_argument("--single-ortho",  default=None)
    ap.add_argument("--hyper",           action="store_true", default=False)
    ap.add_argument("--hyper-step",      type=int, default=5)
    ap.add_argument("--feature-selection", action="store_true", default=False)
    ap.add_argument("--boruta",      action="store_true", default=False)
    ap.add_argument("--optuna",      action="store_true", default=False)
    ap.add_argument("--optuna-trials", type=int, default=None)
    ap.add_argument("--stacking",    action="store_true", default=False)
    ap.add_argument("--n-perm",      type=int, default=None)
    ap.add_argument("--skip-perm",   action="store_true")
    ap.add_argument("--bca-b",       type=int, default=None)
    ap.add_argument("--sweep",       action="store_true")
    run(ap.parse_args())
