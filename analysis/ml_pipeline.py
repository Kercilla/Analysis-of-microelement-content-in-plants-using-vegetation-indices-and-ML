from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class PLSRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        nc = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model_ = PLSRegression(n_components=max(1, nc))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X).ravel()


def _registry():
    reg = {
        "PLSR":      (PLSRWrapper(10), True),
        "ElasticNet": (ElasticNet(max_iter=5000, random_state=42), True),
        "Ridge":     (Ridge(), True),
        "Lasso":     (Lasso(max_iter=5000, random_state=42), True),
        "RF":        (RandomForestRegressor(random_state=42, n_jobs=-1), False),
        "GBR":       (GradientBoostingRegressor(random_state=42), False),
        "SVR":       (SVR(), True),
        "kNN":       (KNeighborsRegressor(), True),
        "GPR":       (GaussianProcessRegressor(
                        kernel=ConstantKernel() * RBF() + WhiteKernel(),
                        random_state=42, n_restarts_optimizer=3), True),
    }
    try:
        from xgboost import XGBRegressor
        reg["XGBoost"] = (XGBRegressor(random_state=42, n_jobs=-1,
                                        verbosity=0, tree_method="hist"), False)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        reg["LightGBM"] = (LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), False)
    except ImportError:
        pass
    return reg


def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "RPD": np.nan, "n": len(yt)}
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    sd = np.std(yt)
    rpd = sd / rmse if rmse > 0 else np.inf
    iq = np.percentile(yt, 75) - np.percentile(yt, 25)
    rpiq = iq / rmse if rmse > 0 else np.inf
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "RPD": round(rpd, 2), "RPIQ": round(rpiq, 2), "n": len(yt)}


def evaluate_model(model, X, y, cv_strategy="kfold", n_splits=5,
                   groups=None, scale=True):
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    if len(yc) < n_splits * 2:
        return {"R2": np.nan, "RMSE": np.nan}

    if cv_strategy == "spatial" and groups is not None:
        gc = groups[mask]
        cv = GroupKFold(n_splits=min(n_splits, len(np.unique(gc))))
        splits = list(cv.split(Xc, yc, gc))
    elif cv_strategy == "repeated_kfold":
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=10, random_state=42)
        splits = list(cv.split(Xc))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(Xc))

    y_pred = np.full_like(yc, np.nan)
    fold_r2 = []

    for tr_idx, te_idx in splits:
        Xtr, Xte = Xc[tr_idx], Xc[te_idx]
        ytr, yte = yc[tr_idx], yc[te_idx]
        if scale:
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        try:
            m = clone(model)
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
            y_pred[te_idx] = yp
            fold_r2.append(regression_metrics(yte, yp)["R2"])
        except Exception:
            pass

    valid = np.isfinite(y_pred)
    overall = regression_metrics(yc[valid], y_pred[valid])
    overall["R2_fold_mean"] = round(np.mean(fold_r2), 4) if fold_r2 else np.nan
    overall["R2_fold_std"] = round(np.std(fold_r2), 4) if fold_r2 else np.nan
    overall["y_pred"] = y_pred
    overall["y_true"] = yc
    return overall


def compare_models(X, y, model_names=None, cv_strategy="kfold",
                   n_splits=5, groups=None):
    reg = _registry()
    if model_names is None:
        model_names = list(reg.keys())

    results, predictions = [], {}
    for name in model_names:
        if name not in reg:
            continue
        est, needs_scale = reg[name]
        print(f"  {name}...", end=" ", flush=True)

        metrics = evaluate_model(est, X, y, cv_strategy, n_splits, groups, needs_scale)
        r2 = metrics.get("R2", np.nan)
        rmse = metrics.get("RMSE", np.nan)
        rpd = metrics.get("RPD", np.nan)
        print(f"R2={r2:.4f}, RMSE={rmse:.4f}, RPD={rpd:.2f}")

        results.append({
            "model": name, "R2": r2,
            "R2_fold_mean": metrics.get("R2_fold_mean", np.nan),
            "R2_fold_std": metrics.get("R2_fold_std", np.nan),
            "RMSE": rmse, "MAE": metrics.get("MAE", np.nan),
            "RPD": rpd, "RPIQ": metrics.get("RPIQ", np.nan),
            "n": metrics.get("n", 0),
        })
        if "y_pred" in metrics:
            predictions[name] = {"y_true": metrics["y_true"], "y_pred": metrics["y_pred"]}

    df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return df, predictions


def train_final_model(model_name, X, y, scale=True):
    reg = _registry()
    if model_name not in reg:
        raise ValueError(f"Unknown model: {model_name}")
    est, needs_scale = reg[model_name]
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    scaler = None
    if scale or needs_scale:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(Xc)
    model = clone(est)
    model.fit(Xc, yc)
    return model, scaler


_PARAM_GRIDS = {
    "PLSR":      {"n_components": [3, 5, 7, 10, 15]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
    "Ridge":     {"alpha": [0.01, 0.1, 1.0, 10, 100]},
    "Lasso":     {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "RF":        {"n_estimators": [100, 300, 500], "max_depth": [None, 10, 20],
                  "min_samples_leaf": [2, 5, 10]},
    "GBR":       {"n_estimators": [100, 300], "max_depth": [3, 5, 7],
                  "learning_rate": [0.01, 0.05, 0.1]},
    "SVR":       {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10, 100],
                  "epsilon": [0.01, 0.1, 0.5]},
    "kNN":       {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "XGBoost":   {"n_estimators": [100, 300], "max_depth": [3, 5, 7],
                  "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
    "LightGBM":  {"n_estimators": [100, 300], "max_depth": [3, 5, -1],
                  "learning_rate": [0.01, 0.05, 0.1]},
}


def tune_model(model_name, X, y, n_splits=5, scoring="neg_root_mean_squared_error"):
    from sklearn.model_selection import GridSearchCV
    reg = _registry()
    if model_name not in reg:
        raise ValueError(f"Unknown model: {model_name}")
    est, needs_scale = reg[model_name]
    grid = _PARAM_GRIDS.get(model_name, {})
    if not grid:
        return est, {}

    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    if needs_scale:
        sc = StandardScaler()
        Xc = sc.fit_transform(Xc)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(clone(est), grid, cv=cv, scoring=scoring,
                      n_jobs=-1, refit=True)
    gs.fit(Xc, yc)
    print(f"  {model_name} best: {gs.best_params_}, score={gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def stacking_ensemble(X, y, base_models=None, n_splits=5):
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    if base_models is None:
        base_models = ["ElasticNet", "RF", "GBR", "SVR"]

    reg = _registry()
    estimators = []
    for name in base_models:
        if name in reg and name != "PLSR":
            est, _ = reg[name]
            estimators.append((name, clone(est)))

    if len(estimators) < 2:
        return None

    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    sc = StandardScaler()
    Xc = sc.fit_transform(Xc)

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
        cv=KFold(n_splits=min(n_splits, len(yc) // 4), shuffle=True, random_state=42),
        n_jobs=-1,
    )
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        yp = cross_val_predict(stack, Xc, yc, cv=cv)
        m = regression_metrics(yc, yp)
        m["y_true"] = yc
        m["y_pred"] = yp
        return m
    except Exception as e:
        print(f"stacking error: {e}")
        return None


def multi_output_gpr(X, y_matrix, n_splits=5):
    from sklearn.multioutput import MultiOutputRegressor
    mask = np.all(np.isfinite(y_matrix), axis=1)
    Xc, Yc = X[mask], y_matrix[mask]
    sc = StandardScaler()
    Xc = sc.fit_transform(Xc)

    base = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(),
        random_state=42, n_restarts_optimizer=2,
    )
    mo = MultiOutputRegressor(base, n_jobs=-1)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for fold_i, (tr, te) in enumerate(cv.split(Xc)):
        mo_clone = clone(mo)
        mo_clone.fit(Xc[tr], Yc[tr])
        pred = mo_clone.predict(Xc[te])
        for j in range(Yc.shape[1]):
            if j not in results:
                results[j] = {"yt": [], "yp": []}
            results[j]["yt"].extend(Yc[te, j])
            results[j]["yp"].extend(pred[:, j])

    out = {}
    for j, v in results.items():
        yt, yp = np.array(v["yt"]), np.array(v["yp"])
        out[j] = regression_metrics(yt, yp)
    return out
