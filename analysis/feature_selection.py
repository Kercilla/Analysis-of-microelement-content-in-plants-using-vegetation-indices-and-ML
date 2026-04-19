import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def vip_scores(X, y, n_components=10):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = np.diag(T.T @ T @ Q.T @ Q).ravel()
    p = X.shape[1]
    vip = np.zeros(p)
    for j in range(p):
        vip[j] = np.sqrt(p * np.sum(ss * W[j, :] ** 2) / ss.sum())
    return vip


def select_by_vip(X, y, threshold=1.0, n_components=10):
    return np.where(vip_scores(X, y, n_components) >= threshold)[0]


def spa(X, n_select=20, start=None):
    n, p = X.shape
    n_select = min(n_select, p)
    if start is None: start = np.argmax(X.std(axis=0))
    sel = [start]
    rem = list(range(p))
    rem.remove(start)
    Xp = X.copy()
    for _ in range(n_select - 1):
        if not rem: break
        last = sel[-1]
        xl = Xp[:, last]
        ns = xl @ xl
        if ns == 0: break
        for j in rem:
            Xp[:, j] -= (Xp[:, j] @ xl) / ns * xl
        norms = [np.linalg.norm(Xp[:, j]) for j in rem]
        best = rem[np.argmax(norms)]
        sel.append(best)
        rem.remove(best)
    return np.array(sorted(sel))


def cars(X, y, n_iter=50, n_pls=10, cv=5):
    n, p = X.shape
    n_pls = min(n_pls, n - 1, p)
    r = np.power(p / 2, 1 / (n_iter - 1))
    n_vars = [max(2, int(p * np.exp(-i * np.log(r)))) for i in range(n_iter)]
    best_score, best_sel = np.inf, np.arange(p)
    active = np.arange(p)
    rng = np.random.default_rng(42)
    for it in range(n_iter):
        Xa = X[:, active]
        nc = min(n_pls, Xa.shape[1], n - 1)
        if nc < 1 or Xa.shape[1] < 2: break
        pls = PLSRegression(n_components=nc)
        try: pls.fit(Xa, y)
        except: break
        coefs = np.abs(pls.coef_.ravel())
        if coefs.sum() == 0: break
        w = coefs / coefs.sum()
        try:
            scores = cross_val_score(PLSRegression(n_components=nc), Xa, y,
                                     cv=min(cv, n), scoring="neg_mean_squared_error")
            rmsecv = np.sqrt(-scores.mean())
        except: rmsecv = np.inf
        if rmsecv < best_score:
            best_score, best_sel = rmsecv, active.copy()
        nk = min(n_vars[it], len(active))
        if nk < 2: break
        try: idx = rng.choice(len(active), size=nk, replace=False, p=w)
        except: idx = np.argsort(w)[-nk:]
        active = active[np.sort(idx)]
    return np.sort(best_sel)


def pca_reduce(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca


def rf_importance_select(X, y, n_select=30, n_est=200):
    mask = np.isfinite(y)
    rf = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(X[mask], y[mask])
    return np.sort(np.argsort(rf.feature_importances_)[-n_select:])


def combined_selection(X, y, method="cars_spa", n_select=25, **kw):
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    if method == "vip": return select_by_vip(Xc, yc, **kw)
    if method == "spa": return spa(Xc, n_select=n_select)
    if method == "cars": return cars(Xc, yc, **kw)
    if method == "cars_spa":
        ci = cars(Xc, yc, **kw)
        if len(ci) > n_select:
            si = spa(Xc[:, ci], n_select=n_select)
            return ci[si]
        return ci
    if method == "rf": return rf_importance_select(Xc, yc, n_select=n_select)
    raise ValueError(f"Unknown method: {method}")
