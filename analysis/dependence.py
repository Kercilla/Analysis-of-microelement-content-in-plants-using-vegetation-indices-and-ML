"""
analysis/dependence.py — продвинутые меры зависимости.

Обёртки для: dCor, MI (KSG), HSIC, partial corr, Graphical LASSO, CCA.
Все функции принимают 1D массивы (x, y) или матрицы (X, Y).
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from sklearn.covariance import GraphicalLassoCV
from scipy import stats


# ── Distance correlation ──────────────────────────────────────────────────────

def distance_correlation(x, y):
    """
    Коэффициент дистанционной корреляции (Székely 2007).
    Обнаруживает любые зависимости, не только линейные.
    Возвращает (dCor, p-value) через permutation test (199 перест.).
    """
    try:
        import dcor
        res = dcor.independence.distance_correlation_t_test(
            x.reshape(-1, 1), y.reshape(-1, 1))
        return {"dCor": round(float(dcor.distance_correlation(x, y)), 4),
                "p_value": round(float(res.pvalue), 4)}
    except ImportError:
        # fallback: вычисляем вручную без p-value
        return {"dCor": round(_dcor_manual(x, y), 4), "p_value": np.nan}


def _dcor_manual(x, y):
    """Ручное вычисление dCor без пакета dcor."""
    def _cent(a):
        d = np.abs(a[:, None] - a[None, :])
        row = d.mean(axis=1, keepdims=True)
        col = d.mean(axis=0, keepdims=True)
        grd = d.mean()
        return d - row - col + grd
    n = len(x)
    Ax = _cent(x.astype(float))
    Ay = _cent(y.astype(float))
    dcov2_xy = (Ax * Ay).sum() / n**2
    dcov2_xx = (Ax * Ax).sum() / n**2
    dcov2_yy = (Ay * Ay).sum() / n**2
    denom = np.sqrt(abs(dcov2_xx) * abs(dcov2_yy))
    return float(np.sqrt(abs(dcov2_xy) / denom)) if denom > 0 else 0.0


# ── Mutual Information ────────────────────────────────────────────────────────

def mutual_information(x, y, n_neighbors=5):
    """
    Взаимная информация (KSG estimator, Kraskov 2004).
    Устойчив к нелинейным зависимостям и не требует нормальности.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return {"MI": np.nan}
    xc, yc = x[mask].reshape(-1, 1), y[mask]
    mi = mutual_info_regression(xc, yc, n_neighbors=n_neighbors, random_state=42)[0]
    return {"MI": round(float(mi), 6)}


# ── HSIC ─────────────────────────────────────────────────────────────────────

def hsic_test(x, y):
    """
    Hilbert-Schmidt Independence Criterion (Gretton 2005).
    Kernel-based мера зависимости.
    """
    try:
        from hyppo.independence import Hsic
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return {"HSIC_stat": np.nan, "p_value": np.nan}
        stat, pval = Hsic().test(
            x[mask].reshape(-1, 1),
            y[mask].reshape(-1, 1),
            reps=199, random_state=42)
        return {"HSIC_stat": round(float(stat), 6),
                "p_value":   round(float(pval), 4)}
    except ImportError:
        return {"HSIC_stat": np.nan, "p_value": np.nan}


# ── Partial correlation ───────────────────────────────────────────────────────

def partial_correlation(df, x_col, y_col, covar_cols, method="pearson"):
    """
    Частная корреляция x и y при контроле covar_cols.
    Использует pingouin.partial_corr.
    """
    try:
        import pingouin as pg
        result = pg.partial_corr(
            data=df, x=x_col, y=y_col,
            covar=covar_cols, method=method)
        return {
            "partial_r":   round(float(result["r"].values[0]), 4),
            "p_value":     round(float(result["p-val"].values[0]), 4),
            "CI95_lo":     round(float(result["CI95%"].values[0][0]), 4),
            "CI95_hi":     round(float(result["CI95%"].values[0][1]), 4),
        }
    except ImportError:
        # fallback через scipy
        return _partial_corr_scipy(df, x_col, y_col, covar_cols)
    except Exception:
        return {"partial_r": np.nan, "p_value": np.nan}


def _partial_corr_scipy(df, x_col, y_col, covar_cols):
    """Fallback partial correlation через регрессию остатков."""
    from sklearn.linear_model import LinearRegression
    sub = df[[x_col, y_col] + covar_cols].dropna()
    if len(sub) < 5:
        return {"partial_r": np.nan, "p_value": np.nan}
    Z = sub[covar_cols].values
    lr = LinearRegression()
    rx = sub[x_col].values - lr.fit(Z, sub[x_col].values).predict(Z)
    ry = sub[y_col].values - lr.fit(Z, sub[y_col].values).predict(Z)
    r, p = stats.pearsonr(rx, ry)
    return {"partial_r": round(r, 4), "p_value": round(p, 4)}


# ── Graphical LASSO ───────────────────────────────────────────────────────────

def graphical_lasso(Z, feature_names=None):
    """
    Graphical LASSO для оценки условной структуры зависимостей.
    Z: матрица (n, p) — объединённая [нутриенты | индексы].
    Возвращает: precision matrix и список значимых рёбер.
    """
    mask = np.all(np.isfinite(Z), axis=1)
    Zc = Z[mask]
    if Zc.shape[0] < Zc.shape[1] + 5:
        return None

    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Zc)

    try:
        gl = GraphicalLassoCV(max_iter=500, cv=3)
        gl.fit(Zs)
        prec = gl.precision_
        edges = []
        p = prec.shape[0]
        for i in range(p):
            for j in range(i+1, p):
                if abs(prec[i, j]) > 1e-6:
                    ni = feature_names[i] if feature_names else str(i)
                    nj = feature_names[j] if feature_names else str(j)
                    edges.append({
                        "node_a": ni, "node_b": nj,
                        "partial_corr": round(
                            -prec[i,j] / np.sqrt(prec[i,i]*prec[j,j]), 4)
                    })
        return {"precision": prec, "edges": pd.DataFrame(edges)}
    except Exception as e:
        print(f"  GraphicalLasso error: {e}")
        return None


# ── CCA ───────────────────────────────────────────────────────────────────────

def canonical_correlation(X, Y, n_components=3):
    """
    Canonical Correlation Analysis между блоком спектральных признаков X
    и блоком нутриентов Y.
    Возвращает канонические корреляции и объяснённую дисперсию.
    """
    from sklearn.preprocessing import StandardScaler
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    Xc = StandardScaler().fit_transform(X[mask])
    Yc = StandardScaler().fit_transform(Y[mask])
    nc = min(n_components, Xc.shape[1], Yc.shape[1], Xc.shape[0]-2)
    if nc < 1:
        return None
    cca = CCA(n_components=nc)
    Xt, Yt = cca.fit_transform(Xc, Yc)
    corrs = [stats.pearsonr(Xt[:, i], Yt[:, i])[0] for i in range(nc)]
    return {
        "canonical_corrs":  [round(c, 4) for c in corrs],
        "n_components_used": nc,
    }


# ── All-in-one для одной пары (index, nutrient) ───────────────────────────────

def full_dependence_profile(x, y, df=None, covar_cols=None,
                             x_name="x", y_name="y"):
    """
    Считает все меры зависимости для пары (x, y).
    Возвращает dict со всеми метриками.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    n = mask.sum()

    r, pr   = stats.pearsonr(xc, yc)   if n >= 3 else (np.nan, np.nan)
    rho, ps = stats.spearmanr(xc, yc)  if n >= 3 else (np.nan, np.nan)
    tau, pt = stats.kendalltau(xc, yc) if n >= 3 else (np.nan, np.nan)

    out = {
        "n":            n,
        "pearson_r":    round(r,   4) if np.isfinite(r)   else np.nan,
        "pearson_p":    round(pr,  4) if np.isfinite(pr)  else np.nan,
        "spearman_rho": round(rho, 4) if np.isfinite(rho) else np.nan,
        "spearman_p":   round(ps,  4) if np.isfinite(ps)  else np.nan,
        "kendall_tau":  round(tau, 4) if np.isfinite(tau) else np.nan,
        "kendall_p":    round(pt,  4) if np.isfinite(pt)  else np.nan,
    }

    dc = distance_correlation(xc, yc)
    out["dCor"]      = dc["dCor"]
    out["dCor_p"]    = dc["p_value"]

    mi = mutual_information(xc, yc)
    out["MI"] = mi["MI"]

    if df is not None and covar_cols:
        pc = partial_correlation(df, x_name, y_name, covar_cols)
        out["partial_r"] = pc.get("partial_r")
        out["partial_p"] = pc.get("p_value")

    return out
