
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    model_type: str = "tree",
    max_samples: int = 100,
) -> tuple[np.ndarray, list[str]]:
    
    try:
        import shap
    except ImportError:
        raise ImportError("Установите shap: pip install shap")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    elif model_type == "kernel":
        bg = shap.sample(X, min(max_samples, len(X)))
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X, nsamples=max_samples)
    else:
        raise ValueError(f"Неизвестный тип: {model_type}")

    return sv, feature_names

def plot_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "",
) -> plt.Figure:
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(mean_shap)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(
        range(len(idx_sorted)),
        mean_shap[idx_sorted],
        color="#5B9BD5",
    )
    ax.set_yticks(range(len(idx_sorted)))
    ax.set_yticklabels([feature_names[i] for i in idx_sorted], fontsize=9)
    ax.set_xlabel("mean |SHAP value|")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return fig

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "",
) -> plt.Figure:
    
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    shap.summary_plot(
        shap_values[:, top_idx],
        X[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        plot_size=None,
    )
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    return fig

def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    element: str = "",
) -> plt.Figure:
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, alpha=0.5, s=30, c="#2E75B6", edgecolors="none")

    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)

    z = np.polyfit(yt, yp, 1)
    p = np.poly1d(z)
    ax.plot(sorted(yt), p(sorted(yt)), color="#C00000", linewidth=1.5)

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    ax.text(
        0.05, 0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(yt)}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(f"Observed {element}")
    ax.set_ylabel(f"Predicted {element}")
    ax.set_title(f"{model_name}: {element}", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig

def plot_model_comparison(results_df: pd.DataFrame, metric: str = "R2") -> plt.Figure:
    
    df = results_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.4)))

    colors = ["#5B9BD5" if v >= 0 else "#E06666" for v in df[metric]]
    ax.barh(range(len(df)), df[metric], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlabel(metric)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_title(f"Сравнение моделей по {metric}", fontsize=12)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row[metric] + 0.01, i, f"{row[metric]:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig
