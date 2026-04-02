"""
Визуализация результатов корреляционного анализа.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .correlation import significance_label
from .indices import INDEX_REGISTRY, get_indices_by_tier


def _setup_style():
    """Единые настройки стиля для всех графиков."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def plot_heatmap(
    corr_df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (14, None),
    title: str | None = None,
) -> plt.Figure:
    
    _setup_style()

    value_col = {"pearson": "pearson_r", "spearman": "spearman_rho", "kendall": "kendall_tau"}[method]
    p_col = {"pearson": "pearson_p", "spearman": "spearman_p", "kendall": "kendall_p"}[method]
    method_label = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ"}[method]

    # Порядок индексов по тирам
    ordered_indices = get_indices_by_tier(1) + get_indices_by_tier(2) + get_indices_by_tier(3)
    available = corr_df["index"].unique()
    ordered_indices = [i for i in ordered_indices if i in available]

    elements = corr_df["element"].unique().tolist()

    pivot = corr_df.pivot_table(index="index", columns="element", values=value_col)
    pivot_p = corr_df.pivot_table(index="index", columns="element", values=p_col)

    pivot = pivot.reindex(index=ordered_indices, columns=elements)
    pivot_p = pivot_p.reindex(index=ordered_indices, columns=elements)

    # Аннотации со значимостью
    annot = pivot.copy().astype(str)
    for i in pivot.index:
        for j in pivot.columns:
            r_val = pivot.loc[i, j]
            p_val = pivot_p.loc[i, j]
            if pd.isna(r_val):
                annot.loc[i, j] = ""
            else:
                sig = significance_label(p_val)
                annot.loc[i, j] = f"{r_val:.2f}{sig}"

    # Автоподбор высоты
    h = figsize[1] or max(8, len(ordered_indices) * 0.35)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    sns.heatmap(
        pivot.astype(float),
        annot=annot, fmt="",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"label": method_label, "shrink": 0.5},
        ax=ax, annot_kws={"size": 7},
    )

    # Разделители тиров
    tier_sizes = [
        len([i for i in get_indices_by_tier(t) if i in available])
        for t in [1, 2, 3]
    ]
    pos = 0
    for size in tier_sizes[:-1]:
        pos += size
        ax.axhline(y=pos, color="black", linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Вегетативный индекс")
    ax.set_xlabel("Химический элемент")

    plt.tight_layout()
    return fig


def plot_scatter_top(
    corr_df: pd.DataFrame,
    indices_df: pd.DataFrame,
    chemistry_df: pd.DataFrame,
    top_n: int = 12,
    title: str | None = None,
) -> plt.Figure:
    
    _setup_style()

    df = corr_df.copy()
    df["abs_r"] = df["pearson_r"].abs()
    top = df.nlargest(top_n, "abs_r")

    common = indices_df.index.intersection(chemistry_df.index)

    ncols = min(4, top_n)
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if top_n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        idx_name = row["index"]
        elem_name = row["element"]

        x = indices_df.loc[common, idx_name].values.astype(float)
        y = chemistry_df.loc[common, elem_name].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, alpha=0.5, s=20, c="#2E75B6", edgecolors="none")

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color="#C00000", linewidth=1.5, linestyle="--")

        r = row["pearson_r"]
        sig = significance_label(row["pearson_p"])
        ax.set_title(f"{idx_name} vs {elem_name}", fontsize=9)
        ax.text(0.05, 0.95, f"r={r:.3f}{sig}\nn={len(x)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_xlabel(idx_name, fontsize=8)
        ax.set_ylabel(elem_name, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_distributions(
    df: pd.DataFrame,
    title: str = "",
    max_cols: int = 4,
    color: str = "#5B9BD5",
) -> plt.Figure:
    
    _setup_style()

    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ncols = min(max_cols, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(cols):
        vals = df[col].dropna().values
        if len(vals) > 0:
            axes[i].hist(vals, bins=20, color=color, edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontsize=9)
            axes[i].axvline(np.mean(vals), color="#C00000", linestyle="--", linewidth=1)
            axes[i].tick_params(labelsize=7)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


def plot_method_comparison(
    corr_df: pd.DataFrame,
    title: str = "",
) -> plt.Figure:
    """
    Барплот сравнения Pearson/Spearman/Kendall по элементам.
    """
    _setup_style()
    elements = corr_df["element"].unique()

    bars = []
    for elem in elements:
        sub = corr_df[corr_df["element"] == elem]
        row = {"element": elem}
        if "pearson_r" in sub.columns:
            row["Pearson |r|"] = sub["pearson_r"].abs().max()
        if "spearman_rho" in sub.columns:
            row["Spearman |ρ|"] = sub["spearman_rho"].abs().max()
        if "kendall_tau" in sub.columns:
            row["Kendall |τ|"] = sub["kendall_tau"].abs().max()
        bars.append(row)

    bars_df = pd.DataFrame(bars)
    x = np.arange(len(bars_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(elements) * 1.2), 5))

    colors = ["#2E75B6", "#E06666", "#70AD47"]
    for i, (col, color) in enumerate(zip(
        ["Pearson |r|", "Spearman |ρ|", "Kendall |τ|"], colors
    )):
        if col in bars_df.columns:
            ax.bar(x + (i - 1) * width, bars_df[col], width, label=col, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(bars_df["element"], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Макс. |корреляция|")
    ax.legend(fontsize=9)
    ax.axhline(y=0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, format: str = "png", dpi: int = 150) -> bytes:
    """Конвертирует matplotlib Figure в байты для скачивания."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
