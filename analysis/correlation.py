"""
Корреляционный анализ: Pearson, Spearman, Kendall.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig


def run_correlation(
    indices_df: pd.DataFrame,
    chemistry_df: pd.DataFrame,
    target_columns: list[str] | None = None,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
   
    if config is None:
        config = AnalysisConfig()

    # Общие точки
    common_ids = indices_df.index.intersection(chemistry_df.index)
    if len(common_ids) < config.min_samples:
        raise ValueError(
            f"Недостаточно общих точек: {len(common_ids)} < {config.min_samples}"
        )

    idx_df = indices_df.loc[common_ids]
    chem_df = chemistry_df.loc[common_ids]

    # Определяем целевые столбцы
    if target_columns is None:
        target_columns = chem_df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for element in target_columns:
        if element not in chem_df.columns:
            continue

        chem_vals = chem_df[element].values.astype(float)

        for idx_name in idx_df.columns:
            idx_vals = idx_df[idx_name].values.astype(float)

            # Маска валидных значений (оба конечные)
            mask = np.isfinite(chem_vals) & np.isfinite(idx_vals)
            n = mask.sum()

            if n < config.min_samples:
                continue

            x = idx_vals[mask]
            y = chem_vals[mask]

            row = {
                "element": element,
                "index": idx_name,
                "n": n,
            }

            # Pearson
            if "pearson" in config.methods:
                r, p = stats.pearsonr(x, y)
                row["pearson_r"] = r
                row["pearson_p"] = p
                row["pearson_r2"] = r ** 2

            # Spearman
            if "spearman" in config.methods:
                rho, p = stats.spearmanr(x, y)
                row["spearman_rho"] = rho
                row["spearman_p"] = p

            # Kendall
            if "kendall" in config.methods:
                tau, p = stats.kendalltau(x, y)
                row["kendall_tau"] = tau
                row["kendall_p"] = p

            results.append(row)

    return pd.DataFrame(results)


def get_top_correlations(
    corr_df: pd.DataFrame,
    metric: str = "pearson_r",
    top_n: int = 5,
    per_element: bool = True,
) -> pd.DataFrame:
    
    df = corr_df.copy()
    df["abs_metric"] = df[metric].abs()

    if per_element:
        result = (
            df.groupby("element")
            .apply(lambda g: g.nlargest(top_n, "abs_metric"), include_groups=False)
            .reset_index(drop=True)
        )
    else:
        result = df.nlargest(top_n, "abs_metric")

    return result.drop(columns=["abs_metric"])


def significance_label(p_value: float, alpha: float = 0.05) -> str:
    """Возвращает маркер значимости: ***, **, *, или пустую строку."""
    if p_value < alpha / 50:
        return "***"
    if p_value < alpha / 5:
        return "**"
    if p_value < alpha:
        return "*"
    return ""
