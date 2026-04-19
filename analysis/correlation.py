from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AnalysisConfig:
    methods: list[str] = field(default_factory=lambda: ["pearson", "spearman", "kendall"])
    alpha: float = 0.05
    min_samples: int = 10
    index_tiers: list[int] = field(default_factory=lambda: [1, 2, 3])
    scatter_top_n: int = 12


def run_correlation(indices_df, chemistry_df, target_columns=None, config=None):
    """Корреляции между индексами и химическими элементами."""
    if config is None:
        config = AnalysisConfig()

    common = indices_df.index.intersection(chemistry_df.index)
    if len(common) < config.min_samples:
        raise ValueError(f"Мало общих точек: {len(common)} < {config.min_samples}")

    idx = indices_df.loc[common]
    chem = chemistry_df.loc[common]
    if target_columns is None:
        target_columns = chem.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for elem in target_columns:
        if elem not in chem.columns:
            continue
        y = chem[elem].values.astype(float)
        for name in idx.columns:
            x = idx[name].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = mask.sum()
            if n < config.min_samples:
                continue
            row = {"element": elem, "index": name, "n": n}
            xm, ym = x[mask], y[mask]
            if "pearson" in config.methods:
                r, p = stats.pearsonr(xm, ym)
                row.update(pearson_r=r, pearson_p=p, pearson_r2=r**2)
            if "spearman" in config.methods:
                rho, p = stats.spearmanr(xm, ym)
                row.update(spearman_rho=rho, spearman_p=p)
            if "kendall" in config.methods:
                tau, p = stats.kendalltau(xm, ym)
                row.update(kendall_tau=tau, kendall_p=p)
            results.append(row)

    return pd.DataFrame(results)


def get_top_correlations(corr_df, metric="pearson_r", top_n=5, per_element=True):
    df = corr_df.copy()
    df["_abs"] = df[metric].abs()
    if per_element:
        result = (df.groupby("element")
                  .apply(lambda g: g.nlargest(top_n, "_abs"), include_groups=False)
                  .reset_index(drop=True))
    else:
        result = df.nlargest(top_n, "_abs")
    return result.drop(columns=["_abs"])


def significance_label(p, alpha=0.05):
    if p < alpha / 50: return "***"
    if p < alpha / 5: return "**"
    if p < alpha: return "*"
    return ""
