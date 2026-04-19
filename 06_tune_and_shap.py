#!/usr/bin/env python3
"""
06. Тюнинг гиперпараметров + SHAP-анализ лучших моделей.

Берёт лучшую модель из результатов 03/04/05, тюнит через GridSearchCV,
обучает финальную модель, строит SHAP importance.

    python 06_tune_and_shap.py --results results/03_ml_multi/ml_multi.csv
    python 06_tune_and_shap.py --results results/04_ml_hyper/ml_hyper.csv --top 5
"""
import argparse, os, sys, warnings
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_hyper_from_xlsx, load_chemistry
from analysis.indices import calculate_indices
from analysis.preprocessing import preprocess_pipeline
from analysis.feature_selection import pca_reduce, combined_selection
from analysis.ml_pipeline import tune_model, evaluate_model, _registry, regression_metrics
from analysis.correlation import AnalysisConfig
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


def _load_features(cfg, source, date_key):
    """Загружает данные и строит признаки в зависимости от источника."""
    chem_cols = cfg["chemistry"]["columns"]
    dsm = cfg["chemistry"]["date_sampling_map"]

    sampling_key = dsm.get(date_key, "sampling1")
    chem_path = cfg["paths"]["chemistry"].get(sampling_key)
    offset = cfg["chemistry"]["id_offsets"].get(sampling_key, 0)
    chem_df = load_chemistry(chem_path, chem_cols, offset)

    if source in ("multi", "multi_all"):
        pattern = cfg["paths"]["multi"].get(date_key)
        bands = load_multi(pattern, cfg["camera"]["band_map"])
        idx = calculate_indices(bands, tiers=cfg["statistics"]["index_tiers"])
        combined = pd.concat([bands, idx], axis=1)
        common = combined.index.intersection(chem_df.index)
        return combined.loc[common].values, list(combined.columns), chem_df.loc[common], common

    elif source.startswith("hyper") or source.startswith("pca") or source.startswith("cars"):
        xlsx = cfg["paths"]["hyper"].get(f"{date_key}_xlsx")
        hyper_df, wl = load_hyper_from_xlsx(xlsx)
        common = hyper_df.index.intersection(chem_df.index).values
        spectra = hyper_df.loc[common].values
        wl_names = [f"{w:.1f}" for w in wl]

        if "pca" in source:
            X, _ = pca_reduce(spectra, n_components=10)
            names = [f"PC{i+1}" for i in range(X.shape[1])]
        elif "cars" in source or "selected" in source:
            X_sm, _ = preprocess_pipeline(spectra, wl, ["smooth"])
            return X_sm, wl_names, chem_df.loc[common], common
        elif "deriv" in source:
            X, _ = preprocess_pipeline(spectra, wl, ["smooth", "deriv1", "snv"])
            names = [f"d1_{n}" for n in wl_names]
        else:
            X, _ = preprocess_pipeline(spectra, wl, ["smooth"])
            names = wl_names

        return X, names, chem_df.loc[common], common

    return None, None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--results", required=True, help="CSV из скрипта 03/04/05")
    p.add_argument("--top", type=int, default=5, help="сколько лучших моделей тюнить на элемент")
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    cfg = load_config(args.config)
    outdir = args.output or str(Path(args.results).parent / "tuned")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.results)
    print(f"Результатов: {len(df)}")
    print(f"Элементы: {df['element'].unique().tolist()}")
    print(f"Модели: {df['model'].unique().tolist()}")

    elements = df["element"].unique()
    all_tuned = []

    for elem in elements:
        sub = df[df["element"] == elem].nlargest(args.top, "R2")
        sn = short_name(cfg, elem)
        print(f"\n{'='*50}")
        print(f"{sn}: top-{args.top} моделей для тюнинга")
        print(f"{'='*50}")

        for _, row in sub.iterrows():
            model_name = row["model"]
            fs = row.get("feature_set", "multi")
            date = row.get("date", "date1")

            print(f"\n  {model_name} [{fs}]:")

            X, feat_names, chem, common = _load_features(cfg, fs, date)
            if X is None:
                print("    skip: no data")
                continue

            y = chem[elem].values
            if np.isfinite(y).sum() < 15:
                continue

            # Тюнинг
            try:
                best_est, best_params = tune_model(model_name, X, y, n_splits=5)
            except Exception as e:
                print(f"    tune error: {e}")
                continue

            # Оценка тюненой модели
            reg = _registry()
            _, needs_scale = reg.get(model_name, (None, True))
            metrics = evaluate_model(best_est, X, y, n_splits=5, scale=needs_scale)

            r2_before = row["R2"]
            r2_after = metrics.get("R2", np.nan)
            delta = r2_after - r2_before if np.isfinite(r2_after) else 0
            print(f"    R2: {r2_before:.4f} -> {r2_after:.4f} (delta={delta:+.4f})")

            all_tuned.append({
                "element": elem, "model": model_name, "feature_set": fs,
                "R2_before": r2_before, "R2_after": r2_after, "delta": delta,
                "RMSE": metrics.get("RMSE", np.nan),
                "RPD": metrics.get("RPD", np.nan),
                "best_params": str(best_params),
            })

            # SHAP (для tree-based моделей)
            if model_name in ("RF", "GBR", "XGBoost", "LightGBM") and feat_names:
                try:
                    import shap
                    mask = np.isfinite(y)
                    Xc, yc = X[mask], y[mask]
                    if needs_scale:
                        sc = StandardScaler()
                        Xc = sc.fit_transform(Xc)

                    m = clone(best_est)
                    m.fit(Xc, yc)

                    explainer = shap.TreeExplainer(m)
                    sv = explainer.shap_values(Xc)
                    mean_shap = np.abs(sv).mean(axis=0)

                    # Top-20 features
                    top_idx = np.argsort(mean_shap)[-20:]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(range(len(top_idx)), mean_shap[top_idx], color="#4472C4")
                    if feat_names and len(feat_names) == Xc.shape[1]:
                        ax.set_yticks(range(len(top_idx)))
                        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
                    ax.set_xlabel("mean |SHAP|")
                    ax.set_title(f"SHAP: {sn} / {model_name}")
                    plt.tight_layout()
                    fig.savefig(Path(outdir) / f"shap_{sn}_{model_name}.png",
                                dpi=args.dpi, bbox_inches="tight")
                    plt.close(fig)
                    print(f"    SHAP saved")

                except ImportError:
                    print("    shap not installed")
                except Exception as e:
                    print(f"    SHAP error: {e}")

    if all_tuned:
        tdf = pd.DataFrame(all_tuned)
        tdf.to_csv(Path(outdir) / "tuned_results.csv", index=False, float_format="%.4f")
        print(f"\nResults: {outdir}/tuned_results.csv")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for _, r in tdf.iterrows():
            sn = short_name(cfg, r["element"])
            d = r["delta"]
            mark = "+" if d > 0.01 else "=" if abs(d) < 0.01 else "-"
            print(f"  [{mark}] {sn:>5s} {r['model']:>12s}: "
                  f"{r['R2_before']:.3f} -> {r['R2_after']:.3f} ({d:+.3f})")

    print(f"\nDone -> {outdir}/")


if __name__ == "__main__":
    main()
