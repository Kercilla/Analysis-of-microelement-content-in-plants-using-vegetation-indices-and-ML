#!/usr/bin/env python3

import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import (
    load_hyper_date, load_hyper_from_xlsx,
    load_hyper_wavelength_map, load_chemistry,
)
from analysis.preprocessing import preprocess_pipeline
from analysis.hyper_indices import calculate_hyper_indices, HYPER_INDEX_REGISTRY
from analysis.correlation import run_correlation
from analysis.config import AnalysisConfig


def correlogram(spectra, y, wavelengths):
    mask = np.isfinite(y)
    X, yc = spectra[mask], y[mask]
    rows = []
    for i, wl in enumerate(wavelengths):
        b = X[:, i]; v = np.isfinite(b)
        if v.sum() < 10: continue
        r, p = stats.pearsonr(b[v], yc[v])
        rho, sp = stats.spearmanr(b[v], yc[v])
        rows.append({"wl": wl, "idx": i, "r": r, "p": p, "rho": rho, "n": v.sum()})
    return pd.DataFrame(rows)


def plot_correlogram(prep_cgs, element, sname, output):
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#2E75B6", "#E06666", "#70AD47", "#9B59B6", "#F39C12"]
    for i, (name, cg) in enumerate(prep_cgs.items()):
        ax.plot(cg["wl"], cg["r"], lw=1.2, color=colors[i%5], label=name, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.5)
    for thr in [0.3, -0.3]: ax.axhline(thr, color="gray", lw=0.5, ls=":")
    for b, c in [(475,"blue"),(560,"green"),(668,"red"),(717,"orange"),(840,"brown")]:
        ax.axvline(b, color=c, lw=0.7, ls="--", alpha=0.4)
    ax.set(xlabel="λ, нм", ylabel="Pearson r", ylim=(-0.6, 0.6))
    ax.set_title(f"Коррелограмма: {sname}"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--date", nargs="+", default=None)
    p.add_argument("--use-xlsx", action="store_true", help="Использовать собранный xlsx вместо папки")
    p.add_argument("--elements", nargs="+", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    outdir = args.output or cfg["paths"]["output"]["stat_hyper"]
    os.makedirs(outdir, exist_ok=True)

    chem_cols = cfg["chemistry"]["columns"]
    targets = args.elements or cfg["chemistry"]["target_elements"]
    preps = cfg["hyper_preprocessing"]["variants"]
    date_sampling = cfg["chemistry"]["date_sampling_map"]
    hn = cfg["hyper_naming"]
    dpi = cfg["plots"]["dpi"]

    # Маппинг длин волн
    wl_map_path = cfg["paths"]["hyper"]["wavelength_map"]
    wl_map = load_hyper_wavelength_map(wl_map_path)

    # Какие даты
    dates = args.date or ["date1", "date2"]

    print("=" * 60)
    print("02. ГИПЕРСПЕКТР: СТАТИСТИКА + ГИПЕРСПЕКТРАЛЬНЫЕ ИНДЕКСЫ")
    print("=" * 60)

    for date_key in dates:
        folder_key = f"{date_key}_folder"
        folder = cfg["paths"]["hyper"].get(folder_key)
        xlsx_key = f"{date_key}_xlsx"
        xlsx_path = cfg["paths"]["hyper"].get(xlsx_key)

        print(f"\n{'━' * 55}")
        print(f"  Дата: {date_key}")

        # Загрузка гиперспектра
        if args.use_xlsx and xlsx_path and Path(xlsx_path).exists():
            hyper_df, wavelengths = load_hyper_from_xlsx(xlsx_path)
        elif folder and Path(folder).exists():
            hyper_df, wavelengths = load_hyper_date(
                folder, wl_map, prefix_len=hn["prefix_length"],
                value_col=hn["value_column"], id_col=hn["id_column"],
                min_valid_bands=hn["min_valid_bands"],
            )
        elif xlsx_path and Path(xlsx_path).exists():
            hyper_df, wavelengths = load_hyper_from_xlsx(xlsx_path)
        else:
            print(f"  ⚠ Нет данных для {date_key}")
            continue

        # Химия
        sampling_key = None
        for dk, sv in date_sampling.items():
            if dk in date_key or date_key in dk:
                sampling_key = sv; break
        if not sampling_key: sampling_key = "sampling1"

        chem_path = cfg["paths"]["chemistry"].get(sampling_key)
        if not chem_path or not Path(chem_path).exists():
            print(f"  ⚠ Химия не найдена"); continue

        offset = cfg["chemistry"]["id_offsets"].get(sampling_key, 0)
        chem_df = load_chemistry(chem_path, chem_cols, offset)
        common = hyper_df.index.intersection(chem_df.index).values
        spectra = hyper_df.loc[common].values
        print(f"  Общих: {len(common)}")

        if len(common) < 10:
            print(f"  Мало точек"); continue

        # ── A. Коррелограммы r(λ) ──
        print(f"\n  A. КОРРЕЛОГРАММЫ")
        all_rows = []
        for prep_name, steps in preps.items():
            X, _ = preprocess_pipeline(spectra, wavelengths, steps=steps)
            print(f"    [{prep_name}]")
            for elem in targets:
                if elem not in chem_df.columns: continue
                y = chem_df.loc[common, elem].values
                cg = correlogram(X, y, wavelengths)
                if cg.empty: continue
                best = cg.loc[cg["r"].abs().idxmax()]
                sn = short_name(cfg, elem)
                print(f"      {sn:>5s}: r={best['r']:+.3f} @ {best['wl']:.0f} нм")
                for _, row in cg.iterrows():
                    rd = row.to_dict(); rd["element"] = elem; rd["prep"] = prep_name
                    all_rows.append(rd)

        # Графики
        for elem in targets:
            if elem not in chem_df.columns: continue
            sn = short_name(cfg, elem)
            prep_cgs = {}
            for pn, steps in preps.items():
                X, _ = preprocess_pipeline(spectra, wavelengths, steps=steps)
                y = chem_df.loc[common, elem].values
                prep_cgs[pn] = correlogram(X, y, wavelengths)
            plot_correlogram(prep_cgs, elem, sn,
                             str(Path(outdir) / f"correlogram_{sn}_{date_key}.png"))

        # Тепловая карта
        raw_cgs = {}
        X_raw, _ = preprocess_pipeline(spectra, wavelengths, steps=["smooth"])
        for elem in targets:
            if elem not in chem_df.columns: continue
            raw_cgs[elem] = correlogram(X_raw, chem_df.loc[common, elem].values, wavelengths)

        matrix = np.zeros((len(raw_cgs), len(wavelengths)))
        elem_list = list(raw_cgs.keys())
        for i, elem in enumerate(elem_list):
            for _, row in raw_cgs[elem].iterrows():
                matrix[i, int(row["idx"])] = row["r"]

        step = max(1, len(wavelengths) // 50)
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=[f"{w:.0f}" if j%step==0 else "" for j,w in enumerate(wavelengths)],
                    yticklabels=[short_name(cfg, e) for e in elem_list], ax=ax)
        ax.set_title(f"Корреляция r(λ) — {date_key}")
        plt.tight_layout()
        fig.savefig(Path(outdir) / f"heatmap_wl_{date_key}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # ── B. Гиперспектральные индексы ──
        print(f"\n  B. ГИПЕРСПЕКТРАЛЬНЫЕ ИНДЕКСЫ ({len(HYPER_INDEX_REGISTRY)} шт.)")
        hi_df = calculate_hyper_indices(spectra, wavelengths)
        print(f"    Рассчитано: {hi_df.shape[1]} индексов")

        stat_cfg = cfg["statistics"]
        acfg = AnalysisConfig(methods=stat_cfg["methods"], alpha=stat_cfg["alpha"])

        hi_df.index = common
        hi_corr = run_correlation(
            hi_df, chem_df.loc[common],
            [t for t in targets if t in chem_df.columns], acfg,
        )
        hi_corr["date"] = date_key

        for elem in targets:
            sub = hi_corr[(hi_corr["element"] == elem) & hi_corr["pearson_r"].notna()]
            if sub.empty: continue
            best = sub.loc[sub["pearson_r"].abs().idxmax()]
            sn = short_name(cfg, elem)
            print(f"    {sn:>5s}: {best['index']:>20s} r={best['pearson_r']:+.3f}")

        hi_corr.to_csv(Path(outdir) / f"hyper_indices_corr_{date_key}.csv",
                        index=False, float_format="%.4f")

        # Сохранение полной коррелограммы
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                Path(outdir) / f"correlogram_{date_key}.csv",
                index=False, float_format="%.6f",
            )

    print(f"\nГОТОВО → {outdir}/")


if __name__ == "__main__":
    main()
