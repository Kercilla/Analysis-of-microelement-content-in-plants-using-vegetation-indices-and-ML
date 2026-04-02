#!/usr/bin/env python3

import argparse, os, sys, warnings
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multispectral_date, load_chemistry
from analysis.indices import calculate_indices
from analysis.correlation import run_correlation, get_top_correlations
from analysis.config import AnalysisConfig
from analysis.visualization import (
    plot_heatmap, plot_scatter_top, plot_method_comparison, fig_to_bytes,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--date", nargs="+", default=None, help="Какие даты (ключи из config)")
    p.add_argument("--elements", nargs="+", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    outdir = args.output or cfg["paths"]["output"]["stat_multi"]
    os.makedirs(outdir, exist_ok=True)

    band_map = cfg["camera"]["band_map"]
    chem_cols = cfg["chemistry"]["columns"]
    targets = args.elements or cfg["chemistry"]["target_elements"]
    stat_cfg = cfg["statistics"]
    dpi = cfg["plots"]["dpi"]
    date_sampling = cfg["chemistry"]["date_sampling_map"]

    config = AnalysisConfig(
        methods=stat_cfg["methods"], alpha=stat_cfg["alpha"],
        index_tiers=stat_cfg["index_tiers"],
    )

    # Какие даты обрабатывать
    dates_to_run = args.date or list(cfg["paths"]["multi"].keys())

    print("=" * 60)
    print("01. МУЛЬТИСПЕКТР: СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 60)

    all_results = []
    for date_key in dates_to_run:
        pattern = cfg["paths"]["multi"].get(date_key)
        if not pattern:
            print(f"  ⚠ Дата '{date_key}' не найдена в config")
            continue

        files = sorted(glob(pattern))
        if not files:
            print(f"  ⚠ Нет файлов: {pattern}")
            continue

        print(f"\n  Дата: {date_key} ({len(files)} файлов)")

        # Загрузка мультиспектра
        bands_df = load_multispectral_date(pattern, band_map)
        print(f"  Каналы: {list(bands_df.columns)}, точек: {len(bands_df)}")

        # Загрузка химии (определяем какой отбор)
        # Извлекаем дату из паттерна для маппинга
        date_str = None
        for dk, sv in date_sampling.items():
            if dk in pattern or dk == date_key:
                date_str = dk
                break

        sampling_key = date_sampling.get(date_str, "sampling1")
        chem_path = cfg["paths"]["chemistry"].get(sampling_key)
        if not chem_path or not Path(chem_path).exists():
            print(f"  ⚠ Химия не найдена: {chem_path}")
            continue

        offset = cfg["chemistry"]["id_offsets"].get(sampling_key, 0)
        chem_df = load_chemistry(chem_path, chem_cols, offset)
        print(f"  Химия: {len(chem_df)} проб (offset={offset})")

        # Индексы
        indices_df = calculate_indices(bands_df, tiers=stat_cfg["index_tiers"])
        common = indices_df.index.intersection(chem_df.index)
        print(f"  Индексов: {len(indices_df.columns)}, общих: {len(common)}")

        if len(common) < stat_cfg["min_samples"]:
            print(f"  Мало точек, пропуск")
            continue

        # Корреляция
        corr = run_correlation(
            indices_df.loc[common], chem_df.loc[common],
            [t for t in targets if t in chem_df.columns], config,
        )
        corr["date"] = date_key
        all_results.append(corr)

        # Графики
        for method in stat_cfg["methods"]:
            fig = plot_heatmap(corr, method, title=f"{method} — {date_key}")
            fig.savefig(Path(outdir) / f"heatmap_{method}_{date_key}.png",
                        dpi=dpi, bbox_inches="tight")

        fig = plot_scatter_top(corr, indices_df.loc[common], chem_df.loc[common],
                               stat_cfg["scatter_top_n"],
                               title=f"Top-{stat_cfg['scatter_top_n']} — {date_key}")
        fig.savefig(Path(outdir) / f"scatter_{date_key}.png", dpi=dpi, bbox_inches="tight")

        fig = plot_method_comparison(corr, title=f"Методы — {date_key}")
        fig.savefig(Path(outdir) / f"methods_{date_key}.png", dpi=dpi, bbox_inches="tight")

        # Вывод лучших
        sig = (corr["pearson_p"] < stat_cfg["alpha"]).sum() if "pearson_p" in corr.columns else 0
        print(f"  Пар: {len(corr)}, значимых: {sig}")
        for elem in targets:
            sub = corr[(corr["element"] == elem) & corr["pearson_r"].notna()]
            if sub.empty: continue
            best = sub.loc[sub["pearson_r"].abs().idxmax()]
            print(f"    {short_name(cfg, elem):>5s}: {best['index']:>12s} r={best['pearson_r']:+.3f}")

    # Сохранение
    if all_results:
        full = pd.concat(all_results, ignore_index=True)
        full.to_csv(Path(outdir) / "correlations.csv", index=False, float_format="%.6f")
        get_top_correlations(full, top_n=3, per_element=True).to_csv(
            Path(outdir) / "top3.csv", index=False, float_format="%.4f",
        )
        print(f"\nГОТОВО → {outdir}/")


if __name__ == "__main__":
    main()
