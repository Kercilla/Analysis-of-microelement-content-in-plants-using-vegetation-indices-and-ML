#!/usr/bin/env python3
"""
00. Агрегация гиперспектральных данных из папок с .gpkg в единый xlsx.

Читает 30K+ файлов, группирует по каналам, усредняет по наборам/пролётам,
сохраняет компактную матрицу (точки × каналы).

    python 00_aggregate_hyper.py
    python 00_aggregate_hyper.py --date date1 --min-bands 100
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis.cfg import load_config
from analysis.loaders import load_hyper_date, load_hyper_wavelength_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--date", nargs="+", default=None)
    p.add_argument("--min-bands", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    hn = cfg["hyper_naming"]
    min_bands = args.min_bands or hn["min_valid_bands"]
    wl_map = load_hyper_wavelength_map(cfg["paths"]["hyper"]["wavelength_map"])
    dates = args.date or ["date1", "date2"]
    out_dir = args.output_dir or "data/hyper"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Каналов в маппинге: {len(wl_map)}")
    print(f"Мин. валидных каналов: {min_bands}")

    for date_key in dates:
        folder = cfg["paths"]["hyper"].get(f"{date_key}_folder")
        if not folder or not Path(folder).exists():
            print(f"\n{date_key}: папка не найдена ({folder})")
            continue

        print(f"\n{'='*60}")
        print(f"Дата: {date_key} -> {folder}")
        print(f"{'='*60}")

        t0 = time.time()
        hyper_df, wavelengths = load_hyper_date(
            folder, wl_map,
            prefix_len=hn["prefix_length"],
            value_col=hn["value_column"],
            id_col=hn["id_column"],
            min_valid_bands=min_bands,
        )
        elapsed = time.time() - t0
        print(f"  Время: {elapsed:.1f} сек")

        if hyper_df.empty:
            print("  Нет данных")
            continue

        # Статистика
        valid_per_point = hyper_df.notna().sum(axis=1)
        nan_per_point = hyper_df.isna().sum(axis=1)
        print(f"  Точек: {len(hyper_df)}")
        print(f"  Валидных каналов/точку: мин={valid_per_point.min()}, "
              f"медиана={valid_per_point.median():.0f}, макс={valid_per_point.max()}")
        print(f"  NaN каналов/точку: мин={nan_per_point.min()}, "
              f"макс={nan_per_point.max()}")

        # Спектральный диапазон
        print(f"  Диапазон: {wavelengths[0]:.1f} — {wavelengths[-1]:.1f} нм")

        # Сохранение
        out_path = Path(out_dir) / f"hyper_aggregated_{date_key}.xlsx"

        # Формат: строка 0 = длины волн, строки 1+ = данные
        header_df = pd.DataFrame([wavelengths], columns=hyper_df.columns)
        header_df.index = ["wavelength_nm"]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            combined = pd.concat([header_df, hyper_df])
            combined.to_excel(writer, sheet_name="spectra")

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Сохранено: {out_path} ({size_mb:.1f} MB)")

        # CSV-копия (быстрее читать потом)
        csv_path = Path(out_dir) / f"hyper_aggregated_{date_key}.csv"
        hyper_df.to_csv(csv_path, float_format="%.4f")
        print(f"  CSV: {csv_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
