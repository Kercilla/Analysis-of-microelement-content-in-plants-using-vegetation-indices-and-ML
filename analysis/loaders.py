

import re
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd


# ═══════════════════════════════════════════════════════════
#  Мультиспектр
# ═══════════════════════════════════════════════════════════

def load_multispectral_date(
    gpkg_pattern: str,
    band_map: dict[str, str],
    value_col: str = "_mean",
    id_col: str = "id",
) -> pd.DataFrame:
    
    from glob import glob
    files = sorted(glob(gpkg_pattern))
    if not files:
        raise FileNotFoundError(f"Нет файлов по паттерну: {gpkg_pattern}")

    records = {}
    for fpath in files:
        stem = Path(fpath).stem.split("_")[0]  # "11_05_29" → "11"
        channel = band_map.get(stem)
        if channel is None:
            continue

        gdf = gpd.read_file(fpath)
        for _, row in gdf.iterrows():
            pid = int(row[id_col])
            val = row[value_col]
            if pd.notna(val):
                if pid not in records:
                    records[pid] = {}
                records[pid][channel] = float(val)

    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "id"
    return df


# ═══════════════════════════════════════════════════════════
#  Гиперспектр: агрегация из папки с подпапками
# ═══════════════════════════════════════════════════════════

def _parse_hyper_channel(filename: str, prefix_len: int = 1) -> int | None:
    
    stem = Path(filename).stem
    if not stem.isdigit():
        return None
    channel_str = stem[prefix_len:]
    if not channel_str:
        return None
    return int(channel_str)


def load_hyper_wavelength_map(xlsx_path: str) -> dict[int, float]:
    
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb[wb.sheetnames[0]]
    wl_row = [cell.value for cell in ws[1]]
    wl_map = {}
    for i, val in enumerate(wl_row[5:], start=1):
        if val is not None and isinstance(val, (int, float)):
            wl_map[i] = float(val)
    return wl_map


def load_hyper_date(
    date_folder: str,
    wl_map: dict[int, float],
    prefix_len: int = 1,
    value_col: str = "_mean",
    id_col: str = "id",
    min_valid_bands: int = 50,
) -> tuple[pd.DataFrame, np.ndarray]:
    
    date_folder = Path(date_folder)
    if not date_folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {date_folder}")

    subfolders = sorted([d for d in date_folder.iterdir() if d.is_dir()])
    print(f"  Папка: {date_folder.name}, наборов: {len(subfolders)}")

    if not subfolders:
        raise ValueError(f"Нет подпапок в {date_folder}")

    # {point_id: {channel_num: [list of values from different sets]}}
    all_values = defaultdict(lambda: defaultdict(list))
    sets_with_data = 0

    for si, subfolder in enumerate(subfolders):
        gpkg_files = list(subfolder.glob("*.gpkg"))
        if not gpkg_files:
            continue

        set_has_data = False
        for fpath in gpkg_files:
            ch = _parse_hyper_channel(fpath.name, prefix_len)
            if ch is None or ch not in wl_map:
                continue

            try:
                gdf = gpd.read_file(fpath)
            except Exception:
                continue

            if value_col not in gdf.columns:
                continue

            for _, row in gdf.iterrows():
                pid = int(row[id_col])
                val = row[value_col]
                if pd.notna(val) and val != 0:
                    all_values[pid][ch].append(float(val))
                    set_has_data = True

        if set_has_data:
            sets_with_data += 1

        # Прогресс
        if (si + 1) % 20 == 0 or si == len(subfolders) - 1:
            print(f"    Прочитано {si+1}/{len(subfolders)} наборов "
                  f"({sets_with_data} с данными)")

    # Агрегация: среднее по наборам
    sorted_channels = sorted(wl_map.keys())
    wavelengths = np.array([wl_map[ch] for ch in sorted_channels])
    col_names = [f"{wl_map[ch]:.2f}" for ch in sorted_channels]

    rows = {}
    for pid, channels in all_values.items():
        spectrum = []
        valid_count = 0
        for ch in sorted_channels:
            values = channels.get(ch, [])
            if values:
                spectrum.append(np.mean(values))
                valid_count += 1
            else:
                spectrum.append(np.nan)

        if valid_count >= min_valid_bands:
            rows[pid] = spectrum

    df = pd.DataFrame.from_dict(rows, orient="index", columns=col_names)
    df.index.name = "id"
    df = df.sort_index()

    n_sets_per_point = {
        pid: np.mean([len(channels.get(ch, [])) for ch in sorted_channels if channels.get(ch)])
        for pid, channels in all_values.items()
        if pid in rows
    }
    avg_sets = np.mean(list(n_sets_per_point.values())) if n_sets_per_point else 0

    print(f"  Результат: {len(df)} точек, {len(wavelengths)} каналов, "
          f"среднее {avg_sets:.1f} наборов/точку")

    return df, wavelengths


def load_hyper_from_xlsx(xlsx_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Загружает уже собранный гиперспектр из xlsx (giper_point.xlsx)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb[wb.sheetnames[0]]

    wl_row = [cell.value for cell in ws[1]]
    wavelengths = np.array([
        float(w) for w in wl_row[4:]
        if w is not None and isinstance(w, (int, float))
    ])

    records = {}
    for row in ws.iter_rows(min_row=3, max_row=ws.max_row, values_only=True):
        if row[1] is None or not isinstance(row[1], (int, float)):
            continue
        pid = int(row[1])
        if len(row) > 4 and row[4] == "stdev":
            continue
        spectrum = [
            float(v) if v is not None and isinstance(v, (int, float)) else np.nan
            for v in row[4:4 + len(wavelengths)]
        ]
        if pid not in records:
            records[pid] = []
        records[pid].append(spectrum)

    data = {pid: np.nanmean(specs, axis=0) for pid, specs in records.items()}
    col_names = [f"{w:.2f}" for w in wavelengths]
    df = pd.DataFrame.from_dict(data, orient="index", columns=col_names)
    df.index.name = "id"
    df = df.sort_index()

    print(f"  Гипер xlsx: {len(df)} точек, {len(wavelengths)} каналов "
          f"({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} нм)")
    return df, wavelengths


# ═══════════════════════════════════════════════════════════
#  Химический анализ
# ═══════════════════════════════════════════════════════════

def load_chemistry_doc(
    filepath: str,
    columns: list[str],
    id_offset: int = 0,
) -> pd.DataFrame:
    """Парсит .doc через antiword."""
    result = subprocess.run(
        ["antiword", "-m", "UTF-8.txt", filepath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"antiword error: {result.stderr}")

    parts = result.stdout.split("|")
    values = []
    for p in parts:
        p = p.strip().replace(",", ".")
        if p and re.match(r"^[\d\.<]+", p):
            values.append(p)

    samples = []
    i = 0
    while i < len(values) - 16:
        try:
            sid = int(values[i])
            if 1 <= sid <= 9999:
                row_vals = values[i + 1 : i + 17]
                parsed = [sid - id_offset]
                for v in row_vals:
                    parsed.append(np.nan if v.startswith("<") else float(v))
                samples.append(parsed)
                i += 17
                continue
        except (ValueError, IndexError):
            pass
        i += 1

    df = pd.DataFrame(samples, columns=["id"] + columns)
    return df.set_index("id").sort_index()


def load_chemistry(filepath: str, columns: list[str], id_offset: int = 0) -> pd.DataFrame:
    """Универсальный загрузчик химии."""
    ext = Path(filepath).suffix.lower()
    if ext == ".doc":
        return load_chemistry_doc(filepath, columns, id_offset)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, index_col=0)
        if id_offset:
            df.index = df.index - id_offset
        return df.sort_index()
    elif ext == ".csv":
        return pd.read_csv(filepath, index_col=0).sort_index()
    raise ValueError(f"Неподдерживаемый формат: {ext}")
