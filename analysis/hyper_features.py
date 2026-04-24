"""
analysis/hyper_features.py — гиперспектральные признаки для 02_ml_honest.py.

Pipeline для каждой точки:
  1. Сырой спектр (300 каналов, 390–1031 нм)
  2. Savitzky-Golay сглаживание
  3. Четыре варианта предобработки: raw, SNV, deriv1, deriv1_SNV
  4. Гиперспектральные VI (24 индекса из hyper_indices.py)
  5. Субвыборка каналов (каждый 5-й → 60 каналов) для снижения размерности

Итого ~300 признаков:
  4 предобработки × 60 каналов + 24 VI = 264 признака

Feature selection в 02_ml_honest.py выберет лучшие 50.
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── Предобработка ─────────────────────────────────────────────────────────────

def sg_smooth(spectra: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay сглаживание. Каждая строка = один спектр."""
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(spectra, window_length=window,
                              polyorder=poly, axis=1)
    except Exception:
        return spectra


def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — убирает мультипликативный шум рассеяния."""
    mean = spectra.mean(axis=1, keepdims=True)
    std  = spectra.std(axis=1, keepdims=True) + 1e-10
    return (spectra - mean) / std


def deriv1(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Первая производная по длине волны (конечные разности)."""
    dW = np.diff(wavelengths)
    dS = np.diff(spectra, axis=1)
    # Нормируем на шаг длины волны; добавляем нулевой канал в конец
    deriv = dS / (dW[np.newaxis, :] + 1e-10)
    # Дополняем до исходного числа каналов
    return np.hstack([deriv, deriv[:, -1:]])


def preprocess_spectra(spectra: np.ndarray,
                        wavelengths: np.ndarray,
                        variant: str = "raw",
                        sg_window: int = 11,
                        sg_poly: int = 2) -> np.ndarray:
    """
    Применяет вариант предобработки к матрице спектров.

    Варианты:
      raw        — только SG сглаживание
      snv        — SG + SNV
      deriv1     — SG + 1я производная
      deriv1_snv — SG + 1я производная + SNV
    """
    S = sg_smooth(spectra, window=sg_window, poly=sg_poly)
    if variant == "raw":
        return S
    elif variant == "snv":
        return snv(S)
    elif variant == "deriv1":
        return deriv1(S, wavelengths)
    elif variant == "deriv1_snv":
        return snv(deriv1(S, wavelengths))
    else:
        raise ValueError(f"Неизвестный вариант предобработки: {variant}")


# ── Субвыборка каналов ────────────────────────────────────────────────────────

def subsample_bands(spectra: np.ndarray,
                     wavelengths: np.ndarray,
                     step: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Берёт каждый step-й канал (по умолч. каждый 5-й из 300 = 60 каналов).
    Снижает размерность при сохранении спектральной формы.
    """
    idx = np.arange(0, len(wavelengths), step)
    return spectra[:, idx], wavelengths[idx]


# ── Главная функция ───────────────────────────────────────────────────────────

def build_hyper_features(spectra_df: pd.DataFrame,
                          wavelengths: np.ndarray,
                          variants: list[str] = None,
                          band_step: int = 5,
                          include_indices: bool = True,
                          prefix: str = "") -> pd.DataFrame:
    """
    Строит полный feature matrix из гиперспектральных данных.

    Параметры
    ----------
    spectra_df   : DataFrame (n_points × n_bands), индекс = point_id
    wavelengths  : 1D array длин волн (нм)
    variants     : список вариантов предобработки (по умолч. все 4)
    band_step    : шаг субвыборки каналов
    include_indices: добавлять ли гиперспектральные VI
    prefix       : префикс для имён признаков (например, дата съёмки)

    Возвращает
    ----------
    DataFrame с признаками, индекс = point_id
    """
    if variants is None:
        variants = ["raw", "snv", "deriv1", "deriv1_snv"]

    S = spectra_df.values.astype(np.float64)
    ids = spectra_df.index

    all_features = {}

    # ── 1. Предобработанные + субвыборочные каналы ────────────────────────────
    for var in variants:
        try:
            S_proc = preprocess_spectra(S, wavelengths, variant=var)
            S_sub, wl_sub = subsample_bands(S_proc, wavelengths, step=band_step)
            for i, wl in enumerate(wl_sub):
                col = f"{prefix}hyper_{var}_{wl:.0f}nm"
                all_features[col] = S_sub[:, i]
        except Exception as e:
            print(f"  Предобработка {var}: ошибка {e}")
            continue

    # ── 2. Гиперспектральные VI ───────────────────────────────────────────────
    if include_indices:
        try:
            from analysis.hyper_indices import calculate_hyper_indices
            # Используем snv-предобработку для индексов (стандарт в литературе)
            S_snv = preprocess_spectra(S, wavelengths, variant="snv")
            idx_df = calculate_hyper_indices(S_snv, wavelengths)
            for col in idx_df.columns:
                all_features[f"{prefix}hvi_{col}"] = idx_df[col].values
        except Exception as e:
            print(f"  Hyper VI: ошибка {e}")

    feat_df = pd.DataFrame(all_features, index=ids)

    # Убираем константные признаки и признаки с >50% NaN
    valid_cols = [c for c in feat_df.columns
                  if feat_df[c].std() > 1e-10
                  and feat_df[c].isna().mean() < 0.5]
    feat_df = feat_df[valid_cols]

    wl_min = wavelengths.min()
    wl_max = wavelengths[wavelengths <= 1100].max() if (wavelengths <= 1100).any()              else wavelengths.max()
    n_bands_sub = len(range(0, len(wavelengths), band_step))
    n_vi = feat_df.shape[1] - len(variants) * n_bands_sub
    print(f"  Гиперспектр{' '+prefix if prefix else ''}: "
          f"{feat_df.shape[1]} признаков "
          f"(λ={wl_min:.0f}–{wl_max:.0f} нм, "
          f"{len(variants)} предобработки × {n_bands_sub} каналов + {n_vi} VI)")

    return feat_df


# ── Загрузка для обеих дат ────────────────────────────────────────────────────

def load_hyper_for_all_dates(cfg: dict,
                               variants: list[str] = None,
                               band_step: int = 5) -> dict[str, pd.DataFrame]:
    """
    Загружает гиперспектральные признаки для date1 и date2.

    Возвращает dict: {'date1': DataFrame, 'date2': DataFrame}
    Индекс DataFrame = point_id, столбцы = признаки с префиксом даты.
    """
    from analysis.loaders import load_hyper_date, load_hyper_wavelength_map

    hyper_cfg = cfg.get("paths", {}).get("hyper", {})
    wl_path   = hyper_cfg.get("wavelength_map")
    naming    = cfg.get("hyper_naming", {})

    if not wl_path or not Path(wl_path).exists():
        print("  [hyper] wavelength_map не найден — пропускаем")
        return {}

    print("  Загрузка карты длин волн...")
    wl_map = load_hyper_wavelength_map(wl_path)
    # Оставляем только физически осмысленный диапазон Pika L (390–1050 нм)
    wl_map = {ch: wl for ch, wl in wl_map.items() if 350 <= wl <= 1100}
    wl_vals = sorted(wl_map.values())
    print(f"    Карта длин волн: {len(wl_map)} каналов, "
          f"λ={min(wl_vals):.0f}–{max(wl_vals):.0f} нм")

    result = {}
    date_map = {
        "date1": hyper_cfg.get("date1_folder"),
        "date2": hyper_cfg.get("date2_folder"),
    }

    for dk, folder in date_map.items():
        if not folder or not Path(folder).exists():
            print(f"  [hyper {dk}] папка не найдена: {folder}")
            continue

        print(f"  Загрузка гиперспектра [{dk}] из {Path(folder).name}...")
        try:
            spectra_df, wavelengths = load_hyper_date(
                date_folder=folder,
                wl_map=wl_map,
                prefix_len=naming.get("prefix_length", 1),
                value_col=naming.get("value_column", "_mean"),
                id_col=naming.get("id_column", "id"),
                min_valid_bands=naming.get("min_valid_bands", 50),
            )
            print(f"    {len(spectra_df)} точек, {spectra_df.shape[1]} каналов, "
                  f"λ={wavelengths.min():.0f}–{wavelengths.max():.0f} нм")

            # Строим признаки с префиксом даты
            feat_df = build_hyper_features(
                spectra_df, wavelengths,
                variants=variants,
                band_step=band_step,
                include_indices=True,
                prefix=f"{dk}_",
            )
            result[dk] = feat_df

        except Exception as e:
            import traceback
            print(f"  [hyper {dk}] ошибка: {e}")
            traceback.print_exc()
            continue

    return result
