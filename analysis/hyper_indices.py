

import numpy as np
import pandas as pd


def _get_band(spectra: np.ndarray, wavelengths: np.ndarray, target_nm: float) -> np.ndarray:
    """Извлекает ближайший канал к заданной длине волны."""
    idx = np.argmin(np.abs(wavelengths - target_nm))
    return spectra[:, idx]


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


# ═══════════════════════════════════════════════════════════
#  Реестр гиперспектральных индексов
# ═══════════════════════════════════════════════════════════

HYPER_INDEX_REGISTRY = {}


def _reg(name, wl_needed, group, formula, description):
    """Регистрация индекса."""
    def decorator(func):
        HYPER_INDEX_REGISTRY[name] = {
            "func": func,
            "wavelengths": wl_needed,
            "group": group,
            "formula": formula,
            "description": description,
        }
        return func
    return decorator


# ── Группа 1: Хлорофилл / Азот ───────────────────────────

@_reg("mND705", [750, 705, 445], "chlorophyll",
      "(R750-R705)/(R750+R705-2*R445)",
      "Модифицированный ND705 — хлорофилл, устойчив к фону")
def mnd705(S, W):
    return _safe_div(
        _get_band(S, W, 750) - _get_band(S, W, 705),
        _get_band(S, W, 750) + _get_band(S, W, 705) - 2 * _get_band(S, W, 445),
    )

@_reg("VOG1", [740, 720], "chlorophyll",
      "R740/R720",
      "Vogelmann Red Edge Index 1 — хлорофилл")
def vog1(S, W):
    return _safe_div(_get_band(S, W, 740), _get_band(S, W, 720))

@_reg("VOG2", [734, 747, 715, 726], "chlorophyll",
      "(R734-R747)/(R715+R726)",
      "Vogelmann Red Edge Index 2")
def vog2(S, W):
    return _safe_div(
        _get_band(S, W, 734) - _get_band(S, W, 747),
        _get_band(S, W, 715) + _get_band(S, W, 726),
    )

@_reg("REP", [670, 680, 700, 710, 720, 730, 740, 750, 760], "chlorophyll",
      "λ где dR/dλ максимален в 680-760нм",
      "Red Edge Position — точка перегиба красного края, прямой индикатор хлорофилла/N")
def rep(S, W):
    # Ищем длины волн в диапазоне 680-760
    mask = (W >= 680) & (W <= 760)
    if mask.sum() < 3:
        return np.full(S.shape[0], np.nan)
    wl_re = W[mask]
    S_re = S[:, mask]
    # Первая производная
    dR = np.diff(S_re, axis=1)
    dW = np.diff(wl_re)
    deriv = dR / dW[np.newaxis, :]
    # Позиция максимума производной
    max_idx = np.argmax(deriv, axis=1)
    # Интерполяция для субпиксельной точности
    rep_vals = wl_re[:-1][max_idx] + dW[max_idx] / 2
    return rep_vals

@_reg("CHL_RE", [750, 710], "chlorophyll",
      "R750/R710 - 1",
      "Chlorophyll Red Edge — линейная зависимость с Cab")
def chl_re(S, W):
    return _safe_div(_get_band(S, W, 750), _get_band(S, W, 710)) - 1

@_reg("DD", [749, 720, 701, 672], "chlorophyll",
      "(R749-R720)-(R701-R672)",
      "Double Difference — двойная разность, хлорофилл")
def dd(S, W):
    return (_get_band(S, W, 749) - _get_band(S, W, 720)) - \
           (_get_band(S, W, 701) - _get_band(S, W, 672))

@_reg("MCARI_narrow", [700, 670, 550], "chlorophyll",
      "((R700-R670)-0.2*(R700-R550))*(R700/R670)",
      "MCARI узкополосный — хлорофилл, чувствителен к LAI")
def mcari_narrow(S, W):
    R700 = _get_band(S, W, 700)
    R670 = _get_band(S, W, 670)
    R550 = _get_band(S, W, 550)
    return ((R700 - R670) - 0.2 * (R700 - R550)) * _safe_div(R700, R670)

@_reg("TCARI_narrow", [700, 670, 550], "chlorophyll",
      "3*((R700-R670)-0.2*(R700-R550)*(R700/R670))",
      "TCARI узкополосный — глубина абсорбции хлорофилла")
def tcari_narrow(S, W):
    R700 = _get_band(S, W, 700)
    R670 = _get_band(S, W, 670)
    R550 = _get_band(S, W, 550)
    return 3 * ((R700 - R670) - 0.2 * (R700 - R550) * _safe_div(R700, R670))

@_reg("OSAVI_narrow", [800, 670], "chlorophyll",
      "(1+0.16)*(R800-R670)/(R800+R670+0.16)",
      "OSAVI узкополосный")
def osavi_narrow(S, W):
    R800 = _get_band(S, W, 800)
    R670 = _get_band(S, W, 670)
    return _safe_div(1.16 * (R800 - R670), R800 + R670 + 0.16)

@_reg("TCARI_OSAVI_narrow", [700, 670, 550, 800], "chlorophyll",
      "TCARI/OSAVI (узкополосный)",
      "Отношение — хлорофилл очищенный от LAI и почвы")
def tcari_osavi_narrow(S, W):
    t = tcari_narrow(S, W)
    o = osavi_narrow(S, W)
    return _safe_div(t, o)

@_reg("ZM", [750, 710], "chlorophyll",
      "R750/R710",
      "Zarco-Tejada & Miller — хлорофилл")
def zm(S, W):
    return _safe_div(_get_band(S, W, 750), _get_band(S, W, 710))

@_reg("GI", [554, 677], "chlorophyll",
      "R554/R677",
      "Greenness Index — обратно пропорционален хлорофиллу")
def gi(S, W):
    return _safe_div(_get_band(S, W, 554), _get_band(S, W, 677))


# ── Группа 2: Каротиноиды / Стресс ───────────────────────

@_reg("PRI", [531, 570], "stress",
      "(R531-R570)/(R531+R570)",
      "Photochemical Reflectance Index — ксантофилловый цикл, фотосинтетическая эффективность")
def pri(S, W):
    return _safe_div(
        _get_band(S, W, 531) - _get_band(S, W, 570),
        _get_band(S, W, 531) + _get_band(S, W, 570),
    )

@_reg("SIPI_narrow", [800, 445, 680], "stress",
      "(R800-R445)/(R800-R680)",
      "SIPI узкополосный — каротиноиды/хлорофилл, ранний стресс")
def sipi_narrow(S, W):
    return _safe_div(
        _get_band(S, W, 800) - _get_band(S, W, 445),
        _get_band(S, W, 800) - _get_band(S, W, 680),
    )

@_reg("CRI550", [510, 550], "stress",
      "1/R510 - 1/R550",
      "Carotenoid Reflectance Index 550 — каротиноиды")
def cri550(S, W):
    R510 = _get_band(S, W, 510)
    R550 = _get_band(S, W, 550)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((R510 != 0) & (R550 != 0), 1.0/R510 - 1.0/R550, np.nan)

@_reg("ARI_narrow", [550, 700], "stress",
      "1/R550 - 1/R700",
      "Anthocyanin Reflectance Index — антоцианы, стресс, дефицит P")
def ari_narrow(S, W):
    R550 = _get_band(S, W, 550)
    R700 = _get_band(S, W, 700)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((R550 != 0) & (R700 != 0), 1.0/R550 - 1.0/R700, np.nan)

@_reg("PSRI_narrow", [678, 500, 750], "stress",
      "(R678-R500)/R750",
      "PSRI узкополосный — senescence, старение")
def psri_narrow(S, W):
    return _safe_div(
        _get_band(S, W, 678) - _get_band(S, W, 500),
        _get_band(S, W, 750),
    )


# ── Группа 3: Вода / Структура листа ─────────────────────

@_reg("WBI", [900, 970], "water",
      "R900/R970",
      "Water Band Index — содержание воды в листе")
def wbi(S, W):
    return _safe_div(_get_band(S, W, 900), _get_band(S, W, 970))

@_reg("NDWI_hyper", [860, 1020], "water",
      "(R860-R1020)/(R860+R1020)",
      "NDWI гиперспектральный (ограничен до 1031нм)")
def ndwi_hyper(S, W):
    return _safe_div(
        _get_band(S, W, 860) - _get_band(S, W, 1020),
        _get_band(S, W, 860) + _get_band(S, W, 1020),
    )

@_reg("MSI", [819, 1020], "water",
      "R819/R1020",
      "Moisture Stress Index")
def msi(S, W):
    return _safe_div(_get_band(S, W, 819), _get_band(S, W, 1020))


# ── Группа 4: Оптимизированные для микроэлементов ────────

@_reg("R_Fe", [700, 670, 550, 750], "micronutrient",
      "(R700-R670)/(R750-R550)",
      "Ratio для Fe — хлороз от дефицита железа")
def r_fe(S, W):
    return _safe_div(
        _get_band(S, W, 700) - _get_band(S, W, 670),
        _get_band(S, W, 750) - _get_band(S, W, 550),
    )

@_reg("R_Mn", [741, 720], "micronutrient",
      "(R741-R720)/(R741+R720)",
      "ND для Mn — пик корреляции из коррелограммы")
def r_mn(S, W):
    return _safe_div(
        _get_band(S, W, 741) - _get_band(S, W, 720),
        _get_band(S, W, 741) + _get_band(S, W, 720),
    )

@_reg("R_Zn", [903, 860], "micronutrient",
      "(R903-R860)/(R903+R860)",
      "ND для Zn — пик из коррелограммы")
def r_zn(S, W):
    return _safe_div(
        _get_band(S, W, 903) - _get_band(S, W, 860),
        _get_band(S, W, 903) + _get_band(S, W, 860),
    )

@_reg("R_Ca", [798, 750], "micronutrient",
      "R798/R750",
      "Ratio для Ca — пик из deriv1_snv коррелограммы")
def r_ca(S, W):
    return _safe_div(_get_band(S, W, 798), _get_band(S, W, 750))


# ═══════════════════════════════════════════════════════════
#  Расчёт индексов
# ═══════════════════════════════════════════════════════════

def calculate_hyper_indices(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    indices: list[str] | None = None,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    
    wl_range = (wavelengths.min(), wavelengths.max())

    to_compute = {}
    for name, info in HYPER_INDEX_REGISTRY.items():
        if indices is not None and name not in indices:
            continue
        if groups is not None and info["group"] not in groups:
            continue
        # Проверяем что нужные длины волн в диапазоне
        if all(wl_range[0] <= wl <= wl_range[1] for wl in info["wavelengths"]):
            to_compute[name] = info

    results = {}
    for name, info in to_compute.items():
        try:
            values = info["func"](spectra, wavelengths)
            results[name] = values
        except Exception as e:
            results[name] = np.full(spectra.shape[0], np.nan)

    return pd.DataFrame(results)


def list_hyper_indices() -> pd.DataFrame:
    """Справочная таблица всех гиперспектральных индексов."""
    rows = []
    for name, info in HYPER_INDEX_REGISTRY.items():
        rows.append({
            "name": name,
            "group": info["group"],
            "formula": info["formula"],
            "description": info["description"],
            "wavelengths": info["wavelengths"],
        })
    return pd.DataFrame(rows)
