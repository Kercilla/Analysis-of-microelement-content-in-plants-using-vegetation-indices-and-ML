

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class IndexInfo:
    """Описание вегетативного индекса."""
    name: str
    tier: int                     # 1=основной, 2=ценный, 3=экспериментальный
    formula_str: str              # Формула для документации
    description: str              # Что измеряет
    required_bands: list[str]     # Какие каналы нужны
    func: Callable                # Функция расчёта


# Реестр всех зарегистрированных индексов
INDEX_REGISTRY: dict[str, IndexInfo] = {}


def register_index(
    name: str,
    tier: int,
    formula: str,
    description: str,
    bands: list[str],
):
    """Декоратор для регистрации вегетативного индекса."""
    def decorator(func):
        INDEX_REGISTRY[name] = IndexInfo(
            name=name,
            tier=tier,
            formula_str=formula,
            description=description,
            required_bands=bands,
            func=func,
        )
        return func
    return decorator


def _safe_div(a, b):
    """Поэлементное деление массивов с защитой от 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, np.nan)
    return result


# ════════════════════════════════════════════════════════════
#  TIER 1: Основные индексы
# ════════════════════════════════════════════════════════════

@register_index("NDVI", 1, "(NIR-Red)/(NIR+Red)",
                "Общий показатель вегетации. Насыщается при LAI>3.",
                ["NIR", "Red"])
def ndvi(B, G, R, RE, NIR):
    return _safe_div(NIR - R, NIR + R)


@register_index("NDRE", 1, "(NIR-RE)/(NIR+RE)",
                "Хлорофилл/азот. Лучший предиктор N. Не насыщается.",
                ["NIR", "RedEdge"])
def ndre(B, G, R, RE, NIR):
    return _safe_div(NIR - RE, NIR + RE)


@register_index("GNDVI", 1, "(NIR-Green)/(NIR+Green)",
                "В 5× чувствительнее к хлорофиллу, чем NDVI.",
                ["NIR", "Green"])
def gndvi(B, G, R, RE, NIR):
    return _safe_div(NIR - G, NIR + G)


@register_index("CIre", 1, "NIR/RE - 1",
                "Линейная зависимость с концентрацией хлорофилла.",
                ["NIR", "RedEdge"])
def ci_re(B, G, R, RE, NIR):
    return _safe_div(NIR, RE) - 1


@register_index("CIgreen", 1, "NIR/Green - 1",
                "Хлорофилл через зелёный канал.",
                ["NIR", "Green"])
def ci_green(B, G, R, RE, NIR):
    return _safe_div(NIR, G) - 1


@register_index("OSAVI", 1, "1.16*(NIR-Red)/(NIR+Red+0.16)",
                "Коррекция на фон почвы.",
                ["NIR", "Red"])
def osavi(B, G, R, RE, NIR):
    return _safe_div(1.16 * (NIR - R), NIR + R + 0.16)


@register_index("EVI", 1, "2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)",
                "Для плотного полога. Атмосферная + почвенная коррекция.",
                ["NIR", "Red", "Blue"])
def evi(B, G, R, RE, NIR):
    return _safe_div(2.5 * (NIR - R), NIR + 6 * R - 7.5 * B + 1)


@register_index("EVI2", 1, "2.5*(NIR-Red)/(NIR+2.4*Red+1)",
                "EVI без Blue канала.",
                ["NIR", "Red"])
def evi2(B, G, R, RE, NIR):
    return _safe_div(2.5 * (NIR - R), NIR + 2.4 * R + 1)


@register_index("CCCI", 1, "NDRE/NDVI",
                "Хлорофилл независимо от биомассы. R²=0.97 для N в пшенице.",
                ["NIR", "Red", "RedEdge"])
def ccci(B, G, R, RE, NIR):
    ndvi_val = _safe_div(NIR - R, NIR + R)
    ndre_val = _safe_div(NIR - RE, NIR + RE)
    return _safe_div(ndre_val, ndvi_val)


# ════════════════════════════════════════════════════════════
#  TIER 2: Ценные дополнительные
# ════════════════════════════════════════════════════════════

@register_index("MTCI", 2, "(NIR-RE)/(RE-Red)",
                "Широкий диапазон чувствительности к хлорофиллу.",
                ["NIR", "RedEdge", "Red"])
def mtci(B, G, R, RE, NIR):
    return _safe_div(NIR - RE, RE - R)


@register_index("Datt", 2, "(NIR-RE)/(NIR-Red)",
                "Хлорофилл, отделённый от LAI.",
                ["NIR", "RedEdge", "Red"])
def datt(B, G, R, RE, NIR):
    return _safe_div(NIR - RE, NIR - R)


@register_index("SAVI", 2, "1.5*(NIR-Red)/(NIR+Red+0.5)",
                "Поправка на почву (L=0.5).",
                ["NIR", "Red"])
def savi(B, G, R, RE, NIR):
    return _safe_div((NIR - R), (NIR + R + 0.5)) * 1.5


@register_index("WDRVI", 2, "(0.2*NIR-Red)/(0.2*NIR+Red)",
                "Расширенный динамический диапазон для плотных пологов.",
                ["NIR", "Red"])
def wdrvi(B, G, R, RE, NIR):
    return _safe_div(0.2 * NIR - R, 0.2 * NIR + R)


@register_index("SIPI", 2, "(NIR-Blue)/(NIR-Red)",
                "Каротиноиды/хлорофилл. Ранний стресс.",
                ["NIR", "Blue", "Red"])
def sipi(B, G, R, RE, NIR):
    return _safe_div(NIR - B, NIR - R)


@register_index("PSRI", 2, "(Red-Green)/RE",
                "Старение растений (senescence).",
                ["Red", "Green", "RedEdge"])
def psri(B, G, R, RE, NIR):
    return _safe_div(R - G, RE)


@register_index("ARI", 2, "1/Green - 1/RE",
                "Антоцианы. Индикатор дефицита P.",
                ["Green", "RedEdge"])
def ari(B, G, R, RE, NIR):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((G != 0) & (RE != 0), 1.0 / G - 1.0 / RE, np.nan)


@register_index("TCARI", 2, "3*((RE-Red)-0.2*(RE-Green)*(RE/Red))",
                "Глубина абсорбции хлорофилла.",
                ["RedEdge", "Red", "Green"])
def tcari(B, G, R, RE, NIR):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(R != 0, RE / R, np.nan)
    return 3 * ((RE - R) - 0.2 * (RE - G) * ratio)


@register_index("MCARI", 2, "((RE-Red)-0.2*(RE-Green))*(RE/Red)",
                "Модифицированный коэффициент абсорбции хлорофилла.",
                ["RedEdge", "Red", "Green"])
def mcari(B, G, R, RE, NIR):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(R != 0, RE / R, np.nan)
    return ((RE - R) - 0.2 * (RE - G)) * ratio


@register_index("TCARI_OSAVI", 2, "TCARI/OSAVI",
                "Хлорофилл, очищенный от LAI и почвы.",
                ["NIR", "RedEdge", "Red", "Green"])
def tcari_osavi(B, G, R, RE, NIR):
    t = tcari(B, G, R, RE, NIR)
    o = osavi(B, G, R, RE, NIR)
    return _safe_div(t, o)


@register_index("MCARI_OSAVI", 2, "MCARI/OSAVI",
                "Альтернативный комбинированный хлорофилловый индекс.",
                ["NIR", "RedEdge", "Red", "Green"])
def mcari_osavi(B, G, R, RE, NIR):
    m = mcari(B, G, R, RE, NIR)
    o = osavi(B, G, R, RE, NIR)
    return _safe_div(m, o)


# ════════════════════════════════════════════════════════════
#  TIER 3: Экспериментальные
# ════════════════════════════════════════════════════════════

@register_index("SR", 3, "NIR/Red", "Простой ratio.", ["NIR", "Red"])
def sr(B, G, R, RE, NIR):
    return _safe_div(NIR, R)

@register_index("DVI", 3, "NIR-Red", "Разностный.", ["NIR", "Red"])
def dvi(B, G, R, RE, NIR):
    return NIR - R

@register_index("RDVI", 3, "(NIR-Red)/sqrt(NIR+Red)", "Ренормализованный.", ["NIR", "Red"])
def rdvi(B, G, R, RE, NIR):
    s = NIR + R
    with np.errstate(invalid="ignore"):
        return np.where(s > 0, (NIR - R) / np.sqrt(s), np.nan)

@register_index("RENDVI", 3, "(RE-Red)/(RE+Red)", "Red Edge нормализованный.", ["RedEdge", "Red"])
def rendvi(B, G, R, RE, NIR):
    return _safe_div(RE - R, RE + R)

@register_index("LCI", 3, "(NIR-RE)/(NIR+Red)", "Хлорофилл листа.", ["NIR", "RedEdge", "Red"])
def lci(B, G, R, RE, NIR):
    return _safe_div(NIR - RE, NIR + R)

@register_index("CVI", 3, "(NIR*Red)/Green²", "Хлорофилл к структуре.", ["NIR", "Red", "Green"])
def cvi(B, G, R, RE, NIR):
    return _safe_div(NIR * R, G ** 2)

@register_index("NPCI", 3, "(Red-Blue)/(Red+Blue)", "Пигменты/хлорофилл.", ["Red", "Blue"])
def npci(B, G, R, RE, NIR):
    return _safe_div(R - B, R + B)

@register_index("NGRDI", 3, "(Green-Red)/(Green+Red)", "NUE корреляция.", ["Green", "Red"])
def ngrdi(B, G, R, RE, NIR):
    return _safe_div(G - R, G + R)

@register_index("TGI", 3, "-0.5*(218*(R-G)-108*(R-B))", "Видимый хлорофилл.", ["Red", "Green", "Blue"])
def tgi(B, G, R, RE, NIR):
    return -0.5 * (218 * (R - G) - 108 * (R - B))

@register_index("ARVI", 3, "(NIR-(2R-B))/(NIR+(2R-B))", "Атмосферная коррекция.", ["NIR", "Red", "Blue"])
def arvi(B, G, R, RE, NIR):
    rb = 2 * R - B
    return _safe_div(NIR - rb, NIR + rb)

@register_index("BNDVI", 3, "(NIR-Blue)/(NIR+Blue)", "Blue NDVI.", ["NIR", "Blue"])
def bndvi(B, G, R, RE, NIR):
    return _safe_div(NIR - B, NIR + B)

@register_index("GRNDVI", 3, "(NIR-(G+R))/(NIR+(G+R))", "Green-Red NDVI.", ["NIR", "Green", "Red"])
def grndvi(B, G, R, RE, NIR):
    return _safe_div(NIR - (G + R), NIR + (G + R))

@register_index("MSR", 3, "(NIR/R-1)/sqrt(NIR/R+1)", "Модифицированный SR.", ["NIR", "Red"])
def msr(B, G, R, RE, NIR):
    ratio = _safe_div(NIR, R)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(ratio) & ((ratio + 1) > 0),
                        (ratio - 1) / np.sqrt(ratio + 1), np.nan)

@register_index("NLI", 3, "(NIR²-Red)/(NIR²+Red)", "Нелинейный.", ["NIR", "Red"])
def nli(B, G, R, RE, NIR):
    return _safe_div(NIR ** 2 - R, NIR ** 2 + R)

@register_index("NIRv", 3, "NIR*NDVI", "Proxy GPP.", ["NIR", "Red"])
def nirv(B, G, R, RE, NIR):
    return NIR * _safe_div(NIR - R, NIR + R)

@register_index("MSAVI2", 3, "(2*NIR+1-sqrt((2*NIR+1)²-8*(NIR-R)))/2",
                "Авто-коррекция почвы.", ["NIR", "Red"])
def msavi2(B, G, R, RE, NIR):
    d = (2 * NIR + 1) ** 2 - 8 * (NIR - R)
    with np.errstate(invalid="ignore"):
        return np.where(d >= 0, (2 * NIR + 1 - np.sqrt(d)) / 2, np.nan)

@register_index("VARI", 3, "(G-R)/(G+R-B)", "Видимый атмосферно-устойчивый.", ["Green", "Red", "Blue"])
def vari(B, G, R, RE, NIR):
    return _safe_div(G - R, G + R - B)

@register_index("GLI", 3, "(2G-R-B)/(2G+R+B)", "Зелёный лист.", ["Green", "Red", "Blue"])
def gli(B, G, R, RE, NIR):
    return _safe_div(2 * G - R - B, 2 * G + R + B)

@register_index("SRre", 3, "NIR/RE", "Simple Ratio RE.", ["NIR", "RedEdge"])
def srre(B, G, R, RE, NIR):
    return _safe_div(NIR, RE)

@register_index("RTVIcore", 3, "100*(NIR-RE)-10*(NIR-G)", "Структура+хлорофилл.", ["NIR", "RedEdge", "Green"])
def rtvicore(B, G, R, RE, NIR):
    return 100 * (NIR - RE) - 10 * (NIR - G)

@register_index("CRI700", 3, "1/Blue-1/RE", "Каротиноиды.", ["Blue", "RedEdge"])
def cri700(B, G, R, RE, NIR):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((B != 0) & (RE != 0), 1.0 / B - 1.0 / RE, np.nan)

@register_index("BGI", 3, "Blue/Green", "Blue-Green пигмент.", ["Blue", "Green"])
def bgi(B, G, R, RE, NIR):
    return _safe_div(B, G)

@register_index("RGRI", 3, "Red/Green", "Red-Green ratio.", ["Red", "Green"])
def rgri(B, G, R, RE, NIR):
    return _safe_div(R, G)

@register_index("SRPI", 3, "Blue/Red", "Пигментный ratio.", ["Blue", "Red"])
def srpi(B, G, R, RE, NIR):
    return _safe_div(B, R)


# ════════════════════════════════════════════════════════════
#  Основная функция расчёта
# ════════════════════════════════════════════════════════════

def calculate_indices(
    bands_df: pd.DataFrame,
    tiers: list[int] | None = None,
    indices: list[str] | None = None,
) -> pd.DataFrame:
    
    B = bands_df.get("Blue", pd.Series(0, index=bands_df.index)).values.astype(float)
    G = bands_df.get("Green", pd.Series(0, index=bands_df.index)).values.astype(float)
    R = bands_df.get("Red", pd.Series(0, index=bands_df.index)).values.astype(float)
    RE = bands_df.get("RedEdge", pd.Series(0, index=bands_df.index)).values.astype(float)
    NIR = bands_df.get("NIR", pd.Series(0, index=bands_df.index)).values.astype(float)

    available_bands = set(bands_df.columns)
    results = {}

    # Определяем какие индексы считать
    if indices is not None:
        to_compute = {k: v for k, v in INDEX_REGISTRY.items() if k in indices}
    elif tiers is not None:
        to_compute = {k: v for k, v in INDEX_REGISTRY.items() if v.tier in tiers}
    else:
        to_compute = INDEX_REGISTRY

    for name, info in to_compute.items():
        # Проверяем, что все нужные каналы есть
        if not all(b in available_bands for b in info.required_bands):
            continue

        try:
            values = info.func(B, G, R, RE, NIR)
            results[name] = values
        except Exception as e:
            print(f"  ⚠ Ошибка при расчёте {name}: {e}")
            results[name] = np.full(len(bands_df), np.nan)

    df = pd.DataFrame(results, index=bands_df.index)
    return df


def get_indices_by_tier(tier: int) -> list[str]:
    """Возвращает список индексов данного тира."""
    return [name for name, info in INDEX_REGISTRY.items() if info.tier == tier]


def get_index_info(name: str) -> IndexInfo | None:
    """Возвращает описание индекса."""
    return INDEX_REGISTRY.get(name)
