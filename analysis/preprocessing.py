"""
Предобработка спектральных данных.

- Savitzky-Golay сглаживание
- Первая/вторая производная
- Standard Normal Variate (SNV)
- Min-Max / Standard нормализация
- Удаление шумных диапазонов
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def sg_smooth(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(spectra, window_length, polyorder, axis=1)


def sg_derivative(
    spectra: np.ndarray,
    deriv: int = 1,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    
    if window_length % 2 == 0:
        window_length += 1
    polyorder = max(polyorder, deriv)
    return savgol_filter(spectra, window_length, polyorder, deriv=deriv, axis=1)


def snv(spectra: np.ndarray) -> np.ndarray:
    
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (spectra - mean) / std


def continuum_removal(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
   
    from scipy.spatial import ConvexHull

    result = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        y = spectra[i]
        points = np.column_stack([wavelengths, y])

        try:
            hull = ConvexHull(points)
            # Верхняя огибающая — интерполяция верхних точек hull
            hull_vertices = sorted(set(hull.vertices))
            upper = np.interp(wavelengths, wavelengths[hull_vertices], y[hull_vertices])
            upper[upper == 0] = 1.0
            result[i] = y / upper
        except Exception:
            result[i] = y

    return result


def remove_noisy_bands(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    noise_ranges: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    if noise_ranges is None:
        noise_ranges = [(925, 975)]

    mask = np.ones(len(wavelengths), dtype=bool)
    for start, end in noise_ranges:
        mask &= ~((wavelengths >= start) & (wavelengths <= end))

    return spectra[:, mask], wavelengths[mask]


def preprocess_pipeline(
    spectra: np.ndarray,
    wavelengths: np.ndarray | None = None,
    steps: list[str] | None = None,
    sg_window: int = 11,
    sg_poly: int = 2,
    noise_ranges: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    
    if steps is None:
        steps = ["smooth", "deriv1", "snv"]

    wl = wavelengths.copy() if wavelengths is not None else None
    X = spectra.copy().astype(float)

    for step in steps:
        if step == "smooth":
            X = sg_smooth(X, sg_window, sg_poly)

        elif step == "deriv1":
            X = sg_derivative(X, deriv=1, window_length=sg_window, polyorder=sg_poly)

        elif step == "deriv2":
            X = sg_derivative(X, deriv=2, window_length=sg_window, polyorder=max(sg_poly, 3))

        elif step == "snv":
            X = snv(X)

        elif step == "remove_noise":
            if wl is not None:
                X, wl = remove_noisy_bands(X, wl, noise_ranges)

        elif step == "continuum":
            if wl is not None:
                X = continuum_removal(X, wl)

        else:
            raise ValueError(f"Неизвестный шаг предобработки: {step}")

    # Замена NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, wl
