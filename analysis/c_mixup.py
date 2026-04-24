"""
analysis/c_mixup.py — C-Mixup аугментация для регрессии.

Тонкая обёртка над spatial_cv.c_mixup для удобного импорта.
Подробная документация — в analysis/spatial_cv.py.
"""
from analysis.spatial_cv import c_mixup, c_mixup_with_coords   # noqa: F401

__all__ = ["c_mixup", "c_mixup_with_coords"]
