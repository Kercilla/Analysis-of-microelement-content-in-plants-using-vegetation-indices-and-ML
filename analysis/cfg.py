"""
Загрузка config.yaml — единая точка доступа ко всем параметрам.
"""

from pathlib import Path
import yaml


def load_config(config_path: str = None) -> dict:
   
    if config_path:
        p = Path(config_path)
    else:
        p = Path("config.yaml")
        if not p.exists():
            p = Path(__file__).parent.parent / "config.yaml"

    if not p.exists():
        raise FileNotFoundError(f"config.yaml не найден: {p}")

    with open(p) as f:
        cfg = yaml.safe_load(f)

    return cfg


def short_name(cfg: dict, element: str) -> str:
    """Короткое имя элемента из конфига."""
    return cfg["chemistry"]["short_names"].get(element, element)
