from pathlib import Path
import yaml

def load_config(path=None):
    if path:
        p = Path(path)
    else:
        p = Path("config.yaml")
        if not p.exists():
            p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"config.yaml не найден: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)

def short_name(cfg, element):
    return cfg["chemistry"]["short_names"].get(element, element)
