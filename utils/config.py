from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config.yaml not found at {p.resolve()}")
    with open(p) as f:
        return yaml.safe_load(f)
