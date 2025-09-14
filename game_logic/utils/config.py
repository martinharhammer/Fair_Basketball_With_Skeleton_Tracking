from __future__ import annotations
from pathlib import Path
import json
from typing import Callable, Tuple, Any

def load_config(config_path: str | Path = "config.json") -> Tuple[dict, Callable[[str | Path | None], str]]:
    """Load JSON config and return (config_dict, resolve) where resolve()
    makes any relative path absolute to the config file's directory.
    """
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        C = json.load(f)
    cfg_dir = cfg_path.parent

    def resolve(p: str | Path | None) -> str:
        if p is None or p == "":
            return ""
        p = Path(p)
        return str(p if p.is_absolute() else (cfg_dir / p).resolve())

    return C, resolve

