from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML: pip install pyyaml") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
