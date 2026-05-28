from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.workflow import train_model
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the dual-branch transformer for speech enhancement.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--data-dir", default=None, help="Audio dataset root directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and artifacts.")
    return parser.parse_args()


def resolve_config_path(config_arg: str, project_root: Path) -> Path:
    raw_path = Path(config_arg).expanduser()
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        (Path.cwd() / raw_path),
        (project_root / raw_path),
    ]
    if raw_path.parent == Path("."):
        candidates.append(project_root / "configs" / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked_paths = "\n".join(f"- {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        "Config file not found. Checked:\n"
        f"{checked_paths}"
    )


def main():
    args = parse_args()
    project_root = PROJECT_ROOT

    config_path = resolve_config_path(args.config, project_root)
    config = load_config(config_path)

    data_dir = args.data_dir if args.data_dir is not None else config.get("data", {}).get("data_dir")
    if data_dir is not None:
        data_dir = str((project_root / data_dir).resolve()) if not Path(data_dir).is_absolute() else data_dir

    output_dir = args.output_dir if args.output_dir is not None else config.get("runtime", {}).get("output_dir")
    if output_dir is not None:
        output_dir = str((project_root / output_dir).resolve()) if not Path(output_dir).is_absolute() else output_dir

    train_model(config, data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
