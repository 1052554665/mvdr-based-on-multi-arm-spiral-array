from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.workflow import train_model
from src.utils.config import load_config
from src.utils.train_eval import save_enhanced_audio, generate_enhancement_report


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


def _run_single_category(config, data_dir, output_dir, project_root):
    """Train and export enhanced outputs for a single data directory."""
    model, test_dataset, _, device = train_model(config, data_dir=data_dir, output_dir=output_dir)

    if test_dataset is not None and len(test_dataset) > 0:
        train_cfg = config.get("train", {})
        batch_size = int(train_cfg.get("batch_size", 4))
        num_workers = int(train_cfg.get("num_workers", 0))
        pin_memory = device.type == "cuda"
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        output_root = Path(output_dir or config.get("runtime", {}).get("output_dir", "experiments/default")).resolve()
        saved_count = save_enhanced_audio(
            model,
            test_loader,
            device,
            output_dir=str(output_root / "enhanced_outputs"),
            num_samples=len(test_dataset),
        )
        print(f"Saved {saved_count} enhanced spectrogram samples to {output_root / 'enhanced_outputs'}")
        try:
            report_path = generate_enhancement_report(
                model, test_loader, device,
                output_dir=str(output_root / "enhancement_report"),
                num_samples=min(10, len(test_dataset)),
            )
            print(f"Evaluation report saved to {report_path}")
        except Exception as exc:
            print(f"[WARNING] Failed to generate enhancement report: {exc}")
    else:
        print("[WARNING] Test dataset is empty, skipping enhanced output export.")

    return model, device


def main():
    args = parse_args()
    project_root = PROJECT_ROOT

    config_path = resolve_config_path(args.config, project_root)
    config = load_config(config_path)

    data_cfg = config.get("data", {})
    categories = data_cfg.get("categories", [])
    base_data_dir = args.data_dir if args.data_dir is not None else data_cfg.get("data_dir")

    base_output_dir = args.output_dir if args.output_dir is not None else config.get("runtime", {}).get("output_dir")
    if base_output_dir is not None and not Path(base_output_dir).is_absolute():
        base_output_dir = str((project_root / base_output_dir).resolve())

    if categories and base_data_dir:
        # ── Multi-category mode ──────────────────────────────────────
        for category in categories:
            print(f"\n{'#' * 60}")
            print(f"#  Category: {category}")
            print(f"{'#' * 60}")

            cat_data_dir = str((project_root / base_data_dir / category).resolve())
            cat_output_dir = str(Path(base_output_dir) / category) if base_output_dir else None

            _run_single_category(config, cat_data_dir, cat_output_dir, project_root)
    else:
        # ── Single-category mode (original behavior) ─────────────────
        data_dir = base_data_dir
        if data_dir is not None and not Path(data_dir).is_absolute():
            data_dir = str((project_root / data_dir).resolve())

        _run_single_category(config, data_dir, base_output_dir, project_root)


if __name__ == "__main__":
    main()
