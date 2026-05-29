from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_branch_transformer import DualBranchTransformer
from src.trainers.workflow import build_dataloaders
from src.utils.config import load_config
from src.utils.train_eval import generate_enhancement_report, save_enhanced_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained speech enhancement model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default="experiments/default/best_model.pth", help="Path to the model checkpoint.")
    parser.add_argument("--data-dir", default=None, help="Audio dataset root directory.")
    parser.add_argument("--output-dir", default="enhancement_report", help="Directory for evaluation outputs.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = PROJECT_ROOT

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_config(config_path)

    data_dir = args.data_dir if args.data_dir is not None else config.get("data", {}).get("data_dir")
    if data_dir is not None:
        data_dir = str((project_root / data_dir).resolve()) if not Path(data_dir).is_absolute() else data_dir

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (project_root / checkpoint_path).resolve()

    _, val_loader, test_loader, _, _, _ = build_dataloaders(config, data_dir)
    model_cfg = config.get("model", {})
    device_name = config.get("runtime", {}).get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = DualBranchTransformer(
        in_channels=int(model_cfg.get("in_channels", 2)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        depth=int(model_cfg.get("depth", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        patch_size=tuple(model_cfg.get("patch_size", (16, 16))),
        input_size=tuple(model_cfg.get("input_size", (128, 64))),
        suppression_mode=model_cfg.get("suppression_mode", "frequency_aware"),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    active_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    report_path = generate_enhancement_report(model, active_loader, device, output_dir=str(output_dir), num_samples=args.num_samples)
    saved_count = save_enhanced_audio(model, active_loader, device, output_dir=str(output_dir / "enhanced_outputs"), num_samples=args.num_samples)
    print(f"Report saved to {report_path}")
    print(f"Saved {saved_count} enhanced spectrogram samples to {output_dir / 'enhanced_outputs'}")


if __name__ == "__main__":
    main()
