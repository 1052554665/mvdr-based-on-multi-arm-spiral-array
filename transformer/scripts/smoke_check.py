from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.speech_enhancement import SpeechEnhancementDataset
from src.models.dual_branch_transformer import DualBranchTransformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchTransformer(input_size=(128, 64)).to(device)
    dataset = SpeechEnhancementDataset(data_dir=None, num_samples=2, input_size=(128, 64))
    batch = torch.stack([dataset[0]["noisy"], dataset[1]["noisy"]]).to(device)
    noise = torch.stack([dataset[0]["noise"], dataset[1]["noise"]]).to(device)

    with torch.no_grad():
        enhanced, mask = model(batch, noise)

    print(f"device={device}")
    print(f"enhanced_shape={tuple(enhanced.shape)}")
    print(f"mask_shape={tuple(mask.shape)}")
    print("smoke_check=passed")


if __name__ == "__main__":
    main()
