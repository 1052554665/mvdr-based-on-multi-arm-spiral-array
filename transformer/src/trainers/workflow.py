from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.datasets.speech_enhancement import SpeechEnhancementDataset
from src.losses.speech_enhancement_loss import compute_loss
from src.models.dual_branch_transformer import DualBranchTransformer


def _is_cublas_init_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "CUBLAS_STATUS_NOT_INITIALIZED" in message or "cublas" in message.lower()


def _maybe_fallback_to_cpu(model, optimizer, device, config: Dict[str, Any], exc: RuntimeError):
    runtime_cfg = config.get("runtime", {})
    allow_fallback = bool(runtime_cfg.get("cuda_fallback_to_cpu", True))
    if device.type == "cuda" and allow_fallback and _is_cublas_init_error(exc):
        print("[WARNING] CUDA cuBLAS initialization failed. Falling back to CPU for this run.")
        print("[WARNING] To keep CUDA enabled, reduce train.batch_size / model size or verify CUDA driver+PyTorch compatibility.")
        fallback_device = torch.device("cpu")
        model = model.to(fallback_device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(fallback_device)
        return model, optimizer, fallback_device, True
    return model, optimizer, device, False


def build_datasets(config: Dict[str, Any], data_dir: str | None):
    data_cfg = config.get("data", {})
    split_ratio = tuple(data_cfg.get("split_ratio", (0.7, 0.15, 0.15)))
    model_input_size = tuple(config.get("model", {}).get("input_size", (128, 64)))
    common_kwargs = dict(
        data_dir=data_dir,
        input_size=model_input_size,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        n_fft=int(data_cfg.get("n_fft", 512)),
        hop_length=int(data_cfg.get("hop_length", 256)),
        n_mels=int(data_cfg.get("n_mels", 128)),
        split_ratio=split_ratio,
        seed=int(config.get("train", {}).get("seed", 42)),
    )

    train_dataset = SpeechEnhancementDataset(num_samples=int(data_cfg.get("train_samples", 800)), split="train", **common_kwargs)
    val_dataset = SpeechEnhancementDataset(num_samples=int(data_cfg.get("val_samples", 200)), split="val", **common_kwargs)
    test_dataset = SpeechEnhancementDataset(num_samples=int(data_cfg.get("test_samples", 100)), split="test", **common_kwargs)
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(config: Dict[str, Any], data_dir: str | None):
    train_dataset, val_dataset, test_dataset = build_datasets(config, data_dir)
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train_one_epoch(model, dataloader, optimizer, device, epoch, lambda_mask: float = 0.5):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        x_s = batch["noisy"].to(device)
        x_n = batch["noise"].to(device)
        clean = batch["clean"].to(device)

        enhanced, mask = model(x_s, x_n)
        loss, loss_dict = compute_loss(enhanced, mask, clean, x_s, lambda_mask=lambda_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss_dict['total']:.4f} "
                f"(enhance: {loss_dict['enhance']:.4f}, mask: {loss_dict['mask']:.4f})"
            )

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, device, lambda_mask: float = 0.5):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x_s = batch["noisy"].to(device)
            x_n = batch["noise"].to(device)
            clean = batch["clean"].to(device)

            enhanced, mask = model(x_s, x_n)
            loss, loss_dict = compute_loss(enhanced, mask, clean, x_s, lambda_mask=lambda_mask)
            total_loss += loss_dict["total"]
            num_batches += 1

    if num_batches == 0:
        print("[WARNING] Validation set is empty. Returning 0 loss.")
        return 0.0

    return total_loss / num_batches


def train_model(config: Dict[str, Any], data_dir: str | None = None, output_dir: str | None = None):
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    device_name = config.get("runtime", {}).get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        try:
            # Trigger CUDA context and a tiny GEMM early to fail fast with a clearer error path.
            torch.cuda.current_device()
            _ = torch.empty((1, 1), device=device) @ torch.empty((1, 1), device=device)
        except RuntimeError as exc:
            if _is_cublas_init_error(exc):
                print("[WARNING] CUDA warmup failed before training loop.")
                device = torch.device("cpu")
            else:
                raise

    output_root = Path(output_dir or config.get("runtime", {}).get("output_dir", "experiments/default")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = build_dataloaders(config, data_dir)

    model = DualBranchTransformer(
        in_channels=int(model_cfg.get("in_channels", 2)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        depth=int(model_cfg.get("depth", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        patch_size=tuple(model_cfg.get("patch_size", (16, 16))),
        input_size=tuple(model_cfg.get("input_size", (128, 64))),
        suppression_mode=model_cfg.get("suppression_mode", "frequency_aware"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    num_epochs = int(train_cfg.get("num_epochs", 1))
    lambda_mask = float(train_cfg.get("lambda_mask", 0.5))
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"Using device: {device}")
    print(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    for epoch in range(1, num_epochs + 1):
        print("\n" + "=" * 60)
        print(f"Epoch [{epoch}/{num_epochs}]")
        print("=" * 60)

        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, lambda_mask=lambda_mask)
        except RuntimeError as exc:
            model, optimizer, new_device, switched = _maybe_fallback_to_cpu(model, optimizer, device, config, exc)
            if switched:
                device = new_device
                train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, lambda_mask=lambda_mask)
            else:
                raise
        val_loss = validate(model, val_loader, device, lambda_mask=lambda_mask)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_root / "best_model.pth")
            print(f"Saved best model to {output_root / 'best_model.pth'}")

        if epoch % int(train_cfg.get("checkpoint_interval", 10)) == 0:
            torch.save(checkpoint, output_root / f"checkpoint_epoch_{epoch}.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        output_root / "last_model.pth",
    )

    return model, test_dataset, history, device
