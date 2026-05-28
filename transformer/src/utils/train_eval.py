from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_speech_quality(model, dataloader, device, num_samples: int = 50):
    try:
        from pesq import pesq  # type: ignore
        from pystoi import stoi  # type: ignore
    except ImportError:
        pesq = None
        stoi = None

    model.eval()
    metrics = {"pesq": [], "stoi": [], "snr_before": [], "snr_after": [], "snr_improvement": [], "mse": []}
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            x_s = batch["noisy"].to(device)
            x_n = batch["noise"].to(device)
            clean_spec = batch["clean"].to(device)
            enhanced_spec, _ = model(x_s, x_n)

            for index in range(x_s.size(0)):
                if sample_count >= num_samples:
                    break

                mse = F.mse_loss(enhanced_spec[index], clean_spec[index, :1, :, :]).item()
                metrics["mse"].append(mse)

                noisy_mag = x_s[index, :1, :, :]
                clean_mag = clean_spec[index, :1, :, :]
                enhanced_mag = enhanced_spec[index]

                snr_before = 10 * np.log10((clean_mag**2).mean().item() / (((noisy_mag - clean_mag) ** 2).mean().item() + 1e-10))
                snr_after = 10 * np.log10((clean_mag**2).mean().item() / (((enhanced_mag - clean_mag) ** 2).mean().item() + 1e-10))
                metrics["snr_before"].append(snr_before)
                metrics["snr_after"].append(snr_after)
                metrics["snr_improvement"].append(snr_after - snr_before)

                sample_count += 1

    avg_metrics = {
        "MSE": float(np.mean(metrics["mse"])) if metrics["mse"] else 0.0,
        "SNR_before (dB)": float(np.mean(metrics["snr_before"])) if metrics["snr_before"] else 0.0,
        "SNR_after (dB)": float(np.mean(metrics["snr_after"])) if metrics["snr_after"] else 0.0,
        "SNR_Improvement (dB)": float(np.mean(metrics["snr_improvement"])) if metrics["snr_improvement"] else 0.0,
    }
    return avg_metrics, metrics


def visualize_spectrogram_comparison(
    model,
    dataloader,
    device,
    output_dir: str = "spectrogram_comparison",
    num_samples: int = 10,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Please install matplotlib: pip install matplotlib") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    vis_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if vis_count >= num_samples:
                break

            x_s = batch["noisy"].to(device)
            x_n = batch["noise"].to(device)
            clean_spec = batch["clean"].to(device)
            enhanced_spec, mask = model(x_s, x_n)

            for index in range(x_s.size(0)):
                if vis_count >= num_samples:
                    break

                clean_mag = clean_spec[index, 0].cpu().numpy()
                noisy_mag = x_s[index, 0].cpu().numpy()
                enhanced_mag = enhanced_spec[index, 0].cpu().numpy()
                mask_val = mask[index, 0].cpu().numpy()

                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                vmin = min(clean_mag.min(), noisy_mag.min(), enhanced_mag.min())
                vmax = max(clean_mag.max(), noisy_mag.max(), enhanced_mag.max())

                images = [
                    (axes[0, 0], clean_mag, "Clean Speech Spectrogram", "inferno", vmin, vmax),
                    (axes[0, 1], noisy_mag, "Noisy Speech Spectrogram", "inferno", vmin, vmax),
                    (axes[1, 0], enhanced_mag, "Enhanced Speech Spectrogram", "inferno", vmin, vmax),
                    (axes[1, 1], mask_val, "Predicted Mask", "viridis", 0, 1),
                ]
                for axis, image, title, cmap, lower, upper in images:
                    rendered = axis.imshow(image, aspect="auto", origin="lower", cmap=cmap, vmin=lower, vmax=upper)
                    axis.set_title(title, fontsize=14, fontweight="bold")
                    axis.set_xlabel("Time Frame")
                    axis.set_ylabel("Mel Frequency Bin")
                    plt.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

                snr_before = 10 * np.log10((clean_mag**2).mean() / (((noisy_mag - clean_mag) ** 2).mean() + 1e-10))
                snr_after = 10 * np.log10((clean_mag**2).mean() / (((enhanced_mag - clean_mag) ** 2).mean() + 1e-10))
                snr_improvement = snr_after - snr_before
                fig.suptitle(
                    f"Sample {vis_count + 1} | SNR: {snr_before:.2f}dB → {snr_after:.2f}dB (Improvement: +{snr_improvement:.2f}dB)",
                    fontsize=16,
                    fontweight="bold",
                    y=1.02,
                )
                plt.tight_layout()
                plt.savefig(output_path / f"comparison_{vis_count + 1:04d}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                vis_count += 1

    return vis_count


def generate_enhancement_report(model, dataloader, device, output_dir: str = "enhancement_report", num_samples: int = 10):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    spectrogram_dir = output_path / "spectrograms"

    vis_error = None
    metrics_error = None

    try:
        vis_count = visualize_spectrogram_comparison(model, dataloader, device, output_dir=str(spectrogram_dir), num_samples=min(10, num_samples))
    except Exception as exc:
        vis_count = 0
        vis_error = str(exc)

    try:
        avg_metrics, detailed_metrics = evaluate_speech_quality(model, dataloader, device, num_samples=num_samples)
    except Exception as exc:
        avg_metrics = {}
        detailed_metrics = {"mse": [], "snr_before": [], "snr_after": [], "snr_improvement": []}
        metrics_error = str(exc)

    report_path = output_path / "report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Speech Enhancement Evaluation Report\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Output directory: {output_path}\n")
        handle.write(f"Number of samples requested: {num_samples}\n")
        handle.write(f"Generated spectrograms: {vis_count}\n\n")
        if vis_error:
            handle.write(f"[WARNING] Spectrogram visualization skipped: {vis_error}\n")
        if metrics_error:
            handle.write(f"[WARNING] Metric evaluation failed: {metrics_error}\n")
        if avg_metrics:
            handle.write("Average Metrics:\n")
            handle.write("-" * 60 + "\n")
            for key, value in avg_metrics.items():
                handle.write(f"  {key:30s}: {value:.4f}\n")
            handle.write("-" * 60 + "\n\n")
        handle.write("Per-sample Metrics:\n")
        handle.write("-" * 60 + "\n")
        if not detailed_metrics.get("mse"):
            handle.write("No per-sample metrics available.\n")
        else:
            for index in range(len(detailed_metrics["mse"])):
                handle.write(
                    f"Sample {index + 1:3d}: MSE={detailed_metrics['mse'][index]:.4f}, "
                    f"SNR_before={detailed_metrics['snr_before'][index]:.2f}dB, "
                    f"SNR_after={detailed_metrics['snr_after'][index]:.2f}dB, "
                    f"Improvement={detailed_metrics['snr_improvement'][index]:.2f}dB\n"
                )

    return str(report_path)


def save_enhanced_audio(model, dataloader, device, output_dir: str = "enhanced_output", num_samples: int = 10):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if saved_count >= num_samples:
                break

            x_s = batch["noisy"].to(device)
            x_n = batch["noise"].to(device)
            enhanced_spec, mask = model(x_s, x_n)

            for index in range(x_s.size(0)):
                if saved_count >= num_samples:
                    break
                np.savez(
                    output_path / f"enhanced_{saved_count + 1:04d}.npz",
                    enhanced=enhanced_spec[index].cpu().numpy(),
                    mask=mask[index].cpu().numpy(),
                    noisy=x_s[index].cpu().numpy(),
                )
                saved_count += 1

    return saved_count
