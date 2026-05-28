from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(
    enhanced: torch.Tensor,
    mask: torch.Tensor,
    clean_target: torch.Tensor,
    noisy_input: torch.Tensor,
    lambda_mask: float = 0.5,
):
    noisy_magnitude = noisy_input[:, :1, :, :]
    clean_magnitude = clean_target[:, :1, :, :]

    mask_loss = F.mse_loss(mask * noisy_magnitude, clean_magnitude)
    enhance_loss = F.mse_loss(enhanced, clean_magnitude)
    sparsity_loss = torch.mean(torch.abs(mask))

    total_loss = enhance_loss + lambda_mask * mask_loss + 0.01 * sparsity_loss
    loss_dict = {
        "total": total_loss.item(),
        "enhance": enhance_loss.item(),
        "mask": mask_loss.item(),
        "sparsity": sparsity_loss.item(),
    }
    return total_loss, loss_dict
