from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 2, embed_dim: int = 128, patch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        batch_size, embed_dim, freq_patch, time_patch = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (freq_patch, time_patch)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 4, dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, suppression_mode: str = "adaptive"):
        super().__init__()
        self.suppression_mode = suppression_mode
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        if suppression_mode == "frequency_aware":
            self.horizontal_detector = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
                nn.BatchNorm1d(dim),
                nn.Sigmoid(),
            )
            self.vertical_detector = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.BatchNorm1d(dim),
                nn.Sigmoid(),
            )
            self.stripe_enhancer = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.amplifier = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

    def forward(self, z_s: torch.Tensor, z_n: torch.Tensor) -> torch.Tensor:
        z_cross, _ = self.cross_attn(z_s, z_n, z_n, need_weights=False)
        z_diff = z_s - z_cross

        if self.suppression_mode == "frequency_aware":
            z_cross_t = z_cross.transpose(1, 2)
            h_response = self.horizontal_detector(z_cross_t)
            v_response = self.vertical_detector(z_cross_t)
            stripe_score = torch.cat([h_response, 1 - v_response], dim=1).transpose(1, 2)
            freq_attention = self.stripe_enhancer(stripe_score)
            z_noise_enhanced = z_cross * freq_attention
            gate = self.gate(torch.cat([z_s, z_noise_enhanced, z_diff], dim=-1))
            z_noise_amplified = self.amplifier(z_noise_enhanced)
            z = z_s - gate * (z_noise_amplified + z_cross)
        elif self.suppression_mode == "aggressive":
            gate = self.gate(torch.cat([z_s, z_cross, z_diff], dim=-1))
            z_noise_strong = self.amplifier(z_cross)
            z = (z_s - gate * z_noise_strong) + 0.5 * z_diff
        else:
            gate = self.gate(torch.cat([z_s, z_cross, z_diff], dim=-1))
            z = z_s - gate * z_cross

        return self.norm(z)


class Decoder(nn.Module):
    def __init__(self, dim: int, out_channels: int = 1, patch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_channels),
        )
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        freq_patch, time_patch = shape
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(batch_size, 1, freq_patch, time_patch)
        x = self.upsample(x)
        return torch.sigmoid(x)


class DualBranchTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (16, 16),
        input_size: tuple[int, int] = (128, 64),
        suppression_mode: str = "frequency_aware",
    ):
        super().__init__()

        if input_size[0] % patch_size[0] != 0 or input_size[1] % patch_size[1] != 0:
            raise ValueError("input_size must be divisible by patch_size in both dimensions.")

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.encoder = TransformerEncoder(depth, embed_dim, num_heads)
        self.fusion = CrossAttentionFusion(embed_dim, num_heads, suppression_mode=suppression_mode)
        self.decoder = Decoder(embed_dim, patch_size=patch_size)

    def forward(self, x_s: torch.Tensor, x_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_s, shape = self.patch_embed(x_s)
        z_n, _ = self.patch_embed(x_n)
        z_s = z_s + self.pos_embed[:, : z_s.size(1), :]
        z_n = z_n + self.pos_embed[:, : z_n.size(1), :]
        z_s = self.encoder(z_s)
        z_n = self.encoder(z_n)
        z = self.fusion(z_s, z_n)
        mask = self.decoder(z, shape)
        enhanced = mask * x_s[:, :1, :, :]
        return enhanced, mask


def build_model(config: Dict[str, Any]) -> DualBranchTransformer:
    return DualBranchTransformer(
        in_channels=config.get("in_channels", 2),
        embed_dim=config.get("embed_dim", 128),
        depth=config.get("depth", 4),
        num_heads=config.get("num_heads", 8),
        patch_size=tuple(config.get("patch_size", (16, 16))),
        input_size=tuple(config.get("input_size", (128, 64))),
        suppression_mode=config.get("suppression_mode", "frequency_aware"),
    )
