"""
DocLoom Multi-Modal Thalamic Perception Stem
Handles sensory projection (2D Vision Patches with 2D-RoPE, Audio Mel-Spectrograms)
and the Dual-Manifold 'Mirror World' Cross-Binding Bridge.
Strictly compliant with Rule 1 (<= 700 lines).
"""
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_AI.model.config import DocLoomModelConfig


class Rotary2DEmbedding(nn.Module):
    """
    2D Rotary Position Embeddings (2D-RoPE) for spatial patch grids.
    Encodes exact (x, y) grid coordinates to solve multi-column reading order,
    diagonal text, and diagram-to-caption geometric relationships.
    """
    def __init__(self, dim: int, max_grid_size: int = 128):
        super().__init__()
        assert dim % 4 == 0, "2D RoPE dimension must be divisible by 4"
        self.dim = dim
        self.half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, h_grid: int, w_grid: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cos and sin tensors shaped (1, h_grid * w_grid, dim)."""
        y_pos = torch.arange(h_grid, device=device, dtype=self.inv_freq.dtype)
        x_pos = torch.arange(w_grid, device=device, dtype=self.inv_freq.dtype)

        y_freqs = torch.einsum("i,j->ij", y_pos, self.inv_freq)  # (h, half_dim/2)
        x_freqs = torch.einsum("i,j->ij", x_pos, self.inv_freq)  # (w, half_dim/2)

        # Expand across 2D grid
        y_grid = y_freqs.unsqueeze(1).repeat(1, w_grid, 1)  # (h, w, half_dim/2)
        x_grid = x_freqs.unsqueeze(0).repeat(h_grid, 1, 1)  # (h, w, half_dim/2)

        freqs = torch.cat([y_grid, x_grid], dim=-1)  # (h, w, half_dim)
        freqs = freqs.view(-1, self.half_dim)         # (h*w, half_dim)
        freqs = torch.cat([freqs, freqs], dim=-1)     # (h*w, dim)

        return freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)


def apply_2d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary 2D positional rotation to patch token embeddings."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


class VisualPatchStem(nn.Module):
    """
    Native 2D Visual Patch Ingestion Stem.
    Decomposes images (PDF pages, charts, photos) into 16x16 pixel patches,
    projects into Cortex latent dimension (n_embd), and applies 2D-RoPE.
    """
    def __init__(self, cfg: DocLoomModelConfig, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        self.patch_size = patch_size
        self.n_embd = cfg.n_embd

        # 2D Patch Convolution Stem (Converts 16x16 pixels straight to n_embd)
        self.patch_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cfg.n_embd,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.rope_2d = Rotary2DEmbedding(cfg.n_embd)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        pixel_values: (batch, 3, H, W) normalized image tensors.
        Returns:
            patch_tokens: (batch, num_patches, n_embd)
            grid_shape: (H_grid, W_grid)
        """
        b, c, h, w = pixel_values.shape
        h_grid = h // self.patch_size
        w_grid = w // self.patch_size

        # (b, n_embd, h_grid, w_grid) -> (b, h_grid*w_grid, n_embd)
        x = self.patch_proj(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        cos, sin = self.rope_2d(h_grid, w_grid, x.device)
        x = apply_2d_rope(x, cos, sin)
        return x, (h_grid, w_grid)


class AudioSpectralStem(nn.Module):
    """
    Native Audio Spectral Ingestion Stem.
    Transduces continuous mel-spectrogram acoustic frequencies into Cortex latent tokens.
    Prepared for subsequent audio/speech perception capabilities.
    """
    def __init__(self, cfg: DocLoomModelConfig, n_mels: int = 80, kernel_size: int = 16):
        super().__init__()
        self.cfg = cfg
        self.n_embd = cfg.n_embd

        self.spec_conv = nn.Conv1d(
            in_channels=n_mels,
            out_channels=cfg.n_embd,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 4
        )
        self.norm = nn.LayerNorm(cfg.n_embd)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        mel_spectrogram: (batch, n_mels, time_frames)
        Returns: (batch, audio_tokens, n_embd)
        """
        x = self.spec_conv(mel_spectrogram).transpose(1, 2)
        return self.norm(x)


class DualManifoldBridge(nn.Module):
    """
    The 'Mirror World' Cross-Binding Invariant.
    Binds visual perceptual topology (Side A) with symbolic text propositions (Side B)
    into a single unified crystalline coordinate WITHOUT duplicating data.
    """
    def __init__(self, cfg: DocLoomModelConfig, crystal_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.crystal_dim = crystal_dim

        # Projections to common crystalline phase coordinate space
        self.visual_to_crystal = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, crystal_dim)
        )
        self.semantic_to_crystal = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, crystal_dim)
        )
        self.temperature = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def project_visual_crystal(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pools visual tokens and outputs normalized crystal coordinate."""
        v_pooled = visual_tokens.mean(dim=1)  # (batch, n_embd)
        v_coord = self.visual_to_crystal(v_pooled)
        return F.normalize(v_coord, p=2, dim=-1)

    def project_semantic_crystal(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pools text tokens and outputs normalized crystal coordinate."""
        t_pooled = text_tokens.mean(dim=1)  # (batch, n_embd)
        t_coord = self.semantic_to_crystal(t_pooled)
        return F.normalize(t_coord, p=2, dim=-1)

    def compute_mirror_invariant_loss(self, v_coord: torch.Tensor, t_coord: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive alignment ensuring visual representation and text description
        resonate at the exact same physical coordinates in the crystal manifold.
        """
        temp = self.temperature.exp().clamp(max=100.0)
        sim = torch.matmul(v_coord, t_coord.T) * temp
        labels = torch.arange(v_coord.shape[0], device=v_coord.device)
        loss_v = F.cross_entropy(sim, labels)
        loss_t = F.cross_entropy(sim.T, labels)
        return (loss_v + loss_t) / 2.0
