from __future__ import annotations
import math
import torch
from torch.utils.data import TensorDataset


def synthetic_bci(samples: int = 512, timesteps: int = 64, channels: int = 16,
                  classes: int = 4, seed: int = 7) -> TensorDataset:
    """Deterministic synthetic neural time series for CI/smoke tests.

    Each class combines a distinct temporal frequency, spatial channel group and
    low-frequency drift. This is *not* a scientific BCI dataset; it only makes the
    repo executable without a network download.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(samples) % classes
    t = torch.linspace(0.0, 1.0, timesteps)
    x = 0.25 * torch.randn(samples, timesteps, channels, generator=g)
    for n in range(samples):
        c = int(y[n])
        freq = 5.0 + 3.0 * c
        wave = torch.sin(2 * math.pi * freq * t)
        start = (c * max(1, channels // classes)) % channels
        width = max(2, channels // 4)
        idx = [(start + k) % channels for k in range(width)]
        x[n, :, idx] += wave[:, None]
        x[n] += (0.08 * c) * torch.cos(2 * math.pi * 0.5 * t)[:, None]
    # per-channel z-score to mimic preprocessing contract
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-5)
    x = (x - mu) / sd
    return TensorDataset(x.float(), y.long())


def delta_events(x: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
    """Convert continuous samples [B,T,C] to ternary delta events {-1,0,+1}."""
    dx = torch.zeros_like(x)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]
    return (dx > threshold).to(x.dtype) - (dx < -threshold).to(x.dtype)
