"""Lightweight, trainable per-channel affine recalibration for frozen speech decoder.

Hardware feasibility: 512 gain and 512 offset values, eventually stored in SRAM.
NOT a claim of on-chip learning: optimization uses PyTorch backpropagation on a host.
"""
from __future__ import annotations

import torch
from torch import nn


class ChannelAffineDecoder(nn.Module):
    def __init__(self, base: nn.Module, features: int = 512, clip: float = 8.0):
        super().__init__()
        if features <= 0 or clip <= 0:
            raise ValueError('features and clip must be positive')
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.gain = nn.Parameter(torch.ones(features))
        self.offset = nn.Parameter(torch.zeros(features))
        self.clip = clip

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep pretrained decoder deterministic while calibrating.
        self.base.eval()
        return self

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != self.gain.numel():
            raise ValueError(f'Expected [B,T,{self.gain.numel()}]')
        return self.base(torch.clamp(x * self.gain + self.offset, -self.clip, self.clip))

    def regularization(self):
        return (self.gain - 1).square().mean() + self.offset.square().mean()

    def calibration_state(self) -> dict:
        return {'gain': self.gain.detach().cpu().clone(),
                'offset': self.offset.detach().cpu().clone(),
                'clip': self.clip}
