"""Causal, stride-4, hardware-directed 512-input phoneme-CTC student.

The convolution kernels are left-padded only: output at bin t cannot depend on
future features. Intermediate activations are deterministic for streaming RTL.
This is an original compact *baseline*, not the official RNN or a measured ASIC.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from neuroedge.speech.ctc import NUM_CLASSES


class CausalDepthwise(nn.Module):
    def __init__(self, channels: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel, stride=stride, groups=channels, bias=True)
    def forward(self, x):
        return self.conv(F.pad(x, (self.kernel - 1, 0)))


class CompactCTC(nn.Module):
    def __init__(self, hidden: int = 64, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.hidden, self.n_classes = hidden, n_classes
        self.project = nn.Linear(512, hidden)
        self.dw1 = CausalDepthwise(hidden)
        self.pw1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dw2 = CausalDepthwise(hidden)
        self.pw2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.head = nn.Linear(hidden, n_classes)
    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return (lengths + 3) // 4
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, 512] -> [B, ceil(T/4), vocab]
        if x.ndim != 3 or x.shape[-1] != 512:
            raise ValueError("Expected [B,T,512] inputs")
        z = F.relu(self.project(x)).transpose(1, 2)
        z = F.relu(self.pw1(self.dw1(z)))
        z = F.relu(self.pw2(self.dw2(z)))
        return self.head(z.transpose(1, 2))
