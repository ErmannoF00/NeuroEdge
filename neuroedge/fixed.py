from __future__ import annotations
import torch


def symmetric_quantize(x: torch.Tensor, bits: int = 8):
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().max().clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax-1, qmax).to(torch.int32)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor | float):
    return q.float() * scale


def qmul_acc(a: torch.Tensor, b: torch.Tensor, acc_bits: int = 32):
    """Integer MAC helper for bit-accurate reference experiments."""
    acc = (a.to(torch.int64) * b.to(torch.int64)).sum(dim=-1)
    lo = -(1 << (acc_bits - 1))
    hi = (1 << (acc_bits - 1)) - 1
    return torch.clamp(acc, lo, hi)
