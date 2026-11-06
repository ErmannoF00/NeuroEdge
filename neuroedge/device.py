from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class DeviceConfig:
    states: int
    read_sigma_rel: float
    write_sigma_rel: float
    drift: float = 0.0

FEFET_EXAMPLE = DeviceConfig(states=16, read_sigma_rel=0.02, write_sigma_rel=0.03, drift=0.002)
RRAM_EXAMPLE  = DeviceConfig(states=32, read_sigma_rel=0.04, write_sigma_rel=0.06, drift=0.01)


def map_weight_to_device(w: torch.Tensor, cfg: DeviceConfig, seed: int = 0):
    """Illustrative device-aware weight mapping.

    Parameters are deliberately *example values*, not a claim about a particular
    fabricated technology. Replace them with measured distributions/compact-model
    data for publishable hardware conclusions.
    """
    g = torch.Generator(device=w.device).manual_seed(seed)
    wmax = w.abs().max().clamp_min(1e-12)
    z = (w / wmax + 1.0) * 0.5
    levels = max(2, cfg.states)
    zq = torch.round(z * (levels - 1)) / (levels - 1)
    write_noise = torch.randn(zq.shape, generator=g, device=zq.device, dtype=zq.dtype) * cfg.write_sigma_rel
    stored = torch.clamp(zq * (1.0 + write_noise), 0.0, 1.0)
    stored = stored * (1.0 - cfg.drift)
    read_noise = torch.randn(stored.shape, generator=g, device=stored.device, dtype=stored.dtype) * cfg.read_sigma_rel
    read = torch.clamp(stored * (1.0 + read_noise), 0.0, 1.0)
    return (2.0 * read - 1.0) * wmax


def perturb_model(model, cfg: DeviceConfig, seed=0):
    clone = type(model)(**_infer_ctor(model))
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        k = 0
        for p in clone.parameters():
            if p.ndim >= 2:
                p.copy_(map_weight_to_device(p, cfg, seed + k)); k += 1
    return clone


def _infer_ctor(model):
    from .models import TinyQNN, TinySNN, DiagonalSSM
    if isinstance(model, TinyQNN):
        return dict(channels=model.temporal.in_channels, hidden=model.temporal.out_channels, classes=model.head.out_features)
    if isinstance(model, TinySNN):
        return dict(channels=model.inp.in_features, hidden=model.inp.out_features, classes=model.out.out_features, beta=model.beta)
    if isinstance(model, DiagonalSSM):
        return dict(channels=model.inp.in_features, state=model.inp.out_features, classes=model.out.out_features)
    raise TypeError(type(model))
