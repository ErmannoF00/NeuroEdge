from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class BackendSpec:
    name: str
    weight_bits: int = 8
    input_bits: int = 8
    accum_bits: int = 32
    analog: bool = False

class ComputeBackend:
    """Software contract mirroring the RTL/backend boundary."""
    def __init__(self, spec: BackendSpec): self.spec = spec
    def matmul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SRAMNMCBackend(ComputeBackend):
    def __init__(self, spec: BackendSpec | None = None):
        super().__init__(spec or BackendSpec('sram_nmc'))
    def matmul(self, x, w):
        return x @ w

class StatisticalNVMBackend(ComputeBackend):
    """Research abstraction only; not a compact device model."""
    def __init__(self, sigma_rel=0.03, levels=16, seed=1, name='nvm_stat'):
        super().__init__(BackendSpec(name, weight_bits=4, input_bits=8, accum_bits=32, analog=True))
        self.sigma_rel, self.levels, self.seed = sigma_rel, levels, seed
    def map_weights(self, w):
        scale = w.abs().max().clamp_min(1e-9)
        wn = (w/scale).clamp(-1,1)
        q = torch.round((wn+1)*(self.levels-1)/2) * 2/(self.levels-1) - 1
        g = torch.Generator(device=w.device).manual_seed(self.seed)
        noise = torch.randn(q.shape, generator=g, device=w.device, dtype=w.dtype) * self.sigma_rel
        return (q*(1+noise)).clamp(-1,1)*scale
    def matmul(self, x, w):
        return x @ self.map_weights(w)
