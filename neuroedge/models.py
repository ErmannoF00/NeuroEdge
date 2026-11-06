from __future__ import annotations
import torch
from torch import nn

class TinyQNN(nn.Module):
    def __init__(self, channels=16, hidden=32, classes=4):
        super().__init__()
        self.temporal = nn.Conv1d(channels, hidden, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, classes)
    def forward(self, x):
        h = self.temporal(x.transpose(1,2)).transpose(1,2)
        h = torch.relu(self.norm(h))
        return self.head(h.mean(dim=1))

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + x.abs()).pow(2)

spike = SurrogateSpike.apply

class TinySNN(nn.Module):
    def __init__(self, channels=16, hidden=32, classes=4, beta=0.90):
        super().__init__()
        self.inp = nn.Linear(channels, hidden)
        self.out = nn.Linear(hidden, classes)
        self.beta = beta
    def forward(self, x):
        mem = torch.zeros(x.size(0), self.inp.out_features, device=x.device)
        count = torch.zeros_like(mem)
        for t in range(x.size(1)):
            cur = self.inp(x[:,t])
            mem = self.beta * mem + cur
            spk = spike(mem - 1.0)
            mem = mem * (1.0 - spk.detach())
            count = count + spk
        return self.out(count / x.size(1))

class DiagonalSSM(nn.Module):
    """Hardware-friendly diagonal state-space encoder, not an exact Mamba2 implementation."""
    def __init__(self, channels=16, state=32, classes=4):
        super().__init__()
        self.inp = nn.Linear(channels, state)
        self.logit_a = nn.Parameter(torch.full((state,), 2.0))
        self.b = nn.Parameter(torch.ones(state))
        self.out = nn.Linear(state, classes)
    def forward(self, x):
        h = torch.zeros(x.size(0), self.inp.out_features, device=x.device)
        a = torch.sigmoid(self.logit_a) * 0.995
        for t in range(x.size(1)):
            u = torch.tanh(self.inp(x[:,t]))
            h = a * h + self.b * u
        return self.out(h)

class MaskedSSM(nn.Module):
    def __init__(self, channels=16, state=32):
        super().__init__()
        self.encoder = DiagonalSSM(channels, state, classes=state)
        self.reconstruct = nn.Linear(state, channels)
    def forward(self, x):
        # expose final representation through DiagonalSSM internals
        h = torch.zeros(x.size(0), self.encoder.inp.out_features, device=x.device)
        a = torch.sigmoid(self.encoder.logit_a) * 0.995
        for t in range(x.size(1)):
            u = torch.tanh(self.encoder.inp(x[:,t]))
            h = a * h + self.encoder.b * u
        return self.reconstruct(h)
