"""Independent cycle-level *specification* for the streaming classifier protocol.

This does not execute the HDL, and is not a substitute for Icarus simulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SerialClassifierProtocol:
    weights: tuple[int, ...]
    bias: int
    index: int = 0
    accumulated: int = 0
    response_valid: bool = False
    response_score: int = 0

    def __post_init__(self):
        if len(self.weights) != 8:
            raise ValueError('v1 expects exactly 8 weights')
        if any(not -128 <= w <= 127 for w in self.weights):
            raise ValueError('weights must be INT8')

    @property
    def input_ready(self):
        return not self.response_valid

    def clock(self, in_valid=False, feature=0, out_ready=False, reset_n=True):
        old_valid = self.response_valid
        ready = not old_valid or out_ready
        if not reset_n:
            self.index=0
            self.accumulated=0
            self.response_valid=False
            self.response_score=0
            return
        if old_valid and out_ready:
            self.response_valid=False
        if in_valid and ready:
            if not -128 <= feature <= 127:
                raise ValueError('input must be signed INT8')
            total = self.accumulated + feature*self.weights[self.index]
            if self.index == 7:
                score=total+self.bias
                if not -(2**31) <= score < 2**31:
                    raise OverflowError('INT32 ABI score overflow')
                self.response_score=score
                self.response_valid=True
                self.index=0
                self.accumulated=0
            else:
                self.accumulated=total
                self.index+=1
