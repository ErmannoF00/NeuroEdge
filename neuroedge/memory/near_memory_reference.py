"""Bit-exact int8 dot product and analytical cost counters for the near-memory tile.

Counters describe explicit transactions, NOT silicon energy or measured time.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class DotCounters:
    macs: int
    weight_reads: int
    input_handshakes: int
    minimum_compute_cycles: int

def int8_dot(x, w):
    if len(x) != len(w) or not x:
        raise ValueError('nonempty matching vectors required')
    if any(not isinstance(v, int) or v < -128 or v > 127 for v in (*x, *w)):
        raise ValueError('signed INT8 operands required')
    if len(x) > 512:
        raise ValueError('max 512 products; 32-bit accumulator safety bound')
    result = sum(a * b for a,b in zip(x,w))
    return result, DotCounters(len(x),len(x),len(x),2*len(x))
