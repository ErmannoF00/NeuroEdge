"""Architectural bit-plane crossbar emulator, NOT a measured memristor/ADC.

Two signed input phases (positive, negative) and 8 binary weight planes
produce ideal column sums. Weight bit 7 is negative (two's complement).
Optional quantization acts on the integer column sum, before recombination;
it is NOT a calibrated electrical transfer function.
"""
from dataclasses import dataclass
from neuroedge.memory.near_memory_reference import int8_dot

@dataclass(frozen=True)
class CrossbarEstimate:
    dot: int
    column_conversions: int
    clipped_conversions: int
    quantized_conversions: int
    exact_reference: int


def bitplane_dot(x, w, *, column_step=1, column_limit=None):
    reference,_ = int8_dot(x,w)
    if not isinstance(column_step,int) or column_step < 1:
        raise ValueError('column_step must be a positive integer')
    if column_limit is not None and (not isinstance(column_limit,int) or column_limit < 0):
        raise ValueError('column_limit must be nonnegative')
    result=0; clipped=0; quantized=0
    for bit in range(8):
        pos=sum(max(a,0) for a,b in zip(x,w) if (b & (1<<bit)))
        neg=sum(max(-a,0) for a,b in zip(x,w) if (b & (1<<bit)))
        def digitize(val):
            nonlocal clipped,quantized
            if column_limit is not None and val>column_limit:
                val=column_limit; clipped+=1
            q=((val + column_step//2)//column_step)*column_step
            if q!=val: quantized+=1
            return q
        sign=-1 if bit==7 else 1
        result += sign*(1<<bit)*(digitize(pos)-digitize(neg))
    return CrossbarEstimate(result,16,clipped,quantized,reference)
