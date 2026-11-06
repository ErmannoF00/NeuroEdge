"""Exact, bank-partitioned 512x64 projection against existing fixed-point model.

This is a *software functional simulator* of row-local SRAM/MAC banks; all
weight reads and MACs are counted and projection compute-cycle lower bounds
are reported. It is NOT a synthesized 64-bank SRAM, a CIM macro, or a cycle-
accurate model of the remaining decoder.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, FixedParameters, INPUTS, HIDDEN

ALLOWED_BANKS = (1, 2, 4, 8, 16, 32, 64)

@dataclass
class ProjectionCounters:
    bins: int = 0
    macs: int = 0
    weight_reads: int = 0
    feature_reads: int = 0
    # Two cycles/MAC follows the separately verified sequential 1RW dot tile.
    compute_only_cycles: int = 0


def exact_banked_projection(features: np.ndarray, p: FixedParameters, *, banks: int) -> np.ndarray:
    """Requantize per OUTPUT ROW, never per partial sum or bank chunk."""
    if banks not in ALLOWED_BANKS:
        raise ValueError('bank count must divide 64 and be 1,2,4,8,16,32,64')
    x = np.asarray(features)
    if x.dtype != np.int8 or x.shape != (INPUTS,):
        raise ValueError('input must be 512 signed INT8 features')
    w = p.row_weights('project')
    bias, shifts = p.layer_metadata('project')
    out = np.empty(HIDDEN, dtype=np.int8)
    # Physical interpretation: one independent MAC per bank, with ceil(64/B)
    # sequential output rows processed by each bank (not a timing simulation).
    for bank in range(banks):
        for row in range(bank, HIDDEN, banks):
            acc = int(w[row].astype(np.int64) @ x.astype(np.int64))
            q = (acc >> int(shifts[row])) + int(bias[row])
            out[row] = np.int8(max(0, min(127, q)))
    return out


class BankedStreamingDecoder(FixedStreamingDecoder):
    """Drop-in integer decoder: project uses banked SRAM reference, all other
    layers, causal histories and frame cadence remain the original implementation.
    """
    def __init__(self, p: FixedParameters, *, banks: int = 64):
        if banks not in ALLOWED_BANKS:
            raise ValueError('unsupported bank count')
        super().__init__(p)
        self.banks = banks
        self.projection_counters = ProjectionCounters()

    def _eval(self, name, x, head=False):
        if name != 'project':
            return super()._eval(name, x, head=head)
        y = exact_banked_projection(x, self.p, banks=self.banks)
        self.saturation_hits['project'] += int(np.count_nonzero(y == 127))
        self.layer_outputs['project'] += HIDDEN
        c = self.projection_counters
        c.bins += 1
        c.macs += INPUTS * HIDDEN
        c.weight_reads += INPUTS * HIDDEN
        c.feature_reads += INPUTS  # Broadcast input into B row-local lanes
        c.compute_only_cycles += (HIDDEN // self.banks) * INPUTS * 2
        return y
