"""Deterministic, non-learned model and inputs for public logic tests only.

These parameters do not represent a neural model trained on human data.
"""
from __future__ import annotations
import numpy as np
from .fixed_decoder import FixedParameters, TOTAL_WEIGHTS, TOTAL_META, WEIGHT_OFFSETS, INPUTS, HIDDEN


def synthetic_parameters() -> FixedParameters:
    weights = np.zeros(TOTAL_WEIGHTS, dtype=np.int8)
    for c in range(4):
        weights[WEIGHT_OFFSETS['project'] + c * INPUTS + c] = 1
        weights[WEIGHT_OFFSETS['dw1'] + c * 5 + 3] = 1
        weights[WEIGHT_OFFSETS['dw1'] + c * 5 + 4] = 1
        weights[WEIGHT_OFFSETS['pw1'] + c * HIDDEN + c] = 1
        weights[WEIGHT_OFFSETS['dw2'] + c * 5 + 4] = 1
        weights[WEIGHT_OFFSETS['pw2'] + c * HIDDEN + c] = 1
        weights[WEIGHT_OFFSETS['head'] + c * HIDDEN + c] = 1
    return FixedParameters(weights, np.zeros(TOTAL_META, dtype=np.int32),
                           np.zeros(TOTAL_META, dtype=np.uint8))


def synthetic_features(bins: int = 32) -> np.ndarray:
    if bins <= 0 or bins % 4:
        raise ValueError('bins must be a positive multiple of four')
    data = np.zeros((bins, INPUTS), dtype=np.int8)
    for t in range(bins):
        for c in range(4):
            data[t, c] = 1 + ((t + 2 * c) % 7)
    return data
