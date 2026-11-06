"""Leakage-controlled, label-free neural-session calibration and paired PER analysis.

All coefficients are estimated from the NEW SESSION calibration TRAIN split only.
No phoneme labels or validation values enter the coefficient calculations.
Clinical validation and on-chip learning are explicitly out of scope.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import h5py
from neuroedge.speech.dataset import TrialRef, N_FEATURES


@dataclass(frozen=True)
class AffineParameters:
    gain: np.ndarray
    offset: np.ndarray
    n_bins: int
    rule: str


def moments_from_calibration(refs: Sequence[TrialRef], src_mean: np.ndarray,
                             src_std: np.ndarray, clip: float = 8.0):
    """Moments of frozen-source-normalized calibration features; no labels read."""
    src_mean = np.asarray(src_mean, np.float64)
    src_std = np.asarray(src_std, np.float64)
    if not refs or src_mean.shape != (N_FEATURES,) or src_std.shape != (N_FEATURES,):
        raise ValueError('Nonempty calibration refs and exactly 512 source stats required')
    if np.any(src_std <= 0) or clip <= 0:
        raise ValueError('Invalid source std or clipping')
    first = np.zeros(N_FEATURES, dtype=np.float64)
    second = np.zeros(N_FEATURES, dtype=np.float64)
    n_bins = 0
    for ref in refs:
        with h5py.File(ref.file, 'r') as h5:
            features = np.asarray(h5[ref.key]['input_features'][:ref.bins], dtype=np.float64)
        if features.shape != (ref.bins, N_FEATURES) or not np.isfinite(features).all():
            raise ValueError('Invalid calibration neural features')
        z = np.clip((features - src_mean) / src_std, -clip, clip)
        first += z.sum(axis=0)
        second += (z*z).sum(axis=0)
        n_bins += z.shape[0]
    if n_bins < 2:
        raise ValueError('Need at least two neural bins')
    mean = first / n_bins
    std = np.sqrt(np.maximum(second / n_bins - mean*mean, 1.e-4))
    return mean, std, n_bins


def calibrate_rules(mean: np.ndarray, std: np.ndarray, n_bins: int,
                    gain_limit: float = 0.25, offset_limit: float = 2.0):
    """Prespecified bounded methods, NOT chosen using held-out validation."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
        raise ValueError('512 channels required')
    if gain_limit < 0 or gain_limit >= 1 or offset_limit < 0 or np.any(std <= 0):
        raise ValueError('Invalid gain/offset bounds')
    gain0 = np.ones(N_FEATURES, np.float32)
    offset0 = np.zeros(N_FEATURES, np.float32)
    offset = np.clip(-mean, -offset_limit, offset_limit).astype(np.float32)
    gain = np.clip(1 / std, 1-gain_limit, 1+gain_limit).astype(np.float32)
    offset_aff = np.clip(-mean*gain, -offset_limit, offset_limit).astype(np.float32)
    return {
        'fixed': AffineParameters(gain0, offset0, n_bins, 'no_adaptation'),
        'unsup_offset': AffineParameters(gain0.copy(), offset, n_bins, 'mean_only_no_labels'),
        'unsup_affine': AffineParameters(gain, offset_aff, n_bins, 'mean_variance_no_labels'),
    }


def paired_per_bootstrap(reference: Sequence[int], edits_before: Sequence[int],
                         edits_after: Sequence[int], seed: int = 2026,
                         iterations: int = 1000) -> dict:
    """Resample WHOLE trials, preserving pairs and phoneme-count weights."""
    n = len(reference)
    if n == 0 or not (n == len(edits_before) == len(edits_after)) or iterations <= 0:
        raise ValueError('Nonempty equal trial arrays and positive iterations required')
    refs = np.asarray(reference, dtype=np.int64)
    bef = np.asarray(edits_before, dtype=np.int64)
    aft = np.asarray(edits_after, dtype=np.int64)
    if np.any(refs <= 0) or np.any(bef < 0) or np.any(aft < 0):
        raise ValueError('Invalid edit counts')
    denom = refs.sum()
    before = float(bef.sum()/denom)
    after = float(aft.sum()/denom)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, np.float64)
    for k in range(iterations):
        ix = rng.integers(0, n, n)
        deltas[k] = (aft[ix].sum() - bef[ix].sum()) / refs[ix].sum()
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        'before_per': before, 'after_per': after,
        'after_minus_before': after-before,
        'delta_95pct_trial_bootstrap_ci': [float(lo), float(hi)],
        'bootstrap_iterations': iterations,
        'n_trials': n,
        'scope': 'paired trial bootstrap, not participant population generalization',
    }
