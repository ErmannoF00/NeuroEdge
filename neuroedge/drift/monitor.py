"""Frozen-normalizer neural activity excursions, policy replay, and audit.

No label/phoneme/transcription is accessed. This is a *monitor*, not an
established neural drift detector. An update_request is not a parameter write.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import h5py
from neuroedge.control.budget import BudgetController

N_CHANNELS = 512
SAMPLES_PER_BIN = 512


def quantize_neural(features: np.ndarray, mean: np.ndarray, std: np.ndarray,
                    fraction_bits: int) -> np.ndarray:
    """Exact deployment input rounding: numpy ties-to-even, signed INT8 clamp."""
    x = np.asarray(features, dtype=np.float32)
    mu = np.asarray(mean, dtype=np.float32)
    sigma = np.asarray(std, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != N_CHANNELS or x.shape[0] < 1:
        raise ValueError('expected non-empty [bins,512] neural features')
    if mu.shape != (N_CHANNELS,) or sigma.shape != (N_CHANNELS,):
        raise ValueError('mean/std must be [512]')
    if not (1 <= fraction_bits <= 5):
        raise ValueError('fraction bits must be between 1 and 5 for 2-sigma monitor')
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('nonfinite values or nonpositive std')
    # Matches SpeechTrials training-frozen normalization and input export.
    y = np.clip((x - mu) / sigma, -8., 8.)
    return np.clip(np.rint(y * (1 << fraction_bits)), -128, 127).astype(np.int8)


def excursion_scores(q: np.ndarray, magnitude_threshold: int) -> np.ndarray:
    """Number of INT8 features with |value| >= integer threshold for each bin."""
    a = np.asarray(q)
    if a.ndim != 2 or a.shape[1] != N_CHANNELS or a.dtype != np.int8:
        raise ValueError('expected signed INT8 [bins,512]')
    if not (1 <= magnitude_threshold <= 128):
        raise ValueError('invalid magnitude threshold')
    # Upcast BEFORE abs: abs(int8(-128)) overflows.
    return np.sum(np.abs(a.astype(np.int16)) >= magnitude_threshold, axis=1,
                  dtype=np.int32).astype(np.uint16)


def iter_hdf5_trials(path: Path, *, max_trials: int = 0) -> Iterator[tuple[str, np.ndarray]]:
    if max_trials < 0:
        raise ValueError('max_trials must be non-negative')
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, 'r') as f:
        n = 0
        for key in sorted(f.keys()):
            group = f[key]
            if 'input_features' not in group:
                raise ValueError(f'{path}:{key}: missing input_features')
            d = group['input_features']
            if d.ndim != 2 or d.shape[1] != N_CHANNELS:
                raise ValueError(f'{path}:{key}: incorrect input shape')
            count = int(group.attrs['n_time_steps'])
            if not 0 < count <= d.shape[0]:
                raise ValueError(f'{path}:{key}: invalid n_time_steps')
            x = np.asarray(d[:count], dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f'{path}:{key}: nonfinite inputs')
            yield str(key), x
            n += 1
            if max_trials and n >= max_trials:
                break
        if not n:
            raise ValueError(f'{path}: no trials')


def calibrated_thresholds(train_scores: np.ndarray) -> tuple[int, int]:
    """Use calibration train split only. p90 reset / p99 assert thresholds."""
    scores = np.asarray(train_scores, dtype=np.int32)
    if scores.ndim != 1 or len(scores) < 10 or (scores < 0).any() or (scores > N_CHANNELS).any():
        raise ValueError('invalid training score distribution')
    low = int(np.ceil(np.percentile(scores, 90.0)))
    high = int(np.ceil(np.percentile(scores, 99.0)))
    if low >= N_CHANNELS:
        low = N_CHANNELS - 1
    high = max(low + 1, min(N_CHANNELS, high))
    return low, high


@dataclass(frozen=True)
class ActivitySummary:
    trials: int
    bins: int
    mean_excursions: float
    p95_excursions: float
    excursion_bins_ge_high_fraction: float
    active_bin_fraction: float
    requested_actions: int
    suppressed_actions: int
    requested_actions_per_1000_bins: float
    trial_requests: tuple[int, ...]
    trial_mean_excursions: tuple[float, ...]


def evaluate_trials(path: Path, mean: np.ndarray, std: np.ndarray, *,
                    fraction_bits: int, magnitude_threshold: int,
                    low_threshold: int, high_threshold: int,
                    max_trials: int = 0, policy_kwargs: dict | None = None) -> ActivitySummary:
    """Replay each trial with a fresh controller to avoid inventing gaps between trials."""
    if not (0 <= low_threshold < high_threshold <= N_CHANNELS):
        raise ValueError('invalid hysteresis thresholds')
    all_scores: list[np.ndarray] = []
    reqs = []
    means = []
    total_active = total_req = total_supp = total_bins = 0
    for _, x in iter_hdf5_trials(path, max_trials=max_trials):
        scores = excursion_scores(quantize_neural(x, mean, std, fraction_bits), magnitude_threshold)
        ctrl = BudgetController(high_threshold=high_threshold,
                                low_threshold=low_threshold, **(policy_kwargs or {}))
        req = 0
        active = 0
        for score in scores.tolist():
            a, r, _, _ = ctrl.sample(int(score))
            active += a
            req += r
        total_bins += len(scores)
        total_active += active
        total_req += req
        total_supp += ctrl.suppressed
        all_scores.append(scores)
        reqs.append(req)
        means.append(float(scores.mean()))
    merged = np.concatenate(all_scores)
    return ActivitySummary(
        trials=len(reqs), bins=total_bins,
        mean_excursions=float(merged.mean()),
        p95_excursions=float(np.percentile(merged, 95.0)),
        excursion_bins_ge_high_fraction=float(np.mean(merged >= high_threshold)),
        active_bin_fraction=total_active / total_bins,
        requested_actions=total_req,
        suppressed_actions=total_supp,
        requested_actions_per_1000_bins=1000.0 * total_req / total_bins,
        trial_requests=tuple(reqs),
        trial_mean_excursions=tuple(means),
    )


def trial_bootstrap_difference(reference_trial_means: tuple[float, ...],
                               shift_trial_means: tuple[float, ...],
                               seed: int, repeats: int = 2000) -> tuple[float, float]:
    """Independent-session trial-cluster bootstrap; not a subject-population CI."""
    if repeats < 100:
        raise ValueError('bootstrap requires at least 100 replicates')
    a = np.asarray(reference_trial_means, dtype=np.float64)
    b = np.asarray(shift_trial_means, dtype=np.float64)
    if not len(a) or not len(b):
        raise ValueError('empty trial means')
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        values[i] = rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean()
    return tuple(float(z) for z in np.quantile(values, [0.025, 0.975]))
