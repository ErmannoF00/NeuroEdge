"""Lazy, strict, subject/session-preserving Brain-to-Text 25 HDF5 reader.

No transcriptions or raw neural data are written to result files. The trial
labels are consumed as-is, trimmed by the authoritative seq_len attribute.
No train/validation window mixing or transductive normalization.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

from neuroedge.speech.ctc import BLANK, NUM_CLASSES, ctc_min_length

N_FEATURES = 512

@dataclass(frozen=True)
class TrialRef:
    file: str
    key: str
    bins: int
    label_len: int


def list_labeled_trials(path: str | Path, limit: int = 0) -> list[TrialRef]:
    """Enumerate labeled trials by metadata only; 0=all. Fail on wrong schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit < 0:
        raise ValueError("limit must be >=0")
    refs = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            if "input_features" not in g or "seq_class_ids" not in g:
                raise ValueError(f"{path}:{key}: labeled training trial required")
            shp = g["input_features"].shape
            if len(shp) != 2 or shp[1] != N_FEATURES:
                raise ValueError(f"{key}: expected [T,512], got {shp}")
            bins = int(g.attrs["n_time_steps"])
            if not 0 < bins <= shp[0]:
                raise ValueError(f"{key}: invalid n_time_steps")
            if "seq_len" not in g.attrs:
                raise ValueError(f"{key}: missing seq_len attribute")
            seq_len = int(g.attrs["seq_len"])
            if not 0 < seq_len <= g["seq_class_ids"].size:
                raise ValueError(f"{key}: invalid seq_len={seq_len}")
            refs.append(TrialRef(str(path), str(key), bins, seq_len))
            if limit and len(refs) >= limit:
                break
    if not refs:
        raise ValueError(f"{path}: no labeled trials")
    return refs


def read_trial(ref: TrialRef) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(ref.file, "r") as f:
        g = f[ref.key]
        x = np.asarray(g["input_features"][:ref.bins], dtype=np.float32)
        target = np.asarray(g["seq_class_ids"][:ref.label_len], dtype=np.int64).reshape(-1)
    if x.shape != (ref.bins, N_FEATURES) or not np.isfinite(x).all():
        raise ValueError(f"Invalid neural array in {ref.key}")
    if len(target) != ref.label_len or ((target <= BLANK) | (target >= NUM_CLASSES)).any():
        raise ValueError(f"Invalid phoneme label IDs in {ref.key}: expected 1..{NUM_CLASSES-1}")
    if (ref.bins + 3) // 4 < ctc_min_length(target.tolist()):
        raise ValueError(f"CTC target impossible after stride-4 downsampling: {ref.key}")
    return x, target


def training_channel_stats(refs: list[TrialRef]) -> tuple[np.ndarray, np.ndarray]:
    """Numerically stable moments over training bins only (no validation leakage)."""
    n_total = 0
    mean = np.zeros((N_FEATURES,), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for ref in refs:
        x, _ = read_trial(ref)
        x = x.astype(np.float64)
        n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = np.square(x - batch_mean).sum(axis=0)
        if n_total == 0:
            mean, m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            total = n_total + n
            mean = mean + delta * (n / total)
            m2 = m2 + batch_m2 + delta ** 2 * (n_total * n / total)
        n_total += n
    if n_total < 2:
        raise ValueError("Not enough training frames")
    std = np.sqrt(np.maximum(m2 / n_total, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class SpeechTrials(Dataset):
    def __init__(self, refs: list[TrialRef], mean: np.ndarray, std: np.ndarray):
        self.refs = refs
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (N_FEATURES,) or self.std.shape != (N_FEATURES,) or (self.std <= 0).any():
            raise ValueError("Invalid normalizer")
    def __len__(self):
        return len(self.refs)
    def __getitem__(self, i):
        x, y = read_trial(self.refs[i])
        # Frozen training-only statistics; do not normalize independently per trial.
        x = np.clip((x - self.mean) / self.std, -8, 8)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def collate_ctc(batch):
    """Right-padding only; lengths used in CTC loss and evaluation."""
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    x = torch.zeros((len(xs), int(lengths.max()), N_FEATURES), dtype=torch.float32)
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
    targets = torch.cat(ys).to(dtype=torch.long)
    return x, lengths, targets, target_lengths
