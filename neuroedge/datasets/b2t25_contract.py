"""Small, original contract validator for the public Brain-to-Text '25 HDF5 data.

This does not copy upstream model/training code. It follows the public dataset
schema documented by Neuroprosthetics-Lab/nejm-brain-to-text.

Install optional dependency: pip install h5py
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


N_FEATURES = 512
BIN_MS = 20


@dataclass(frozen=True)
class TrialHeader:
    source: str
    key: str
    n_time_steps: int
    feature_shape: tuple[int, ...]
    has_phoneme_labels: bool
    has_transcription: bool
    session: str | None
    block_num: str | None
    trial_num: str | None


def _scalar_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return _scalar_str(value.item())
            return ",".join(_scalar_str(x) or "" for x in value.tolist())
    except ImportError:
        pass
    return str(value)


def inspect_file(path: str | Path, *, limit: int | None = None) -> Iterator[TrialHeader]:
    """Yield metadata without materializing the large neural recordings.

    Data file stores one HDF5 group per trial. `input_features` has 512 columns,
    `n_time_steps` is an attribute. Labels may be absent in held-out test split.
    Failure is explicit: never silently truncate or 'repair' unsupported input.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install optional speech dependency: pip install h5py") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    count = 0
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if limit is not None and count >= limit:
                break
            trial = f[key]
            if not isinstance(trial, h5py.Group):
                raise ValueError(f"{key}: expected trial group")
            if "input_features" not in trial:
                raise ValueError(f"{key}: missing input_features")
            features = trial["input_features"]
            shape = tuple(int(x) for x in features.shape)
            if len(shape) != 2 or shape[1] != N_FEATURES:
                raise ValueError(f"{key}: expected [T,{N_FEATURES}], got {shape}")
            if "n_time_steps" not in trial.attrs:
                raise ValueError(f"{key}: missing n_time_steps attribute")
            steps = int(trial.attrs["n_time_steps"])
            if not (0 < steps <= shape[0]):
                raise ValueError(f"{key}: invalid n_time_steps={steps} for shape {shape}")
            yield TrialHeader(
                source=str(path), key=key, n_time_steps=steps,
                feature_shape=shape,
                has_phoneme_labels="seq_class_ids" in trial,
                has_transcription="transcription" in trial,
                session=_scalar_str(trial.attrs.get("session")),
                block_num=_scalar_str(trial.attrs.get("block_num")),
                trial_num=_scalar_str(trial.attrs.get("trial_num")),
            )
            count += 1


def load_trial_features(path: str | Path, key: str):
    """Load one trial's valid neural feature bins, not the whole dataset."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        trial = f[key]
        arr = trial["input_features"]
        if len(arr.shape) != 2 or arr.shape[1] != N_FEATURES:
            raise ValueError(f"Expected [T,{N_FEATURES}] feature matrix")
        steps = int(trial.attrs["n_time_steps"])
        if not (0 < steps <= arr.shape[0]):
            raise ValueError("Invalid valid-bin count")
        x = np.asarray(arr[:steps, :], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite neural features")
        return x
