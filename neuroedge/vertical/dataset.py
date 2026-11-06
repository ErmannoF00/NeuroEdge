"""Motor imagery data with strict train/validation/test subject separation.

Synthetic fixtures are for deterministic CI only. A publication-ready run must
use --dataset physionet and follow the protocol in docs/REAL_EEG_PROTOCOL.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import numpy as np

CHANNELS = ('C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'Pz')
IMAGERY_RUNS = (4, 8, 12)   # unilateral left/right *imagery*, not execution

@dataclass
class FeatureSet:
    X: np.ndarray       # shape [epochs, 8], log10(channel power)
    y: np.ndarray       # 0=T1 left, 1=T2 right for run 4/8/12
    subject: np.ndarray
    provenance: str
    source_files: list[dict] = field(default_factory=list)


def synthetic_fixture(subjects: list[int], seed: int = 20261014,
                      n_per_subject: int = 36) -> FeatureSet:
    """A deliberately trivial smoke test, NOT scientific EEG data."""
    rng = np.random.default_rng(seed)
    all_x, all_y, all_subj = [], [], []
    for sid in subjects:
        labels = rng.integers(0, 2, size=n_per_subject, dtype=np.int64)
        offset = rng.normal(0, 0.12, size=(1, 8))
        # Mimic a log-bandpower feature, without pretending it's EEG.
        X = rng.normal(-10.0, 0.3, size=(n_per_subject, 8)) + offset
        X += (labels[:, None] * 2 - 1) * np.array(
            [-0.25, 0.28, 0.02, -0.10, 0.12, -0.06, 0.05, 0.00])
        all_x.append(X)
        all_y.append(labels)
        all_subj.append(np.full(n_per_subject, sid))
    return FeatureSet(np.concatenate(all_x), np.concatenate(all_y),
                      np.concatenate(all_subj), 'synthetic_demo_only')


def physionet_features(subjects: list[int], cache_root: str) -> FeatureSet:
    """Load official PhysioNet EEGMMIDB from MNE, download when absent.

    Data: 64 channels at 160 Hz, runs 4/8/12 (imagined left/right fist).
    Features: log10(mean(x**2)) per EEG channel over 0.5..2.5s epochs
    after 8-30 Hz *offline* IIR band-pass filtering and average reference.
    This filter is currently performed in software; no on-chip DSP claim.
    """
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as exc:
        raise RuntimeError('Install EEG dependencies: pip install -e ".[eeg]"') from exc
    cache = Path(cache_root).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    xs, ys, subs, file_infos = [], [], [], []
    for sid in subjects:
        if sid < 1 or sid > 109:
            raise ValueError('PhysioNet subject IDs must be in 1..109')
        paths = eegbci.load_data(sid, list(IMAGERY_RUNS), path=str(cache), update_path=False)
        for fname in paths:
            fpath=Path(fname)
            hasher=hashlib.sha256()
            with fpath.open('rb') as fin:
                for chunk in iter(lambda:fin.read(1024*1024), b''):
                    hasher.update(chunk)
            file_infos.append({'file':fpath.name,'sha256':hasher.hexdigest(),'bytes':fpath.stat().st_size})
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            eegbci.standardize(raw)
            found = [ch for ch in CHANNELS if ch in raw.ch_names]
            if tuple(found) != CHANNELS:
                raise RuntimeError(f'Channel mismatch in {fname}: {found}; expected {CHANNELS}')
            raw.pick(list(CHANNELS))
            raw.set_eeg_reference('average', verbose=False)
            raw.filter(8., 30., method='iir',
                       iir_params={'order': 4, 'ftype': 'butter'}, verbose=False)
            events, code = mne.events_from_annotations(
                raw, event_id={'T1': 1, 'T2': 2}, verbose=False)
            ep = mne.Epochs(raw, events, event_id=code, tmin=0.5, tmax=2.5,
                            baseline=None, preload=True, verbose=False)
            data = ep.get_data()  # volts: [epoch, channel, sample]
            if len(data) == 0:
                continue
            X = np.log10(np.maximum(np.mean(data**2, axis=2), 1e-20))
            # T1/T2 in unilateral *imagery* runs have a consistent interpretation.
            y = (ep.events[:, -1] == 2).astype(np.int64)
            xs.append(X)
            ys.append(y)
            subs.append(np.full(len(y), sid, dtype=np.int32))
    if not xs:
        raise RuntimeError('No EEG epochs extracted. Inspect MNE annotations.')
    return FeatureSet(np.vstack(xs), np.concatenate(ys), np.concatenate(subs),
                      'physionet_eegmmidb_v1.0.0_mne_runs_4_8_12', file_infos)


def load_features(dataset: str, subjects: list[int], cache_root: str) -> FeatureSet:
    if dataset == 'synthetic':
        return synthetic_fixture(subjects)
    if dataset == 'physionet':
        return physionet_features(subjects, cache_root)
    raise ValueError(f'Unknown dataset {dataset!r}')
