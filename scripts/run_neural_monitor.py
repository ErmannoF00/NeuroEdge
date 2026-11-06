#!/usr/bin/env python3
"""Train-only activity thresholds; evaluate August vs December; export real-neural RTL vectors.

Unsupervised data-characterization study, NOT validated drift detection or adaptation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from neuroedge.control.budget import BudgetController
from neuroedge.drift.monitor import (
    calibrated_thresholds, evaluate_trials, excursion_scores, iter_hdf5_trials,
    quantize_neural, trial_bootstrap_difference,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def emit_rtl_vectors(out: Path, q: np.ndarray, low: int, high: int, magnitude: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    score = excursion_scores(q, magnitude)
    c = BudgetController(low_threshold=low, high_threshold=high)
    with (out/'features.mem').open('w') as f:
        for v in q.reshape(-1):
            f.write(f'{int(v)&255:02x}\n')
    with (out/'expected.txt').open('w') as f:
        for s in score:
            a, r, w, sup = c.sample(int(s))
            f.write(f'{int(s)} {a} {r} {w} {sup}\n')
    (out/'contract.json').write_text(json.dumps({
        'contract': 'neuroedge-INT8-excursion-activity',
        'frames': len(q), 'features_per_frame': 512,
        'low_threshold': low, 'high_threshold': high,
        'magnitude_threshold': magnitude,
        'n_requests': int(sum(int(line.split()[2]) for line in (out/'expected.txt').read_text().splitlines())),
        'source': 'real neural recording; frozen normalization',
        'not_update_execution': True,
    }, indent=2) + '\n')


def run(args):
    for p in (args.train, args.reference_val, args.shift_val, args.checkpoint, args.profile):
        if not p.is_file():
            raise FileNotFoundError(p)
    if len({args.train.resolve(), args.reference_val.resolve(), args.shift_val.resolve()}) != 3:
        raise ValueError('train/reference/shift must be separate data files')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    mean = np.asarray(ck['mean'], dtype=np.float32)
    std = np.asarray(ck['std'], dtype=np.float32)
    profile = json.loads(args.profile.read_text())
    bits = int(profile['input_frac_bits'])
    magnitude = 2 * (1 << bits)
    if not 1 <= bits <= 5:
        raise ValueError('input precision unsupported by existing comparator')
    manifest = ck.get('data_manifest', {})
    true_train_hash = sha256(args.train)
    if manifest.get('train_sha256') != true_train_hash:
        raise ValueError('checkpoint training provenance differs from provided train HDF5')
    if profile.get('checkpoint_sha256') != sha256(args.checkpoint):
        raise ValueError('profile belongs to a different checkpoint')
    if profile.get('calibration_hdf5_sha256') != true_train_hash:
        raise ValueError('activation profile calibration train file differs')
    if args.max_cal_trials < 1 or args.max_evaluation_trials < 0 or args.export_bins < 1:
        raise ValueError('invalid trial counts or export bin count')

    # Thresholds are fitted strictly on training recordings, never on either validation file.
    train_scores = []
    train_trial_count = 0
    for _, x in iter_hdf5_trials(args.train, max_trials=args.max_cal_trials):
        train_trial_count += 1
        q = quantize_neural(x, mean, std, bits)
        train_scores.append(excursion_scores(q, magnitude))
    train_scores = np.concatenate(train_scores)
    low, high = calibrated_thresholds(train_scores)
    print('TRAIN_ONLY_ACTIVITY_THRESHOLDS_PASS',
          f'low={low}', f'high={high}', f'calibration_bins={len(train_scores)}', flush=True)

    kwargs = dict(mean=mean, std=std, fraction_bits=bits,
                  magnitude_threshold=magnitude, low_threshold=low,
                  high_threshold=high, max_trials=args.max_evaluation_trials)
    reference = evaluate_trials(args.reference_val, **kwargs)
    shift = evaluate_trials(args.shift_val, **kwargs)
    delta = shift.mean_excursions - reference.mean_excursions
    equal_trial_delta = float(np.mean(shift.trial_mean_excursions) - np.mean(reference.trial_mean_excursions))
    ci = trial_bootstrap_difference(reference.trial_mean_excursions,
                                    shift.trial_mean_excursions, args.seed, args.bootstrap)
    # The exact RTL score+policy vector uses true later-session recordings.
    key, first = next(iter_hdf5_trials(args.shift_val, max_trials=1))
    frames = min(args.export_bins, len(first))
    q = quantize_neural(first[:frames], mean, std, bits)
    out = args.out
    emit_rtl_vectors(out/'rtl_vectors', q, low, high, magnitude)
    def public_summary(d):
        # No raw neural values, transcriptions, or per-trial identifiers in report.
        obj = asdict(d)
        obj.pop('trial_requests')
        obj.pop('trial_mean_excursions')
        return obj
    report = {
        'contract': 'neuroedge-unlabeled-neural-excursion-activity',
        'recording_sessions': ['reference', 'shift_development'],
        'estimator': 'number of |INT8 normalized neural features| >= 2 sigma per 20 ms bin',
        'no_labels_used': True,
        'calibration_only_on_source_train': True,
        'normalizer_source': 'frozen model checkpoint; train split',
        'input_fraction_bits': bits, 'magnitude_threshold': magnitude,
        'hysteresis_low': low, 'hysteresis_high': high,
        'controller_settings': {'window_bins': 50, 'max_requests': 4, 'cooldown_bins': 5},
        'controller_reset': 'at each trial boundary; recording gaps unknown',
        'train_calibration_trials': train_trial_count,
        'train_calibration_bins': len(train_scores),
        'train_score_mean': float(train_scores.mean()),
        'reference': public_summary(reference),
        'shift_development': public_summary(shift),
        'difference_bin_weighted_mean_excursions': float(delta),
        'difference_equal_trial_mean_excursions': equal_trial_delta,
        'difference_equal_trial_mean_bootstrap_ci95': list(ci),
        'difference_mean_definitions': 'main delta bin-weighted; CI on equal-trial means (not exact same estimand when trial durations differ)',
        'rtl_vector_trial_key_only_not_saved': True,
        'rtl_vector_bins': int(frames),
        'limitations': [
            'A neural activity excursion is not a validated drift detector.',
            'Both comparisons are single-participant and December is already a development session.',
            'An update request does not change model parameters and is not a physical write.',
            'No accuracy, energy, detector sensitivity/specificity, or device-endurance claim is supported.',
            'The trial-cluster bootstrap does not establish generalization to other participants.',
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/'activity_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REFERENCE_ACTIVITY_PASS',
          f'trials={reference.trials}', f'bins={reference.bins}',
          f'count_mean={reference.mean_excursions:.3f}',
          f'requests={reference.requested_actions}', flush=True)
    print('SHIFT_ACTIVITY_PASS',
          f'trials={shift.trials}', f'bins={shift.bins}',
          f'count_mean={shift.mean_excursions:.3f}',
          f'requests={shift.requested_actions}', flush=True)
    print(f'SHIFT_MINUS_REFERENCE_COUNT_MEAN={delta:+.4f}', flush=True)
    print('UNLABELED_NEURAL_AUDIT_PASS', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--reference-val', type=Path, required=True)
    p.add_argument('--shift-val', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-cal-trials', type=int, default=64)
    p.add_argument('--max-evaluation-trials', type=int, default=0)
    p.add_argument('--export-bins', type=int, default=16)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    run(a)


if __name__ == '__main__':
    main()
