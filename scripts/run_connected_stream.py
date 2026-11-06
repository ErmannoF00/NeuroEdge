#!/usr/bin/env python3
"""Privacy-conscious host replay: 20ms features -> decoder -> delayed feedback.

Pure SOFTWARE functional replay; no physical sampler, RTL cosimulation, real-
time guarantee, TTS or implicit reading of private inner speech.

Examples:
  python scripts/run_connected_stream.py --synthetic --compare
  python scripts/run_connected_stream.py --hdf5 LOCAL_VAL.hdf5 \
      --checkpoint LOCAL_BEST.pt --max-trials 4 --compare
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neuroedge.speech.fixed_decoder import (
    FixedParameters, TOTAL_META, TOTAL_WEIGHTS, quantize_input, quantize_model,
    FixedStreamingDecoder,
)
from neuroedge.speech.ctc import ctc_collapse
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters
from neuroedge.system.streaming import run_trial_series


def load_hex_array(path: Path, count: int, bits: int, signed: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text().splitlines()
    try:
        encoded = [int(line.strip(), 16) for line in lines if line.strip()]
    except ValueError as exc:
        raise ValueError(f'{path.name}: non-hex token') from exc
    if len(encoded) != count or any(v >= 1 << bits or v < 0 for v in encoded):
        raise ValueError(f'{path.name}: invalid count or width')
    if signed:
        encoded = [v - (1 << bits) if (v & (1 << (bits-1))) else v for v in encoded]
    return np.asarray(encoded, dtype={8: np.int8 if signed else np.uint8,
                                       16: np.int16, 32: np.int32}[bits])


def from_rtl_weights(directory: Path) -> FixedParameters:
    weights = load_hex_array(directory/'all_weights.mem', TOTAL_WEIGHTS, 8, True)
    biases = load_hex_array(directory/'all_biases.mem', TOTAL_META, 32, True)
    shifts = load_hex_array(directory/'all_shifts.mem', TOTAL_META, 8, False)
    return FixedParameters(weights, biases, shifts)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_real_trials(*, hdf5: Path, checkpoint: Path, profile_path: Path | None,
                     weights_dir: Path | None, max_trials: int):
    import torch
    from neuroedge.speech.model import CompactCTC
    from neuroedge.speech.dataset import list_labeled_trials, read_trial
    from neuroedge.contracts import normalize_contract

    if max_trials < 1:
        raise ValueError('specify a bounded positive --max-trials for initial real-data replay')
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    mean = np.asarray(ck['mean'], dtype=np.float32)
    std = np.asarray(ck['std'], dtype=np.float32)
    if mean.shape != (512,) or std.shape != (512,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('invalid checkpoint train-only normalization')
    profile = json.loads(profile_path.read_text()) if profile_path else None
    if profile:
        if normalize_contract(profile.get('contract')) != 'neuroedge-train-only-power2-activation-calibration':
            raise ValueError('unexpected quantization profile')
        if profile['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('checkpoint/profile mismatch')
        if profile['calibration_hdf5_sha256'] == sha256(hdf5):
            raise ValueError('would evaluate on the quantization calibration HDF5')
    model = CompactCTC(hidden=int(ck['architecture']['hidden']))
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    p = quantize_model(model, profile)
    if weights_dir is not None:
        exported = from_rtl_weights(weights_dir)
        if not (np.array_equal(exported.weights, p.weights)
                and np.array_equal(exported.biases, p.biases)
                and np.array_equal(exported.shifts, p.shifts)):
            raise ValueError('RTL weight directory does not match checkpoint and quantization profile')
        p = exported
    frac = int(profile['input_frac_bits']) if profile else 4
    if frac != 4:
        raise ValueError(f'profile input_frac_bits={frac} conflicts with the existing Q4 transactional offset interface; use a validated Q4 export')
    items = []
    refs = list_labeled_trials(hdf5, limit=max_trials)
    for ref in refs:
        raw, target = read_trial(ref)
        x = np.clip((raw - mean) / std, -8, 8).astype(np.float32)
        items.append((quantize_input(x, frac_bits=frac), target.tolist()))
    return p, items


def main():
    a = argparse.ArgumentParser(description=__doc__)
    source = a.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--hdf5', type=Path)
    a.add_argument('--checkpoint', type=Path)
    a.add_argument('--quant-profile', type=Path)
    a.add_argument('--weights-dir', type=Path)
    a.add_argument('--max-trials', type=int, default=4)
    a.add_argument('--banks', type=int, choices=(1,2,4,8,16,32,64), default=None)
    a.add_argument('--compare', action='store_true', help='Frozen vs sham vs post-trial supervised controls')
    a.add_argument('--feedback-mode', choices=('off','supervised','sham'), default='off')
    a.add_argument('--feedback-channel', type=int, default=0)
    a.add_argument('--step-q4', type=int, default=4)
    a.add_argument('--delay-trials', type=int, default=0)
    a.add_argument('--release-trials', type=str, default='', help='External user-armed trial IDs, e.g. 0,2')
    a.add_argument('--out', type=Path, default=Path('build/connected_stream_report.json'))
    args = a.parse_args()
    if args.synthetic:
        if args.checkpoint or args.quant_profile or args.hdf5 or args.weights_dir:
            a.error('synthetic mode cannot take human dataset/model arguments')
        p = synthetic_parameters()
        clean = synthetic_features(32)
        teacher = FixedStreamingDecoder(p).run(clean)
        # Controlled synthetic input disturbance; labels are derived from a
        # separate teacher stream ONLY for harness coverage.
        disturbed = clean.copy()
        disturbed[:, 0] = np.clip(disturbed[:, 0].astype(np.int32)+9, -128, 127).astype(np.int8)
        trials = [(disturbed[4*i:4*i+4].copy(),
                   ctc_collapse(teacher[i:i+1].argmax(axis=1).tolist()))
                  for i in range(8)]
        provenance = 'nonlearned 32-bin synthetic fixture with injected drift'
    else:
        if not args.checkpoint:
            a.error('--hdf5 requires --checkpoint with frozen train-only normalization')
        p, trials = load_real_trials(hdf5=args.hdf5, checkpoint=args.checkpoint,
                                     profile_path=args.quant_profile,
                                     weights_dir=args.weights_dir,
                                     max_trials=args.max_trials)
        provenance = 'local attempted-speech HDF5 with frozen checkpoint normalization (no data redistributed)'
    try:
        armed = frozenset(int(x) for x in args.release_trials.split(',') if x.strip())
    except ValueError:
        a.error('bad --release-trials; expected comma-separated integer IDs')
    if any(x < 0 or x >= len(trials) for x in armed):
        a.error('release trial IDs out of range')
    modes = ('off', 'sham', 'supervised') if args.compare else (args.feedback_mode,)
    # A trained checkpoint, if used, is unchanged in all three control arms.
    reports = {mode: run_trial_series(p, trials, banks=args.banks, mode=mode,
                                      armed_trials=armed, feedback_channel=args.feedback_channel,
                                      step_q4=args.step_q4,
                                      delay_trials=args.delay_trials)
               for mode in modes}
    output = {'provenance': provenance,
              'numerical_contract': 'signed INT8 features + INT16 Q4 offsets, INT16 41-way logits',
              'virtual_stream': '20ms exact bin timestamps; software replay, NOT wall-time acquisition',
              'feedback_supervision': 'retrospective offline task labels revealed post trial (not automatically available in deployment)',
              'neural_intent': 'explicit release authorization only; not a thought-reading classifier',
              'backend_banks': args.banks, 'mode_results': reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print('NEUROEDGE_CONNECTED_STREAM_SOFTWARE_PASS')
    for name, r in reports.items():
        print(f"  {name}: trials={r['trials']} bins={r['bins']} logits={r['outputs']*41} "
              f"greedy_PER={r['greedy_per']} memory_writes={r['committed_writes']}")
    print('  local_report=', args.out)


if __name__ == '__main__':
    main()
