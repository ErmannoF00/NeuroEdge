#!/usr/bin/env python3
"""Train a small causal neural speech student on official labeled HDF5 trials.

PER is phoneme-token Levenshtein / reference phonemes, no language model.
This is NOT WER, the reference RNN, or clinical-level evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neuroedge.speech.ctc import BLANK, NUM_CLASSES, ctc_collapse, edit_distance
from neuroedge.speech.dataset import (
    list_labeled_trials, training_channel_stats, SpeechTrials, collate_ctc,
)
from neuroedge.speech.model import CompactCTC


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_epoch(model, loader, optimizer, device, max_steps=0):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CTCLoss(blank=BLANK, reduction='sum', zero_infinity=False)
    total_loss = 0.0
    n_targets = 0
    edits = 0
    n_reference = 0
    n_predictions = 0
    blank_frames = 0
    total_frames = 0
    trials = 0
    steps = 0

    for x, x_lens, targets, t_lens in loader:
        steps += 1
        if max_steps and steps > max_steps:
            break
        x, targets = x.to(device), targets.to(device)
        # lengths as actual integers; padded bins have no influence on CTC.
        out_lens = CompactCTC.output_lengths(x_lens)
        if (out_lens < t_lens).any():
            raise ValueError('Impossible target length after temporal decimation')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1).contiguous()
            loss_sum = loss_fn(log_probs, targets, out_lens, t_lens)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError('Non-finite CTC loss: inspect data/label alignment')
            loss = loss_sum / torch.clamp(t_lens.sum().to(device), min=1)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss_sum.item())
        n_targets += int(t_lens.sum())
        path_ids = logits.detach().argmax(dim=-1).cpu()
        offset = 0
        for b in range(len(x_lens)):
            truth = targets[offset:offset+int(t_lens[b])].detach().cpu().tolist()
            path = path_ids[b, :int(out_lens[b])].tolist()
            pred = ctc_collapse(path)
            edits += edit_distance(pred, truth)
            n_reference += len(truth)
            n_predictions += len(pred)
            blank_frames += sum(v == BLANK for v in path)
            total_frames += len(path)
            offset += len(truth)
            trials += 1
    if n_targets <= 0:
        raise ValueError('No training/evaluation labels processed')
    return {
        'ctc_nll_per_target': total_loss/n_targets,
        'phoneme_error_rate_greedy_no_lm': edits/n_reference,
        'reference_phonemes': n_reference,
        'predicted_phonemes': n_predictions,
        'decoded_blank_frame_fraction': blank_frames/total_frames,
        'trials': trials,
        'optimizer_steps': steps if not max_steps else min(steps, max_steps),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--val', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-train-trials', type=int, default=0, help='0=ALL, >0=pilot subset')
    p.add_argument('--max-val-trials', type=int, default=0, help='0=ALL, >0=pilot subset')
    p.add_argument('--max-steps', type=int, default=0, help='0=ALL; otherwise engineering dry-run')
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden <= 0 or args.threads <= 0 or args.max_steps < 0:
        p.error('epochs, batch-size, hidden and threads must be positive; max-steps >= 0')
    if args.train.resolve() == args.val.resolve():
        p.error('Train and validation cannot refer to the same file')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    train_refs = list_labeled_trials(args.train, args.max_train_trials)
    val_refs = list_labeled_trials(args.val, args.max_val_trials)
    if set((r.file, r.key) for r in train_refs) & set((r.file,r.key) for r in val_refs):
        raise ValueError('Train/validation leakage: duplicated source/key')
    print('TRAINING_STATS_FROM_TRAIN_ONLY_START', flush=True)
    mean, std = training_channel_stats(train_refs)
    train_ds = SpeechTrials(train_refs, mean, std)
    val_ds = SpeechTrials(val_refs, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_ctc, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_ctc, num_workers=0)
    model = CompactCTC(hidden=args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print('PARAMETERS=', sum(x.numel() for x in model.parameters()))
    pilot = args.max_train_trials > 0 or args.max_val_trials > 0 or args.max_steps > 0
    manifest = {
        'dataset': 'Brain-to-Text 25: one participant, one recording session',
        'train_sha256': sha256_file(args.train), 'val_sha256': sha256_file(args.val),
        'train_trial_count': len(train_ds), 'validation_trial_count': len(val_ds),
        'training_mean_std_source': 'training split only',
        'ctc_blank_index': BLANK, 'phoneme_classes': NUM_CLASSES,
        'temporal_downsample': 4, 'causal': True,
        'evaluation_metric': 'greedy phoneme error rate without a language model',
        'is_pilot': pilot, 'not_official_baseline': True,
        'not_word_error_rate': True, 'not_clinically_validated': True,
    }
    (args.out/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    best_val = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        tr = run_epoch(model, train_loader, optimizer, args.device, args.max_steps)
        va = run_epoch(model, val_loader, None, args.device, args.max_steps)
        entry = {'epoch': epoch, 'train': tr, 'validation': va,
                 'elapsed_s': round(time.monotonic() - started, 2)}
        history.append(entry)
        print(f"EPOCH={epoch:03d} CTC_TRAIN={tr['ctc_nll_per_target']:.4f} "
              f"CTC_VAL={va['ctc_nll_per_target']:.4f} "
              f"PER_VAL={va['phoneme_error_rate_greedy_no_lm']:.4f} "
              f"BLANK_FRAC={va['decoded_blank_frame_fraction']:.3f} "
              f"SEC={entry['elapsed_s']}", flush=True)
        if va['ctc_nll_per_target'] < best_val:
            best_val = va['ctc_nll_per_target']
            torch.save({
                'state_dict': model.state_dict(),
                'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                'architecture': {'hidden': args.hidden, 'n_classes': NUM_CLASSES, 'temporal_downsample': 4},
                'data_manifest': manifest,
                'best_epoch': epoch,
            }, args.out/'best_model.pt')
        (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.out/'metrics.json').write_text(json.dumps({
        'checkpoint_selection': 'minimum validation CTC NLL',
        'best_validation_ctc_nll_per_target': best_val,
        'last_epoch': history[-1], 'pilot_only': pilot,
        'warning': 'Single session, single participant; not cross-session drift or WER',
    }, indent=2)+'\n')
    print('REAL_SPEECH_PILOT_EXECUTED_PASS' if pilot else 'REAL_SPEECH_SINGLE_SESSION_TRAINING_PASS', flush=True)
    print('CHECKPOINT=', args.out/'best_model.pt', flush=True)

if __name__ == '__main__': main()
