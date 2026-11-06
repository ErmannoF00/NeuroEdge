#!/usr/bin/env python3
"""Compare fixed, unsupervised shift-only, unsupervised bounded affine decoder.

The chosen session is DEVELOPMENT ONLY after its validation scores are inspected.
A truly independent final assessment requires a never-inspected later session.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroedge.speech.dataset import list_labeled_trials, SpeechTrials, collate_ctc
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.ctc import ctc_collapse, edit_distance, BLANK
from neuroedge.speech.adaptation import ChannelAffineDecoder
from neuroedge.speech.session_ablation import (
    moments_from_calibration, calibrate_rules, paired_per_bootstrap,
)
from train_speech_ctc import run_epoch


def sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def trial_errors(model, ds):
    """Return validation edit totals per trial, not patient text or labels."""
    model.eval()
    errors, lengths = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(x.unsqueeze(0))
            pred = ctc_collapse(logits.argmax(dim=-1)[0, :int(CompactCTC.output_lengths(torch.tensor([len(x)]))[0])].tolist())
            truth = y.tolist()
            errors.append(edit_distance(pred,truth))
            lengths.append(len(truth))
    return errors, lengths


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--calibration-train', required=True, type=Path)
    ap.add_argument('--development-val', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--calibration-trials', type=int, default=8)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    a=ap.parse_args()
    if a.calibration_trials <= 0 or a.bootstrap <= 0 or a.threads <= 0:
        ap.error('calibration-trials, bootstrap and threads must be positive')
    if a.calibration_train.resolve() == a.development_val.resolve() or a.calibration_train.parent.resolve() != a.development_val.parent.resolve():
        ap.error('Distinct train/val files from the SAME new session required')
    torch.set_num_threads(a.threads)
    np.random.seed(a.seed); torch.manual_seed(a.seed); random.seed(a.seed)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cal_sha=sha(a.calibration_train)
    val_sha=sha(a.development_val)
    prior=cp['data_manifest']
    if len({prior['train_sha256'],prior['val_sha256'],cal_sha,val_sha}) != 4:
        raise ValueError('Data SHA overlap detected: no new-session or split leakage permitted')
    source_mean,source_std=cp['mean'].numpy(),cp['std'].numpy()
    cal_refs=list_labeled_trials(a.calibration_train,a.calibration_trials)
    dev_refs=list_labeled_trials(a.development_val)
    mu,sigma,n_bins=moments_from_calibration(cal_refs,source_mean,source_std)
    methods=calibrate_rules(mu,sigma,n_bins)
    dev=SpeechTrials(dev_refs,source_mean,source_std)
    loader=DataLoader(dev,batch_size=4,shuffle=False,collate_fn=collate_ctc,num_workers=0)
    results={}
    errors={}
    reference=None
    for name,params in methods.items():
        base=CompactCTC(hidden=int(cp['architecture']['hidden']))
        base.load_state_dict(cp['state_dict'],strict=True)
        model=ChannelAffineDecoder(base)
        with torch.no_grad():
            model.gain.copy_(torch.from_numpy(params.gain))
            model.offset.copy_(torch.from_numpy(params.offset))
        scores=run_epoch(model,loader,None,'cpu')
        edits,refs=trial_errors(model,dev)
        per=sum(edits)/sum(refs)
        if abs(per - scores['phoneme_error_rate_greedy_no_lm']) > 1.e-7:
            raise AssertionError('Per-trial PER and aggregate decoder result disagree')
        if reference is not None and refs != reference:
            raise AssertionError('Paired validation trials changed')
        reference=refs
        errors[name]=edits
        results[name]={
            'metrics':scores,
            'gain':params.gain.tolist(),
            'offset':params.offset.tolist(),
            'n_calibration_bins':n_bins,
            'method':params.rule,
        }
        print(f'ABLATED_METHOD={name} DEV_PER={per:.6f} BLANK={scores["decoded_blank_frame_fraction"]:.5f}',flush=True)
    comparisons={
        n:paired_per_bootstrap(reference,errors['fixed'],errors[n],seed=a.seed,iterations=a.bootstrap)
        for n in methods if n != 'fixed'
    }
    manifest={
      'protocol':'DEVELOPMENT_ONLY: source checkpoint fixed, target calibration TRAIN for moments, target VAL for diagnostics',
      'source_checkpoint_sha256':sha(a.checkpoint),
      'source_training_sha256':prior['train_sha256'],
      'source_validation_sha256':prior['val_sha256'],
      'calibration_train_sha256':cal_sha,
      'development_val_sha256':val_sha,
      'calibration_trials':len(cal_refs), 'development_trials':len(dev_refs),
      'source_feature_normalization':'frozen source checkpoint statistics',
      'uses_calibration_labels':False,
      'uses_development_labels_for_parameter_estimation':False,
      'validation_has_been_inspected':True,
      'not_final_heldout_evidence':True,
      'not_onchip_learning':True,
      'method_hyperparams_preset':{'gain_limit':0.25,'offset_limit':2.,'clip':8.},
      'paired_ci_scope':'Within one participant, resampled whole trials; not independent subjects',
      'methods':results,
      'comparison_to_fixed':comparisons,
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'session_ablation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('6_SESSION_ABLATION_DEV_PASS',flush=True)

if __name__=='__main__': main()
