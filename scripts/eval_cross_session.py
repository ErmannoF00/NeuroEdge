#!/usr/bin/env python3
"""Evaluate session drift and host-calibrated input gains/offsets.

Strict protocol:
- frozen prior-session model/normalization
- update only 512 gains+512 offsets on NEW session train-file trials
- evaluate only on NEW session validation file
- no adaptation gradients or selection using validation labels
- NO on-chip learning, NO claim of generalization to other patients
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neuroedge.speech.ctc import BLANK
from neuroedge.speech.dataset import list_labeled_trials, SpeechTrials, collate_ctc
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.adaptation import ChannelAffineDecoder
from train_speech_ctc import run_epoch


def checksum(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(model, loader):
    return run_epoch(model, loader, None, 'cpu')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--calibration-train', required=True, type=Path)
    p.add_argument('--heldout-val', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--calibration-trials', type=int, default=8)
    p.add_argument('--calibration-steps', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--regularization', type=float, default=0.01)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    a=p.parse_args()
    if a.calibration_trials <= 0 or a.calibration_steps <= 0 or a.batch_size <= 0 or a.threads <= 0:
        p.error('trials, steps, batch-size, threads must be positive')
    if a.lr <= 0 or a.regularization < 0:
        p.error('lr must be positive; regularization >=0')
    if a.calibration_train.resolve() == a.heldout_val.resolve():
        p.error('Calibration and evaluation must use different files')
    if a.calibration_train.parent.resolve() != a.heldout_val.parent.resolve():
        p.error('Calibration and validation must belong to same new recording session')
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train_sha=checksum(a.calibration_train)
    val_sha=checksum(a.heldout_val)
    cp=torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cp_manifest=cp['data_manifest']
    if train_sha in (cp_manifest['train_sha256'],cp_manifest['val_sha256']) or val_sha in (cp_manifest['train_sha256'],cp_manifest['val_sha256']):
        raise ValueError('Requested files must be from a DIFFERENT recording session')
    if train_sha == val_sha:
        raise ValueError('Same content in calibration/evaluation files')
    base=CompactCTC(hidden=int(cp['architecture']['hidden']))
    base.load_state_dict(cp['state_dict'],strict=True)
    mean,std=cp['mean'].numpy(),cp['std'].numpy()
    calibration_refs=list_labeled_trials(a.calibration_train, a.calibration_trials)
    heldout_refs=list_labeled_trials(a.heldout_val)
    train_ds=SpeechTrials(calibration_refs,mean,std)
    val_ds=SpeechTrials(heldout_refs,mean,std)
    loader_train=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,collate_fn=collate_ctc,num_workers=0)
    loader_val=DataLoader(val_ds,batch_size=a.batch_size,shuffle=False,collate_fn=collate_ctc,num_workers=0)
    model=ChannelAffineDecoder(base)
    before=evaluate(model,loader_val)
    print('CROSS_SESSION_UNADAPTED_PER=',round(before['phoneme_error_rate_greedy_no_lm'],5),flush=True)
    loss_fn=nn.CTCLoss(blank=BLANK,reduction='sum',zero_infinity=False)
    optim=torch.optim.Adam([model.gain,model.offset],lr=a.lr)
    started=time.monotonic()
    iterator=iter(loader_train)
    for step in range(1,a.calibration_steps+1):
        try:
            x,x_lens,targets,t_lens=next(iterator)
        except StopIteration:
            iterator=iter(loader_train)
            x,x_lens,targets,t_lens=next(iterator)
        model.train(True)
        optim.zero_grad(set_to_none=True)
        logits=model(x)
        out_lens=CompactCTC.output_lengths(x_lens)
        if (out_lens<t_lens).any():
            raise ValueError('Invalid CTC lengths for calibration trial')
        logp=logits.log_softmax(-1).transpose(0,1).contiguous()
        ctc=loss_fn(logp,targets,out_lens,t_lens)/t_lens.sum().clamp(min=1)
        loss=ctc+a.regularization*model.regularization()
        if not torch.isfinite(loss):
            raise FloatingPointError('Calibration loss is nonfinite')
        loss.backward()
        nn.utils.clip_grad_norm_([model.gain,model.offset],max_norm=1.)
        optim.step()
        # Only calibration training loss is used for optimization; validation is untouched until the end.
        if step==1 or step%10==0 or step==a.calibration_steps:
            print(f'CAL_STEP={step:03d} CTC_TRAIN={ctc.item():.5f}',flush=True)
    model.eval()
    after=evaluate(model,loader_val)
    a.out.mkdir(parents=True,exist_ok=True)
    torch.save({'calibration':model.calibration_state(),
                'base_checkpoint_sha256':checksum(a.checkpoint),
                'calibration_train_sha256':train_sha,
                'heldout_val_sha256':val_sha,
                'calibration_trial_count':len(calibration_refs),
                'algorithm':'host-backprop per-channel affine'},a.out/'calibration.pt')
    outcome={
        'protocol':'new-session train calibration; new-session heldout validation',
        'frozen_pretrained_decoder':True,
        'pretrained_train_sha256':cp_manifest['train_sha256'],
        'pretrained_val_sha256':cp_manifest['val_sha256'],
        'calibration_train_sha256':train_sha,
        'heldout_val_sha256':val_sha,
        'calibration_train_trials':len(calibration_refs),
        'heldout_validation_trials':len(heldout_refs),
        'calibration_steps':a.calibration_steps,
        'calibration_lr':a.lr,
        'regularization':a.regularization,
        'seed':a.seed,
        'adaptable_params':sum(p.numel() for p in (model.gain,model.offset)),
        'before':before,'after':after,
        'per_delta_after_minus_before':after['phoneme_error_rate_greedy_no_lm']-before['phoneme_error_rate_greedy_no_lm'],
        'elapsed_s':round(time.monotonic()-started,2),
        'limitations':['One participant, dates differ',
                       'Host backprop for affine calibration, no silicon learning yet',
                       'No guarantee adaptation improves held-out phoneme error',
                       'PER without language model, not WER',
                       'Validation used only for final evaluation; hyperparameters must be fixed in advance'],
    }
    (a.out/'cross_session.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print('CROSS_SESSION_ADAPTED_PER=',round(after['phoneme_error_rate_greedy_no_lm'],5),flush=True)
    print('CROSS_SESSION_DELTA_PER=',round(outcome['per_delta_after_minus_before'],5),flush=True)
    print('CROSS_SESSION_RESEARCH_EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
