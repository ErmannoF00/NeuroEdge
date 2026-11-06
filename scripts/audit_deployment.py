#!/usr/bin/env python3
"""Frozen August-trained model, train-calibrated INT8 vs float on two dates.

Important: December has ALREADY BEEN USED for method development. This is a
cross-session *diagnostic*, NOT an untouched scientific test, nor clinical data.
Persists only aggregate phoneme errors and trial-level COUNTS, no neural arrays,
patient transcriptions, frame-level predictions or trial names.
"""
from __future__ import annotations
from neuroedge.contracts import normalize_contract
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from neuroedge.speech.ctc import ctc_collapse,edit_distance
from neuroedge.speech.dataset import list_labeled_trials,read_trial
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.fixed_decoder import quantize_model,quantize_input,FixedStreamingDecoder


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def aggregate(trial_rows):
    if not trial_rows: raise ValueError('empty trial data')
    tokens=sum(r['target_tokens'] for r in trial_rows)
    if tokens<=0:raise ValueError('no targets')
    return {
        'trials':len(trial_rows),'target_phonemes':tokens,
        'float_PER':sum(r['float_edits'] for r in trial_rows)/tokens,
        'int8_PER':sum(r['int8_edits'] for r in trial_rows)/tokens,
        'int_minus_float_PER':sum(r['int8_edits']-r['float_edits'] for r in trial_rows)/tokens,
    }


def bootstrap_delta_ci(trial_rows, seed=2026, repetitions=2000):
    """Trial-resampled micro-PER delta. Descriptive; not a patient-level CI."""
    if repetitions<100:raise ValueError('Need >=100 bootstrap repetitions')
    n=len(trial_rows)
    r=np.random.default_rng(seed)
    token=np.array([t['target_tokens'] for t in trial_rows],dtype=np.int64)
    diff=np.array([t['int8_edits']-t['float_edits'] for t in trial_rows],dtype=np.int64)
    draws=r.integers(0,n,size=(repetitions,n))
    d=np.sum(diff[draws],axis=1)/np.sum(token[draws],axis=1)
    return list(map(float,np.percentile(d,[2.5,97.5])))


def evaluate_file(path, model, params, mean, std, input_frac_bits):
    trial_rows=[]
    saturation={k:{'hits':0,'n':0} for k in ('project','dw1','pw1','dw2','pw2','head')}
    for ref in list_labeled_trials(path):
        x,y=read_trial(ref)
        normalized=np.clip((x-mean)/std,-8,8).astype(np.float32)
        with torch.no_grad():
            logits=model(torch.from_numpy(normalized.copy())[None])[0]
            float_path=logits.argmax(dim=-1).cpu().numpy().tolist()
        dec=FixedStreamingDecoder(params)
        q=quantize_input(normalized,frac_bits=input_frac_bits)
        int_path=dec.run(q).argmax(axis=1).tolist()
        if len(float_path)!=len(int_path):raise AssertionError('CTC timeline mismatch')
        truth=y.tolist()
        trial_rows.append({
            'target_tokens':len(truth),
            'float_edits':edit_distance(ctc_collapse(float_path),truth),
            'int8_edits':edit_distance(ctc_collapse(int_path),truth),
        })
        for k in saturation:
            saturation[k]['hits']+=dec.saturation_hits[k]
            saturation[k]['n']+=dec.layer_outputs[k]
    return trial_rows,saturation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--train',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--reference-val',type=Path,required=True)
    ap.add_argument('--shift-val',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args()
    if args.bootstrap<100:ap.error('--bootstrap requires >=100')
    torch.set_num_threads(args.threads)
    sha={k:sha256_file(p) for k,p in {
        'checkpoint':args.checkpoint,'calibration_train':args.train,
        'reference_val':args.reference_val,'shift_val':args.shift_val
    }.items()}
    if len({sha['calibration_train'],sha['reference_val'],sha['shift_val']})!=3:
        raise ValueError('Train, reference validation and shift validation MUST differ')
    profile=json.loads(args.profile.read_text())
    if normalize_contract(profile.get('contract'))!='neuroedge-train-only-power2-activation-calibration':
        raise ValueError('Invalid PTQ profile contract')
    if profile['checkpoint_sha256']!=sha['checkpoint']:
        raise ValueError('PTQ checkpoint mismatch')
    if profile['calibration_hdf5_sha256']!=sha['calibration_train']:
        raise ValueError('PTQ profile did not originate from specified TRAIN file')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=CompactCTC(hidden=int(ck['architecture']['hidden']))
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    params=quantize_model(model,profile)
    mean=np.asarray(ck['mean'],np.float32)
    std=np.asarray(ck['std'],np.float32)
    result={
      'contract':'neuroedge-frozen-deployment-diagnostic',
      'provenance_sha256':sha,'profile_sha256':sha256_file(args.profile),
      'important_limitations':[
        'Shift session previously used in method development; NOT an untouched holdout',
        'Trial bootstrap assumes exchangeability of trial observations; does not establish participant-level generalization',
        'Training-normalization and PTQ parameters fixed from original session',
        'Only float and INT8 software inference; silicon PPA and speech utility not established',
      ],
      'datasets':{},
    }
    for label,path in [('reference',args.reference_val),('shift_diagnostic',args.shift_val)]:
        rows,sat=evaluate_file(path,model,params,mean,std,int(profile['input_frac_bits']))
        entry=aggregate(rows)
        entry['int_minus_float_trial_bootstrap_ci95']=bootstrap_delta_ci(rows,seed=2026,repetitions=args.bootstrap)
        entry['saturation']=sat
        result['datasets'][label]=entry
        print(f"{label.upper()}_PASS trials={entry['trials']} float_PER={entry['float_PER']:.6f} int8_PER={entry['int8_PER']:.6f} delta={entry['int_minus_float_PER']:+.6f}",flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('9_FROZEN_DEPLOYMENT_AUDIT_PASS')

if __name__=='__main__': main()
