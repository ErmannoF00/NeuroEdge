#!/usr/bin/env python3
"""Retrospective neural-monitor/decoder-error utility audit on real trials.

Frozen August trained model and PTQ profile, existing activity thresholds.
Labels are used ONLY to evaluate trial-level phoneme errors after replay.
Never trains a policy, reselects thresholds, updates weights, or writes labels.
"""
from __future__ import annotations
from neuroedge.contracts import normalize_contract
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from neuroedge.control.budget import BudgetController
from neuroedge.drift.monitor import quantize_neural,excursion_scores
from neuroedge.drift.utility import logit_metrics,summarize_trial_utility
from neuroedge.speech.dataset import list_labeled_trials,read_trial
from neuroedge.speech.ctc import ctc_collapse,edit_distance
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder,quantize_model
from neuroedge.speech.model import CompactCTC


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def provenance(args):
    for p in (args.train,args.reference_val,args.shift_val,args.checkpoint,args.profile,args.monitor_report):
        if not p.is_file(): raise FileNotFoundError(p)
    h={k:digest(getattr(args,k)) for k in ('train','reference_val','shift_val','checkpoint','profile','monitor_report')}
    if len(set(h[k] for k in ('train','reference_val','shift_val'))) != 3:
        raise ValueError('train/reference/shift files must be distinct')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    profile=json.loads(args.profile.read_text())
    mon=json.loads(args.monitor_report.read_text())
    if ck.get('data_manifest',{}).get('train_sha256')!=h['train']:
        raise ValueError('checkpoint not trained on supplied train HDF5')
    if profile.get('checkpoint_sha256')!=h['checkpoint'] or profile.get('calibration_hdf5_sha256')!=h['train']:
        raise ValueError('PTQ profile provenance mismatch')
    if normalize_contract(profile.get('contract'))!='neuroedge-train-only-power2-activation-calibration':
        raise ValueError('unexpected profile contract')
    if normalize_contract(mon.get('contract'))!='neuroedge-unlabeled-neural-excursion-activity' or not mon.get('calibration_only_on_source_train'):
        raise ValueError('unexpected activity monitor contract')
    if int(profile['input_frac_bits'])!=int(mon['input_fraction_bits']):
        raise ValueError('monitor and decoder input quantization differ')
    if int(mon['magnitude_threshold']) != 2*(1 << int(profile['input_frac_bits'])):
        raise ValueError('monitor excursion threshold differs')
    return ck,profile,mon,h


def evaluate(path,params,ck,profile,monitor,max_trials=0):
    mean=np.asarray(ck['mean'],dtype=np.float32)
    std=np.asarray(ck['std'],dtype=np.float32)
    frac=int(profile['input_frac_bits'])
    rows=[]
    for ref in list_labeled_trials(path,limit=max_trials):
        x,y=read_trial(ref)
        q=quantize_neural(x,mean,std,frac)
        scores=excursion_scores(q,int(monitor['magnitude_threshold']))
        ctrl=BudgetController(high_threshold=int(monitor['hysteresis_high']),
                              low_threshold=int(monitor['hysteresis_low']))
        requests=0
        for score in scores.tolist():
            _,action,_,_=ctrl.sample(int(score))
            requests+=action
        logits=FixedStreamingDecoder(params).run(q)
        output=ctc_collapse(logits.argmax(axis=1).tolist())
        confidence=logit_metrics(logits)
        rows.append({'bins':len(x),'target_tokens':len(y),
                     'edits':edit_distance(output,y.tolist()),
                     'requests':requests,'score_mean':float(scores.mean()),
                     **confidence})
    return rows


def run(args):
    if args.permutations<100 or args.bootstraps<100 or args.max_trials<0:
        raise ValueError('Invalid repetitions or trial limit')
    torch.set_num_threads(args.threads)
    ck,profile,monitor,hashes=provenance(args)
    model=CompactCTC(hidden=int(ck['architecture']['hidden']))
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    params=quantize_model(model,profile)
    result={'contract':'NeuroEdge-existing-retrospective-trial-risk-utility',
            'dataset_hashes_sha256':hashes,
            'frozen_monitor':{'low':monitor['hysteresis_low'],
                              'high':monitor['hysteresis_high'],
                              'magnitude':monitor['magnitude_threshold']},
            'datasets':{},
            'important_limitations':[
                'All risk associations are retrospective, not prospectively validated.',
                'December data were previously used for model/policy development: NOT untouched.',
                'Single participant; trial bootstrap/permutation does not imply population generalization.',
                'CTC PER is a per-trial endpoint, not an observable online per-bin failure signal.',
                'Labels are read to assess outcomes ONLY; they are not used to set monitor thresholds.',
                'Four exploratory metrics were compared; p-values are uncorrected.',
                'An adaptation request is not a parameter update, benefit or energy saving.',
                'No patient raw features, phoneme labels, text, or predicted strings persisted.'
            ]}
    for label,p in [('reference',args.reference_val),('shift_development',args.shift_val)]:
        rows=evaluate(p,params,ck,profile,monitor,args.max_trials)
        summary=summarize_trial_utility(rows,seed=args.seed,permutations=args.permutations,
                                        bootstraps=args.bootstraps)
        result['datasets'][label]=summary
        a=summary['associations']
        print(f'{label.upper()}_PASS trials={summary["trials"]} PER={summary["micro_PER"]:.6f} '
              f'requests_per_1000={summary["requests_per_1000_bins"]:.3f} '
              f'request_rho={a["request_rate_per_1000_bins"]["spearman_rho"]:+.4f} '
              f'margin_rho={a["negative_top2_margin"]["spearman_rho"]:+.4f}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('REAL_NEURAL_FAILURE_ASSOCIATION_AUDIT_PASS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','reference_val','shift_val','checkpoint','profile','monitor_report','out'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--permutations',type=int,default=2000)
    p.add_argument('--bootstraps',type=int,default=1000)
    p.add_argument('--max-trials',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026)
    run(p.parse_args())

if __name__=='__main__': main()
