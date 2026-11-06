#!/usr/bin/env python3
"""existing real-data lagged trial association with frozen existing system.
No thresholds tuned, no checkpoint changes, no waveform/labels persisted.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from neuroedge.drift.causal_audit import audit_trial_risk
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.fixed_decoder import quantize_model
from scripts.audit_monitor_utility import provenance, evaluate


def run(args):
    if args.threads<1 or args.permutations<100 or args.max_trials<0:
        raise ValueError('invalid parameters')
    torch.set_num_threads(args.threads)
    ck,profile,monitor,hashes=provenance(args)
    model=CompactCTC(hidden=int(ck['architecture']['hidden']))
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    params=quantize_model(model,profile)
    report={
        'contract':'NeuroEdge-existing-temporally-lagged-trial-risk-diagnostic',
        'dataset_hashes_sha256':hashes,
        'session_ordering':'dataset trial_XXXX keys, within each HDF5 file',
        'policy_frozen':True,
        'model_frozen':True,
        'dataset_results':{},
    }
    for i,(name,p) in enumerate([('reference',args.reference_val),('shift_development',args.shift_val)]):
        rows=evaluate(p,params,ck,profile,monitor,args.max_trials)
        result=audit_trial_risk(rows,seed=args.seed+10000*i,permutations=args.permutations)
        report['dataset_results'][name]=result
        print(f'{name.upper()}_PASS trials={result["trials"]} lag_pairs={result["lagged_pairs"]}',flush=True)
        for measure in ('request_rate_per_1000_bins','blank_fraction'):
            a=result['same_trial'][measure]
            b=result['previous_trial_predicts_next_trial'][measure]
            c=result['rank_adjusted_for_bin_and_label_count'][measure]
            print(f'  {measure}: same={a["rho"]:+.4f} adjusted={c["rho"]:+.4f} '
                  f'lagged={b["rho"]:+.4f} lag_BH_q={b["bh_q_exploratory"]:.4f}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('TEMPORALLY_LAGGED_REAL_NEURAL_AUDIT_PASS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','reference_val','shift_val','checkpoint','profile','monitor_report','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--permutations',type=int,default=2000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-trials',type=int,default=0)
    run(p.parse_args())
if __name__=='__main__': main()
