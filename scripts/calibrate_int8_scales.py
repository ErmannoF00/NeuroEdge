#!/usr/bin/env python3
"""Train-only activation calibration: power-of-two scales for a signed INT8
complete causal CompactCTC. NO validation/test data may be used here.

This only selects deployment parameters. Accuracy gain is not guaranteed.
It does not change model weights or use gradients.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.dataset import list_labeled_trials,read_trial
from neuroedge.speech.fixed_decoder import LAYER_ORDER,quantize_model


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def select_bits(abs_q, max_int=110, max_bits=9):
    if not np.isfinite(abs_q) or abs_q < 0: raise ValueError('Bad calibration quantile')
    if abs_q == 0: return max_bits
    return int(min(max_bits, max(0, math.floor(math.log2(max_int / abs_q)))))


def calibrate(model, refs, mean, std, percentile=99.9):
    seen={k:[] for k in ('input',)+LAYER_ORDER}
    cache={}
    layers={
        'project':model.project,
        'dw1':model.dw1,
        'pw1':model.pw1,
        'dw2':model.dw2,
        'pw2':model.pw2,
        'head':model.head,
    }
    hooks=[]
    for name,mod in layers.items():
        def hook(module,inp,out,n=name):
            o=out.detach().cpu().numpy()
            if n in ('project','pw1','pw2'):
                o=np.maximum(o,0.0)
            cache[n]=o
        hooks.append(mod.register_forward_hook(hook))
    try:
        for ref in refs:
            raw,_=read_trial(ref)
            x=np.clip((raw-mean)/std,-8,8).astype(np.float32)
            # Bounded deterministic sampling; never retain full session tensors.
            seen['input'].append(np.abs(x.reshape(-1)[::64]).copy())
            with torch.no_grad():
                model(torch.from_numpy(x.copy())[None])
            for name in LAYER_ORDER:
                seen[name].append(np.abs(cache[name].reshape(-1)[::16]).copy())
            cache.clear()
    finally:
        for hook in hooks:hook.remove()
    q={name:float(np.percentile(np.concatenate(vals),percentile)) for name,vals in seen.items()}
    # INT8 activation output headroom; head is stored in signed INT16 and
    # can preserve substantially more precision than the hidden layers.
    bits={'input':select_bits(q['input'])}
    for name in LAYER_ORDER:
        bits[name]=8 if name=='head' else select_bits(q[name])
    return q,bits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-trials',type=int,default=64)
    p.add_argument('--percentile',type=float,default=99.9)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if not 0<a.percentile<=100:p.error('percentile must be (0,100]')
    if a.max_trials<=0:p.error('max-trials must be positive')
    torch.set_num_threads(a.threads)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    m=CompactCTC(hidden=int(ck['architecture']['hidden']))
    m.load_state_dict(ck['state_dict'],strict=True);m.eval()
    refs=list_labeled_trials(a.train,a.max_trials)
    mean=np.asarray(ck['mean'],np.float32)
    std=np.asarray(ck['std'],np.float32)
    quantiles,bits=calibrate(m,refs,mean,std,a.percentile)
    profile={
        'contract':'neuroedge-train-only-power2-activation-calibration',
        'input_frac_bits':bits['input'],
        'output_frac_bits':{k:bits[k] for k in LAYER_ORDER},
        'observed_abs_percentile':quantiles,
        'percentile':a.percentile,
        'train_calibration_trials':len(refs),
        'calibration_hdf5_sha256':sha256_file(a.train),
        'checkpoint_sha256':sha256_file(a.checkpoint),
        'note':'Train-only PTQ scales. Does not demonstrate recovered accuracy or QAT.'
    }
    # Fail BEFORE exporting incompatible RTL shifts.
    quantize_model(m,profile)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(profile,indent=2)+'\n')
    print('CALIBRATED_INT8_PROFILE_PASS')
    print('FRACTION_BITS=',json.dumps(bits,sort_keys=True))
    print('CALIBRATION_SOURCE=train_only',len(refs),'trials')

if __name__=='__main__':main()
