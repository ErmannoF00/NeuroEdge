#!/usr/bin/env python3
"""Produce actual neural-feature vectors for fixed-point affine hardware checking.

Only frozen source-model input normalization and host-estimated coefficients;
no claim that calibration training happens in RTL.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import torch

FRAC=12
SCALE=16    # normalized-feature Q scale: 1.0 normal unit -> INT8 value 16


def clip_i8(x):
    return int(np.clip(x,-128,127))


def fixed_affine(x, g, b):
    return int(max(-128,min(127,(int(x)*int(g)+int(b)) >> FRAC)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--ablation',required=True,type=Path)
    p.add_argument('--validation',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--method',choices=['fixed','unsup_offset','unsup_affine'],default='unsup_affine')
    p.add_argument('--bins',type=int,default=8)
    a=p.parse_args()
    if a.bins <= 0: p.error('bins must be positive')
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    obj=json.loads(a.ablation.read_text())
    if obj['development_val_sha256'] != sha256(a.validation):
        raise ValueError('The input validation HDF5 does not match ablation provenance')
    coeff=obj['methods'][a.method]
    g=np.asarray(coeff['gain'],dtype=np.float64)
    b=np.asarray(coeff['offset'],dtype=np.float64)
    if g.shape != (512,) or b.shape != (512,): raise ValueError('512 coefficients expected')
    gainq=np.rint(g*(1<<FRAC)).astype(np.int64)
    biasq=np.rint(b*SCALE*(1<<FRAC)).astype(np.int64)
    if np.max(np.abs(gainq)) >= (1<<23) or np.max(np.abs(biasq)) >= (1<<23):
        raise ValueError('Fixed-point coefficient exceeds signed 24-bit range')
    with h5py.File(a.validation,'r') as f:
        key=sorted(f.keys())[0]
        grp=f[key]
        t=min(a.bins,int(grp.attrs['n_time_steps']))
        raw=np.asarray(grp['input_features'][:t],dtype=np.float64)
    mean=cp['mean'].numpy().astype(np.float64)
    std=cp['std'].numpy().astype(np.float64)
    norm=np.clip((raw-mean)/std,-8,8)
    x=np.clip(np.rint(norm*SCALE),-128,127).astype(np.int64)
    rows=[]
    for time_i in range(len(x)):
        for channel in range(512):
            xi=int(x[time_i,channel]); gi=int(gainq[channel]); bi=int(biasq[channel])
            rows.append((xi,gi,bi,fixed_affine(xi,gi,bi)))
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'affine_vectors.txt'
    with path.open('w') as f:
        f.write(f'{len(rows)}\n')
        for row in rows: f.write(' '.join(map(str,row))+'\n')
    (a.out/'affine_contract.json').write_text(json.dumps({
        'source_checkpoint':str(a.checkpoint),
        'validation_sha256':sha256(a.validation),
        'method':a.method,
        'bins':len(x),'channels_per_bin':512,
        'samples':len(rows),
        'input_scale':SCALE,
        'coeff_fraction_bits':FRAC,
        'rounding':'arithmetic_right_shift_towards_negative_infinity',
        'output':'signed_int8_saturated',
        'host_estimated_parameters_not_hardware_learning':True
    },indent=2)+'\n')
    print(f'AFFINE_VECTORS_EXPORTED_PASS samples={len(rows)} method={a.method}')


def sha256(path):
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

if __name__=='__main__': main()
