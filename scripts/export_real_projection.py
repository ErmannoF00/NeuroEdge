#!/usr/bin/env python3
"""Export actual trained projection weights + one real normalized neural frame to RTL.

The DUT is ONLY the first affine dot-product (bias/ReLU/temporal/CTC are NOT RTL).
Generated vectors include patient-derived data and must remain under ignored results/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from neuroedge.speech.dataset import TrialRef, read_trial
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.quant import quantize_weights,quantize_activation,project_int32


def write_mem(path, arr, hex_digits):
    path.write_text(''.join(f'{int(v) & ((1 << (4*hex_digits)) - 1):0{hex_digits}x}\n' for v in arr))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--hdf5',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trial',default=None,help='default: first labeled trial')
    p.add_argument('--frame',type=int,default=0)
    args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=CompactCTC(hidden=int(ckpt['architecture']['hidden']))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    w=model.project.weight.detach().numpy()
    if w.shape != (64,512):
        raise ValueError('previous prototype RTL core fixed to 64x512; train with --hidden 64')
    with h5py.File(args.hdf5,'r') as f:
        key=args.trial or sorted(f.keys())[0]
        if key not in f: raise ValueError(f'Missing trial: {key}')
        g=f[key]
        ref=TrialRef(str(args.hdf5),key,int(g.attrs['n_time_steps']),int(g.attrs['seq_len']))
    x,_=read_trial(ref)
    if not 0<=args.frame<len(x): raise ValueError('frame outside trial')
    mean=ckpt['mean'].detach().numpy()
    std=ckpt['std'].detach().numpy()
    norm=np.clip((x[args.frame]-mean)/std,-8,8)
    wq,sw=quantize_weights(w)
    xq=quantize_activation(norm)
    y=project_int32(wq,xq)
    args.out.mkdir(parents=True,exist_ok=True)
    write_mem(args.out/'weights.mem',wq.reshape(-1),2)
    write_mem(args.out/'features.mem',xq,2)
    write_mem(args.out/'expected.mem',y,8)
    data={'source': 'REAL neural data, train-normalized',
          'projection_only_not_whole_model':True,
          'mac_count':32768,'input_dims':512,'output_dims':64,
          'quantization':'signed INT8, symmetric per-row weight scales, x_scale=1/16',
          'bias_not_included':True,'temporal_not_included':True,
          'scale_x':1/16,'weight_scales':sw.tolist(),
          'trial_key':key,'frame_idx':args.frame}
    (args.out/'contract.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f'PROJECTION_EXPORT_PASS macs=32768 outputs=64 frame={args.frame}')
    print('RTL_VECTORS=',args.out)
if __name__=='__main__':main()
