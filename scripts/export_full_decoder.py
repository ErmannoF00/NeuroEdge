#!/usr/bin/env python3
"""Export ALL six causal CompactCTC layers + N consecutive *real* 512-channel bins to RTL.

Write artifacts under ignored results/, never publish neural features or trial text.
Passing HW equivalence proves fixedpoint kernel fidelity, NOT float/INT8 PER.
"""
from __future__ import annotations
from neuroedge.contracts import normalize_contract
import argparse,hashlib,json
from pathlib import Path
import torch
import numpy as np
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.dataset import list_labeled_trials,read_trial
from neuroedge.speech.fixed_decoder import (
    FixedStreamingDecoder,quantize_model,quantize_input,write_rtl_vectors,
    TOTAL_WEIGHTS,TOTAL_META,ACT_SCALE,
)

def checksum(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--hdf5',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--trial',default=None)
    ap.add_argument('--frames',type=int,default=8)
    ap.add_argument('--quant-profile',type=Path,default=None)
    args=ap.parse_args()
    if args.frames < 8 or args.frames > 128 or args.frames % 4:
        ap.error('frames must be a multiple of 4 between 8 and 128')
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=CompactCTC(hidden=int(ck['architecture']['hidden']))
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    profile=json.loads(args.quant_profile.read_text()) if args.quant_profile else None
    if profile is not None:
        if normalize_contract(profile.get('contract'))!='neuroedge-train-only-power2-activation-calibration':
            raise ValueError('Incompatible quantization profile')
        if profile['checkpoint_sha256']!=checksum(args.checkpoint):
            raise ValueError('Profile checkpoint SHA mismatch')
        if profile['calibration_hdf5_sha256']==checksum(args.hdf5):
            raise ValueError('Cannot export using calibration training data as validation')
    params=quantize_model(model,profile)
    input_frac_bits=int(profile['input_frac_bits']) if profile else 4
    trials=list_labeled_trials(args.hdf5)
    ref=next((r for r in trials if r.key==args.trial),None) if args.trial else trials[0]
    if ref is None: raise ValueError(f'unknown trial {args.trial}')
    raw,_=read_trial(ref)
    if len(raw)<args.frames: raise ValueError('Not enough bins')
    mean=np.asarray(ck['mean'],dtype=np.float32)
    std=np.asarray(ck['std'],dtype=np.float32)
    x=np.clip((raw[:args.frames]-mean)/std,-8,8)
    qx=quantize_input(x,frac_bits=input_frac_bits)
    decoder=FixedStreamingDecoder(params)
    y=decoder.run(qx)
    # Architectural functional check only, not equal float/int8 predictions.
    with torch.no_grad():
        float_logits=model(torch.from_numpy(x.copy())[None])[0].numpy()
    if y.shape!=float_logits.shape:
        raise AssertionError(f'Stride mismatch int={y.shape} float={float_logits.shape}')
    write_rtl_vectors(args.out,params,qx,y)
    contract={
        'version':'neuroedge-full-quantized-inference',
        'source_type':'patient-derived intracortical features; DO NOT PUBLISH input_frames.mem',
        'checkpoint_sha256':checksum(args.checkpoint),
        'source_hdf5_sha256':checksum(args.hdf5),
        'source_trial_key':ref.key,
        'frames':args.frames,
        'output_steps':len(y),
        'output_classes':y.shape[1],
        'weight_bytes':TOTAL_WEIGHTS,
        'bias_entries':TOTAL_META,
        'shift_entries':TOTAL_META,
        'input_scale': 2.0**(-input_frac_bits),
        'output_frac_bits': profile['output_frac_bits'] if profile else {k:4 for k in ('project','dw1','pw1','dw2','pw2','head')},
        'quant_profile': str(args.quant_profile) if args.quant_profile else 'default',
        'float_argmax':np.argmax(float_logits,axis=1).tolist(),
        'int16_argmax':np.argmax(y,axis=1).tolist(),
        'saturation_per_layer':{k:{'hits':decoder.saturation_hits[k],
                                   'n':decoder.layer_outputs[k]} for k in decoder.saturation_hits},
        'provenance':{
            'input_features':'real HDF5, checkpoint train-only normalization',
            'network':'6 layers, DW1 and DW2 temporal stride 2; causal',
            'precision':'INT8 MAC/INT32 accumulator/INT16 CTC logits',
            'software_hardware_check':'must run scripts/verify_full_decoder.sh',
            'not_yet_verified':'RTL simulation, quantized PER and ASIC PPA on this machine',
        },
    }
    (args.out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(f'FULL_DECODER_VECTORS_PASS frames={args.frames} logits={y.size} params=44224 real_neural=True')
    print('SATURATION_DIAGNOSTIC=',json.dumps(contract['saturation_per_layer']))
    print('VECTORS=',args.out)
if __name__=='__main__':main()
