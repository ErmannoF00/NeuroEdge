#!/usr/bin/env python3
"""Research control: compare original float vs full INT8 greedy PER on labeled trials.

This prevents confusing RTL bit exactness with useful neural decoding performance.
Untouched heldout validation only; do not use results to claim final drift benefit.
"""
from neuroedge.contracts import normalize_contract
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from neuroedge.speech.ctc import ctc_collapse,edit_distance
from neuroedge.speech.dataset import list_labeled_trials,read_trial
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.fixed_decoder import quantize_input,quantize_model,FixedStreamingDecoder


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--val',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-trials',type=int,default=0,help='0 means ALL')
    ap.add_argument('--quant-profile',type=Path,default=None,
                    help='Optional TRAIN-calibrated fixed-point deployment profile')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.max_trials<0:ap.error('--max-trials must be >=0')
    torch.set_num_threads(args.threads)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    m=CompactCTC(hidden=int(ck['architecture']['hidden']))
    m.load_state_dict(ck['state_dict'],strict=True);m.eval()
    profile=json.loads(args.quant_profile.read_text()) if args.quant_profile else None
    if profile is not None:
        if normalize_contract(profile.get('contract'))!='neuroedge-train-only-power2-activation-calibration':
            raise ValueError('Unsupported or unaudited quantization profile')
        def filehash(p):
            h=hashlib.sha256()
            with open(p,'rb') as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b)
            return h.hexdigest()
        if profile['checkpoint_sha256']!=filehash(args.checkpoint):
            raise ValueError('Quantization profile belongs to a different checkpoint')
        if profile['calibration_hdf5_sha256']==filehash(args.val):
            raise ValueError('Validation and calibration files must differ')
    params=quantize_model(m,profile)
    input_frac_bits=int(profile['input_frac_bits']) if profile else 4
    refs=list_labeled_trials(args.val,args.max_trials)
    mean=np.asarray(ck['mean'],dtype=np.float32)
    std=np.asarray(ck['std'],dtype=np.float32)
    tally={'float_edits':0,'integer_edits':0,'target_tokens':0,
           'float_blank':0,'integer_blank':0,'output_steps':0,'agreement':0}
    sat={k:{'hits':0,'n':0} for k in ('project','dw1','pw1','dw2','pw2','head')}
    for r in refs:
        raw,truth=read_trial(r)
        x=np.clip((raw-mean)/std,-8,8).astype(np.float32)
        with torch.no_grad():
            out=m(torch.from_numpy(x.copy())[None])[0].argmax(dim=1).numpy().tolist()
        fixed=FixedStreamingDecoder(params)
        logits=fixed.run(quantize_input(x,frac_bits=input_frac_bits))
        out_q=np.argmax(logits,axis=1).tolist()
        if len(out)!=len(out_q):raise AssertionError('FLOAT vs INT output timeline differs')
        tgt=truth.tolist()
        tally['float_edits']+=edit_distance(ctc_collapse(out),tgt)
        tally['integer_edits']+=edit_distance(ctc_collapse(out_q),tgt)
        tally['target_tokens']+=len(tgt)
        tally['float_blank']+=out.count(0)
        tally['integer_blank']+=out_q.count(0)
        tally['output_steps']+=len(out)
        tally['agreement']+=sum(a==b for a,b in zip(out,out_q))
        for k in sat:
            sat[k]['hits']+=fixed.saturation_hits[k]
            sat[k]['n']+=fixed.layer_outputs[k]
    metrics={
        'float_greedy_PER':tally['float_edits']/tally['target_tokens'],
        'int_greedy_PER':tally['integer_edits']/tally['target_tokens'],
        'delta_int_minus_float':(tally['integer_edits']-tally['float_edits'])/tally['target_tokens'],
        'logit_argmax_agreement':tally['agreement']/tally['output_steps'],
        'float_blank_fraction':tally['float_blank']/tally['output_steps'],
        'int_blank_fraction':tally['integer_blank']/tally['output_steps'],
        'trials':len(refs),
        'input_frac_bits':input_frac_bits,
        'quantization_profile':str(args.quant_profile) if args.quant_profile else 'default',
        'all_val_trials_evaluated':args.max_trials==0,
        'per_layer_saturation':sat,
        'research_warning':'INT8 inference comparison ONLY; not trained QAT; not a clinical benchmark; no RTL power characterization.',
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'quantized_per.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('FLOAT_PER=',round(metrics['float_greedy_PER'],6))
    print('INTEGER_PER=',round(metrics['int_greedy_PER'],6))
    print('DELTA_PER=',round(metrics['delta_int_minus_float'],6))
    print('QUANTIZED_CTC_EVALUATION_PASS trials=',len(refs))
if __name__=='__main__':main()
