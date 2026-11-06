#!/usr/bin/env python3
"""Evaluate a saved checkpoint on labeled HDF5 only (no LM, report PER)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from neuroedge.speech.dataset import list_labeled_trials,SpeechTrials,collate_ctc
from neuroedge.speech.model import CompactCTC
from train_speech_ctc import run_epoch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--val',required=True,type=Path)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    expected_sha=cp['data_manifest']['val_sha256']
    actual_sha=sha(a.val)
    if expected_sha!=actual_sha:
        raise ValueError('Dataset SHA mismatch: refusing to evaluate another validation file')
    model=CompactCTC(hidden=int(cp['architecture']['hidden']))
    model.load_state_dict(cp['state_dict'],strict=True)
    refs=list_labeled_trials(a.val)
    ds=SpeechTrials(refs,cp['mean'].numpy(),cp['std'].numpy())
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,collate_fn=collate_ctc,num_workers=0)
    metrics=run_epoch(model,loader,None,'cpu')
    a.out.mkdir(parents=True,exist_ok=True)
    results={'evaluation':'checkpoint heldout validation', 'metrics':metrics,
             'val_sha256':actual_sha,'best_epoch':cp['best_epoch'],
             'not_wer':True,'not_official_rnn':True}
    (a.out/'checkpoint_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print('VALIDATION_PER=',round(metrics['phoneme_error_rate_greedy_no_lm'],4))
    print('HELDOUT_SPEECH_CHECKPOINT_EVAL_PASS')
if __name__=='__main__':main()
