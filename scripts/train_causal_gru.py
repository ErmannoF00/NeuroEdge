#!/usr/bin/env python3
"""Leakage-audited causal GRU CTC comparison; experimental, NOT existing RTL.
Reuses exact CTC objective, data loading and train-only normalization of baseline.
Results are local only and must be compared at matched computational budgets.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from research.neural.causal_gru import CausalGRUCTC
from neuroedge.speech.dataset import list_labeled_trials, training_channel_stats, SpeechTrials, collate_ctc
from scripts.train_speech_ctc import run_epoch, sha256_file

def main():
    ap=argparse.ArgumentParser()
    for x in ('train','val','out'):ap.add_argument('--'+x,required=True,type=Path)
    ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--lr',type=float,default=0.0005)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=ap.parse_args()
    if args.train.resolve()==args.val.resolve():ap.error('same training and validation file')
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    if min(args.hidden,args.depth,args.epochs,args.batch_size)<=0:ap.error('dimensions must be positive')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    train_refs=list_labeled_trials(args.train)
    val_refs=list_labeled_trials(args.val)
    if {(r.file,r.key) for r in train_refs} & {(r.file,r.key) for r in val_refs}:
        raise ValueError('overlapping labeled training/validation trials')
    mean,std=training_channel_stats(train_refs)
    train=SpeechTrials(train_refs,mean,std);val=SpeechTrials(val_refs,mean,std)
    t_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate_ctc)
    v_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,collate_fn=collate_ctc)
    model=CausalGRUCTC(hidden=args.hidden,depth=args.depth).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={'model':'causal_gru_experimental','training_file_sha256':sha256_file(args.train),
              'development_file_sha256':sha256_file(args.val),'train_trials':len(train),
              'development_trials':len(val),'normalization':'TRAIN-ONLY',
              'labels_used_for_training':True,'model_selection_on_development':True,
              'no_test_holdout_claim':True,'greedy_per_no_language_model':True,
              'hardware_export_available':False,'parameter_count':sum(p.numel() for p in model.parameters()),
              'architecture':{'hidden':args.hidden,'depth':args.depth,'stride':4},'seed':args.seed}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    history=[]; best=float('inf')
    for epoch in range(1,args.epochs+1):
        tr=run_epoch(model,t_loader,optimizer,args.device)
        va=run_epoch(model,v_loader,None,args.device)
        history.append({'epoch':epoch,'train':tr,'development':va})
        if va['ctc_nll_per_target']<best:
            best=va['ctc_nll_per_target']
            torch.save({'model_state':model.state_dict(),'architecture':manifest['architecture'],
                        'train_mean':mean,'train_std':std,'manifest':manifest,'best_epoch':epoch},
                       args.out/'best_model.pt')
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('GRU_EPOCH',epoch,'development_PER',va['phoneme_error_rate_greedy_no_lm'],flush=True)
    print('GRU_EXPERIMENT_COMPLETE_LOCAL_NOT_RTL_OR_CLINICAL')
if __name__=='__main__':main()
