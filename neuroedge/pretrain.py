from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import synthetic_bci
from .models import MaskedSSM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--samples',type=int,default=512)
    ap.add_argument('--channels',type=int,default=16); ap.add_argument('--timesteps',type=int,default=64)
    ap.add_argument('--mask-frac',type=float,default=0.5); ap.add_argument('--out',default='results/pretrained_ssm.pt')
    a=ap.parse_args(); torch.manual_seed(3)
    ds=synthetic_bci(a.samples,a.timesteps,a.channels); dl=DataLoader(ds,batch_size=32,shuffle=True)
    m=MaskedSSM(a.channels,32); opt=torch.optim.Adam(m.parameters(),3e-3)
    for _ in range(a.epochs):
        for x,_ in dl:
            channel_target=x.mean(dim=1)
            mask=(torch.rand(x.size(0),a.channels)<a.mask_frac)
            xin=x.clone()
            # independent channel mask per sample
            for b in range(x.size(0)): xin[b,:,mask[b]]=0
            pred=m(xin)
            loss=((pred-channel_target)**2 * mask.float()).sum()/mask.float().sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); torch.save(m.state_dict(),p)
    print(f'pretraining_loss={float(loss):.6f}')
if __name__=='__main__': main()
