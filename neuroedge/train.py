from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from .data import synthetic_bci, delta_events
from .models import TinyQNN, TinySNN, DiagonalSSM
from .device import perturb_model, FEFET_EXAMPLE, RRAM_EXAMPLE


def build(name, channels, classes):
    if name == 'qnn': return TinyQNN(channels=channels, hidden=32, classes=classes)
    if name == 'snn': return TinySNN(channels=channels, hidden=32, classes=classes)
    if name == 'ssm': return DiagonalSSM(channels=channels, state=32, classes=classes)
    raise ValueError(name)

@torch.no_grad()
def accuracy(model, loader, event=False):
    model.eval(); n=ok=0
    for x,y in loader:
        if event: x=delta_events(x)
        pred=model(x).argmax(-1)
        ok += int((pred==y).sum()); n += y.numel()
    return ok/max(n,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['qnn','snn','ssm'],default='qnn')
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--samples',type=int,default=512)
    ap.add_argument('--channels',type=int,default=16)
    ap.add_argument('--timesteps',type=int,default=64)
    ap.add_argument('--classes',type=int,default=4)
    ap.add_argument('--event-input',action='store_true')
    ap.add_argument('--out',default='results/model.pt')
    args=ap.parse_args()
    torch.manual_seed(4)
    ds=synthetic_bci(args.samples,args.timesteps,args.channels,args.classes)
    ntr=int(0.8*len(ds)); tr,va=random_split(ds,[ntr,len(ds)-ntr],generator=torch.Generator().manual_seed(1))
    tl=DataLoader(tr,batch_size=32,shuffle=True); vl=DataLoader(va,batch_size=64)
    model=build(args.model,args.channels,args.classes)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
    for e in range(args.epochs):
        model.train()
        for x,y in tl:
            if args.event_input or args.model=='snn': x=delta_events(x)
            loss=torch.nn.functional.cross_entropy(model(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
    event=args.event_input or args.model=='snn'
    base=accuracy(model,vl,event)
    fe=accuracy(perturb_model(model,FEFET_EXAMPLE,11),vl,event)
    rr=accuracy(perturb_model(model,RRAM_EXAMPLE,22),vl,event)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'model':args.model,'channels':args.channels,'classes':args.classes},out)
    metrics={'model':args.model,'accuracy':base,'fefet_example_accuracy':fe,'rram_example_accuracy':rr,'note':'device parameters are illustrative'}
    out.with_suffix('.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
