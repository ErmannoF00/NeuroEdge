from __future__ import annotations
import csv, json
from pathlib import Path
from .train import build, accuracy
from .data import synthetic_bci
from torch.utils.data import DataLoader
import torch

MODELS=('qnn','snn','ssm')

def run(out='results/smoke_benchmark.csv', samples=256, timesteps=48, channels=16, classes=4, epochs=2):
    from .data import delta_events
    ds=synthetic_bci(samples,timesteps,channels,classes)
    dl=DataLoader(ds,batch_size=64)
    rows=[]
    for name in MODELS:
        torch.manual_seed(7)
        model=build(name,channels,classes)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
        for _ in range(epochs):
            for x,y in dl:
                event=(name=='snn')
                xx=delta_events(x) if event else x
                loss=torch.nn.functional.cross_entropy(model(xx),y)
                opt.zero_grad(); loss.backward(); opt.step()
        acc=accuracy(model,dl,event=(name=='snn'))
        rows.append({'model':name,'synthetic_ci_accuracy':acc,'scientific_result':False})
    p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    p.with_suffix('.json').write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
