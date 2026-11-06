#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neuroedge.fixed import symmetric_quantize

ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--bits',type=int,default=8)
a=ap.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu')
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
manifest={}
for name,w in ck['state_dict'].items():
    if not torch.is_floating_point(w): continue
    q,s=symmetric_quantize(w,a.bits)
    flat=q.flatten().tolist(); fn=f'{out}_{name.replace(".","_")}.mem'
    width=(a.bits+3)//4; mask=(1<<a.bits)-1
    with open(fn,'w') as f:
        for v in flat: f.write(f'{(int(v)&mask):0{width}x}\n')
    manifest[name]={'file':fn,'shape':list(w.shape),'scale':float(s),'bits':a.bits}
Path(f'{out}_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'exported {len(manifest)} tensors')
