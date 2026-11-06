#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neuroedge.device import DeviceConfig,map_weight_to_device
ap=argparse.ArgumentParser(); ap.add_argument('--states',default='4,8,16,32'); ap.add_argument('--sigmas',default='0.0,0.02,0.05,0.1'); ap.add_argument('--out',default='results/device_sweep.csv')
a=ap.parse_args(); w=torch.linspace(-1,1,4096)
rows=[]
for s in map(int,a.states.split(',')):
  for sig in map(float,a.sigmas.split(',')):
    cfg=DeviceConfig(s,sig,sig,0); q=map_weight_to_device(w,cfg,1); mse=float(torch.mean((q-w)**2)); rows.append((s,sig,mse))
p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
with p.open('w',newline='') as f:
  cw=csv.writer(f); cw.writerow(['states','sigma_rel','weight_mse']); cw.writerows(rows)
print(p)
