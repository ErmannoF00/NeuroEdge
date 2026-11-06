#!/usr/bin/env python3
"""Summarize SPICE/external-model conductance measurements WITHOUT inventing data.

Input rows must be supplied by user from external simulator, and provenance
recorded explicitly. This converter is not a JART compact model or simulator.
"""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import defaultdict
import numpy as np

REQUIRED = {'state','conductance_s','read_energy_j','program_energy_j',
            'read_latency_s','program_latency_s'}


def characterize(csv_path: str, out_path: str, source_model: str, model_sha256: str):
    inp = Path(csv_path).resolve()
    rows = list(csv.DictReader(inp.open(newline='')))
    if not rows:
        raise ValueError('No measurement rows')
    if not REQUIRED.issubset(set(rows[0])):
        raise ValueError(f'Missing columns: {sorted(REQUIRED - set(rows[0]))}')
    if len(model_sha256) != 64 or not all(c in '0123456789abcdef' for c in model_sha256.lower()):
        raise ValueError('model SHA256 must be exactly 64 hex characters')
    by_state = defaultdict(list)
    for n,row in enumerate(rows):
        vals = {name:float(row[name]) for name in REQUIRED-{'state'}}
        if not all(np.isfinite(x) and x>=0 for x in vals.values()):
            raise ValueError(f'Invalid negative/nonfinite SPICE measurement row {n}')
        if vals['conductance_s']<=0:
            raise ValueError(f'Conductance must be positive row {n}')
        by_state[int(row['state'])].append(vals)
    summary = {}
    for state, group in sorted(by_state.items()):
        summary[str(state)]={'samples':len(group)}
        for col in sorted(REQUIRED-{'state'}):
            x=np.array([r[col] for r in group])
            summary[str(state)][col]={'mean':float(np.mean(x)),
                                      'std':float(np.std(x,ddof=1)) if len(x)>1 else None,
                                      'p05':float(np.quantile(x,0.05)),
                                      'p95':float(np.quantile(x,0.95))}
    result = {
        'source_model':source_model,'source_model_sha256':model_sha256.lower(),
        'input_measurement_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
        'source_path':str(inp),'state_statistics':summary,
        'note':'Measured/modelled numbers transcribed from supplied CSV; no source-specific physics inferred.'}
    dest=Path(out_path); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--source-model',required=True)
    ap.add_argument('--model-sha256',required=True)
    args=ap.parse_args()
    out=characterize(args.csv,args.out,args.source_model,args.model_sha256)
    print(f"CHARACTERIZATION_IMPORTED states={len(out['state_statistics'])} source={out['source_model']}")
