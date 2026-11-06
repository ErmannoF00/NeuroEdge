"""Idealized lower bounds for projection layer, not chip cycle/area measurement."""
import json
from pathlib import Path

def banking_sweep(n_inputs=512,n_outputs=64, max_banks=64):
    if n_inputs<1 or n_outputs<1 or max_banks<1: raise ValueError
    rows=[]
    for banks in [1,2,4,8,16,32,64]:
        if banks>max_banks: continue
        # Each bank independently executes one row, two compute cycles per feature.
        waves=(n_outputs+banks-1)//banks
        lower_bound=2*n_inputs*waves
        rows.append({'banks':banks,'weight_bits':n_inputs*n_outputs*8,
                     'cycles_compute_lower_bound':lower_bound,
                     'projection_ms_at_assumed_25MHz':round(lower_bound/25000,6),
                     'macs':n_inputs*n_outputs})
    return rows

if __name__=='__main__':
    print(json.dumps({'assumptions':'Independent per-row banks, no arbitration, loading, control or routing cost; NOT timing signoff',
                      'projection':banking_sweep()},indent=2))
