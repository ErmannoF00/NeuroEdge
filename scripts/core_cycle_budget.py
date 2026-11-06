#!/usr/bin/env python3
"""Peak one-MAC/cycle schedule projection, NOT a routed timing result.

Includes 512 feature-write cycles and RTL shifting, excluding stalls, SRAM
read latency, cache/DMA and first-time parameter configuration. System clock
rate is a user hypothesis until mapped STA proves it at the chosen corner.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def estimate(nbins:int,frequency_mhz:float):
    if nbins<1 or frequency_mhz<=0:raise ValueError('invalid cycle budget')
    result=[]
    for t in range(nbins):
        # One clock per MAC, one per input byte, one per history channel,
        # one per output handshake with always-ready receiver.
        core=512+32768+64
        if t%2==0:core+=320+4096+64
        if t%4==0:core+=320+4096+2624+41
        result.append(core)
    return {
        'bins':nbins,'frequency_mhz_hypothetical':frequency_mhz,
        'total_ideal_cycles':sum(result),
        'mean_cycles_per_bin':sum(result)/len(result),
        'worst_case_cycles_per_bin':max(result),
        'mean_latency_ms':sum(result)/len(result)/(frequency_mhz*1000),
        'peak_bin_processing_ms':max(result)/(frequency_mhz*1000),
        'neural_bin_period_ms':20,
        'timing_feasible_under_assumed_1cycle_mac':max(result)/(frequency_mhz*1000)<20,
        'status':'IDEAL FULL-DECODER CYCLE MODEL ONLY; 25MHz not proven; no mapped SRAM/STA/placement/power.'
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bins',type=int,default=8)
    p.add_argument('--clock-mhz',type=float,default=25)
    p.add_argument('--out',type=Path,default=Path('results/asic/cycle_budget.json'))
    a=p.parse_args()
    metrics=estimate(a.bins,a.clock_mhz)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    print('IDEAL_CYCLE_MODEL_PASS (not silicon timing)')
if __name__=='__main__':main()
