#!/usr/bin/env python3
"""Reproduce a causal, supervised *synthetic* calibration feedback replay."""
from __future__ import annotations
import argparse
from pathlib import Path
from neuroedge.control.closed_loop_replay import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('build/closed_loop_replay.json'))
    ap.add_argument('--feedback-delay-trials', type=int, default=0)
    args = ap.parse_args()
    report = write_report(args.out, feedback_delay_trials=args.feedback_delay_trials)
    print('NEUROEDGE_SYNTHETIC_LAGGED_FEEDBACK_PASS'
          f' fixed_l1={report["fixed_logit_l1"]}'
          f' controlled_l1={report["controlled_logit_l1"]}'
          f' accepted_writes={report["accepted_memory_writes"]}'
          f' evidence={args.out}')

if __name__ == '__main__':
    main()
