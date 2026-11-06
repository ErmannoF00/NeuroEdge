#!/usr/bin/env python3
"""Validate and summarize a *local* connected-stream trial report.

No participant arrays, decoded texts, identifiers or absolute paths are emitted.
This tool does not infer causal efficacy, confidence intervals or generalization.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

MODES = ('off', 'sham', 'supervised')


def _check_number(x: object, field: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f'invalid {field}')
    return float(x)


def inspect(report: dict) -> dict:
    modes = report.get('mode_results')
    if not isinstance(modes, dict) or any(name not in modes for name in MODES):
        raise ValueError('missing frozen/sham/supervised controls')
    summary: dict = {'interpretation': 'DESCRIPTIVE ONLY; no demonstrated benefit unless tested on held-out trials',
                     'sensitive_content_included': False,
                     'feedback_source': report.get('feedback_supervision'),
                     'modes': {}}
    ntrials = nbins = nsteps = None
    for name in MODES:
        mode = modes[name]
        trials, bins, outputs = [mode[k] for k in ('trials','bins','outputs')]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (trials,bins,outputs)):
            raise ValueError(f'{name}: invalid sample size')
        if ntrials is None: ntrials,nbins,nsteps = trials,bins,outputs
        elif (trials,bins,outputs) != (ntrials,nbins,nsteps):
            raise ValueError('different trial/bin/output counts across arms')
        if not isinstance(mode.get('records'), list) or len(mode['records']) != trials:
            raise ValueError(f'{name}: incomplete per-trial records')
        if any(rec.get('released') or rec.get('released_phonemes') is not None
               for rec in mode['records']):
            raise ValueError('unarmed local study has output releases; do not summarize automatically')
        writes = mode['committed_writes']
        if isinstance(writes, bool) or not isinstance(writes,int) or writes < 0:
            raise ValueError('invalid write count')
        target = mode.get('target_phonemes')
        edits = mode.get('ctc_edits')
        per = mode.get('greedy_per')
        if target is not None:
            if not isinstance(target,int) or target < 1 or not isinstance(edits,int) or edits < 0:
                raise ValueError('invalid edit/target totals')
            calc = edits / target
            if per is None or abs(_check_number(per, 'greedy_per') - calc) > 1e-10:
                raise ValueError('PER mismatch with integer error count')
        elif per is not None or edits is not None:
            raise ValueError('partial PER fields')
        summary['modes'][name] = {'greedy_per': per, 'edits': edits,
                                  'target_phonemes': target, 'committed_writes': writes}
    summary['trials'] = ntrials
    summary['bins'] = nbins
    summary['ctc_steps'] = nsteps
    summary['scalar_logits'] = nsteps * 41
    frozen = summary['modes']['off']['greedy_per']
    sup = summary['modes']['supervised']['greedy_per']
    summary['supervised_minus_frozen_per'] = None if frozen is None or sup is None else sup - frozen
    summary['supervised_commits'] = summary['modes']['supervised']['committed_writes']
    summary['supervised_benefit_proven'] = False
    summary['notes'] = ('No policy update accepted: the supervised run is not an adaptation success.'
                        if summary['supervised_commits'] == 0 else
                        'Writes occurred; prospective held-out effectiveness is unestablished.')
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path, help='LOCAL generated stream report in ignored build/')
    p.add_argument('--out', type=Path, help='Optional privacy-filtered summary in build/')
    args = p.parse_args()
    result = inspect(json.loads(args.report.read_text()))
    print('NEUROEDGE_LOCAL_REPLAY_AUDIT_PASS')
    print(f"  trials={result['trials']} bins={result['bins']} scalar_logits={result['scalar_logits']}")
    for name, vals in result['modes'].items():
        print(f"  {name}: PER={vals['greedy_per']} commits={vals['committed_writes']}")
    print('  ' + result['notes'])
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print('  summary_saved=',args.out)

if __name__ == '__main__': main()
