"""Audit the real-neural summary tool using synthetic JSON only."""
from copy import deepcopy
from scripts.inspect_neural_replay import inspect
import pytest


def report():
    recs = [{'trial_id': i,'released': False, 'released_phonemes': None} for i in range(4)]
    return {'feedback_supervision':'post-trial oracle',
            'mode_results': {
                name: {'trials':4,'bins':3123,'outputs':783,
                       'target_phonemes':116,'ctc_edits':e,'greedy_per':e/116,
                       'committed_writes':writes,'records':deepcopy(recs)}
                for name,e,writes in [('off',69,0),('sham',70,3),('supervised',69,0)]}}


def test_four_trial_negative_result():
    s = inspect(report())
    assert s['scalar_logits'] == 32103 and s['supervised_commits'] == 0
    assert s['supervised_minus_frozen_per'] == 0
    assert not s['supervised_benefit_proven']
    assert 'not an adaptation success' in s['notes']


def test_reject_mismatched_denominator():
    r=report();r['mode_results']['sham']['greedy_per']=0.99
    with pytest.raises(ValueError,match='PER mismatch'): inspect(r)


def test_reject_unequal_data_counts():
    r=report();r['mode_results']['sham']['bins']=3122
    with pytest.raises(ValueError,match='different trial'): inspect(r)


def test_reject_accidental_release():
    r=report();r['mode_results']['off']['records'][0]['released']=True
    with pytest.raises(ValueError,match='output releases'): inspect(r)
