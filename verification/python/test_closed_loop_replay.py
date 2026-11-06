"""Evidence-oriented tests of causal, supervised synthetic replay."""
import numpy as np
import pytest
from neuroedge.control.closed_loop_replay import run_synthetic_replay, select_lagged_proposal
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder


def test_real_decoder_and_transaction_table_form_lagged_synthetic_loop():
    result = run_synthetic_replay()
    assert result['bins'] == 32 and result['outputs'] == 328
    assert 0 < result['controlled_logit_l1'] < result['fixed_logit_l1']
    assert result['controlled_logit_l1'] < result['sham_logit_l1']
    assert result['sham_accepted_writes'] == result['trials'] - 1
    assert 0 < result['accepted_memory_writes'] <= result['trials'] - 1
    assert result['trials_log'][0]['active_offset_q4'] == 0
    assert not result['trials_log'][0]['accepted']
    for r in result['trials_log']:
        assert abs(r['active_offset_q4']) <= 32
        assert abs(r['proposal_delta_q4']) <= 4
        assert r['target_available_after_trial'] >= r['trial']
    assert result['no_hardware_closed_loop_claim']
    assert result['no_real_data_accuracy_claim']


def test_no_feedback_available_means_no_update_and_no_improvement():
    result = run_synthetic_replay(feedback_delay_trials=100)
    assert result['accepted_memory_writes'] == 0
    assert result['controlled_logit_l1'] == result['fixed_logit_l1']
    assert all(t['active_offset_q4'] == 0 for t in result['trials_log'])


def test_zero_drift_causes_no_memory_commits():
    result = run_synthetic_replay(drift_q4=0)
    assert result['fixed_logit_l1'] == result['controlled_logit_l1'] == 0
    assert result['accepted_memory_writes'] == 0


def test_delayed_feedback_is_not_used_before_observed():
    result = run_synthetic_replay(feedback_delay_trials=2)
    assert [x['active_offset_q4'] for x in result['trials_log'][:3]] == [0, 0, 0]
    assert [x['accepted'] for x in result['trials_log'][:3]] == [False] * 3
    assert result['accepted_memory_writes'] <= 5


def test_candidate_refuses_partial_trials_and_corrupt_targets():
    p = synthetic_parameters()
    state = FixedStreamingDecoder(p)
    raw = synthetic_features(4)
    logits = FixedStreamingDecoder(p).run(raw)
    with pytest.raises(ValueError, match='one completed'):
        select_lagged_proposal(decoder_before_trial=state, completed_raw_trial=raw[:3],
                               completed_logits=logits, revealed_target_logits=logits,
                               current_offset_q4=0, source_trial=0)
    with pytest.raises(ValueError, match='41-logit'):
        select_lagged_proposal(decoder_before_trial=state, completed_raw_trial=raw,
                               completed_logits=logits, revealed_target_logits=np.zeros((1,40), dtype=np.int16),
                               current_offset_q4=0, source_trial=0)
