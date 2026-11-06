"""Bounded, privacy-preserving diagnostic must not mutate parameters."""
import numpy as np
import pytest
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters
from scripts.diagnose_feedback_candidates import diagnose_trial


def test_no_weight_or_input_mutation():
    p=synthetic_parameters()
    inputs=synthetic_features(4)
    before_w=p.weights.copy()
    before_x=inputs.copy()
    r=diagnose_trial(p,inputs,[1],channels=(0,304),steps=(4,))
    assert len(r['candidates'])==4
    assert r['ctc_steps']==1
    assert all(c['baseline_edits']==r['baseline_edits'] for c in r['candidates'])
    assert np.array_equal(inputs,before_x)
    assert np.array_equal(p.weights,before_w)
    assert all(c['candidate_edits']>=0 for c in r['candidates'])


@pytest.mark.parametrize('channel,step',[(512,4),(0,0),(0,17),(-1,4)])
def test_invalid_candidate_rejected(channel,step):
    with pytest.raises(ValueError):
        diagnose_trial(synthetic_parameters(),synthetic_features(4),[1],
                      channels=(channel,),steps=(step,))
