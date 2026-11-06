import numpy as np
import pytest
from neuroedge.speech.session_ablation import calibrate_rules, paired_per_bootstrap


def test_known_moments():
    mu=np.full(512,0.5)
    sd=np.full(512,2.)
    m=calibrate_rules(mu,sd,100)
    assert np.allclose(m['unsup_offset'].offset,-0.5)
    assert np.allclose(m['unsup_affine'].gain,0.75)  # bounded from 0.5
    assert np.allclose(m['unsup_affine'].offset,-0.375)
    assert np.allclose(m['fixed'].gain,1.)


def test_paired_ci_equal_models():
    x=paired_per_bootstrap([4,5,6],[1,3,4],[1,3,4],iterations=200)
    assert x['after_minus_before']==0
    assert x['delta_95pct_trial_bootstrap_ci']==[0.0,0.0]


def test_paired_improvement_sign():
    x=paired_per_bootstrap([10,20],[8,16],[4,8],iterations=200)
    assert x['after_minus_before'] < 0
    assert x['delta_95pct_trial_bootstrap_ci'][1] < 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        paired_per_bootstrap([0],[2],[1])
    with pytest.raises(ValueError):
        calibrate_rules(np.zeros(5), np.ones(5), 12)
