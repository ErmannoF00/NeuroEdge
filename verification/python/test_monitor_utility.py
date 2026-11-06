import numpy as np
import pytest
from neuroedge.drift.utility import (
    rank_average,spearman,permutation_rho,bootstrap_rho_ci,
    logit_metrics,summarize_trial_utility,
)


def _rows(coupled=True):
    x=[]
    for i in range(24):
        per=0.20+0.02*i
        req = i+1 if coupled else 12
        x.append(dict(bins=500,target_tokens=100,edits=round(per*100),
                      requests=req,score_mean=10.+i,blank_fraction=0.2+i/100,
                      mean_top2_margin=40.-i))
    return x


def test_spearman_ties_and_constant():
    assert np.allclose(rank_average([2,2,9,1]),[2.5,2.5,4,1])
    assert spearman([1,2,3,4],[3,6,9,12])==pytest.approx(1)
    assert spearman([1,1,1,1],[3,6,9,12])==0
    with pytest.raises(ValueError):spearman([1,2,3],[1,2,3])


def test_permutation_positive_vs_degenerate():
    x=np.arange(24)
    rho,p=permutation_rho(x,x,seed=99,repetitions=399)
    assert rho==pytest.approx(1)
    assert p<0.02
    assert permutation_rho(np.ones(24),x,seed=99,repetitions=199)==(0.,1.)


def test_logit_margin_int16_safe():
    a=np.zeros((2,41),dtype=np.int16)
    a[0,0]=32767;a[0,1]=-32768
    a[1,1]=32767;a[1,2]=32766
    d=logit_metrics(a)
    assert d['blank_fraction']==pytest.approx(0.5)
    assert d['mean_top2_margin']>=1
    with pytest.raises(ValueError):logit_metrics(np.zeros((2,40)))


def test_trial_utility_no_patient_data_and_positive_control():
    r=summarize_trial_utility(_rows(),seed=101,permutations=199,bootstraps=199)
    assert r['trials']==24
    assert r['requests']==sum(range(1,25))
    assert r['associations']['request_rate_per_1000_bins']['spearman_rho']>0.99
    assert r['retrospective_worst_quartile']['enrichment_vs_uniform_per_bin_null']>1
    assert r['micro_PER']>0
    assert 'text' not in repr(r) and 'transcription' not in repr(r)


def test_constant_monitor_failure_does_not_fabricate_signal():
    r=summarize_trial_utility(_rows(False),seed=102,permutations=199,bootstraps=199)
    assert r['associations']['request_rate_per_1000_bins']['spearman_rho']==0
    assert r['associations']['request_rate_per_1000_bins']['permutation_two_sided_p_exploratory']==1


def test_error_rejection():
    with pytest.raises(ValueError):summarize_trial_utility(_rows()[:3])
    with pytest.raises(ValueError):summarize_trial_utility(_rows()[:-1]+[dict(_rows()[-1],bins=0)])
    with pytest.raises(ValueError):bootstrap_rho_ci([1,2,3],[1,2,3])
