import numpy as np
import pytest
from neuroedge.drift.causal_audit import (
    benjamini_hochberg, adjusted_rho, audit_trial_risk
)
from neuroedge.control.transactional import OffsetCommitBank


def example_rows(n=26):
    rows=[]
    for i in range(n):
        bins=250+i*17
        tokens=15+(i%7)
        req=(i*11+3)%13
        edits=max(0, int(tokens*(0.1+0.8*(i%9)/8)))
        rows.append(dict(bins=bins,target_tokens=tokens,edits=edits,
                         requests=req,score_mean=float((i*7)%60),
                         blank_fraction=(i%8)/8,
                         mean_top2_margin=10.+i/10))
    return rows


def test_bh_exact_example():
    q=benjamini_hochberg({'a':.0065,'b':.0195,'c':.1759,'d':.8331})
    assert abs(q['a']-.026)<1e-12
    assert abs(q['b']-.039)<1e-12
    assert abs(q['c']-.23453333333333334)<1e-10
    assert q['d']==.8331


def test_bh_reject_bad_input():
    with pytest.raises(ValueError): benjamini_hochberg({})
    with pytest.raises(ValueError): benjamini_hochberg({'x':-1})


def test_adjusted_rank_linear_length_confound():
    rng=np.random.default_rng(51)
    z=np.arange(40.,dtype=float)
    x=z+rng.normal(0,.01,len(z))
    y=z+rng.normal(0,.01,len(z))
    assert abs(adjusted_rho(x,y,(z,z*.1))) < .6


def test_trial_lag_respects_separate_rows():
    a=audit_trial_risk(example_rows(),seed=12,permutations=150)
    assert a['trials']==26 and a['lagged_pairs']==25
    for family in ('same_trial','rank_adjusted_for_bin_and_label_count',
                   'previous_trial_predicts_next_trial'):
        assert set(a[family])=={'request_rate_per_1000_bins',
                                'activity_excursions_mean','blank_fraction',
                                'negative_top2_margin'}
        for m in a[family].values():
            assert -1<=m['rho']<=1
            assert 0<=m['bh_q_exploratory']<=1


def test_trial_lag_constant_target_is_null():
    rows=example_rows(12)
    for r in rows:
        r['target_tokens']=20
        r['edits']=10
    a=audit_trial_risk(rows,seed=4,permutations=100)
    for family in ('same_trial','previous_trial_predicts_next_trial'):
        assert all(v['rho']==0. for v in a[family].values())


def test_invalid_trials_guard():
    rows=example_rows(12)
    rows[2]['bins']=0
    with pytest.raises(ValueError): audit_trial_risk(rows,permutations=100)


def test_commits_are_gated_and_atomic():
    b=OffsetCommitBank()
    assert b.read(7)==0
    assert b.stage(7,-30)
    assert b.read(7)==0
    assert not b.stage(9,50)
    assert not b.tick(update_request=True,decoder_idle=False)
    assert b.read(7)==0 and b.pending==(7,-30)
    assert b.tick(update_request=True,decoder_idle=True)
    assert b.read(7)==-30
    assert b.writes==1 and b.rejected==1 and b.accepted_stages==1
    assert not b.tick(update_request=True,decoder_idle=True)
    assert b.rejected==2


def test_abort_reset_range():
    b=OffsetCommitBank()
    assert b.stage(511,32767)
    b.tick(abort=True,update_request=True)
    assert b.read(511)==0 and b.aborted==1
    assert b.stage(0,-32768)
    assert b.tick(update_request=True)
    assert b.read(0)==-32768
    b.reset()
    assert b.read(0)==0 and b.writes==0
    with pytest.raises(ValueError): b.stage(512,0)
    with pytest.raises(ValueError): b.stage(0,32768)
