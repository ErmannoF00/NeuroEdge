"""Exploratory trial-lag risk audit; NOT causal identification or online decoding.

The first previous-trial signal is available before the next trial begins.
Labels are used only after all features are computed. Trials within one session
are not independent biological subjects. existing never alters decoder parameters.
"""
from __future__ import annotations
import numpy as np
from neuroedge.drift.utility import spearman, rank_average

INDICATORS = ('request_rate_per_1000_bins', 'activity_excursions_mean',
              'blank_fraction', 'negative_top2_margin')


def _vectors(rows):
    if len(rows) < 8:
        raise ValueError('Need >=8 trials for lagged audit')
    bins = np.array([r['bins'] for r in rows], dtype=float)
    tokens = np.array([r['target_tokens'] for r in rows], dtype=float)
    edits = np.array([r['edits'] for r in rows], dtype=float)
    req = np.array([r['requests'] for r in rows], dtype=float)
    if not np.isfinite(np.c_[bins, tokens, edits, req]).all():
        raise ValueError('nonfinite input')
    if np.any(bins<=0) or np.any(tokens<=0) or np.any(edits<0) or np.any(req<0):
        raise ValueError('invalid trial')
    x = {
        'request_rate_per_1000_bins': 1000*req/bins,
        'activity_excursions_mean': np.asarray([r['score_mean'] for r in rows],float),
        'blank_fraction': np.asarray([r['blank_fraction'] for r in rows],float),
        'negative_top2_margin': -np.asarray([r['mean_top2_margin'] for r in rows],float),
    }
    for values in x.values():
        if not np.isfinite(values).all(): raise ValueError('nonfinite feature')
    return x, edits/tokens, bins, tokens


def benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    """BH adjusted p (exploratory); no claim of independent tests."""
    if not pvalues: raise ValueError('Empty family')
    names=list(pvalues)
    p=np.array([pvalues[k] for k in names],float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('p-values must be within [0,1]')
    order=np.argsort(p,kind='stable')
    ranked=p[order]
    adjusted=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    values=np.empty(len(p))
    values[order]=np.clip(adjusted,0.,1.)
    return {k:float(v) for k,v in zip(names,values)}


def _rank_residual(values, confounders):
    y=rank_average(np.asarray(values,float))
    z=np.column_stack([np.ones(len(y))]+[rank_average(c) for c in confounders])
    return y-z@np.linalg.lstsq(z,y,rcond=None)[0]


def _pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    x=x-x.mean(); y=y-y.mean()
    den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/den) if den>1e-12 else 0.


def adjusted_rho(x, y, confounders):
    """Rank correlation after linear residualization of ranked covariates."""
    if len(x)!=len(y) or any(len(c)!=len(x) for c in confounders):
        raise ValueError('length mismatch')
    if len(x)<8: raise ValueError('too few trials')
    return _pearson(_rank_residual(x,confounders),_rank_residual(y,confounders))


def _perm_test(x,y, *, seed, permutations, confounders=()):
    if permutations<100: raise ValueError('Need at least 100 permutations')
    if confounders:
        x0=_rank_residual(x,confounders)
        y0=_rank_residual(y,confounders)
        fun=_pearson
    else:
        x0=np.asarray(x); y0=np.asarray(y)
        fun=spearman
    observed=fun(x0,y0)
    rng=np.random.default_rng(seed)
    hits=sum(abs(fun(x0,rng.permutation(y0))) >= abs(observed)-1e-12
             for _ in range(permutations))
    return {'rho':float(observed), 'permutation_p_exploratory':(hits+1)/(permutations+1)}


def audit_trial_risk(rows, *, seed=2026, permutations=1000):
    """Same-trial and previous-trial risk; length/label confounding diagnostic.

    Lag-one i->i+1 predicts next trial's PER using only finished trial i.
    Comparisons are observational, restricted to one session, and are never
    used as if a label were available to firmware.
    """
    x, per, bins, tokens = _vectors(rows)
    same={}; adjusted={}; lagged={}
    for i,k in enumerate(INDICATORS):
        same[k]=_perm_test(x[k],per,seed=seed+i*100,permutations=permutations)
        adjusted[k]=_perm_test(x[k],per,seed=seed+1000+i*100,
                               permutations=permutations,confounders=(bins,tokens))
        lagged[k]=_perm_test(x[k][:-1],per[1:],seed=seed+2000+i*100,
                             permutations=permutations)
    for family in (same,adjusted,lagged):
        q=benjamini_hochberg({k:v['permutation_p_exploratory'] for k,v in family.items()})
        for k,val in q.items(): family[k]['bh_q_exploratory']=val
    return {
      'trials':len(rows),'lagged_pairs':len(rows)-1,
      'mean_trial_PER':float(np.mean(per)),
      'same_trial':same,
      'rank_adjusted_for_bin_and_label_count':adjusted,
      'previous_trial_predicts_next_trial':lagged,
      'caveats':[
        'Within-session trial permutation assumes exchangeability; temporal dependence can invalidate p-values.',
        'Rank covariate adjustment is an observational sensitivity check, not causal adjustment.',
        'Labels and their counts are NOT available to the hardware; labels used for evaluation/confound check only.',
        'Lagged preceding-trial score is temporally available, but does not establish clinical prediction or intervention benefit.',
        'Same participant; one-session evidence; no novel generalization claim.',
        'Multiple exploratory families compared; BH within each family only, not across sessions/families.',
      ]}
