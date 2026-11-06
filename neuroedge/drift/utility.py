"""Trial-cluster retrospective monitor utility, not online calibration or diagnosis.

All metrics are computed per trial and are never used to modify the decoder.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def rank_average(x):
    """Ranks with average rank for ties, independent of scipy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('rank expects finite 1D vector')
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(x, y):
    x = rank_average(x)
    y = rank_average(y)
    if len(x) != len(y) or len(x) < 4:
        raise ValueError('Need >=4 paired trials')
    dx, dy = x-x.mean(), y-y.mean()
    den = np.sqrt((dx@dx)*(dy@dy))
    if den <= 1e-12:
        return 0.0
    return float((dx@dy)/den)


def permutation_rho(x, y, *, seed=2026, repetitions=2000):
    """Two-sided shuffle test for fixed per-trial score association; exploratory."""
    if repetitions < 100:
        raise ValueError('Need at least 100 permutations')
    x, y = np.asarray(x), np.asarray(y)
    observed = spearman(x, y)
    rng = np.random.default_rng(seed)
    more = 0
    for _ in range(repetitions):
        if abs(spearman(x, rng.permutation(y))) >= abs(observed)-1e-12:
            more += 1
    return observed, (more+1)/(repetitions+1)


def bootstrap_rho_ci(x, y, *, seed=2027, repetitions=1000):
    if repetitions < 100:
        raise ValueError('Need at least 100 resamples')
    x,y = np.asarray(x), np.asarray(y)
    if len(x) != len(y) or len(x) < 4:
        raise ValueError('Need >=4 paired trials')
    rng = np.random.default_rng(seed)
    vals=[]
    for _ in range(repetitions):
        ids = rng.integers(len(x),size=len(x))
        # Trials with identical ranks yield undefined correlation; conservative
        # zero is used to avoid falsely precise CIs on degenerate resamples.
        vals.append(spearman(x[ids],y[ids]))
    return [float(z) for z in np.quantile(vals,[0.025,0.975])]


def logit_metrics(logits):
    """Cheap, causal per-logit-frame confidence summaries, no softmax."""
    a=np.asarray(logits)
    if a.ndim != 2 or a.shape[1] != 41 or a.shape[0] == 0:
        raise ValueError('expected [T,41] logits')
    best = np.argmax(a,axis=1)
    margin = np.partition(a.astype(np.int32),-2,axis=1)
    diff=margin[:,-1]-margin[:,-2]
    return {'blank_fraction':float(np.mean(best==0)),
            'mean_top2_margin':float(np.mean(diff))}


def summarize_trial_utility(rows, *, seed=2026, permutations=2000, bootstraps=1000):
    """Compute diagnostics without storing trial labels, raw signals or outputs."""
    if len(rows) < 4:
        raise ValueError('At least four trials required')
    keys=('bins','target_tokens','edits','requests','score_mean','blank_fraction','mean_top2_margin')
    if any(any(k not in r for k in keys) for r in rows):
        raise ValueError('missing trial fields')
    bins=np.array([r['bins'] for r in rows],dtype=np.float64)
    tokens=np.array([r['target_tokens'] for r in rows],dtype=np.float64)
    edits=np.array([r['edits'] for r in rows],dtype=np.float64)
    requests=np.array([r['requests'] for r in rows],dtype=np.float64)
    if np.any(bins<=0) or np.any(tokens<=0) or np.any(edits<0) or np.any(requests<0):
        raise ValueError('invalid trial quantities')
    per=edits/tokens
    metrics={
      'request_rate_per_1000_bins': 1000 * requests / bins,
      'activity_excursions_mean': np.array([r['score_mean'] for r in rows],float),
      'blank_fraction': np.array([r['blank_fraction'] for r in rows],float),
      'negative_top2_margin': -np.array([r['mean_top2_margin'] for r in rows],float),
    }
    associations={}
    for i,(name,values) in enumerate(metrics.items()):
        rho,p=permutation_rho(values,per,seed=seed+10*i,repetitions=permutations)
        lo,hi=bootstrap_rho_ci(values,per,seed=seed+100+i,repetitions=bootstraps)
        associations[name]={'spearman_rho':rho,'permutation_two_sided_p_exploratory':p,
                            'trial_bootstrap_rho_ci95':[lo,hi]}
    # Null assumption: requests distributed uniformly over neural bins, not
    # equally across trials. This is purely retrospective and descriptive.
    high=per >= np.quantile(per,0.75)
    total_requests=requests.sum()
    risk_bin_fraction=float(bins[high].sum()/bins.sum())
    fraction_in_high=(float(requests[high].sum()/total_requests)
                      if total_requests else None)
    enrichment=(float(fraction_in_high/risk_bin_fraction)
                if total_requests and risk_bin_fraction > 0 else None)
    return {
       'trials':len(rows),'bins':int(bins.sum()),
       'target_tokens':int(tokens.sum()),'edits':int(edits.sum()),
       'micro_PER':float(edits.sum()/tokens.sum()),
       'macro_trial_PER':float(per.mean()),
       'requests':int(total_requests),
       'requests_per_1000_bins':float(1000*total_requests/bins.sum()),
       'associations':associations,
       'retrospective_worst_quartile': {
           'trial_count':int(high.sum()),'share_of_bins':risk_bin_fraction,
           'share_of_requests':fraction_in_high,
           'enrichment_vs_uniform_per_bin_null':enrichment},
    }
