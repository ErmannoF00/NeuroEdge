#!/usr/bin/env python3
"""Generate reproducible controller vectors; policy trace only, not neural accuracy."""
import argparse, random
from pathlib import Path
from neuroedge.control.budget import BudgetController

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--bins', type=int, default=256)
a = p.parse_args()
assert a.bins > 0
rng = random.Random(20261015)
c = BudgetController()
path = Path(a.out)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w') as f:
    for i in range(a.bins):
        score = (500 if i % 67 < 15 else 1550 if i % 67 < 35 else rng.randrange(700, 1300))
        active, request, writes, suppressed = c.sample(score)
        f.write(f'{score} {active} {request} {writes} {suppressed}\n')
print(f'POLICY_VECTORS_PASS bins={c.observed} requests={c.writes} total_suppressed={c.suppressed}')
