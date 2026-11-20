"""Section 6, paragraph "The true exponent, measured": replicate counts and accuracy checks of the field simulation,
read from the stored JSON lines results.jsonl and results_rows_33_34.jsonl, pooled by (k,d,lambda) as analyze.py
does. Gives the number of replicates per point and in all under the alternatives (paper: from 2e4 to 2e6 per point,
1.0e7 in all), the smallest number of failures at a point (paper: at least 78), and the recheck of 2% of the
replicates with four times as many starting points (paper: lambda_1 never changed, lambda_2 rose 3 times in 279 000,
no decision changed); and the order relation (eq:order) under the alternatives (lam2>lam1 counts).
Run: python3 replicate_counts.py (a fraction of a second).
"""
import json
from collections import defaultdict
import os
R = os.path.dirname(os.path.abspath(__file__)) + "/"
recs = [json.loads(l) for fn in ("results.jsonl", "results_rows_33_34.jsonl") for l in open(R + fn) if l.strip()]
cfgs = {json.dumps(r['cfg'], sort_keys=True) for r in recs}
print("distinct Cfg in the records:", cfgs)
print("seeds:", sorted({r['seed'] for r in recs}), " nrep per record / lam:")
F = ['n', 'fail', 'n_check', 'chk_lam1_up', 'chk_lam1_diff', 'chk_lam2_up', 'chk_fail_change', 'n_fix',
     'viol_lam2_gt_lam1', 'not_local_max', 'fibre_is_sup', 'h_only', 'd_only', 'wall_s']
tot = {'null': defaultdict(float), 'alt': defaultdict(float)}
pooled = defaultdict(lambda: defaultdict(float))
for r in recs:
    key = (r['k'], r['d'], r['lam'])
    for f in F:
        pooled[key][f] += r.get(f, 0)
    pooled[key]['nrec'] += 1
for key in sorted(pooled):
    p = pooled[key]
    g = 'null' if key[2] == 0 else 'alt'
    for f in F:
        tot[g][f] += p[f]
    print(f"  {key}: n={int(p['n'])} records={int(p['nrec'])} fail={int(p['fail'])} checked={int(p['n_check'])} "
          f"lam1 changed={int(p['chk_lam1_diff'])} lam2 up={int(p['chk_lam2_up'])} decision changes={int(p['chk_fail_change'])} "
          f"Z restarts={int(p['n_fix'])} lam2>lam1={int(p['viol_lam2_gt_lam1'])} wall={p['wall_s']:.0f}s")
for g in ('null', 'alt'):
    t = tot[g]
    print(f"TOTAL {g}: n={int(t['n'])}, checked={int(t['n_check'])} ({100*t['n_check']/t['n']:.2f}%), lam1 changed={int(t['chk_lam1_diff'])}, "
          f"lam2 up={int(t['chk_lam2_up'])}, decision changes={int(t['chk_fail_change'])}, Z restarts={int(t['n_fix'])}, "
          f"lam2>lam1={int(t['viol_lam2_gt_lam1'])}, not local max={int(t['not_local_max'])}, wall={t['wall_s']/3600:.1f} h")
alt = [p for key, p in pooled.items() if key[2] > 0]
print(f"under the alternatives: {len(alt)} points, from {int(min(p['n'] for p in alt))} to {int(max(p['n'] for p in alt))} "
      f"replicates per point, {int(sum(p['n'] for p in alt))} in all; smallest number of failures at a point "
      f"{int(min(p['fail'] for p in alt))}; rechecked replicates (null and alternatives) "
      f"{int(tot['null']['n_check'] + tot['alt']['n_check'])}")
print("(paper: 2e4 to 2e6 per point, 1.0e7 in all, at least 78 failures; lambda_2 rose 3 times in 279 000 rechecked "
      "replicates, lambda_1 never changed, no decision changed)")
