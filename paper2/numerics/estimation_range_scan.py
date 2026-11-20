"""Section 6, paragraph "The estimation sandwich", range of usefulness of Theorem 4(ii): at (3,3), (3,5), (5,3), (3,8)
and lambda = 5 to 30 by 0.5, then 31 to 100 by 1, the right-hand side of Theorem 4(ii) (eps_A of Lemma C.2 in the
product form, varrho exact, Rem_far by Kac-Rice quadrature with E det) minimised over (theta_1, u) under the
hypotheses of (ii), and the Kac-Rice column minimised over u (E det in place of E|det|); with the lower bound (i) and
the target (d-1)/(k lambda^2). One JSON line per point in estimation_range_scan.jsonl, read by
estimation_thresholds.py.
Run: PROCS=4 python3 estimation_range_scan.py (about 65 minutes of one core in all; the stored output was produced on
2026-11-10 and was not run again for this release).
"""
import os
import sys
import json
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np
from estimation_lib import Case

PAIRS = [(3, 3), (3, 5), (5, 3), (3, 8)]
LAMS = sorted(set([5, 8, 10, 12, 15, 20, 30, 40] + list(np.arange(5, 30.01, 0.5)) + list(range(31, 101, 1))))

def job(kd):
    k, d = kd
    out = []
    for lam in LAMS:
        t = time.time()
        C = Case(k, d, lam)
        lo = C.lower()
        b = C.best_both(nu=40)
        ukr, kr = C.best_kacrice(nu=40)
        rec = dict(k=k, d=d, lam=float(lam), lower=lo, tgt=C.tgt, kr=kr, u_kr=ukr, sec=time.time() - t)
        if b:
            rec.update({("x_" + kk): vv for kk, vv in b.items()})
        out.append(rec)
        print(json.dumps(rec), flush=True)
    return out

if __name__ == "__main__":
    with Pool(int(os.environ.get("PROCS", "4"))) as p:
        res = p.map(job, PAIRS)
    with open(os.path.join(HERE, "estimation_range_scan.jsonl"), "w") as f:
        for r in res:
            for x in r:
                f.write(json.dumps(x) + "\n")
