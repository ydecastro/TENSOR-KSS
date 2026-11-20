"""Section 6, paragraph "The estimation sandwich": the Kac-Rice bound, minimised over u, against the lower bound of
Theorem 4(i), at (3,3), (3,5), (5,3), (3,8) and lambda = 5, 8, 9, 10, 12, 15, with E|det| estimated by Monte Carlo
(4000 GOE draws; the E det proxy turns negative at small u for d = 8). estimation_thresholds.py reads the output
estimation_kacrice_scan.jsonl and gives the signal from which this bound is within 10% of the lower bound
(paper: from lambda about 9, and 10.4 at (3,8)).
Run: PROCS=4 python3 estimation_kacrice_scan.py (several minutes on 4 processes; the stored output was produced on
2026-11-10 and was not run again for this release).
"""
import os
import sys
import json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
from scipy import integrate, stats
from estimation_lib import Case

def kr_mc(C, u):
    th = C.th[1:]
    g = np.abs(C.g(th, u)); keep = th[g > 1e-12 * g.max()]
    t = np.linspace(keep.min(), keep.max(), 301)
    return integrate.simpson(C.g(t, u, "mc") * np.sin(t) ** 2, x=t) + stats.norm.cdf(u - C.lam)

def job(arg):
    kind, k, d, lam = arg
    C = Case(k, d, lam)
    if kind == "kr":
        us = np.linspace(max(0.3 * lam, 1.0), lam, 14)
        v = [kr_mc(C, u) for u in us]
        i = int(np.argmin(v))
        uu = np.linspace(us[max(i - 1, 0)], us[min(i + 1, 13)], 7)
        v2 = [kr_mc(C, u) for u in uu]
        j = int(np.argmin(v2))
        out = dict(kind=kind, k=k, d=d, lam=lam, u=float(uu[j]), kr_mc=float(v2[j]), lower=C.lower(), tgt=C.tgt)
        out["kr_mc_over_lower"] = out["kr_mc"] / out["lower"]
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    jobs = [("kr", k, d, lam) for (k, d) in [(3, 3), (3, 5), (5, 3), (3, 8)] for lam in (5, 8, 9, 10, 12, 15)]
    with Pool(int(os.environ.get("PROCS", "4"))) as p:
        res = p.map(job, jobs, chunksize=1)
    with open(os.path.join(HERE, "estimation_kacrice_scan.jsonl"), "w") as f:
        for x in res: f.write(json.dumps(x) + "\n")
