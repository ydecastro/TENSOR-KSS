"""Section 6, paragraph "The true exponent, measured", second part, and the last sentence of Remark 1 (rem:bracket):
the measured deficiencies against the exponent E of Theorem 2 (thm:exponent).

For each simulated pair and each tabulated lambda: the deficiency p, its Clopper-Pearson 95% interval,
-log(p)/lambda^2 against E (paper: below E at every lambda of the table), and the factor p / exp(-E lambda^2)
(paper: 1.7 at lambda = 6 and 1.1 at lambda = 12 at (3,3)). The slopes printed in the paper are compared with E
(paper: 6% above at (3,3), 7% above at (5,3) where the difference is 2.7 times the quoted uncertainty, within its
error at (3,4), 10% below at (3,5)). The slopes themselves are the weighted least squares of ../sim/analyze.py
(weights n p/(1-p), the inverse variance of log p), repeated here on the windows quoted in the paper.

Input: the pooled counts of ../sim/results.jsonl and ../sim/results_rows_33_34.jsonl (as in ../sim/analyze.py), and
E from deficiency_exponent_constants.json (run deficiency_exponent_constants.py first).
Run: python3 deficiency_vs_exponent.py (a second). Deterministic.
"""
import json
import math
import os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
T1 = json.load(open(os.path.join(HERE, "deficiency_exponent_constants.json")))
pool = {}
for fn in ("results.jsonl", "results_rows_33_34.jsonl"):
    for line in open(os.path.join(SIM, fn)):
        if line.strip():
            r = json.loads(line)
            if r["lam"] > 0:
                key = (r["k"], r["d"], float(r["lam"]))
                n0, f0 = pool.get(key, (0, 0))
                pool[key] = (n0 + r["n"], f0 + r["fail"])


def cp(x, n):
    return stats.beta.ppf(0.025, x, n - x + 1), stats.beta.ppf(0.975, x + 1, n - x)


def wls(Y, W, cols):
    X = np.stack(cols, 1)
    cov = np.linalg.inv(X.T @ np.diag(W) @ X)
    beta = cov @ X.T @ np.diag(W) @ Y
    return beta, np.sqrt(np.diag(cov))


PRINTED = {(3, 3): (0.065, 0.002), (3, 4): (0.115, 0.012), (3, 5): (0.14, 0.01), (5, 3): (0.124, 0.003)}
PAPER_REL = {(3, 3): "6% above", (3, 4): "within its error", (3, 5): "10% below", (5, 3): "7% above, 2.7 times the uncertainty"}
WINDOWS = {(3, 3): [(6, 12)], (3, 4): [(6, 10), (7, 10)], (3, 5): [(6, 9.5), (7, 9.5), (8, 9.5)], (5, 3): [(6, 9)]}
for kd in [(3, 3), (3, 4), (3, 5), (5, 3)]:
    k, d = kd
    E = float(T1[f"{k},{d},0.05"]["E"])
    print(f"== (k,d) = {kd}: E = {E:.6f}")
    rows = sorted((l, n, f) for (kk, dd, l), (n, f) in pool.items() if (kk, dd) == kd)
    tab = []
    print("   lambda  n         fail  deficiency  CP 95%                    -log(p)/lambda^2  vs E   p/exp(-E lambda^2) [CP]")
    for l, n, f in rows:
        p = f / n
        lo, hi = cp(f, n)
        le = -math.log(p) / l ** 2
        fac = p * math.exp(E * l * l)
        tab.append(dict(lam=l, n=n, p=p, local=le))
        print(f"   {l:5.1f}  {n:8d}  {f:6d}  {p:.4e}  [{lo:.3e}, {hi:.3e}]   {le:.5f}           "
              f"{'below' if le < E else 'ABOVE'}  {fac:7.3f} [{lo * math.exp(E * l * l):.3f}, {hi * math.exp(E * l * l):.3f}]")
    L = np.array([t["lam"] for t in tab]); Y = -np.log([t["p"] for t in tab])
    Wt = np.array([t["n"] * t["p"] / (1 - t["p"]) for t in tab])
    for (a, b) in WINDOWS[kd]:
        s_ = (L >= a) & (L <= b)
        beta, se = wls(Y[s_], Wt[s_], [np.ones(s_.sum()), L[s_] ** 2])
        print(f"   WLS slope of -log p on lambda^2 over [{a}, {b}]: {beta[1]:.4f} +- {se[1]:.4f};  (slope - E)/E = {(beta[1] - E) / E * 100:+.1f}%")
    mu, sd = PRINTED[kd]
    print(f"   printed slope {mu} +- {sd}: (slope - E)/E = {(mu - E) / E * 100:+.1f}%, (slope - E)/uncertainty = "
          f"{(mu - E) / sd:+.1f}   (paper: {PAPER_REL[kd]})")
    print(f"   max over the table of -log(p)/lambda^2 = {max(t['local'] for t in tab):.5f} (E = {E:.5f}); "
          f"all below E: {all(t['local'] < E for t in tab)}")
print("(paper, (3,3): the deficiency exceeds exp(-E lambda^2) by a factor 1.7 at lambda = 6 and 1.1 at lambda = 12)")
