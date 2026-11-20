"""Supplementary Material, Appendix C, before Lemma C.2 (lem:Aupper): the q-th root of q 2^(q/2) Gamma(q/2) is at
most (5/6) sqrt(q) for q >= 8 (2.29 against 2.36 at q = 8), and the ratio of the two sides decreases in q.
Run: python3 goe_norm_moment.py (a fraction of a second).
"""
from math import lgamma, log, exp, sqrt

prev = None
mono = True
for q in range(8, 401):
    lhs = exp((log(q) + q / 2 * log(2) + lgamma(q / 2)) / q)
    rhs = 5 / 6 * sqrt(q)
    r = lhs / rhs
    if q <= 12 or q in (20, 50, 100, 200, 400):
        print(f"q = {q:3d}: (q 2^(q/2) Gamma(q/2))^(1/q) = {lhs:.4f}, (5/6) sqrt(q) = {rhs:.4f}, ratio {r:.4f}")
    if prev is not None and r >= prev:
        mono = False
    prev = r
print(f"ratio decreasing on q = 8, ..., 400: {mono}   (paper: 2.29 against 2.36 at q = 8, ratio decreasing)")
