"""Remark after Corollary 6 (cor:minimax): the lower bound of Theorem 4(i) (thm:estim) solved for the risk at
(k,d) = (3,5), lambda = 5, 10, 20, 40, and its ratio to (d-1)/(k lambda^2).

Theorem 4(i) reads R (d-1) k lambda^2 >= ((d-1) - d R)^2, that is
d^2 R^2 - (2 d (d-1) + (d-1) k lambda^2) R + (d-1)^2 <= 0; the bound is the smallest root.
The same value is the column "lower bound, (i)" of the table of Section 6, "The estimation sandwich".
Run: python3 estimation_lower_bound.py (a fraction of a second).
"""
from math import sqrt

k, d = 3, 5
N = d - 1
paper = {5: ("0.0472", "88.54%"), 10: ("0.01291", "96.80%"), 20: ("0.003306", "99.18%"), 40: ("0.000832", "99.79%")}
for lam, (pv, pp) in paper.items():
    a = d * d
    b = -(2 * d * N + N * k * lam * lam)
    c = N * N
    R = (-b - sqrt(b * b - 4 * a * c)) / (2 * a)
    tgt = N / (k * lam * lam)
    print(f"lambda = {lam:2d}: lower bound R = {R:.6e}, (d-1)/(k lambda^2) = {tgt:.6e}, ratio {100 * R / tgt:.2f}%"
          f"   (paper {pv}, {pp})")
