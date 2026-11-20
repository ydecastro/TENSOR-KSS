"""Theorem 4(iii) (thm:estim): the threshold lambda_0(k,d), the least number such that the five conditions
(eq:lam0a) to (eq:lam0e) of Appendix C of the Supplementary Material hold at every lambda > lambda_0(k,d), at
(k,d) = (3,3), (3,5), (5,3), (3,8), (7,5) (paper, rounded up: 28.83, 33.11, 29.50, 38.52, 34.52; the last condition
(eq:lam0e) is the one that binds).

  (a) lambda^2 >= 128 log lambda;
  (b) lambda > 4N/(3 rho) and lambda >= 16 sqrt(2N)/(9 rho);
  (c) g_1 = (87/128) rho lambda - sqrt(2N) >= 0 and K_N lambda^2 exp(-g_1^2/4) <= 1;
  (d) g_3 = (9/16) rho lambda - sqrt(2N) >= 0 and (5/4 + K_N exp(-g_3^2/4)) exp(128 N (N+1)/(169 lambda^2))
      (4k-5)/(3k-4) <= 2;
  (e) lambda >= 4 sqrt(d+3) and (10(4k-1)/3) omega_{d-1} ((k-1)/pi)^(N/2) (rho lambda + rho sqrt(N) + m_N)^N lambda^4
      exp(-lambda^2/32) <= 1;
with N = d - 1, rho^2 = k/(2(k-1)), q* = max(2N, 8), m_N = sqrt(2N) + (5/6) sqrt(q*), K_N = 2 sqrt(2) (5/4 + m_N/N)^N,
omega_{d-1} = 2 pi^(d/2)/Gamma(d/2).
Each condition is evaluated on a grid of [2, 200] (40000 points) and of [200, 1e5] (geometric, 20000 points); the
last grid point where it fails is refined by bisection. The script also checks on the grid that each condition,
once true, stays true. Run: python3 lambda0.py (a few seconds).
"""
import math
from math import log, sqrt, pi, gamma
import numpy as np


def pars(k, d):
    N = d - 1
    rho = sqrt(k / (2 * (k - 1)))
    qs = max(2 * N, 8)
    mN = sqrt(2 * N) + 5 / 6 * sqrt(qs)
    KN = 2 * sqrt(2) * (1.25 + mN / N) ** N
    omd1 = 2 * pi ** (d / 2) / gamma(d / 2)
    return N, rho, mN, KN, omd1


def conditions(k, d):
    N, rho, mN, KN, omd1 = pars(k, d)

    def a(lam):
        return lam * lam >= 128 * log(lam)

    def b(lam):
        return lam > 4 * N / (3 * rho) and lam >= 16 * sqrt(2 * N) / (9 * rho)

    def c(lam):
        g1 = 87 / 128 * rho * lam - sqrt(2 * N)
        return g1 >= 0 and log(KN) + 2 * log(lam) - g1 * g1 / 4 <= 0

    def dd(lam):
        g3 = 9 / 16 * rho * lam - sqrt(2 * N)
        return g3 >= 0 and ((1.25 + KN * math.exp(-g3 * g3 / 4)) * math.exp(128 * N * (N + 1) / (169 * lam ** 2))
                            * (4 * k - 5) / (3 * k - 4) <= 2)

    def e(lam):
        logF = (log(10 * (4 * k - 1) / 3) + log(omd1) + N / 2 * log((k - 1) / pi)
                + N * log(rho * lam + rho * sqrt(N) + mN) + 4 * log(lam) - lam * lam / 32)
        return lam >= 4 * sqrt(d + 3) and logF <= 0

    return {"(a)": a, "(b)": b, "(c)": c, "(d)": dd, "(e)": e}


def last_failure(cond, grid):
    v = np.array([cond(l) for l in grid])
    bad = np.where(~v)[0]
    first_true = int(np.argmax(v))
    mono = not np.any(~v[first_true:])
    if len(bad) == 0:
        return grid[0], mono
    i = bad[-1]
    lo, hi = grid[i], grid[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cond(mid):
            hi = mid
        else:
            lo = mid
    return hi, mono


PAPER = {(3, 3): "28.83", (3, 5): "33.11", (5, 3): "29.50", (3, 8): "38.52", (7, 5): "34.52"}
grid = np.unique(np.concatenate([np.linspace(2.0, 200, 40000), np.geomspace(200, 1e5, 20000)]))
for (k, d), pv in PAPER.items():
    per = {name: last_failure(f, grid) for name, f in conditions(k, d).items()}
    lam0 = max(p[0] for p in per.values())
    bind = max(per, key=lambda n: per[n][0])
    up = math.ceil(lam0 * 100) / 100
    print(f"({k},{d}) lambda_0 = {lam0:.6f}, rounded up {up:.2f}, binding condition {bind}; "
          + "  ".join(f"{n} {p[0]:.3f}{'' if p[1] else ' (not monotone on the grid)'}" for n, p in per.items())
          + f"   (paper {pv})")
