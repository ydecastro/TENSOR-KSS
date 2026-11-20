"""The power bounds in closed form, at alpha = 0.05.

- Section 3.2, after Corollary 3 (cor:closed): the signal lambda at which the bound (eq:closed-nct) equals 0.05,
  at (3,5) and (3,3) (paper: 11.0 and 11.4).
- Section 3.2, after Corollary 4 (cor:Sclosed): the same for the known-variance bound (eq:Sclosed-nct)
  (paper: 6.8 at (3,3) and 10.0 at (3,5)).
- Section 6, paragraph "The power bound", at (3,5): the bound (eq:closed-nct) equals 0.05 and 0.01 at lambda = 11.0
  and 12.0, its Gaussian form (eq:closed-chernoff) at 13.1 and 14.1; the simulated deficiency equals 0.05 and 0.01
  near lambda = 6.2 and 7.2 (log-linear interpolation of the table of the next paragraph); the bound needs a signal
  1.7 to 1.8 times larger.

  (eq:closed-nct)      1 - P_lambda{T <= alpha} <= B(lambda) = P{lambda + xi < (A/rho) chi_{m-1}}
                       = int_0^inf Phi((A/rho) s - lambda) f_{chi_{m-1}}(s) ds
                       (the noncentral Student distribution function at A sqrt(m-1)/rho);
  (eq:closed-chernoff) 1 - P_lambda{T <= alpha} <= exp(-(rho lambda - A sqrt(m-1))^2 / (2 (A^2 + rho^2))),
                       equal to beta at lambda = (A sqrt(m-1) + (2 (A^2 + rho^2) log(1/beta))^(1/2)) / rho;
  (eq:Sclosed-nct)     1 - P_lambda{S <= alpha} <= P{rho^2 (lambda + xi)^2 < chi^2_{m-1} + rho^2 c_alpha},
                       c_alpha the (1-alpha)-quantile of chi^2_{2d}.
A is the constant of Remark 1, read from deficiency_exponent_constants.json (run deficiency_exponent_constants.py
first). mpmath at 30 digits; each root is checked against scipy.stats.nct. The simulated deficiencies are the pooled
counts of ../sim/results.jsonl and ../sim/results_rows_33_34.jsonl, as in ../sim/analyze.py.
Run: python3 power_bound_student.py (about a minute). Deterministic.
"""
import json
import math
import os
import mpmath as mp
from scipy import stats
import exponent_lib as C

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
T1 = json.load(open(os.path.join(HERE, "deficiency_exponent_constants.json")))


def chi_pdf(s, nu):
    return mp.exp((1 - mp.mpf(nu) / 2) * mp.log(2) + (nu - 1) * mp.log(s) - s * s / 2 - mp.loggamma(mp.mpf(nu) / 2))


def bound_nct(lam, A, P):
    nu = P["m"] - 1
    c = A / mp.sqrt(P["rho2_mp"])
    s0 = mp.sqrt(nu - 1)
    f = lambda s: mp.ncdf(c * s - lam) * chi_pdf(s, nu)
    return mp.quad(f, [0, s0 - 6, s0 - 3, s0, s0 + 3, s0 + 6, s0 + 12, mp.inf] if s0 > 6 else [0, s0, s0 + 3, s0 + 6, s0 + 12, mp.inf])


def bound_S(lam, P, alpha, ca):
    """(eq:Sclosed-nct), conditioning on xi = x: P{chi^2_nu > rho^2 (lam + x)^2 - rho^2 c_alpha}."""
    nu = P["m"] - 1
    rho2 = P["rho2_mp"]
    def f(x):
        tq = rho2 * ((lam + x) ** 2 - ca)
        return mp.npdf(x) * (1 if tq <= 0 else mp.gammainc(mp.mpf(nu) / 2, tq / 2, mp.inf, regularized=True))
    return mp.quad(f, [-mp.inf, -lam - mp.sqrt(ca), -lam, -lam + mp.sqrt(ca), 0, mp.inf])


def solve(fun, beta, lo, hi):
    g = lambda l: fun(l) - beta
    for _ in range(60):
        mid = (lo + hi) / 2
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return mp.findroot(g, (lo, hi), solver="anderson")


def lam_gauss(A, P, beta):
    rho = mp.sqrt(P["rho2_mp"])
    return (A * mp.sqrt(P["m"] - 1) + mp.sqrt(2 * (A * A + P["rho2_mp"]) * mp.log(1 / beta))) / rho


pool = {}
for fn in ("results.jsonl", "results_rows_33_34.jsonl"):
    for line in open(os.path.join(SIM, fn)):
        if line.strip():
            r = json.loads(line)
            if r["lam"] > 0:
                key = (r["k"], r["d"], float(r["lam"]))
                n0, f0 = pool.get(key, (0, 0))
                pool[key] = (n0 + r["n"], f0 + r["fail"])


def sim_cross(k, d, beta):
    row = sorted((l, f / n) for (kk, dd, l), (n, f) in pool.items() if (kk, dd) == (k, d))
    for (l0, p0), (l1, p1) in zip(row[:-1], row[1:]):
        if p0 >= beta > p1:
            return l0 + (math.log(p0) - math.log(beta)) / (math.log(p0) - math.log(p1)) * (l1 - l0)
    return float("nan")


PAPER_T = {(3, 3): "0.05 at 11.4", (3, 5): "0.05 at 11.0, 0.01 at 12.0; Gaussian form 13.1 and 14.1", (5, 3): None}
PAPER_S = {(3, 3): "6.8", (3, 5): "10.0", (5, 3): None}
al = "0.05"
for (k, d) in [(3, 3), (3, 5), (5, 3)]:
    P = C.mp_params(k, d)
    A = mp.mpf(T1[f"{k},{d},{al}"]["A"])
    lam_b = {}
    for bstr in ("0.05", "0.01"):
        beta = mp.mpf(bstr)
        ln = solve(lambda l: bound_nct(l, A, P), beta, mp.mpf(0), mp.mpf(200))
        lg = lam_gauss(A, P, beta)
        chk = stats.nct.cdf(float(A) * math.sqrt(P["m"] - 1) / P["rho"], P["m"] - 1, float(ln))
        lam_b[bstr] = float(ln)
        print(f"({k},{d}) alpha = {al}, A = {mp.nstr(A, 12)}: bound (eq:closed-nct) = {bstr} at lambda = {mp.nstr(ln, 8)} "
              f"(scipy nct cdf there: {chk:.10f}); Gaussian form (eq:closed-chernoff) = {bstr} at lambda = {mp.nstr(lg, 8)}")
    if PAPER_T[(k, d)]:
        print(f"      (paper: {PAPER_T[(k, d)]})")
    ca = mp.mpf(stats.chi2.ppf(1 - float(al), 2 * d))
    lS = solve(lambda l: bound_S(l, P, al, ca), mp.mpf("0.05"), mp.mpf(0), mp.mpf(100))
    print(f"({k},{d}) known-variance bound (eq:Sclosed-nct), c_alpha = {mp.nstr(ca, 8)}: = 0.05 at lambda = {mp.nstr(lS, 8)}"
          + (f"   (paper {PAPER_S[(k, d)]})" if PAPER_S[(k, d)] else ""))
    for bstr in ("0.05", "0.01"):
        sc = sim_cross(k, d, float(bstr))
        print(f"({k},{d}) simulated deficiency = {bstr} at lambda = {sc:.4f} (log-linear interpolation); "
              f"ratio lambda(bound)/lambda(simulated) = {lam_b[bstr] / sc:.4f}"
              + ("   (paper at (3,5): 6.2 and 7.2, ratio 1.7 to 1.8)" if (k, d) == (3, 5) else ""))
