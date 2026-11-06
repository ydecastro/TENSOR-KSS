"""Section 6 (sec:low) and the local-efficiency table of Section 7, at (k,d) = (3,3).

The field X(t) = sum_{|a|=3} sqrt(3!/a!) g_a t^a (Cov = <s,t>^3) is evaluated on a Fibonacci grid of 40 000 points
of S^2, with 2e5 replicates (seed 7); sup X is the grid maximum and T_q = int X^2 dsigma is the mean of X^2 over
the grid. At alpha = 0.05 and 0.01 the script prints u_alpha (the 1-alpha quantile of sup X), the local coefficient
of the maximum test C_max = Cov_0(1{sup X > u_alpha}, T_q)/2, the optimal coefficient
C_opt = (E[T_q; T_q > t_alpha] - alpha)/2 for the exact null law T_q = (1/5) chi2_3 + (2/35) chi2_7 (by quadrature),
and the local efficiency C_max/C_opt. Output: local_efficiency.out.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import math
import numpy as np
from scipy import integrate, optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "local_efficiency.out"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


k, d = 3, 3
NPTS, M, BATCH, SEED = 40000, 200000, 5000, 7

# Kostlan basis on S^{d-1}: X(t) = sum_{|a|=k} sqrt(k!/a!) g_a t^a, Cov = <s,t>^k
alphas = [a for a in itertools.product(range(k + 1), repeat=d) if sum(a) == k]
coef = np.array([math.sqrt(math.factorial(k) / np.prod([math.factorial(x) for x in a])) for a in alphas])
A = np.array(alphas)


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


B = np.prod(fib(NPTS)[:, None, :] ** A[None, :, :], axis=2) * coef     # (npts, nmon)
rng = np.random.default_rng(SEED)
sup = np.empty(M)
T = np.empty(M)
for b in range(0, M, BATCH):
    g = rng.normal(size=(BATCH, len(alphas)))
    V = g @ B.T                                                       # (BATCH, npts)
    sup[b:b + BATCH] = V.max(axis=1)
    T[b:b + BATCH] = (V ** 2).mean(axis=1)

ET2k = math.prod(range(1, 2 * k, 2)) / math.prod(d + 2 * i for i in range(k))
P(f"(k,d)=({k},{d}), {NPTS} grid points, {M} replicates, seed {SEED}")
P(f"E T_q = {T.mean():.5f} (exact 1), Var T_q = {T.var():.5f} (exact 2 E T_cor^2k = {2 * ET2k:.5f})")

# exact null law of T_q = a chi2_3 + b chi2_7
a1, b1 = 1 / 5, 2 / 35


def sf(x):
    """P(T_q > x)."""
    if x <= 0:
        return 1.0
    f = lambda s: stats.chi2.pdf(s, 3) * stats.chi2.sf((x - a1 * s) / b1, 7)
    return integrate.quad(f, 0, x / a1, epsabs=0, epsrel=1e-12, limit=200)[0] + stats.chi2.sf(x / a1, 3)


def tail_mean(t):
    """E[T_q; T_q > t] = t P(T_q > t) + int_t^inf P(T_q > x) dx."""
    return t * sf(t) + integrate.quad(sf, t, np.inf, epsabs=0, epsrel=1e-10, limit=200)[0]


for alpha in (0.05, 0.01):
    u = np.quantile(sup, 1 - alpha)
    ind = (sup > u).astype(float)
    cmax = np.cov(ind, T)[0, 1] / 2
    se = np.std(ind * (T - T.mean())) / math.sqrt(M) / 2
    t_a = optimize.brentq(lambda x: sf(x) - alpha, 0.5, 10, xtol=1e-12)
    copt = (tail_mean(t_a) - alpha) / 2
    P(f"alpha={alpha}: u_alpha={u:.4f}  C_max={cmax:.4e} (s.e. {se:.1e})  t_alpha={t_a:.4f}  C_opt={copt:.4e}"
      f"  efficiency={cmax / copt:.4f}")
P("paper: u_alpha 3.181, 3.678; t_alpha 2.022, 2.732; C_max 2.959e-2, 8.398e-3; C_opt 3.655e-2, 1.081e-2;"
  " efficiency 0.810, 0.777")
OUT.close()
