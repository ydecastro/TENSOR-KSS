"""Section 7, the price of knowing sigma (and the introduction): level maps u_alpha(r) = G_r^{-1}(alpha G_r(v))
for the known-variance test and H_r^{-1}(alpha H_r(v)) for the t-test, from the closed forms of Proposition 2.
"""
import numpy as np
from math import comb, sqrt
from scipy import stats, integrate, optimize, special

def Gr(l, eigs):
    f = lambda u: np.prod(u - eigs) * stats.norm.pdf(u)
    return integrate.quad(f, l, np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]

def Hr(l, eigs, kap, nu):
    gam = sqrt(kap / nu)
    f = lambda u: np.prod(u - eigs) * stats.t.pdf(u / gam, nu)
    return integrate.quad(f, l, np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]

def level(F, v, alpha):
    t = alpha * F(v)
    return optimize.brentq(lambda u: F(u) - t, v, 10 * v + 20, xtol=1e-13)

table = [((3, 5), 0.05, 4, 4.897, 5.486), ((3, 5), 0.01, 4, 5.276, 6.171), ((3, 8), 0.05, 4, 5.036, 5.202), ((5, 6), 0.05, 4, 4.862, 4.923)]

def both(k, d, alpha, v, eigs):
    m = comb(d + k - 1, k); kap = m - d; nu = m - 1
    eigs = np.asarray(eigs, float)
    uG = level(lambda l: Gr(l, eigs), v, alpha)
    uH = level(lambda l: Hr(l, eigs, kap, nu), v, alpha)
    return uG, uH

print("--- Omega = 0 ---")
for (k, d), a, v, pG, pH in table:
    uG, uH = both(k, d, a, v, np.zeros(d - 1))
    print((k, d), a, v, f"known {uG:.4f} (paper {pG})  t {uH:.4f} (paper {pH})  extra {uH-uG:.4f} (paper {pH-pG:+.3f})")

# Omega = G/rho, G ~ GOE(N) with Var(v'Gv)=1, for a few seeds; and Omega = sqrt(2N)/rho * Id etc.
def goe(N, rng):
    A = rng.standard_normal((N, N))
    return (A + A.T) / 2  # diag var 1, off-diag var 1/2

print("\n--- Omega = G/rho for random GOE draws: distribution of the level maps ---")
for (k, d), a, v, pG, pH in table:
    rho = sqrt(k / (2 * (k - 1)))
    rng = np.random.default_rng(0)
    out = []
    for rep in range(200):
        e = np.linalg.eigvalsh(goe(d - 1, rng)) / rho
        if e.max() >= v:
            continue
        out.append(both(k, d, a, v, e))
    out = np.array(out)
    print((k, d), a, f"{len(out)} draws with lmax<v: known median {np.median(out[:,0]):.3f} [{np.percentile(out[:,0],10):.3f},{np.percentile(out[:,0],90):.3f}]"
          f"  t median {np.median(out[:,1]):.3f}  extra median {np.median(out[:,1]-out[:,0]):.3f}")

# 'inflation of a single quantile ... 2.8% at (3,5)'
print("\n--- single-quantile inflation at (3,5), alpha=0.05 ---")
k, d = 3, 5; m = comb(d + k - 1, k); kap = m - d; nu = m - 1
z = stats.norm.ppf(0.95)
print("t_{m-1}/z - 1 =", stats.t.ppf(0.95, nu) / z - 1, "; t_kappa/z - 1 =", stats.t.ppf(0.95, kap) / z - 1,
      "; gamma t_{m-1} / z - 1 =", sqrt(kap / nu) * stats.t.ppf(0.95, nu) / z - 1)
for (k, d), a, v, pG, pH in table:
    m = comb(d + k - 1, k); kap = m - d; nu = m - 1
    z = stats.norm.ppf(1 - a)
    print((k, d), a, " t_{m-1}/z-1 =", round(stats.t.ppf(1 - a, nu) / z - 1, 4), " t_kap/z-1 =", round(stats.t.ppf(1 - a, kap) / z - 1, 4))
