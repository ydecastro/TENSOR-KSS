"""Section 6 (sec:low), exact parts.
 - Funk-Hecke eigenvalues mu_j (numerical Gegenbauer integral), n_j, and the checks
   sum mu_j n_j = 1 and sum mu_j^2 n_j = E T_cor^{2k} = (2k-1)!!/prod_{i<k}(d+2i);
 - null law of the quadratic statistic T_q at (3,3): quantiles t_alpha and C_opt = (E[T_q;T_q>t_alpha]-alpha)/2
   (local-efficiency table of Section 7);
 - power of the quadratic test, T_q = sum mu_j chi2_{n_j}(lam^2 mu_j n_j), exact (Imhof inversion).
Saves quadpower.npy, used by maximum_test.py and ftest.py.
"""
import os
import numpy as np
from math import comb, gamma, sqrt, pi
import mpmath as mp
from scipy import special, integrate, stats, optimize

mp.mp.dps = 30

def mu_FH(k, d, j):
    nud = (d - 2) / 2
    if d == 2:
        raise ValueError
    C = lambda t: special.eval_gegenbauer(j, nud, t)
    c1 = special.eval_gegenbauer(j, nud, 1.0)
    w = gamma(d / 2) / (sqrt(pi) * gamma((d - 1) / 2))
    return integrate.quad(lambda t: t ** k * C(t) / c1 * w * (1 - t * t) ** ((d - 3) / 2), -1, 1, epsabs=1e-15, epsrel=1e-13, limit=200)[0]

def n_harm(d, j):
    return comb(d + j - 1, j) - (comb(d + j - 3, j - 2) if j >= 2 else 0)

def dfact(n):
    return 1 if n <= 0 else n * dfact(n - 2)

print("=== Funk-Hecke checks ===")
for (k, d) in [(3, 3), (3, 4), (3, 5), (5, 3), (5, 6)]:
    js = [j for j in range(k + 1) if (k - j) % 2 == 0]
    mus = [mu_FH(k, d, j) for j in js]; ns = [n_harm(d, j) for j in js]
    s1 = sum(m * n for m, n in zip(mus, ns)); s2 = sum(m * m * n for m, n in zip(mus, ns))
    ET = dfact(2 * k - 1) / np.prod([d + 2 * i for i in range(k)])
    print((k, d), "j:", js, "mu:", [f"{m:.10f}" for m in mus], "n:", ns, f" sum mu n = {s1:.12f}, sum mu^2 n = {s2:.12f}, E T^2k = {ET:.12f}")
print("(3,3): mu_1 = 1/5 =", 1 / 5, ", mu_3 = 2/35 =", 2 / 35)

# ---------- weighted sum of noncentral chi-squares: Imhof inversion ----------
def imhof_sf(x, w, n, nc):
    """P(sum w_j chi2_{n_j}(nc_j) > x), Imhof (1961)."""
    w = [mp.mpf(a) for a in w]; n = [mp.mpf(a) for a in n]; nc = [mp.mpf(a) for a in nc]; x = mp.mpf(x)
    def theta(u):
        return sum(n_j * mp.atan(w_j * u) + nc_j * w_j * u / (1 + (w_j * u) ** 2) for w_j, n_j, nc_j in zip(w, n, nc)) / 2 - x * u / 2
    def rho(u):
        r = mp.mpf(1)
        for w_j, n_j, nc_j in zip(w, n, nc):
            r *= (1 + (w_j * u) ** 2) ** (n_j / 4) * mp.exp(nc_j * (w_j * u) ** 2 / (2 * (1 + (w_j * u) ** 2)))
        return r
    f = lambda u: mp.sin(theta(u)) / (u * rho(u)) if u != 0 else 0
    return mp.mpf(1) / 2 + mp.quad(f, [0, 1, 5, 20, 100, mp.inf]) / mp.pi

w = [mp.mpf(1) / 5, mp.mpf(2) / 35]; n = [3, 7]
# cross-check Imhof with exact convolution at the null
def sf_null_conv(x):
    # P(w1 X1 + w2 X2 > x) = int_0^{x/w1} f_{chi2_3}(y) P(chi2_7 > (x - w1 y)/w2) dy + P(chi2_3 > x/w1)
    x = float(x)
    a = integrate.quad(lambda y: stats.chi2.pdf(y, 3) * stats.chi2.sf((x - 0.2 * y) / (2 / 35), 7), 0, x / 0.2, epsabs=0, epsrel=1e-12, limit=400)[0]
    return a + stats.chi2.sf(x / 0.2, 3)
print("\ncheck Imhof vs convolution at x=2:", mp.nstr(imhof_sf(2, w, n, [0, 0]), 12), sf_null_conv(2.0))

print("\n=== null quantiles t_alpha and C_opt at (3,3) (local-efficiency table, Section 7) ===")
res = {}
for alpha in [0.05, 0.01]:
    t = optimize.brentq(lambda x: sf_null_conv(x) - alpha, 0.5, 6, xtol=1e-14)
    # E[T;T>t] = sum_j w_j n_j P(T^{(j)} > t), T^{(j)}: n_j -> n_j+2 (size-biasing)
    def sf_shift(x, j):
        nn = list(n); nn[j] += 2
        return float(imhof_sf(x, w, nn, [0, 0]))
    ETt = sum(float(w[j]) * n[j] * sf_shift(t, j) for j in range(2))
    Copt = 0.5 * (ETt - alpha)
    res[alpha] = (t, Copt)
    print(f"alpha={alpha}: t_alpha = {t:.6f},  E[T;T>t] = {ETt:.8f},  C_opt = {Copt:.6e}")
# MC cross-check of C_opt
rng = np.random.default_rng(11)
X1 = rng.chisquare(3, 4_000_000); X2 = rng.chisquare(7, 4_000_000); T = 0.2 * X1 + 2 / 35 * X2
for alpha in [0.05, 0.01]:
    t = res[alpha][0]
    print(f"  MC: P(T>t)={np.mean(T>t):.5f}, C_opt MC = {0.5*(np.mean(T*(T>t))-alpha):.5e}")

print("\n=== power of the quadratic test (non-central mixture, Section 6) ===")
lams = [1.2, 1.3, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 2.0, 3.0, 5.0]
powq = {}
for alpha in [0.05, 0.01]:
    t = res[alpha][0]
    row = []
    for lam in lams:
        nc = [lam ** 2 * float(w[0]) * n[0], lam ** 2 * float(w[1]) * n[1]]
        p = float(imhof_sf(t, w, n, nc))
        row.append(p)
    powq[alpha] = row
    print(f"alpha={alpha}: " + "  ".join(f"lam={l}: {p:.5f}" for l, p in zip(lams, row)))
# MC cross-check at lam=2
lam = 2.0
Y1 = rng.noncentral_chisquare(3, lam ** 2 * 0.6, 2_000_000); Y2 = rng.noncentral_chisquare(7, lam ** 2 * 0.4, 2_000_000)
print("  MC check lam=2, alpha=0.05:", np.mean(0.2 * Y1 + 2 / 35 * Y2 > res[0.05][0]))
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quadpower.npy'),
        np.array([lams, powq[0.05], powq[0.01]]))
print("t_alpha:", {a: res[a][0] for a in res})
