"""Proposition 2 (prop:closed): the closed forms of G_r and H_r against high-precision quadrature, for random
spectra at (k,d) = (3,5), (3,8), (5,6).
"""
import mpmath as mp, numpy as np
from itertools import combinations
from math import comb
mp.mp.dps = 40
rng = np.random.default_rng(2)
def esym(e, j): return mp.fsum(mp.fprod(c) for c in combinations(e, j)) if j else mp.mpf(1)
for (k, d) in [(3, 5), (3, 8), (5, 6)]:
    N = d - 1; m = comb(d + k - 1, k); kap = m - d; nu = m - 1
    gam = mp.sqrt(mp.mpf(kap) / nu)
    cnu = mp.gamma((nu + 1) / mp.mpf(2)) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / mp.mpf(2)))
    e = [mp.mpf(float(x)) for x in rng.normal(size=N) * 1.5]
    def I(p, l):
        if p == 0: return mp.ncdf(-l)
        if p == 1: return mp.npdf(l)
        return l ** (p - 1) * mp.npdf(l) + (p - 1) * I(p - 2, l)
    def K(p, a):
        z = nu / (nu + a * a); A = (nu - p) / mp.mpf(2); B = (p + 1) / mp.mpf(2)
        return cnu * mp.mpf(nu) ** B / 2 * mp.beta(A, B) * mp.betainc(A, B, 0, z, regularized=True)
    for l in [mp.mpf(max(e)) + 0.5, mp.mpf(4), mp.mpf(9)]:
        Gc = mp.fsum((-1) ** j * esym(e, j) * I(N - j, l) for j in range(N + 1))
        Gq = mp.quad(lambda u: mp.fprod([u - x for x in e]) * mp.npdf(u), [l, l + 3, l + 10, mp.inf])
        Hc = gam * mp.fsum((-1) ** j * esym(e, j) * gam ** (N - j) * K(N - j, l / gam) for j in range(N + 1))
        Hq = mp.quad(lambda u: mp.fprod([u - x for x in e]) * cnu * (1 + (u / gam) ** 2 / nu) ** (-(nu + 1) / mp.mpf(2)), [l, l + 3, l + 10, l + 50, l + 300, mp.inf])
        print((k, d), "l=", mp.nstr(l, 5), " G rel.err", mp.nstr(abs(Gc / Gq - 1), 3), "  H rel.err", mp.nstr(abs(Hc / Hq - 1), 3))
