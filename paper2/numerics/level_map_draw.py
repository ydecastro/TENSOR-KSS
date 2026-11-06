"""The level-map table of Section 7 with the draw of Omega used in the paper: numpy default_rng(3), one
symmetrised N(0,1) matrix (A+A^T)/2 per (k,d), drawn in the order (3,5), (3,8), (5,6); Omega = G/rho.
"""
import numpy as np
from math import comb, sqrt
from scipy import stats, integrate, optimize
import mpmath as mp
mp.mp.dps = 30
def Gr(l, e):
    return mp.quad(lambda u: mp.fprod([u - x for x in e]) * mp.npdf(u), [l, l + 5, l + 20, mp.inf])
def Hr(l, e, kap, nu):
    return mp.quad(lambda u: mp.fprod([u - x for x in e]) * (1 + u * u / kap) ** (-(mp.mpf(nu) + 1) / 2), [l, l + 5, l + 20, l + 200, mp.inf])
def level(F, v, a):
    t = a * F(v)
    return mp.findroot(lambda u: mp.log(F(u) / t), (mp.mpf(v) + 1e-6, mp.mpf(v) + 8), solver='illinois')
rng = np.random.default_rng(3)
paper = {((3, 5), 0.05): (4.897, 5.486), ((3, 5), 0.01): (5.276, 6.171), ((3, 8), 0.05): (5.036, 5.202), ((5, 6), 0.05): (4.862, 4.923)}
for k, d in ((3, 5), (3, 8), (5, 6)):
    N = d - 1; rho = sqrt(k / (2 * (k - 1))); m = comb(d + k - 1, k); kap = m - d; nu = m - 1
    A = rng.normal(size=(N, N)); A = (A + A.T) / 2; ev = np.linalg.eigvalsh(A) / rho
    e = [mp.mpf(float(x)) for x in ev]
    print(f"(k,d)=({k},{d}) eigenvalues of Omega:", np.round(ev, 4), " lmax=", round(ev.max(), 4))
    for a in (0.05, 0.01):
        v = 4
        if ((k, d), a) not in paper:
            continue
        uG = level(lambda l: Gr(l, e), v, a); uH = level(lambda l: Hr(l, e, kap, nu), v, a)
        pG, pH = paper[((k, d), a)]
        print(f"   alpha={a}: known {mp.nstr(uG,7)} (paper {pG})  t {mp.nstr(uH,7)} (paper {pH})  extra {mp.nstr(uH-uG,6)} (paper {pH-pG:+.3f})"
              f"  rel. extra {mp.nstr((uH-uG)/uG*100,4)}%")
