"""How typical is the draw of level_map_draw.py: percentiles of the two level maps and of the extra level over
3000 GOE draws (closed forms of Proposition 2), alpha = 0.05 and 0.01, v = 4.
"""
import numpy as np
from math import comb, sqrt, gamma, pi
from scipy import stats, special, optimize
def esym(ev):
    N = len(ev); e = np.zeros(N + 1); e[0] = 1
    for x in ev:
        for j in range(N, 0, -1): e[j] += x * e[j - 1]
    return e
def Imom(p, l):
    if p == 0: return stats.norm.sf(l)
    if p == 1: return stats.norm.pdf(l)
    return l ** (p - 1) * stats.norm.pdf(l) + (p - 1) * Imom(p - 2, l)
def make(k, d):
    N = d - 1; m = comb(d + k - 1, k); kap = m - d; nu = m - 1; gam = sqrt(kap / nu)
    def K(p, a):
        z = nu / (nu + a * a); A = (nu - p) / 2; B = (p + 1) / 2
        return nu ** B / 2 * special.beta(A, B) * special.betainc(A, B, z)
    G = lambda e, l: sum((-1) ** j * e[j] * Imom(N - j, l) for j in range(N + 1))
    H = lambda e, l: sum((-1) ** j * e[j] * gam ** (N - j) * K(N - j, l / gam) for j in range(N + 1))
    return G, H
def lev(F, e, v, a):
    t = a * F(e, v); return optimize.brentq(lambda l: F(e, l) - t, v, 40)
for (k, d) in [(3, 5), (3, 8), (5, 6)]:
    G, H = make(k, d); rho = sqrt(k / (2 * (k - 1))); N = d - 1
    rng = np.random.default_rng(3)
    # the specific draw (order (3,5),(3,8),(5,6) in the original script)
    draws = {}
    for kk, dd in [(3, 5), (3, 8), (5, 6)]:
        A = rng.normal(size=(dd - 1, dd - 1)); A = (A + A.T) / 2; draws[(kk, dd)] = np.linalg.eigvalsh(A) / np.sqrt(kk / (2 * (kk - 1)))
    ev0 = draws[(k, d)]
    rng2 = np.random.default_rng(99); out = []
    for i in range(3000):
        A = rng2.normal(size=(N, N)); A = (A + A.T) / 2; ev = np.linalg.eigvalsh(A) / rho
        if ev.max() >= 4: continue
        e = esym(ev); out.append((lev(G, e, 4, 0.05), lev(H, e, 4, 0.05)))
    out = np.array(out); e0 = esym(ev0); g0, h0 = lev(G, e0, 4, 0.05), lev(H, e0, 4, 0.05)
    pct = lambda arr, x: 100 * np.mean(arr <= x)
    print(f"{(k,d)}: {len(out)} draws with lmax<4 (of 3000). rng(3) draw: known {g0:.4f} (pct {pct(out[:,0],g0):.0f}), t {h0:.4f} (pct {pct(out[:,1],h0):.0f}),"
          f" extra {h0-g0:.4f} (pct {pct(out[:,1]-out[:,0],h0-g0):.0f}); medians: known {np.median(out[:,0]):.4f}, t {np.median(out[:,1]):.4f}, extra {np.median(out[:,1]-out[:,0]):.4f}")
