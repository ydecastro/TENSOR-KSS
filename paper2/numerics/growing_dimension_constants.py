"""Theorem 3 (thm:sqrtd), Section 3.5, and the paragraph "Growing dimension" of Section 1: the constants
Upsilon_k and C_k for k = 3, 5, 7 (paper: Upsilon_3 = 2.214, C_3 = 7.70, C_5 = 21.74, C_7 = 47.06).

  g_k(theta) = [(|cos| + sin)^k - |cos|^k - k |cos|^(k-1) sin] / (1 - cos^k theta),  Upsilon_k = sup over (0, pi) of g_k
               (Lemma D.2, lem:banach, of the Supplementary Material);
  v_k = (Upsilon_k + k - 1) sqrt(k),  l_k = 2 (k-1) sqrt(k);
  C_k > v_k solves (C^2 - v_k^2)/2 - log(C/v_k) = log(v_k/(v_k - l_k)).

Method: a scan of 200001 points locates the maximum of g_k and shows that it is the largest interior local maximum
on the grid; golden-section refinement at 40 digits; C_k by a bracketing root finder. Deterministic.
Run: python3 growing_dimension_constants.py (a few seconds).
"""
import math
import numpy as np
import mpmath as mp

mp.mp.dps = 40
PAPER = {3: ("2.214", "7.70"), 5: (None, "21.74"), 7: (None, "47.06")}


def golden_min(F, a, b, tol):
    g = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(a), mp.mpf(b)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = F(c), F(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = F(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = F(d)
    return (c, fc) if fc < fd else (d, fd)


def scan(f, xs):
    v = np.array([f(x) for x in xs])
    dv = np.diff(v)
    locmax = np.where((dv[:-1] > 0) & (dv[1:] <= 0))[0] + 1
    return v, [(float(xs[i]), float(v[i])) for i in locmax]


for k in (3, 5, 7):
    gk_np = lambda th: ((abs(math.cos(th)) + math.sin(th)) ** k - abs(math.cos(th)) ** k
                        - k * abs(math.cos(th)) ** (k - 1) * math.sin(th)) / (1 - math.cos(th) ** k)
    gk_mp = lambda th: ((abs(mp.cos(th)) + mp.sin(th)) ** k - abs(mp.cos(th)) ** k
                        - k * abs(mp.cos(th)) ** (k - 1) * mp.sin(th)) / (1 - mp.cos(th) ** k)
    ths = np.linspace(1e-4, math.pi - 1e-4, 200001)
    v, nmax = scan(gk_np, ths)
    i = int(np.argmax(v))
    th_s, mG = golden_min(lambda t: -gk_mp(t), ths[i - 2], ths[i + 2], mp.mpf(10) ** -30)
    Ups = -mG
    vk = (Ups + k - 1) * mp.sqrt(k)
    lk = 2 * (k - 1) * mp.sqrt(k)
    rhs = mp.log(vk / (vk - lk))
    Fc = lambda Cc: (Cc * Cc - vk * vk) / 2 - mp.log(Cc / vk) - rhs
    Ck = mp.findroot(Fc, (vk * (1 + mp.mpf(10) ** -12), 10 * vk), solver="anderson")
    pu, pc = PAPER[k]
    print(f"k = {k}:")
    print(f"   Upsilon_k = {mp.nstr(Ups, 20)} at theta = {mp.nstr(th_s, 15)} (interior local maxima of g_k on the "
          f"grid: {[(round(a, 6), round(b, 6)) for a, b in nmax]}; limit at 0 is k-1 = {k - 1})"
          + (f"   (paper {pu})" if pu else ""))
    print(f"   v_k = {mp.nstr(vk, 15)}, l_k = {mp.nstr(lk, 15)}")
    print(f"   C_k = {mp.nstr(Ck, 20)} (residual {mp.nstr(Fc(Ck), 3)})   (paper {pc})")
