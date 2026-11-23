"""Section 3.5, Figure 2 (fig:Ck) and the sentences before it; Supplementary Material, Lemma D.5 (lem:Ck): the ratio
C_k / (2^(k/2) sqrt(k)) for the odd orders k = 3, 5, ..., 999.

  g_k(theta) = [(|cos| + sin)^k - |cos|^k - k |cos|^(k-1) sin] / (1 - cos^k theta),  Upsilon_k = sup over (0, pi) of g_k;
  v_k = (Upsilon_k + k - 1) sqrt(k),  l_k = 2 (k-1) sqrt(k),  L_k = log(v_k / (v_k - l_k));
  C_k > v_k solves (C^2 - v_k^2)/2 - log(C/v_k) = L_k.

For k odd, g_k(pi - theta) <= g_k(theta) on (0, pi/2] (same numerator, larger denominator), so the supremum is
searched on (0, pi/2).

Two computations.
1. mpmath at 560 digits, every odd k from 3 to 999: the maximiser of g_k is the zero of (log g_k)' next to the
   maximiser on a grid of 200000 points (bisection, then secant); delta_k = C_k - v_k by Newton's method from the
   upper bracket of the lemma. At k = 999 the ratio is 1 + 4.3e-148 and delta_k/v_k is of order 1e-451: hence the
   number of digits. This computation gives the figure and every statement printed below.
2. Double precision, every odd k: the same quantities by a scan and a golden-section search, delta_k by bisection.
   It is compared with computation 1.
Checked for every odd k from 3 to 999 (computation 1): the bounds of Lemma D.5,
   2^(k/2) (1 - (k+1) 2^(-k)) <= Upsilon_k <= max{ 2^(k/2) / (1 - cos^k(1/4)),  k (1 + k/32) e^(k/4) },
   v_k < C_k <= v_k + v_k L_k / (v_k^2 - 1);
the ratio is largest at k = 5; it decreases strictly from k = 5 to k = 999; it is below 1.02 for 21 <= k <= 999.
Output: fig_Ck.pdf (and .png) in the directory given by --out (default: this directory), and the printed table.
Deterministic. Run: python3 constant_Ck_growth.py (about one minute).
"""
import argparse, math, os
import numpy as np
import mpmath as mp

DPS = 560
mp.mp.dps = DPS
NGRID = 200000
TH = (np.arange(NGRID // 2) + 0.5) * math.pi / NGRID      # midpoints of a grid of (0, pi/2)
GR = (math.sqrt(5) - 1) / 2


def g_double(th, k):
    c = np.cos(th); a = np.abs(c); b = np.sin(th)
    return ((a + b) ** k - a ** k - k * a ** (k - 1) * b) / (1.0 - c ** k)


def g_mp(th, k):
    c = mp.cos(th); a = abs(c); b = mp.sin(th)
    return ((a + b) ** k - a ** k - k * a ** (k - 1) * b) / (1 - c ** k)


def dlogg_mp(th, k):
    """(log g_k)' on (0, pi/2), where |cos| = cos."""
    c = mp.cos(th); s = mp.sin(th)
    n = (c + s) ** k - c ** k - k * c ** (k - 1) * s
    dn = k * (c + s) ** (k - 1) * (c - s) + k * c ** (k - 1) * s + k * (k - 1) * c ** (k - 2) * s * s - k * c ** k
    return dn / n - k * c ** (k - 1) * s / (1 - c ** k)


def upsilon_mp(k):
    vals = g_double(TH, k)
    i = int(np.argmax(vals)); th0 = TH[i]; h = math.pi / NGRID
    f = lambda t: dlogg_mp(t, k)
    a, b = mp.mpf(th0 - 1.5 * h), mp.mpf(th0 + 1.5 * h)
    fa, fb = f(a), f(b)
    assert fa > 0 > fb, (k, fa, fb)                        # sign change of (log g_k)' around the grid maximiser
    for _ in range(40):
        m = (a + b) / 2; fm = f(m)
        if fm > 0: a, fa = m, fm
        else: b, fb = m, fm
    x0, x1, f0, f1 = a, b, fa, fb
    for _ in range(60):
        if f1 == f0: break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
        if abs(x1 - x0) < mp.mpf(10) ** (-(DPS - 30)): break
    assert abs(f1) < mp.mpf(10) ** (-(DPS // 2)), (k, f1)
    U = g_mp(x1, k)
    assert float(U) >= vals[i] * (1 - 1e-9), k
    return U


def constants_mp(k):
    U = upsilon_mp(k)
    sk = mp.sqrt(k); v = (U + k - 1) * sk; l = 2 * (k - 1) * sk; L = mp.log(v / (v - l))
    F = lambda d: d * (2 * v + d) / 2 - mp.log1p(d / v) - L
    hi = L * v / (v * v - 1)                               # upper bracket of the lemma
    d = hi
    for _ in range(200):
        dn = d - F(d) / (v + d - 1 / (v + d))
        if abs(dn - d) <= abs(d) * mp.mpf(10) ** (-(DPS - 15)):
            d = dn; break
        d = dn
    assert abs(F(d)) <= L * mp.mpf(10) ** (-(DPS - 40)), k
    return U, v, l, L, d, hi


def ratio_double(k):
    v = g_double(TH[::5], k); i = int(np.argmax(v)); ths = TH[::5]
    a, b = ths[max(i - 2, 0)], ths[min(i + 2, len(ths) - 1)]
    f = lambda t: -float(g_double(np.array([t]), k)[0])
    c, d = b - GR * (b - a), a + GR * (b - a); fc, fd = f(c), f(d)
    while b - a > 1e-13:
        if fc < fd: b, d, fd = d, c, fc; c = b - GR * (b - a); fc = f(c)
        else: a, c, fc = c, d, fd; d = a + GR * (b - a); fd = f(d)
    U = max(-fc, -fd, float(v[i]))
    vk = (U + k - 1) * math.sqrt(k); l = 2 * (k - 1) * math.sqrt(k); L = -math.log1p(-l / vk)
    F = lambda dl: dl * (2 * vk + dl) / 2 - math.log1p(dl / vk) - L
    lo, hi = 0.0, L * vk / (vk * vk - 1)
    for _ in range(200):
        mid = (lo + hi) / 2
        if F(mid) < 0: lo = mid
        else: hi = mid
    return (vk + hi) / (2.0 ** (k / 2) * math.sqrt(k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    ks = list(range(3, 1000, 2))
    r1 = {}; rows = {}; ok = True; worst = 0.0
    for k in ks:
        U, v, l, L, d, hi = constants_mp(k)
        s = mp.mpf(2) ** (mp.mpf(k) / 2)
        lowU = s * (1 - (k + 1) * mp.mpf(2) ** (-k))
        upU = max(s / (1 - mp.cos(mp.mpf(1) / 4) ** k), k * (1 + mp.mpf(k) / 32) * mp.exp(mp.mpf(k) / 4))
        ok = ok and (lowU <= U <= upU) and U > k - 1 and v > l > 1 and 0 < d <= hi
        r1[k] = (v + d) / (s * mp.sqrt(k)) - 1               # ratio - 1
        rows[k] = (U, v + d, d / v)
        worst = max(worst, abs(float(1 + r1[k]) - ratio_double(k)) / float(1 + r1[k]))
    print(f"mpmath at {DPS} digits, odd k = 3, ..., 999 ({len(ks)} orders)")
    print(f"bounds of Lemma D.5 hold for every odd k from 3 to 999: {ok}")
    print(f"largest relative difference between the ratio at {DPS} digits and in double precision: {worst:.1e}")
    print("  k   Upsilon_k           C_k                 C_k/v_k - 1   C_k/(2^(k/2) sqrt k) - 1")
    for k in (3, 5, 7, 9, 11, 15, 19, 21, 23, 31, 51, 101, 301, 999):
        U, C, dv = rows[k]
        print(f"{k:4d}  {mp.nstr(U, 10):>18s}  {mp.nstr(C, 10):>18s}  {mp.nstr(dv, 3):>12s}   {mp.nstr(r1[k], 6):>14s}")
    kmax = max(ks, key=lambda k: r1[k])
    dec = all(r1[k + 2] < r1[k] for k in ks if 5 <= k < 999)
    below = all(r1[k] < mp.mpf("0.02") for k in ks if k >= 21)
    print(f"the ratio is largest at k = {kmax}, where it is {mp.nstr(1 + r1[kmax], 5)}; at k = 3 it is {mp.nstr(1 + r1[3], 5)}")
    print(f"it decreases strictly from k = 5 to k = 999: {dec}")
    print(f"it is below 1.02 for every odd k from 21 to 999: {below}; at k = 19 it is {mp.nstr(1 + r1[19], 5)}, at k = 21 it is {mp.nstr(1 + r1[21], 5)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.0,
        "lines.markersize": 3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    ax.axhline(1.0, color="0.6", lw=0.6, zorder=0)
    ax.plot(ks, [float(1 + r1[k]) for k in ks], "o", color="k", ms=1.8, mew=0)
    ax.set_xscale("log"); ax.set_xlim(2.6, 1150); ax.set_ylim(0.95, 1.8)
    ax.set_xticks([3, 5, 7, 10, 30, 100, 300, 1000]); ax.set_xticklabels(["3", "5", "7", "10", "30", "100", "300", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(r"order $k$ (odd)"); ax.set_ylabel(r"$C_k\,/\,(2^{k/2}\sqrt{k})$")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_Ck.{ext}"), dpi=200)
    print("written: fig_Ck.pdf, fig_Ck.png")
