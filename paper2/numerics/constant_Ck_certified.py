"""Theorem 3 (thm:sqrtd), last line, and Lemma D.5 (lem:Ck) of the Supplementary Material: the bound
C_k <= 1.72 * 2^(k/2) sqrt(k) for the odd orders 3 <= k <= 123, and C_k <= 1.02 * 2^(k/2) sqrt(k) for 21 <= k <= 123,
by interval arithmetic (mpmath.iv). Lemma D.5 gives the bounds for the odd k >= 125 by hand, so this script covers the finite range that the lemma leaves.

  g_k(theta) = [(|cos| + sin)^k - |cos|^k - k |cos|^(k-1) sin] / (1 - cos^k theta),  Upsilon_k = sup over (0, pi) of g_k;
  v_k = (Upsilon_k + k - 1) sqrt(k),  l_k = 2 (k-1) sqrt(k);  C_k > v_k solves (C^2 - v_k^2)/2 - log(C/v_k) = log(v_k/(v_k - l_k)).

What is certified, and how.
1. An enclosure [U_lo, U_hi] of Upsilon_k. For k odd, g_k(pi - t) <= g_k(t) on (0, pi/2] (same numerator, larger
   denominator), so the supremum is over (0, pi/2]. There, with a = cos t and b = sin t, 1 - a = b^2/(1 + a) gives the
   form without singularity at 0
       g_k = (1 + a) * sum_{j=2..k} binom(k,j) a^(k-j) b^(j-2) / sum_{i=0..k-1} a^i,
   two polynomials with non-negative coefficients. On a cell [t0, t1] an upper bound is obtained with a at t0 in
   the numerator, b at t1, and a at t1 in the denominator; a lower bound is the value at the midpoint. All the
   evaluations are intervals. Branch and bound: a cell whose upper bound is below the best lower bound is
   discarded, the others are split, until the upper bounds exceed the best lower bound by a relative TOL at most.
2. An enclosure of C_k. C_k is increasing in v_k when v_k (v_k - l_k) > 1 (implicit function; checked on the
   enclosure). Upper end: the bracket of Lemma D.5 at the upper end of v_k, C <= v + v log(v/(v-l))/(v^2-1). Lower
   end: a point c with F(c) < 0 at the lower end of v_k, F increasing in c.
3. With r_k = C_k/(2^(k/2) sqrt k): r_k <= 1.72 for every odd k in [3, 123]; r_5 > 1.7186; r_k < r_5 for k != 5;
   r_k < 1.02 for every odd k in [21, 123].
The script stops with an error if one of these fails. Deterministic. Run: python3 constant_Ck_certified.py
(about fifteen minutes).
"""
import math
from mpmath import iv, mp, mpf

iv.dps = 40
mp.dps = 40
TOL = {5: mpf("2e-5")}          # relative width asked for the enclosure of Upsilon_k (default 1e-3)
BIN = {}


def pt(x, n=9, up=False):
    """Decimal string with n significant digits of an end point, rounded down (lower end) or up (upper end)."""
    from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING
    from mpmath.libmp import to_rational
    p, q = to_rational(x._mpi_[0] if hasattr(x, "_mpi_") else x._mpf_)
    return str(Context(prec=n, rounding=ROUND_CEILING if up else ROUND_FLOOR).divide(Decimal(p), Decimal(q)))


def binom(k):
    if k not in BIN:
        BIN[k] = [math.comb(k, j) for j in range(k + 1)]
    return BIN[k]


def cell_bounds(k, t0, t1):
    """Certified (upper bound of g_k on [t0, t1], lower bound of g_k at the midpoint), 0 <= t0 < t1 as mpf points; the last t1 is 2.3e-41 above pi/2, where the lower bound of cos is set to 0."""
    c = binom(k)
    a_hi = iv.cos(iv.mpf(t0)).b; a_lo = iv.cos(iv.mpf(t1)).a; b_hi = iv.sin(iv.mpf(t1)).b
    if a_lo < 0: a_lo = mpf(0)
    A, B, Al = iv.mpf(a_hi), iv.mpf(b_hi), iv.mpf(a_lo)
    num = iv.mpf(0)
    for j in range(2, k + 1):
        num += c[j] * A ** (k - j) * B ** (j - 2)
    den = iv.mpf(0)
    for i in range(k):
        den += Al ** i
    up = ((1 + A) * num / den).b
    tm = iv.mpf((t0 + t1) / 2); am, bm = iv.cos(tm), iv.sin(tm)
    numm = iv.mpf(0)
    for j in range(2, k + 1):
        numm += c[j] * am ** (k - j) * bm ** (j - 2)
    denm = iv.mpf(0)
    for i in range(k):
        denm += am ** i
    lo = ((1 + am) * numm / denm).a
    return up, lo


def upsilon_enclosure(k):
    tol = TOL.get(k, mpf("1e-3"))
    # the last cell ends at a point at or above pi/2; there the lower bound of cos is set to 0 (see cell_bounds)
    n0 = 64
    edges = [mpf(i) * (iv.pi.a / 2) / n0 for i in range(n0)] + [iv.pi.b / 2]
    cells = [(edges[i], edges[i + 1]) for i in range(n0)]
    best_lo = mpf(0); work = []
    for (t0, t1) in cells:
        up, lo = cell_bounds(k, t0, t1); best_lo = max(best_lo, lo); work.append((up, t0, t1))
    nsplit = 0
    while True:
        work = [w for w in work if not (w[0] < best_lo)]      # an undecided comparison keeps the cell
        top = max(w[0] for w in work)
        if top <= best_lo * (1 + tol):
            return best_lo, top, nsplit
        new = []
        for (up, t0, t1) in work:
            if up <= best_lo * (1 + tol / 2):
                new.append((up, t0, t1)); continue
            tm = (t0 + t1) / 2; nsplit += 1
            for (s0, s1) in ((t0, tm), (tm, t1)):
                u, l = cell_bounds(k, s0, s1); best_lo = max(best_lo, l); new.append((u, s0, s1))
        work = new


def ratio_enclosure(k):
    U_lo, U_hi, nsplit = upsilon_enclosure(k)
    sk = iv.sqrt(iv.mpf(k)); l = 2 * (k - 1) * sk
    v_lo = (iv.mpf(U_lo) + k - 1) * sk; v_hi = (iv.mpf(U_hi) + k - 1) * sk
    assert (v_lo - l).a > 0 and (v_lo * (v_lo - l)).a > 1 and l.a > 1, k      # C_k increasing in v_k on the enclosure
    vh = iv.mpf(v_hi.b)
    c_hi = (vh + vh * iv.log(vh / (vh - l)) / (vh * vh - 1)).b               # bracket of Lemma D.5 at the upper end
    vl = iv.mpf(v_lo.a)
    F = lambda c: (c * c - vl * vl) / 2 - iv.log(c / vl) - iv.log(vl / (vl - l))
    lo, hi = vl.a, (vl + vl * iv.log(vl / (vl - l)) / (vl * vl - 1)).b      # bisection in mpf, then a certified sign
    for _ in range(60):
        mid = (lo + hi) / 2
        if F(iv.mpf(mid)).b < 0: lo = mid
        else: hi = mid
    assert F(iv.mpf(lo)).b <= 0 or lo == vl.a, k
    c_lo = lo
    scale = iv.mpf(2) ** (iv.mpf(k) / 2) * sk
    return (iv.mpf(c_lo) / scale).a, (iv.mpf(c_hi) / scale).b, U_lo, U_hi, nsplit


if __name__ == "__main__":
    ks = list(range(3, 125, 2)); enc = {}
    print("interval arithmetic, mpmath.iv at 40 digits; odd k = 3, ..., 123")
    print("  k    Upsilon_k in                      C_k/(2^(k/2) sqrt k) in            cells split")
    for k in ks:
        r_lo, r_hi, U_lo, U_hi, ns = ratio_enclosure(k); enc[k] = (r_lo, r_hi)
        if k <= 25 or k % 20 == 3:
            print(f"{k:4d}   [{pt(U_lo)}, {pt(U_hi, up=True)}]   [{pt(r_lo, 8)}, {pt(r_hi, 8, up=True)}]   {ns}")
    c1 = all(enc[k][1] <= iv.mpf("1.72").a for k in ks)
    c2 = enc[5][0] > iv.mpf("1.7186").b
    c3 = all(enc[k][1] < enc[5][0] for k in ks if k != 5)
    c4 = all(enc[k][1] < iv.mpf("1.02").a for k in ks if k >= 21)
    print(f"certified: C_k <= 1.72 * 2^(k/2) sqrt(k) for every odd k in [3, 123]: {c1}")
    print(f"certified: the ratio at k = 5 is in [{pt(enc[5][0], 8)}, {pt(enc[5][1], 8, up=True)}], above 1.7186: {c2}")
    print(f"certified: the ratio at every other odd k in [3, 123] is below the ratio at k = 5: {c3}")
    print(f"certified: the ratio is below 1.02 for every odd k in [21, 123]: {c4}; largest upper end there: "
          f"{pt(max(enc[k][1] for k in ks if k >= 21), 6, up=True)}")
    assert c1 and c2 and c3 and c4
