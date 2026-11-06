"""
Layer weights of the helix process of the KSS field -- shape of w_j(theta).

    w_j(theta) = binom(k,j)^{1/2} r^{k-j} (1-r^2)^{j/2} / (1-r^k),   r = cos(theta),

the coefficients of the layer decomposition zeta^t = sum_{j=2}^k w_j(theta) Z_j(v)
(Proposition 3.4 of the KSS paper).  This file answers: is w_j monotone, and is it
maximal at the pole?

Closed form for the critical points.  With x = cos(theta), s = sin(theta),

    w_j'/w_j = -(k-j) s/x + j x/s - k x^{k-1} s/(1-x^k) = P_j(x) / [x s (1-x^k)],
    P_j(x)   = k(x^2-1) + j(1-x^k) = (1-x) Q_j(x),
    Q_j(x)   = j (1 + x + ... + x^{k-1}) - k (1+x),

so the interior critical points of w_j are exactly the roots of Q_j in (-1,1), i.e.

    (1 - x^k)/(1 - x^2) = k/j.                                            (*)

Since Q_j(1) = k(j-2), the root at x=1 is picked up only by j=2: that is why j=2 is
the exceptional layer (w_2 -> sigma = sqrt(2(k-1)/k) at the pole, w_j -> 0 for j>=3).

Only the standard library is used.  Running the file prints the table of the critical
points and writes the pgfplots data (w2.dat to w5.dat, V.dat) read by layer_weights_k5.tex,
the source of Figure 2 of the KSS paper (compile it with pdflatex after running this file).
"""
from fractions import Fraction
from math import acos, comb, cos, pi, sin, sqrt
from pathlib import Path

K = 5
SIGMA = sqrt(2 * (K - 1) / K)
HERE = Path(__file__).resolve().parent


def w(j, th, k=K):
    x = cos(th)
    return sqrt(comb(k, j)) * x ** (k - j) * sin(th) ** j / (1 - x ** k)


def Q(j, x, k=K):
    return j * sum(x ** i for i in range(k)) - k * (1 + x)


def critical_points(j, k=K, n=20000):
    """Roots of Q_j in (-1,1), as polar angles, sign changes located by bisection.

    A double root of Q_j produces no sign change; it is a critical point of w_j only
    through the 1/x of the logarithmic derivative, handled by the caller (case j=k,
    where Q_k(x) = k x^2 (1 + x + ... + x^{k-2}) has the double root x=0)."""
    out, xs = [], [-1 + 2 * i / n for i in range(n + 1)]
    for a, b in zip(xs, xs[1:]):
        if Q(j, a, k) * Q(j, b, k) < 0:
            for _ in range(200):
                m = (a + b) / 2
                if Q(j, a, k) * Q(j, m, k) <= 0:
                    b = m
                else:
                    a = m
            out.append((a + b) / 2)
    ths = sorted(acos(x) for x in out)
    if j == k:
        ths.append(pi / 2)          # the double root x = 0
    return sorted(ths)


def report():
    print(f"k = {K},  sigma = sqrt(2(k-1)/k) = {SIGMA:.6f}")
    print("limits:  w_2(0+) = sigma,  w_j(0+) = 0 for j >= 3 (w_j = O(theta^{j-2}));"
          "  w_j(pi-) = 0 for all j\n")
    hdr = f"{'j':>2}  {'w_j(0+)':>9}  {'interior extrema (theta/pi -> w_j)':<44}  behaviour"
    print(hdr + "\n" + "-" * len(hdr))
    for j in range(2, K + 1):
        ths = critical_points(j)
        ext = ", ".join(f"{t/pi:.5f} -> {w(j,t):+.5f}" for t in ths)
        if j == 2:
            beh = (f"decreasing on (0,{ths[0]/pi:.5f} pi), increasing after; "
                   f"sup = sigma as theta->0, not attained")
        elif j == K:
            beh = "single max, = 1 exactly at pi/2"
        elif (K - j) % 2 == 0:
            beh = "two maxima, tangential zero (local min) at pi/2"
        else:
            beh = "one max then one min, sign change at pi/2"
        print(f"{j:>2}  {SIGMA if j == 2 else 0.0:>9.6f}  {ext:<44}  {beh}")

    # V = sum_j w_j^2 = [1 - r^{2k} - k r^{2k-2}(1-r^2)] / (1-r^k)^2, exact rationals
    Vr = lambda r: (1 - r ** (2 * K) - K * r ** (2 * K - 2) * (1 - r ** 2)) / (1 - r ** K) ** 2
    N = 4000
    rs = [Fraction(N - 1 - i, N) for i in range(2 * N - 1)]   # r decreasing <=> theta increasing
    vs = [Vr(r) for r in rs]
    up = [i for i in range(1, len(vs)) if vs[i] > vs[i - 1]]
    print(f"\nV = sum_j w_j^2 on a {len(rs)}-point exact-rational grid in r: "
          f"{'strictly DECREASING in theta throughout' if not up else 'NOT monotone'};"
          f"  V(0+) = sigma^2 = {SIGMA**2:.4f}, V(pi/2) = {float(Vr(Fraction(0))):.4f}, V(pi-) = 0")


def write_data(m=1500):
    for j in range(2, K + 1):
        with open(HERE / f"w{j}.dat", "w") as f:
            f.write("t w\n0 %.10f\n" % (SIGMA if j == 2 else 0.0))
            for i in range(1, m):
                t = pi * i / m
                f.write(f"{t/pi:.8f} {w(j,t):.10f}\n")
            f.write("1 0\n")
    with open(HERE / "V.dat", "w") as f:
        f.write("t v\n0 %.10f\n" % SIGMA ** 2)
        for i in range(1, m):
            t = pi * i / m
            f.write(f"{t/pi:.8f} {sum(w(j,t)**2 for j in range(2,K+1)):.10f}\n")
        f.write("1 0\n")
    print(f"\npgfplots data written to {HERE}")


if __name__ == "__main__":
    report()
    write_data()
