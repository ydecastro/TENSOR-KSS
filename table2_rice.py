"""Table 2 of the KSS paper (Non-asymptotic First- and Second-Order Tail Brackets for the Kostlan-Shub-Smale
Field and Spherical k-Spin Complexity): at the six points (k, d, u) of the table, the Rice integral of Lemma 4.4
with Hadamard's bound, evaluated numerically (Gauss-Legendre in theta, trapezoid in y), the explicit bound
Pbar(u) of Proposition 4.8 with W_k = 43 k^3 (k odd), 9 k^3 (k even) and the factor 1/2 for k even
(Proposition 4.10), and the two values of eps of the proof of Theorem 1.1: with the numerical integral and
A(rho u) exact (A = 2^{-N} H_N), and with the explicit bound (1.7). Then the levels u/u_IMF where each bound
reaches 1. Conventions: GOE in the Mehta normalization, rho^2 = k/(2(k-1)), u_IMF = sqrt(2d-1)/rho.
Requirements: numpy, scipy, sympy. Run: python table2_rice.py (about ten minutes). Output: table2_rice_output.txt."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np, sympy as sp
from math import comb, sqrt, pi, gamma, exp
from scipy.stats import norm
from scipy.special import eval_hermite
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(HERE, "table2_rice_output.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

def W(k): return 43 * k**3 if k % 2 else 9 * k**3
def C_rice_W(k, d): return sqrt(pi) * W(k) * 2**(2 * k) * (k - 1)**((d - 2) / 2) / gamma((d - 1) / 2)
def half(k): return 0.5 if k % 2 == 0 else 1.0
def Pbar(k, d, u):  # bound on P(sup zeta > u), includes the 1/2 for k even
    rho = sqrt(k / (2 * (k - 1))); a = rho * u
    return half(k) * C_rice_W(k, d) * (1 + a)**(d - 1) / a * exp(-a**2 / 2)
def holder(d): return 4 * (2 + 5 / 6 * sqrt(d - 1))**(d - 1)
def uIMF(k, d): return sqrt(2 * d - 1) / sqrt(k / (2 * (k - 1)))

th = sp.symbols('theta', positive=True)
def rice_integral(k, d, x, nth=600, ny=400, t0=1e-3):
    n = d - 2; rho = sqrt(k / (2 * (k - 1)))
    r, s = sp.cos(th), sp.sin(th)
    Sk = sum(r**i for i in range(k)); ws = sp.Matrix([sp.sqrt(comb(k, j)) * r**(k - j) * (1 + r) * s**(j - 2) / Sk for j in range(2, k + 1)])
    f0, f1, f2 = (sp.lambdify(th, sp.diff(ws, th, m), 'numpy') for m in (0, 1, 2))
    gx, gw = np.polynomial.legendre.leggauss(nth)
    ths = t0 + (gx + 1) * (pi - 2 * t0) / 2; wth = gw * (pi - 2 * t0) / 2
    omega_n = 2 * pi**((n + 1) / 2) / gamma((n + 1) / 2)
    ys = np.linspace(x, x + 12, ny)
    total = 0.0
    for t, wt in zip(ths, wth):
        w = np.array(f0(t), float).ravel(); w1 = np.array(f1(t), float).ravel(); w2 = np.array(f2(t), float).ravel()
        V = w @ w; C = w @ w1; gr = w1 @ w1; De = V * gr - C**2
        gt = np.sum(np.arange(2, k + 1) * w**2) / np.sin(t)**2
        btt = np.sum(np.arange(2, k + 1) * (np.arange(2, k + 1) - 1) * w**2) / np.sin(t)**4
        g1 = -(w2 @ w); g2 = -(w2 @ w1)
        Sig = np.array([[V, C], [C, gr]]); Si = np.linalg.inv(Sig)
        m = ys * (Si @ np.array([g1, g2]))[0]
        s2 = max(w2 @ w2 - np.array([g1, g2]) @ Si @ np.array([g1, g2]), 0.0); sb = sqrt(s2)
        Ebrr = m * norm.cdf(m / sb) + sb * norm.pdf(m / sb) if sb > 0 else np.maximum(m, 0)
        hh = rho * ys * norm.cdf(rho * ys) + norm.pdf(rho * ys)
        H = Ebrr * (sqrt(2 * btt) * hh)**n
        p = De**-0.5 * gt**(-n / 2) * (2 * pi)**(-d / 2) * np.exp(-ys**2 * gr / (2 * De))
        total += wt * np.sin(t)**n * np.trapezoid(p * H, ys)
    return omega_n * total

P("Table 2 of the KSS paper: W_k = 43k^3 (k odd), 9k^3 (k even); the factor 1/2 for k even is in Pbar and in both eps columns.")
P("    Columns: (k,d) | u | u/uIMF | Rice integral (count, unchanged) | Pbar(u) | eps numerical (A exact) | eps explicit | Pbar/RI")
rows = {}
for (k, d) in [(3, 5), (3, 6), (4, 6)]:
    N = d - 1; rho = sqrt(k / (2 * (k - 1)))
    for ratio in [2.5, 4.0]:
        u = ratio * uIMF(k, d); a = rho * u
        RI = rice_integral(k, d, u)
        pb = Pbar(k, d, u)
        Aex = eval_hermite(N, a) / 2**N / a**N
        eps_num = half(k) * 3 * (2 + 5 / 6 * sqrt(N))**N * RI / Aex
        eps_ex = holder(d) * pb
        rows[(k, d, ratio)] = (u, RI, pb, eps_num, eps_ex)
        P(f"    ({k},{d}) | {u:.2f} | {ratio} | {RI:.3e} | {pb:.4e} | {eps_num:.4e} | {eps_ex:.4e} | {pb/RI:.3e}")
P("    C_Rice (with the 1/2 for k even): " + ", ".join(f"({k},{d}): {half(k)*C_rice_W(k,d):.4e}" for (k, d) in [(3, 5), (3, 6), (4, 6)]))
P("    Thresholds u/u_IMF where eps_expl = 1:")
for (k, d) in [(3, 5), (3, 6), (4, 6)]:
    r1 = brentq(lambda rr: holder(d) * Pbar(k, d, rr * uIMF(k, d)) - 1, 1.0, 10.0)
    P(f"      ({k},{d}): {r1:.4f}")
P("    Thresholds u/u_IMF where eps_num = 1 (bisection on the Rice integral):")
for (k, d) in [(3, 5), (3, 6), (4, 6)]:
    N = d - 1; rho = sqrt(k / (2 * (k - 1)))
    def g(rr):
        u = rr * uIMF(k, d); a = rho * u
        return half(k) * 3 * (2 + 5 / 6 * sqrt(N))**N * rice_integral(k, d, u, nth=200, ny=200) / (eval_hermite(N, a) / 2**N / a**N) - 1
    P(f"      ({k},{d}): {brentq(g, 1.2, 3.0, xtol=1e-4):.4f}")
P(f"    Hoelder factor 3(2+5/6 sqrt(d-1))^(d-1): d=5: {3*(2+5/6*2)**4:.3e}, d=6: {3*(2+5/6*sqrt(5))**5:.3e}")

out.close()
