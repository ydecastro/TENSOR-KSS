"""Section 7, 'The power bound': the Kac-Rice mass delta^lambda(u_+; B_b0) and the power bound of Theorem 5
(thm:main) at the tuple of Remark 9 (rem:tuple); and the estimation sandwich at (3,5) of Section 7: the explicit
upper bound of Theorem 6(ii) (thm:estim) and its direct Kac-Rice version.

delta^lambda(u;B) = omega_{d-2} ((k-1)/pi)^{N/2} int_I (1-c^2)^{(d-3)/2} exp(-k lam^2 c^{2k-2}(1-c^2)/2)
                      int_u^inf A_lam(rho x;c) phi(x - lam c^k) dx dc
with A_lam(nu;c) = D_N(nu) - beta_2(c) D_{N-1}(nu) (expected determinant, Lemma 29, lem:spikedet), N = d-1.
We integrate in theta (c = cos theta): (1-c^2)^{(d-3)/2} dc = sin^{d-2}(theta) dtheta.
"""
import numpy as np
import mpmath as mp
from math import comb, sqrt, pi, gamma, factorial, log, exp
from scipy import stats, integrate, special

mp.mp.dps = 30

def Dn_coeffs(n):
    # D_n(nu) = sum_i (-1)^i n!/(i!(n-2i)! 4^i) nu^{n-2i}
    c = np.zeros(n + 1)
    for i in range(n // 2 + 1):
        c[n - 2 * i] += (-1) ** i * factorial(n) / (factorial(i) * factorial(n - 2 * i) * 4 ** i)
    return c  # c[p] = coefficient of nu^p

def gauss_trunc_moments(p_max, u, m):
    """M_p = int_u^inf x^p phi(x-m) dx, p=0..p_max, exact via recursion on y = x - m."""
    a = u - m
    # J_q = int_a^inf y^q phi(y) dy : J_0 = 1-Phi(a), J_1 = phi(a), J_q = a^{q-1}phi(a) + (q-1) J_{q-2}
    J = [mp.ncdf(-a), mp.npdf(a)]
    for q in range(2, p_max + 1):
        J.append(a ** (q - 1) * mp.npdf(a) + (q - 1) * J[q - 2])
    M = []
    for p in range(p_max + 1):
        M.append(mp.fsum(mp.binomial(p, q) * m ** (p - q) * J[q] for q in range(p + 1)))
    return M

class KR:
    def __init__(self, k, d, lam):
        self.k, self.d, self.lam = k, d, mp.mpf(lam)
        self.N = d - 1
        self.rho = mp.sqrt(mp.mpf(k) / (2 * (k - 1)))
        self.cN = Dn_coeffs(self.N); self.cN1 = Dn_coeffs(self.N - 1)
        self.pref = 2 * mp.pi ** (mp.mpf(d - 1) / 2) / mp.gamma(mp.mpf(d - 1) / 2) * (mp.mpf(k - 1) / mp.pi) ** (mp.mpf(self.N) / 2)

    def inner(self, u, c):
        """int_u^inf A_lam(rho x;c) phi(x - lam c^k) dx with A = D_N - beta_2 D_{N-1} (as polynomials in x)."""
        k, lam, rho, N = self.k, self.lam, self.rho, self.N
        beta2 = lam * mp.sqrt(mp.mpf(k * (k - 1)) / 2) * c ** (k - 2) * (1 - c * c)
        M = gauss_trunc_moments(N, u, lam * c ** k)
        s = mp.fsum(mp.mpf(self.cN[p]) * rho ** p * M[p] for p in range(N + 1))
        s -= beta2 * mp.fsum(mp.mpf(self.cN1[p]) * rho ** p * M[p] for p in range(N))
        return s

    def density_theta(self, u, th):
        k, lam, d = self.k, self.lam, self.d
        c = mp.cos(th); s = mp.sin(th)
        return self.pref * s ** (d - 2) * mp.exp(-k * lam ** 2 * c ** (2 * k - 2) * s * s / 2) * self.inner(u, c)

    def mass(self, u, th_lo, th_hi, weight=None, pts=None):
        f = (lambda th: self.density_theta(u, th)) if weight is None else (lambda th: weight(th) * self.density_theta(u, th))
        grid = [th_lo] + (pts or []) + [th_hi]
        return mp.quad(f, grid)

# ---------------------------------------------------------------------------------------------
print("=== Section 7, 'The power bound': delta^lambda(u_+; B_b0) at (3,5), lam=150, b0=6 ===")
k, d, lam, b0 = 3, 5, 150, 6
kr = KR(k, d, lam)
u_plus = mp.mpf('109.73995605332618')  # from a1 (u^dagger = 60.966642..., u_+ = 1.8 u^dagger)
th_b = mp.asin(mp.sqrt(mp.mpf(b0)) / lam)
sc = 1 / (lam * mp.sqrt(k))
delta = kr.mass(u_plus, 0, th_b, pts=[sc * t for t in [1, 2, 3, 4] if sc * t < th_b])
lim = 1 - mp.gammainc(mp.mpf(d - 1) / 2, k * b0 / mp.mpf(2), mp.inf, regularized=True)
print("delta =", mp.nstr(delta, 10), "  limit P(chi2_4 <= 18) =", mp.nstr(lim, 10), "  diff =", mp.nstr(lim - delta, 4))
# with u_+ rounded as in the text
for uu in [mp.mpf('109.7'), mp.mpf('109.74')]:
    print("  (u=", uu, ") delta =", mp.nstr(kr.mass(uu, 0, th_b, pts=[sc * t for t in [1, 2, 3, 4]]), 10))
E = mp.mpf('0.006911631859572436')  # from a1
print("power bound (1-E) delta =", mp.nstr((1 - E) * delta, 8), " ; with E=6.9e-3 exactly:", mp.nstr((1 - mp.mpf('0.0069')) * delta, 8))
print("u_+ in sd below lam:", mp.nstr(lam - u_plus, 6), "; rho u_+ =", mp.nstr(kr.rho * u_plus, 6))
# the whole sphere, for orientation: expected number of critical points above u_+ (near + far)
tot = kr.mass(u_plus, 0, mp.pi / 2, pts=[th_b, 0.05, 0.1, 0.3])
print("delta(u_+; whole hemisphere c>=0) =", mp.nstr(tot, 10))

# ---------------------------------------------------------------------------------------------
print("\n=== Section 7, 'The estimation sandwich' at (3,5) ===")
k, d = 3, 5; N = d - 1
rho = sqrt(k / (2 * (k - 1)))

def rho_var(th1):
    ths = np.linspace(1e-6, th1, 20001)
    return 1 - np.min(np.cos(ths) ** (2 * k - 2) * (np.sin(ths) / ths) ** 2)

def upper_ii(lam, u, th1, kr):
    """Thm estim (ii): (d-1)/(k lam^2) (1+eps_A)/(1-varrho)^{(d+1)/2} + Rem_far + Phi(u-lam)."""
    vr = rho_var(th1)
    bstar = lam * sqrt(k * (k - 1) / 2) * np.sin(th1) ** 2
    m1 = lam * np.cos(th1) ** k
    rN = exp(N ** 2 / (2 * m1 ** 2)) - 1
    qs = max(2 * N, 8)
    edet_paper = 2 * sqrt(2) * (5 / 4 + 5 / 6 * sqrt(2 * qs / N ** 2)) ** N * exp(-0.25 * (rho * u - bstar - sqrt(2 * N)) ** 2)
    epsA = bstar / (rho * u) + rN + edet_paper
    rem = kr.mass(mp.mpf(u), mp.mpf(th1), mp.pi, pts=[mp.mpf(th1) * 1.5, mp.mpf(th1) * 3, mp.pi / 4, mp.pi / 2])
    main = (d - 1) / (k * lam ** 2) * (1 + epsA) / (1 - vr) ** ((d + 1) / 2)
    return main + float(rem) + stats.norm.cdf(u - lam), dict(varrho=vr, epsA=epsA, rem=float(rem), main=main,
                                                             valid=(u < lam * np.cos(th1) ** k) and (rho * u > d - 1))

def lower_i(lam):
    a = d * d; b = -(2 * d * (d - 1) + (d - 1) * k * lam ** 2); c = (d - 1) ** 2
    return (-b - sqrt(b * b - 4 * a * c)) / (2 * a)

rows = [(20, 16), (20, 18), (40, 34), (40, 36)]
res = {}
for lam, u in rows:
    kr = KR(k, d, lam)
    tgt = (d - 1) / (k * lam ** 2)
    # (a) literal Thm estim (ii): optimise over theta1
    best = None
    for th1 in np.concatenate([np.linspace(0.02, 0.6, 59)]):
        if not (u < lam * np.cos(th1) ** k):
            continue
        val, info = upper_ii(lam, u, th1, kr)
        if best is None or val < best[0]:
            best = (val, th1, info)
    # refine
    th_grid = np.linspace(max(best[1] - 0.012, 0.005), best[1] + 0.012, 25)
    for th1 in th_grid:
        if not (u < lam * np.cos(th1) ** k):
            continue
        val, info = upper_ii(lam, u, th1, kr)
        if val < best[0]:
            best = (val, th1, info)
    # the appendix's recommended theta1 = sqrt(6 log lam / k)/lam
    th_app = sqrt(6 * log(lam) / k) / lam
    val_app, info_app = upper_ii(lam, u, th_app, kr)
    # (b) direct Kac-Rice: E sin^2 <= int (1-c^2) d delta^lam(u; dc) [c>=0] + delta(u; c<0) + Phi(u-lam)
    w = lambda th: mp.sin(th) ** 2
    s1 = kr.mass(mp.mpf(u), 0, mp.pi / 2, weight=w, pts=[mp.mpf(t) / lam for t in [0.5, 1, 2, 3, 5, 8]] + [mp.mpf(0.5)])
    s2 = kr.mass(mp.mpf(u), mp.pi / 2, mp.pi)
    direct = float(s1 + s2) + stats.norm.cdf(u - lam)
    # (c) b-integral version with the exact count: (1/lam^2) int_0^{lam^2} min(1, delta(u; B_b^c)) db = int min(...) ; same as (b) when count<1
    lo = lower_i(lam)
    res[(lam, u)] = (lo, tgt, best, val_app, direct)
    print(f"lam={lam} u={u}: lower(i)={lo:.6e} target={tgt:.6e}")
    print(f"   (ii) optimised over theta1: {best[0]:.6e} (ratio {best[0]/tgt:.5f}) at theta1={best[1]:.4f}, varrho={best[2]['varrho']:.4g},"
          f" epsA={best[2]['epsA']:.3g}, Rem_far={best[2]['rem']:.3g}, main={best[2]['main']:.6e}")
    print(f"   (ii) at appendix theta1={th_app:.4f}: {val_app:.6e} (ratio {val_app/tgt:.5f}); Rem_far={info_app['rem']:.3g}, varrho={info_app['varrho']:.4g}")
    print(f"   direct Kac-Rice int sin^2 d delta + Phi: {direct:.6e} (ratio {direct/tgt:.5f});  of which Phi(u-lam)={stats.norm.cdf(u-lam):.4e}"
          f" = {stats.norm.cdf(u-lam)/tgt*100:.3f}% of target; far side c<0: {float(s2):.2e}")
    print(f"   upper/lower: (ii)opt {best[0]/lo:.5f}, direct {direct/lo:.5f}")
