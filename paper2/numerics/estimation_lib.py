"""Theorem 4 (thm:estim) of the main paper and Appendix C of the Supplementary Material: the three columns of the
table of Section 6, "The estimation sandwich", and the explicit bound of Theorem 4(ii) over a range of signals.
Written from the formulas of the paper:

  lower (i):   smallest R with d^2 R^2 - (2 d N + N k lam^2) R + N^2 <= 0      (Theorem 4(i) solved for R)
  Kac-Rice:    int_0^pi sin^2(theta) delta^lam(u; d theta) + Phi(u - lam)
  explicit:    N/(k lam^2) (1+eps_A)/(1-varrho)^{(d+1)/2} + Rem_far + Phi(u - lam)   (Theorem 4(ii))
               with 1 + eps_A = (1 + beta*/(rho u) + eps_det)(1 + r_N) as in Lemma C.2 (lem:Aupper), varrho exact,
               Rem_far = delta^lam(u; theta > theta1).

delta^lam(u; d theta) = pref sin^{N-1} th exp(-k lam^2 c^{2k-2} s^2 / 2) V(th, u) d theta, c = cos th, s = sin th,
pref = omega_{d-2} ((k-1)/pi)^{N/2}, V = int_u^inf A_lam(rho x; c) phi(x - lam c^k) dx.
A_lam = E|det(z - G - beta_2 e1 e1^T)|; "proxy" uses E det = D_N(z) - beta_2 D_{N-1}(z) (Lemma C.1, lem:spikedet),
"mc" estimates E|det| by Monte Carlo over GOE(N) with Var(v'Gv) = 1 (fixed seed).
D_n by the three-term recursion D_n = z D_{n-1} - (n-1)/2 D_{n-2}, D_0 = 1, D_1 = z.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from math import sqrt, pi, gamma, log, exp
from scipy import stats, optimize, integrate

SEED = 777001


class Case:
    def __init__(self, k, d, lam):
        self.k, self.d, self.lam = k, d, float(lam)
        self.N = N = d - 1
        self.rho = sqrt(k / (2 * (k - 1)))
        self.pref = 2 * pi ** ((d - 1) / 2) / gamma((d - 1) / 2) * ((k - 1) / pi) ** (N / 2)
        self.tgt = N / (k * self.lam ** 2)
        qs = max(2 * N, 8)
        self.KN = 2 * sqrt(2) * (1.25 + (sqrt(2 * N) + 5 / 6 * sqrt(qs)) / N) ** N   # K_N of Appendix C, q* = max(2N, 8)
        sc = 1 / (self.lam * sqrt(k))
        h = sc / 80
        n = int(np.ceil(pi / h))
        self.th = np.linspace(0.0, pi, n + 1)
        self.yg, self.yw = np.polynomial.legendre.leggauss(40)

    # ---------- determinant polynomials ----------
    @staticmethod
    def D(n, z):
        a, b = np.ones_like(z), z
        if n == 0:
            return a
        for m in range(2, n + 1):
            a, b = b, z * b - (m - 1) / 2 * a
        return b

    def beta2(self, c):
        k = self.k
        return self.lam * sqrt(k * (k - 1) / 2) * c ** (k - 2) * (1 - c * c)

    def xnodes(self, l0, u):
        """composite Gauss-Legendre nodes in x on [max(u, l0-13), that + 27], 3 panels; arrays (M, P)."""
        a = np.maximum(u, l0 - 13.0)
        panels = []
        W = []
        for j in range(3):
            lo = a + 9.0 * j
            panels.append(lo[:, None] + 4.5 * (self.yg[None, :] + 1))
            W.append(np.full((len(a), len(self.yg)), 4.5) * self.yw[None, :])
        return np.concatenate(panels, 1), np.concatenate(W, 1)

    def V_proxy(self, th, u):
        c = np.cos(th)
        l0 = self.lam * c ** self.k
        X, W = self.xnodes(l0, u)
        z = self.rho * X
        A = self.D(self.N, z) - self.beta2(c)[:, None] * self.D(self.N - 1, z)
        return np.sum(W * A * stats.norm.pdf(X - l0[:, None]), 1)

    def V_mc(self, th, u, nmc=4000):
        rng = np.random.default_rng(SEED + 31 * self.N)
        Aa = rng.standard_normal((nmc, self.N, self.N))
        G = (Aa + np.transpose(Aa, (0, 2, 1))) / 2.0       # diag var 1, off-diag var 1/2
        c = np.cos(th)
        l0 = self.lam * c ** self.k
        X, W = self.xnodes(l0, u)
        b2 = self.beta2(c)
        out = np.empty(len(th))
        for i in range(len(th)):
            Gi = G.copy(); Gi[:, 0, 0] += b2[i]
            mu = np.linalg.eigvalsh(Gi)                           # (nmc, N)
            z = self.rho * X[i]                                   # (P,)
            Aabs = np.mean(np.prod(np.abs(z[:, None, None] - mu[None, :, :]), axis=2), axis=1)
            out[i] = np.sum(W[i] * Aabs * stats.norm.pdf(X[i] - l0[i]))
        return out

    def g(self, th, u, which="proxy"):
        s, c = np.sin(th), np.cos(th)
        pos = s ** (self.N - 1) * np.exp(-self.k * self.lam ** 2 / 2 * c ** (2 * self.k - 2) * s * s)
        V = self.V_proxy(th, u) if which == "proxy" else self.V_mc(th, u)
        return self.pref * pos * V

    # ---------- the three columns ----------
    def lower(self):
        d, N, k, lam = self.d, self.N, self.k, self.lam
        a, b, c = d * d, -(2 * d * N + N * k * lam ** 2), N * N
        return (-b - sqrt(b * b - 4 * a * c)) / (2 * a)

    def kacrice(self, u, which="proxy"):
        th = self.th
        G = self.g(th, u, which) * np.sin(th) ** 2
        return integrate.simpson(G, x=th) + stats.norm.cdf(u - self.lam)

    def remfar_curve(self, u):
        """Rem_far(theta1) = int_{theta1}^pi g, on the grid self.th (cumulative trapezoid from pi)."""
        G = self.g(self.th, u)
        cum = integrate.cumulative_trapezoid(G[::-1], -self.th[::-1], initial=0.0)[::-1]
        return cum

    def remfar_quad(self, th1, u, which="proxy"):
        f = lambda t: float(self.g(np.array([t]), u, which)[0])
        sc = 1 / (self.lam * sqrt(self.k))
        pts = [th1 + j * sc for j in (1, 2, 4, 8, 16) if th1 + j * sc < pi / 2] + [pi / 2]
        tot, edges = 0.0, [th1] + pts + [pi]
        for a, b in zip(edges[:-1], edges[1:]):
            tot += integrate.quad(f, a, b, epsabs=0, epsrel=1e-11, limit=400)[0]
        return tot

    # ---------- explicit right-hand side ----------
    def varrho(self, th1):
        t = np.linspace(1e-7, th1, 4001)
        f = np.cos(t) ** (2 * self.k - 2) * (np.sin(t) / t) ** 2
        return 1 - f.min()

    def hyp(self, th1, u):
        k, lam, N, rho = self.k, self.lam, self.N, self.rho
        s2 = np.sin(th1) ** 2
        return (u > N / rho) & (u < lam * np.cos(th1) ** k) & (u >= 4 * (k - 1) * lam * s2) & \
               (u >= (k - 1) * lam * s2 + sqrt(2 * N) / rho)

    def terms(self, th1, u):
        k, lam, N, rho = self.k, self.lam, self.N, self.rho
        th1 = np.asarray(th1, float)
        bst = rho * (k - 1) * lam * np.sin(th1) ** 2
        l1 = lam * np.cos(th1) ** k
        rN = np.exp(N * N / (2 * l1 * l1)) - 1
        edet = self.KN * np.exp(-0.25 * (rho * u - bst - sqrt(2 * N)) ** 2)
        f1 = np.cos(th1) ** (2 * k - 2) * (np.sin(th1) / th1) ** 2   # = 1 - varrho (f decreasing on (0, pi/2))
        main_fac = (1 + bst / (rho * u) + edet) * (1 + rN) / f1 ** ((self.d + 1) / 2)
        return dict(beta=bst / (rho * u), rN=rN, edet=edet, varrho=1 - f1, main_fac=main_fac)

    def best_theta(self, u):
        """min over theta1 of the explicit RHS at fixed u, theta1 on the fine grid, hypotheses enforced."""
        th = self.th[1:]
        rem = self.remfar_curve(u)[1:]
        ok = self.hyp(th, u) & (th < pi / 2)
        if not ok.any():
            return None
        T = self.terms(th, u)
        val = self.tgt * T["main_fac"] + rem + stats.norm.cdf(u - self.lam)
        val = np.where(ok, val, np.inf)
        i = int(np.argmin(val))
        out = dict(val=float(val[i]), th1=float(th[i]), u=float(u), rem=float(rem[i]), Phi=float(stats.norm.cdf(u - self.lam)))
        out.update({kk: float(vv[i]) for kk, vv in T.items()})
        out["main"] = self.tgt * out["main_fac"]
        return out

    def best_both(self, nu=60):
        lo = self.N / self.rho * (1 + 1e-9)
        us = np.linspace(lo, self.lam * 0.999, nu)
        res = [self.best_theta(u) for u in us]
        vals = [r["val"] if r else np.inf for r in res]
        i = int(np.argmin(vals))
        if not np.isfinite(vals[i]):
            return None
        a, b = us[max(i - 1, 0)], us[min(i + 1, nu - 1)]
        f = lambda u: (self.best_theta(u) or {"val": np.inf})["val"]
        o = optimize.minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(xatol=1e-4 * self.lam))
        r = self.best_theta(o.x)
        return r if r["val"] <= vals[i] else res[i]

    def best_kacrice(self, nu=50):
        us = np.linspace(0.3 * self.lam, self.lam, nu)
        v = [self.kacrice(u) for u in us]
        i = int(np.argmin(v))
        a, b = us[max(i - 1, 0)], us[min(i + 1, nu - 1)]
        o = optimize.minimize_scalar(self.kacrice, bounds=(a, b), method="bounded", options=dict(xatol=1e-4 * self.lam))
        return (float(o.x), float(o.fun)) if o.fun <= v[i] else (float(us[i]), float(v[i]))
