"""Shared definitions for the constant A and the exponent E of Theorem 2 (thm:exponent) and Remark 1 (rem:bracket).

Notation of the paper (Sections 2 and 3 of the main text):
  N = d - 1, m = binom(d+k-1, k), kappa = m - d, nu = m - 1, rho^2 = k / (2(k-1)), T_max = sqrt(kappa)/rho.
  H_r(l) = int_l^inf det(u Id - r) f_nu(u sqrt(nu/kappa)) du.
  Up to a factor that does not depend on (r, l), f_nu(u sqrt(nu/kappa)) = (1 + u^2/kappa)^(-m/2), so
      H_r(l) = int_l^inf prod_i (u - r_i) (1 + u^2/kappa)^(-m/2) du.
  u_alpha(r, v) is the solution u > v of H_r(u) = alpha H_r(v), for v >= lmax(r).
  A = max { u_alpha(r, lmax(r)) : r psd of size N, ||r||_F = T_max } / T_max   (eq:Adef).
  E = rho^2 / (2 (A^2 + rho^2)).

Two independent evaluations of H_r in mpmath:
  * mp_logH:      closed form of Appendix F (app:closed) of the Supplementary Material (sum of incomplete Beta
                  functions), any precision;
  * mp_logH_beta: the Beta form J_r(l) = int_0^1 x^(kappa-1) prod(1 - r_i x/l) (1 + kappa x^2/l^2)^(-m/2) dx, with
                  H_r(l) = kappa^(m/2) l^(-kappa) J_r(l), by tanh-sinh quadrature.
dp_logJ is the same Beta form by Gauss-Legendre in double precision, vectorised; it is used only for scans and
searches, never for a printed digit that the mpmath code does not confirm.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import math
from math import comb
import numpy as np
import mpmath as mp

PAIRS = [(3, 3), (3, 4), (3, 5), (5, 3), (5, 5), (7, 5)]


def params(k, d):
    m = comb(d + k - 1, k)
    N = d - 1
    kappa = m - d
    rho2 = k / (2.0 * (k - 1))
    return dict(k=k, d=d, m=m, N=N, kappa=kappa, nu=m - 1, rho=math.sqrt(rho2), rho2=rho2,
                Tmax=math.sqrt(kappa / rho2))


def mp_params(k, d):
    P = params(k, d)
    rho2 = mp.mpf(k) / (2 * (k - 1))
    return dict(P, rho2_mp=rho2, Tmax_mp=mp.sqrt(mp.mpf(P["kappa"]) / rho2))


def _esym(r):
    """elementary symmetric polynomials e_0..e_N of the list r."""
    e = [mp.mpf(1)] + [mp.mpf(0)] * len(r)
    for x in r:
        for j in range(len(r), 0, -1):
            e[j] += e[j - 1] * x
    return e


def mp_logH(r, ell, P):
    """log H_r(ell), closed form: prod(u - r_i) = sum_j (-1)^j e_j u^(N-j) and
    M_p(ell) = int_ell^inf u^p (1 + u^2/kappa)^(-m/2) du = (kappa^((p+1)/2)/2) B(w; (m-p-1)/2, (p+1)/2),
    w = kappa/(kappa + ell^2), ell >= 0. Needs the working precision to absorb the cancellation (dps >= 40)."""
    kap, m = mp.mpf(P["kappa"]), P["m"]
    N = len(r)
    e = _esym(r)
    w = kap / (kap + ell * ell)
    tot = mp.mpf(0)
    for j in range(N + 1):
        p = N - j
        Mp = kap ** (mp.mpf(p + 1) / 2) / 2 * mp.betainc(mp.mpf(m - p - 1) / 2, mp.mpf(p + 1) / 2, 0, w)
        tot += (-1) ** j * e[j] * Mp
    if tot <= 0:
        raise ValueError("H_r(l) <= 0: precision too low or l < lmax(r)")
    return mp.log(tot)


def mp_logH_beta(r, ell, P):
    """log H_r(ell) through the Beta form, H = kappa^(m/2) ell^(-kappa) J_r(ell), J by tanh-sinh quadrature on
    [0,1] with 30 extra digits (independent check of mp_logH)."""
    kap, m = mp.mpf(P["kappa"]), P["m"]
    with mp.extradps(30):
        ell_ = mp.mpf(ell)
        rr = [mp.mpf(x) for x in r]
        f = lambda x: x ** (kap - 1) * mp.fprod([1 - ri * x / ell_ for ri in rr]) * (1 + kap * x * x / (ell_ * ell_)) ** (-mp.mpf(m) / 2)
        pts = [mp.mpf(0)] + [1 - c / kap for c in (40, 20, 5, 1, mp.mpf(1) / 4) if c < kap] + [mp.mpf(1)]
        J = mp.quad(f, sorted(set(pts)))
        val = mp.log(J) + mp.mpf(m) / 2 * mp.log(kap) - kap * mp.log(ell_)
    return +val


def mp_u_alpha(r, v, alpha, P, logH=mp_logH):
    """solution u > v of H_r(u) = alpha H_r(v), mpmath; requires max(r) <= v."""
    la = mp.log(alpha)
    h2 = logH(r, v, P)
    f = lambda u: logH(r, u, P) - h2 - la
    hi = v * mp.mpf("1.5")
    while f(hi) > 0:
        hi = v + 2 * (hi - v)
    lo = v
    for _ in range(30):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return mp.findroot(f, (lo, hi), solver="anderson", tol=mp.mpf(10) ** (-(mp.mp.dps - 5)), maxsteps=200)


def mp_U(spec, alpha, P, logH=mp_logH):
    """u_alpha(r, lmax(r)) / T_max for the spectrum r = T_max * spec (spec given in units of T_max)."""
    T = P["Tmax_mp"]
    r = [T * mp.mpf(s) for s in spec]
    v = max(r)
    return mp_u_alpha(r, v, alpha, P, logH) / T


def mp_golden_max(F, lo, hi, tol):
    """golden-section search for the maximum of a unimodal F on [lo, hi]; returns (x, F(x))."""
    g = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(lo), mp.mpf(hi)
    c, d_ = b - g * (b - a), a + g * (b - a)
    fc, fd = F(c), F(d_)
    while b - a > tol:
        if fc > fd:
            b, d_, fd = d_, c, fc
            c = b - g * (b - a)
            fc = F(c)
        else:
            a, c, fc = c, d_, fd
            d_ = a + g * (b - a)
            fd = F(d_)
    return (c, fc) if fc > fd else (d_, fd)


_GL = {}


def _gl(n):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL[n]


def dp_logJ(R, ell, P, n=600):
    """log J_r(ell) (Beta form) for the rows of R (n_rep x N) at ell (n_rep,); needs max(r) <= ell."""
    kap, m = P["kappa"], P["m"]
    x, w = _gl(n)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    ell = np.broadcast_to(np.asarray(ell, dtype=float), (R.shape[0],))
    lg = (kap - 1) * np.log(x)[None, :] - 0.5 * m * np.log1p(kap * x[None, :] ** 2 / ell[:, None] ** 2)
    for i in range(R.shape[1]):
        lg = lg + np.log(np.maximum(1.0 - R[:, i:i + 1] * x[None, :] / ell[:, None], 1e-300))
    mx = lg.max(axis=1, keepdims=True)
    return np.log((np.exp(lg - mx) * w[None, :]).sum(axis=1)) + mx[:, 0]


def dp_U(S, alpha, P, n=600, iters=70):
    """u_alpha(r, lmax(r))/T_max for each row of S (spectra in units of T_max, entries <= row max), vectorised
    bisection on theta = u/lmax in [1, theta_hi]."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    T = P["Tmax"]
    R = S * T
    v = R.max(axis=1)
    base = dp_logJ(R, v, P, n)
    la = math.log(alpha)
    f = lambda th: -P["kappa"] * np.log(th) + dp_logJ(R, v * th, P, n) - base - la
    lo = np.ones(len(S))
    hi = np.full(len(S), 2.0)
    for _ in range(60):
        bad = f(hi) > 0
        if not bad.any():
            break
        hi = np.where(bad, hi * 2, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi) * v / T
