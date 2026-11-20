"""Competing tests and estimators on the same replicates of the field simulation, for Figure 1 (fig:intro) and the
paragraph "The figure of the introduction" of Section 6. Imports spike_lib and mc from ../sim.

On one replicate y ~ lam f(e) + N(0, I_m), e = e_1, sigma = 1 (every test below is scale-free or
uses sigma = 1 as the known value), the functions compute:
  * the t-spacing p-value T (mc.run_batch, unchanged),
  * the known-variance spacing p-value S = G_r(lambda_1)/G_r(lambda_2), r = Omega (Proposition F.1),
  * lambda_1 and lambda_1/sigma_hat (maximum tests, calibrated by Monte Carlo at figure time),
  * the harmonic energies Q_j = |P_j y|^2 and the invariant statistic of Section 5; for k = 3 the
    exact p-value of F = (Q_1/n_1)/(Q_3/n_3) ~ F(n_1, n_3) under the null,
  * the unfolding statistic s_1^2/|Y|_F^2 and the unfolding estimator (first left singular vector),
  * sin^2 of the angle between the argmax and e,
  * the oracle invariant statistic log R_lam(Y), the likelihood ratio of the maximal invariant under
    rotations and scalings (Wijsman), estimated by importance sampling on the sphere.

run_batch_ext is a copy of mc.run_batch (../sim/mc.py) that also returns x1, U,
Tr, the full spectrum of Omega and y. The check branch is not copied.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from math import comb, lgamma, log, sqrt
from scipy import stats, special

REPO_SIM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
if REPO_SIM not in sys.path:
    sys.path.insert(0, REPO_SIM)
from spike_lib import (Model, multinom, ascend_sphere, newton_sphere, random_sphere,  # noqa: E402
                       householder_frame, rotate, Helix, points_to_polar, logTstat, threshold_T2)
from mc import Cfg, maximize_Z, helix_max, polar_to_ambient  # noqa: E402


# ----------------------------------------------------------------------------- field simulation
def run_batch_ext(model, rng, B, lam, cfg):
    """Copy of mc.run_batch without the check branch; returns the extra keys x1, U, Tr, wO, y, fro2."""
    k, d, n = model.k, model.d, model.d - 1
    y = model.draw_y(rng, B, lam)
    Tf = model.tensor(y)
    fro2 = (y ** 2).sum(1)
    etaZ = 1.0 / (k * (k - 1) * (abs(lam) + 3.0))
    X0 = random_sphere(rng, (B, cfg.SZ + 1), d)
    X0[:, 0, :] = 0
    X0[:, 0, 0] = 1.0
    x1, v1, g1, HR, XZ, valZ = maximize_Z(Tf, k, d, X0, cfg.itZ, etaZ)
    n_fix = np.zeros(B, dtype=np.int64)
    for rnd in range(4):
        U = householder_frame(x1)
        Tr = rotate(Tf, U, k, d)
        Hel = Helix(Tr, k, d)
        cosang = (XZ * x1[:, None, :]).sum(-1)
        vm = np.where(cosang < 0.995, valZ, -np.inf)
        oi = np.argsort(-vm, axis=1)[:, :cfg.n_other]
        ar = np.arange(B)[:, None]
        Xo = XZ[ar, oi]
        Ao = np.einsum('bji,bsj->bsi', U, Xo)
        tho, uo, nwo = points_to_polar(Ao)
        th2, u2, lam2h, gnH, wO = helix_max(Hel, rng, B, d, cfg, tho, uo)
        lmaxO = wO[:, -1]
        lam2 = np.maximum(lam2h, lmaxO)
        bad = lam2 > v1 + 1e-9 * (1 + np.abs(v1))
        if not bad.any():
            break
        s2 = polar_to_ambient(U, th2, u2)
        Xs = np.stack([x1, s2], 1)
        Xs, vs, _ = ascend_sphere(Tf, Xs, k, d, cfg.itZ, etaZ)
        xb = Xs[np.arange(B), vs.argmax(1)]
        x1n, v1n, g1n, HRn = newton_sphere(Tf, xb, k, d)
        upd = bad & (v1n > v1)
        x1 = np.where(upd[:, None], x1n, x1)
        v1 = np.where(upd, v1n, v1)
        g1 = np.where(upd, g1n, g1)
        HR = np.where(upd[:, None, None], HRn, HR)
        n_fix += upd
        if not upd.any():
            break
    V = U[:, :, 1:]
    hr_eig = np.linalg.eigvalsh(np.einsum('bia,bij,bjc->bac', V, HR, V))
    s2perp = fro2 - Tr[:, 0] ** 2 - k * (Tr[:, 1:d] ** 2).sum(1)
    sig = np.sqrt(s2perp / model.kappa)
    T1 = v1 / sig
    T2 = lam2 / sig
    r = wO / sig[:, None]
    logT = logTstat(model, T1, T2, r)
    tau, always = threshold_T2(model, T1, r, cfg.alpha)
    ell = sig * tau
    fail_h = logT > np.log(cfg.alpha)
    Td = Tr.copy()
    Td[:, 0] = ell
    Td[:, model.grad_entries] = 0.0
    Xd0 = random_sphere(rng, (B, cfg.SD), d)
    xd, vd, gd, _, _, _ = maximize_Z(Td, k, d, Xd0, cfg.itD, etaZ)
    tol = 1e-9 * (1 + np.abs(ell))
    fail_d = vd > ell + tol
    thd, ud, nwd = points_to_polar(xd[:, None, :])
    Gd = Hel.eval(thd, ud, grad=False)[:, 0]
    Gd = np.where(nwd[:, 0] > 1e-6, Gd, -np.inf)
    lam2c = np.maximum(lam2, Gd)
    logTc = logTstat(model, T1, lam2c / sig, r)
    fail = logTc > np.log(cfg.alpha)
    return dict(lam1=v1, lam2=lam2, lam2h=lam2h, lam2c=lam2c, lmaxO=lmaxO, lminO=wO[:, 0], sig=sig,
                logT=logT, logTc=logTc, tau=tau, always=always, fail_h=fail_h, fail_d=fail_d, fail=fail,
                defl_excess=vd - ell, gradZ=g1, hr_max=hr_eig[:, -1], gradH=gnH, n_fix=n_fix,
                c_e=np.abs(x1[:, 0]), th2=np.abs(th2), smallT2=(T2 < 0.05),
                x1=x1, U=U, Tr=Tr, wO=wO, y=y, fro2=fro2)


# ----------------------------------------------------------------------------- features
class Features:
    """f_a(t) = multinom(k,a)^{1/2} t^a in the coordinate order of Model, and the harmonic projectors."""

    def __init__(self, model):
        self.model = model
        k, d, m = model.k, model.d, model.m
        self.A = np.array(model.alphas, dtype=np.int64)            # (m, d)
        self.sqm = np.array([sqrt(multinom(k, a)) for a in model.alphas])
        # representative flat index of each multi-index in the full tensor
        rep = np.empty(m, dtype=np.int64)
        seen = np.zeros(m, dtype=bool)
        for flat, a in enumerate(model.idx):
            if not seen[a]:
                rep[a] = flat
                seen[a] = True
        self.rep = rep
        # Sigma = int f(t) f(t)^T dt, uniform t on the sphere, from the moments of the sphere
        S = np.empty((m, m))
        for i in range(m):
            for j in range(i, m):
                c = self.A[i] + self.A[j]
                S[i, j] = S[j, i] = self.sqm[i] * self.sqm[j] * sphere_moment(c, d)
        w, V = np.linalg.eigh(S)
        order = np.argsort(-w)
        w, V = w[order], V[:, order]
        # group the eigenvalues: harmonic degrees j <= k, j = k mod 2, dimension n_j
        self.degrees, self.mu, self.nj, self.V = [], [], [], []
        pos = 0
        for j in range(k % 2, k + 1, 2):
            n_j = comb(d + j - 1, j) - (comb(d + j - 3, j - 2) if j >= 2 else 0)
            self.degrees.append(j)
            self.nj.append(n_j)
            block = w[pos:pos + n_j]
            self.mu.append(float(block.mean()))
            assert block.max() - block.min() < 1e-9 * max(1.0, block.max()), (j, block)
            self.V.append(V[:, pos:pos + n_j].copy())
            pos += n_j
        assert pos == m, (pos, m)
        self.mu = np.array(self.mu)                                # increasing degree

    def eval(self, t):
        """f(t) for t (n, d): (n, m)."""
        F = np.ones((t.shape[0], self.model.m))
        for i in range(self.model.d):
            ai = self.A[:, i]
            nz = ai > 0
            F[:, nz] *= t[:, i:i + 1] ** ai[nz][None, :]
        return F * self.sqm[None, :]

    def y_rot(self, Tr):
        """coordinates of the rotated tensor: Z(U w) = <y_rot, f(w)>."""
        return Tr[:, self.rep] * self.sqm[None, :]

    def energies(self, y):
        """Q_j = |P_j y|^2 for each harmonic degree, (B, number of degrees)."""
        return np.stack([((y @ V) ** 2).sum(1) for V in self.V], 1)


def sphere_moment(c, d):
    """E prod t_i^{c_i} for t uniform on S^{d-1}: prod (c_i-1)!! / prod_{j<|c|/2} (d+2j), zero if some c_i odd."""
    c = np.asarray(c)
    if (c % 2).any():
        return 0.0
    num = 1.0
    for ci in c:
        for q in range(1, int(ci), 2):
            num *= q
    den = 1.0
    for j in range(int(c.sum()) // 2):
        den *= d + 2 * j
    return num / den


def mu_funk_hecke(k, d, j):
    """Eigenvalue mu_j of the kernel <e,e'>^k on the harmonics of degree j (Funk-Hecke)."""
    lam = (d - 2) / 2
    from scipy.special import gegenbauer, gamma
    C = gegenbauer(j, lam)
    cnorm = C(1.0)
    const = gamma(d / 2) / (sqrt(np.pi) * gamma((d - 1) / 2))
    from scipy import integrate
    f = lambda t: t ** k * C(t) / cnorm * const * (1 - t * t) ** ((d - 3) / 2)
    return integrate.quad(f, -1, 1, epsabs=0, epsrel=1e-12, limit=200)[0]


# ----------------------------------------------------------------------------- closed forms
def logG_closed(l, eig):
    """log G_r(l), G_r(l) = sum_j (-1)^j e_j(r) I_{N-j}(l) (Proposition F.1, known variance), l (B,), eig (B,N)."""
    B, N = eig.shape
    e = np.zeros((B, N + 1))
    e[:, 0] = 1.0
    for i in range(N):
        e[:, 1:] = e[:, 1:] + eig[:, i:i + 1] * e[:, :-1]
    phi = stats.norm.pdf(l)
    sf = stats.norm.sf(l)
    I = [sf, phi]
    for p in range(2, N + 1):
        I.append(l ** (p - 1) * phi + (p - 1) * I[p - 2])
    G = np.zeros(B)
    for j in range(N + 1):
        G += (-1) ** j * e[:, j] * I[N - j]
    return np.log(G)


def unfolding(model, y):
    """s_1^2/|Y|_F^2 and the first left singular vector of the d x d^{k-1} unfolding."""
    k, d = model.k, model.d
    Mat = model.tensor(y).reshape(y.shape[0], d, d ** (k - 1))
    u, s, _ = np.linalg.svd(Mat, full_matrices=False)
    fro2 = (y ** 2).sum(1)
    return s[:, 0] ** 2 / fro2, u[:, :, 0]


# ----------------------------------------------------------------------------- oracle
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(20)
_GH_LW = np.log(_GH_W) + _GH_X ** 2


def log_I(c, m):
    """log int_0^inf r^{m-1} exp(-r^2/2 + c r) dr, Gauss-Hermite around the mode (c any shape)."""
    c = np.asarray(c, dtype=float)
    rs = 0.5 * (c + np.sqrt(c * c + 4.0 * (m - 1)))
    f2 = (m - 1) / rs ** 2 + 1.0
    s = np.sqrt(2.0 / f2)
    out = np.full(c.shape, -np.inf)
    for x, lw in zip(_GH_X, _GH_LW):
        r = rs + s * x
        ok = r > 0
        f = np.where(ok, (m - 1) * np.log(np.where(ok, r, 1.0)) - 0.5 * r * r + c * r, -np.inf)
        out = np.logaddexp(out, lw + f)
    return out + np.log(s)


def log_M(kappa, d):
    """log E exp(kappa <e', pole>), e' uniform on S^{d-1}: Gamma(d/2) (2/kappa)^{d/2-1} I_{d/2-1}(kappa)."""
    nu = d / 2 - 1
    if kappa < 1e-12:
        return 0.0
    return lgamma(d / 2) + nu * log(2.0 / kappa) + log(special.ive(nu, kappa)) + kappa


def sample_vmf_w(rng, kappa, d, n):
    """w = <e', pole> under vMF(kappa) on S^{d-1}, Wood (1994), vectorised rejection."""
    if kappa < 1e-12:
        return 2 * rng.beta((d - 1) / 2, (d - 1) / 2, n) - 1
    b = (-2 * kappa + sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    cc = kappa * x0 + (d - 1) * log(1 - x0 * x0)
    out = np.empty(n)
    filled = 0
    while filled < n:
        nn = int(1.5 * (n - filled)) + 16
        z = rng.beta((d - 1) / 2, (d - 1) / 2, nn)
        u = rng.uniform(size=nn)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        acc = kappa * w + (d - 1) * np.log(1 - x0 * w) - cc >= np.log(u)
        w = w[acc][:n - filled]
        out[filled:filled + w.size] = w
        filled += w.size
    return out


def make_cloud(rng, d, kappas, sizes):
    """Points on S^{d-1} in the pole frame from the mixture of vMF(kappa) around the pole with weights
    proportional to `sizes` (deterministic mixture), kappa = 0 meaning uniform; returns (points (n, d),
    log q(point) w.r.t. the uniform probability)."""
    pts, ws = [], []
    for kap, n in zip(kappas, sizes):
        w = sample_vmf_w(rng, kap, d, n)
        u = random_sphere(rng, (n,), d - 1)
        pts.append(np.concatenate([w[:, None], np.sqrt(np.clip(1 - w * w, 0, 1))[:, None] * u], 1))
        ws.append(w)
    P = np.concatenate(pts, 0)
    w = np.concatenate(ws, 0)
    ntot = float(sum(sizes))
    comp = np.full(P.shape[0], -np.inf)
    for kap, n in zip(kappas, sizes):
        comp = np.logaddexp(comp, log(n / ntot) + kap * w - log_M(kap, d))
    return P, comp


def oracle_kappas(lam, k, m):
    """Two concentrations for the proposal at signal lam: a null-like peak (z = 0.3) and a signal-like one."""
    out = []
    for z in (0.3, lam / sqrt(lam * lam + m)):
        c = lam * z
        rs = 0.5 * (c + sqrt(c * c + 4.0 * (m - 1)))
        out.append(max(k * lam * rs * z, 1.0))
    return out


def oracle_logR(feat, y_rot, fro2, lam, cloud, logq, block=200):
    """log R_lam(Y) = log int de' I(lam z(e'))/I(0) - lam^2/2, z = Z(e')/|Y|, by importance sampling on
    `cloud` (points in the frame of the argmax, log density logq w.r.t. the uniform probability)."""
    m = feat.model.m
    F = feat.eval(cloud)                                   # (n, m)
    n = cloud.shape[0]
    out = np.empty(y_rot.shape[0])
    lI0 = log_I(0.0, m)
    for s in range(0, y_rot.shape[0], block):
        yb = y_rot[s:s + block]
        Z = F @ yb.T                                       # (n, b)
        z = Z / np.sqrt(fro2[s:s + block])[None, :]
        lh = log_I(lam * z, m) - lI0 - 0.5 * lam * lam - logq[:, None]
        out[s:s + block] = special.logsumexp(lh, axis=0) - log(n)
    return out


def lower_bound_risk(lam, d, k):
    """Smallest R with R >= ((d-1) - d R)^2 / ((d-1) k lam^2) (Theorem 4(i)): the smaller root."""
    a = d * d
    b = -(2 * d * (d - 1) + (d - 1) * k * lam ** 2)
    c = (d - 1) ** 2
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
