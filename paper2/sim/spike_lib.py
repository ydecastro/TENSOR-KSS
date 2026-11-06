"""
Direct Monte Carlo of the t-spacing test in the rank-one spiked tensor model
Y = lam * e^{(x)k} + W   (sigma = 1),  Z(t) = <Y, t^{(x)k}> = <y, f(t)>,
f_alpha(t) = binom(k,alpha)^{1/2} t^alpha  (orthonormal feature system, m = binom(d+k-1,k)).

Conventions (checked in validate.py):
  * y ~ lam f(e) + N(0, I_m);  full symmetric tensor T[i1..ik] = y_alpha / binom(k,alpha)^{1/2};
    Z(t) = T[t,...,t];  ||y||^2 = ||T||_F^2.
  * lambda1 = max_{S^{d-1}} Z at that.
  * Rotated tensor T' = T x_1 U ... x_k U with U = [that, V] orthonormal (index 0 = normal direction).
  * Regression residual (helix numerator) q = <P_perp y, f(.)>: in rotated coordinates it is T' with the
    entries of type (0^k) and (0^{k-1}, i) set to zero, i.e. value AND gradient at that removed
    (this is the regression on (Z(t), grad Z(t)) of Azais-Dalmao-De Castro, eq. (e:Xbarra)).
  * Helix, in polar coordinates s = cos(th) that + sin(th) V u, u in S^{d-2}:
        G(th,u) = h(c) * sum_{j=2..k} C(k,j) c^{k-j} s^{j-2} Q_j(u),
        Q_j(u) = T'[0^{k-j}, u^j],  c = cos th, s = sin th,  h(c) = s^2/(1-c^k) = (1+c)/sum_{i<k} c^i.
    G is smooth on R x S^{d-2}, G(0,u) = u' Omega u (the fibre, a removable singularity), and
    G(th,u) = (Z(s) - <s,that>^k Z(that) - grad-term)/(1 - <s,that>^k) for th != 0.
  * Omega = (k-1) T'[0^{k-2}, i, j]  (= P D^2Z P / k on the tangent space = (Hess_R Z + k Z Id)/k).
  * sigma_hat^2 = ||P_perp y||^2 / kappa = (||T||_F^2 - T'[0^k]^2 - k sum_i T'[0^{k-1} i]^2) / kappa,
    kappa = m - d.
  * T = H_r(T1)/H_r(T2), r = Omega/sigma_hat, Beta form: T = theta^{-kappa} J_r(T1)/J_r(T2),
    J_r(l) = int_0^1 x^{kappa-1} prod_i (1 - r_i x / l) (1 + kappa x^2/l^2)^{-(nu+1)/2} dx, nu = m-1.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from math import comb, factorial
from itertools import combinations_with_replacement


def multinom(k, alpha):
    r = factorial(k)
    for a in alpha:
        r //= factorial(a)
    return r


class Model:
    def __init__(self, k, d, n_gl=200):
        self.k, self.d = k, d
        alphas = []
        for c in combinations_with_replacement(range(d), k):
            a = [0] * d
            for i in c:
                a[i] += 1
            alphas.append(tuple(a))
        self.alphas = alphas
        self.m = len(alphas)
        self.kappa = self.m - d
        self.N = d - 1
        self.nu = self.m - 1
        aidx = {a: i for i, a in enumerate(alphas)}
        idx = np.empty(d ** k, dtype=np.int64)
        scale = np.empty(d ** k)
        for flat in range(d ** k):
            digits = np.unravel_index(flat, (d,) * k)
            a = [0] * d
            for i in digits:
                a[int(i)] += 1
            a = tuple(a)
            idx[flat] = aidx[a]
            scale[flat] = 1.0 / np.sqrt(multinom(k, a))
        self.idx, self.scale = idx, scale
        e = [0] * d
        e[0] = k
        self.e_index = aidx[tuple(e)]
        # Gauss-Legendre on [0,1]
        xg, wg = np.polynomial.legendre.leggauss(n_gl)
        self.xg = 0.5 * (xg + 1.0)
        self.lwg = np.log(0.5 * wg)
        self.lxg = np.log(self.xg)
        # entries of type (0^{k-1}, i), i>=1 in the rotated tensor (all positions)
        full = np.arange(d ** k).reshape((d,) * k)
        grad_entries = []
        for pos in range(k):
            for i in range(1, d):
                ix = [0] * k
                ix[pos] = i
                grad_entries.append(full[tuple(ix)])
        self.grad_entries = np.array(sorted(set(grad_entries)))

    # ---------------------------------------------------------------- data
    def draw_y(self, rng, B, lam):
        y = rng.standard_normal((B, self.m))
        y[:, self.e_index] += lam
        return y

    def tensor(self, y):
        return y[:, self.idx] * self.scale  # (B, d^k)


# -------------------------------------------------------------------- polynomial evaluation
def eval_sym(Tf, X, k, n, hess=False):
    """Symmetric k-tensor Tf (B, n^k) at points X (B,S,n).
    Returns value (B,S), Euclidean gradient (B,S,n), optionally Euclidean Hessian (B,S,n,n)."""
    B, S, _ = X.shape
    Xt = np.ascontiguousarray(X.transpose(0, 2, 1))  # (B,n,S)
    W = np.matmul(Tf.reshape(B, n ** (k - 1), n), Xt)  # (B, n^{k-1}, S)
    rem = k - 1
    H = G = None
    if k == 2 and hess:
        raise NotImplementedError
    while rem > 0:
        if rem == 2 and hess:
            H = W.reshape(B, n, n, S)
        if rem == 1:
            G = W
        W4 = W.reshape(B, n ** (rem - 1), n, S)
        W = np.einsum('bmis,bis->bms', W4, Xt)
        rem -= 1
    val = W[:, 0, :]
    grad = k * G.transpose(0, 2, 1)
    if hess:
        return val, grad, (k * (k - 1)) * H.transpose(0, 3, 1, 2)
    return val, grad


def normalize(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def ascend_sphere(Tf, X, k, n, iters, eta0):
    """Multi-start Riemannian gradient ascent with per-start adaptive step (monotone)."""
    X = normalize(X)
    val, grad = eval_sym(Tf, X, k, n)
    eta = np.full(val.shape, eta0)
    for _ in range(iters):
        rg = grad - (grad * X).sum(-1, keepdims=True) * X
        Xn = normalize(X + eta[..., None] * rg)
        vn, gn = eval_sym(Tf, Xn, k, n)
        acc = vn >= val
        X = np.where(acc[..., None], Xn, X)
        val = np.where(acc, vn, val)
        grad = np.where(acc[..., None], gn, grad)
        eta = np.where(acc, eta * 1.3, eta * 0.5)
    rg = grad - (grad * X).sum(-1, keepdims=True) * X
    return X, val, np.linalg.norm(rg, axis=-1)


def newton_sphere(Tf, x, k, n, steps=8):
    """Riemannian Newton polish of single points x (B,n), accepted only if the value does not drop."""
    B = x.shape[0]
    I = np.eye(n)[None]
    for _ in range(steps):
        v, g, H = eval_sym(Tf, x[:, None, :], k, n, hess=True)
        v, g, H = v[:, 0], g[:, 0], H[:, 0]
        P = I - x[:, :, None] * x[:, None, :]
        rg = np.einsum('bij,bj->bi', P, g)
        HR = P @ H @ P - (x * g).sum(-1)[:, None, None] * P
        A = HR + x[:, :, None] * x[:, None, :]
        try:
            xi = np.linalg.solve(A, -rg[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        xn = normalize(x + xi)
        vn, _ = eval_sym(Tf, xn[:, None, :], k, n)
        acc = vn[:, 0] >= v - 1e-12 * (1 + np.abs(v))
        x = np.where(acc[:, None], xn, x)
    v, g, H = eval_sym(Tf, x[:, None, :], k, n, hess=True)
    v, g, H = v[:, 0], g[:, 0], H[:, 0]
    P = I - x[:, :, None] * x[:, None, :]
    rg = np.einsum('bij,bj->bi', P, g)
    HR = P @ H @ P - (x * g).sum(-1)[:, None, None] * P
    return x, v, np.linalg.norm(rg, axis=-1), HR


def random_sphere(rng, shape, n):
    return normalize(rng.standard_normal(shape + (n,)))


def householder_frame(t):
    """U (B,n,n) orthonormal with U[:,:,0] = t."""
    B, n = t.shape
    s = np.where(t[:, 0] >= 0, 1.0, -1.0)
    v = t.copy()
    v[:, 0] += s
    Hh = np.eye(n)[None] - 2 * v[:, :, None] * v[:, None, :] / (v * v).sum(-1)[:, None, None]
    # Hh e0 = -s t
    U = Hh.copy()
    U[:, :, 0] = t
    return U


def rotate(Tf, U, k, n):
    """T'[a1..ak] = sum T[i1..ik] U[i1,a1]...U[ik,ak]."""
    B = Tf.shape[0]
    W = Tf.reshape(B, n, n ** (k - 1))
    Ut = U.transpose(0, 2, 1)
    for _ in range(k):
        W = np.matmul(Ut, W)
        W = np.ascontiguousarray(W.transpose(0, 2, 1)).reshape(B, n, n ** (k - 1))
    return W.reshape(B, n ** k)


# -------------------------------------------------------------------- helix
class Helix:
    """Helix G(th,u) of a rotated tensor Tr (B, d^k) (normal direction = index 0)."""

    def __init__(self, Tr, k, d):
        self.k, self.d, self.n = k, d, d - 1
        B = Tr.shape[0]
        F = Tr.reshape((B,) + (d,) * k)
        self.Q = {}
        for j in range(2, k + 1):
            sl = (slice(None),) + (0,) * (k - j) + (slice(1, None),) * j
            self.Q[j] = np.ascontiguousarray(F[sl]).reshape(B, self.n ** j)
        self.Omega = (k - 1) * self.Q[2].reshape(B, self.n, self.n) if k >= 2 else None
        self.C = {j: comb(k, j) for j in range(k + 1)}

    def eval(self, th, U, grad=True):
        k, n = self.k, self.n
        c, s = np.cos(th), np.sin(th)
        Sc = np.zeros_like(c)
        Sp = np.zeros_like(c)
        cp = np.ones_like(c)
        for i in range(k):
            Sc += cp
            if i + 1 < k:
                Sp += (i + 1) * cp
            cp = cp * c
        h = (1 + c) / Sc
        hp = -s * (Sc - (1 + c) * Sp) / Sc ** 2
        F = np.zeros_like(c)
        Fp = np.zeros_like(c)
        Fu = np.zeros(U.shape)
        for j in range(2, k + 1):
            if j == 2:
                qv, qg = self._eval_q2(U)
            else:
                qv, qg = eval_sym(self.Q[j], U, j, n)
            a, b = k - j, j - 2
            A = self.C[j] * c ** a * s ** b
            Ap = 0.0
            if a > 0:
                Ap = Ap - a * c ** (a - 1) * s ** (b + 1)
            if b > 0:
                Ap = Ap + b * c ** (a + 1) * s ** (b - 1)
            Ap = self.C[j] * Ap
            F += A * qv
            Fp += Ap * qv
            Fu += A[..., None] * qg
        G = h * F
        if not grad:
            return G
        dth = hp * F + h * Fp
        gu = h[..., None] * Fu
        gu = gu - (gu * U).sum(-1, keepdims=True) * U
        return G, dth, gu

    def _eval_q2(self, U):
        B = U.shape[0]
        M = self.Q[2].reshape(B, self.n, self.n)
        MU = np.einsum('bij,bsj->bsi', M, U)
        return (MU * U).sum(-1), 2 * MU

    def ascend(self, th, U, iters, eta0):
        G, dth, gu = self.eval(th, U)
        eta = np.full(G.shape, eta0)
        for _ in range(iters):
            thn = th + eta * dth
            Un = normalize(U + eta[..., None] * gu)
            Gn, dn, gn = self.eval(thn, Un)
            acc = Gn >= G
            th = np.where(acc, thn, th)
            U = np.where(acc[..., None], Un, U)
            G = np.where(acc, Gn, G)
            dth = np.where(acc, dn, dth)
            gu = np.where(acc[..., None], gn, gu)
            eta = np.where(acc, eta * 1.3, eta * 0.5)
        gnorm = np.sqrt(dth ** 2 + (gu ** 2).sum(-1))
        return th, U, G, gnorm


def points_to_polar(A):
    """Rotated-coordinate points A (...,d) on the sphere -> (th, u) with th in [0,pi]."""
    a0 = np.clip(A[..., 0], -1, 1)
    w = A[..., 1:]
    nw = np.linalg.norm(w, axis=-1)
    th = np.arctan2(nw, a0)
    u = w / np.maximum(nw, 1e-300)[..., None]
    return th, u, nw


# -------------------------------------------------------------------- statistic
def logJ(model, ell, r):
    """log J_r(ell) for ell (B,) > 0, r (B,N) eigenvalues of Omega/sigma_hat."""
    x = model.xg[None, :]
    kap, nu = model.kappa, model.nu
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (kap - 1) * model.lxg[None, :] - 0.5 * (nu + 1) * np.log1p(kap * x ** 2 / ell[:, None] ** 2)
        for i in range(r.shape[1]):
            t = t + np.log1p(-r[:, i:i + 1] * x / ell[:, None])
        t = t + model.lwg[None, :]
        mx = np.max(t, axis=1, keepdims=True)
        return mx[:, 0] + np.log(np.exp(t - mx).sum(1))


def logTstat(model, T1, T2, r):
    return -model.kappa * np.log(T1 / T2) + logJ(model, T1, r) - logJ(model, T2, r)


def threshold_T2(model, T1, r, alpha, iters=70):
    """tau with Tstat(T1, tau, r) = alpha; failure (Tstat > alpha) iff T2 > tau.
    Returns tau and a flag 'always' (Tstat > alpha for every admissible T2 >= max(lmax(r),0+))."""
    la = np.log(alpha)
    # Beta-form Gauss-Legendre is accurate (<1e-13) for ell >= 0.05 (checked against quadrature);
    # T2 < 0.05 is flagged separately by the caller.
    lo = np.maximum(r.max(1) * (1 + 1e-13), 0.05)
    hi = T1.copy()
    flo = logTstat(model, T1, lo, r) - la
    always = flo >= 0
    a, b = lo.copy(), hi.copy()
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = logTstat(model, T1, mid, r) - la
        pos = fm >= 0
        b = np.where(pos, mid, b)
        a = np.where(pos, a, mid)
    tau = np.where(always, lo, 0.5 * (a + b))
    return tau, always
