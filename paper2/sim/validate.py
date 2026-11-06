"""Independent checks of every formula used in spike_lib.py.
Run: python3 validate.py  (prints max discrepancies)."""
import numpy as np
from math import comb, lgamma, sqrt, pi
from scipy import integrate, stats, special
from spike_lib import (Model, eval_sym, normalize, householder_frame, rotate, Helix, logJ, logTstat,
                       multinom, ascend_sphere, newton_sphere, random_sphere, threshold_T2)


def features(model, t):
    """f_alpha(t) = binom(k,alpha)^{1/2} t^alpha, t (...,d) -> (...,m)."""
    out = []
    for a in model.alphas:
        v = np.sqrt(multinom(model.k, a)) * np.prod([t[..., i] ** a[i] for i in range(model.d)], axis=0)
        out.append(v)
    return np.stack(out, -1)


def dfeatures(model, t, h):
    """directional derivative D f(t)[h]."""
    eps = 1e-6
    return (features(model, t + eps * h) - features(model, t - eps * h)) / (2 * eps)


def H_quad(model, r, ell):
    nu, kap = model.nu, model.kappa
    f = lambda u: np.prod(u - r) * stats.t.pdf(u * np.sqrt(nu / kap), nu)
    v, err = integrate.quad(f, ell, np.inf, epsabs=0, epsrel=1e-13, limit=500)
    return v


def H_closed(model, r, ell):
    """Prop. closed."""
    nu, kap, N = model.nu, model.kappa, model.N
    gam = np.sqrt(kap / nu)
    cnu = np.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / np.sqrt(nu * pi)
    # elementary symmetric functions
    e = np.poly(r)  # coefficients of prod(u - r_i) = sum (-1)^j e_j u^{N-j}
    a = ell / gam
    z = nu / (nu + a * a)
    tot = 0.0
    for j in range(N + 1):
        p = N - j
        K = cnu * nu ** ((p + 1) / 2) / 2 * special.beta((nu - p) / 2, (p + 1) / 2) * \
            special.betainc((nu - p) / 2, (p + 1) / 2, z)
        tot += e[j] * gam ** p * K   # e[j] already carries (-1)^j e_j
    return gam * tot


def main():
    rng = np.random.default_rng(1)
    for (k, d) in [(3, 3), (3, 4), (3, 5), (5, 3)]:
        M = Model(k, d)
        print(f"==== (k,d)=({k},{d}) m={M.m} kappa={M.kappa} N={M.N}")
        B = 7
        y = M.draw_y(rng, B, 3.0)
        Tf = M.tensor(y)
        # 1. feature consistency
        t = random_sphere(rng, (B, 5), d)
        vt, gt = eval_sym(Tf, t, k, d)
        vf = np.einsum('bm,bsm->bs', y, features(M, t))
        print(" Z(tensor) vs <y,f>:", np.abs(vt - vf).max(),
              " | ||y||-||T||_F:", np.abs(np.linalg.norm(y, axis=1) - np.linalg.norm(Tf, axis=1)).max())
        # kernel identity
        s = random_sphere(rng, (B, 5), d)
        ker = np.einsum('bsm,bsm->bs', features(M, s), features(M, t))
        print(" sum f(s)f(t) - <s,t>^k:", np.abs(ker - (s * t).sum(-1) ** k).max())
        # gradient / hessian finite differences
        eps = 1e-5
        gfd = np.stack([(eval_sym(Tf, t + eps * np.eye(d)[i], k, d)[0] - eval_sym(Tf, t - eps * np.eye(d)[i], k, d)[0])
                        / (2 * eps) for i in range(d)], -1)
        _, _, Hs = eval_sym(Tf, t, k, d, hess=True)
        Hfd = np.stack([(eval_sym(Tf, t + eps * np.eye(d)[i], k, d)[1] - eval_sym(Tf, t - eps * np.eye(d)[i], k, d)[1])
                        / (2 * eps) for i in range(d)], -1)
        print(" grad FD:", np.abs(gfd - gt).max(), " hess FD:", np.abs(Hfd - Hs).max())
        # 2. find a local max (so that t is critical), rotation
        X0 = random_sphere(rng, (B, 30), d)
        X, val, _ = ascend_sphere(Tf, X0, k, d, 200, 0.02)
        xb = X[np.arange(B), val.argmax(1)]
        x1, v1, gn, HR = newton_sphere(Tf, xb, k, d)
        print(" critical grad norm:", gn.max())
        U = householder_frame(x1)
        print(" frame orthonormality:", np.abs(np.einsum('bij,bik->bjk', U, U) - np.eye(d)).max(),
              " U e0 - t:", np.abs(U[:, :, 0] - x1).max())
        Tr = rotate(Tf, U, k, d)
        a = random_sphere(rng, (B, 5), d)
        v_rot = eval_sym(Tr, a, k, d)[0]
        v_dir = eval_sym(Tf, np.einsum('bij,bsj->bsi', U, a), k, d)[0]
        print(" rotation T'(a) - T(Ua):", np.abs(v_rot - v_dir).max(), " T'[0^k]-lambda1:",
              np.abs(Tr[:, 0] - v1).max())
        # 3. Omega vs (Hess_R + k Z)/k and P D2Z P / k
        Hel = Helix(Tr, k, d)
        V = U[:, :, 1:]
        Om_R = np.einsum('bia,bij,bjc->bac', V, HR, V) / k + v1[:, None, None] * np.eye(d - 1)
        _, _, He = eval_sym(Tf, x1[:, None, :], k, d, hess=True)
        Om_E = np.einsum('bia,bij,bjc->bac', V, He[:, 0], V) / k
        print(" Omega vs (HessR+kZ)/k:", np.abs(Hel.Omega - Om_R).max(), " vs P D2Z P/k:",
              np.abs(Hel.Omega - Om_E).max())
        # 4. helix vs direct regression quotient at a NON-critical point too
        for label, tt, TrX in [("critical", x1, Tr)]:
            th = rng.uniform(0.05, np.pi - 0.05, (B, 6))
            uu = random_sphere(rng, (B, 6), d - 1)
            G = Hel.eval(th, uu, grad=False)
            sp = np.cos(th)[..., None] * tt[:, None, :] + np.sin(th)[..., None] * np.einsum('bia,bsa->bsi', V, uu)
            Zs = eval_sym(Tf, sp, k, d)[0]
            c = (sp * tt[:, None, :]).sum(-1)
            # regression residual through explicit projection in feature space
            ft = features(M, tt)                                  # (B,m)
            Dft = np.stack([dfeatures(M, tt, V[:, :, i]) for i in range(d - 1)], 1)  # (B,N,m)
            basis = np.concatenate([ft[:, None, :], Dft], 1)     # (B,d,m)
            Qb, _ = np.linalg.qr(basis.transpose(0, 2, 1))        # (B,m,d)
            yperp = y - np.einsum('bmi,bi->bm', Qb, np.einsum('bmi,bm->bi', Qb, y))
            qs = np.einsum('bm,bsm->bs', yperp, features(M, sp))
            g_direct = qs / (1 - c ** k)
            g_simple = (Zs - c ** k * v1[:, None]) / (1 - c ** k)
            print(f" helix G vs projection quotient ({label}):", np.abs(G - g_direct).max(),
                  " vs (Z(s)-c^k l1)/(1-c^k):", np.abs(G - g_simple).max())
            # sigma_hat^2
            s2_proj = (yperp ** 2).sum(1)
            s2_form = (y ** 2).sum(1) - Tr[:, 0] ** 2 - k * (Tr[:, 1:d] ** 2).sum(1)
            print("  ||P_perp y||^2 projection vs formula:", np.abs(s2_proj - s2_form).max(),
                  " rank(basis)=", np.linalg.matrix_rank(basis[0]))
        # fibre limit
        uu = random_sphere(rng, (B, 4), d - 1)
        G0 = Hel.eval(np.zeros((B, 4)), uu, grad=False)
        Gsmall = Hel.eval(np.full((B, 4), 1e-7), uu, grad=False)
        sp = np.cos(1e-3) * x1[:, None, :] + np.sin(1e-3) * np.einsum('bia,bsa->bsi', V, uu)
        c = (sp * x1[:, None, :]).sum(-1)
        gq = (eval_sym(Tf, sp, k, d)[0] - c ** k * v1[:, None]) / (1 - c ** k)
        uOu = np.einsum('bsi,bij,bsj->bs', uu, Hel.Omega, uu)
        print(" fibre: G(0,u)-u'Om u:", np.abs(G0 - uOu).max(), " G(1e-7,u)-u'Om u:", np.abs(Gsmall - uOu).max(),
              " direct quotient at th=1e-3 minus u'Om u:", np.abs(gq - uOu).max())
        # gradient of helix by FD
        th = rng.uniform(-3, 3, (B, 3))
        uu = random_sphere(rng, (B, 3), d - 1)
        G, dth, gu = Hel.eval(th, uu)
        e = 1e-6
        dfd = (Hel.eval(th + e, uu, grad=False) - Hel.eval(th - e, uu, grad=False)) / (2 * e)
        w = rng.standard_normal(uu.shape)
        w = w - (w * uu).sum(-1, keepdims=True) * uu
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        gfd = (Hel.eval(th, normalize(uu + e * w), grad=False) - Hel.eval(th, normalize(uu - e * w), grad=False)) / (2 * e)
        print(" helix dG/dth FD:", np.abs(dfd - dth).max(), " dG/du FD:", np.abs(gfd - (gu * w).sum(-1)).max())
        # symmetry G(-th,-u) = G(th,u)
        print(" symmetry G(-th,-u)-G(th,u):", np.abs(Hel.eval(-th, -uu, grad=False) - G).max())
        # 5. statistic: Beta form vs quadrature vs closed form
        sig = np.sqrt(s2_form / M.kappa)
        r = np.linalg.eigvalsh(Hel.Omega) / sig[:, None]
        worst = 0
        worstc = 0
        for b in range(B):
            l2 = max(r[b].max(), 0) + abs(rng.normal()) + 0.3
            l1 = l2 * (1 + abs(rng.normal()))
            Tq = H_quad(M, r[b], l1) / H_quad(M, r[b], l2)
            Tc = H_closed(M, r[b], l1) / H_closed(M, r[b], l2)
            Tb = np.exp(logTstat(M, np.array([l1]), np.array([l2]), r[b:b + 1]))[0]
            worst = max(worst, abs(Tb / Tq - 1))
            worstc = max(worstc, abs(Tc / Tq - 1))
            # absolute Beta form H = kappa^{(nu+1)/2} l^{-kappa} J (up to the Student constant)
        cst = stats.t.pdf(0, M.nu)
        Hb = cst * M.kappa ** ((M.nu + 1) / 2) * 2.0 ** (-M.kappa) * np.exp(logJ(M, np.array([2.0]), r[:1]))[0]
        print(f" T: Beta-form vs quad max rel err {worst:.2e}; closed vs quad {worstc:.2e};"
              f" absolute H_r(2): Beta {Hb:.10e} quad {H_quad(M, r[0], 2.0):.10e}")
        # threshold consistency
        T1 = np.full(B, 12.0)
        tau, always = threshold_T2(M, T1, r, 0.05)
        chk = np.exp(logTstat(M, T1, tau, r))
        print(" threshold: T(T1,tau) =", np.round(chk[~always], 12), " always:", always.sum())


if __name__ == "__main__":
    main()
