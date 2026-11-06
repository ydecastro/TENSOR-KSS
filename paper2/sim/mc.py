"""One Monte Carlo batch of the t-spacing test (all replicates vectorised)."""
import numpy as np
from dataclasses import dataclass
from spike_lib import (Model, eval_sym, normalize, ascend_sphere, newton_sphere, random_sphere,
                       householder_frame, rotate, Helix, points_to_polar, logTstat, threshold_T2)


@dataclass
class Cfg:
    SZ: int = 40          # random starts for max of Z (plus one at e)
    itZ: int = 80
    SH: int = 40          # random starts for the helix
    n_other: int = 6      # other local maxima of Z used as helix starts
    itH: int = 120
    itH_polish: int = 300
    n_polish: int = 4     # helix candidates polished
    SD: int = 40          # random starts for the deflation check
    itD: int = 80
    alpha: float = 0.05


def maximize_Z(Tf, k, d, X0, iters, eta0):
    X, val, gn = ascend_sphere(Tf, X0, k, d, iters, eta0)
    B = X.shape[0]
    ib = val.argmax(1)
    xb = X[np.arange(B), ib]
    x1, v1, g1, HR = newton_sphere(Tf, xb, k, d)
    return x1, v1, g1, HR, X, val


def helix_max(Hel, rng, B, d, cfg, extra_th=None, extra_u=None, SH=None, n_polish=None):
    n = d - 1
    SH = cfg.SH if SH is None else SH
    n_polish = cfg.n_polish if n_polish is None else n_polish
    # starts: random, eigenvectors of Omega at th = 0, extra (other maxima of Z)
    th_r = rng.uniform(0, np.pi, (B, SH))
    u_r = random_sphere(rng, (B, SH), n)
    w, Vo = np.linalg.eigh(Hel.Omega)             # ascending
    u_e = np.concatenate([Vo.transpose(0, 2, 1), -Vo.transpose(0, 2, 1)], 1)  # (B,2n,n)
    th_e = np.zeros((B, 2 * n))
    ths = [th_r, th_e]
    us = [u_r, u_e]
    if extra_th is not None:
        ths.append(extra_th)
        us.append(extra_u)
    th = np.concatenate(ths, 1)
    U = np.concatenate(us, 1)
    th, U, G, gn = Hel.ascend(th, U, cfg.itH, 0.05)
    # polish the best candidates
    order = np.argsort(-G, axis=1)[:, :n_polish]
    ar = np.arange(B)[:, None]
    th2, U2, G2, gn2 = Hel.ascend(th[ar, order], U[ar, order], cfg.itH_polish, 0.05)
    ib = G2.argmax(1)
    a = np.arange(B)
    return th2[a, ib], U2[a, ib], G2[a, ib], gn2[a, ib], w


def polar_to_ambient(U, th, u):
    """rotated coords (cos th, sin th u) -> ambient via frame U."""
    a = np.concatenate([np.cos(th)[:, None], np.sin(th)[:, None] * u], 1)
    return np.einsum('bij,bj->bi', U, a)


def run_batch(model, rng, B, lam, cfg, check=False):
    k, d, n = model.k, model.d, model.d - 1
    y = model.draw_y(rng, B, lam)
    Tf = model.tensor(y)
    fro2 = (y ** 2).sum(1)
    etaZ = 1.0 / (k * (k - 1) * (abs(lam) + 3.0))
    # ---------------- lambda1
    X0 = random_sphere(rng, (B, cfg.SZ + 1), d)
    X0[:, 0, :] = 0
    X0[:, 0, 0] = 1.0                              # start at e
    x1, v1, g1, HR, XZ, valZ = maximize_Z(Tf, k, d, X0, cfg.itZ, etaZ)
    n_fix = np.zeros(B, dtype=np.int64)
    out = {}
    for rnd in range(4):
        U = householder_frame(x1)
        Tr = rotate(Tf, U, k, d)
        Hel = Helix(Tr, k, d)
        # other local maxima of Z (far from x1) as helix starts
        cosang = (XZ * x1[:, None, :]).sum(-1)
        vm = np.where(cosang < 0.995, valZ, -np.inf)
        oi = np.argsort(-vm, axis=1)[:, :cfg.n_other]
        ar = np.arange(B)[:, None]
        Xo = XZ[ar, oi]
        Ao = np.einsum('bji,bsj->bsi', U, Xo)       # rotated coords
        tho, uo, nwo = points_to_polar(Ao)
        th2, u2, lam2h, gnH, wO = helix_max(Hel, rng, B, d, cfg, tho, uo)
        lmaxO = wO[:, -1]
        lam2 = np.maximum(lam2h, lmaxO)
        bad = lam2 > v1 + 1e-9 * (1 + np.abs(v1))
        if not bad.any():
            break
        # helix exceeded lambda1: Z has a larger value at the helix argmax -> re-maximise Z there
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
    # local-max check: Riemannian Hessian eigenvalues on the tangent space
    V = U[:, :, 1:]
    hr_eig = np.linalg.eigvalsh(np.einsum('bia,bij,bjc->bac', V, HR, V))
    # ---------------- statistic
    s2perp = fro2 - Tr[:, 0] ** 2 - k * (Tr[:, 1:d] ** 2).sum(1)
    sig = np.sqrt(s2perp / model.kappa)
    T1 = v1 / sig
    T2 = lam2 / sig
    r = wO / sig[:, None]
    logT = logTstat(model, T1, T2, r)
    tau, always = threshold_T2(model, T1, r, cfg.alpha)
    ell = sig * tau
    fail_h = logT > np.log(cfg.alpha)
    # ---------------- deflation check (independent random starts)
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
    out.update(dict(lam1=v1, lam2=lam2, lam2h=lam2h, lam2c=lam2c, lmaxO=lmaxO, lminO=wO[:, 0], sig=sig,
                    logT=logT, logTc=logTc, tau=tau, always=always, fail_h=fail_h, fail_d=fail_d, fail=fail,
                    defl_excess=vd - ell, gradZ=g1, hr_max=hr_eig[:, -1], gradH=gnH, n_fix=n_fix,
                    c_e=np.abs(x1[:, 0]), th2=np.abs(th2), smallT2=(T2 < 0.05)))
    if check:
        # re-run both searches with 4x starts and more iterations (independent random starts)
        cfg4 = Cfg(SZ=4 * cfg.SZ, itZ=2 * cfg.itZ, SH=4 * cfg.SH, n_other=cfg.n_other, itH=2 * cfg.itH,
                   itH_polish=cfg.itH_polish, n_polish=2 * cfg.n_polish, SD=cfg.SD, itD=cfg.itD, alpha=cfg.alpha)
        X0 = random_sphere(rng, (B, cfg4.SZ + 1), d)
        X0[:, 0, :] = 0
        X0[:, 0, 0] = 1.0
        x1b, v1b, _, _, _, _ = maximize_Z(Tf, k, d, X0, cfg4.itZ, etaZ)
        Ub = householder_frame(x1b)
        Helb = Helix(rotate(Tf, Ub, k, d), k, d)
        _, _, l2b, _, wb = helix_max(Helb, rng, B, d, cfg4)
        l2b = np.maximum(l2b, wb[:, -1])
        out.update(dict(chk_lam1=v1b, chk_lam2=l2b))
    return out
