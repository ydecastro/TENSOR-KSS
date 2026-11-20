"""Checks of tests_lib.py, made before the simulation of Figure 1: harmonic projectors against the Funk-Hecke
eigenvalues, Z computed from the features against the tensor contraction, log I against quadrature, the von Mises-Fisher
sampler against the Bessel ratios, the rotation that sends the argmax to the pole, and the noise of the importance
sampling of the most powerful invariant test. Output check_tests_lib.log. Run: python3 check_tests_lib.py (a few seconds)."""
import os, sys, time
import numpy as np
from scipy import integrate, special, stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tests_lib import *  # noqa

def say(*a): print(*a, flush=True)

# 1. harmonic projectors
for (k, d) in [(3, 3), (3, 5), (3, 8), (5, 3)]:
    M = Model(k, d); F = Features(M)
    mus = [mu_funk_hecke(k, d, j) for j in F.degrees]
    say(f'({k},{d}) degrees {F.degrees} n_j {F.nj} mu {np.round(F.mu,8)} Funk-Hecke {np.round(mus,8)} '
        f'sum mu n = {float((F.mu*np.array(F.nj)).sum()):.12f}  max|diff| {max(abs(a-b) for a,b in zip(F.mu,mus)):.2e}')
    # f(t) reproduces Z: <y,f(t)> against the full tensor contraction
    rng = np.random.default_rng(0); y = M.draw_y(rng, 3, 2.0); t = random_sphere(rng, (5,), d)
    Zf = F.eval(t) @ y.T
    Tf = M.tensor(y).reshape((3,) + (d,) * k)
    Zt = np.array([[np.einsum(Tf[b], list(range(k)), *[x for i in range(k) for x in (t[j], [i])]) for b in range(3)] for j in range(5)])
    say(f'   Z by features vs tensor: max|diff| {np.abs(Zf-Zt).max():.2e}')

# 2. log_I against quadrature
for m in (10, 35, 120):
    for c in (-5.0, 0.0, 3.0, 20.0, 60.0):
        q = integrate.quad(lambda r: np.exp((m-1)*np.log(r) - r*r/2 + c*r - (lambda rs: (m-1)*np.log(rs) - rs*rs/2 + c*rs)(0.5*(c+np.sqrt(c*c+4*(m-1))))), 0, np.inf, epsabs=0, epsrel=1e-12, limit=500)[0]
        rs = 0.5*(c+np.sqrt(c*c+4*(m-1))); lq = np.log(q) + (m-1)*np.log(rs) - rs*rs/2 + c*rs
        say(f'log_I m={m} c={c}: GH {float(log_I(c, m)):.10f} quad {lq:.10f} diff {float(log_I(c,m))-lq:.2e}')

# 3. log_M and the vMF sampler
rng = np.random.default_rng(1)
for d in (3, 8):
    for kap in (0.5, 10.0, 100.0, 300.0):
        f = lambda w: np.exp(kap*w) * (1-w*w)**((d-3)/2)
        g = lambda w: (1-w*w)**((d-3)/2)
        num = integrate.quad(f, -1, 1, limit=500, points=[1-10/kap] if kap > 20 else None)[0]
        den = integrate.quad(g, -1, 1)[0]
        w = sample_vmf_w(rng, kap, d, 200000)
        Ad = special.ive(d/2, kap) / special.ive(d/2-1, kap)
        say(f'log_M d={d} kappa={kap}: bessel {log_M(kap,d):.8f} quad {np.log(num/den):.8f} | sampler mean w {w.mean():.5f} vs A_d {Ad:.5f} (se {w.std()/np.sqrt(w.size):.5f})')

# 4. y_rot: Z(U pole) = lambda_1
M = Model(3, 8); F = Features(M); rng = np.random.default_rng(2)
o = run_batch_ext(M, rng, 200, 4.0, Cfg())
yr = F.y_rot(o['Tr'])
say(f'y_rot[e_index] vs lam1: max|diff| {np.abs(yr[:, M.e_index]-o["lam1"]).max():.2e}; Z(U w) check:',
    f'{np.abs((F.eval(np.array([[0.6,0.8,0,0,0,0,0,0]])) @ yr.T)[0] - (F.eval(np.einsum("bij,j->bi", o["U"], np.array([0.6,0.8,0,0,0,0,0,0]))) * o["y"]).sum(1)).max():.2e}')

# 5. oracle importance-sampling noise at (3,8): two independent clouds, and a cloud twice as large
for lam_sim in (0.0, 6.0):
    rng = np.random.default_rng(3)
    o = run_batch_ext(M, rng, 300, lam_sim, Cfg()); yr = F.y_rot(o['Tr'])
    for lo in (3.0, 6.0, 10.0):
        kap = [0.0] + oracle_kappas(lo, 3, M.m)
        c1, q1 = make_cloud(rng, 8, kap, [500, 750, 750]); c2, q2 = make_cloud(rng, 8, kap, [500, 750, 750])
        c3, q3 = make_cloud(rng, 8, kap, [2000, 3000, 3000])
        r1 = oracle_logR(F, yr, o['fro2'], lo, c1, q1); r2 = oracle_logR(F, yr, o['fro2'], lo, c2, q2); r3 = oracle_logR(F, yr, o['fro2'], lo, c3, q3)
        say(f'oracle noise sim lam={lam_sim} test lam={lo}: kappas {np.round(kap,1)} sd(logR1-logR2) {np.std(r1-r2):.4f}  sd(logR1-logR3) {np.std(r1-r3):.4f}  '
            f'spread sd(logR) {np.std(r1):.3f}  mean logR {r1.mean():.3f}')
