"""Section 6, paragraph "The figure of the introduction": on 3000 replicates at (3,8), the power of the most powerful
invariant test at lambda = 4, 6, 8 with the importance-sampling cloud of 2000 points used for Figure 1 ("mid": four
components of 500 points) and with a cloud four times larger ("4x": 8000 points) (paper: the power changes by at most
0.002). The first line of each lambda ("base") is a three-component cloud of 2000 points that was not used. The
studentised maximum and T are computed on the same replicates. Run: python3 check_cloud_size.py (about 2 minutes).
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tests_lib import *  # noqa
def say(*a): print(*a, flush=True)
M = Model(3, 8); F = Features(M)
rng = np.random.default_rng(11)
o0 = run_batch_ext(M, rng, 3000, 0.0, Cfg()); yr0 = F.y_rot(o0['Tr'])
for lo in (4.0, 6.0, 8.0):
    o1 = run_batch_ext(M, rng, 3000, lo, Cfg()); yr1 = F.y_rot(o1['Tr'])
    kap = [0.0] + oracle_kappas(lo, 3, M.m)
    line = f'lam={lo}: '
    for name, sizes in (('base', [500,750,750]), ('4x', [2000,3000,3000]), ('mid', [500,500,500,500])):
        kk = kap if len(sizes) == 3 else [0.0, 10.0] + oracle_kappas(lo, 3, M.m)
        cl, q = make_cloud(rng, 8, kk, sizes); r0 = oracle_logR(F, yr0, o0['fro2'], lo, cl, q)
        cl, q = make_cloud(rng, 8, kk, sizes); r1 = oracle_logR(F, yr1, o1['fro2'], lo, cl, q)
        thr = np.quantile(r0, 0.95)
        line += f'{name}: thr {thr:.3f} power {np.mean(r1 > thr):.4f} | '
    # the other sigma-free tests at the same replicates, MC calibrated on the same null set
    for nm, s0, s1 in (('studmax', o0['lam1']/o0['sig'], o1['lam1']/o1['sig']),):
        thr = np.quantile(s0, 0.95); line += f'{nm} power {np.mean(s1 > thr):.4f} | '
    pT = np.exp(o1['logTc']); line += f'T power {np.mean(pT <= 0.05):.4f}'
    say(line)
