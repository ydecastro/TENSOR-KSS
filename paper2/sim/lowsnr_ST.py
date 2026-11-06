"""Section 6 (sec:low), table at (k,d) = (3,3): power of the t-spacing test T and of the known-variance spacing
test S, alpha = 0.05, lambda in {0 (null calibration), 1.4, 1.6, 1.8, 2.0, 3.0, 5.0}.

Uses the field simulation of this folder (spike_lib.py, mc.py): run_batch returns lambda_1 (lam1), the second
maximum after the deflation check (lam2c), the eigenvalues of Omega (lminO, lmaxO; d = 3, so N = 2 and these
are all of them), sigma_hat (sig) and log T (logTc, Beta form).

The known-variance statistic is added from Proposition 2 (prop:closed), sigma = 1:
    S = G_r(lambda_1) / G_r(lambda_2),  r = Omega,
    G_r(l) = sum_{j=0}^N (-1)^j e_j(r) I_{N-j}(l),  I_0 = 1-Phi, I_1 = phi, I_p = l^{p-1} phi(l) + (p-1) I_{p-2}.
For N = 2: G_r(l) = (1 + e_2) (1-Phi(l)) + (l - e_1) phi(l). Cross-checked against quadrature below.

Writes one JSON line per lambda (after each lambda) and the null values to null_vals.npz.
usage: python3 lowsnr_ST.py [--nnull 1000000] [--nrep 500000] [--procs 10] [--seed 20261019]
"""
import os, sys, json, time, argparse
import numpy as np
from multiprocessing import Pool
from scipy import stats, integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = HERE
sys.path.insert(0, os.path.abspath(OLD))
from spike_lib import Model  # noqa: E402
from mc import Cfg, run_batch  # noqa: E402

ALPHA = 0.05


def logG_closed(l, eig):
    """log G_r(l) for l (B,), eig (B,N) eigenvalues of r (any N), via the I_p recursion."""
    B, N = eig.shape
    # elementary symmetric polynomials e_0..e_N
    e = np.zeros((B, N + 1)); e[:, 0] = 1.0
    for i in range(N):
        e[:, 1:] = e[:, 1:] + eig[:, i:i + 1] * e[:, :-1]
    phi = stats.norm.pdf(l); sf = stats.norm.sf(l)
    I = [sf, phi]
    for p in range(2, N + 1):
        I.append(l ** (p - 1) * phi + (p - 1) * I[p - 2])
    G = np.zeros(B)
    for j in range(N + 1):
        G += (-1) ** j * e[:, j] * I[N - j]
    return np.log(G)


def check_closed_form():
    rng = np.random.default_rng(1)
    for _ in range(5):
        eig = np.sort(rng.standard_normal(2))
        l = eig.max() + rng.uniform(0.1, 3)
        q = integrate.quad(lambda u: (u - eig[0]) * (u - eig[1]) * stats.norm.pdf(u), l, np.inf)[0]
        c = np.exp(logG_closed(np.array([l]), eig[None, :]))[0]
        assert abs(q - c) < 1e-12 * max(1, abs(q)), (q, c)
    return True


def work(args):
    k, d, lam, B, seed_words, check, cfgd = args
    M = Model(k, d)
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    cfg = Cfg(**cfgd)
    o = run_batch(M, rng, B, lam, cfg, check=check)
    eig = np.stack([o['lminO'], o['lmaxO']], 1)
    logS = logG_closed(o['lam1'], eig) - logG_closed(o['lam2c'], eig)
    T2 = o['lam2c'] / o['sig']
    s = dict(n=B,
             rejT=int((o['logTc'] <= np.log(ALPHA)).sum()),
             rejS=int((logS <= np.log(ALPHA)).sum()),
             rejT_h=int((o['logT'] <= np.log(ALPHA)).sum()),
             both=int(((o['logTc'] <= np.log(ALPHA)) & (logS <= np.log(ALPHA))).sum()),
             nanT=int((~np.isfinite(o['logTc'])).sum()), nanS=int((~np.isfinite(logS)).sum()),
             viol_order=int((o['lmaxO'] > o['lam2c'] + 1e-12).sum() + (o['lam2c'] > o['lam1'] + 1e-9 * (1 + np.abs(o['lam1']))).sum()),
             T2_over_bound=int((T2 > np.sqrt(M.kappa) / np.sqrt(k / (2 * (k - 1))) + 1e-9).sum()),
             S_gt1=int((logS > 1e-12).sum()),
             n_fix=int(o['n_fix'].sum()), gradZ_bad=int((o['gradZ'] > 1e-8).sum()),
             not_local_max=int((o['hr_max'] >= 0).sum()),
             n_check=0, chk_lam1_diff=0, chk_lam2_up=0)
    if check:
        tol = 1e-7
        diff1 = np.abs(o['chk_lam1'] - o['lam1']) > tol * (1 + np.abs(o['lam1']))
        up2 = (~diff1) & (o['chk_lam2'] > o['lam2c'] + tol * (1 + np.abs(o['lam2c'])))
        s.update(n_check=B, chk_lam1_diff=int(diff1.sum()), chk_lam2_up=int(up2.sum()))
    keep = dict(logT=o['logTc'].astype(np.float64), logS=logS.astype(np.float64))
    return s, keep


def run_point(k, d, lam, nrep, procs, seed, chunk=1000, check_frac=0.02, out=None):
    nch = (nrep + chunk - 1) // chunk
    ncheck = int(round(check_frac * nch))
    tasks = [(k, d, lam, min(chunk, nrep - c * chunk), [seed, k, d, int(round(lam * 1000)), c], c < ncheck, {})
             for c in range(nch)]
    t0 = time.time(); tot = None; keeps = {'logT': [], 'logS': []}
    with Pool(procs) as pool:
        for s, keep in pool.imap_unordered(work, tasks):
            if tot is None:
                tot = {kk: 0 for kk in s}
            for kk, v in s.items():
                tot[kk] += v
            for kk in keeps:
                keeps[kk].append(keep[kk])
    raw = {kk: np.concatenate(v) for kk, v in keeps.items()}
    n = tot['n']
    rec = dict(k=k, d=d, lam=lam, nrep=n, alpha=ALPHA, seed=seed, wall_s=round(time.time() - t0, 1),
               time=time.strftime('%Y-%m-%d %H:%M:%S'))
    rec.update(tot)
    for nm in ('T', 'S'):
        p = tot['rej' + nm] / n
        rec['power_' + nm] = p
        rec['se_' + nm] = float(np.sqrt(p * (1 - p) / n))
    if lam == 0.0:
        for nm in ('T', 'S'):
            v = np.exp(raw['log' + nm]); v = v[np.isfinite(v)]
            ks = stats.kstest(v, 'uniform')
            rec['ks_stat_' + nm] = float(ks.statistic); rec['ks_p_' + nm] = float(ks.pvalue)
            rec['level_0.05_' + nm] = float((v <= 0.05).mean()); rec['level_0.01_' + nm] = float((v <= 0.01).mean())
        np.savez_compressed(os.path.join(HERE, 'null_vals.npz'), **raw)
    if out:
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nnull', type=int, default=1000000)
    ap.add_argument('--nrep', type=int, default=500000)
    ap.add_argument('--procs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20261019)
    ap.add_argument('--lams', default='1.4,1.6,1.8,2.0,3.0,5.0')
    ap.add_argument('--skip-null', action='store_true')
    a = ap.parse_args()
    assert check_closed_form()
    print('closed form of G_r checked against quadrature', flush=True)
    out = os.path.join(HERE, 'results_lowsnr.jsonl')
    if not a.skip_null:
        run_point(3, 3, 0.0, a.nnull, a.procs, a.seed, out=out)
    for lam in [float(x) for x in a.lams.split(',')]:
        run_point(3, 3, lam, a.nrep, a.procs, a.seed, out=out)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
