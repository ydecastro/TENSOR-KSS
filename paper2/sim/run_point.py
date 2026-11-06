"""Run one (k,d,lambda) point with multiprocessing; append a JSON line of raw counts to results.jsonl
and save per-replicate key quantities to raw/<tag>.npz.

usage: python3 run_point.py k d lam nrep [--chunk 1000] [--check-frac 0.05] [--tag name] [--seed s]
       [--procs p] [--out results.jsonl] [--raw-dir raw]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spike_lib import Model  # noqa: E402
from mc import Cfg, run_batch  # noqa: E402

KEEP = ['lam1', 'lam2c', 'lmaxO', 'lminO', 'sig', 'logTc', 'fail', 'fail_h', 'fail_d']


def work(args):
    k, d, lam, B, seed_words, check, cfgd = args
    M = Model(k, d)
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    cfg = Cfg(**cfgd)
    o = run_batch(M, rng, B, lam, cfg, check=check)
    tol = 1e-7
    s = dict(
        n=B,
        fail=int(o['fail'].sum()), fail_h=int(o['fail_h'].sum()), fail_d=int(o['fail_d'].sum()),
        h_only=int((o['fail_h'] & ~o['fail_d']).sum()), d_only=int((o['fail_d'] & ~o['fail_h']).sum()),
        lam2c_gt_lam2=int((o['lam2c'] > o['lam2'] + 1e-9).sum()),
        fail_changed_by_defl=int((o['fail'] != o['fail_h']).sum()),
        viol_lam2_gt_lam1=int((o['lam2c'] > o['lam1'] + 1e-9 * (1 + np.abs(o['lam1']))).sum()),
        viol_lmaxO_gt_lam2=int((o['lmaxO'] > o['lam2c'] + 1e-12).sum()),
        viol_lmaxO_ge_lam1=int((o['lmaxO'] >= o['lam1']).sum()),
        fibre_is_sup=int((o['lam2h'] < o['lmaxO']).sum()),
        not_local_max=int((o['hr_max'] >= 0).sum()),
        n_fix=int(o['n_fix'].sum()),
        gradZ_bad=int((o['gradZ'] > 1e-8).sum()),
        gradH_bad=int((o['gradH'] > 1e-5).sum()),
        always=int(o['always'].sum()), smallT2=int(o['smallT2'].sum()),
        nan=int((~np.isfinite(o['logTc'])).sum()),
        sum_sig2=float((o['sig'] ** 2).sum()),
        n_check=0, chk_lam1_up=0, chk_lam2_up=0, chk_lam2_down=0, chk_fail_change=0, chk_lam1_diff=0,
    )
    if check:
        up1 = o['chk_lam1'] > o['lam1'] + tol * (1 + np.abs(o['lam1']))
        diff1 = np.abs(o['chk_lam1'] - o['lam1']) > tol * (1 + np.abs(o['lam1']))
        same = ~diff1
        up2 = same & (o['chk_lam2'] > o['lam2c'] + tol * (1 + np.abs(o['lam2c'])))
        dn2 = same & (o['chk_lam2'] < o['lam2c'] - tol * (1 + np.abs(o['lam2c'])))
        # where lambda1 agrees (so sigma_hat, Omega and the threshold are identical) a decision changes
        # iff the 4x-start helix exceeds the threshold ell* = sigma_hat * tau while the main run did not fail
        thr = o['tau'] * o['sig']
        chg = same & ((o['chk_lam2'] > thr) & ~o['fail'])
        s.update(n_check=B, chk_lam1_up=int(up1.sum()), chk_lam1_diff=int(diff1.sum()),
                 chk_lam2_up=int(up2.sum()), chk_lam2_down=int(dn2.sum()), chk_fail_change=int(chg.sum()))
    keep = {kk: o[kk] for kk in KEEP}
    return s, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('k', type=int)
    ap.add_argument('d', type=int)
    ap.add_argument('lam', type=float)
    ap.add_argument('nrep', type=int)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--check-frac', type=float, default=0.05)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--procs', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(HERE, 'results.jsonl'))
    ap.add_argument('--raw-dir', default=os.path.join(HERE, 'raw'))
    ap.add_argument('--cfg', default='{}', help='JSON overrides of Cfg fields')
    a = ap.parse_args()
    cfgd = json.loads(a.cfg)
    cfg = Cfg(**cfgd)
    nch = (a.nrep + a.chunk - 1) // a.chunk
    ncheck = int(round(a.check_frac * nch))
    tag = a.tag or f"k{a.k}d{a.d}_lam{a.lam:g}_n{a.nrep}_s{a.seed}"
    tasks = []
    for c in range(nch):
        B = min(a.chunk, a.nrep - c * a.chunk)
        tasks.append((a.k, a.d, a.lam, B, [a.seed, a.k, a.d, int(round(a.lam * 1000)), c],
                      c < ncheck, cfgd))
    t0 = time.time()
    tot = None
    keeps = {kk: [] for kk in KEEP}
    with Pool(a.procs) as pool:
        for s, keep in pool.imap_unordered(work, tasks):
            if tot is None:
                tot = {kk: 0 for kk in s}
            for kk, v in s.items():
                tot[kk] += v
            for kk in KEEP:
                keeps[kk].append(keep[kk])
    dt = time.time() - t0
    M = Model(a.k, a.d)
    rec = dict(tag=tag, k=a.k, d=a.d, m=M.m, kappa=M.kappa, lam=a.lam, nrep=a.nrep, alpha=cfg.alpha,
               seed=a.seed, cfg=cfg.__dict__, wall_s=round(dt, 1), time=time.strftime('%Y-%m-%d %H:%M:%S'))
    rec.update(tot)
    rec['deficiency'] = tot['fail'] / tot['n']
    raw = {kk: np.concatenate(v) for kk, v in keeps.items()}
    os.makedirs(a.raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.raw_dir, tag + '.npz'),
                        **{kk: (v.astype(np.float32) if v.dtype == np.float64 and kk != 'logTc' else v)
                           for kk, v in raw.items()})
    if a.lam == 0.0:
        from scipy import stats
        Tv = np.exp(raw['logTc'])
        ks = stats.kstest(Tv[np.isfinite(Tv)], 'uniform')
        rec['ks_stat'] = float(ks.statistic)
        rec['ks_p'] = float(ks.pvalue)
        rec['level_0.05'] = float((Tv <= 0.05).mean())
        rec['level_0.01'] = float((Tv <= 0.01).mean())
    with open(a.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
