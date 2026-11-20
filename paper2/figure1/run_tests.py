"""One point (k, d, lam) of Figure 1: every test and estimator of tests_lib on the same replicates, in parallel.

usage: python3 run_tests.py k d lam nrep [--chunk 1000] [--procs 4] [--seed 777] [--tag name]
                            [--oracle "1,2,3"|"self"|""] [--cloud 500,500,500,500] [--out DIR]

Each chunk of `chunk` replicates has its own seed SeedSequence([seed, k, d, round(1000 lam), chunk]).
Writes DIR/<tag>.npz (per-replicate arrays, float32 except the p-values) and appends one JSON line
to DIR/results.jsonl.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tests_lib import (Model, Cfg, Features, run_batch_ext, logG_closed, unfolding,  # noqa: E402
                       make_cloud, oracle_kappas, oracle_logR)

_FEAT = {}


def work(args):
    k, d, lam, B, seed_words, oracle_lams, cloud_sizes = args
    key = (k, d)
    if key not in _FEAT:
        M = Model(k, d)
        _FEAT[key] = (M, Features(M))
    M, feat = _FEAT[key]
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    cfg = Cfg()
    o = run_batch_ext(M, rng, B, lam, cfg)
    y, fro2 = o['y'], o['fro2']
    out = {}
    out['pT'] = np.exp(o['logTc'])
    out['pS'] = np.exp(logG_closed(o['lam1'], o['wO']) - logG_closed(o['lam2c'], o['wO']))
    out['lam1'] = o['lam1']
    out['lam2c'] = o['lam2c']
    out['sig'] = o['sig']
    out['T1'] = o['lam1'] / o['sig']
    Q = feat.energies(y)
    out['Q'] = Q
    if k == 3:
        n1, n3 = feat.nj
        Fst = (Q[:, 0] / n1) / (Q[:, 1] / n3)
        out['pF'] = stats.f.sf(Fst, n1, n3)
    out['Ttilde'] = (Q * feat.mu[None, :]).sum(1) / Q.sum(1)
    ust, u1 = unfolding(M, y)
    out['unfold'] = ust
    out['sin2_svd'] = 1.0 - u1[:, 0] ** 2
    out['sin2_argmax'] = 1.0 - o['c_e'] ** 2
    out['cos_argmax_signed'] = o['x1'][:, 0]
    out['fail'] = o['fail']
    out['not_local_max'] = o['hr_max'] >= 0
    out['gradZ'] = o['gradZ']
    if oracle_lams:
        yr = feat.y_rot(o['Tr'])
        for lo in oracle_lams:
            kap = [0.0, 10.0] + oracle_kappas(lo, k, M.m)
            cloud, logq = make_cloud(rng, d, kap, cloud_sizes)
            out['logR_%g' % lo] = oracle_logR(feat, yr, fro2, lo, cloud, logq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('k', type=int)
    ap.add_argument('d', type=int)
    ap.add_argument('lam', type=float)
    ap.add_argument('nrep', type=int)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--oracle', default='')
    ap.add_argument('--cloud', default='500,500,500,500')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    a = ap.parse_args()
    k, d, lam = a.k, a.d, a.lam
    if a.oracle == 'self':
        oracle_lams = [lam]
    elif a.oracle:
        oracle_lams = [float(x) for x in a.oracle.split(',')]
    else:
        oracle_lams = []
    cloud_sizes = [int(x) for x in a.cloud.split(',')]
    tag = a.tag or f'k{k}d{d}_lam{lam:g}_n{a.nrep}_s{a.seed}'
    os.makedirs(a.out, exist_ok=True)
    nchunks = (a.nrep + a.chunk - 1) // a.chunk
    tasks = []
    for c in range(nchunks):
        B = min(a.chunk, a.nrep - c * a.chunk)
        tasks.append((k, d, lam, B, [a.seed, k, d, int(round(1000 * lam)), c], oracle_lams, cloud_sizes))
    t0 = time.time()
    with Pool(a.procs) as pool:
        outs = pool.map(work, tasks, chunksize=1)
    wall = time.time() - t0
    raw = {kk: np.concatenate([o[kk] for o in outs]) for kk in outs[0]}
    n = raw['pT'].size
    summ = dict(tag=tag, k=k, d=d, lam=lam, n=int(n), seed=a.seed, chunk=a.chunk, oracle=oracle_lams,
                cloud=cloud_sizes, wall_s=round(wall, 1), time=time.strftime('%Y-%m-%d %H:%M:%S'),
                rejT=int((raw['pT'] <= 0.05).sum()), rejS=int((raw['pS'] <= 0.05).sum()),
                mean_sin2_argmax=float(raw['sin2_argmax'].mean()), mean_sin2_svd=float(raw['sin2_svd'].mean()),
                se_sin2_argmax=float(raw['sin2_argmax'].std() / np.sqrt(n)),
                se_sin2_svd=float(raw['sin2_svd'].std() / np.sqrt(n)),
                not_local_max=int(raw['not_local_max'].sum()), gradZ_bad=int((raw['gradZ'] > 1e-8).sum()),
                nan=int((~np.isfinite(raw['pT'])).sum()))
    if 'pF' in raw:
        summ['rejF'] = int((raw['pF'] <= 0.05).sum())
    if lam == 0:
        for key in ('pT', 'pS', 'pF'):
            if key in raw:
                v = raw[key][np.isfinite(raw[key])]
                summ['ks_p_' + key] = float(stats.kstest(v, 'uniform').pvalue)
    save = {}
    for kk, v in raw.items():
        if kk in ('pT', 'pS', 'pF'):
            save[kk] = v
        elif v.dtype == bool:
            save[kk] = v
        else:
            save[kk] = v.astype(np.float32)
    np.savez_compressed(os.path.join(a.out, tag + '.npz'), **save)
    with open(os.path.join(a.out, 'results.jsonl'), 'a') as f:
        f.write(json.dumps(summ) + '\n')
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
