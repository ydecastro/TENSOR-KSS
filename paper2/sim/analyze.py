"""Section 7: the deficiency table, the null calibration (KS p-values) and the fits of -log(deficiency) on lambda^2,
from the JSON lines written by run_point.py. Records with the same (k,d,lambda) are pooled.
usage: python3 analyze.py [results.jsonl results_rows_33_34.jsonl ...]"""
import sys
import json
import numpy as np
from math import comb, sqrt
from scipy import stats

fns = sys.argv[1:] or ['results.jsonl', 'results_rows_33_34.jsonl']
recs = [json.loads(l) for fn in fns for l in open(fn) if l.strip()]
ALPHA = 0.05


def cp_ci(x, n, a=0.05):
    lo = stats.beta.ppf(a / 2, x, n - x + 1) if x > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi


def constants(k, d, alpha=ALPHA):
    m = comb(d + k - 1, k)
    kap = m - d
    N = d - 1
    rho2 = k / (2 * (k - 1))
    rho = sqrt(rho2)
    A_bin = comb(kap + N, N) ** (1 / kap) * alpha ** (-1 / kap)
    q = stats.beta.ppf(alpha, kap, N + 1)
    A_sharp = 1 / q
    return dict(kappa=kap, mech=rho2 / (2 * (rho + A_bin) ** 2), bin=rho2 / (2 * (A_bin ** 2 + rho2)),
                sharp=rho2 / (2 * (A_sharp ** 2 + rho2)), A_bin=A_bin, A_sharp=A_sharp)


def wls(x, yv, w):
    X = np.stack([np.ones_like(x), x], 1)
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ yv
    res = yv - X @ beta
    chi2 = float((w * res ** 2).sum())
    dof = len(x) - 2
    return beta, cov, chi2, dof


# merge records by (k,d,lam) (production seed only unless flagged)
pts = {}
for r in recs:
    key = (r['k'], r['d'], r['lam'])
    p = pts.setdefault(key, dict(n=0, fail=0, recs=[]))
    p['n'] += r['n']
    p['fail'] += r['fail']
    p['recs'].append(r)

print("=" * 100)
print("Null calibration")
for key, p in sorted(pts.items()):
    if key[2] != 0:
        continue
    for r in p['recs']:
        print(f" (k,d)=({r['k']},{r['d']}) n={r['n']}  KS stat={r['ks_stat']:.5f} p={r['ks_p']:.3f}  "
              f"P(T<=.05)={r['level_0.05']:.5f} P(T<=.01)={r['level_0.01']:.5f}  "
              f"lam2>lam1:{r['viol_lam2_gt_lam1']} lmaxO>=lam1:{r['viol_lmaxO_ge_lam1']} lmaxO>lam2:{r['viol_lmaxO_gt_lam2']} "
              f"not-local-max:{r['not_local_max']} Z-restarts(helix>lam1 caught):{r['n_fix']} "
              f"| check n={r['n_check']} lam1 up:{r['chk_lam1_up']} lam1 diff:{r['chk_lam1_diff']} "
              f"lam2 up:{r['chk_lam2_up']} decision changes:{r['chk_fail_change']} | helix-only:{r['h_only']} "
              f"defl-only:{r['d_only']} nan:{r['nan']} T2<0.05:{r['smallT2']} mean sig2:{r['sum_sig2']/r['n']:.4f} wall {r['wall_s']}s")

paper33 = {6: 0.179, 7: 0.0771, 8: 2.86e-2, 9: 9.63e-3, 10: 2.89e-3, 12: 1.81e-4}
for kd in sorted({(k, d) for (k, d, l) in pts}):
    k, d = kd
    C = constants(k, d)
    rows = sorted([(l, p) for (kk, dd, l), p in pts.items() if (kk, dd) == kd and l > 0])
    if not rows:
        continue
    print("=" * 100)
    print(f"(k,d)=({k},{d})  kappa={C['kappa']}  alpha={ALPHA}   A_bin={C['A_bin']:.4f} A_sharp=1/q={C['A_sharp']:.4f}")
    print(f"  recomputed comparison exponents: mechanism {C['mech']:.4f}, ratio-binomial {C['bin']:.4f}, "
          f"ratio-sharp {C['sharp']:.4f}")
    print(f"  {'lam':>5} {'reps':>9} {'fail':>7} {'deficiency':>11} {'95% CI (Clopper-Pearson)':>28}  "
          f"{'-log':>7}  diag: lam2>lam1 lmaxO>=lam1 notmax Zfix | chk_n lam1up lam2up dec.chg | h_only d_only")
    L, Y, Wt = [], [], []
    for l, p in rows:
        n, x = p['n'], p['fail']
        ph = x / n
        lo, hi = cp_ci(x, n)
        r = p['recs'][0]
        agg = {f: sum(rr[f] for rr in p['recs']) for f in ['viol_lam2_gt_lam1', 'viol_lmaxO_ge_lam1', 'not_local_max',
                                                            'n_fix', 'n_check', 'chk_lam1_up', 'chk_lam2_up',
                                                            'chk_fail_change', 'h_only', 'd_only']}
        extra = f"   paper {paper33[l]:.3g}" if kd == (3, 3) and l in paper33 else ""
        print(f"  {l:5.2f} {n:9d} {x:7d} {ph:11.4e} [{lo:10.3e}, {hi:10.3e}]  {-np.log(ph) if x else np.inf:7.3f}"
              f"  {agg['viol_lam2_gt_lam1']:>5} {agg['viol_lmaxO_ge_lam1']:>5} {agg['not_local_max']:>5} {agg['n_fix']:>4}"
              f" | {agg['n_check']:>6} {agg['chk_lam1_up']:>3} {agg['chk_lam2_up']:>3} {agg['chk_fail_change']:>3}"
              f" | {agg['h_only']:>5} {agg['d_only']:>3}{extra}")
        if kd == (3, 3) and l in paper33:
            z = (ph - paper33[l]) / sqrt(paper33[l] * (1 - paper33[l]) / n)
            print(f"         vs paper: ratio {ph/paper33[l]:.3f}, z = {z:+.2f} (binomial, paper value as truth), "
                  f"rel. SE {sqrt((1-ph)/(n*ph)):.3f}")
        if x > 0:
            L.append(l)
            Y.append(-np.log(ph))
            Wt.append(n * ph / (1 - ph))  # 1/var(log p_hat)
    L, Y, Wt = map(np.array, (L, Y, Wt))
    if len(L) >= 2:
        print("  local slopes  d(-log def)/d(lam^2):")
        for i in range(len(L) - 1):
            s = (Y[i + 1] - Y[i]) / (L[i + 1] ** 2 - L[i] ** 2)
            se = sqrt(1 / Wt[i] + 1 / Wt[i + 1]) / (L[i + 1] ** 2 - L[i] ** 2)
            print(f"    [{L[i]:.2f},{L[i+1]:.2f}]  {s:.4f} +- {se:.4f}")
    if len(L) >= 3:
        for lo_i in range(0, len(L) - 2):
            beta, cov, chi2, dof = wls(L[lo_i:] ** 2, Y[lo_i:], Wt[lo_i:])
            print(f"  WLS fit -log def = a + b lam^2 on lam in [{L[lo_i]:.2f},{L[-1]:.2f}]:  b = {beta[1]:.4f} "
                  f"+- {sqrt(cov[1,1]):.4f} (stat), a = {beta[0]:.3f};  chi2/dof = {chi2:.1f}/{dof}")
        # unweighted, whole grid
        b_unw = np.polyfit(L ** 2, Y, 1)[0]
        print(f"  OLS (unweighted) whole grid: b = {b_unw:.4f}")
        # fit with a log-prefactor: -log def = a + b lam^2 + c log lam
        X = np.stack([np.ones_like(L), L ** 2, np.log(L)], 1)
        Wm = np.diag(Wt)
        cov = np.linalg.inv(X.T @ Wm @ X)
        beta = cov @ X.T @ Wm @ Y
        print(f"  WLS fit with log-prefactor -log def = a + b lam^2 + c log lam: b = {beta[1]:.4f} +- "
              f"{sqrt(cov[1,1]):.4f}, c = {beta[2]:.2f}")
        # fit with a linear term: -log def = a + b lam^2 + c lam (the correction of order lambda of the lower bound)
        X = np.stack([np.ones_like(L), L ** 2, L], 1)
        cov = np.linalg.inv(X.T @ Wm @ X)
        beta = cov @ X.T @ Wm @ Y
        res = Y - X @ beta
        chi2 = float((Wt * res ** 2).sum())
        print(f"  WLS fit with a linear term -log def = a + b lam^2 + c lam on lam in [{L[0]:.2f},{L[-1]:.2f}]: "
              f"b = {beta[1]:.4f} +- {sqrt(cov[1,1]):.4f}, c = {beta[2]:.3f} +- {sqrt(cov[2,2]):.3f}, "
              f"a = {beta[0]:.3f};  chi2/dof = {chi2:.1f}/{len(L)-3}")
