"""Section 6 (sec:low): the studentised quadratic statistic at (k,d) = (3,3).
Model in R^m, m = 10: Y ~ N(lam psi_e, sigma^2 I). Q_1 = |P_1 Y|^2 ~ sigma^2 chi2_3(lam^2 mu_1 n_1/sigma^2),
Q_3 = |P_3 Y|^2 ~ sigma^2 chi2_7(lam^2 mu_3 n_3/sigma^2), independent; mu_1 = 1/5, mu_3 = 2/35.
T_q = mu_1 Q_1 + mu_3 Q_3; the invariant statistic T_q/(Q_1+Q_3) is increasing in Q_1/Q_3, and
F = (7/3) Q_1/Q_3 ~ F_{3,7} under the null.
Computes (1) the null law of F (KS against F_{3,7}); (2) the local coefficient C_F = Cov_0(1{F>f_alpha}, T_q)/2,
exactly and by Monte Carlo, against C_opt and C_max; (3) the power of the F-test on the lambda grid, against the
exact power of the quadratic test (quadratic_test.py).
"""
import numpy as np, os, json
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261019)
mu1, mu3, n1, n3 = 1 / 5, 2 / 35, 3, 7
C_opt = {0.05: 3.655341e-02, 0.01: 1.080666e-02}   # a4 (exact)
C_max = {0.05: 2.9607e-02, 0.01: 8.4527e-03}        # a5 (MC, 4e6 null replicates)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

n = 4_000_000
Q1 = rng.chisquare(n1, n); Q3 = rng.chisquare(n3, n)
F = (n3 / n1) * Q1 / Q3
T = mu1 * Q1 + mu3 * Q3
ks = stats.kstest(F, 'f', args=(n1, n3))
say(f"null law of F=(7/3)Q1/Q3 vs F_(3,7): KS stat {ks.statistic:.5f}, p = {ks.pvalue:.3f}  ({n} replicates)")
for a in (0.05, 0.01):
    f = stats.f.ppf(1 - a, n1, n3)
    phi = (F > f).astype(float)
    C_mc = 0.5 * np.cov(phi, T)[0, 1]
    se = 0.5 * np.std((phi - phi.mean()) * (T - T.mean())) / np.sqrt(n)
    # exact: E[T; F>f] = mu1*3*P(F_{5,7} > 3f/5) + mu3*7*P(F_{3,9} > 9f/7)
    ETf = mu1 * n1 * stats.f.sf(3 * f / 5, 5, 7) + mu3 * n3 * stats.f.sf(9 * f / 7, 3, 9)
    C_ex = 0.5 * (ETf - a)
    say(f"alpha={a}: f_alpha={f:.4f}; C_F exact = {C_ex:.5e}, MC = {C_mc:.5e} (+-{se:.1e}); "
        f"C_F/C_opt = {C_ex/C_opt[a]:.4f}, C_F/C_max = {C_ex/C_max[a]:.4f}, C_max/C_opt = {C_max[a]/C_opt[a]:.4f}")

# power of the F-test (doubly non-central), MC with common random numbers, 4e6 per lambda
lams = [1.4, 1.6, 1.8, 2.0, 3.0, 5.0]
qp = np.load(os.path.join(HERE, 'quadpower.npy'))
qlams = list(qp[0])
Z1 = rng.standard_normal((n, n1)); Z3 = rng.standard_normal((n, n3))
res = {}
for a in (0.05, 0.01):
    f = stats.f.ppf(1 - a, n1, n3)
    row = []
    for lam in lams:
        # non-centrality 0.6 lam^2 in Q1: mean vector of norm sqrt(0.6) lam along the first coordinate
        q1 = ((Z1[:, 0] + np.sqrt(mu1 * n1) * lam) ** 2 + (Z1[:, 1:] ** 2).sum(1))
        q3 = ((Z3[:, 0] + np.sqrt(mu3 * n3) * lam) ** 2 + (Z3[:, 1:] ** 2).sum(1))
        p = float(np.mean((n3 / n1) * q1 / q3 > f))
        row.append(p)
        say(f"alpha={a} lam={lam}: power F-test = {p:.4f} (se {np.sqrt(p*(1-p)/n):.4f}); quadratic (known sigma, exact) = {qp[1 + (a == 0.01)][qlams.index(lam)]:.4f}")
    res[a] = row
with open(os.path.join(HERE, 'ftest_output.txt'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
json.dump(dict(lams=lams, power=res), open(os.path.join(HERE, 'ftest_power.json'), 'w'))
