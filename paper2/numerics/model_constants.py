"""Model constants used across the paper.

- Section 6, paragraph "The model constants": the table of m = binom(d+k-1, k), kappa = m - d and nu = m - 1
  at (k,d) = (3,3), (3,5), (3,8), (5,6).
- Section 3.1, after Theorem 1 (thm:upper): alpha^(-1/kappa) at (3,3) and (5,5), alpha = 0.05.
- Section 3.2, after Lemma 5 (lem:T2sharp): the bound T_max = sqrt(kappa)/rho on T_2 at (3,3), (3,5), (5,5).
- Section 3.2, after Corollary 4 (cor:Sclosed): c_alpha, the (1-alpha)-quantile of the chi-square law with 2d
  degrees of freedom, at d = 3 and d = 5, alpha = 0.05; and rho^2/(2(1+rho^2)) = k/(2(3k-2)), the column S of the
  table of Remark 1 (rem:bracket).

Here rho^2 = k/(2(k-1)). Run: python3 model_constants.py (a fraction of a second).
"""
from math import comb, sqrt
from scipy import stats

ALPHA = 0.05


def params(k, d):
    m = comb(d + k - 1, k)
    rho2 = k / (2 * (k - 1))
    return m, m - d, m - 1, rho2


print("Section 6, the model constants")
paper = {(3, 3): (10, 7, 9), (3, 5): (35, 30, 34), (3, 8): (120, 112, 119), (5, 6): (252, 246, 251)}
for (k, d), pv in paper.items():
    m, kap, nu, _ = params(k, d)
    print(f"  ({k},{d}): m = {m}, kappa = {kap}, nu = {nu}   (paper {pv[0]}, {pv[1]}, {pv[2]})")

print("Section 3.1, alpha^(-1/kappa) at alpha = 0.05")
for (k, d), pv in {(3, 3): (7, 1.534), (5, 5): (121, 1.025)}.items():
    m, kap, nu, _ = params(k, d)
    print(f"  ({k},{d}): kappa = {kap}, alpha^(-1/kappa) = {ALPHA ** (-1 / kap):.5f}   (paper kappa = {pv[0]}, {pv[1]})")

print("Section 3.2, T_max = sqrt(kappa)/rho")
for (k, d), pv in {(3, 3): 3.06, (3, 5): 6.32, (5, 5): 13.9}.items():
    m, kap, nu, rho2 = params(k, d)
    print(f"  ({k},{d}): T_max = {sqrt(kap / rho2):.4f}   (paper {pv})")

print("Section 3.2, c_alpha = (1-alpha)-quantile of chi^2 with 2d degrees of freedom, alpha = 0.05")
for d, pv in {3: 12.6, 5: 18.3}.items():
    print(f"  d = {d}: c_alpha = {stats.chi2.ppf(1 - ALPHA, 2 * d):.4f}   (paper {pv})")

print("Remark 1, column S: rho^2/(2(1+rho^2)) = k/(2(3k-2))")
for k, pv in {3: 0.2143, 5: 0.1923, 7: 0.1842}.items():
    rho2 = k / (2 * (k - 1))
    print(f"  k = {k}: rho^2/(2(1+rho^2)) = {rho2 / (2 * (1 + rho2)):.5f}, k/(2(3k-2)) = {k / (2 * (3 * k - 2)):.5f}"
          f"   (paper {pv})")
