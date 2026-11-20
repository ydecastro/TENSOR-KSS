"""Section 6, paragraph "The price of knowing sigma": the 0.95-quantile of the reference Student law of H_r against
the Gaussian one, and the point where the two upper tails cross, at (k,d) = (3,5).

The reference density of H_r is the Student density with nu = m - 1 degrees of freedom taken at
u sqrt(nu/kappa) (Section 2.4), that is the law of gamma t_nu with the scale gamma = sqrt(kappa/nu).
Run: python3 reference_student_quantile.py (a fraction of a second).
"""
from math import comb, sqrt
from scipy import optimize, stats

z95 = stats.norm.ppf(0.95)
for (k, d) in [(3, 5), (3, 3), (3, 8), (5, 6)]:
    m = comb(d + k - 1, k)
    kap, nu = m - d, m - 1
    gam = sqrt(kap / nu)
    q = gam * stats.t.ppf(0.95, nu)
    note = "   (paper: 3.4% below at (3,5))" if (k, d) == (3, 5) else ""
    print(f"({k},{d}) m = {m}, kappa = {kap}, nu = {nu}: Gaussian 0.95-quantile {z95:.5f}; "
          f"gamma t_nu 0.95-quantile {q:.5f} ({100 * (q / z95 - 1):+.2f}%){note}")

k, d = 3, 5
m = comb(d + k - 1, k)
kap, nu = m - d, m - 1
gam = sqrt(kap / nu)
u = optimize.brentq(lambda x: stats.t.sf(x / gam, nu) - stats.norm.sf(x), 1.0, 4.0)
print(f"(3,5): the upper tail of gamma t_nu is above the Gaussian one only for u > {u:.4f}   (paper: beyond u = 2.7)")
