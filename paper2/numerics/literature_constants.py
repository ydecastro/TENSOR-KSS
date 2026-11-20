"""Constants of the literature at k = 3, written in the variables of the paper.

- Section 1, paragraph "Growing dimension": sup X / sqrt(d) tends to about 1.657 for k = 3. This is the ground-state
  energy E_0(3) of the spherical 3-spin model in the normalisation of the paper. It is computed in two ways: the closed
  form used by Jagannath, Lopatto, and Miolane (E_0 = sqrt(k) (1 + z/k)/sqrt(1 + z), z the positive root of
  (1+z) log(1+z)/z^2 - 1/z = 1/k), and the zero above E_inf = 2 sqrt((k-1)/k) of the annealed complexity of the
  spherical k-spin model.
- Supplementary Material, Appendix E (app:norm), and Section 1 and Remark 3 (rem:dim) of the main paper: the threshold
  of Perry, Wein, and Bandeira (their Theorem 3.2), lambda_PWB^2 = min over t in (0,1) of
  -log(1-t^2) (1+t^k)/t^k, which is 1.5847 for k = 3, that is lambda/sigma = 1.1205 sqrt(d) (main paper: 1.12);
  and the critical value of Jagannath, Lopatto, and Miolane, lambda_c^2 = min over t in (0,1) of
  -(log(1-t) + t)/t^k, which is 1.2066 for k = 3 (main paper: 1.21).
Run: python3 literature_constants.py (a few seconds).
"""
import math
import numpy as np
from scipy import optimize

for k in (3, 5, 7):
    ts = np.linspace(1e-6, 1 - 1e-9, 2000001)
    lc = math.sqrt(np.min(-(np.log1p(-ts) + ts) / ts ** k))
    lp = math.sqrt(np.min(-np.log1p(-ts ** 2) * (1 + ts ** k) / ts ** k))
    zk = optimize.brentq(lambda z: (1 + z) / z ** 2 * math.log1p(z) - 1 / z - 1 / k, 1e-6, 1e6)
    E0_closed = math.sqrt(k) * (1 + zk / k) / math.sqrt(1 + zk)
    Einf = 2 * math.sqrt((k - 1) / k)

    def complexity(u):
        I1 = u / Einf ** 2 * math.sqrt(u * u - Einf ** 2) - math.log(u + math.sqrt(u * u - Einf ** 2)) + math.log(Einf)
        return 0.5 * math.log(k - 1) - (k - 2) / (4.0 * (k - 1)) * u * u - I1

    E0_zero = optimize.brentq(complexity, Einf + 1e-12, 10)
    note = ("   (paper: 1.657; 1.5847 and 1.1205, main paper 1.12; 1.2066, main paper 1.21)" if k == 3 else "")
    print(f"k = {k}: E_0 = {E0_closed:.6f} (closed form), {E0_zero:.6f} (zero of the complexity); "
          f"lambda_PWB = {lp:.4f}, lambda_PWB/sqrt(2) = {lp / math.sqrt(2):.4f}; lambda_c = {lc:.4f}{note}")
