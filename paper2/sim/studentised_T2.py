"""Section 7, the studentised second maximum: T2 = lambda_2/sigma_hat <= sqrt(kappa)/rho on every replicate of the
field simulation, and the laws of T2, lmax(Omega)/lambda_2 and lambda_1 on the replicates where the test fails.
Reads the raw files written by run_point.py (seed 777); set RAW to their folder (default ./raw).
"""
import numpy as np, glob, os
from math import comb, sqrt
R = os.environ.get('RAW', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw'))
for f in sorted(glob.glob(os.path.join(R, 'k*_s777.npz'))):
    tag = os.path.basename(f)[:-4]; k = int(tag[1]); d = int(tag[3]); lam = float(tag.split('_')[1][3:])
    m = comb(d + k - 1, k); kap = m - d; rho = sqrt(k / (2 * (k - 1))); bound = sqrt(kap) / rho
    z = np.load(f); T2 = z['lam2c'] / z['sig']; fail = z['fail']
    q = np.quantile(T2, [0.5, 0.99, 1.0])
    line = (f"{tag:28s} n={T2.size:8d} bound={bound:6.3f} T2 max={q[2]:6.3f} q99={q[1]:6.3f} med={q[0]:6.3f} "
            f"frac>bound={np.mean(T2 > bound * (1 + 1e-6)):.2e}")
    if fail.sum() > 0:
        Tf = T2[fail]; ratio = (z['lmaxO'] / z['lam2c'])[fail]
        line += (f" | FAIL n={fail.sum():6d}: T2/bound med={np.median(Tf)/bound:.3f} q10={np.quantile(Tf,0.1)/bound:.3f} "
                 f"lmaxO/lam2 med={np.median(ratio):.3f} q10={np.quantile(ratio,0.1):.3f} lam1/lam med={np.median(z['lam1'][fail])/lam if lam else 0:.3f} "
                 f"lam2/lam med={np.median(z['lam2c'][fail])/lam if lam else 0:.3f}")
    print(line)
