"""Section 7, the estimation sandwich at (3,5): continuous optimisation over theta_1 of the explicit bound of
Theorem 6(ii) (thm:estim), and the widths of the bracket quoted in the text. Reuses the functions of
kacrice_mass_and_sandwich.py.
"""
import numpy as np, sys, os
from scipy import optimize, stats
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kacrice_mass_and_sandwich.py')).read()
head = src.split('# ---------------------------------------------------------------------------------------------')[0]
exec(head)
sand = src.split('print("\\n=== Section 7, \'The estimation sandwich\' at (3,5) ===")')[1].split('rows = [')[0]
exec(sand)
for lam, u in [(20, 16), (20, 18), (40, 34), (40, 36)]:
    kr = KR(k, d, lam); tgt = (d - 1) / (k * lam ** 2)
    f = lambda th1: upper_ii(lam, u, th1, kr)[0]
    lo_b = 0.05 if lam == 20 else 0.03
    r = optimize.minimize_scalar(f, bounds=(lo_b, 0.3), method='bounded', options={'xatol': 1e-5})
    val, info = upper_ii(lam, u, r.x, kr)
    lo = lower_i(lam)
    print(f"lam={lam} u={u}: Thm (ii) optimal theta1={r.x:.5f}: bound {val:.6e}, ratio to target {val/tgt:.5f}, "
          f"varrho={info['varrho']:.5f}, eps_A={info['epsA']:.5f}, Rem_far={info['rem']:.3e}, valid={info['valid']}; ratio to lower (i) {val/lo:.5f}")
# bracket widths with the direct Kac-Rice upper bound (values from a2_output.txt)
direct = {(20, 16): 3.390474e-03, (40, 34): 8.349052e-04}
for (lam, u), up in direct.items():
    lo = lower_i(lam); tgt = (d - 1) / (k * lam ** 2)
    print(f"lam={lam}: lower={lo:.7e} upper={up:.7e}: upper/lower-1 = {100*(up/lo-1):.3f}%, (up-lo)/target = {100*(up-lo)/tgt:.3f}%,"
          f" (up-lo)/up = {100*(up-lo)/up:.3f}%; with table-rounded values: {100*(float(f'{up:.4g}')/float(f'{lo:.4g}' if lam==20 else f'{lo:.3g}')-1):.3f}%")
