"""Section 6, paragraph "The estimation sandwich": the four rows of the table at (k,d) = (3,5) and the sentences
under it.

For each row (lambda, u): the lower bound of Theorem 4(i); the target (d-1)/(k lambda^2); the Kac-Rice column
int sin^2 theta delta^lambda(u; d theta) + Phi(u - lambda) (paper: 3.390e-3, 2.604e-2, 8.349e-4, 8.666e-4, ratios 1.017,
7.81, 1.0019, 1.0399), its excess over the lower bound (paper: 2.6% at lambda = 20 and 0.40% at lambda = 40) and the
share of Phi(u - lambda) (paper: 3.8% for lambda - u = 4 at lambda = 40); the right-hand side of Theorem 4(ii)
minimised over theta_1 at the printed u, with eps_A in the product form of Lemma C.2 (paper: 4.261e-3, 2.695e-2,
8.934e-4, 9.244e-4, ratios 1.278, 8.09, 1.072, 1.109). For comparison, the same minimum with the additive form
1 + beta*/(rho u) + r_N + eps_det used by kacrice_mass_and_sandwich.py and sandwich_refine.py (version of 2026-10-26).
Rem_far is computed by adaptive quadrature (scipy quad, epsrel 1e-11) with E det in place of E|det|; the last line of
each row gives the effect of E|det| (Monte Carlo, 4000 GOE draws) on Rem_far and on the Kac-Rice integral.
Run: python3 estimation_sandwich.py (about a minute).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, stats, integrate
from estimation_lib import Case

def region(C, u, a):
    th = C.th[C.th >= a]; g = np.abs(C.g(th, u)); keep = th[g > 1e-12 * g.max()]
    return keep.min(), keep.max()

PAPER = {(20, 16): "lower 3.306e-3, Kac-Rice 3.390e-3 (1.017), explicit 4.261e-3 (1.278); 2.6%",
         (20, 18): "lower 3.306e-3, Kac-Rice 2.604e-2 (7.81), explicit 2.695e-2 (8.09)",
         (40, 34): "lower 8.316e-4, Kac-Rice 8.349e-4 (1.0019), explicit 8.934e-4 (1.072); 0.40%",
         (40, 36): "lower 8.316e-4, Kac-Rice 8.666e-4 (1.0399), explicit 9.244e-4 (1.109); Phi costs 3.8%"}
for lam, u in [(20, 16), (20, 18), (40, 34), (40, 36)]:
    C = Case(3, 5, lam)
    res = {}
    for form in ("product", "additive"):
        def f(th1):
            T = C.terms(th1, u)
            mf = T["main_fac"] if form == "product" else (1 + T["beta"] + T["rN"] + T["edet"]) / (1 - T["varrho"]) ** 3
            return C.tgt * mf + C.remfar_quad(th1, u) + stats.norm.cdf(u - lam)
        ths = np.linspace(0.02, 0.32, 61)
        v = [f(t) if C.hyp(t, u) else np.inf for t in ths]
        i = int(np.argmin(v))
        o = optimize.minimize_scalar(f, bounds=(ths[i - 1], ths[i + 1]), method="bounded", options=dict(xatol=1e-6))
        res[form] = (o.fun, o.x, bool(C.hyp(o.x, u)))
    th1 = res["product"][1]
    a, b = region(C, u, th1)
    th = np.linspace(a, b, 401)
    rp = integrate.simpson(C.g(th, u), x=th); rm = integrate.simpson(C.g(th, u, "mc"), x=th)
    a2, b2 = region(C, u, 1e-9); th2 = np.linspace(a2, b2, 401)
    kp = integrate.simpson(C.g(th2, u) * np.sin(th2) ** 2, x=th2); km = integrate.simpson(C.g(th2, u, "mc") * np.sin(th2) ** 2, x=th2)
    kr = C.kacrice(u); lo = C.lower(); t = C.tgt
    T = C.terms(th1, u)
    print(f"lam={lam} u={u}: lower={lo:.5e} ({lo/t:.5f} tgt) target={t:.5e}  KR={kr:.5e} (KR/tgt {kr/t:.5f}, KR/lower-1 {100*(kr/lo-1):.3f}%)"
          f"  Phi(u-lam)/tgt={100*stats.norm.cdf(u-lam)/t:.3f}%")
    print(f"   explicit, product (Lemma C.2): {res['product'][0]:.5e} ratio {res['product'][0]/t:.5f} at theta1={th1:.5f} hyp={res['product'][2]}"
          f"  [varrho={T['varrho']:.5f} beta*/(rho u)={T['beta']:.5f} r_N={T['rN']:.5f} eps_det={T['edet']:.1e}]")
    print(f"   explicit, additive form (version of 2026-10-26): {res['additive'][0]:.5e} ratio {res['additive'][0]/t:.5f} at theta1={res['additive'][1]:.5f}")
    print(f"   E|det|/E det: Rem_far {rm/rp:.5f} (Rem_far {rp:.4e}); KR integral {km/kp:.5f}")
    print(f"   (paper: {PAPER[(lam, u)]})")
