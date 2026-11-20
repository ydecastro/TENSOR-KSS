"""Section 6, paragraph "The estimation sandwich", the sentences on the range of usefulness of Theorem 4(ii), read
from the stored scans estimation_range_scan.jsonl and estimation_kacrice_scan.jsonl.

- The explicit bound of (ii), optimised over (theta_1, u), divided by the target (d-1)/(k lambda^2): the signal from
  which it is below 2 and below 1.1 (paper: lambda about 11, 13, 13, 16 and 26, 28, 34, 45 at (3,3), (5,3), (3,5),
  (3,8)), by linear interpolation between the two grid points that bracket the level; its value at lambda = 10
  (paper: 4 to 46 times the target) and the split of the excess over the target between the terms of (ii).
- The Kac-Rice bound, minimised over u with E|det| by Monte Carlo, divided by the lower bound (i): the signal from which
  it is within 10% (paper: from lambda about 9, and 10.4 at (3,8)).
Run: python3 estimation_thresholds.py (a second).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "estimation_range_scan.jsonl"))]
kr = [json.loads(l) for l in open(os.path.join(HERE, "estimation_kacrice_scan.jsonl"))]


def share(r):
    """split of the excess of the explicit bound over the target, in units of the target."""
    t = r["tgt"]; mf = r["x_main_fac"]; d = r["d"]
    lr = -(d + 1) / 2 * np.log(1 - r["x_varrho"]); ln = np.log1p(r["x_rN"]); lb = np.log1p(r["x_beta"] + r["x_edet"])
    tot = lr + ln + lb; ex = mf - 1
    return {"varrho": ex * lr / tot, "r_N": ex * ln / tot,
            "beta*/(rho u)": ex * lb / tot * r["x_beta"] / (r["x_beta"] + r["x_edet"]),
            "eps_det": ex * lb / tot * r["x_edet"] / (r["x_beta"] + r["x_edet"]),
            "Rem_far": r["x_rem"] / t, "Phi(u-lam)": r["x_Phi"] / t}


def crossing(lam, ratio, lev):
    i = int(np.argmax(ratio < lev))
    if ratio[i] >= lev:
        return None
    return lam[i - 1] + (ratio[i - 1] - lev) / (ratio[i - 1] - ratio[i]) * (lam[i] - lam[i - 1])


PAPER = {(3, 3): (11, 26), (3, 5): (13, 34), (5, 3): (13, 28), (3, 8): (16, 45)}
for kd in [(3, 3), (3, 5), (5, 3), (3, 8)]:
    R = sorted([r for r in rows if (r["k"], r["d"]) == kd], key=lambda r: r["lam"])
    lam = np.array([r["lam"] for r in R])
    ex = np.array([r["x_val"] / r["tgt"] if "x_val" in r else np.inf for r in R])
    print(f"{kd}: explicit bound / target decreasing on the grid: {bool(np.all(np.diff(ex[np.isfinite(ex)]) < 0))}")
    for lev in (2, 1.5, 1.1):
        l0 = crossing(lam, ex, lev)
        print(f"   explicit / target < {lev} from lambda = {l0:.1f}" if l0 else f"   below {lev}: not reached by lambda = 100")
    print(f"   (paper: within a factor 2 from lambda about {PAPER[kd][0]}, within 10% from about {PAPER[kd][1]})")
    for r in R:
        if r["lam"] in (10, 20) and "x_val" in r:
            p = share(r)
            print(f"   lambda = {r['lam']:4.1f}: explicit / target = {r['x_val'] / r['tgt']:7.3f}; excess over the target: "
                  + ", ".join(f"{a} {b:.3f}" for a, b in p.items()))
    K = sorted([r for r in kr if (r["k"], r["d"]) == kd], key=lambda r: r["lam"])
    kl = np.array([r["lam"] for r in K]); kv = np.array([r["kr_mc_over_lower"] for r in K])
    l0 = crossing(kl, kv, 1.1)
    print("   Kac-Rice (E|det|) / lower bound (i): " + ", ".join(f"lambda = {a:g}: {b:.4f}" for a, b in zip(kl, kv))
          + f"; within 10% from lambda = {l0:.1f}")
print("(paper: at lambda = 10 the explicit bound is 4 to 46 times the target; the Kac-Rice bound is within 10% of the "
      "lower bound from lambda about 9, and 10.4 at (3,8))")
