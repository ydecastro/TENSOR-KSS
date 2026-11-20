"""The numbers of the caption of Figure 1 (fig:intro) and of the paragraph "The figure of the introduction" of
Section 6, read from figure1_data.json (written by reduce_raw.py from the field simulation).
Run: python3 figure_numbers.py (a fraction of a second).
"""
import os
import json
import numpy as np
from make_figure import lower_bound_risk

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "figure1_data.json")))

print(f"(a) under lambda = 0, (3,8), {data['null_n']} replicates: rejection rates at nominal 0.05 and 0.01")
for t, v in data["null_levels"].items():
    print(f"   {t:8s} {v['0.05']:.4f}  {v['0.01']:.4f}")
print("   (paper: sup Z/sigma_hat with the quantile of sup Z/sigma ('plugin') rejects with probability 0.097 at 0.05 "
      "and 0.026 at 0.01)")

P = sorted(data["power_k3d8"], key=lambda r: r["lam"])
print(f"(b) power at alpha = 0.05, (3,8), {P[0]['n']} replicates per point:")
names = ["T", "S", "max", "studmax", "oracle", "F", "unfold"]
print("   lambda   lambda^2 " + " ".join(f"{n:>8s}" for n in names))
for r in P:
    print(f"   {r['lam']:7.4f} {r['lam'] ** 2:7.2f}  " + " ".join(f"{r.get(n, np.nan):8.4f}" for n in names))
with_or = [r for r in P if "oracle" in r]
d1 = max(r["oracle"] - r["studmax"] for r in with_or)
d2 = max(r["unfold"] - r["T"] for r in P)
gap = [(max(r["oracle"], r["studmax"]) - r["T"], r["lam"]) for r in with_or]
d3, l3 = max(gap)
d4 = max(max(r["S"], r["max"]) - r["T"] for r in P)
F4 = [r["F"] for r in P if abs(r["lam"] - 4) < 1e-9][0]
F12 = [r["F"] for r in P if abs(r["lam"] - 12) < 1e-9][0]
print(f"   most powerful invariant test minus studentised maximum: at most {d1:.4f} (paper 0.011)")
print(f"   unfolding test minus T: at most {d2:.4f} (paper 0.011)")
print(f"   T below the larger of these two by at most {d3:.4f}, at lambda = {l3:g} (paper 0.08, at lambda = 5)")
print(f"   tests with sigma known (S, maximum test) minus T: at most {d4:.4f} (paper 0.16)")
print(f"   locally optimal invariant test: power {F4:.4f} at lambda = 4 and {F12:.4f} at lambda = 12 (paper 0.20 and 0.94)")

print("(c) risk divided by the lower bound of Theorem 4(i)")
E = data["estimation"]
for lam in (0.125, 10, 20):
    for (k, d) in [(3, 3), (3, 5), (3, 8), (5, 3)]:
        rr = [r for r in E if (r["k"], r["d"], r["lam"]) == (k, d, lam)]
        if not rr:
            continue
        r = rr[0]
        lb = lower_bound_risk(np.array(lam, float), d, k)
        print(f"   ({k},{d}) lambda = {lam:g}: argmax {r['mean_sin2_argmax'] / lb:.4f} (+- {2 * r['se_sin2_argmax'] / lb:.4f}),"
              f" unfolding {r['mean_sin2_svd'] / lb:.4f} (+- {2 * r['se_sin2_svd'] / lb:.4f})  [n = {r['n']}]")
print("   (paper: at lambda = 10 the argmax is within 10% of the lower bound at the four pairs and the unfolding estimator"
      " is above it by 7 to 24%; at lambda = 20 and (3,8), 2.5% and 5.5%; the ratio tends to one at both ends)")
