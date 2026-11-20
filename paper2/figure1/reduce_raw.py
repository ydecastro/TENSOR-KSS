"""Figure 1 (fig:intro): reduce the per-replicate output of run_tests.py (one .npz per point, about 630 MB in all,
not in the repository) to the two small files that make_figure.py and figure_numbers.py read:

  figure1_data.json      the rejection rates of every test at alpha = 0.05 and 0.01 under the null (panel (a) and the
                         levels quoted in Section 6), the power at alpha = 0.05 of every test at every simulated
                         lambda of (3,8) (panel (b)), and the summary lines of every estimation point (panel (c));
  figure1_null_ecdf.npz  for panel (a), the sorted p-values of each test on the 1e5 null replicates at the ranks
                         1, 50, 100, ..., 100000 (2001 of the 100000 vertices of the empirical distribution function).

The p-values are computed exactly as in the original figure script: T, S and the F-test in closed form; the other
tests by Monte Carlo against the 1e6 null replicates of seed 777 (proportion of null values at least as large); the
most powerful invariant test against the null values of its own statistic at the same lambda (at lambda = 6 for
panel (a)).
Run: python3 reduce_raw.py RAW   (RAW: folder with the .npz files and results.jsonl written by production.sh).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RAW", os.path.join(HERE, "raw"))
ALPHA = 0.05
LAMS = list(range(1, 13))
POINTS_B = [(2, "k3d8_lamsq2"), (4, "k3d8_lam2"), (6, "k3d8_lamsq6"), (8, "k3d8_lamsq8"), (10, "k3d8_lamsq10"),
            (12, "k3d8_lamsq12"), (14, "k3d8_lamsq14"), (16, "k3d8_lam4")]
CALIB_TAGS = ["calib_k3d8_null_1e6", "calib_k3d8_null_1e6_b"]
ORACLE_H0_LAM = 6
TESTS = ["T", "S", "F", "max", "studmax", "plugin", "unfold", "oracle"]
THIN = 50


def load(tag):
    return np.load(os.path.join(PROD, tag + ".npz"))


def tail_p(calib_sorted, x):
    """Monte Carlo p-value of a right-tail statistic: proportion of null values >= x."""
    n = calib_sorted.size
    return 1.0 - np.searchsorted(calib_sorted, x, side="left") / n


def load_calib():
    """The null calibration set (seed 777, identical replicates in both files) with every oracle column."""
    out = {}
    for tag in CALIB_TAGS:
        f = os.path.join(PROD, tag + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            for k in z.files:
                out.setdefault(k, z[k])
    return out


def oracle_key(z, lam):
    keys = [k for k in (z.files if hasattr(z, "files") else z.keys()) if k.startswith("logR_")]
    for k in keys:
        if abs(float(k[5:]) - lam) < 1e-4:
            return k
    return None


def pvalues(z, calib, lam=None):
    """p-values of every test on the replicates z (npz), calibration set calib (dict of arrays)."""
    p = {"T": z["pT"], "S": z["pS"], "F": z["pF"]}
    c_lam1 = np.sort(calib["lam1"].astype(float))
    c_T1 = np.sort(calib["T1"].astype(float))
    c_unf = np.sort(calib["unfold"].astype(float))
    p["max"] = tail_p(c_lam1, z["lam1"].astype(float))
    p["studmax"] = tail_p(c_T1, z["T1"].astype(float))
    p["plugin"] = tail_p(c_lam1, z["T1"].astype(float))
    p["unfold"] = tail_p(c_unf, z["unfold"].astype(float))
    if lam is not None:
        kz, kc = oracle_key(z, lam), oracle_key(calib, lam)
        if kz and kc:
            c_or = np.sort(calib[kc].astype(float))
            p["oracle"] = tail_p(c_or, z[kz].astype(float))
    return p


def power_table(calib):
    """Powers at every point of (3,8): the integer grid and the points of panel (b), keyed by lambda."""
    rows = {}
    pts = [(float(l), "k3d8_lam%d" % l) for l in LAMS] + [(np.sqrt(sq), tag) for sq, tag in POINTS_B]
    for lam, tag in pts:
        if lam in rows or not os.path.exists(os.path.join(PROD, tag + ".npz")):
            continue
        z = load(tag)
        p = pvalues(z, calib, lam=lam)
        rows[lam] = {name: float((v <= ALPHA).mean()) for name, v in p.items()}
        rows[lam]["n"] = int(z["pT"].size)
        rows[lam]["tag"] = tag
    return rows


def main():
    calib = load_calib()
    # panel (a): the null replicates of seed 778
    z = load("panel_k3d8_null_1e5")
    p = pvalues(z, calib, lam=ORACLE_H0_LAM)
    n = p["T"].size
    ranks = np.unique(np.r_[1, np.arange(THIN, n + 1, THIN)])
    ecdf = {"n": np.array(n), "rank": ranks}
    levels = {}
    for name in TESTS:
        v = np.sort(p[name])
        ecdf["p_" + name] = v[ranks - 1]
        levels[name] = {"0.05": float((p[name] <= 0.05).mean()), "0.01": float((p[name] <= 0.01).mean())}
    np.savez_compressed(os.path.join(HERE, "figure1_null_ecdf.npz"), **ecdf)
    # panel (b)
    rows = power_table(calib)
    power = [dict(lam=float(l), **v) for l, v in sorted(rows.items())]
    # panel (c): the summary lines of the estimation points
    est = []
    for line in open(os.path.join(PROD, "results.jsonl")):
        r = json.loads(line)
        if r["tag"].startswith(("calib", "panel")):
            continue
        est.append({k: r[k] for k in ("tag", "k", "d", "lam", "n", "seed", "mean_sin2_argmax", "mean_sin2_svd",
                                       "se_sin2_argmax", "se_sin2_svd")})
    out = dict(null_n=int(n), null_levels=levels, power_k3d8=power, estimation=est)
    with open(os.path.join(HERE, "figure1_data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"null replicates {n}; levels at 0.05: " + ", ".join(f"{t} {levels[t]['0.05']:.4f}" for t in TESTS))
    print(f"{len(power)} power points at (3,8); {len(est)} estimation points")


if __name__ == "__main__":
    main()
