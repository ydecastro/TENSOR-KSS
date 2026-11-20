"""Remark 1 (rem:bracket), Section 3.3: the constant A of (eq:Adef) and the exponent E = rho^2/(2(A^2 + rho^2)) of
Theorem 2 (thm:exponent) at the six pairs of the table, alpha = 0.05; the maximiser (a, b) T_max at (3,3); the gap
between the column A and its upper bound at d >= 4. These values are also used in Section 3.2 (the power bound of
Corollary 3) and in Section 6 ("The power bound", "The true exponent, measured").

  U(r) = u_alpha(r, lmax(r))/T_max for r psd on the Frobenius sphere ||r||_F = T_max;  A = max_r U(r).
    d = 3 (N = 2): the psd spectra of the sphere are T_max (cos phi, sin phi), 0 <= phi <= pi/4 up to order, so A is a
           maximum in one variable; a scan of 20001 points shows a single interior local maximum, refined by a
           golden-section search in 50-digit arithmetic.
    d >= 4: A_eq  = max over the family T_max (cos phi, sin phi/sqrt(N-1), ..., sin phi/sqrt(N-1)) (same method): the
                    column A of the table, a lower bound on A;
            A_bar = max over a in [1/sqrt(N), 1] of u_alpha(T_max (a, c, ..., c), a T_max)/T_max with
                    c = min(a, sqrt(1-a^2)) (scanned in two parts: a = cos psi, c = sin psi, psi in [0, pi/4]; and
                    c = a on [1/sqrt N, 1/sqrt 2]): an upper bound on A, by Lemma 6 (lem:mono);
            so A_eq <= A <= A_bar; and the best value of a search over the whole sphere (20000 random spectra, then
            Nelder-Mead from the 12 best and 12 random starts; seed 20261116 + 100 k + d), evaluated again at 50 digits.
H_r is the closed form of Appendix F at 50 digits; every maximiser is checked again with the Beta-form quadrature at
80 digits. Scans and searches use the double precision Beta form (Gauss-Legendre, 600 nodes).

Run: python3 deficiency_exponent_constants.py [k d [k d ...]]  (all six pairs by default, about 10 minutes on one
core: (3,3) and (5,3) take 20 seconds each, the four pairs with d >= 4 about 2 minutes each).
Writes deficiency_exponent_constants.json (updated pair by pair), read by power_bound_student.py and
deficiency_vs_exponent.py.
"""
import json
import math
import os
import sys
import time
import numpy as np
import mpmath as mp
from scipy import optimize
import exponent_lib as C

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "deficiency_exponent_constants.json")
DIG = 30
ALPHA = "0.05"
# printed in the paper: A, E (table of Remark 1); A_bar - A at d >= 4 (text of Remark 1)
PAPER = {(3, 3): ("2.312", "0.0615", None), (3, 4): ("1.543", "0.120", "1.3e-3"), (3, 5): ("1.287", "0.156", "4.6e-4"),
         (5, 3): ("1.440", "0.1158", None), (5, 5): ("1.064", "0.1778", "1.9e-6"), (7, 5): ("1.023", "0.1789", "3.6e-8")}


def s(x, n=DIG):
    return mp.nstr(x, n)


def count_local_max(U):
    """strict local maxima of a sampled function (interior), and the number of sign changes of the differences."""
    dU = np.diff(U)
    loc = np.where((dU[:-1] > 0) & (dU[1:] <= 0))[0] + 1
    sg = np.sign(dU[np.abs(dU) > 0])
    changes = int(np.sum(sg[1:] != sg[:-1]))
    return len(loc), changes, float(np.min(np.abs(dU))), loc


def scan_and_refine(family, lo, hi, P, alpha, label, n=20001, power=2):
    """scan F(x) = U(family(x)) on a grid of [lo, hi] denser near lo (x = lo + (hi-lo) (i/n)^power), then refine the
    maximum by golden section at 50 digits. family(x, lib) returns the spectrum in units of T_max."""
    xs = lo + (hi - lo) * (np.arange(n) / (n - 1)) ** power
    S = np.array([family(x, np) for x in xs])
    U = np.concatenate([C.dp_U(S[i:i + 2000], alpha, P) for i in range(0, n, 2000)])
    nmax, changes, mind, loc = count_local_max(U)
    i = int(np.argmax(U))
    print(f"   {label}: scan of {n} points on [{lo:.6g}, {hi:.6g}]: {nmax} interior strict local maximum(s), "
          f"{changes} sign change(s) of the differences, smallest |difference| {mind:.2e}; grid argmax x = {xs[i]:.10g} "
          f"(index {i}), U = {U[i]:.12f}; U at the ends {U[0]:.12f}, {U[-1]:.12f}", flush=True)
    if nmax > 1:
        print("      all interior local maxima of the grid (x, U): " +
              ", ".join(f"({xs[j]:.8g}, {U[j]:.12f})" for j in loc))
    a_, b_ = xs[max(i - 2, 0)], xs[min(i + 2, n - 1)]
    F = lambda x: C.mp_U(family(x, mp), alpha, P)
    if i == 0 or i == n - 1:
        print(f"   {label}: maximum at an end of the interval")
        x = mp.mpf(xs[i]); val = F(x)
    else:
        x, val = C.mp_golden_max(F, a_, b_, mp.mpf(10) ** -22)
        assert val >= F(mp.mpf(xs[i - 1])) and val >= F(mp.mpf(xs[i + 1]))
    return dict(x=x, val=val, nmax=nmax, changes=changes)


def arc(x, lib):
    return [lib.cos(x), lib.sin(x)]


def make_family(N):
    def fam(x, lib):
        b = lib.sin(x) / lib.sqrt(N - 1)
        return [lib.cos(x)] + [b] * (N - 1)
    return fam


def make_bar_hi(N):
    """a = cos psi, c = sin psi <= a (psi in [0, pi/4]): the part a >= 1/sqrt(2) of the A_bar curve."""
    def bar(psi, lib):
        return [lib.cos(psi)] + [lib.sin(psi)] * (N - 1)
    return bar


def make_bar_lo(N):
    """c = a (a in [1/sqrt(N), 1/sqrt(2)]): the part a <= 1/sqrt(2) of the A_bar curve."""
    def bar(a, lib):
        return [a] * N
    return bar


def sphere_search(P, alpha, seed):
    """search of max U over psd spectra of the Frobenius sphere (double precision): random spectra, then
    Nelder-Mead on x in R^N with spectrum |x|/||x||."""
    N = P["N"]
    rng = np.random.default_rng(seed)
    blocks = [np.abs(rng.standard_normal((5000, N))),
              np.exp(rng.uniform(-8, 0, (5000, N))),
              np.c_[np.ones(5000), np.abs(rng.standard_normal((5000, N - 1))) * 0.1],
              np.c_[np.ones(5000), np.exp(rng.uniform(-9, -1, (5000, N - 1)))]]
    X = np.concatenate(blocks)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    U = np.concatenate([C.dp_U(X[i:i + 2000], alpha, P) for i in range(0, len(X), 2000)])
    order = np.argsort(-U)
    best_rand = float(U[order[0]])
    f = lambda x: -C.dp_U((np.abs(x) / np.linalg.norm(x))[None], alpha, P)[0] if np.linalg.norm(x) > 0 else 0.0
    starts = [X[j] for j in order[:12]] + [X[j] for j in rng.choice(len(X), 12, replace=False)]
    best = (-np.inf, None)
    for x0 in starts:
        o = optimize.minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-15, maxiter=4000,
                                                                       maxfev=6000))
        if -o.fun > best[0]:
            v = np.abs(o.x) / np.linalg.norm(o.x)
            best = (-o.fun, np.sort(v)[::-1])
    return best_rand, best[0], best[1]


def run(k, d):
    t0 = time.time()
    alpha = mp.mpf(ALPHA)
    P = C.mp_params(k, d)
    N, T, rho2 = P["N"], P["Tmax_mp"], P["rho2_mp"]
    print(f"== (k,d) = ({k},{d}), alpha = {ALPHA}: N = {N}, m = {P['m']}, kappa = {P['kappa']}, rho^2 = {s(rho2, 12)}, "
          f"T_max = {s(T, 20)}", flush=True)
    E = lambda A: rho2 / (2 * (A * A + rho2))
    R = dict(k=k, d=d, alpha=ALPHA)
    if N == 2:
        r1 = scan_and_refine(arc, 0.0, math.pi / 4, P, alpha, "arc (cos phi, sin phi), phi in [0, pi/4]")
        Aeq, phi = r1["val"], r1["x"]
        spec = [mp.cos(phi), mp.sin(phi)]
        Abar = Aeq
    else:
        fam = make_family(N)
        r1 = scan_and_refine(fam, 0.0, math.atan(math.sqrt(N - 1)), P, alpha,
                             "family (cos phi, sin phi/sqrt(N-1), ...), phi in [0, atan sqrt(N-1)]")
        Aeq, phi = r1["val"], r1["x"]
        spec = fam(phi, mp)
        r2 = scan_and_refine(make_bar_hi(N), 0.0, math.pi / 4, P, alpha,
                             "A_bar, part a >= 1/sqrt 2: (cos psi, sin psi, ..., sin psi), psi in [0, pi/4]")
        r3 = scan_and_refine(make_bar_lo(N), 1 / math.sqrt(N), 1 / math.sqrt(2), P, alpha,
                             "A_bar, part a <= 1/sqrt 2: (a, a, ..., a), a in [1/sqrt N, 1/sqrt 2]", power=1)
        Abar = max(r2["val"], r3["val"])
        abar = mp.cos(r2["x"]) if r2["val"] >= r3["val"] else r3["x"]
        print(f"   A_bar attained at a = {s(abar, 15)} (A_bar curve: max {s(r2['val'], 15)} on a >= 1/sqrt 2, "
              f"{s(r3['val'], 15)} on a <= 1/sqrt 2)")
        br, bs, bv = sphere_search(C.params(k, d), float(alpha), 20261116 + 100 * k + d)
        bv_mp = [mp.mpf(x) for x in bv]
        nrm = mp.sqrt(mp.fsum([x * x for x in bv_mp]))
        val_mp = C.mp_U([x / nrm for x in bv_mp], alpha, P)
        print(f"   whole-sphere search: best of 20000 random spectra U = {br:.10f}; best after Nelder-Mead "
              f"U = {bs:.13f} at spectrum/T_max {np.array2string(bv, precision=7)}; at 50 digits {s(val_mp, 20)}; "
              f"A_eq - best = {s(Aeq - val_mp, 3)}")
        R.update(search_best=s(val_mp, 20), Aeq_minus_search=s(Aeq - val_mp, 3))
    with mp.workdps(80):
        chk = C.mp_U([mp.mpf(x) for x in spec], alpha, P, logH=C.mp_logH_beta)
    print(f"   check: U at the maximiser with the Beta-form quadrature (80 digits) minus closed form: {s(chk - Aeq, 3)}")
    R.update(A=s(Aeq), Abar=s(Abar), E=s(E(Aeq)), E_Abar=s(E(Abar)), a=s(spec[0], 15), b=s(spec[1], 15),
             Abar_minus_A=s(Abar - Aeq, 6), S_column=s(rho2 / (2 * (1 + rho2)), 10))
    pa, pe, pg = PAPER[(k, d)]
    print(f"   A      = {s(Aeq)}" + ("   (maximum over the arc)" if N == 2 else "   (A_eq, family (a,b,...,b))")
          + f"   (paper {pa})")
    if N > 2:
        print(f"   A_bar  = {s(Abar)}   (A_bar - A_eq = {s(Abar - Aeq, 4)}; paper {pg})")
    print(f"   maximiser: spectrum/T_max = ({s(spec[0], 12)}, {s(spec[1], 12)}{', ...' if N > 2 else ''})"
          + ("   (paper at (3,3): (0.995, 0.103))" if (k, d) == (3, 3) else ""))
    print(f"   E      = {s(E(Aeq))}" + ("" if N == 2 else f"   E(A_bar) = {s(E(Abar))}") + f"   (paper {pe})")
    print(f"   column S of the table: rho^2/(2(1+rho^2)) = {s(rho2 / (2 * (1 + rho2)), 8)}")
    assert Aeq <= Abar
    print(f"   [{time.time() - t0:.0f} s]", flush=True)
    return R


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    pairs = list(zip(args[::2], args[1::2])) or C.PAIRS
    res = json.load(open(JSON)) if os.path.exists(JSON) else {}
    for (k, d) in pairs:
        res[f"{k},{d},{ALPHA}"] = run(k, d)
        json.dump(res, open(JSON, "w"), indent=1)
