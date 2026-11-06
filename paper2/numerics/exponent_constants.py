"""Constants of Theorem 23 (thm:lower) and of the table of Remark 24 (rem:bracket).

A        = (Lambda(T_max)/alpha)^(1/kappa)           (Corollary 19, Proposition 17)
E*       = rho^2 / (2 (A^2 + rho^2))                  (Remark 21)
A_low    : F(A_low) = alpha,  F(th) = H_{r1}(th T_max)/H_{r1}(T_max), r1 = T_max e1 e1^T
E_low    = rho^2 / (2 (A_low^2 + rho^2))

F is computed three ways: (i) the Beta form (eq:betaform), F = th^{-kappa} J1(th T_max)/J1(T_max), integrand in
log space; (ii) the direct u-integral of H_{r1}; (iii) mpmath in high precision of (i). Also the threshold
alpha^{-1/kappa} and its exponent, and the split of A_low into its Student and determinant parts.
"""
import numpy as np
from scipy import integrate, optimize, special
from math import comb, sqrt, log, exp
import mpmath as mp
import sys, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exponent_constants.out")
fout = open(OUT, "w")
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); fout.write(s + "\n"); fout.flush()

ALPHA = 0.05
PAIRS = [(3,3),(3,4),(3,5),(5,3),(5,5),(7,5)]

def consts(k, d):
    N = d-1; m = comb(d+k-1, k); kappa = m-d; nu = m-1
    rho = sqrt(k/(2*(k-1))); Tmax = sqrt(kappa)/rho
    return N, m, kappa, nu, rho, Tmax

def J_general(ell, eig, kappa, nu, student=True):
    """J_r(ell) = int_0^1 x^{kappa-1} prod(1 - r_i x/ell) (1 + kappa x^2/ell^2)^{-(nu+1)/2} dx, log-space."""
    def f(x):
        if x <= 0.0: return 0.0
        lg = (kappa-1)*log(x)
        pr = 1.0
        for r in eig:
            pr *= (1 - r*x/ell)
        if student:
            lg += -0.5*(nu+1)*np.log1p(kappa*x*x/(ell*ell))
        return pr*exp(lg)
    # x^{kappa-1} peaks at 1: split the interval
    pts = [1 - 1.0/kappa, 1 - 5.0/kappa, 1 - 20.0/kappa]
    pts = sorted(set([p for p in pts if 0 < p < 1]))
    val, err = integrate.quad(f, 0, 1, points=pts, limit=400, epsabs=0, epsrel=1e-13)
    return val

def F_beta(th, k, d, student=True, det=True):
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    eig = [Tmax] + [0.0]*(N-1) if det else [0.0]*N
    return th**(-kappa) * J_general(th*Tmax, eig, kappa, nu, student) / J_general(Tmax, eig, kappa, nu, student)

def H_direct(ell, k, d):
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    def f(u):
        return (u - Tmax) * u**(N-1) * exp(-0.5*(nu+1)*np.log1p(u*u/kappa))
    val, err = integrate.quad(f, ell, np.inf, limit=400, epsabs=0, epsrel=1e-12)
    return val

def F_direct(th, k, d):
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    return H_direct(th*Tmax, k, d) / H_direct(Tmax, k, d)

def F_mp(th, k, d):
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    mp.mp.dps = 40
    T = mp.mpf(kappa)**0.5 / mp.sqrt(mp.mpf(k)/(2*(k-1)))
    def J(ell):
        return mp.quad(lambda x: x**(kappa-1)*(1 - T*x/ell)*(1 + kappa*x*x/(ell*ell))**(-(nu+1)/mp.mpf(2)), [0, 1])
    return float(mp.mpf(th)**(-kappa) * J(th*T) / J(T))

def Lambda(ell, k, d):
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    vals = []
    for j in range(N+1):
        def f(x, j=j):
            if x <= 0: return 0.0
            return exp((kappa+j-1)*log(x) + (N-j)*np.log1p(-x) - 0.5*(nu+1)*np.log1p(kappa*x*x/(ell*ell))) if x < 1 else 0.0
        pts = sorted(set([p for p in [1-1.0/kappa, 1-5.0/kappa, 1-20.0/kappa] if 0 < p < 1]))
        v, e = integrate.quad(f, 0, 1, points=pts, limit=400, epsabs=0, epsrel=1e-13)
        vals.append(1.0/((kappa+j)*v))
    return max(vals), int(np.argmax(vals))

P("# exponent_constants.py -- alpha =", ALPHA)
P("# columns: (k,d) kappa nu Tmax | A E* | A_low(beta) A_low(direct) A_low(mp) E_low | ratio | alpha^-1/k E(crude) | A_low(Student only) A_low(det only)")
results = {}
for (k, d) in PAIRS:
    N, m, kappa, nu, rho, Tmax = consts(k, d)
    Lam, jstar = Lambda(Tmax, k, d)
    A = (Lam/ALPHA)**(1.0/kappa)
    Estar = rho**2/(2*(A**2 + rho**2))
    # A_low by Beta form
    g = lambda th: F_beta(th, k, d) - ALPHA
    Alow = optimize.brentq(g, 1.0+1e-12, 50.0, xtol=1e-12)
    Alow_dir = optimize.brentq(lambda th: F_direct(th, k, d) - ALPHA, 1.0+1e-12, 50.0, xtol=1e-12)
    Alow_mp = optimize.brentq(lambda th: F_mp(th, k, d) - ALPHA, 1.0+1e-12, 50.0, xtol=1e-10)
    Elow = rho**2/(2*(Alow**2 + rho**2))
    crude = ALPHA**(-1.0/kappa)
    Ecrude = rho**2/(2*(crude**2 + rho**2))
    Alow_stud = optimize.brentq(lambda th: F_beta(th, k, d, student=True, det=False) - ALPHA, 1.0+1e-12, 50.0, xtol=1e-12)
    Alow_det = optimize.brentq(lambda th: F_beta(th, k, d, student=False, det=True) - ALPHA, 1.0+1e-12, 50.0, xtol=1e-12)
    results[(k,d)] = dict(kappa=kappa, nu=nu, N=N, rho=rho, Tmax=Tmax, Lam=Lam, A=A, Estar=Estar, Alow=Alow, Elow=Elow)
    P(f"({k},{d}) kappa={kappa:3d} nu={nu:3d} Tmax={Tmax:6.3f} | Lambda(Tmax)={Lam:.6g} (j*={jstar}) A={A:.4f} E*={Estar:.4f} | "
      f"A_low={Alow:.4f} {Alow_dir:.4f} {Alow_mp:.4f} E_low={Elow:.4f} | E_low/E*={Elow/Estar:.3f} | "
      f"alpha^-1/k={crude:.4f} E_crude={Ecrude:.4f} | A_low[Student only]={Alow_stud:.4f} A_low[det only]={Alow_det:.4f}")

P("")
P("# consistency A_low <= A (must hold by eq:ratiobound):")
for kd, r in results.items():
    P(f"  {kd}: A_low={r['Alow']:.4f} <= A={r['A']:.4f} : {r['Alow'] <= r['A']}   E*={r['Estar']:.4f} <= E_low={r['Elow']:.4f} : {r['Estar'] <= r['Elow']}")

P("")
P("# F properties at (3,3): F(1), monotonicity on [1,6], F(large)")
k, d = 3, 3
ths = np.linspace(1, 6, 51)
Fv = np.array([F_beta(t, k, d) for t in ths])
P("  F(1) =", Fv[0], " (direct:", F_direct(1.0, k, d), ")")
P("  strictly decreasing on grid:", bool(np.all(np.diff(Fv) < 0)))
P("  F(6) =", Fv[-1], " F(50) =", F_beta(50.0, k, d))
N, m, kappa, nu, rho, Tmax = consts(k, d)
P("  H_{r1}(Tmax) direct (positive, finite):", H_direct(Tmax, k, d))
P("  F(th) > th^-kappa on grid (so A_low > alpha^-1/kappa):", bool(np.all(Fv[1:] > ths[1:]**(-kappa))))

P("")
P("# Section 7: deficiency / exp(-E_low lambda^2) at every lambda of the simulation (sim/*.jsonl)")
import json
SIMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim")
pool = {}
for fn in ("results.jsonl", "results_rows_33_34.jsonl"):
    for line in open(os.path.join(SIMDIR, fn)):
        if line.strip():
            r = json.loads(line)
            if r["lam"] > 0:
                key = (r["k"], r["d"], r["lam"])
                n0, f0 = pool.get(key, (0, 0))
                pool[key] = (n0 + r["n"], f0 + r["fail"])
table = {kd: {(int(l) if l == int(l) else l): f / n for (k_, d_, l), (n, f) in sorted(pool.items()) if (k_, d_) == kd}
         for kd in [(3, 3), (3, 4), (3, 5), (5, 3)]}
Elow_paper = {(3,3):0.0618,(3,4):0.1199,(3,5):0.1558,(5,3):0.1158}
for kd, row in table.items():
    El = results[kd]['Elow']; Ep = Elow_paper[kd]
    s = f"  {kd} E_low(mine)={El:.4f} E_low(paper)={Ep}: "
    for lam, dfc in row.items():
        s += f" l={lam}: {dfc/exp(-El*lam*lam):.2f}({dfc/exp(-Ep*lam*lam):.2f})"
    P(s)
    lams = np.array(sorted(row)); y = -np.log(np.array([row[l] for l in lams]))
    slope = np.polyfit(lams**2, y, 1)[0]
    P(f"     regression slope of -log(def) on lambda^2 over the row: {slope:.4f};  local slopes: " +
      " ".join(f"{(y[i+1]-y[i])/(lams[i+1]**2-lams[i]**2):.4f}" for i in range(len(lams)-1)))

P("")
P("# Bracket check: measured slopes as printed in the paper vs [E*, E_low]")
meas = {(3,3):(0.065,0.002),(3,4):(0.115,0.012),(3,5):(0.15,0.01),(5,3):(0.124,0.003)}
for kd,(mu,se) in meas.items():
    r = results[kd]
    P(f"  {kd}: measured {mu}+-{se}; bracket [{r['Estar']:.4f},{r['Elow']:.4f}]; (meas-E_low)/E_low = {(mu-r['Elow'])/r['Elow']*100:+.1f}%  ({(mu-r['Elow'])/se:+.1f} sigma)")

P("")
P("# x*, y* of the optimal configuration (for the data check)")
for kd, r in results.items():
    xs = r['Alow']**2/(r['Alow']**2 + r['rho']**2); ys = xs/r['Alow']
    P(f"  {kd}: A_low={r['Alow']:.4f} x*={xs:.4f} y*={ys:.4f}  (1-x*)={1-xs:.4f}  rho*y*={r['rho']*ys:.4f}")
fout.close()
print("written", OUT)
