"""Recomputation of numbers of Sections 3 to 6.

1. Lemma 22 (lem:Phibounds) and Remark 21 (rem:Estar): A, E = I(1), D_A, c_near, c_far, K(c_far)/2,
   K(c_near)/2, w_1, the minimum of I on a grid, and a direct check of the lemma on a grid, at the six pairs.
2. A_low (Theorem 23), E_low, the balance exponent (A_0), the percentages of Section 4.
3. Remark 9 (rem:tuple) with the Rice constant of eq:ricebound: Pbar, q_eta, q_tau, E, share, u_dagger,
   u_+, power bound.
4. F-test at (3,3): F_{3,7} quantile, local coefficient C_F, powers at lambda = 1.4 to 5 (exact).
5. Exact power of the quadratic test at lambda = 5.
6. The bounds sqrt(kappa)/rho on T_2; rho u_+.
"""
import numpy as np
from scipy import integrate, special, stats, optimize
from math import comb, gamma, sqrt, pi, log, exp

def rho_of(k): return sqrt(k/(2*(k-1)))

# ---------- 1. Phibounds ----------
def Lambda(ell, kap, N, nu):
    best = 0.0
    for j in range(N+1):
        f = lambda x: x**(kap+j-1)*(1-x)**(N-j)*(1+kap*x*x/ell**2)**(-(nu+1)/2)
        val, _ = integrate.quad(f, 0, 1, epsabs=0, epsrel=1e-12, limit=200)
        best = max(best, 1.0/((kap+j)*val))
    return best

def I_of(c, k, A):
    rho = rho_of(k)
    V = 1 - c**(2*k) - k*c**(2*k-2)*(1-c*c)
    K = k*c**(2*k-2)*(1-c*c)
    t1 = max(1 - c**k - K, 0.0)**2/(2*((1-c**k)**2 + V)) if (1-c**k)**2 + V > 0 else 0.0
    t2 = rho**2*max(c**k - A*sqrt(max(V,0))/rho, 0.0)**2/(2*(A*A+rho*rho))
    return K/2 + max(t1, t2)

pairs = [(3,3),(3,4),(3,5),(5,3),(5,5),(7,5)]
alpha = 0.05
print("=== 1. cor:def A, lem:Phibounds table")
print("(k,d)  kap   A      E=I(1)   D       c_near   c_b      K(c_b)/2  K(c_near)/2  w_1     minI(grid)  minI@c<=0.9")
Avals = {}
for (k,d) in pairs:
    N = d-1; m = comb(d+k-1,k); kap = m-d; nu = m-1; rho = rho_of(k)
    Tmax = sqrt(kap)/rho
    A = (Lambda(Tmax, kap, N, nu)/alpha)**(1/kap)
    Avals[(k,d)] = A
    E = rho**2/(2*(A*A+rho*rho))
    D = k/2 + A*(k-1)
    dA = A*(A-1)/(A*A+rho*rho)
    cnear = sqrt(1-1/D); cb = (1-4*E)**(1/k)
    Kc = lambda c: k*c**(2*k-2)*(1-c*c)/2
    den = k*(k-1) - 2*E*D*D
    w1 = k*dA/den if den > 0 else float('inf')
    cs = np.linspace(-1, 1, 400001)
    Is = np.array([I_of(c,k,A) for c in cs])
    minI = Is.min(); minI09 = Is[cs<=0.9].min()
    print(f"({k},{d})  {kap:3d}  {A:.3f}  {E:.5f}  {D:.3f}  {cnear:.4f}  {cb:.4f}  {Kc(cb):.4f}    {Kc(cnear):.4f}      {w1:.2f}   {minI:.5f}     {minI09:.4f}")
    # lemma (i)-(iii) grid checks
    bad_i = [c for c in cs[cs<=0] if I_of(c,k,A) < 0.25-1e-12]
    bad_ii = [c for c in cs[(cs>=0)&(cs<1)] if I_of(c,k,A) < max(Kc(c), (1-c**k)/4)-1e-12]
    ws = np.linspace(0, 1/D, 2001)[1:]
    bad_iii = [w for w in ws if I_of(sqrt(1-w),k,A) - E < w*(k/2*dA - (k*(k-1)/2 - E*D*D)*w) - 1e-12]
    print(f"      lemma checks: (i) viol {len(bad_i)}, (ii) viol {len(bad_ii)}, (iii) viol {len(bad_iii)};  I(1)={I_of(1.0,k,A):.5f}, I(0)={I_of(0.0,k,A):.4f}, I(-1)={I_of(-1.0,k,A):.4f}")

# ---------- 2. A_low, E_low, balance ----------
print("\n=== 2. A_low, E_low, balance A_0 and exponent")
for (k,d) in pairs:
    N = d-1; m = comb(d+k-1,k); kap = m-d; nu = m-1; rho = rho_of(k); Tmax = sqrt(kap)/rho
    def J1(ell):
        f = lambda x: x**(kap-1)*(1-Tmax*x/ell)*(1+kap*x*x/ell**2)**(-(nu+1)/2)
        return integrate.quad(f, 0, 1, epsabs=0, epsrel=1e-12, limit=200)[0]
    F = lambda th: th**(-kap)*J1(th*Tmax)/J1(Tmax)
    Alow = optimize.brentq(lambda th: F(th)-alpha, 1.0000001, 20)
    Elow = rho**2/(2*(Alow**2+rho**2))
    A0 = comb(kap+N,N)**(1/kap)*alpha**(-1/kap)
    bal = rho**2/(2*(rho+A0)**2)
    Es = rho**2/(2*(1+rho**2))
    print(f"({k},{d})  A_low={Alow:.3f}  E_low={Elow:.4f}  A_0={A0:.3f}  balance={bal:.4f} ({100*bal/0.5:.1f}% of 1/2)  E*={rho**2/(2*(Avals[(k,d)]**2+rho**2)):.4f}  ratio E_low/E*={Elow/(rho**2/(2*(Avals[(k,d)]**2+rho**2))):.3f}  S={Es:.4f}  alpha^(-1/kap)={alpha**(-1/kap):.3f}")

# ---------- 3. rem:tuple ----------
print("\n=== 3. rem:tuple at (3,5)")
k,d = 3,5; N = d-1; m = comb(d+k-1,k); kap = m-d; nu = m-1; rho = rho_of(k)
eta, tau, b0, v, lam = 0.8, 9.0, 6.0, 50.0, 150.0
vm = (1-eta)*v; mu = (k-1)*b0/lam; beta = sqrt(k*(k-1)/2)*b0/lam
print(f"v_-={vm}, mu_b0={mu:.3f}, beta_b0={beta:.4f}, rho v_-={rho*vm:.3f} (>= {d-1}), rho(v_- - mu)={rho*(vm-mu):.3f} (>= {sqrt(2*(d-1)):.3f}), rho v_-/4={rho*vm/4:.3f}, beta+sqrt(2N)={beta+sqrt(2*N):.3f} <= rho tau={rho*tau:.3f} < rho v_-={rho*vm:.3f}")
CR = 43*sqrt(pi)*k**3*2**(2*k)*(k-1)**((d-2)/2)/gamma((d-1)/2)
CR_old = 13*sqrt(pi)*k**4*2**(5*k/2)*(k-1)**((d-2)/2)/gamma((d-1)/2)
print(f"C_Rice new = {CR:.4e}, old = {CR_old:.4e}, ratio new/old = {CR/CR_old:.4f}")
y = vm - mu
Pbar = CR*(1+rho*y)**(d-1)/(rho*y)*exp(-rho**2*y**2/2)
p_ = (rho*vm)**2
Ckd = 2*(9/4 + 5/6*sqrt(d-1))**(d-1)
# q_eta: sup over c in cap of P{|sigma_hat-1|>eta}, kap*sigma_hat^2 ~ chi2_kap(delta^2), delta^2 <= beta^2
def q_eta(delta2):
    lo = stats.ncx2.cdf(kap*(1-eta)**2, kap, delta2)
    hi = stats.ncx2.sf(kap*(1+eta)**2, kap, delta2)
    return lo+hi
qe = max(q_eta(0.0), q_eta(beta**2))
# q_tau: sup over c of 2 exp(-(rho tau - |beta2(c)| - sqrt(2N))^2/2), |beta2| <= beta_b0
qt = 2*exp(-0.5*(rho*tau - beta - sqrt(2*N))**2)
E = Ckd*(2*Pbar**(1-1/p_) + qe**(1-1/p_) + qt**(1-1/p_))
share = Ckd*qt**(1-1/p_)/E
print(f"Pbar={Pbar:.3e}, q_eta={qe:.3e}, q_tau={qt:.3e}, p_-={p_:.1f}, C_kd={Ckd:.2f}, E={E:.4e}, share q_tau={100*share:.2f}%")
print(f"power bound (1-E)*0.99876 = {(1-E)*0.99876:.5f}; rho*u_+ = {rho*109.7:.2f}")
# u_dagger: sup over ||r||<=tau, |s-1|<=eta of u_alpha(r/s); claimed at r=tau Id, s=1-eta
def H(r_eigs, ell):
    gam = sqrt(kap/(m-1))
    f = lambda u: np.prod([u-ri for ri in r_eigs])*stats.t.pdf(u/gam, m-1)/gam
    return integrate.quad(f, ell, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
def u_alpha(r_eigs, v):
    target = alpha*H(r_eigs, v)
    return optimize.brentq(lambda u: H(r_eigs,u)-target, v, 10*v)
ud = u_alpha([tau/(1-eta)]*N, v)
print(f"u_dagger (r=tau Id, s=1-eta) = {ud:.2f}, u_+ = (1+eta) u_dagger = {(1+eta)*ud:.1f}")
for s in [0.2, 0.5, 1.0, 1.8]:
    for r in [tau, -tau, 0.0]:
        print(f"   check s={s}, r={r:+.0f} Id: u_alpha={u_alpha([r/s]*N, v):.2f}")

# ---------- 4. F-test at (3,3) ----------
print("\n=== 4. F-test at (3,3)")
k,d = 3,3; m = 10
mu1, mu3 = 1/5, 2/35; n1, n3 = 3, 7
print(f"check sum mu_j n_j = {mu1*n1+mu3*n3:.6f}")
for a in [0.05, 0.01]:
    f_a = stats.f.ppf(1-a, 3, 7)
    EQ1 = n1*stats.f.sf(3*f_a/5, 5, 7)
    EQ3 = n3*stats.f.sf(9*f_a/7, 3, 9)
    CF = 0.5*(mu1*EQ1 + mu3*EQ3 - a)
    print(f"alpha={a}: f_alpha={f_a:.4f}, C_F={CF:.4e}, ratio to C_opt (3.655e-2 / 1.081e-2) = {CF/(3.655e-2 if a==0.05 else 1.081e-2):.3f}, to C_max (2.959e-2 / 8.398e-3) = {CF/(2.959e-2 if a==0.05 else 8.398e-3):.3f}")
f_a = stats.f.ppf(0.95, 3, 7)
def power_F(lam):
    nc1, nc3 = 0.6*lam**2, 0.4*lam**2
    # P{(7/3) Q1/Q3 > f} = int P{Q1 > 3 f q/7} f_Q3(q) dq
    g = lambda q: stats.ncx2.sf(3*f_a*q/7, n1, nc1)*stats.ncx2.pdf(q, n3, nc3)
    return integrate.quad(g, 0, np.inf, epsabs=1e-12, epsrel=1e-10, limit=400)[0]
print("F-test power (exact):", ", ".join(f"lam={l}: {power_F(l):.4f}" for l in [1.4,1.6,1.8,2.0,3.0,5.0]))

# ---------- 5. quadratic test exact power ----------
print("\n=== 5. quadratic test T = mu1 chi2_3 + mu3 chi2_7 at (3,3), alpha=0.05")
def sf_T(t, lam):
    nc1, nc3 = mu1*n1*lam**2, mu3*n3*lam**2
    g = lambda q: stats.ncx2.sf((t-mu3*q)/mu1, n1, nc1)*stats.ncx2.pdf(q, n3, nc3) if t-mu3*q > 0 else stats.ncx2.pdf(q, n3, nc3)
    return integrate.quad(g, 0, np.inf, epsabs=1e-12, epsrel=1e-10, limit=400)[0]
t_a = optimize.brentq(lambda t: sf_T(t, 0.0)-0.05, 0.5, 10)
print(f"t_alpha={t_a:.4f} (paper 2.022)")
print("quadratic power:", ", ".join(f"lam={l}: {sf_T(t_a,l):.4f}" for l in [1.4,1.6,1.8,2.0,3.0,5.0]))
print(f"S at lam=5 from simulation: 0.96189 +- 0.00027; difference to quadratic: {sf_T(t_a,5.0)-0.96189:.4f}")

# ---------- 6. misc ----------
print("\n=== 6. T2 bounds and rho u_+")
for (k,d) in [(3,3),(3,5),(5,5),(5,3)]:
    m = comb(d+k-1,k); kap = m-d; print(f"({k},{d}) sqrt(kap)/rho = {sqrt(kap)/rho_of(k):.3f}")
print(f"k/(3k-2) at k=3: {3/7:.4f}; rho^2/(2(1+rho^2)) = {0.75/(2*1.75):.4f}")
