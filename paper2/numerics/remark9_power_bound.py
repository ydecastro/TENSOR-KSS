"""Remark 9 (rem:tuple) and Section 7, 'The power bound': the ingredients of Theorem 5 (thm:main) at
(k,d) = (3,5), alpha = 0.05, eta = 0.8, tau = 9, b0 = 6, v = 50, lambda = 150, with the Rice constant of
eq:ricebound; the check of Remark 10 (rem:Shigh); the level-map claim of Section 7; and the values of the
remark after Corollary 7 (the lower bound of Theorem 6(i) solved at (3,5)).
"""
import numpy as np
from math import comb, sqrt, log, exp, pi, gamma, factorial
from scipy import integrate, optimize, special, stats

k, d = 3, 5
alpha, eta, tau, b0, v, lam = 0.05, 0.8, 9.0, 6.0, 50.0, 150.0
m = comb(d + k - 1, k); kappa = m - d; N = d - 1; nu = m - 1
rho = sqrt(k / (2 * (k - 1)))
v_minus = (1 - eta) * v
mu_b0 = (k - 1) * b0 / lam
beta_b0 = sqrt(k * (k - 1) / 2) * b0 / lam
print(f"m={m} kappa={kappa} N={N} nu={nu} rho={rho:.4f}")
print(f"v_-={v_minus}, mu_b0={mu_b0:.4f}, beta_b0={beta_b0:.4f}")
print(f"rho v_-={rho*v_minus:.3f} >= d-1={d-1}; rho(v_- - mu)={rho*(v_minus-mu_b0):.3f} >= sqrt(2(d-1))={sqrt(2*(d-1)):.3f}")
print(f"beta_b0 <= rho v_-/4 = {rho*v_minus/4:.3f}; beta_b0+sqrt(2(d-1))={beta_b0+sqrt(2*(d-1)):.3f} <= rho tau={rho*tau:.3f} < rho v_-={rho*v_minus:.3f}")

# H_r(ell) up to constants: int_ell^inf prod(u - r_i) (1+u^2/kappa)^{-(nu+1)/2} du
def H(ell, eig):
    f = lambda u: np.prod([u - r for r in eig]) * (1 + u * u / kappa) ** (-(nu + 1) / 2)
    I, _ = integrate.quad(f, ell, np.inf, epsabs=0, epsrel=1e-11, limit=400)
    return I

def u_alpha(eig, v=v, alpha=alpha):
    target = alpha * H(v, eig)
    return optimize.brentq(lambda u: H(u, eig) - target, v, 50 * v, xtol=1e-10)

print()
print("=== u^dagger: u_alpha(r/s) over a few admissible spectra (|r_i|<=tau, |s-1|<=eta) ===")
cands = {
    "r=tau Id, s=1-eta (attained)": [tau / (1 - eta)] * N,
    "r=tau Id, s=1": [tau] * N,
    "r=tau Id, s=1+eta": [tau / (1 + eta)] * N,
    "r=-tau Id, s=1-eta": [-tau / (1 - eta)] * N,
    "r=diag(tau,0,0,0), s=1-eta": [tau / (1 - eta), 0, 0, 0],
    "r=diag(tau,tau,-tau,-tau), s=1-eta": [tau / (1 - eta)] * 2 + [-tau / (1 - eta)] * 2,
    "r=0": [0.0] * N,
}
best = None
for name, eig in cands.items():
    ua = u_alpha(eig)
    print(f"  {name:38s} u_alpha = {ua:.3f}")
    if best is None or ua > best[1]:
        best = (name, ua)
u_dag = u_alpha([tau / (1 - eta)] * N)
u_plus = (1 + eta) * u_dag
p_minus = (rho * v_minus) ** 2
print(f"u^dagger (paper 60.97) = {u_dag:.3f}; u_+ = (1+eta) u^dagger = {u_plus:.2f} (paper 109.7); p_- = {p_minus:.1f}")

# monotonicity probe: random spectra in the admissible box
rng = np.random.default_rng(1)
worst = 0
for _ in range(60):
    eig = rng.uniform(-tau, tau, N) / rng.uniform(1 - eta, 1 + eta)
    worst = max(worst, u_alpha(list(eig)))
print(f"max u_alpha over 60 random admissible spectra = {worst:.3f} (should not exceed u^dagger)")

print()
print("=== the three mark probabilities ===")
C_rice = 43 * sqrt(pi) * k ** 3 * 2 ** (2 * k) * (k - 1) ** ((d - 2) / 2) / gamma((d - 1) / 2)
def Pbar(y):
    return C_rice * (1 + rho * y) ** (d - 1) / (rho * y) * exp(-rho ** 2 * y ** 2 / 2)
Pb = Pbar(v_minus - mu_b0)
print(f"C_rice={C_rice:.4g}; Pbar(v_- - mu_b0)={Pb:.3g} (paper 3.5e-8)")
# q_eta: sup over noncentrality delta^2 in [0, beta_b0^2] of P{|sigma_hat-1|>eta}
def q_eta(delta2):
    lo = kappa * (1 - eta) ** 2; hi = kappa * (1 + eta) ** 2
    return stats.ncx2.cdf(lo, kappa, delta2) + stats.ncx2.sf(hi, kappa, delta2)
qe = max(q_eta(x) for x in np.linspace(0, beta_b0 ** 2, 50))
print(f"q_eta = {qe:.3g} (paper 5.2e-9); central value {q_eta(0):.3g}")
qt = 2 * exp(-0.5 * (rho * tau - beta_b0 - sqrt(2 * N)) ** 2)
print(f"q_tau bound = {qt:.3g} (paper 1.3e-5)")
C_kd = 2 * (9 / 4 + 5 / 6 * sqrt(d - 1)) ** (d - 1)
e = 1 - 1 / p_minus
E_err = C_kd * (2 * Pb ** e + qe ** e + qt ** e)
print(f"C_kd={C_kd:.2f}; calE = {E_err:.4g} (paper 6.85e-3); share of q_tau = {100*C_kd*qt**e/E_err:.1f}% (paper 99.4%)")

print()
print("=== Kac-Rice mass delta^lambda(u_+; B_b0) with A = D_N - beta_2 D_{N-1} ===")
def D(n, z):
    return sum((-1) ** i * factorial(n) / (factorial(i) * factorial(n - 2 * i) * 4 ** i) * z ** (n - 2 * i) for i in range(n // 2 + 1))
def beta2(c):
    return lam * sqrt(k * (k - 1) / 2) * c ** (k - 2) * (1 - c * c)
def A_lam(z, c):
    return D(N, z) - beta2(c) * D(N - 1, z)
omega = 2 * pi ** ((d - 1) / 2) / gamma((d - 1) / 2)
pref = omega * ((k - 1) / pi) ** (N / 2)
c0 = sqrt(1 - b0 / lam ** 2)
def inner(c):
    f = lambda x: A_lam(rho * x, c) * stats.norm.pdf(x - lam * c ** k)
    I, _ = integrate.quad(f, u_plus, lam * c ** k + 12, epsabs=0, epsrel=1e-10, limit=400)
    return I
def outer(c):
    return (1 - c * c) ** ((d - 3) / 2) * exp(-k * lam ** 2 / 2 * c ** (2 * k - 2) * (1 - c * c)) * inner(c)
# substitute w = 1 - c^2 for accuracy near c = 1
def outer_w(w):
    c = sqrt(1 - w)
    return outer(c) / (2 * c)
mass, err = integrate.quad(outer_w, 0, b0 / lam ** 2, epsabs=0, epsrel=1e-10, limit=400)
mass *= pref
print(f"delta^lambda(u_+; B_b0) = {mass:.6f} (paper 0.99876); P(chi2_4 <= k b0) = {stats.chi2.cdf(k*b0, d-1):.6f} (paper 0.99877)")
print(f"power bound (1-calE)*mass = {(1-E_err)*mass:.4f} (paper 0.991)")
print(f"rho u_+ = {rho*u_plus:.1f} (paper 95.0); (lambda-u_+) = {lam-u_plus:.1f} sd (paper 40)")

print()
print("=== Remark 10: q_eta >= P(chi2_kappa > 4 kappa) ===")
print(f"P(chi2_{kappa} > {4*kappa}) = {stats.chi2.sf(4*kappa, kappa):.3g}; C_kd * that = {C_kd*stats.chi2.sf(4*kappa, kappa):.3g}")

print()
print("=== Section 7 level-map claim: t_{m-1} 0.95 quantile vs Gaussian at (3,5) ===")
print(f"t_34(0.95)/z(0.95) - 1 = {stats.t.ppf(0.95, 34)/stats.norm.ppf(0.95)-1:.4f} (paper 2.8%)")

print()
print("=== Theorem 6(i) at (3,5): smallest R with R (d-1) k lam^2 >= ((d-1) - d R)^2 ===")
for lam_ in (5, 10, 20, 40):
    a_, b_, c_ = d * d, -(2 * d * (d - 1) + (d - 1) * k * lam_ ** 2), (d - 1) ** 2
    R_ = (-b_ - sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    print(f"  lambda={lam_:2d}: R >= {R_:.6f}, that is {100 * R_ / ((d - 1) / (k * lam_ ** 2)):.2f}% of (d-1)/(k lambda^2)")
print("  paper: 0.0472, 0.01290, 0.003305, 0.000831; 88.54%, 96.80%, 99.18%, 99.79%")
