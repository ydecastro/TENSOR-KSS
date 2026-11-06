"""Section 6 (sec:low): Monte Carlo of the maximum test at (k,d) = (3,3). Field X(t) = sum_{|a|=3} sqrt(3!/a!) g_a t^a
on S^2 (Cov = <s,t>^3); sup found by a Fibonacci grid (10^4 points) followed by 4 Riemannian Newton steps from
the grid argmax. T_q = int X^2 dsigma computed exactly as g^T M g. Outputs: u_alpha, C_max =
Cov_0(1{sup X>u_alpha},T_q)/2, efficiency against the exact C_opt, power of the maximum test on a lambda grid
(common random numbers across lambda), crossings with the exact power of the quadratic test (quadratic_test.py).
usage: python3 maximum_test.py [nnull] [nalt]; the paper uses 4000000 2000000 (defaults 600000 300000).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, itertools, math, time, sys
from math import factorial

rng = np.random.default_rng(20260927)
k, d = 3, 3
alph = [a for a in itertools.product(range(k + 1), repeat=d) if sum(a) == k]
A = np.array(alph)
coef = np.array([math.sqrt(factorial(k) / np.prod([factorial(x) for x in a])) for a in alph])
nm = len(alph)

def fib(n):
    i = np.arange(n) + 0.5; ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]
P = fib(10000)
B = (np.prod(P[:, None, :] ** A[None, :, :], axis=2) * coef).astype(np.float32)   # (npts, nm)
e = np.array([0., 0., 1.]); spike = ((P @ e) ** k).astype(np.float32)
i_spike = alph.index((0, 0, 3))

# exact Gram matrix M_ab = int f_a f_b dsigma on S^2 : E prod t_i^{2c_i} = prod (2c_i-1)!! / (3*5*7)
def dfact(n): return 1 if n <= 0 else n * dfact(n - 2)
M = np.zeros((nm, nm))
for p, a in enumerate(alph):
    for q, b in enumerate(alph):
        s = np.array(a) + np.array(b)
        if np.all(s % 2 == 0):
            M[p, q] = coef[p] * coef[q] * np.prod([dfact(x - 1) for x in s]) / (3 * 5 * 7)
print("trace check: tr M = sum mu_j n_j =", np.trace(M), " ; eigenvalues of M:", np.round(np.sort(np.linalg.eigvalsh(M)), 6))

# symmetric tensor from polynomial coefficients c (n, nm): Y[ijk] = c_a / multinomial(a) for index multiset a
idx_of = {}
for p, a in enumerate(alph):
    idx = sum(([i] * a[i] for i in range(3)), [])
    mult = factorial(k) / np.prod([factorial(x) for x in a])
    for perm in set(itertools.permutations(idx)):
        idx_of[perm] = (p, mult)
def tensor(c):
    Y = np.zeros((c.shape[0], 3, 3, 3))
    for perm, (p, mult) in idx_of.items():
        Y[:, perm[0], perm[1], perm[2]] = c[:, p] / mult
    return Y

def refine(c, x0, iters=4):
    Y = tensor(c); x = x0.copy()
    for _ in range(iters):
        H = 6 * np.einsum('nijk,nk->nij', Y, x)
        g = 0.5 * np.einsum('nij,nj->ni', H, x)          # 3 Y(x,x,.)
        # tangent basis
        a = np.where(np.abs(x[:, [0]]) < 0.9, np.array([[1., 0, 0]]), np.array([[0., 1, 0]]))
        u1 = a - np.sum(a * x, 1, keepdims=True) * x; u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
        u2 = np.cross(x, u1)
        U = np.stack([u1, u2], 2)                           # (n,3,2)
        gr = np.einsum('nij,ni->nj', U, g)
        xg = np.sum(x * g, 1)
        Hr = np.einsum('nia,nij,njb->nab', U, H, U) - xg[:, None, None] * np.eye(2)[None]
        det = Hr[:, 0, 0] * Hr[:, 1, 1] - Hr[:, 0, 1] ** 2
        ok = (Hr[:, 0, 0] < 0) & (det > 0)
        inv = np.stack([np.stack([Hr[:, 1, 1], -Hr[:, 0, 1]], 1), np.stack([-Hr[:, 1, 0], Hr[:, 0, 0]], 1)], 1) / det[:, None, None]
        step = -np.einsum('nab,nb->na', inv, gr)
        step = np.where(ok[:, None], step, 0.05 * gr)       # fallback: small gradient step
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        step = np.where(nrm > 0.2, step * 0.2 / nrm, step)
        x = x + np.einsum('nia,na->ni', U, step); x /= np.linalg.norm(x, axis=1, keepdims=True)
    val = np.einsum('nijk,ni,nj,nk->n', Y, x, x, x)
    return val

def sup_batch(g, lam):
    V = g.astype(np.float32) @ B.T
    if lam:
        V += np.float32(lam) * spike[None, :]
    j = V.argmax(1)
    c = g * coef[None, :]          # polynomial coefficients c_a = sqrt(3!/a!) g_a
    c[:, i_spike] += lam
    val = refine(c, P[j])
    return np.maximum(val, V[np.arange(len(j)), j].astype(float))

t0 = time.time()
nnull = int(sys.argv[1]) if len(sys.argv) > 1 else 600000
blk = 5000
sup0 = np.empty(nnull); T0 = np.empty(nnull)
for b in range(0, nnull, blk):
    g = rng.standard_normal((blk, nm))
    sup0[b:b + blk] = sup_batch(g, 0.0)
    T0[b:b + blk] = np.einsum('na,ab,nb->n', g, M, g)
print(f"null: {nnull} reps in {time.time()-t0:.0f}s; E T = {T0.mean():.5f} (1), Var T = {T0.var():.5f} (2/7={2/7:.5f})")
C_opt = {0.05: 3.655341e-02, 0.01: 1.080666e-02}   # exact, a4
u = {}
for a in (0.05, 0.01):
    u[a] = np.quantile(sup0, 1 - a)
    phi = (sup0 > u[a]).astype(float)
    Cmax = 0.5 * np.cov(phi, T0)[0, 1]
    # delta-method s.e. of Cmax (ignoring quantile noise)
    se = 0.5 * np.std((phi - phi.mean()) * (T0 - T0.mean())) / np.sqrt(nnull)
    print(f"alpha={a}: u_alpha = {u[a]:.4f}, C_max = {Cmax:.4e} (+-{se:.1e}), efficiency = {Cmax/C_opt[a]:.4f} (+-{se/C_opt[a]:.3f})")

lams = [1.3, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 2.0, 3.0, 5.0]
nalt = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
hits = {a: np.zeros(len(lams)) for a in (0.05, 0.01)}
for b in range(0, nalt, blk):
    g = rng.standard_normal((blk, nm))
    for il, lam in enumerate(lams):
        s = sup_batch(g, lam)
        for a in (0.05, 0.01):
            hits[a][il] += np.sum(s > u[a])
print(f"alternatives: {nalt} reps per lambda, total time {time.time()-t0:.0f}s")
qp = np.load(os.path.join(HERE, 'quadpower.npy'))
qlams = list(qp[0])
for ia, a in enumerate((0.05, 0.01)):
    pm = hits[a] / nalt
    print(f"\nalpha={a}:  lambda | max test (MC, s.e.) | quadratic (exact)")
    diffs = []
    for il, lam in enumerate(lams):
        pq = qp[1 + ia][qlams.index(lam)]
        se = np.sqrt(pm[il] * (1 - pm[il]) / nalt)
        diffs.append(pm[il] - pq)
        print(f"   {lam:4.2f} | {pm[il]:.4f} ({se:.4f}) | {pq:.4f}   diff {pm[il]-pq:+.4f}")
    diffs = np.array(diffs)
    for il in range(len(lams) - 1):
        if diffs[il] < 0 <= diffs[il + 1]:
            l0, l1 = lams[il], lams[il + 1]
            lc = l0 + (l1 - l0) * (-diffs[il]) / (diffs[il + 1] - diffs[il])
            pq0, pq1 = qp[1 + ia][qlams.index(l0)], qp[1 + ia][qlams.index(l1)]
            print(f"   crossing (linear interp.) at lambda ~ {lc:.3f}, common power ~ {pq0 + (pq1-pq0)*(lc-l0)/(l1-l0):.4f}")
np.savez(os.path.join(HERE, 'maximum_test.npz'),
         lams=lams, h05=hits[0.05], h01=hits[0.01], nalt=nalt, u05=u[0.05], u01=u[0.01])
