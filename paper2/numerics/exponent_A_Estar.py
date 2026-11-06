"""Lambda(sqrt(kappa)/rho), A and E* (table of Remark 24, rem:bracket, column of Theorem 20), the balance column
(eq:exponent), the column S, k/(2(3k-2)), sqrt(kappa)/rho, and the threshold alpha^{-1/kappa} at (3,3).
Log-space integrals in mpmath, 30 digits.
"""
import mpmath as mp
from math import comb
import os
mp.mp.dps = 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exponent_A_Estar.out")
log = open(OUT, "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

def Lambda_terms(l, kappa, N, nu):
    out = []
    for j in range(N + 1):
        def f(x, j=j):
            if x <= 0 or x >= 1:
                return mp.mpf(0)
            return mp.exp((kappa + j - 1) * mp.log(x) + (N - j) * mp.log(1 - x)
                          - (nu + 1) / mp.mpf(2) * mp.log(1 + kappa * x * x / l / l))
        I = mp.quad(f, [0, 0.5, 0.8, 0.95, 1])
        out.append(1 / ((kappa + j) * I))
    return out

alpha = mp.mpf("0.05")
table = {(3,3):(7,3.06,3.153,0.0351,0.0320,0.2143),(3,4):(16,4.62,2.331,0.0606,0.0507,0.2143),
         (3,5):(30,6.32,2.015,0.0780,0.0626,0.2143),(5,3):(18,5.37,1.952,0.0705,0.0556,0.1923),
         (5,5):(121,13.91,1.478,0.1112,0.0812,0.1923),(7,5):(325,23.60,1.346,0.1218,0.0864,0.1842)}
P("(k,d) kappa  Tmax=sqrt(kappa)/rho  Lambda(Tmax)  argmax_j  A=(Lambda/alpha)^(1/kappa)  E*=rho^2/(2(A^2+rho^2))  A_mech  E_mech  S=k/(2(3k-2))   | paper: Tmax A E* E_mech S")
for (k, d), row in table.items():
    m = comb(d + k - 1, k); N = d - 1; kappa = m - d; nu = m - 1
    rho = mp.sqrt(mp.mpf(k) / (2 * (k - 1)))
    Tmax = mp.sqrt(kappa) / rho
    terms = Lambda_terms(Tmax, kappa, N, nu)
    Lam = max(terms); jarg = terms.index(Lam)
    A = mp.power(Lam / alpha, mp.mpf(1) / kappa)
    Estar = rho**2 / (2 * (A**2 + rho**2))
    Amech = mp.power(comb(kappa + N, N) / alpha, mp.mpf(1) / kappa)
    Emech = rho**2 / (2 * (rho + Amech)**2)
    S = mp.mpf(k) / (2 * (3 * k - 2))
    P(f"({k},{d}) {kappa:4d}  {mp.nstr(Tmax,5):>7}  {mp.nstr(Lam,8):>12}  j={jarg}  A={mp.nstr(A,6)}  E*={mp.nstr(Estar,5)}  Amech={mp.nstr(Amech,5)}  Emech={mp.nstr(Emech,5)}  S={mp.nstr(S,5)}   | {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}")
    P(f"      improvement E*/E_mech = {mp.nstr(Estar/Emech,4)};  Lambda(Tmax)/binom = {mp.nstr(Lam/comb(kappa+N,N),5)};  all Lambda terms: {[mp.nstr(t,5) for t in terms]}")

# threshold numbers at (3,3), Remark 24: alpha^{-1/kappa}, rho^2/(2(alpha^{-2/kappa}+rho^2))
k, d = 3, 3; m = comb(d+k-1,k); kappa = m-d; rho = mp.sqrt(mp.mpf(k)/(2*(k-1)))
P("\nRemark 24 at (3,3): alpha^{-1/kappa} =", mp.nstr(mp.power(alpha, -mp.mpf(1)/kappa), 6),
  " rho^2/(2(alpha^{-2/kappa}+rho^2)) =", mp.nstr(rho**2/(2*(mp.power(alpha,-mp.mpf(2)/kappa)+rho**2)), 6))
P("Theorem 16 (thm:upper), text: (5,5) alpha^{-1/121} =", mp.nstr(mp.power(alpha, -mp.mpf(1)/121), 6))
# Section 4: A(1+eta)/(1-eta) at (3,5) eta=0.8 -> exponent
k, d = 3, 5; m = comb(d+k-1,k); kappa = m-d; N = d-1
Amech = mp.power(comb(kappa+N,N)/alpha, mp.mpf(1)/kappa); A9 = Amech*9
P("Section 4: (3,5) A_mech =", mp.nstr(Amech,5), " 9A =", mp.nstr(A9,5), " exponent rho^2/(2(rho+9A)^2) =", mp.nstr(rho**2/(2*(rho+A9)**2),5))
# Section 4 table check
P("\nSection 4 (balance exponent):")
for (k,d) in [(3,3),(3,5),(5,3),(5,5),(7,5)]:
    m = comb(d+k-1,k); kappa=m-d; N=d-1; rho = mp.sqrt(mp.mpf(k)/(2*(k-1)))
    Amech = mp.power(comb(kappa+N,N)/alpha, mp.mpf(1)/kappa)
    P(f"({k},{d}) kappa={kappa} rho={mp.nstr(rho,4)} A={mp.nstr(Amech,4)} v*/lam=1/(rho+A)={mp.nstr(1/(rho+Amech),3)} exp={mp.nstr(rho**2/(2*(rho+Amech)**2),4)} share={mp.nstr(rho**2/(rho+Amech)**2*100,3)}%")
log.close()
