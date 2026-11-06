import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tapping_model import Link
from math import lgamma
def pois(n, lam): return np.exp(-lam + n*np.log(lam) - lgamma(n+1)) if lam > 0 else float(n == 0)
maxdev = 0.0
for mu0 in [0.05, 0.1, 0.5]:
    for r in [0.1, 0.5, 0.9]:
        for th in [0.1, 0.6, 1.5]:
            L = Link(mu0, th, r=r); L0 = Link(mu0, 0.0, r=r, DE=L.DE)
            DE = L.DE; n = np.arange(DE)
            om = [O / L.GB for O in L.omega()]; om0 = [O / L0.GB for O in L0.omega()]
            lamB = (1 - r) * mu0; sB = L.sB; pB = L.p_B
            UE = np.diag(np.exp(-0.5j * th * n * (n - 1))); U2 = np.kron(UE, UE)
            N = 40
            for k in (0, 1):
                acc = np.zeros_like(om0[k])
                for a in range(N):
                    for b in range(N):
                        q = pois(a, lamB) * pois(b, lamB) * (1 - (1 - pB)**2 * (1 - sB)**(a + b)) / L.GB
                        if q < 1e-18: continue
                        Rd = np.kron(np.exp(-1j * a * th * n), np.exp(-1j * b * th * n))
                        acc += q * (Rd[:, None] * om0[k] * Rd.conj()[None, :])
                pred = U2 @ acc @ U2.conj().T
                maxdev = max(maxdev, np.abs(pred - om[k]).max())
print(f'channel form vs direct construction: max |difference| = {maxdev:.2e}')
