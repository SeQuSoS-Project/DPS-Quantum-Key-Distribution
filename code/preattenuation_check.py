"""Check of equation (preatt): lossless Kerr on N photons followed by attenuation tau
gives visibility V = |<a>|^2/<n> = exp[-2N(1-cos Theta)], and Poissonian photon statistics."""
import numpy as np
from math import factorial, comb
D = 70
def coh(N):
    a = np.sqrt(N); return np.array([np.exp(-N/2) * a**n / np.sqrt(float(factorial(n))) for n in range(D)], dtype=complex)
def kerr(psi, th): return psi * np.exp(-1j * th * np.array([n*(n-1)/2 for n in range(D)]))
def loss(rho, t):
    out = np.zeros_like(rho)
    for n in range(D):
        for m in range(D):
            s = 0
            for l in range(D - max(n, m)):
                s += np.sqrt(float(comb(n+l, l) * comb(m+l, l))) * (1-t)**l * rho[n+l, m+l]
            out[n, m] = t**((n+m)/2) * s
    return out
for N, th, tau in [(5, .3, .1), (12, .1, .02), (3, 1.2, .5)]:
    psi = kerr(coh(N), th); rho = loss(np.outer(psi, psi.conj()), tau)
    a = np.trace(np.diag(np.sqrt(np.arange(1, D)), 1) @ rho); n = np.real(np.trace(np.diag(np.arange(D)) @ rho))
    pops = np.real(np.diag(rho)); mu = tau * N
    pois = np.array([np.exp(-mu) * mu**k / float(factorial(k)) for k in range(D)])
    print(f"N={N} th={th} tau={tau}: V={abs(a)**2/n:.6f} vs {np.exp(-2*N*(1-np.cos(th))):.6f}; max|p-Poisson|={np.max(abs(pops-pois)):.1e}")
