"""Numerical check of the Kerr error bounds: for the lossy Kerr channel,
V = |m1|^2/mu >= exp(-4 mu0) for every (A, Theta), and for any distribution of Theta
at fixed A, V >= exp(-4 mu0) cos^2(2 mu0). Also checks Poisson populations."""
import numpy as np
from math import factorial
rng = np.random.default_rng(7)
def f(z): return np.where(np.abs(z) < 1e-12, 1.0, (1 - np.exp(-z)) / np.where(np.abs(z) < 1e-12, 1, z))
def m1n(mu0, A, th):  # m1/sqrt(mu)
    return np.exp(-1j * mu0 * th * f(A + 1j * th))
worst_det, worst_mix = np.inf, np.inf
for mu0 in [1e-4, 1e-3, 1e-2, .1, .3]:
    A = rng.uniform(0, 5, 20000); th = rng.uniform(-2 * np.pi, 2 * np.pi, 20000)
    V = np.abs(m1n(mu0, A, th))**2
    worst_det = min(worst_det, np.min(V / np.exp(-4 * mu0)))
    for _ in range(3000):
        a = rng.uniform(0, 5); ths = rng.uniform(-2*np.pi, 2*np.pi, rng.integers(2, 6)); w = rng.dirichlet(np.ones(len(ths)))
        Vm = np.abs(np.sum(w * m1n(mu0, a, ths)))**2
        worst_mix = min(worst_mix, Vm / (np.exp(-4 * mu0) * np.cos(2 * mu0)**2))
print('min V/exp(-4mu0) over deterministic (A,Theta):', round(worst_det, 6), '(>=1 required)')
print('min V/[exp(-4mu0)cos^2(2mu0)] over mixtures:', round(worst_mix, 6), '(>=1 required)')
# Poisson populations of the exact lossy-Kerr output (Kunz et al.), diagonal n=m
mu0, A, th = .3, .7, 1.1
pops = [mu0**n / factorial(n) * np.exp(-mu0 - A * n) * np.real(np.exp(mu0 * A * f(np.array(A + 0j)))) for n in range(12)]
mu = mu0 * np.exp(-A)
print('max |p_n - Poisson(mu)|:', max(abs(p - mu**n / factorial(n) * np.exp(-mu)) for n, p in enumerate(pops)))
