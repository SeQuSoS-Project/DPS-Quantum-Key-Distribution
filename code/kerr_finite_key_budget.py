"""Finite-key cost of Kerr deformation for DPS-QKD under the general-attack proof
of Mizutani, Takeuchi & Tamaki, Phys. Rev. Research 5, 023132 (2023).

The finite-key kernels (Kato concentration, constants, channel model) are imported
unchanged from the companion package (audit_numeric.py). The only modification is
the signal bit-error rate, which acquires the Kerr photon-weighted error
e_ph = (1 - V)/2 in the weak-detection limit:
    e_sig = e_d + (1 - 2 e_d) e_ph,  ebit = (s e_sig + n/2)/(s + n).
Usage: python3 kerr_finite_key_budget.py <dir containing audit_numeric.py> <outdir>
"""
import sys, json, os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import gammainc
sys.path.insert(0, sys.argv[1])
import audit_numeric as k

ED = 0.01          # intrinsic misalignment error
NEM = 1e13         # emitted three-pulse blocks, as in Mizutani et al.
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

def eph_weak(mu, theta):
    """Photon-weighted Kerr error for Kerr acting on the emitted weak pulses (A=0)."""
    return (1 - np.exp(-2 * mu * (1 - np.cos(theta)))) / 2

def key(mu, t, eta, noise, eph):
    """Copy of audit_numeric.key with the Kerr error added to the signal error."""
    s = mu * eta; n = noise
    esig = ED + (1 - 2 * ED) * eph
    ebit = (s * esig + n / 2) / (s + n)
    ndet = NEM * 2 * (s + n) * np.exp(-2 * (s + n))
    nc = t * ndet
    q = gammainc(np.arange(1, 4), 3 * mu)
    ch = .5 * (-k.LOGEPS + np.sqrt(k.LOGEPS**2 - 8 * t * q * NEM * k.LOGEPS))
    counts = t * q * NEM + ch
    delta = np.sqrt(-2 * ndet * k.LOGEPS)
    d = max(k.LAM / (1 - t) + 1, 1 / t + k.LAM + 1)
    mstar = min(counts[2], np.floor((ndet - 1) / 2))
    if mstar < 0: return -np.inf
    a, b, ba = k.kato(ndet, mstar)
    k3 = counts[2] * (1 + 2 * a / np.sqrt(ndet)) + ba * np.sqrt(ndet)
    nph = k.LAM * ebit * nc + counts[1] + k.LAM * np.sqrt((counts[0] + delta) * k3) + t * d * delta
    eph_bound = nph / nc
    ell = nc * (1 - k.entropy(eph_bound) - 1.16 * k.entropy(ebit)) - 86
    return ell / (3 * NEM)

def optimise(eta, noise, eph_of_mu):
    scale = max(eta**2 * 1e-3, 1e-13)
    fun = lambda x: -key(10**x[0], x[1], eta, noise, eph_of_mu(10**x[0])) / scale
    best = None
    for s in [[np.log10(.009 * eta), .85], [np.log10(.003 * eta), .5], [np.log10(.015 * eta), .7]]:
        f = minimize(fun, s, method='Nelder-Mead', bounds=[(-9, -.5), (.001, .9999)],
                     options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2500})
        if best is None or f.fun < best.fun: best = f
    return dict(rate=float(-best.fun * scale), mu=float(10**best.x[0]), t=float(best.x[1]))

def check_baseline():
    # With eph = 0 the copy must reproduce the companion package exactly.
    eta, n = k.channel(25)
    for mu, t in [(1.5e-3, .8), (4e-4, .6)]:
        a = key(mu, t, eta, n, 0.); b = k.key(mu, t, eta, 1e13, noise=n)
        assert abs(a - b) <= 1e-12 * abs(b), (a, b)

check_baseline()
data = dict(parameters=dict(e_d=ED, N_em=NEM, f_EC=1.16, eps_sec='2^-27 (as in Mizutani et al.)',
            channel='eta = 10^(-0.2 L/10) * 0.8 * 10^(-0.2); dark probability 1e-8 per detector'),
            distances={})
thetas = np.linspace(0, np.pi, 61)
deltas = np.logspace(-5, -1, 41)
for L in [10, 25, 50]:
    eta, noise = k.channel(L)
    base = optimise(eta, noise, lambda mu: 0.)
    rel_theta = [optimise(eta, noise, lambda mu, th=th: eph_weak(mu, th))['rate'] / base['rate'] for th in thetas]
    rel_delta = [optimise(eta, noise, lambda mu, dl=dl: dl)['rate'] / base['rate'] for dl in deltas]
    th1 = brentq(lambda th: 1 - optimise(eta, noise, lambda mu: eph_weak(mu, th))['rate'] / base['rate'] - .01,
                 1e-3, np.pi, xtol=1e-5)
    worst = optimise(eta, noise, lambda mu: eph_weak(mu, np.pi))
    rand = optimise(eta, noise, lambda mu: (1 - np.exp(-2 * mu)) / 2)   # uniformly random Kerr phase
    d1 = brentq(lambda dl: 1 - optimise(eta, noise, lambda mu: dl)['rate'] / base['rate'] - .01,
                1e-7, 1e-1, xtol=1e-9)
    data['distances'][str(L)] = dict(eta=float(eta), noise=float(noise), baseline=base,
        theta_1pct=float(th1), loss_theta_pi=float(1 - worst['rate'] / base['rate']),
        loss_random_phase=float(1 - rand['rate'] / base['rate']),
        delta_1pct=float(d1), thetas=thetas.tolist(), rel_theta=rel_theta,
        deltas=deltas.tolist(), rel_delta=rel_delta)
    print(f"L={L}: R0={base['rate']:.3e} mu*={base['mu']:.2e} | Theta_1%={th1:.3f} | "
          f"loss(Theta=pi)={1-worst['rate']/base['rate']:.1%} | loss(random)={1-rand['rate']/base['rate']:.1%} | "
          f"delta_1%={d1:.2e}")
(OUT / 'finite_key_budget.json').write_text(json.dumps(data, indent=1))
