"""Data for the enhancement-window figure: (a) receiver-independent key-margin gain versus
Kerr phase; (b) Kerr phase accumulated along a fibre before the tap.
Usage: python3 compute_window_channel.py <output json>"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tapping_model import Link, h2

F = 1.16
out = dict(a={}, b={})
ths = np.geomspace(1e-3, 1, 40)
mu0, r = 0.05, 0.9
for ed in [0.01, 0.03, 0.05]:
    L0 = Link(mu0, 0.0, r=r, e_d=ed); c0 = L0.holevo_trace()[0]; q0 = L0.bob()['QB']
    gain, dq = [], []
    for th in ths:
        L = Link(mu0, th, r=r, e_d=ed); q = L.bob()['QB']
        gain.append((c0 - L.holevo_trace()[0]) - F * (h2(q) - h2(q0))); dq.append(q - q0)
    out['a'][str(ed)] = dict(gain=gain, dQB=dq)
out['a']['theta'] = ths.tolist(); out['a']['mu0'] = mu0; out['a']['r'] = r

# (b) fibre: heralded cross-phase kappa*z and Bob-side lossless-equivalent phase
hbar_omega = 6.62607015e-34 * 299792458 / 1550e-9
z = np.linspace(0.05, 100, 400)
for name, gamma, adb, tau in [('SMF', 1.3, 0.2, 100e-12), ('HNLF', 20.0, 1.0, 10e-12)]:
    kappa = gamma * hbar_omega / tau                    # per km
    al = adb * np.log(10) / 10
    herald = kappa * z
    theta_eff = np.sqrt(2 * kappa**2 / al**2 * (1 - np.exp(-al * z) * (1 + al * z)))
    out['b'][name] = dict(gamma=gamma, alpha_dB=adb, tau=tau, kappa=kappa, z=z.tolist(),
                          herald=herald.tolist(), theta_eff=theta_eff.tolist(),
                          saturation=float(np.sqrt(2) * kappa / al))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for ed in ['0.01', '0.03', '0.05']:
    g = np.array(out['a'][ed]['gain']); pos = ths[g > 0]; big = ths[g >= 0.1 * g.max()]
    print(ed, 'max', g.max(), 'at', ths[g.argmax()], 'positive up to', pos.max(), 'all positive below?', bool((g[ths < pos.max()] > 0).all()),
          '>=10% of max in', big.min(), big.max())
for n, v in out['b'].items():
    print(n, 'kappa', v['kappa'], 'herald@100km', v['herald'][-1], 'theta_eff sat', v['saturation'])
