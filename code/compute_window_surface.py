"""Receiver-independent margin gain over (Kerr strength, intensity) for the 3D window figure.
Usage: python3 compute_window_surface.py <output json> [e_d] [r]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tapping_model import Link, h2
F = 1.16
ed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
ths = np.geomspace(3e-3, 1, 33)
mus = np.geomspace(0.01, 1.0, 21)
G = np.zeros((len(mus), len(ths))); E = np.zeros_like(G)
t0 = time.time()
for i, mu0 in enumerate(mus):
    L0 = Link(mu0, 0.0, r=r, e_d=ed); c0 = L0.holevo_trace()[0]; q0 = L0.bob()['QB']
    for j, th in enumerate(ths):
        L = Link(mu0, th, r=r, e_d=ed); q = L.bob()['QB']; dc = c0 - L.holevo_trace()[0]
        dh = F * (h2(q) - h2(q0))
        G[i, j] = dc - dh; E[i, j] = dc / dh
json.dump(dict(theta=ths.tolist(), mu0=mus.tolist(), gain=G.tolist(), eps=E.tolist(), e_d=ed, r=r), open(sys.argv[1], 'w'))
k = np.unravel_index(G.argmax(), G.shape)
print(f'e_d={ed} r={r}: max gain {G.max():.3e} at mu0={mus[k[0]]:.3f}, Theta={ths[k[1]]:.3f}; time {time.time()-t0:.0f}s')
for i in range(0, len(mus), 4):
    pos = ths[G[i] > 0]
    print(f'  mu0={mus[i]:.3f}: window up to Theta={pos.max() if len(pos) else 0:.3f}, max {G[i].max():.2e}')
