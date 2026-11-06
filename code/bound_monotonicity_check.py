import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tapping_model import Link
mus = [0.05, 0.1, 0.2, 0.5, 1.0]; rs = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]; ths = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0]
pts = 0; viol = 0; worst = -1
for mu0 in mus:
    for r in rs:
        c0, t0 = Link(mu0, 0.0, r=r).holevo_trace()
        for th in ths:
            c, t = Link(mu0, th, r=r).holevo_trace(); pts += 1
            d = max(c - c0, t - t0); worst = max(worst, d)
            if c > c0 + 1e-12 or t > t0 + 1e-12: viol += 1
print(f'points {pts}, violations {viol}, largest increase {worst:.2e}')
