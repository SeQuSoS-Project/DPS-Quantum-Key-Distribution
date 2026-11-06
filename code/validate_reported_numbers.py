"""Recompute the headline numbers quoted in the manuscript and check them.
Usage: python3 validate_reported_numbers.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tapping_model import Link, h2, mean_field

checks = []
def check(name, value, reported, tol):
    ok = abs(value - reported) <= tol
    checks.append(ok)
    print(f"{'OK ' if ok else 'BAD'} {name:<52} computed {value:.6g}  reported {reported:.6g}")

L0, L6, P6 = Link(0.1, 0.0), Link(0.1, 0.6), Link(0.1, 0.6, model='diffusion')
b6 = L6.bob()
check('Bob QBER at mu0=0.1, Theta=0.6 (%)', 100 * b6['QB'], 2.6875, 5e-5)
check('Bob gain G_B', b6['GB'], 9.9970e-4, 5e-9)
i0, _ = L0.disp_optimise(); i6, _ = L6.disp_optimise(); p6, _ = P6.disp_optimise()
check('adjusted displacement, Theta=0 (bits)', i0, 0.017325, 5e-7)
check('adjusted displacement, Theta=0.6 (bits)', i6, 0.013739, 5e-7)
check('adjusted displacement, phase diffusion (bits)', p6, 0.016155, 5e-7)
bn = np.sqrt(L6.r * L6.mu / 2)
check('nominal displacement, Theta=0.6 (bits)', L6.disp_info(bn), 0.004636, 5e-7)
check('mean-phase tracking, Theta=0.6 (bits)', L6.disp_info(bn * np.exp(1j * np.angle(mean_field(L6.rhoE[0])))), 0.005097, 5e-7)
check('delay interferometer, Theta=0 (bits)', L0.di_info(), 0.048771, 5e-7)
check('delay interferometer, Theta=0.6 (bits)', L6.di_info(), 0.026095, 5e-7)
check('delay interferometer, phase diffusion (bits)', P6.di_info(), 0.042741, 5e-7)
check('interferometer error on accepted slots, Theta=0.6 (%)', 100 * L6.di_error(), 9.80, 5e-3)
check('interferometer error, phase diffusion (%)', 100 * P6.di_error(), 1.68, 5e-3)
c0, T0 = L0.holevo_trace(); c6, T6 = L6.holevo_trace(); cP, TP = P6.holevo_trace()
check('Holevo chi_K, Theta=0 (bits)', c0, 0.11348, 5e-6)
check('Holevo chi_K, Theta=0.6 (bits)', c6, 0.07204, 5e-6)
check('Holevo chi_K, phase diffusion (bits)', cP, 0.10039, 5e-6)
check('trace distance, Theta=0', T0, 0.1813, 5e-5)
check('trace distance, Theta=0.6', T6, 0.1627, 5e-5)
dh = 1.16 * (h2(b6['QB']) - h2(L0.bob()['QB']))
check("Bob's weighted error-entropy growth (bits)", dh, 0.1126, 5e-5)
check('exchange ratio, displacement', (i0 - i6) / dh, 0.032, 5e-4)
check('exchange ratio, Holevo', (c0 - c6) / dh, 0.368, 5e-4)
check('exchange ratio, interferometer (Table 5 prints 0.202)', (L0.di_info() - L6.di_info()) / dh, 0.2015, 5e-4)
G0, G = Link(0.05, 0.0, r=0.9), Link(0.05, 0.1, r=0.9)
gain = (G0.holevo_trace()[0] - G.holevo_trace()[0]) - 1.16 * (h2(G.bob()['QB']) - h2(G0.bob()['QB']))
check('largest Holevo-level gain, e_d=1% (bits/slot)', gain, 2.4e-4, 5e-6)
G0, G = Link(0.1, 0.0, r=0.9, e_d=0.05), Link(0.1, 0.3, r=0.9, e_d=0.05)
gain = (G0.holevo_trace()[0] - G.holevo_trace()[0]) - 1.16 * (h2(G.bob()['QB']) - h2(G0.bob()['QB']))
check('largest Holevo-level gain, e_d=5% (bits/slot)', gain, 3.8e-3, 5e-5)
print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
