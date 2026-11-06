"""Draw the enhancement-window figure from data/window_channel.json.
Usage: python3 plot_window_channel.py <repo root>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(sys.argv[1]); d = json.loads((root / 'data' / 'window_channel.json').read_text())
C1, C2, C3, C7, C8 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.linewidth': .6, 'legend.frameon': False, 'legend.fontsize': 7.5,
                     'axes.titlesize': 9, 'lines.linewidth': 1.5})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), constrained_layout=True)
a = ax[0]; th = np.array(d['a']['theta'])
lo, hi = [], []
for ed, c, ls in [('0.01', C1, '-'), ('0.03', C2, '--'), ('0.05', C3, '-.')]:
    g = np.array(d['a'][ed]['gain'])
    a.semilogx(th, 1e3 * g, color=c, ls=ls, label=rf'$e_d={float(ed)*100:.0f}\%$')
    big = th[g >= 0.1 * g.max()]; lo.append(big.min()); hi.append(big.max())
a.axhline(0, color=MUTED, lw=.8)
a.set_ylim(-1.5, 3.6); a.set_xlim(1e-3, 1)
a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel(r'Gain $\Delta M_\chi$ ($10^{-3}$ bits per accepted slot)')
a.set_title(r'(a) Enhancement window; $\mu_0=0.05$, $r=0.9$'); a.legend(loc='upper left')
a.text(1.5e-3, -1.2, 'below zero: SPM costs key', fontsize=7.5, color=MUTED)
b = ax[1]
win = (lo[0], hi[0])                     # e_d = 1% curve, gain >= 10% of its maximum
b.axhspan(*win, color=GRID, alpha=.9, lw=0)
b.text(1.5, win[0] * 1.6, 'enhancement window of (a)', fontsize=7.5, color=MUTED, va='bottom')
for name, c in [('SMF', C7), ('HNLF', C8)]:
    v = d['b'][name]; z = np.array(v['z'])
    b.semilogy(z, v['herald'], color=c, label=f'{name}: cross-phase $\\kappa z$')
    b.semilogy(z, v['theta_eff'], color=c, ls='--', label=f'{name}: Bob-side $\\Theta_{{\\rm eff}}$')
b.set_ylim(1e-10, 1); b.set_xlim(0, 100)
b.set_xlabel('Fibre length before the tap $z$ (km)'); b.set_ylabel('Kerr phase (rad)')
b.set_title('(b) Kerr phase supplied by fibre'); b.legend(loc='center', bbox_to_anchor=(0.5, 0.68), ncol=2, fontsize=6.5, columnspacing=1.0)
for x in ax:
    x.grid(color=GRID, lw=.5); x.set_axisbelow(True)
    for s in ('top', 'right'): x.spines[s].set_visible(False)
fig.savefig(root / 'paper' / 'figures' / 'window_channel.pdf'); fig.savefig(root / 'paper' / 'figures' / 'window_channel.png', dpi=160)
print('window band', win)
