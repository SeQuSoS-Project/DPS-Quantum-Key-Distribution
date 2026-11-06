"""Figure: finite-key cost of Kerr deformation (reads data/finite_key_budget.json)."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(sys.argv[1])
d = json.loads((root / 'data' / 'finite_key_budget.json').read_text())['distances']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.linewidth': .6})
series = [('10', '#2a78d6', '-'), ('25', '#eb6834', '--'), ('50', '#1baf7a', '-.')]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
for L, c, ls in series:
    v = d[L]
    ax[0].plot(v['thetas'], v['rel_theta'], color=c, ls=ls, lw=1.6, label=f'{L} km')
    ax[0].annotate(f'{L} km', (v['thetas'][-1], v['rel_theta'][-1]), xytext=(4, 0),
                   textcoords='offset points', va='center', fontsize=8, color='#0b0b0b')
    ax[1].semilogx(v['deltas'], v['rel_delta'], color=c, ls=ls, lw=1.6, label=f'{L} km')
for a in ax:
    a.axhline(.99, color='#52514e', lw=.8, ls=':')
    a.grid(color='#e4e3df', lw=.5); a.set_axisbelow(True)
    for s in ['top', 'right']: a.spines[s].set_visible(False)
ax[0].set_xlim(0, np.pi * 1.16); ax[0].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax[0].set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
ax[0].set_xlabel(r'Two-photon Kerr phase $\Theta$ on the emitted pulses (rad)')
ax[0].set_ylabel('Finite key relative to $\\Theta=0$')
ax[0].set_title('(a) Kerr acting on the weak pulses', fontsize=9)
ax[0].text(2.35, .992, '1% key loss', va='bottom', fontsize=7.5, color='#52514e')
ax[1].set_xlabel(r'Added visibility error $\delta e$')
ax[1].set_ylabel('Finite key relative to $\\delta e=0$')
ax[1].set_title('(b) Fixed added error (any Kerr stage)', fontsize=9)
ax[1].legend(frameon=False, fontsize=8, loc='lower left')
fig.savefig(root / 'paper' / 'figures' / 'finite_key_budget.pdf')
fig.savefig(root / 'paper' / 'figures' / 'finite_key_budget.png', dpi=160)
