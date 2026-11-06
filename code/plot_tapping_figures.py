"""Draw Figures 3-7 from data/tapping_figures.json.
Usage: python3 plot_tapping_figures.py <repo root>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]); d = json.loads((root / 'data' / 'tapping_figures.json').read_text())
figdir = root / 'paper' / 'figures'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.linewidth': .6, 'legend.frameon': False, 'legend.fontsize': 7.5,
                     'axes.titlesize': 9, 'lines.linewidth': 1.5, 'lines.markersize': 4.5})
MU = [('0.05', C1), ('0.1', C2), ('0.5', C3)]
MU3 = [('0.1', C1), ('0.5', C2), ('1.0', C3)]

def style(ax):
    ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

def save(fig, name):
    fig.savefig(figdir / f'{name}.pdf'); fig.savefig(figdir / f'{name}.png', dpi=160); plt.close(fig)

# ---------------- Figure 3: operating windows
f3 = d['fig3']; th = np.array(f3['theta'])
fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0), constrained_layout=True)
a = ax[0, 0]
a.plot(th, f3['a']['nominal'], color=C2, ls='--', label='Nominal (USD) setting')
a.plot(th, f3['a']['tracked'], color=C3, ls=':', label='Mean-phase tracking')
a.plot(th, f3['a']['adjusted'], color=C1, label='Optimised')
a.set_ylabel(r'$I_E^{\rm loc}(\Theta)/I_E^{\rm loc}(0)$'); a.set_title(r'(a) Eve; $\mu_0=0.1$'); a.legend(loc='lower left')
a = ax[0, 1]
for mu, c in MU:
    a.plot(th, 100 * np.array(f3['b'][mu]), color=c, label=rf'$\mu_0={mu}$')
a.axhline(3, color=MUTED, lw=.8, ls='--'); a.text(0.0, 3.2, '3% budget', ha='left', va='bottom', fontsize=7.5, color=MUTED)
a.set_ylim(0, 10); a.set_ylabel('Bob QBER (%)'); a.set_title("(b) Bob's receiver"); a.legend(loc='upper left')
a = ax[1, 0]
a.plot(th, f3['c']['HE'], color=C1, label=r'$H(K|E_1,E_2,S)$')
a.plot(th, f3['c']['fh2'], color=C2, label=r'$f_{\rm EC}h_2(Q_B)$')
a.set_ylim(0, 1.05); a.set_ylabel('Entropy term (bits)'); a.set_xlabel(r'Kerr strength $\Theta$')
a.set_title(r'(c) Entropy balance; $\mu_0=0.1$'); a.legend(loc='center right')
a = ax[1, 1]
for mu, c in MU:
    a.plot(th, f3['d'][mu], color=c, label=rf'$\mu_0={mu}$')
a.set_ylabel('Margin $M$ (bits)'); a.set_xlabel(r'Kerr strength $\Theta$'); a.set_title('(d) Margin'); a.legend(loc='lower left')
for a in ax.flat: style(a)
save(fig, 'operating_windows')

# ---------------- Figure 4: heralded cross-phase
f4 = d['fig4']; th4 = np.array(f4['theta']); tt = np.linspace(0, 1, 101)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
a = ax[0]
a.plot(tt, -0.1 * np.sin(tt), color=C1, lw=1)
a.plot(tt, -0.1 * np.sin(tt) - tt, color=C2, lw=1)
a.plot(tt, -0.1 * np.sin(tt) - tt / 2, color=MUTED, lw=.9, ls='--', label='midpoint')
a.plot(th4, f4['phase_all'], 'o', color=C1, label=r'arg $m_E$ (all slots)')
a.plot(th4, f4['phase_heralded'], 's', color=C2, label=r'arg $m_{E|B}$ (Bob detection heralded)')
a.plot(th4, f4['phase_beta'], '^', color=C3, label=r'arg $\beta$ (optimised displacement)')
a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel('Phase (rad)'); a.set_title(r'(a) Heralded cross-phase; $\mu_0=0.1$')
a.legend(loc='lower left', fontsize=7)
a = ax[1]
a.plot(th4, 100 * np.array(f4['zeta_sifted']), 'o-', color=C1, label='Kerr, conditioned on acceptance')
a.plot(th4, 100 * np.array(f4['zeta_unsifted']), 's-', color=C2, label='Kerr, without conditioning')
a.plot(th4, 100 * np.array(f4['zeta_diffusion']), '^--', color=C3, label='phase diffusion, equal $V$')
a.axvline(f4['theta_3pct'], color=MUTED, lw=.8, ls=':'); a.text(f4['theta_3pct'] + .015, 0.6, '3% QBER', fontsize=7.5, color=MUTED, va='bottom')
a.set_ylim(0, 50); a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel(r'Information reduction $\zeta$ (%)')
a.set_title(r'(b) Origin of the suppression; $\mu_0=0.1$'); a.legend(loc='upper left', fontsize=7)
for a in ax: style(a)
save(fig, 'heralded_phase')

# ---------------- Figure 5: interferometric eavesdropper
f5 = d['fig5']; th5 = np.array(f5['theta']); V = np.exp(-2 * 0.1 * (1 - np.cos(tt)))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
a = ax[0]
a.plot(tt, 100 * (1 - V * np.cos(tt)) / 2, color=C1, lw=1)
a.plot(th5[::2], 100 * np.array(f5['eve_kerr'])[::2], 'o', color=C1, label=r'Eve, Kerr (line: $(1-V\cos\Theta)/2$)')
a.plot(th5[::2], 100 * np.array(f5['eve_diffusion'])[::2], '^', color=C3, label='Eve, phase diffusion')
a.plot(th5, 100 * np.array(f5['bob_eph']), color=MUTED, ls='--', lw=1, label=r'Bob, $e_V=(1-V)/2$')
a.plot(th5[::2], 100 * np.array(f5['bob_QB'])[::2], 's', color=C2, label=r"Bob's QBER $Q_B$")
a.set_ylim(0, 25); a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel('Error rate on accepted slots (%)')
a.set_title(r'(a) Interferometric Eve; $\mu_0=0.1$, $r=0.5$'); a.legend(loc='upper left', fontsize=7)
a = ax[1]; tl = f5['theta_log']
for mu, c in MU3:
    a.loglog(tl, f5['eps'][mu]['kerr'], 'o-', color=c, label=rf'$\mu_0={mu}$')
    a.loglog(tl, f5['eps'][mu]['diffusion'], '--', color=c)
a.axhline(1, color=MUTED, lw=.8, ls=':'); a.text(0.95, 1.06, 'break-even', ha='right', fontsize=7.5, color=MUTED)
a.plot([], [], 'k-', label='Kerr'); a.plot([], [], 'k--', label='phase diffusion')
a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel(r'Exchange ratio $\varepsilon$')
a.set_ylim(0.02, 1.4); a.set_title(r'(b) $\varepsilon$ for the interferometric Eve; $r=0.5$'); a.legend(loc='lower left', fontsize=7, ncol=2)
for a in ax: style(a)
save(fig, 'interferometric_eve')

# ---------------- Figure 6: receiver-independent bounds
f6 = d['fig6']; th6 = np.array(f6['theta'])
fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.6), constrained_layout=True)
a = ax[0]
a.plot(th6[::2], np.array(f6['chi_kerr'])[::2], 'o-', color=C1, label=r'Holevo $\chi_K$, Kerr')
a.plot(th6, f6['chi_diffusion'], '--', color=C1, label=r'Holevo $\chi_K$, phase diffusion')
a.plot(th6[::2], np.array(f6['di_kerr'])[::2], 's-', color=C2, label='delay interferometer, Kerr')
a.plot(th6[::2], np.array(f6['disp_kerr'])[::2], '^-', color=C3, label='optimised USD-type, Kerr')
a.set_ylim(0, 0.2); a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel("Eve's information (bits per accepted slot)")
a.set_title(r'(a) Bound and receivers; $\mu_0=0.1$'); a.legend(loc='upper right', fontsize=6.5)
a = ax[1]; rs = f6['r']
for t, c in [('0.3', C1), ('0.6', C2)]:
    a.semilogy(rs, 100 * np.array(f6['reduction'][t]['chi']), 'o-', color=c, label=rf'accepted slot, $\chi_K$; $\Theta={t}$')
    a.semilogy(rs, np.maximum(100 * np.array(f6['reduction'][t]['trace']), 1e-4), '--', color=c, label=rf'single pulse, $T_X$; $\Theta={t}$')
a.set_ylim(1e-3, 1e2); a.set_xlabel('Tap fraction $r$'); a.set_ylabel(r'Reduction relative to $\Theta=0$ (%)')
a.set_title('(b) Heralded versus unconditioned'); a.legend(loc='lower left', fontsize=6.5)
a = ax[2]; tl = f6['theta_log']
for mu, c in MU3:
    a.loglog(tl, f6['eps_chi'][mu]['kerr'], 'o-', color=c, label=rf'$\mu_0={mu}$')
    a.loglog(tl, f6['eps_chi'][mu]['diffusion'], '--', color=c)
a.axhline(1, color=MUTED, lw=.8, ls=':'); a.text(0.95, 1.08, 'break-even', ha='right', fontsize=7, color=MUTED)
a.plot([], [], 'k-', label='Kerr'); a.plot([], [], 'k--', label='phase diffusion')
a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel(r'Exchange ratio $\varepsilon_\chi$')
a.set_ylim(0.04, 1.6); a.set_title(r'(c) Receiver-independent; $r=0.5$'); a.legend(loc='lower left', fontsize=6, ncol=2)
for a in ax: style(a)
save(fig, 'holevo_bounds')

# ---------------- Figure 7: exchange ratio of the displacement receiver
f7 = d['fig7']; th7 = f7['theta']
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
a = ax[0]
for mu, c in MU3:
    a.plot(th7, f7['eps'][mu]['kerr'], 'o-', color=c, label=rf'$\mu_0={mu}$')
    a.plot(th7, f7['eps'][mu]['diffusion'], '--', color=c)
a.axhline(1, color=MUTED, lw=.8, ls=':'); a.text(0.98, 1.01, 'break-even', ha='right', va='bottom', fontsize=7.5, color=MUTED)
a.plot([], [], 'k-', label='Kerr'); a.plot([], [], 'k--', label='phase diffusion')
a.set_ylim(0, 1.1); a.set_xlabel(r'Kerr strength $\Theta$'); a.set_ylabel(r'Exchange ratio $\varepsilon$')
a.set_title(r'(a) $\varepsilon(\Theta)$; $r=0.5$'); a.legend(loc='center right', fontsize=7, ncol=2)
a = ax[1]; mus = np.array(f7['mu0'])
for (r, c, mk) in [('0.1', C1, 'o'), ('0.5', C2, 's'), ('0.9', C3, '^')]:
    e = np.array(f7['small'][r]['eps']); M0 = np.array(f7['small'][r]['M0'])
    a.semilogx(mus, e, '-', color=c, lw=1.2)
    a.semilogx(mus[M0 > 0], e[M0 > 0], mk, color=c, label=f'$r={r}$')
    if (M0 <= 0).any():
        a.semilogx(mus[M0 <= 0], e[M0 <= 0], mk, mfc='white', color=c)
a.semilogx([], [], 'o', mfc='white', color=MUTED, label=r'$M(0)\leq0$')
a.axhline(1, color=MUTED, lw=.8, ls=':')
a.set_ylim(0, 1.1); a.set_xlabel(r'Input mean photon number $\mu_0$'); a.set_ylabel(r'$\varepsilon$ at $\Theta=0.05$')
a.set_title(r'(b) Small-$\Theta$ exchange ratio'); a.legend(loc='center left', bbox_to_anchor=(0, 0.62), fontsize=7)
for a in ax: style(a)
save(fig, 'exchange_ratio')
print('figures written')
