"""3D view of the enhancement window: positive part of the margin gain over
(Kerr strength, intensity), with a top-view inset of the window edges.
Usage: python3 plot_window_surface.py <repo root>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

root = Path(sys.argv[1])
D = {ed: json.loads((root / 'data' / f'window_surface_ed{ed}.json').read_text()) for ed in (1, 3, 5)}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NEG = '#f3c5ad'
plt.rcParams.update({'font.size': 8.5, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': MUTED, 'axes.linewidth': .6})
styles = {1: ('#2a78d6', '-'), 3: ('#eb6834', '--'), 5: ('#1baf7a', '-.')}

d = D[1]
X, Y = np.meshgrid(np.log10(d['theta']), np.log10(d['mu0']))
Z = 1e4 * np.array(d['gain'])
Zc = np.clip(Z, 0.0, None)
pos_cmap = LinearSegmentedColormap.from_list('pos', ['#e8e7e3', '#a9c8ef', '#2a78d6', '#1c5aa6'])
fc = pos_cmap(Normalize(0, Zc.max())(Zc)); fc[Z < 0] = to_rgba(NEG)

fig = plt.figure(figsize=(7.0, 4.3))
ax = fig.add_axes([0.0, 0.0, 0.68, 1.0], projection='3d')
ax.set_proj_type('ortho')
ax.plot_surface(X, Y, Zc, facecolors=fc, rstride=1, cstride=1, linewidth=0.15,
                edgecolor=(1, 1, 1, 0.35), antialiased=True, shade=False)
ax.contour(X, Y, Z, levels=[0.0], colors=INK, linewidths=1.3)            # window edge on the surface
k = np.unravel_index(np.argmax(Z), Z.shape)
ax.scatter([X[k]], [Y[k]], [Z[k]], color=INK, s=12, depthshade=False)
ax.text(X[k], Y[k], Z[k] + 0.3, rf'$\mu_0\approx{10**Y[k]:.2f}$, $\Theta\approx{10**X[k]:.2f}$' + '\n' +
        rf'$\Delta M_\chi\approx{Z[k]:.1f}\times10^{{-4}}$', fontsize=7.2, color=INK, ha='center', va='bottom')
ax.set_zlim(0, Zc.max() + 0.9); ax.set_zticks([0, 1, 2, 3])
ax.set_xticks([-2, -1, 0]); ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
ax.set_yticks([-2, -1, 0]); ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
ax.set_xlabel(r'Kerr strength $\Theta$', labelpad=2); ax.set_ylabel(r'Mean photon number $\mu_0$', labelpad=2)
ax.set_zlabel('Positive margin gain\n($10^{-4}$ bits per accepted slot)', labelpad=8)
ax.view_init(elev=36, azim=-125)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((0.82, 0.82, 0.80, 1))
    axis._axinfo['grid']['color'] = (0.89, 0.89, 0.87, 1); axis._axinfo['grid']['linewidth'] = 0.4

# top-view inset: window edges for three baseline errors
ins = fig.add_axes([0.73, 0.50, 0.25, 0.40])
ins.contourf(X, Y, Z, levels=[-1e9, 0.0], colors=[NEG])
for ed, (c, ls) in styles.items():
    dd = D[ed]; Xe, Ye = np.meshgrid(np.log10(dd['theta']), np.log10(dd['mu0']))
    ins.contour(Xe, Ye, np.array(dd['gain']), levels=[0.0], colors=c, linestyles=ls, linewidths=1.4)
ins.plot(X[k], Y[k], 'o', color=INK, ms=3)
ins.set_xticks([-2, -1, 0]); ins.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$'], fontsize=7)
ins.set_yticks([-2, -1, 0]); ins.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$'], fontsize=7)
ins.set_xlabel(r'$\Theta$', fontsize=8, labelpad=1); ins.set_ylabel(r'$\mu_0$', fontsize=8, labelpad=1)
ins.set_title('Top view', fontsize=8)
handles = [Line2D([], [], color=c, ls=ls, lw=1.5, label=rf'window edge, $e_d={ed}\%$') for ed, (c, ls) in styles.items()]
handles.append(Patch(color=NEG, label='negative margin gain,\nclipped at zero'))
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.70, 0.38), frameon=False, fontsize=7.2)
fig.savefig(root / 'paper' / 'figures' / 'window_surface.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig(root / 'paper' / 'figures' / 'window_surface.png', dpi=170, bbox_inches='tight', pad_inches=0.05)
print('max', Z[k], 'at Theta', 10 ** X[k], 'mu0', 10 ** Y[k])
