"""Compute the data behind Figures 3-7 and Tables 4-5 (tapping analysis).
Usage: python3 compute_tapping_figures.py <output json>"""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import brentq
from tapping_model import Link, h2, mean_field

F = 1.16
out = {}
t0 = time.time()

def adjusted(L, start=None):
    return L.disp_optimise(start=start)

# ---------------- Table 5 / benchmark at the main example
L0, L6, P6 = Link(0.1, 0.0), Link(0.1, 0.6), Link(0.1, 0.6, model='diffusion')
i0, b0 = adjusted(L0); i6, b6 = adjusted(L6); p6, _ = adjusted(P6)
c0, T0 = L0.holevo_trace(); c6, T6 = L6.holevo_trace(); cP, TP = P6.holevo_trace()
q0, q6 = L0.bob()['QB'], L6.bob()['QB']
dh = F * (h2(q6) - h2(q0))
out['benchmark'] = dict(
    QB=q6, GB=L6.bob()['GB'],
    disp=[i0, i6, p6], di=[L0.di_info(), L6.di_info(), P6.di_info()], holevo=[c0, c6, cP], trace=[T0, T6, TP],
    nominal=L6.disp_info(np.sqrt(L6.r * L6.mu / 2)),
    tracked=L6.disp_info(np.sqrt(L6.r * L6.mu / 2) * np.exp(1j * np.angle(mean_field(L6.rhoE[0])))),
    bob_entropy_growth=dh,
    eps=[(i0 - i6) / dh, (L0.di_info() - L6.di_info()) / dh, (c0 - c6) / dh])
print('benchmark done', time.time() - t0)

# ---------------- Table 4
rows = []
for mu0 in [0.05, 0.1, 0.2, 0.5, 1.0]:
    A0 = Link(mu0, 0.0); a0, _ = adjusted(A0); Q0 = A0.bob()['QB']
    row = dict(mu0=mu0)
    for q in [0.03, 0.05]:
        th = brentq(lambda t: Link(mu0, t).bob()['QB'] - q, 1e-4, 2.5, xtol=1e-8)
        A = Link(mu0, th); a, _ = adjusted(A)
        row[str(q)] = dict(theta=th, zeta=1 - a / a0, eps=(a0 - a) / (F * (h2(A.bob()['QB']) - h2(Q0))))
    rows.append(row)
out['table4'] = rows
print('table4 done', time.time() - t0)

# ---------------- Figure 3 (operating windows)
th_lin = np.linspace(0, 1, 21)
fig3 = dict(theta=th_lin.tolist(), a={}, b={}, c={}, d={})
base = Link(0.1, 0.0); bn0 = np.sqrt(base.r * base.mu / 2)
nom0 = base.disp_info(bn0); adj0, bstart = adjusted(base)
nom, trk, adj = [], [], []
for th in th_lin:
    L = Link(0.1, th); bn = np.sqrt(L.r * L.mu / 2); ph = np.angle(mean_field(L.rhoE[0]))
    nom.append(L.disp_info(bn) / nom0); trk.append(L.disp_info(bn * np.exp(1j * ph)) / nom0)
    v, bstart = adjusted(L, bstart); adj.append(v / adj0)
fig3['a'] = dict(nominal=nom, tracked=trk, adjusted=adj)
for mu0 in [0.05, 0.1, 0.5]:
    qb, HE, fh, M = [], [], [], []
    bstart = None
    for th in th_lin:
        L = Link(mu0, th); qb.append(L.bob()['QB'])
        v, bstart = adjusted(L, bstart)
        HE.append(1 - v); fh.append(F * h2(qb[-1])); M.append(HE[-1] - fh[-1])
    fig3['b'][str(mu0)] = qb; fig3['d'][str(mu0)] = M
    if mu0 == 0.1:
        fig3['c'] = dict(HE=HE, fh2=fh)
out['fig3'] = fig3
print('fig3 done', time.time() - t0)

# ---------------- Figure 4 (heralded cross-phase)
th_h = np.array([0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0])
allp, herp, betap, z_sift, z_raw, z_pd = [], [], [], [], [], []
r0, _ = L0.disp_optimise(objective='raw')
bstart = None
for th in th_h:
    L = Link(0.1, th); a, b = L.heralded_phase(); allp.append(a); herp.append(b)
    v, bstart = adjusted(L, bstart)
    ang = np.angle(bstart); ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    betap.append(ang); z_sift.append(1 - v / i0)
    rr, _ = L.disp_optimise(objective='raw'); z_raw.append(1 - rr / r0)
    P = Link(0.1, th, model='diffusion'); pv, _ = adjusted(P); z_pd.append(1 - pv / i0)
out['fig4'] = dict(theta=th_h.tolist(), phase_all=allp, phase_heralded=herp, phase_beta=betap,
                   zeta_sifted=z_sift, zeta_unsifted=z_raw, zeta_diffusion=z_pd,
                   theta_3pct=out['table4'][1]['0.03']['theta'])
print('fig4 done', time.time() - t0)

# ---------------- Figure 5 (interferometric eavesdropper)
eE, eP, eph, qb = [], [], [], []
for th in th_lin:
    L = Link(0.1, th); P = Link(0.1, th, model='diffusion')
    eE.append(L.di_error()); eP.append(P.di_error()); eph.append(L.bob()['eph']); qb.append(L.bob()['QB'])
th_log = np.logspace(-2, 0, 9)
epsDI = {}
for mu0 in [0.1, 0.5, 1.0]:
    K0 = Link(mu0, 0.0); d0 = K0.di_info(); Q0 = K0.bob()['QB']
    ek, ep = [], []
    for th in th_log:
        K = Link(mu0, th); P = Link(mu0, th, model='diffusion')
        ek.append((d0 - K.di_info()) / (F * (h2(K.bob()['QB']) - h2(Q0))))
        ep.append((d0 - P.di_info()) / (F * (h2(P.bob()['QB']) - h2(Q0))))
    epsDI[str(mu0)] = dict(kerr=ek, diffusion=ep)
out['fig5'] = dict(theta=th_lin.tolist(), eve_kerr=eE, eve_diffusion=eP, bob_eph=eph, bob_QB=qb,
                   theta_log=th_log.tolist(), eps=epsDI)
print('fig5 done', time.time() - t0)

# ---------------- Figure 6 (receiver-independent bounds)
chiK, chiP, diK, adK = [], [], [], []
bstart = None
for th in th_lin:
    L = Link(0.1, th); P = Link(0.1, th, model='diffusion')
    chiK.append(L.holevo_trace()[0]); chiP.append(P.holevo_trace()[0]); diK.append(L.di_info())
    v, bstart = adjusted(L, bstart); adK.append(v)
rs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 0.999]
red = {}
for th in [0.3, 0.6]:
    rc, rt = [], []
    for r in rs:
        B0 = Link(0.1, 0.0, r=r); B = Link(0.1, th, r=r)
        rc.append(1 - B.holevo_trace()[0] / B0.holevo_trace()[0])
        rt.append(1 - B.single_pulse_trace() / B0.single_pulse_trace())
    red[str(th)] = dict(chi=rc, trace=rt)
epsChi = {}
for mu0 in [0.1, 0.5, 1.0]:
    K0 = Link(mu0, 0.0); h0 = K0.holevo_trace()[0]; Q0 = K0.bob()['QB']
    ek, ep = [], []
    for th in th_log:
        K = Link(mu0, th); P = Link(mu0, th, model='diffusion')
        ek.append((h0 - K.holevo_trace()[0]) / (F * (h2(K.bob()['QB']) - h2(Q0))))
        ep.append((h0 - P.holevo_trace()[0]) / (F * (h2(P.bob()['QB']) - h2(Q0))))
    epsChi[str(mu0)] = dict(kerr=ek, diffusion=ep)
out['fig6'] = dict(theta=th_lin.tolist(), chi_kerr=chiK, chi_diffusion=chiP, di_kerr=diK, disp_kerr=adK,
                   r=rs, reduction=red, theta_log=th_log.tolist(), eps_chi=epsChi)
print('fig6 done', time.time() - t0)

# ---------------- Figure 7 (exchange ratio, displacement receiver)
th_e = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
epsD = {}
for mu0 in [0.1, 0.5, 1.0]:
    K0 = Link(mu0, 0.0); a0, bs = adjusted(K0); Q0 = K0.bob()['QB']
    ek, ep = [], []
    bk = bp = bs
    for th in th_e:
        K = Link(mu0, th); P = Link(mu0, th, model='diffusion')
        vk, bk = adjusted(K, bk); vp, bp = adjusted(P, bp)
        ek.append((a0 - vk) / (F * (h2(K.bob()['QB']) - h2(Q0))))
        ep.append((a0 - vp) / (F * (h2(P.bob()['QB']) - h2(Q0))))
    epsD[str(mu0)] = dict(kerr=ek, diffusion=ep)
mus = np.array([0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
small = {}
for r in [0.1, 0.5, 0.9]:
    vals, M0s = [], []
    for mu0 in mus:
        K0 = Link(mu0, 0.0, r=r); a0, bs = adjusted(K0); Q0 = K0.bob()['QB']
        K = Link(mu0, 0.05, r=r); vk, _ = adjusted(K, bs)
        vals.append((a0 - vk) / (F * (h2(K.bob()['QB']) - h2(Q0))))
        M0s.append((1 - a0) - F * h2(Q0))
    small[str(r)] = dict(eps=vals, M0=M0s)
out['fig7'] = dict(theta=th_e.tolist(), eps=epsD, mu0=mus.tolist(), small=small)
print('fig7 done', time.time() - t0)

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, complex) or isinstance(o, np.complexfloating): return [float(o.real), float(o.imag)]
    return o
json.dump(clean(out), open(sys.argv[1], 'w'), indent=1)
print('saved', time.time() - t0)
