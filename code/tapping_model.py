"""Joint optical model of DPS-QKD with a Kerr (SPM) segment before a passive tap.

Reconstructed from the equations of the manuscript (sections 3, 5-7 and appendices A-B):
lossy-Kerr state, pure-loss and acceptance-weighted tap maps, Bob's delay-interferometer
statistics, Eve's displacement (USD-type) receiver, Eve's delay interferometer, the
accepted-slot operators Omega_k, local information, Holevo quantity and trace distance,
and the equal-visibility Gaussian phase-diffusion control.
"""
import numpy as np
from math import lgamma
from scipy.linalg import expm
from scipy.optimize import minimize

# ---------------------------------------------------------------- basics
def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def _lfact(n):
    return np.array([lgamma(k + 1) for k in np.atleast_1d(n)])

def coherent_vec(gamma, D):
    n = np.arange(D)
    if abs(gamma) == 0:
        v = np.zeros(D, complex); v[0] = 1; return v
    logmag = -abs(gamma)**2 / 2 + n * np.log(abs(gamma)) - 0.5 * _lfact(n)
    return np.exp(logmag) * np.exp(1j * n * np.angle(gamma))

def f_kunz(z):
    z = np.asarray(z, complex)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-14
    out[nz] = (1 - np.exp(-z[nz])) / z[nz]
    return out

def kerr_state(mu0, A, theta, D, label=0):
    """Exact lossy-Kerr output (Kunz, Paris, Banaszek 2018), eq. (rho) of the paper."""
    n = np.arange(D)[:, None]; m = np.arange(D)[None, :]
    logamp = (n + m) / 2 * np.log(mu0) - 0.5 * (_lfact(np.arange(D))[:, None] + _lfact(np.arange(D))[None, :])
    rho = np.exp(logamp - mu0 - A * (n + m) / 2 + 0j)
    rho = rho * np.exp(-0.5j * theta * (n * (n - 1) - m * (m - 1)))
    if A > 0:
        rho = rho * np.exp(mu0 * A * f_kunz(A + 1j * theta * (n - m)))
    if label == 1:
        rho = rho * (-1.0) ** (n + m)
    return rho

def diffusion_state(mu, sigma2, D, label=0):
    """Coherent state with Gaussian phase diffusion of variance sigma2."""
    n = np.arange(D)[:, None]; m = np.arange(D)[None, :]
    logamp = (n + m) / 2 * np.log(mu) - 0.5 * (_lfact(np.arange(D))[:, None] + _lfact(np.arange(D))[None, :])
    rho = np.exp(logamp - mu + 0j) * np.exp(-sigma2 * (n - m) ** 2 / 2)
    if label == 1:
        rho = rho * (-1.0) ** (n + m)
    return rho

def _binom_sqrt(D):
    # S[n, l] = sqrt(C(n+l, l))
    n = np.arange(D)[:, None]; l = np.arange(D)[None, :]
    lf = lambda x: np.vectorize(lambda y: lgamma(y + 1))(x)
    return np.exp(0.5 * (lf(n + l) - lf(n) - lf(l)))

def weighted_loss(R, t, w, Dout):
    """[L(R)]_nm = t^{(n+m)/2} sum_l sqrt(C(n+l,l)C(m+l,l)) w^l R_{n+l,m+l}.
    w = 1-t gives the pure-loss channel; w = (1-r)(1-s_B), t = r gives chi_x."""
    D = R.shape[0]
    S = _binom_sqrt(D)
    out = np.zeros((Dout, Dout), complex)
    n = np.arange(Dout)
    for l in range(D):
        k = n + l
        ok = k < D
        if not ok.any():
            break
        idx = n[ok]
        sub = R[np.ix_(k[ok], k[ok])]
        fac = S[idx, l]
        out[np.ix_(idx, idx)] += (w ** l) * fac[:, None] * fac[None, :] * sub
    tp = t ** (n / 2)
    return out * tp[:, None] * tp[None, :]

def loss(R, t, Dout):
    return weighted_loss(R, t, 1 - t, Dout)

def mean_field(R):
    D = R.shape[0]
    a = np.diag(np.sqrt(np.arange(1, D)), 1)
    return np.trace(a @ R)

# ------------------------------------------------- balanced interferometer
_BS_CACHE = {}
def _bs_block(N):
    if N not in _BS_CACHE:
        G = np.zeros((N + 1, N + 1))
        for n in range(N):          # a1^dag a2 |n, N-n> = sqrt((n+1)(N-n)) |n+1, N-n-1>
            G[n + 1, n] = np.sqrt((n + 1) * (N - n))
        _BS_CACHE[N] = expm(np.pi / 4 * (G - G.T))
    return _BS_CACHE[N]

def port_vacuum_probs(R1, R2, phi=0.0):
    """For the (possibly unnormalised) two-mode product R1 (x) R2 entering a balanced
    beam splitter (phase phi on mode 2), return Tr, P(port c empty), P(port d empty),
    P(both empty). Port c is the bright port for equal coherent inputs."""
    D1, D2 = R1.shape[0], R2.shape[0]
    if phi != 0.0:
        ph = np.exp(-1j * phi * np.arange(D2))
        R2 = R2 * ph[:, None] * ph.conj()[None, :]
    p_first_empty = 0.0; p_second_empty = 0.0
    for N in range(D1 + D2 - 1):
        ns = [n for n in range(N + 1) if n < D1 and N - n < D2]
        if not ns:
            continue
        tau = np.zeros((N + 1, N + 1), complex)
        for i in ns:
            for j in ns:
                tau[i, j] = R1[i, j] * R2[N - i, N - j]
        U = _bs_block(N)
        out = U @ tau @ U.conj().T
        p_first_empty += out[0, 0].real      # |0, N>
        p_second_empty += out[N, N].real     # |N, 0>
    both = (R1[0, 0] * R2[0, 0]).real
    tr = (np.trace(R1) * np.trace(R2)).real
    return tr, p_first_empty, p_second_empty, both

def _bright_is_second():
    # determine which output port is bright for equal coherent inputs
    v = coherent_vec(0.5, 10); R = np.outer(v, v.conj())
    tr, e1, e2, b = port_vacuum_probs(R, R)
    return e1 > e2   # first output almost always empty -> it is the dark port
BRIGHT_SECOND = _bright_is_second()

def interferometer_records(R1, R2, eff, dark, phi=0.0):
    """Records (c only, d only, none, double) of a balanced interferometer with
    detectors of efficiency eff and dark probability dark; c is the bright port."""
    tr, e1, e2, b = port_vacuum_probs(R1, R2, phi)
    ec, ed = (e2, e1) if BRIGHT_SECOND else (e1, e2)   # P(c empty), P(d empty)
    uc, ud, u00 = (1 - dark) * ec, (1 - dark) * ed, (1 - dark) ** 2 * b
    c_only = ud - u00; d_only = uc - u00; none = u00; double = tr - uc - ud + u00
    return np.array([c_only, d_only, none, double]), uc, ud, u00

# ------------------------------------------------------------- parameters
class Link:
    def __init__(self, mu0, theta, r=0.5, A=0.0, D_km=50.0, alpha_db=0.2, eta_B=0.2,
                 p_B=1e-7, e_d=0.01, eta_E=1.0, p_E=0.0, f_EC=1.16, model='kerr',
                 Din=None, DE=None, DB=8):
        self.mu0, self.theta, self.r, self.A = mu0, theta, r, A
        self.tB = 10 ** (-alpha_db * D_km / 10)
        self.eta_B, self.p_B, self.e_d = eta_B, p_B, e_d
        self.eta_E, self.p_E, self.f_EC = eta_E, p_E, f_EC
        self.mu = mu0 * np.exp(-A)
        self.sB = self.tB * eta_B / 2
        self.eta = (1 - r) * self.tB * eta_B
        Din = Din or int(max(20, mu0 + 12 * np.sqrt(mu0) + 14))
        DE = DE or int(max(12, r * mu0 + 10 * np.sqrt(r * mu0) + 10))
        self.Din, self.DE, self.DB = Din, DE, DB
        # visibility of the Kerr state (eq. moment)
        m1 = np.sqrt(self.mu) * np.exp(-1j * mu0 * theta * f_kunz(A + 1j * theta))
        self.V = abs(m1) ** 2 / self.mu
        if model == 'kerr':
            self.rho = [kerr_state(mu0, A, theta, Din, x) for x in (0, 1)]
        elif model == 'diffusion':
            s2 = -np.log(self.V)
            self.rho = [diffusion_state(self.mu, s2, Din, x) for x in (0, 1)]
        else:
            raise ValueError(model)
        self.model = model
        self.rhoE = [loss(R, r, DE) for R in self.rho]
        self.chi = [weighted_loss(R, r, (1 - r) * (1 - self.sB), DE) for R in self.rho]
        self.GB = 1 - (1 - p_B) ** 2 * np.exp(-self.eta * self.mu)

    # ---------------- Bob
    def bob(self):
        s = self.eta / 2
        RB = loss(self.rho[0], s, self.DB)
        rec, uc, ud, u00 = interferometer_records(RB, RB, 1.0, self.p_B)
        GB = 1 - u00
        Q0 = 0.5 + (uc - ud) / (2 * GB)
        QB = self.e_d + (1 - 2 * self.e_d) * Q0
        return dict(GB=GB, Q0=Q0, QB=QB, eph=(1 - self.V) / 2)

    # ---------------- Omega_k components
    def omega_terms(self):
        """List of (k, weight, R1, R2) with Omega_k = sum weight * R1 (x) R2."""
        c = (1 - self.p_B) ** 2
        terms = []
        for x in (0, 1):
            for y in (0, 1):
                k = x ^ y
                terms.append((k, 0.5, self.rhoE[x], self.rhoE[y]))
                terms.append((k, -0.5 * c, self.chi[x], self.chi[y]))
        return terms

    def omega(self):
        Om = [np.zeros((self.DE ** 2,) * 2, complex) for _ in (0, 1)]
        for k, w, R1, R2 in self.omega_terms():
            Om[k] += w * np.kron(R1, R2)
        return Om

    # ---------------- information from a joint record distribution
    def _info_from_J(self, J):
        """J[k, z] unnormalised; returns local information 1 - H(K|Z)."""
        J = np.maximum(J, 0)
        G = J.sum()
        col = J.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(J > 0, J * np.log2(np.where(col > 0, J / col, 1)), 0)
        H = -t.sum() / G
        return 1 - H, H

    # ---------------- displacement receiver
    def _arm_state(self, R):
        if not hasattr(self, '_arm_cache'):
            self._arm_cache = {}
        key = id(R)
        if key not in self._arm_cache:
            self._arm_cache[key] = (loss(R, self.eta_E / 2, R.shape[0]), np.real(np.sum((1 - self.eta_E) ** np.arange(R.shape[0]) * np.diag(R))), np.real(np.trace(R)))
        return self._arm_cache[key]

    def disp_records(self, R, beta):
        g = np.sqrt(self.eta_E) * beta
        sig, vacw, tr = self._arm_state(R)
        vp = coherent_vec(-g, R.shape[0]); vm = coherent_vec(g, R.shape[0])
        qp = (1 - self.p_E) * np.real(vp.conj() @ sig @ vp)
        qm = (1 - self.p_E) * np.real(vm.conj() @ sig @ vm)
        q00 = (1 - self.p_E) ** 2 * np.exp(-2 * self.eta_E * abs(beta) ** 2) * vacw
        return np.array([qm - q00, qp - q00, q00, tr - qp - qm + q00])

    def disp_J(self, beta):
        F = {}
        J = np.zeros((2, 16))
        cache = {}
        for k, w, R1, R2 in self.omega_terms():
            for R in (R1, R2):
                if id(R) not in cache:
                    cache[id(R)] = self.disp_records(R, beta)
            J[k] += 0.5 * w * np.outer(cache[id(R1)], cache[id(R2)]).ravel()
        return J

    def disp_info(self, beta):
        return self._info_from_J(self.disp_J(beta))[0]

    def disp_optimise(self, start=None, objective='sifted'):
        b0 = np.sqrt(self.r * self.mu / 2)
        mE = np.sqrt(self.r) * mean_field(self.rho[0]) / 1.0
        f = (lambda b: -self.disp_info(b)) if objective == 'sifted' else (lambda b: -self.raw_info(b))
        starts = [b0, b0 * np.exp(1j * np.angle(mE)), 0.6 * b0, 1.5 * b0 * np.exp(1j * np.angle(mE))]
        if start is not None:
            starts.insert(0, start)
        best = None
        for s in starts:
            res = minimize(lambda v: f(v[0] + 1j * v[1]), [s.real, s.imag], method='Nelder-Mead',
                           options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
            if best is None or res.fun < best.fun:
                best = res
        return -best.fun, best.x[0] + 1j * best.x[1]

    def raw_info(self, beta):
        rec = self.disp_records(self.rhoE[0], beta)
        p = rec[0] + rec[1]
        q = rec[1] / p if p > 0 else 0
        Hraw = 1 - p ** 2 * (1 - h2(2 * q * (1 - q)))
        return 1 - Hraw

    # ---------------- delay-interferometer receiver
    def di_J(self, phi=0.0):
        J = np.zeros((2, 4))
        cache = {}
        for k, w, R1, R2 in self.omega_terms():
            key = (id(R1), id(R2))
            if key not in cache:
                S1 = loss(R1, self.eta_E / 2, R1.shape[0]); S2 = loss(R2, self.eta_E / 2, R2.shape[0])
                cache[key] = interferometer_records(S1, S2, 1.0, self.p_E, phi)[0]
            J[k] += 0.5 * w * cache[key]
        return J

    def di_info(self, phi=0.0):
        return self._info_from_J(self.di_J(phi))[0]

    def di_error(self, phi=0.0):
        J = self.di_J(phi)
        right = J[0, 0] + J[1, 1]; wrong = J[0, 1] + J[1, 0]
        return wrong / (right + wrong)

    # ---------------- receiver-independent bounds
    def holevo_trace(self):
        Om = self.omega()
        w = [O / self.GB for O in Om]
        def S(M):
            ev = np.linalg.eigvalsh((M + M.conj().T) / 2)
            ev = ev[ev > 1e-15]
            return -np.sum(ev * np.log2(ev))
        chi = S((w[0] + w[1]) / 2) - (S(w[0]) + S(w[1])) / 2
        T = 0.5 * np.sum(np.abs(np.linalg.eigvalsh((w[0] - w[1] + (w[0] - w[1]).conj().T) / 2)))
        return chi, T

    def single_pulse_trace(self):
        d = self.rhoE[0] - self.rhoE[1]
        return 0.5 * np.sum(np.abs(np.linalg.eigvalsh((d + d.conj().T) / 2)))

    def heralded_phase(self):
        mE = mean_field(self.rhoE[0])
        mEB = mean_field(self.rhoE[0] - self.chi[0]) / (1 - np.real(np.trace(self.chi[0])))
        return np.angle(mE), np.angle(mEB)
