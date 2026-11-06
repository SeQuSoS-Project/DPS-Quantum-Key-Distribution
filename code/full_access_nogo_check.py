import numpy as np
from math import factorial
def coh(alpha, N):
    v=np.array([np.exp(-abs(alpha)**2/2)*alpha**n/np.sqrt(float(factorial(n))) for n in range(N)],dtype=complex)
    return np.outer(v,v.conj())
def dephase_full(rho):
    N=rho.shape[0]; k=np.array([n*(n-1)//2 for n in range(N)])
    M=(k[:,None]==k[None,:]).astype(float)
    return rho*M
def S(r):
    w=np.linalg.eigvalsh(r); w=w[w>1e-15]; return -np.sum(w*np.log2(w))
def kerr(rho,th):
    N=rho.shape[0]; ph=np.exp(-1j*th*np.array([n*(n-1)/2 for n in range(N)]))
    return rho*np.outer(ph,ph.conj())
N=40
for mu in [0.05,0.1,0.2,0.5]:
    a=np.sqrt(mu); r0=coh(a,N); r1=coh(-a,N)
    d0,d1=dephase_full(r0),dephase_full(r1)
    D_coh=0.5*np.abs(np.linalg.eigvalsh(r0-r1)).sum()
    D_dep=0.5*np.abs(np.linalg.eigvalsh(d0-d1)).sum()
    chi_coh=S((r0+r1)/2)-(S(r0)+S(r1))/2
    chi_dep=S((d0+d1)/2)-(S(d0)+S(d1))/2
    usd_coh=1-np.exp(-2*mu); usd_dep_lb=2*mu*np.exp(-mu)
    # Bob photon-weighted error, full dephasing: mean field
    m=np.trace(np.diag(np.sqrt(np.arange(1,N)),1)@d0)  # Tr(a rho)
    eph=(1-abs(m)**2/mu)/2
    # check partial random Kerr (Gaussian theta, sd 0.5) lies above full dephasing
    ths=np.random.default_rng(1).normal(0.6,0.5,400)
    p0=sum(kerr(r0,t) for t in ths)/len(ths); p1=sum(kerr(r1,t) for t in ths)/len(ths)
    D_part=0.5*np.abs(np.linalg.eigvalsh(p0-p1)).sum()
    print(f"mu={mu}: D {D_coh:.5f}->{D_dep:.5f} (lb {2*np.sqrt(mu)*np.exp(-mu):.5f}, rel {1-D_dep/D_coh:.4%}); "
          f"chi {chi_coh:.5f}->{chi_dep:.5f} (rel {1-chi_dep/chi_coh:.4%}); USD {usd_coh:.5f}->>= {usd_dep_lb:.5f} (rel {1-usd_dep_lb/usd_coh:.4%}); "
          f"e_ph Bob {eph:.4f}; D_partialGauss {D_part:.5f}")
