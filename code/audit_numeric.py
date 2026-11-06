"""Finite-key and optical-channel numerical kernels.

Equations for finite-size estimation: Mizutani, Takeuchi & Tamaki,
Phys. Rev. Research 5, 023132 (2023), Eqs. (42)-(49).
Reconstructed from the printed equations during manuscript revision.
See AUTHOR_NOTES.md for the unresolved source-benchmark comparison.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import gammainc, xlogy
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.integrate import quad

LAM=3+np.sqrt(5)
EPS=2.**(-58)/6
LOGEPS=np.log(EPS)
ALPHA=.2*np.log(10)/10
HBAR=1.054571817e-34
HPLANCK=6.62607015e-34
LIGHT=299792458.
WAVELENGTH=1550e-9
PHOTON=HPLANCK*LIGHT/WAVELENGTH

def entropy(x):
    x=np.asarray(x)
    z=np.clip(x,0,.5)
    return -(xlogy(z,z)+xlogy(1-z,1-z))/np.log(2)

def kato(n,m):
    ln=LOGEPS
    v=9*m*(n-m)-2*n*ln
    a=(216*np.sqrt(n)*m*(n-m)*ln-48*n**1.5*ln**2
       +27*np.sqrt(2)*(n-2*m)*np.sqrt(-n*n*ln*v))/(4*(9*n-8*ln)*v)
    a=max(-np.sqrt(n)/2,a)
    c=-ln*(4*a+3*np.sqrt(n))**2/(18*n)
    b=np.sqrt(a*a+c)
    ba=c/(b+a) if a>=0 else b-a
    return a,b,ba

def key(mu,t,eta,nem=1e13,ebit=.01,noise=None,asymptotic=False,details=False):
    s=mu*eta
    if noise is None:
        n=0.
    else:
        n=noise
        ebit=(s*.01+n/2)/(s+n)
    ndet=nem*2*(s+n)*np.exp(-2*(s+n))
    nc=t*ndet
    q=gammainc(np.arange(1,4),3*mu)
    if asymptotic:
        eph=LAM*ebit+(q[1]+LAM*np.sqrt(q[0]*q[2]))/(ndet/nem)
        terms={}
    else:
        ch=.5*(-LOGEPS+np.sqrt(LOGEPS**2-8*t*q*nem*LOGEPS))
        counts=t*q*nem+ch
        delta=np.sqrt(-2*ndet*LOGEPS)
        d=max(LAM/(1-t)+1,1/t+LAM+1)
        mstar=min(counts[2],np.floor((ndet-1)/2))
        if mstar<0: return -np.inf
        a,b,ba=kato(ndet,mstar)
        k3=counts[2]*(1+2*a/np.sqrt(ndet))+ba*np.sqrt(ndet)
        bit=LAM*ebit*nc
        multiphoton=counts[1]
        root=LAM*np.sqrt((counts[0]+delta)*k3)
        azuma=t*d*delta
        nph=bit+multiphoton+root+azuma
        eph=nph/nc
        terms=dict(bit=float(bit),multiphoton=float(multiphoton),root=float(root),azuma=float(azuma),a=float(a),b=float(b),mstar=float(mstar))
    ell=nc*(1-entropy(eph)-1.16*entropy(ebit))-(0 if asymptotic else 86)
    if details:
        return dict(mu=float(mu),t=float(t),eta=float(eta),nem=float(nem),noise=float(n),ebit=float(ebit),ndet=float(ndet),ncode=float(nc),eph=float(eph),ell=float(ell),rate=float(ell/(3*nem)),terms=terms)
    return ell/(3*nem)

def optimize(eta,nem=1e13,noise=None,asymptotic=False):
    scale=max(eta**2*1e-3,1e-13)
    if asymptotic:
        # t -> 1 with a vanishing test fraction, while the number of tests diverges.
        # This is the true asymptotic limit of the selected phase-error bound.
        fits=[]
        for factor in [.003,.009,.015]:
            fit=minimize(lambda z:-key(10**z[0],1.,eta,nem,noise=noise,asymptotic=True)/scale,
                         [np.log10(max(factor*eta,1e-8))],method='Nelder-Mead',bounds=[(-9,-.5)],
                         options={'xatol':1e-11,'fatol':1e-12,'maxiter':1200})
            fits.append(fit)
        fit=min(fits,key=lambda z:z.fun)
        return key(10**fit.x[0],1.,eta,nem,noise=noise,asymptotic=True,details=True)
    def fun(x):
        return -key(10**x[0],x[1],eta,nem,noise=noise,asymptotic=asymptotic)/scale
    options=[]
    for start in [[np.log10(max(.009*eta,1e-8)),.85],[np.log10(max(.003*eta,1e-8)),.5],[np.log10(max(.015*eta,1e-8)),.7]]:
        fit=minimize(fun,start,method='Nelder-Mead',bounds=[(-9,-.5),(.001,.9999)],options={'xatol':1e-10,'fatol':1e-11,'maxiter':1200})
        options.append(fit)
    best=min(options,key=lambda z:z.fun)
    return key(10**best.x[0],best.x[1],eta,nem,noise=noise,asymptotic=asymptotic,details=True)

def channel(L,Pdbm=None,N=8,direction='forward',dark=1e-8,rho=3e-9):
    eta=np.exp(-ALPHA*L)*.8*10**(-.2)
    if Pdbm is None:
        nr=0.
    else:
        p=10**(Pdbm/10)*.001
        geom=L*np.exp(-ALPHA*L) if direction=='forward' else -np.expm1(-2*ALPHA*L)/(2*ALPHA)
        nr=N*p*rho*.2*geom*100e-12/PHOTON*.8*10**(-.2)
    return eta,nr+2*dark

def power_limit(L,N=8,direction='forward',dark=1e-8,asymptotic=False,nem=1e13):
    eta,n=channel(L,None,N,direction,dark)
    if optimize(eta,nem,noise=n,asymptotic=asymptotic)['rate']<=0: return None
    def func(p):
        e,nn=channel(L,p,N,direction,dark)
        return optimize(e,nem,noise=nn,asymptotic=asymptotic)['rate']
    p=brentq(func,-100,10,xtol=1e-7)
    e,n=channel(L,p,N,direction,dark)
    res=optimize(e,nem,noise=n,asymptotic=asymptotic)
    res['Pdbm']=p
    return res

def xpm(L=25,N=8,grid=100,clock=1,Pdbm=0,B=25.,gamma=1.3):
    # f is in GHz; d is in ns/km. S(f) converted to per GHz.
    p=.001*10**(Pdbm/10)
    lam_nm=WAVELENGTH*1e9
    sumphi=0.; sumdiff=0.
    per=[]
    for j in range(1,N+1):
        dl_nm=(WAVELENGTH**2/LIGHT)*(j*grid*1e9)*1e9
        d=17*dl_nm*.001
        def spectrum(f):
            z=ALPHA-2j*np.pi*f*d
            hf=2*gamma*(-np.expm1(-z*L))/z
            return abs(hf)**2*(p*p/B)*np.sinc(f/B)**2
        # Piecewise quadrature resolves walkoff oscillations.
        intervals=np.arange(0,10*B+1,1.)
        ph=2*sum(quad(spectrum,a,b,epsabs=1e-16)[0] for a,b in zip(intervals[:-1],intervals[1:]))
        df=2*sum(quad(lambda f:spectrum(f)*4*np.sin(np.pi*f/clock)**2,a,b,epsabs=1e-16)[0] for a,b in zip(intervals[:-1],intervals[1:]))
        sumphi+=ph; sumdiff+=df
        per.append(dict(j=j,d_ns_km=d,phase_variance=ph,differential_variance=df))
    return dict(e_xpm=-np.expm1(-sumdiff/2)/2,phase_variance=sumphi,differential_variance=sumdiff,ratio=np.sqrt(sumdiff/sumphi),correlation=1-sumdiff/(2*sumphi),per_channel=per)

