"""Heston forward model using normalized characteristic function and Lewis integration."""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad
from . import black


@lru_cache(maxsize=16)
def quadrature(nodes, upper):
    x,w = roots_legendre(nodes)
    return (x+1)*upper/2, w*upper/2


def characteristic(u, maturity, params):
    """E[exp(i*u*log(F_T/F_0))] with zero drift, little-trap branch."""
    kappa,theta,sigma,rho = [float(params[k]) for k in ('kappa','theta','sigma','rho')]
    if not all(np.isfinite(x) for x in (kappa,theta,sigma,rho)):
        raise ValueError('finite Heston parameters required')
    v0 = np.asarray(params['v0'],dtype=float)
    t = np.asarray(maturity,dtype=float)
    u = np.asarray(u,dtype=complex)
    exact_moment=(u==0)|(u==-1j)
    # At u=-i the algebraic ratio is 0/0 when kappa-rho*sigma<0.
    # Both these characteristic-function moments are exactly one.
    u=np.where(exact_moment,1e-12+0j,u)
    if sigma < 1e-4:
        integrated = theta*t+(v0-theta)*(-np.expm1(-kappa*t))/kappa
        return np.where(exact_moment,1.,np.exp(-.5*(u*u+1j*u)*integrated))
    b = kappa-rho*sigma*1j*u
    d = np.sqrt(b*b+sigma*sigma*(u*u+1j*u))
    d = np.where(np.real(d)<0,-d,d)
    g = (b-d)/(b+d)
    e = np.exp(-d*t)
    c = kappa*theta/(sigma*sigma)*((b-d)*t-2*np.log((1-g*e)/(1-g)))
    dd = (b-d)/(sigma*sigma)*(1-e)/(1-g*e)
    return np.where(exact_moment,1.,np.exp(c+dd*v0))


def price(forward, strike, maturity, params, option_type='call', nodes=128, upper=200.0):
    f,k,t,v0 = np.broadcast_arrays(np.asarray(forward,float),np.asarray(strike,float),
                                   np.asarray(maturity,float),np.asarray(params['v0'],float))
    if option_type not in ('call','put'):
        raise ValueError('call or put required')
    if not all(np.all(np.isfinite(a)) for a in (f,k,t,v0)):
        raise ValueError('nonfinite pricing input')
    if np.any(f<=0) or np.any(k<=0) or np.any(t<0) or np.any(v0<0):
        raise ValueError('invalid F,K,T,v0')
    if not (params['kappa']>0 and params['theta']>0 and params['sigma']>=0 and abs(params['rho'])<1):
        raise ValueError('invalid Heston parameters')
    if not all(np.isfinite(params[name]) for name in ('kappa','theta','sigma','rho')):
        raise ValueError('finite Heston parameters required')
    shape=f.shape
    f,k,t,v0=[a.reshape(-1) for a in (f,k,t,v0)]
    if params['sigma']<1e-4:
        ivar=params['theta']*t+(v0-params['theta'])*(-np.expm1(-params['kappa']*t))/params['kappa']
        return black.price(f,k,t,np.sqrt(ivar/np.maximum(t,1e-300)),option_type).reshape(shape)
    u,w=quadrature(int(nodes),float(upper))
    p=dict(params,v0=v0[:,None])
    cf=characteristic(u[None,:]-.5j,t[:,None],p)
    integrand=np.real(np.exp(1j*np.log(f/k)[:,None]*u)*cf)/(u*u+.25)
    call=f-np.sqrt(f*k)/np.pi*(integrand@w)
    # Exact terminal payoff; do not clamp invalid numerical prices into valid bounds.
    call=np.where(t==0,np.maximum(f-k,0),call)
    answer=call if option_type=='call' else call-f+k
    return answer.reshape(shape)


def delta(forward, strike, maturity, params, option_type='call', nodes=128, upper=200.0):
    f=np.asarray(forward,float)
    bump=np.maximum(f*1e-4,1e-6)
    return (price(f+bump,strike,maturity,params,option_type,nodes,upper)-
            price(f-bump,strike,maturity,params,option_type,nodes,upper))/(2*bump)


def greeks(forward,strike,maturity,params):
    f=float(forward); h=f*.001
    p=float(price(f,strike,maturity,params,nodes=256))
    up=float(price(f+h,strike,maturity,params,nodes=256))
    dn=float(price(f-h,strike,maturity,params,nodes=256))
    vol=np.sqrt(params['v0']); dv=.0005
    vp=float(price(f,strike,maturity,dict(params,v0=(vol+dv)**2),nodes=256))
    vm=float(price(f,strike,maturity,dict(params,v0=max(vol-dv,1e-6)**2),nodes=256))
    return dict(delta=(up-dn)/(2*h),gamma=(up-2*p+dn)/(h*h),vega=(vp-vm)/(2*dv))


def delta_adaptive(forward,strike,maturity,params):
    """Scalar independent P1 integration with oscillatory tails; returns delta,error.

    Uses the exact Heston forward delta P1 instead of price finite differences.
    Useful when near-zero variance and short expiry require large Fourier cutoffs.
    """
    x=float(np.log(forward/strike)); t=float(maturity)
    if t<=0: return (float(forward>strike)+.5*float(forward==strike),0.)
    def cf(u):return complex(characteristic(u-1j,t,params))
    def integrand(u):
        u=max(u,1e-10)
        return float(np.real(np.exp(1j*u*x)*cf(u)/(1j*u)))
    head,head_error=quad(integrand,0,1,epsabs=1e-9,epsrel=1e-9,limit=200)
    if abs(x)>1e-6:
        cos_value,cos_error=quad(lambda u:cf(u).imag/u,1,np.inf,weight='cos',wvar=abs(x),epsabs=2e-8,limlst=300,limit=300)
        sin_value,sin_error=quad(lambda u:cf(u).real/u,1,np.inf,weight='sin',wvar=abs(x),epsabs=2e-8,limlst=300,limit=300)
        tail=cos_value+np.sign(x)*sin_value;error=head_error+cos_error+sin_error
    else:
        tail,tail_error=quad(integrand,1,np.inf,epsabs=2e-8,epsrel=2e-8,limit=1000)
        error=head_error+tail_error
    return float(.5+(head+tail)/np.pi),float(error/np.pi)
