"""Constrained SSVI surface, with sufficient static-arbitrage conditions."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from . import black


def total_variance(k,theta,rho,eta):
    theta=np.asarray(theta,float); k=np.asarray(k,float)
    if np.any(theta<=0) or abs(rho)>=1 or eta<0:
        raise ValueError('invalid SSVI parameters')
    phi=eta/np.sqrt(theta*(1+theta))
    return .5*theta*(1+rho*phi*k+np.sqrt((phi*k+rho)**2+1-rho*rho))


def density_factor(k,theta,rho,eta):
    k=np.asarray(k,float); theta=np.asarray(theta,float)
    phi=eta/np.sqrt(theta*(1+theta)); x=phi*k+rho
    root=np.sqrt(x*x+1-rho*rho)
    w=total_variance(k,theta,rho,eta)
    wk=.5*theta*phi*(rho+x/root)
    wkk=.5*theta*phi*phi*(1-rho*rho)/root**3
    return (1-k*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk


def theta_at(maturity,surface):
    """Linear monotone total-variance interpolation including (0,0); flat beyond last knot."""
    return np.interp(maturity,[0.]+surface['maturities'],[0.]+surface['theta'])


def iv(k,maturity,surface):
    t=np.asarray(maturity,float)
    if np.any(t<=0): raise ValueError('positive maturity required')
    return np.sqrt(total_variance(k,theta_at(t,surface),surface['rho'],surface['eta'])/t)


def fit(quotes):
    expiries=sorted(set(q['expiry'] for q in quotes),key=lambda e:np.median([q['maturity'] for q in quotes if q['expiry']==e]))
    maturities=np.array([np.median([q['maturity'] for q in quotes if q['expiry']==e]) for e in expiries])
    indices=np.array([expiries.index(q['expiry']) for q in quotes])
    initial=[]
    for expiry,t in zip(expiries,maturities):
        near_atm=sorted([q for q in quotes if q['expiry']==expiry],
                        key=lambda q:abs(np.log(q['strike']/q['forward'])))[:3]
        initial.append(float(np.median([q['iv'] for q in near_atm]))**2*t)
    initial=np.maximum.accumulate(initial)
    increments=np.maximum(np.diff(np.r_[0,initial]),1e-5)
    f,k,mid,scale=[np.array([q[key] for q in quotes]) for key in ('forward','strike','mid','spread')]
    scale=np.maximum(scale/2,f*.0005)
    types=np.array([q['option_type']=='call' for q in quotes])
    times=np.array([q['maturity'] for q in quotes])
    def unpack(x):
        th=np.cumsum(np.exp(x[:len(expiries)])); rho=.95*np.tanh(x[-2])
        eta=expit(x[-1])*np.sqrt(3.95/(1+abs(rho)))
        return th,rho,eta
    def residual(x):
        th,rho,eta=unpack(x)
        vol=np.sqrt(total_variance(np.log(k/f),th[indices],rho,eta)/times)
        call=black.price(f,k,times,vol)
        values=np.where(types,call,call-f+k)
        return (values-mid)/scale
    x0=np.r_[np.log(increments),-.3,-.5]
    result=least_squares(residual,x0,bounds=(np.r_[np.full(len(expiries),-16),-3,-7],np.r_[np.full(len(expiries),1),3,7]),max_nfev=800,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    th,rho,eta=unpack(result.x)
    s=dict(expiries=expiries,maturities=maturities.tolist(),theta=th.tolist(),rho=float(rho),eta=float(eta))
    kg=np.linspace(-3,3,601); tg=np.geomspace(max(maturities[0]/100,1e-6),maturities[-1]*3,100)
    ws=np.array([total_variance(kg,theta_at(t,s),rho,eta) for t in tg])
    min_g=min(float(np.min(density_factor(kg,theta_at(t,s),rho,eta))) for t in tg)
    s['diagnostics']=dict(optimizer_success=bool(result.success),optimizer_message=result.message,
        weighted_rmse=float(np.sqrt(np.mean(result.fun**2))),min_density_factor=min_g,
        calendar_violations=int(np.count_nonzero(np.diff(ws,axis=0)<-1e-12)),
        butterfly_sufficient_margin=float(4-eta*eta*(1+abs(rho))),
        wing_sufficient_margin=float(4-eta*(1+abs(rho))),
        monotone_theta=bool(np.all(np.diff(th)>=0)),fit_quotes=len(quotes),
        constraint_family='phi(theta)=eta/sqrt(theta*(1+theta)); eta^2*(1+|rho|)<4 and eta*(1+|rho|)<4')
    s['grid']=[dict(maturity=float(t),k=float(kval),iv=float(iv(kval,t,s))) for t in np.linspace(maturities[0],maturities[-1],16) for kval in np.linspace(-.5,.5,31)]
    return s
