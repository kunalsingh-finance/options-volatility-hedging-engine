"""Cross-sectional quote holdout and deterministic multi-start model calibration."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from . import black,heston,surface


def split_quotes(quotes):
    rows=[dict(q) for q in quotes]
    for expiry in set(q['expiry'] for q in rows):
        group=sorted([q for q in rows if q['expiry']==expiry],key=lambda q:q['strike'])
        for i,q in enumerate(group): q['split']='holdout' if i%4==1 else 'calibration'
    return rows


def fit_heston(quotes,uncertainty_refits=True):
    f,k,t,mid,spread=[np.array([q[key] for q in quotes]) for key in ('forward','strike','maturity','mid','spread')]
    calls=np.array([q['option_type']=='call' for q in quotes]); scales=np.maximum(spread/2,f*.0005)
    names=('kappa','theta','sigma','rho','v0')
    bounds=([.15,.01,.05,-.95,.01],[12,2.25,3,.5,2.25])
    median_vol=float(np.median([q['iv'] for q in quotes])); variance=float(np.clip(median_vol**2,.02,1.5))
    def residual(x):
        p=dict(zip(names,x)); call=heston.price(f,k,t,p)
        return (np.where(calls,call,call-f+k)-mid)/scales
    solutions=[]
    for start in ([1.5,variance,.5,-.4,variance],[4.,variance,1.2,-.7,variance],[.5,variance,.25,.1,variance]):
        result=least_squares(residual,start,bounds=bounds,max_nfev=220,ftol=2e-8,xtol=2e-8,gtol=2e-8)
        solutions.append((float(np.mean(result.fun**2)),result))
    solutions.sort(key=lambda x:x[0]); loss,best=solutions[0]; p={name:float(value) for name,value in zip(names,best.x)}
    errors=np.abs(heston.price(f,k,t,p,nodes=256,upper=400)-heston.price(f,k,t,p))
    runs=[dict(weighted_rmse=float(np.sqrt(cost)),success=bool(res.success),params={n:float(v) for n,v in zip(names,res.x)},nfev=int(res.nfev)) for cost,res in solutions]
    at_bounds=[n for n,v,lo,hi in zip(names,best.x,*bounds) if min(v-lo,hi-v)<1e-3*(hi-lo)]
    output=dict(params=p,diagnostics=dict(optimizer_success=bool(best.success),optimizer_message=best.message,
        weighted_rmse=float(np.sqrt(loss)),feller_margin=2*p['kappa']*p['theta']-p['sigma']**2,
        feller_condition=bool(2*p['kappa']*p['theta']>=p['sigma']**2),parameters_near_bounds=at_bounds,
        quadrature_max_error_usd=float(errors.max()),fit_quotes=len(quotes),starts=runs,
        identifiability_note='Three deterministic starts and bid/ask perturbations; one snapshot does not establish stable or unique structural parameters.'))
    if uncertainty_refits:
        refits=[]
        for direction in (-1,1):
            perturbed=[dict(q,mid=q['mid']+direction*.25*q['spread']) for q in quotes]
            refit=fit_heston(perturbed,False)
            refits.append(dict(quote_shift='mid minus quarter spread' if direction<0 else 'mid plus quarter spread',
                               params=refit['params'],weighted_rmse=refit['diagnostics']['weighted_rmse'],
                               optimizer_success=refit['diagnostics']['optimizer_success']))
        output['diagnostics']['quote_uncertainty_refits']=refits
    return output


def fit_currency(quotes):
    rows=split_quotes(quotes); train=[q for q in rows if q['split']=='calibration']
    s=surface.fit(train); h=fit_heston(train)
    bs_vols={}
    for expiry in set(q['expiry'] for q in train):
        group=[q for q in train if q['expiry']==expiry]
        def objective(vol):
            return sum(((float(black.price(q['forward'],q['strike'],q['maturity'],vol,q['option_type']))-q['mid'])/max(q['spread']/2,q['forward']*.0005))**2 for q in group)
        bs_vols[expiry]=float(minimize_scalar(objective,bounds=(.05,3),method='bounded').x)
    for q in rows:
        q['ssvi_iv']=float(surface.iv(np.log(q['strike']/q['forward']),q['maturity'],s))
        q['ssvi_price']=float(black.price(q['forward'],q['strike'],q['maturity'],q['ssvi_iv'],q['option_type']))
        q['heston_price']=float(heston.price(q['forward'],q['strike'],q['maturity'],h['params'],q['option_type']))
        q['black_price']=float(black.price(q['forward'],q['strike'],q['maturity'],bs_vols[q['expiry']],q['option_type']))
        q['black_iv']=bs_vols[q['expiry']]
    metrics={}
    for split in ('calibration','holdout'):
        group=[q for q in rows if q['split']==split]
        for model in ('black','ssvi','heston'):
            err=np.array([q[model+'_price']-q['mid'] for q in group]); scales=np.array([max(q['spread']/2,q['forward']*.0005) for q in group])
            metrics[f'{split}_{model}_rmse_usd']=float(np.sqrt(np.mean(err**2)))
            metrics[f'{split}_{model}_spread_scaled_rmse']=float(np.sqrt(np.mean((err/scales)**2)))
            metrics[f'{split}_{model}_inside_bid_ask']=float(np.mean([q['bid']<=q[model+'_price']<=q['ask'] for q in group]))
    metrics.update(calibration_quotes=len(train),holdout_quotes=len(rows)-len(train))
    return dict(quotes=rows,surface=s,heston=h,metrics=metrics,risk=risk_grid(h['params'],float(surface.iv(0,30/365,s))),
        risk_conventions='Hypothetical USD call F=K=100, T=30/365. Spot shock is relative. Vol shift changes sqrt(v0) and sqrt(theta); skew shifts rho. Greeks are Heston forward delta/gamma and vega per absolute change in sqrt(v0), theta held fixed. All results are precomputed.')


def risk_grid(params,atm_vol):
    baseline=float(heston.price(100,100,30/365,params,nodes=256)); rows=[]
    for spot in (-.2,-.1,0.,.1,.2):
        for vol in (-.1,0.,.1,.2):
            for skew in (-.15,0.,.15):
                p=dict(params,v0=max(.02,np.sqrt(params['v0'])+vol)**2,theta=max(.02,np.sqrt(params['theta'])+vol)**2,rho=float(np.clip(params['rho']+skew,-.98,.98)))
                f=100*(1+spot); hp=float(heston.price(f,100,30/365,p,nodes=256))
                g=heston.greeks(f,100,30/365,p)
                rows.append(dict(spot_shock=spot,vol_shock=vol,skew_shock=skew,effective_rho=p['rho'],heston_price=hp,
                    black_price=float(black.price(f,100,30/365,max(.02,atm_vol+vol))),pnl=hp-baseline,**g))
    return rows
