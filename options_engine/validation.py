"""Independent numerical bounds and artifact-level research checks."""
import numpy as np
from . import heston,surface


def validate(result):
    checks=[]
    def check(name,passed,value=None,limit=None):
        checks.append(dict(name=name,passed=bool(passed),value=value,limit=limit))
    for currency,c in result['currencies'].items():
        diag=c['surface']['diagnostics']; hd=c['heston']['diagnostics']
        check(currency+' SSVI optimizer',diag['optimizer_success'])
        check(currency+' Heston optimizer',hd['optimizer_success'])
        check(currency+' butterfly sufficient condition',diag['butterfly_sufficient_margin']>0,diag['butterfly_sufficient_margin'],0)
        check(currency+' wing sufficient condition',diag['wing_sufficient_margin']>0,diag['wing_sufficient_margin'],0)
        check(currency+' nonnegative density grid',diag['min_density_factor']>=-1e-9,diag['min_density_factor'],-1e-9)
        check(currency+' calendar grid',diag['calendar_violations']==0,diag['calendar_violations'],0)
        check(currency+' monotone total variance',diag['monotone_theta'])
        check(currency+' holdout is present',c['metrics']['holdout_quotes']>=3,c['metrics']['holdout_quotes'],3)
        rows=c['quotes']; errs=[]; bounds=[]
        for q in rows:
            fine=float(heston.price(q['forward'],q['strike'],q['maturity'],c['heston']['params'],q['option_type'],256,400))
            errs.append(abs(fine-q['heston_price'])/q['forward'])
            lower=max((q['forward']-q['strike'])*(1 if q['option_type']=='call' else -1),0)
            upper=q['forward'] if q['option_type']=='call' else q['strike']
            bounds.extend(lower-1e-6*q['forward']<=q[m+'_price']<=upper for m in ('black','heston','ssvi'))
        check(currency+' fine-quadrature price stability',max(errs)<5e-6,float(max(errs)),5e-6)
        check(currency+' model price bounds',all(bounds))
        check(currency+' distinct source instruments',len({q['instrument'] for q in rows})==len(rows))
    hedge=result.get('hedge',{})
    cases=hedge.get('cases',[])
    check('hedge cases complete',len(cases)==24,len(cases),24)
    expected={(scenario,model,policy) for scenario in ('base','high_cost','volatility_stress','spot_jump') for model in ('black_scholes','heston') for policy in ('every_step','every_6_steps','delta_band')}
    def identities(rows):return [(r['scenario'],r['model'],r['policy']) for r in rows]
    for key in ('cases','sample_ledgers','distribution_samples'):
        ids=identities(hedge.get(key,[]))
        check(key+' identities complete and unique',len(ids)==len(expected) and set(ids)==expected)
    for case in cases:
        label=f"{case['scenario']} / {case['model']} / {case['policy']}"
        error=case.get('accounting_max_error',float('inf'))
        check(label+' self-financing accounting',error<1e-8,float(error),1e-8)
        check(label+' nonnegative costs',case['mean_cost']>=0,float(case['mean_cost']),0)
    # Reconstruct the saved trade and cash ledger without trusting its reported errors.
    ledger_errors=[]; attribution_errors=[]
    for ledger in hedge.get('sample_ledgers',[]):
        cash=ledger['premium']; held=0.; gross=ledger['premium']; costs=0.
        rows=ledger['rows']; previous_spot=rows[0]['spot']
        attribution=ledger['premium_gap']
        for i,row in enumerate(rows):
            gross+=held*(row['spot']-previous_spot)
            cash-=row['trade']*row['spot']+row['trade_cost']
            costs+=row['trade_cost'];held+=row['trade']
            if i==len(rows)-1:cash-=ledger['payoff']
            ledger_errors.extend([abs(cash-row['cash']),abs(held-row['held_delta'])])
            wealth=cash+held*row['spot']-(row['option_value'] if i<len(rows)-1 else 0.)
            ledger_errors.append(abs(wealth-row['wealth']))
            attribution+=sum(row.get(key,0.) for key in ('delta_attribution','gamma_attribution','theta_attribution','variance_attribution','attribution_residual'))
            previous_spot=row['spot']
        gross-=ledger['payoff']; attribution-=costs
        ledger_errors.extend([abs(gross-ledger['gross_pnl']),abs(gross-costs-ledger['net_pnl']),abs(held)])
        attribution_errors.append(abs(attribution-ledger['net_pnl']))
    check('independent saved cash-ledger reconstruction',bool(ledger_errors) and max(ledger_errors)<1e-8,float(max(ledger_errors,default=1e99)),1e-8)
    check('independent Greek attribution reconstruction',bool(attribution_errors) and max(attribution_errors)<1e-8,float(max(attribution_errors,default=1e99)),1e-8)
    distributions={(d['scenario'],d['model'],d['policy']):np.asarray(d['pnl']) for d in hedge.get('distribution_samples',[])}
    summary_errors=[]
    for case in cases:
        values=distributions.get((case['scenario'],case['model'],case['policy']),np.array([]))
        if len(values):summary_errors.extend([abs(float(values.mean())-case['mean_pnl']),abs(float(values.std(ddof=1))-case['pnl_std'])])
    check('saved distributions reconcile to summaries',bool(summary_errors) and max(summary_errors)<1e-10,float(max(summary_errors,default=1e99)),1e-10)
    paired_errors=[]
    expected_pairs={(s,p) for s in ('base','high_cost','volatility_stress','spot_jump') for p in ('every_step','every_6_steps','delta_band')}
    pair_ids=[(p['scenario'],p['policy']) for p in hedge.get('paired_comparisons',[])]
    check('paired comparison identities complete and unique',len(pair_ids)==12 and set(pair_ids)==expected_pairs)
    for pair in hedge.get('paired_comparisons',[]):
        diff=distributions[(pair['scenario'],pair['model_a'],pair['policy'])]-distributions[(pair['scenario'],pair['model_b'],pair['policy'])]
        mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
        paired_errors.extend([abs(mean-pair['mean_difference']),abs(se-pair['mc_se']),
                              abs(mean-1.96*se-pair['ci95_low']),abs(mean+1.96*se-pair['ci95_high'])])
    check('paired uncertainty uses common-path differences',bool(paired_errors) and max(paired_errors)<1e-10,float(max(paired_errors,default=1e99)),1e-10)
    hv=hedge.get('validation',{})
    dq=hv.get('delta_quadrature',{})
    check('hedging delta quadrature tolerance',dq.get('max_accepted_grid_difference',1.)<=2e-4,float(dq.get('max_accepted_grid_difference',1.)),2e-4)
    benchmark=hv.get('convergence_benchmark',{})
    for level in benchmark.get('levels',[]):
        z=abs(level['difference_from_analytic'])/max(level['mc_se'],1e-12)
        check(f"Monte Carlo price versus analytic at {level['steps']} steps",z<4,float(z),4)
    return dict(passed=all(c['passed'] for c in checks),checks=checks,
        interpretation='Numerical/accounting consistency under stated conventions. Does not establish market predictive accuracy, historical hedge performance, or regulatory approval.')
