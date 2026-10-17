"""Build the complete offline research pack; explicit --refresh acquires new quotes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from datetime import datetime,timezone
import hashlib
import json
import platform
import traceback
import numpy as np
import scipy
from options_engine import data,calibration,surface


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def fingerprints():
    files=sorted(list((ROOT/'options_engine').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py')))
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--paths',type=int,default=256)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--seed',type=int,default=1729)
    args=parser.parse_args()
    output=ROOT/'output'; output.mkdir(exist_ok=True)
    if args.paths<32 or args.steps<12: parser.error('at least 32 paths and 12 steps required')
    lock=output/'.build.lock'
    try: lock_fd=lock.open('x')
    except FileExistsError: raise RuntimeError('Another build holds output/.build.lock. Inspect before removing a stale lock.')
    try:
        write_json(output/'run_status.json',dict(status='building',started_at=datetime.now(timezone.utc).isoformat()))
        (output/'report.html').write_text('<!doctype html><title>Research build in progress</title><h1>Research build in progress</h1><p>Results are not ready. Check run_status.json.</p>',encoding='utf-8')
        if args.refresh: data.capture(ROOT)
        quotes,metadata=data.load_quotes(ROOT)
        result=dict(metadata=metadata,currencies={})
        result['metadata'].update(title='Options Volatility & Dynamic Hedging',created_at=datetime.now(timezone.utc).isoformat(),
            python_version=platform.python_version(),numpy_version=np.__version__,scipy_version=scipy.__version__,
            code_sha256=fingerprints(),data_classification='Observed Deribit option snapshots; hypothetical model prices and simulated hedge paths.',
            holdout_convention='Every fourth strike within each expiry withheld. Cross-sectional interpolation check, not a future-date forecasting test.')
        for currency in ('BTC','ETH'):
            group=[q for q in quotes if q['currency']==currency]
            if len(group)<16 or len(set(q['expiry'] for q in group))<3:
                raise ValueError(f'Insufficient clean {currency} data: {len(group)} quotes')
            print(f'Calibrating {currency}: {len(group)} clean quotes',flush=True)
            result['currencies'][currency]=calibration.fit_currency(group)
            write_json(output/'calibration.json',result)
        from options_engine import hedge,report,validation
        btc=result['currencies']['BTC']
        atm_vol=float(surface.iv(0,30/365,btc['surface']))
        print(f'Simulating common hedge paths: {args.paths} paths x {args.steps} steps',flush=True)
        result['hedge']=hedge.run_experiment(btc['heston']['params'],atm_vol,args.seed,args.paths,args.steps)
        result['hedge']['assumptions']['calibration_currency']='BTC'
        result['validation']=validation.validate(result)
        write_json(output/'research.json',result)
        write_json(output/'validation.json',result['validation'])
        if not result['validation']['passed']:
            raise RuntimeError('Research validation failed; inspect output/validation.json')
        report.write_report(result,output/'report.html')
        # All numeric quote outputs are exported for independent downstream inspection.
        import csv
        rows=[q for c in result['currencies'].values() for q in c['quotes']]
        with (output/'quotes.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(q.keys() for q in rows))))
            writer.writeheader();writer.writerows(rows)
        manifest={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('research.json','validation.json','report.html','quotes.csv')}
        write_json(output/'run_status.json',dict(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),artifacts_sha256=manifest))
        print(json.dumps(dict(status='complete',checks=len(result['validation']['checks']),report=str(output/'report.html'))),flush=True)
    except Exception as exc:
        write_json(output/'run_status.json',dict(status='failed',error=str(exc),failed_at=datetime.now(timezone.utc).isoformat()))
        (output/'report.html').write_text('<!doctype html><title>Research build failed</title><h1>Research build failed</h1><p>Saved results are not approved. Inspect run_status.json and validation.json, then rebuild.</p>',encoding='utf-8')
        raise
    finally:
        lock_fd.close();lock.unlink(missing_ok=True)


if __name__=='__main__': main()
