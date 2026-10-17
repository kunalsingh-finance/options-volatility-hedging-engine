"""Independently audit hashes, model outputs, and saved hedge ledgers offline."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from options_engine import data,validation


def main():
    output=ROOT/'output';status=json.loads((output/'run_status.json').read_text())
    if status['status']!='complete':raise AssertionError('Build status is not complete')
    for name,digest in status['artifacts_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,f'Changed artifact: {name}'
    result=json.loads((output/'research.json').read_text())
    for name,digest in result['metadata']['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Changed model code: {name}; rebuild'
    quotes,_=data.load_quotes(ROOT)
    saved={q['instrument']:q for c in result['currencies'].values() for q in c['quotes']}
    assert len(quotes)==len(saved)
    for q in quotes:
        for key in ('mid','bid','ask','forward','strike','maturity','iv'):
            assert abs(q[key]-saved[q['instrument']][key])<1e-10,f'Quote mismatch: {key}'
    check=validation.validate(result)
    assert check['passed'],[c for c in check['checks'] if not c['passed']]
    print(json.dumps(dict(status='passed',source_quotes=len(quotes),numeric_checks=len(check['checks']),artifact_hashes=len(status['artifacts_sha256']))))


if __name__=='__main__':main()
