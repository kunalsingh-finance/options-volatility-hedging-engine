"""Mutations that must not receive a clean research audit."""
import copy
import json
from pathlib import Path
import pytest
from options_engine.validation import validate

@pytest.fixture
def result():
    return json.loads((Path(__file__).resolve().parents[1]/'output'/'research.json').read_text(encoding='utf-8'))

def test_false_paired_confidence_interval_is_rejected(result):
    result['hedge']['paired_comparisons'][0]['ci95_low']=999
    result['hedge']['paired_comparisons'][0]['ci95_high']=1000
    assert not validate(result)['passed']

def test_missing_saved_ledger_is_rejected(result):
    result['hedge']['sample_ledgers']=result['hedge']['sample_ledgers'][:1]
    assert not validate(result)['passed']

def test_corrupt_option_mark_is_rejected(result):
    result['hedge']['sample_ledgers'][0]['rows'][1]['option_value']+=100
    assert not validate(result)['passed']
