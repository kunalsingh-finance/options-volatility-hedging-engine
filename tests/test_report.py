"""Checks for the report's data transport and offline artifact guarantees."""

import json
import re

import pytest

from options_engine.report import write_report


def test_embedded_json_cannot_terminate_its_script_element(tmp_path):
    result = {
        "metadata": {
            "source": '</script><script>alert("untrusted source")</script>',
            "label": "BTC\u2028ETH\u2029<>&",
        },
        "currencies": {},
    }
    path = tmp_path / "nested" / "report.html"
    write_report(result, path)
    html = path.read_text(encoding="utf-8")
    payload = re.search(
        r'<script id="research-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    ).group(1)
    assert json.loads(payload) == result
    assert "<" not in payload
    assert '\u2028' not in payload and '\u2029' not in payload
    assert html.count("<script") == 2


def test_report_has_no_external_runtime_dependencies(tmp_path):
    path = tmp_path / "report.html"
    write_report({}, path)
    html = path.read_text(encoding="utf-8")
    resources = re.findall(
        r'''<(?:script|link|img|iframe)\b[^>]*(?:src|href)=["']([^"']+)''',
        html,
    )
    assert resources and all(url.startswith("data:") for url in resources)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_nonfinite_results_are_rejected_instead_of_silently_exported(tmp_path):
    with pytest.raises(ValueError, match="Out of range float"):
        write_report({"calibration_error": float("nan")}, tmp_path / "report.html")
