"""Adversarial quote cleaning and provenance tests; no network required."""
from copy import deepcopy
import hashlib
import json

import pytest

from options_engine import data


NOW = 1_800_000_000_000


def sample():
    instrument = {"instrument_name": "BTC-FIXTURE-100-C", "kind": "option", "is_active": True,
                  "state": "open", "instrument_type": "reversed", "base_currency": "BTC",
                  "settlement_currency": "BTC", "quote_currency": "BTC", "strike": 100,
                  "expiration_timestamp": NOW + 30 * data.DAY_MS, "option_type": "call"}
    book = {"instrument_name": instrument["instrument_name"], "state": "open", "timestamp": NOW,
            "underlying_price": 100, "index_price": 80, "best_bid_price": 0.06,
            "best_ask_price": 0.065, "best_bid_amount": 1, "best_ask_amount": 2,
            "mark_price": 99999, "last_price": 12345}
    return book, instrument


def clean(book=None, instrument=None):
    default_book, default_instrument = sample()
    return data._clean_quote(book or default_book, instrument or default_instrument, "BTC", NOW, "fixture")


def test_usd_conversion_uses_index_not_forward_or_mark():
    row = clean()
    assert row["bid"] == pytest.approx(4.8)
    assert row["ask"] == pytest.approx(5.2)
    assert row["mid"] == pytest.approx(5.0)
    assert row["spread"] == pytest.approx(0.4)
    assert row["relative_spread"] == pytest.approx(0.08)
    assert row["maturity"] == pytest.approx(30 / 365)
    assert row["forward"] == 100
    assert row["iv"] > 0
    from options_engine.black import price
    assert float(price(row["forward"], row["strike"], row["maturity"], row["iv"])) == pytest.approx(row["mid"], abs=1e-8)


@pytest.mark.parametrize("change,reason", [
    ({"best_bid_price": None}, "missing_or_nonfinite_bid_ask"),
    ({"best_bid_price": float("nan")}, "missing_or_nonfinite_bid_ask"),
    ({"best_bid_price": 0}, "nonpositive_or_empty_bid_ask"),
    ({"best_ask_amount": 0}, "nonpositive_or_empty_bid_ask"),
    ({"best_bid_price": 0.07}, "crossed_bid_ask"),
    ({"best_ask_price": 0.15}, "relative_spread_above_0.7"),
    ({"index_price": 0}, "nonpositive_market_field"),
    ({"underlying_price": float("inf")}, "missing_or_nonfinite_market_field"),
    ({"timestamp": NOW - data.MAX_AGE_MS - 1}, "stale_or_future_quote_timestamp"),
    ({"timestamp": NOW + 5_001}, "stale_or_future_quote_timestamp"),
    ({"state": "closed"}, "inactive_instrument"),
    ({"best_bid_price": 1.26, "best_ask_price": 1.27}, "black_forward_price_bounds"),
])
def test_invalid_quote_does_not_use_mark_or_last(change, reason):
    book, instrument = sample()
    book.update(change)
    with pytest.raises(ValueError, match=reason):
        clean(book, instrument)


@pytest.mark.parametrize("days", [7, 366])
def test_maturity_limits(days):
    book, instrument = sample()
    instrument["expiration_timestamp"] = NOW + days * data.DAY_MS
    with pytest.raises(ValueError, match="maturity_outside"):
        clean(book, instrument)


def test_lower_intrinsic_bound_is_enforced():
    book, instrument = sample()
    instrument["strike"] = 90
    with pytest.raises(ValueError, match="black_forward_price_bounds"):
        clean(book, instrument)


def test_linear_contract_cannot_receive_inverse_conversion():
    book, instrument = sample()
    instrument["quote_currency"] = "USDC"
    with pytest.raises(ValueError, match="unsupported_contract"):
        clean(book, instrument)


def write_fixture(root, book_changes=None):
    folder = root / "data" / "raw" / "fixture"
    folder.mkdir(parents=True)
    book, instrument = sample()
    records = [("instruments", {"result": [instrument]})]
    records.append(("order_book", {"result": book}))
    if book_changes is not None:
        rejected_book = deepcopy(book)
        rejected_book.update(book_changes)
        rejected_book["instrument_name"] = "BTC-FIXTURE-101-C"
        rejected_instrument = deepcopy(instrument)
        rejected_instrument["instrument_name"] = rejected_book["instrument_name"]
        records[0][1]["result"].append(rejected_instrument)
        records.append(("order_book", {"result": rejected_book}))
    files = []
    for j, (kind, payload) in enumerate(records):
        payload.update({"testnet": False, "usOut": NOW * 1000})
        raw = json.dumps(payload).encode()
        path = folder / f"{j}.json"
        path.write_bytes(raw)
        files.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                      "kind": kind, "currency": "BTC", "url": "https://www.deribit.com/fixture"})
    manifest = {"schema_version": 1, "retrieval_status": "complete", "files": files}
    (root / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return files


def test_offline_replay_preserves_rejected_rows_and_counts(tmp_path):
    write_fixture(tmp_path, {"best_bid_price": None})
    quotes, metadata = data.load_quotes(tmp_path)
    assert len(quotes) == 1
    assert metadata["raw_count"] == 2
    assert metadata["accepted_count"] == 1
    assert metadata["rejected_count"] == 1
    assert metadata["hashes_verified"] is True
    rejects = json.loads((tmp_path / metadata["rejected_file"]).read_text())
    assert rejects[0]["reason"] == "missing_or_nonfinite_bid_ask"
    assert rejects[0]["raw"]["best_bid_price"] is None


def test_corrupt_raw_snapshot_fails_closed(tmp_path):
    files = write_fixture(tmp_path)
    path = tmp_path / files[0]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.load_quotes(tmp_path)


def test_nonfinite_rejections_remain_json_serializable(tmp_path):
    write_fixture(tmp_path, {"best_bid_price": float("nan")})
    _, metadata = data.load_quotes(tmp_path)
    rejects = json.loads((tmp_path / metadata["rejected_file"]).read_text())
    assert rejects[0]["raw"]["best_bid_price"] == "nan"
    assert rejects[0]["reason"] == "missing_or_nonfinite_bid_ask"


def test_testnet_cannot_be_loaded_even_with_valid_hash(tmp_path):
    files = write_fixture(tmp_path)
    path = tmp_path / files[0]["path"]
    payload = json.loads(path.read_bytes())
    payload["testnet"] = True
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    files[0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid production payload"):
        data.load_quotes(tmp_path)


def test_manifest_path_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="inside data/raw"):
        data._read_verified(tmp_path, {"path": "../outside.json", "sha256": ""})


def test_network_failure_does_not_replace_last_snapshot(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    prior = manifest_path.read_bytes()
    def fail(*args, **kwargs):
        raise data.requests.ConnectionError("fixture network unavailable")
    monkeypatch.setattr(data.requests.Session, "get", fail)
    with pytest.raises(data.requests.ConnectionError, match="network unavailable"):
        data.capture(tmp_path)
    assert manifest_path.read_bytes() == prior
    failures = list((tmp_path / "data" / "raw").glob("*/manifest.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["retrieval_status"] == "failed"
