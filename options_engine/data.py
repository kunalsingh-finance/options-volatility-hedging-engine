"""Bounded, auditable production Deribit snapshots and offline quote cleaning.

Native inverse-option premiums are in BTC/ETH. The USD-equivalent convention
multiplies each premium by that order book's index, while pricing uses its
expiry-specific ``underlying_price``. Mark prices are never quote substitutes.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode

import requests

API = "https://www.deribit.com/api/v2/public/"
DAY_MS = 86_400_000
YEAR_MS = 365 * DAY_MS
MAX_SPREAD = 0.7
MAX_AGE_MS = 120_000
MAX_SKEW_MS = 180_000
DOCUMENTATION = [
    "https://docs.deribit.com/api-reference/market-data/public-get_order_book",
    "https://docs.deribit.com/api-reference/market-data/public-get_instruments",
    "https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency",
    "https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options",
    "https://support.deribit.com/hc/en-us/articles/25944775983133-USD-Order-Options",
]


def _iso(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric quote")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric field")
    return number


def _safe_rejection(value: object) -> object:
    """Keep invalid values reviewable while emitting standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _safe_rejection(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_rejection(item) for item in value]
    return value


def _select(instruments: list[dict], summaries: list[dict], now_ms: int, currency: str) -> tuple[list[dict], dict]:
    """Select up to 11 quoted OTM strikes per expiry without using mark prices."""
    lookup = {s["instrument_name"]: s for s in summaries}
    eligible, excluded = [], Counter()
    for instrument in instruments:
        if not (instrument.get("kind") == "option" and instrument.get("is_active")
                and instrument.get("state", "open") == "open"
                and instrument.get("instrument_type") == "reversed"
                and instrument.get("settlement_currency") == currency
                and instrument.get("quote_currency") == currency):
            excluded["unsupported_or_inactive_instrument"] += 1
            continue
        days = (instrument["expiration_timestamp"] - now_ms) / DAY_MS
        if not 10 <= days <= 190:
            excluded["outside_capture_maturity_window"] += 1
            continue
        eligible.append(instrument)
    expiries = sorted({i["expiration_timestamp"] for i in eligible})
    chosen_expiries = []
    for target in (14, 35, 75, 150):
        remaining = [e for e in expiries if e not in chosen_expiries]
        if remaining:
            chosen_expiries.append(min(remaining, key=lambda e: abs((e - now_ms) / DAY_MS - target)))
    chosen = []
    for expiry in sorted(chosen_expiries):
        candidates = []
        for instrument in eligible:
            if instrument["expiration_timestamp"] != expiry:
                continue
            summary = lookup.get(instrument["instrument_name"], {})
            try:
                forward, strike = _number(summary.get("underlying_price")), _number(instrument["strike"])
                bid, ask = _number(summary.get("bid_price")), _number(summary.get("ask_price"))
                if min(forward, strike, bid, ask) <= 0 or ask < bid:
                    raise ValueError("invalid summary quote")
                if (ask - bid) / ((ask + bid) / 2) > MAX_SPREAD:
                    raise ValueError("wide summary quote")
            except (TypeError, ValueError):
                excluded["unusable_summary_quote"] += 1
                continue
            if instrument["option_type"] != ("put" if strike < forward else "call"):
                excluded["in_the_money_side"] += 1
                continue
            if abs(math.log(strike / forward)) > 0.5:
                excluded["outside_capture_log_moneyness"] += 1
                continue
            candidates.append((instrument, math.log(strike / forward)))
        # Evenly spaced log-moneyness targets provide coverage, not volume weights.
        selected = []
        for j in range(11):
            target = -0.25 + 0.05 * j
            remaining = [(i, k) for i, k in candidates if i not in selected]
            if remaining:
                selected.append(min(remaining, key=lambda pair: abs(pair[1] - target))[0])
        chosen.extend(sorted(selected, key=lambda i: i["strike"]))
    if len(chosen_expiries) != 4 or len({i["expiration_timestamp"] for i in chosen}) != 4:
        raise RuntimeError(f"Insufficient quoted expiries for {currency}; no synthetic substitution")
    return chosen, {"instrument_count": len(instruments), "summary_count": len(summaries),
                    "selected_count": len(chosen), "expiries": [_iso(e) for e in sorted(chosen_expiries)],
                    "selection_exclusions": dict(excluded)}


def capture(root: Path) -> dict:
    """Capture <=88 order books from production; commit manifest only on success.

    Failed captures retain their partial raw records with an explicit failed
    manifest in that capture directory. The last complete offline snapshot is
    preserved. No authentication, account access, or order endpoints are used.
    """
    root = Path(root).resolve()
    capture_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / "data" / "raw" / capture_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "source": "Deribit production public REST",
                "capture_id": capture_id, "retrieval_status": "in_progress",
                "documentation_urls": DOCUMENTATION, "files": [], "currencies": {},
                "conventions": {"premium": "coin premium multiplied by contemporaneous index_price",
                                "pricing_underlying": "reported expiry-specific underlying_price",
                                "discount_factor": 1.0, "day_count": "ACT/365",
                                "selection": "4 expiries nearest 14/35/75/150 days within 10-190 days; <=11 OTM strikes near log-moneyness targets -0.25 to 0.25",
                                "stale_after_seconds": MAX_AGE_MS / 1000,
                                "max_snapshot_skew_seconds": MAX_SKEW_MS / 1000,
                                "max_relative_spread": MAX_SPREAD}}
    session = requests.Session()
    session.headers.update({"User-Agent": "options-volatility-hedging-research/1.0", "Accept": "application/json"})
    last_request = 0.0

    def fetch(method: str, parameters: dict, label: str, kind: str, currency: str | None = None) -> dict:
        nonlocal last_request
        url = API + method + ("?" + urlencode(parameters) if parameters else "")
        # A conservative maximum of two requests/second, well below ordinary
        # public market-data limits. Metadata calls are separated by summaries.
        time.sleep(max(0, 0.5 - (time.monotonic() - last_request)))
        for attempt in range(3):
            last_request = time.monotonic()
            response = session.get(url, timeout=(10, 25))
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload or "result" not in payload:
                raise RuntimeError(f"Deribit API error at {method}: {payload.get('error')}")
            if payload.get("testnet") is not False:
                raise RuntimeError("Production response lacks explicit testnet=false")
            raw = response.content
            path = folder / (label + ".json")
            path.write_bytes(raw)
            entry = {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                     "bytes": len(raw), "kind": kind, "url": response.url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                     "server_timestamp_ms": payload.get("usOut", 0) // 1000}
            if currency:
                entry["currency"] = currency
            manifest["files"].append(entry)
            return payload
        raise RuntimeError("Unreachable retry exhaustion")

    try:
        clock = fetch("get_time", {}, "server_time", "clock")
        start_ms = int(clock["result"])
        manifest["capture_started_at_utc"] = _iso(start_ms)
        manifest["capture_timestamp"] = _iso(start_ms)
        selections = {}
        for currency in ("BTC", "ETH"):
            instruments = fetch("get_instruments", {"currency": currency, "kind": "option", "expired": "false"},
                                currency.lower() + "_instruments", "instruments", currency)
            summaries = fetch("get_book_summary_by_currency", {"currency": currency, "kind": "option"},
                              currency.lower() + "_summary", "summary", currency)
            selected, counts = _select(instruments["result"], summaries["result"], start_ms, currency)
            selections[currency] = selected
            manifest["currencies"][currency] = counts
        timestamps = []
        for currency, instruments in selections.items():
            for instrument in instruments:
                payload = fetch("get_order_book", {"instrument_name": instrument["instrument_name"], "depth": 1},
                                instrument["instrument_name"], "order_book", currency)
                timestamps.append(int(payload["result"]["timestamp"]))
        skew = max(timestamps) - min(timestamps)
        if skew > MAX_SKEW_MS:
            raise RuntimeError(f"Snapshot is not coherent: quote skew {skew / 1000:.1f}s exceeds {MAX_SKEW_MS / 1000:.0f}s")
        manifest.update({"retrieval_status": "complete", "capture_finished_at_utc": _iso(manifest["files"][-1]["server_timestamp_ms"]),
                         "first_quote_at_utc": _iso(min(timestamps)), "last_quote_at_utc": _iso(max(timestamps)),
                         "snapshot_skew_seconds": skew / 1000, "raw_count": len(timestamps),
                         "source_urls": sorted({f["url"] for f in manifest["files"]})})
        _json_write(folder / "manifest.json", manifest)
        _json_write(root / "data" / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"retrieval_status": "failed", "error": str(error)})
        _json_write(folder / "manifest.json", manifest)
        raise
    finally:
        session.close()


def _read_verified(root: Path, entry: dict) -> dict:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root / "data" / "raw"):
        raise ValueError("Manifest path must remain inside data/raw")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Raw data SHA-256 mismatch: {entry['path']}")
    payload = json.loads(raw)
    if payload.get("testnet") is not False or "error" in payload or "result" not in payload:
        raise ValueError(f"Invalid production payload: {entry['path']}")
    return payload


def _clean_quote(book: dict, instrument: dict, currency: str, server_ms: int, source: str) -> dict:
    from .black import implied_vol

    if book.get("instrument_name") != instrument.get("instrument_name"):
        raise ValueError("instrument_name_mismatch")
    if book.get("state") != "open" or instrument.get("is_active") is not True:
        raise ValueError("inactive_instrument")
    if not (instrument.get("kind") == "option" and instrument.get("instrument_type") == "reversed"
            and instrument.get("base_currency") == currency and instrument.get("settlement_currency") == currency
            and instrument.get("quote_currency") == currency):
        raise ValueError("unsupported_contract_or_premium_currency")
    try:
        stamp = int(_number(book["timestamp"]))
        expiry = int(_number(instrument["expiration_timestamp"]))
        forward, index, strike = (_number(book["underlying_price"]), _number(book["index_price"]), _number(instrument["strike"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing_or_nonfinite_market_field") from error
    if min(forward, index, strike) <= 0:
        raise ValueError("nonpositive_market_field")
    if server_ms - stamp > MAX_AGE_MS or stamp - server_ms > 5_000:
        raise ValueError("stale_or_future_quote_timestamp")
    maturity = (expiry - stamp) / YEAR_MS
    if not 7 / 365 < maturity <= 1:
        raise ValueError("maturity_outside_7_to_365_days")
    option_type = instrument.get("option_type")
    if option_type not in ("call", "put"):
        raise ValueError("invalid_option_type")
    try:
        bid_coin, ask_coin = _number(book.get("best_bid_price")), _number(book.get("best_ask_price"))
        bid_amount, ask_amount = _number(book.get("best_bid_amount")), _number(book.get("best_ask_amount"))
    except (TypeError, ValueError) as error:
        raise ValueError("missing_or_nonfinite_bid_ask") from error
    if min(bid_coin, ask_coin, bid_amount, ask_amount) <= 0:
        raise ValueError("nonpositive_or_empty_bid_ask")
    if ask_coin < bid_coin:
        raise ValueError("crossed_bid_ask")
    bid, ask = bid_coin * index, ask_coin * index
    mid = (bid + ask) / 2
    spread = ask - bid
    relative_spread = spread / mid
    if relative_spread > MAX_SPREAD:
        raise ValueError("relative_spread_above_0.7")
    intrinsic = max(forward - strike, 0) if option_type == "call" else max(strike - forward, 0)
    upper = forward if option_type == "call" else strike
    if not intrinsic <= bid <= ask < upper or not intrinsic < mid < upper:
        raise ValueError("black_forward_price_bounds")
    try:
        iv = float(implied_vol(mid, forward, strike, maturity, option_type=option_type))
    except (ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("implied_vol_inversion_failed") from error
    if not math.isfinite(iv) or iv <= 0:
        raise ValueError("invalid_implied_volatility")
    row = {"currency": currency, "instrument": book["instrument_name"], "expiry": _iso(expiry),
           "timestamp_ms": stamp, "maturity": maturity, "forward": forward, "index_price": index,
           "strike": strike, "option_type": option_type, "bid": bid, "ask": ask, "mid": mid,
           "bid_coin": bid_coin, "ask_coin": ask_coin, "iv": iv, "spread": spread,
           "relative_spread": relative_spread, "source": source}
    if not all(math.isfinite(value) for value in row.values() if isinstance(value, (float, int))):
        raise ValueError("nonfinite_derived_field")
    return row


def load_quotes(root: Path) -> tuple[list[dict], dict]:
    """Verify all raw hashes, clean selected executable-sided quotes, save rejects.

    Snapshot freshness is checked against original retrieval time, not today's
    clock: offline replay must remain deterministic after contracts expire.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("retrieval_status") != "complete":
        raise ValueError("A complete version-1 capture is required; run scripts/fetch_data.py explicitly")
    if not manifest.get("files"):
        raise ValueError("Capture manifest contains no raw files")
    records = [(entry, _read_verified(root, entry)) for entry in manifest["files"]]
    instruments = {}
    for entry, payload in records:
        if entry["kind"] == "instruments":
            instruments.update({i["instrument_name"]: i for i in payload["result"]})
    accepted, rejected, raw_by_currency = [], [], Counter()
    timestamps = []
    seen = set()
    for entry, payload in records:
        if entry["kind"] != "order_book":
            continue
        book = payload["result"]
        name = book.get("instrument_name", "unknown")
        currency = entry["currency"]
        raw_by_currency[currency] += 1
        try:
            if name in seen:
                raise ValueError("duplicate_instrument")
            seen.add(name)
            if name not in instruments:
                raise ValueError("missing_instrument_metadata")
            row = _clean_quote(book, instruments[name], currency, int(payload["usOut"] // 1000), entry["url"])
            accepted.append(row)
            timestamps.append(row["timestamp_ms"])
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"instrument": name, "currency": currency, "reason": str(error),
                             "source": entry["url"], "raw_file": entry["path"], "raw": _safe_rejection(book)})
    if timestamps and max(timestamps) - min(timestamps) > MAX_SKEW_MS:
        raise ValueError("Accepted quote timestamps exceed snapshot coherence limit")
    _json_write(root / "data" / "processed" / "rejected_quotes.json", rejected)
    if not accepted:
        raise ValueError("No usable observed quotes; refusing synthetic substitution")
    accepted.sort(key=lambda row: (row["currency"], row["expiry"], row["strike"]))
    _json_write(root / "data" / "processed" / "quotes.json", accepted)
    metadata = dict(manifest)
    metadata.update({"raw_count": sum(raw_by_currency.values()), "accepted_count": len(accepted),
                     "rejected_count": len(rejected), "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
                     "counts_by_currency": {c: {"raw": raw_by_currency[c],
                                                "accepted": sum(q["currency"] == c for q in accepted),
                                                "rejected": sum(q["currency"] == c for q in rejected)} for c in raw_by_currency},
                     "hashes_verified": True, "load_mode": "offline replay of observed snapshot",
                     "rejected_file": "data/processed/rejected_quotes.json",
                     "limitations": ["A bounded, sequential quote snapshot, not historical option data.",
                                     "Top-of-book quotes indicate available sides at capture; fills, market impact and historical hedge performance are not observed.",
                                     "USD equivalents use contemporaneous indices and zero-discount forward pricing; the hedge experiment does not replicate coin collateral.",
                                     "Book timestamps establish snapshot freshness, not the last modification time of each standing order."]})
    return accepted, metadata
