"""Explicitly refresh the observed production snapshot (public read-only API)."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options_engine.data import capture, load_quotes

if __name__ == "__main__":
    capture(ROOT)
    quotes, metadata = load_quotes(ROOT)
    print(json.dumps({key: metadata[key] for key in (
        "capture_timestamp", "retrieval_status", "snapshot_skew_seconds", "raw_count",
        "accepted_count", "rejected_count", "counts_by_currency", "rejection_reasons")}, indent=2))
