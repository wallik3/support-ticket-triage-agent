"""
Data store: loads customers.csv and error_logs.csv at startup.

CSV layout
──────────
customers.csv  — one row per customer; keyed on customer_id
error_logs.csv — technical error logs; keyed on customer_id + timestamp
                 columns: log_id, session_id, level, service, error_code,
                          message, affected_component
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(_DATA_DIR / "customers.csv")
    logs = pd.read_csv(_DATA_DIR / "error_logs.csv", parse_dates=["timestamp"])
    return customers, logs


# Loaded once at import time — fast for all subsequent calls.
_customers_df, _logs_df = _load_frames()


def _sanitise(record: dict) -> dict:
    """Replace float NaN with None for JSON safety."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}


def get_customer_profile(customer_id: str) -> dict | None:
    """Return customer profile as a plain dict, or None if not found."""
    row = _customers_df[_customers_df["customer_id"] == customer_id]
    if row.empty:
        return None
    return _sanitise(row.iloc[0].to_dict())


def get_recent_logs(
    customer_id: str,
    n: int = 10,
    before: datetime | None = None,
) -> list[dict]:
    """Return the n most recent error log entries for a customer.

    Args:
        before: Exclude logs at or after this timestamp for reproducible test results.
    """
    rows = _logs_df[_logs_df["customer_id"] == customer_id]
    if before is not None:
        rows = rows[rows["timestamp"] < pd.Timestamp(before)]
    rows = rows.sort_values("timestamp", ascending=False).head(n)
    raw = rows.to_dict(orient="records")
    records = []
    for r in raw:
        r = _sanitise(r)
        if r.get("timestamp") is not None:
            r["timestamp"] = str(r["timestamp"])
        records.append(r)
    return records
