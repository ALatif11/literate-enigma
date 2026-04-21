"""Data collector — pulls resolved markets from Kalshi and stores them in SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .client import KalshiClient

# Markets that have been fully settled with all payouts processed.
# 'finalized' is the terminal status; 'settled' is the step just before it.
RESOLVED_STATUSES = ("settled",)

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    ticker          TEXT PRIMARY KEY,
    event_ticker    TEXT,
    title           TEXT,
    category        TEXT,
    status          TEXT,
    result          TEXT,
    last_price_cents REAL,
    volume          REAL,
    open_time       TEXT,
    close_time      TEXT,
    duration_hours  REAL,
    collected_at    TEXT,
    raw_json        TEXT
);
"""


@contextmanager
def _db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    with _db(db_path) as conn:
        conn.executescript(SCHEMA)


def _parse_market(raw: dict) -> dict | None:
    """Normalize a raw API market object into our storage schema.

    Returns None if the market lacks the fields needed for backtesting
    (no result, no price, no close time).
    """
    result = raw.get("result", "")
    if result not in ("yes", "no"):
        return None

    # Prices come back as dollars (0.0–1.0); convert to cents (0–100).
    last_price_dollars = raw.get("last_price_dollars") or raw.get("last_price") or 0
    last_price_cents = float(last_price_dollars) * 100

    if last_price_cents <= 0:
        return None

    # Volume may be tagged _fp (floating-point) in newer API versions.
    volume = float(raw.get("volume_fp") or raw.get("volume") or 0)

    open_time = raw.get("open_time") or raw.get("created_time") or ""
    close_time = raw.get("close_time") or raw.get("expiration_time") or ""

    duration_hours = _duration_hours(open_time, close_time)

    return {
        "ticker": raw["ticker"],
        "event_ticker": raw.get("event_ticker", ""),
        "title": raw.get("title", ""),
        "category": raw.get("category", ""),
        "status": raw.get("status", ""),
        "result": result,
        "last_price_cents": last_price_cents,
        "volume": volume,
        "open_time": open_time,
        "close_time": close_time,
        "duration_hours": duration_hours,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "raw_json": json.dumps(raw),
    }


def _duration_hours(open_time: str, close_time: str) -> float:
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        t_open = datetime.strptime(open_time[:19], fmt[:len(fmt)])
        t_close = datetime.strptime(close_time[:19], fmt[:len(fmt)])
        return max(0.0, (t_close - t_open).total_seconds() / 3600)
    except Exception:
        return 0.0


def collect_markets(
    client: KalshiClient,
    db_path: str | Path,
    statuses: tuple[str, ...] = RESOLVED_STATUSES,
    page_size: int = 1000,
    verbose: bool = False,
) -> tuple[int, int]:
    """Fetch resolved markets and upsert them into the SQLite database.

    Returns (fetched_count, stored_count).
    """
    init_db(db_path)
    fetched = 0
    stored = 0

    with _db(db_path) as conn:
        for status in statuses:
            if verbose:
                print(f"  Fetching markets with status={status!r} ...")
            for raw in client.iter_markets(status=status, page_size=page_size):
                fetched += 1
                parsed = _parse_market(raw)
                if parsed is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO markets (
                        ticker, event_ticker, title, category, status, result,
                        last_price_cents, volume, open_time, close_time,
                        duration_hours, collected_at, raw_json
                    ) VALUES (
                        :ticker, :event_ticker, :title, :category, :status, :result,
                        :last_price_cents, :volume, :open_time, :close_time,
                        :duration_hours, :collected_at, :raw_json
                    )
                    ON CONFLICT(ticker) DO UPDATE SET
                        status          = excluded.status,
                        result          = excluded.result,
                        last_price_cents = excluded.last_price_cents,
                        volume          = excluded.volume,
                        close_time      = excluded.close_time,
                        duration_hours  = excluded.duration_hours,
                        collected_at    = excluded.collected_at,
                        raw_json        = excluded.raw_json
                    """,
                    parsed,
                )
                stored += 1

    return fetched, stored


def load_markets(db_path: str | Path) -> list[dict]:
    """Load all stored markets as a list of dicts."""
    with _db(db_path) as conn:
        rows = conn.execute("SELECT * FROM markets").fetchall()
    return [dict(row) for row in rows]
