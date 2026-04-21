"""FastAPI server — bridges the Python backtester modules to the web frontend.

Routes
------
GET  /                          Serves the single-page React frontend
GET  /api/status                System status (DB count, collect state, config check)
GET  /api/scan                  Live open markets from Kalshi, scored on-the-fly
GET  /api/backtest              Full backtest on stored historical data
POST /api/collect               Trigger background collection of resolved markets
GET  /api/market/{ticker}/ai    Claude AI analysis for a specific market (proxied)
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests as _requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .client import KalshiClient, KalshiAPIError
from .collector import collect_markets, load_markets, init_db, _duration_hours
from .scorer import score_markets, filter_by_threshold
from .trader import simulate_trades, _model_probability, _kelly_stake
from .reporter import _brier_score, _sharpe, _max_drawdown, _profit_factor

_STATIC = Path(__file__).parent / "static"
_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Kalshi Command Center")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_collect_state: dict = {"running": False, "last_run": None, "stored": 0, "error": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path() -> str:
    return os.getenv("KALSHI_DB_PATH", "kalshi_backtest.db")


def _get_client() -> KalshiClient | None:
    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    env = os.getenv("KALSHI_ENV", "demo")
    key_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "").strip()

    if not key_id:
        return None
    try:
        if key_path_str:
            return KalshiClient(key_id, Path(key_path_str).expanduser(), env=env)
        elif pem:
            return KalshiClient(key_id, pem.replace("\\n", "\n"), env=env)
    except Exception:
        pass
    return None


def _parse_open_market(raw: dict) -> dict | None:
    """Normalize a live open market for scoring."""
    last_price_dollars = raw.get("last_price_dollars") or raw.get("last_price") or 0
    last_price_cents = float(last_price_dollars) * 100
    if last_price_cents <= 0 or last_price_cents >= 100:
        return None

    volume = float(raw.get("volume_fp") or raw.get("volume") or 0)
    open_time = raw.get("open_time") or raw.get("created_time") or ""
    close_time = raw.get("close_time") or raw.get("expiration_time") or ""
    duration_hours = _duration_hours(open_time, close_time)

    hours_left = 0.0
    try:
        t_close = datetime.strptime(close_time[:19], "%Y-%m-%dT%H:%M:%S")
        hours_left = max(0.0, (t_close - datetime.utcnow()).total_seconds() / 3600)
    except Exception:
        pass

    # Skip markets expiring in less than 2 h (no time to act) or over 30 days
    if hours_left < 2 or hours_left > 720:
        return None

    return {
        "ticker": raw.get("ticker", ""),
        "event_ticker": raw.get("event_ticker", ""),
        "title": raw.get("title", raw.get("ticker", "")),
        "category": raw.get("category", ""),
        "status": "open",
        "result": None,
        "last_price_cents": last_price_cents,
        "volume": volume,
        "open_time": open_time,
        "close_time": close_time,
        "duration_hours": duration_hours,
        "hours_left": round(hours_left, 1),
        "anomaly": abs(last_price_cents - 50) > 30 and volume > 1000,
    }


def _market_to_response(m: dict, kelly_fraction: float = 0.25) -> dict:
    """Add derived fields used by the frontend."""
    entry = min(m["last_price_cents"], 100 - m["last_price_cents"])
    p_model = _model_probability(m.get("score", 0.0), entry)
    kelly_pct = 0.0
    if entry > 0:
        from .trader import _kelly_stake as ks
        # Express as % of bankroll (bankroll=1 for fraction)
        raw_k = ks(p_model, entry, 1.0, kelly_fraction)
        kelly_pct = round(raw_k * 100, 1)
    m["p_model"] = round(p_model, 3)
    m["kelly_pct"] = kelly_pct
    return m


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (_STATIC / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/api/status")
async def status():
    db = Path(_db_path())
    market_count = 0
    if db.exists():
        try:
            market_count = len(load_markets(str(db)))
        except Exception:
            pass
    return {
        "api_configured": _get_client() is not None,
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "env": os.getenv("KALSHI_ENV", "demo"),
        "market_count": market_count,
        "db_path": str(db),
        "collecting": _collect_state["running"],
        "last_collected": _collect_state["last_run"],
        "last_stored": _collect_state["stored"],
        "collect_error": _collect_state["error"],
    }


@app.get("/api/scan")
async def scan(
    limit: int = Query(40),
    kelly_fraction: float = Query(0.25),
):
    """Fetch and score live open markets from Kalshi."""
    client = _get_client()
    if not client:
        raise HTTPException(503, "Kalshi API not configured. Check KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in your .env.")

    loop = asyncio.get_event_loop()
    try:
        raw_markets = await loop.run_in_executor(
            _executor,
            lambda: list(client.iter_markets(status="open", page_size=500)),
        )
    except KalshiAPIError as exc:
        raise HTTPException(502, str(exc))

    parsed = [m for raw in raw_markets if (m := _parse_open_market(raw))]
    if not parsed:
        return {"markets": [], "total_open": len(raw_markets), "scored": 0}

    scored = score_markets(parsed)
    scored.sort(key=lambda m: m["score"], reverse=True)
    enriched = [_market_to_response(m, kelly_fraction) for m in scored[:limit]]
    return {"markets": enriched, "total_open": len(raw_markets), "scored": len(parsed)}


@app.get("/api/backtest")
async def backtest(
    threshold: float = Query(0.4),
    strategy: str = Query("underdog"),
    bankroll_cents: float = Query(0),
    kelly_fraction: float = Query(0.25),
    stake_cents: float = Query(100),
):
    db = _db_path()
    if not Path(db).exists():
        raise HTTPException(404, "No historical data. Run `kalshi-bt collect` or tap COLLECT in settings.")

    markets = load_markets(db)
    if not markets:
        raise HTTPException(404, "Database is empty. Run collect first.")

    scored = score_markets(markets)
    qualifying = filter_by_threshold(scored, threshold)
    trades = simulate_trades(qualifying, threshold, strategy, stake_cents, bankroll_cents, kelly_fraction)

    if not trades:
        return {"metrics": None, "trades": [], "calibration": [], "total_markets": len(markets)}

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    total_stake = sum(t.stake_cents for t in trades)
    total_profit = sum(t.profit_cents for t in trades)

    sharpe = _sharpe(trades)
    pf = _profit_factor(trades)

    # Calibration buckets
    buckets: dict[int, list] = defaultdict(list)
    for t in trades:
        buckets[int(t.p_model * 10) * 10].append(t)
    calibration = [
        {
            "bucket": f"{b}–{b+10}%",
            "count": len(bt),
            "avg_p_model": round(sum(t.p_model for t in bt) / len(bt), 3),
            "actual_win_rate": round(sum(t.won for t in bt) / len(bt), 3),
        }
        for b, bt in sorted(buckets.items())
    ]

    return {
        "total_markets": len(markets),
        "qualifying": len(qualifying),
        "metrics": {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4),
            "total_staked_cents": round(total_stake, 2),
            "net_pnl_cents": round(total_profit, 2),
            "roi": round(total_profit / total_stake if total_stake else 0, 4),
            "sharpe": round(sharpe, 3) if math.isfinite(sharpe) else None,
            "profit_factor": round(pf, 3) if math.isfinite(pf) else None,
            "max_drawdown": round(_max_drawdown(trades), 4),
            "brier_score": round(_brier_score(trades), 4),
            "avg_score_wins": round(sum(t.score for t in wins) / len(wins), 3) if wins else 0,
            "avg_score_losses": round(sum(t.score for t in losses) / len(losses), 3) if losses else 0,
        },
        "calibration": calibration,
        "trades": [
            {
                "ticker": t.ticker[:28],
                "category": t.category,
                "score": round(t.score, 3),
                "p_model": round(t.p_model, 3),
                "direction": t.direction,
                "entry_price_cents": round(t.entry_price_cents, 1),
                "market_result": t.market_result,
                "won": t.won,
                "stake_cents": round(t.stake_cents, 2),
                "profit_cents": round(t.profit_cents, 2),
                "roi": round(t.roi, 4),
            }
            for t in reversed(trades[-60:])
        ],
    }


@app.post("/api/collect")
async def trigger_collect(background_tasks: BackgroundTasks):
    if _collect_state["running"]:
        return {"status": "already_running"}
    client = _get_client()
    if not client:
        raise HTTPException(503, "Kalshi API not configured.")
    background_tasks.add_task(_run_collect, client)
    return {"status": "started"}


async def _run_collect(client: KalshiClient):
    _collect_state["running"] = True
    _collect_state["error"] = None
    try:
        loop = asyncio.get_event_loop()
        _, stored = await loop.run_in_executor(
            _executor,
            lambda: collect_markets(client, _db_path(), verbose=False),
        )
        _collect_state["stored"] = stored
        _collect_state["last_run"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        _collect_state["error"] = str(exc)
    finally:
        _collect_state["running"] = False


@app.get("/api/market/{ticker}/ai")
async def ai_analysis(ticker: str, title: str = "", price: float = 50,
                      volume: float = 0, hours_left: float = 0,
                      score: float = 0, category: str = ""):
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured in .env.")

    prompt = (
        f'You are a sharp prediction market analyst. Give a 2–3 sentence assessment.\n\n'
        f'Market: "{title or ticker}"\n'
        f'YES price: {price:.0f}¢  |  Volume: ${volume/100:,.0f}  |  '
        f'Closes in: {hours_left:.0f}h  |  Score: {score:.2f}  |  Category: {category}\n\n'
        f'Focus on: is the price fair? key risk factors? concrete verdict. No fluff.'
    )

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            _executor,
            lambda: _requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            ),
        )
        data = resp.json()
        text = (data.get("content") or [{}])[0].get("text", "Analysis unavailable.")
    except Exception as exc:
        raise HTTPException(502, f"AI call failed: {exc}")

    return {"analysis": text}
