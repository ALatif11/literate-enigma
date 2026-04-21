"""Click CLI entry point for the Kalshi backtester.

Commands
--------
  kalshi-bt collect   — Pull resolved markets from the Kalshi API and store
                        them locally in SQLite.  Run this first.

  kalshi-bt backtest  — Score every stored market, simulate trades on those
                        that meet the threshold, then print the report.

  kalshi-bt report    — Re-run the report on already-scored/traded data
                        without hitting the API again.

Configuration is read from a .env file (copy .env.example to .env and fill in
your credentials).  All options can also be overridden on the command line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv


def _cents_to_dollars(cents: float) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:.2f}"

load_dotenv()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise click.ClickException(
            f"Missing required config: {name}. "
            f"Set it in your .env file or as an environment variable."
        )
    return value


def _build_client(env: str) -> "KalshiClient":  # noqa: F821  (imported below)
    from .client import KalshiClient

    api_key_id = _require_env("KALSHI_API_KEY_ID")

    key_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if key_path_str:
        key_path = Path(key_path_str).expanduser()
        if not key_path.exists():
            raise click.ClickException(
                f"Private key file not found: {key_path}. "
                f"Check KALSHI_PRIVATE_KEY_PATH in your .env."
            )
        private_key: str | Path = key_path
    else:
        # Allow embedding the PEM directly in the env var (newlines as \n)
        pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "").strip()
        if not pem:
            raise click.ClickException(
                "No private key configured. Set KALSHI_PRIVATE_KEY_PATH "
                "(path to .pem file) or KALSHI_PRIVATE_KEY_PEM (inline PEM) "
                "in your .env."
            )
        private_key = pem.replace("\\n", "\n")

    return KalshiClient(api_key_id=api_key_id, private_key=private_key, env=env)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Kalshi prediction market backtester.

    \b
    Quick start:
      1. cp .env.example .env   # fill in your API credentials
      2. kalshi-bt collect      # download resolved markets
      3. kalshi-bt backtest     # score, trade, and report
    """


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--db",
    "db_path",
    default=lambda: os.getenv("KALSHI_DB_PATH", "kalshi_backtest.db"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--env",
    "kalshi_env",
    default=lambda: os.getenv("KALSHI_ENV", "demo"),
    type=click.Choice(["demo", "prod"]),
    show_default=True,
    help="Kalshi environment to use.",
)
@click.option(
    "--status",
    "statuses",
    default="finalized,settled",
    show_default=True,
    help="Comma-separated market statuses to collect.",
)
@click.option(
    "--page-size",
    default=1000,
    show_default=True,
    help="Markets per API page (max 1000).",
)
def collect(db_path: str, kalshi_env: str, statuses: str, page_size: int) -> None:
    """Fetch resolved markets from the Kalshi API and store them locally.

    Safe to re-run — existing records are updated (upserted) rather than
    duplicated.
    """
    from .client import KalshiAPIError
    from .collector import collect_markets

    status_list = tuple(s.strip() for s in statuses.split(",") if s.strip())
    client = _build_client(kalshi_env)

    click.echo(f"Connecting to Kalshi ({kalshi_env}) ...")
    try:
        fetched, stored = collect_markets(
            client=client,
            db_path=db_path,
            statuses=status_list,
            page_size=page_size,
            verbose=True,
        )
    except KalshiAPIError as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"\nDone. Fetched {fetched} markets, stored/updated {stored} "
        f"(skipped {fetched - stored} without usable data)."
    )
    click.echo(f"Database: {Path(db_path).resolve()}")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--db",
    "db_path",
    default=lambda: os.getenv("KALSHI_DB_PATH", "kalshi_backtest.db"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--threshold",
    default=lambda: float(os.getenv("KALSHI_SCORE_THRESHOLD", "0.4")),
    show_default=True,
    type=float,
    help="Minimum opportunity score (0.0–1.0) required to simulate a trade.",
)
@click.option(
    "--strategy",
    default=lambda: os.getenv("KALSHI_STRATEGY", "underdog"),
    type=click.Choice(["underdog", "favorite"]),
    show_default=True,
    help=(
        "underdog: buy the cheaper (<50¢) side. "
        "favorite: buy the market-favourite (>50¢) side."
    ),
)
@click.option(
    "--stake",
    "stake_cents",
    default=lambda: float(os.getenv("KALSHI_STAKE_CENTS", "100")),
    show_default=True,
    type=float,
    help="Fixed stake per trade in cents (100 = $1.00).  Ignored when --bankroll is set.",
)
@click.option(
    "--bankroll",
    "bankroll_cents",
    default=lambda: float(os.getenv("KALSHI_BANKROLL_CENTS", "0")),
    show_default=True,
    type=float,
    help=(
        "Starting bankroll in cents for Kelly position sizing (e.g. 100000 = $1,000). "
        "When set, overrides --stake and sizes each trade via fractional Kelly."
    ),
)
@click.option(
    "--kelly-fraction",
    default=lambda: float(os.getenv("KALSHI_KELLY_FRACTION", "0.25")),
    show_default=True,
    type=float,
    help=(
        "Fraction of full Kelly to use (0.25 = quarter-Kelly, recommended). "
        "Only applies when --bankroll is set."
    ),
)
def backtest(
    db_path: str,
    threshold: float,
    strategy: str,
    stake_cents: float,
    bankroll_cents: float,
    kelly_fraction: float,
) -> None:
    """Score stored markets, simulate trades, and print a performance report.

    Run `kalshi-bt collect` first to populate the database.

    \b
    Position sizing modes:
      Fixed stake (default): every trade risks --stake cents.
      Kelly sizing:          pass --bankroll to enable. Sizes each trade in
                             proportion to the estimated edge using fractional
                             Kelly. Use --kelly-fraction to control aggression
                             (0.25 = quarter-Kelly is the safe default).
    """
    from .collector import load_markets
    from .scorer import score_markets, filter_by_threshold
    from .trader import simulate_trades
    from .reporter import print_report

    db = Path(db_path)
    if not db.exists():
        raise click.ClickException(
            f"Database not found: {db.resolve()}\n"
            "Run `kalshi-bt collect` first to download market data."
        )

    click.echo(f"Loading markets from {db.resolve()} ...")
    markets = load_markets(db_path)
    if not markets:
        raise click.ClickException(
            "No markets in the database. Run `kalshi-bt collect` first."
        )

    click.echo(f"  Loaded {len(markets)} markets.")

    click.echo("Scoring markets ...")
    scored = score_markets(markets)
    qualifying = filter_by_threshold(scored, threshold)
    click.echo(
        f"  {len(qualifying)} markets meet the score threshold of {threshold:.2f} "
        f"(out of {len(scored)} total)."
    )

    if bankroll_cents > 0:
        click.echo(
            f"Simulating trades (Kelly sizing, {kelly_fraction:.2f}× fraction, "
            f"bankroll={_cents_to_dollars(bankroll_cents)}) ..."
        )
    else:
        click.echo(f"Simulating trades (fixed stake={_cents_to_dollars(stake_cents)}) ...")

    trades = simulate_trades(
        qualifying,
        score_threshold=threshold,
        strategy=strategy,
        stake_cents=stake_cents,
        bankroll_cents=bankroll_cents,
        kelly_fraction=kelly_fraction,
    )
    click.echo(f"  {len(trades)} trades simulated.")

    print_report(trades, threshold, strategy, stake_cents, bankroll_cents, kelly_fraction)


# ---------------------------------------------------------------------------
# score (inspect scores without running trades)
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--db",
    "db_path",
    default=lambda: os.getenv("KALSHI_DB_PATH", "kalshi_backtest.db"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--top",
    default=20,
    show_default=True,
    help="Number of top-scored markets to display.",
)
def score(db_path: str, top: int) -> None:
    """Show the highest-scoring markets in the database without simulating trades."""
    from tabulate import tabulate
    from .collector import load_markets
    from .scorer import score_markets

    db = Path(db_path)
    if not db.exists():
        raise click.ClickException(
            f"Database not found: {db.resolve()}. Run `kalshi-bt collect` first."
        )

    markets = load_markets(db_path)
    scored = sorted(score_markets(markets), key=lambda m: m["score"], reverse=True)

    rows = [
        [
            m["ticker"][:35],
            m.get("category", "")[:14],
            f"{m['score']:.3f}",
            f"{m['score_price_deviation']:.3f}",
            f"{m['score_volume_factor']:.3f}",
            f"{m['score_time_factor']:.3f}",
            f"{m['last_price_cents']:.0f}¢",
            m["result"].upper(),
        ]
        for m in scored[:top]
    ]

    click.echo(f"\nTop {min(top, len(rows))} markets by opportunity score\n")
    click.echo(tabulate(
        rows,
        headers=["Ticker", "Category", "Score", "Price Dev", "Vol", "Time", "Close Price", "Result"],
        tablefmt="simple",
    ))
    click.echo()


# ---------------------------------------------------------------------------
# serve — launch the web command center
# ---------------------------------------------------------------------------


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes (dev mode).")
def serve(host: str, port: int, reload: bool) -> None:
    """Launch the web command center (browser UI).

    \b
    Opens a mobile-friendly dashboard at http://localhost:8000 with:
      · Live market scanner   — scored open markets from Kalshi
      · Decision panel        — Kelly sizing, score breakdown, Claude AI analysis
      · Backtest view         — Brier score, Sharpe, drawdown, calibration table
      · Config                — all settings + historical data collection

    Requires server dependencies:  pip install 'kalshi-backtester[server]'
    """
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "Server dependencies not installed.\n"
            "Run: pip install 'kalshi-backtester[server]'"
        )

    click.echo(f"\n  KALSHI.AI Command Center")
    click.echo(f"  http://{host}:{port}\n")
    uvicorn.run(
        "kalshi_backtester.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )
