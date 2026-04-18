"""Report generator — prints a summary of a backtest run to stdout."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Sequence

from tabulate import tabulate

from .trader import Trade


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _cents_to_dollars(cents: float) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:.2f}"


def print_report(
    trades: list[Trade],
    score_threshold: float,
    strategy: str,
    stake_cents: float,
) -> None:
    """Print a full backtest report to stdout."""
    if not trades:
        print("\nNo trades met the score threshold. Try lowering --threshold.")
        return

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]

    total_stake = sum(t.stake_cents for t in trades)
    total_profit = sum(t.profit_cents for t in trades)
    overall_roi = total_profit / total_stake if total_stake else 0.0
    win_rate = len(wins) / len(trades)

    avg_score_wins = statistics.mean(t.score for t in wins) if wins else 0.0
    avg_score_losses = statistics.mean(t.score for t in losses) if losses else 0.0

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  KALSHI BACKTESTER REPORT")
    print("=" * 60)
    print(f"  Strategy      : {strategy}")
    print(f"  Score threshold: {score_threshold:.2f}")
    print(f"  Stake per trade: {_cents_to_dollars(stake_cents)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    summary = [
        ["Total trades",     len(trades)],
        ["Wins",             f"{len(wins)}  ({_pct(win_rate)})"],
        ["Losses",           len(losses)],
        ["Total staked",     _cents_to_dollars(total_stake)],
        ["Net P&L",          _cents_to_dollars(total_profit)],
        ["Overall ROI",      _pct(overall_roi)],
        ["Avg score (wins)", f"{avg_score_wins:.3f}"],
        ["Avg score (loss)", f"{avg_score_losses:.3f}"],
    ]
    print("\nOVERALL SUMMARY")
    print(tabulate(summary, tablefmt="simple"))

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------
    by_category: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        cat = t.category or "Unknown"
        by_category[cat].append(t)

    cat_rows = []
    for cat, cat_trades in sorted(by_category.items()):
        cat_wins = sum(1 for t in cat_trades if t.won)
        cat_profit = sum(t.profit_cents for t in cat_trades)
        cat_stake = sum(t.stake_cents for t in cat_trades)
        cat_roi = cat_profit / cat_stake if cat_stake else 0.0
        cat_wr = cat_wins / len(cat_trades)
        avg_score = statistics.mean(t.score for t in cat_trades)
        cat_rows.append([
            cat,
            len(cat_trades),
            f"{cat_wins} ({_pct(cat_wr)})",
            _cents_to_dollars(cat_profit),
            _pct(cat_roi),
            f"{avg_score:.3f}",
        ])

    print("\nBREAKDOWN BY CATEGORY")
    print(tabulate(
        cat_rows,
        headers=["Category", "Trades", "Wins", "Net P&L", "ROI", "Avg Score"],
        tablefmt="simple",
    ))

    # ------------------------------------------------------------------
    # Top 10 trades by score
    # ------------------------------------------------------------------
    top = sorted(trades, key=lambda t: t.score, reverse=True)[:10]
    top_rows = [
        [
            t.ticker[:30],
            t.category[:12],
            f"{t.score:.3f}",
            t.direction.upper(),
            f"{t.entry_price_cents:.0f}¢",
            t.market_result.upper(),
            "WIN" if t.won else "LOSS",
            _cents_to_dollars(t.profit_cents),
        ]
        for t in top
    ]
    print("\nTOP 10 TRADES BY SCORE")
    print(tabulate(
        top_rows,
        headers=["Ticker", "Category", "Score", "Dir", "Entry", "Result", "Outcome", "P&L"],
        tablefmt="simple",
    ))

    print("\n" + "=" * 60 + "\n")
