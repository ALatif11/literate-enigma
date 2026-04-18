"""Report generator — prints a summary of a backtest run to stdout."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from tabulate import tabulate

from .trader import Trade


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _cents_to_dollars(cents: float) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:.2f}"


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------


def _brier_score(trades: list[Trade]) -> float:
    """Mean squared error between p_model and the binary outcome.

    BS = mean((p_model - outcome)²)   where outcome is 1 (won) or 0 (lost).
    Range: 0 (perfect) to 1 (perfectly wrong).  Random guessing = 0.25.
    A score below 0.25 means the model has genuine predictive signal.
    """
    return sum((t.p_model - float(t.won)) ** 2 for t in trades) / len(trades)


def _sharpe(trades: list[Trade]) -> float:
    """Risk-adjusted return: mean(ROI) / std(ROI).  Assumes risk-free rate = 0.

    Above 1.0 is decent; above 2.0 is strong.  Infinite if all trades win.
    """
    if len(trades) < 2:
        return float("inf") if trades and trades[0].won else 0.0
    rois = [t.roi for t in trades]
    std = statistics.stdev(rois)
    return statistics.mean(rois) / std if std else float("inf")


def _max_drawdown(trades: list[Trade]) -> float:
    """Largest peak-to-trough decline in cumulative P&L as a fraction of the peak.

    Trades are evaluated in the order supplied (DB insertion order).
    Returns a value in [0, 1] — e.g. 0.08 means an 8% drawdown.
    """
    peak = 0.0
    max_dd = 0.0
    cumulative = 0.0
    for t in trades:
        cumulative += t.profit_cents
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (peak - cumulative) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _profit_factor(trades: list[Trade]) -> float:
    """Gross profit / gross loss.  Above 1.5 is healthy; below 1.0 means net loss."""
    gross_profit = sum(t.profit_cents for t in trades if t.profit_cents > 0)
    gross_loss = abs(sum(t.profit_cents for t in trades if t.profit_cents < 0))
    return gross_profit / gross_loss if gross_loss else float("inf")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(
    trades: list[Trade],
    score_threshold: float,
    strategy: str,
    stake_cents: float,
    bankroll_cents: float = 0.0,
    kelly_fraction: float = 0.25,
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

    bs = _brier_score(trades)
    sharpe = _sharpe(trades)
    mdd = _max_drawdown(trades)
    pf = _profit_factor(trades)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  KALSHI BACKTESTER REPORT")
    print("=" * 62)
    print(f"  Strategy        : {strategy}")
    print(f"  Score threshold : {score_threshold:.2f}")
    if bankroll_cents > 0:
        print(f"  Position sizing : Kelly ({kelly_fraction:.2f}× fraction)")
        print(f"  Starting bankroll: {_cents_to_dollars(bankroll_cents)}")
    else:
        print(f"  Position sizing : Fixed stake {_cents_to_dollars(stake_cents)}/trade")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    sharpe_str = f"{sharpe:.2f}" if math.isfinite(sharpe) else "∞"
    pf_str = f"{pf:.2f}" if math.isfinite(pf) else "∞"

    summary = [
        ["Total trades",      len(trades)],
        ["Wins",              f"{len(wins)}  ({_pct(win_rate)})"],
        ["Losses",            len(losses)],
        ["Total staked",      _cents_to_dollars(total_stake)],
        ["Net P&L",           _cents_to_dollars(total_profit)],
        ["Overall ROI",       _pct(overall_roi)],
        ["Sharpe ratio",      sharpe_str],
        ["Profit factor",     pf_str],
        ["Max drawdown",      _pct(mdd)],
        ["Brier score",       f"{bs:.4f}  {'✓ signal' if bs < 0.25 else '✗ weak'}  (random=0.25)"],
        ["Avg score (wins)",  f"{avg_score_wins:.3f}"],
        ["Avg score (loss)",  f"{avg_score_losses:.3f}"],
    ]
    print("\nOVERALL SUMMARY")
    print(tabulate(summary, tablefmt="simple"))

    # ------------------------------------------------------------------
    # Calibration: p_model vs actual win rate by probability bucket
    # ------------------------------------------------------------------
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        label = f"{int(t.p_model * 10) * 10}–{int(t.p_model * 10) * 10 + 10}%"
        buckets[label].append(t)

    if len(buckets) > 1:
        cal_rows = []
        for label in sorted(buckets):
            bt = buckets[label]
            actual_wr = sum(t.won for t in bt) / len(bt)
            avg_p = statistics.mean(t.p_model for t in bt)
            cal_rows.append([label, len(bt), f"{avg_p:.2f}", _pct(actual_wr)])

        print("\nCALIBRATION  (model probability vs actual win rate)")
        print(tabulate(
            cal_rows,
            headers=["p_model bucket", "Trades", "Avg p_model", "Actual win rate"],
            tablefmt="simple",
        ))

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------
    by_category: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_category[t.category or "Unknown"].append(t)

    cat_rows = []
    for cat, cat_trades in sorted(by_category.items()):
        cat_wins = sum(1 for t in cat_trades if t.won)
        cat_profit = sum(t.profit_cents for t in cat_trades)
        cat_stake = sum(t.stake_cents for t in cat_trades)
        cat_roi = cat_profit / cat_stake if cat_stake else 0.0
        cat_wr = cat_wins / len(cat_trades)
        cat_pf = _profit_factor(cat_trades)
        avg_score = statistics.mean(t.score for t in cat_trades)
        cat_rows.append([
            cat,
            len(cat_trades),
            f"{cat_wins} ({_pct(cat_wr)})",
            _cents_to_dollars(cat_profit),
            _pct(cat_roi),
            f"{cat_pf:.2f}" if math.isfinite(cat_pf) else "∞",
            f"{avg_score:.3f}",
        ])

    print("\nBREAKDOWN BY CATEGORY")
    print(tabulate(
        cat_rows,
        headers=["Category", "Trades", "Wins", "Net P&L", "ROI", "Prof. Factor", "Avg Score"],
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
            f"{t.p_model:.2f}",
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
        headers=["Ticker", "Category", "Score", "p_model", "Dir", "Entry", "Result", "Outcome", "P&L"],
        tablefmt="simple",
    ))

    print("\n" + "=" * 62 + "\n")
