"""Simulated trader — replays resolved markets and tracks P&L.

Trade mechanics
---------------
Every Kalshi binary contract resolves to either $1.00 (YES wins) or $0.00
(NO wins).  Buying YES at P¢ costs P cents per contract.  If YES wins you
collect 100¢, net profit = (100 - P)¢.  If NO wins you lose P¢.

Strategies
----------
underdog (default)
    Always buy the side priced below 50¢ — the "underpriced" side relative to
    fair-coin odds.  This is a mean-reversion / contrarian bet.
    - If last_price_cents < 50  → buy YES at last_price_cents
    - If last_price_cents >= 50 → buy NO  at (100 - last_price_cents)¢

favorite
    Always buy the side the market already favours (priced above 50¢).
    - If last_price_cents > 50  → buy YES at last_price_cents
    - If last_price_cents <= 50 → buy NO  at (100 - last_price_cents)¢

Entry price
-----------
We simulate entering at the market's last_price_cents at close — the price
available to anyone who looked at the market just before it stopped trading.

Stake
-----
We invest a fixed stake (in cents) per trade and scale contracts accordingly.
  contracts = stake_cents / entry_price_cents
  profit    = contracts * (100 - entry_price_cents)   if won
  profit    = contracts * (-entry_price_cents)         if lost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


VALID_STRATEGIES = ("underdog", "favorite")


@dataclass
class Trade:
    ticker: str
    category: str
    score: float
    direction: str          # 'yes' or 'no'
    entry_price_cents: float
    market_result: str      # 'yes' or 'no'
    won: bool
    contracts: float
    stake_cents: float
    profit_cents: float
    roi: float              # profit / stake

    # Sub-scores for reporting
    score_price_deviation: float = 0.0
    score_volume_factor: float = 0.0
    score_time_factor: float = 0.0


def simulate_trades(
    markets: list[dict],
    score_threshold: float = 0.4,
    strategy: str = "underdog",
    stake_cents: float = 100.0,
) -> list[Trade]:
    """Simulate trades on each market that meets the score threshold.

    Parameters
    ----------
    markets:
        Scored market dicts (output of scorer.score_markets).
    score_threshold:
        Only markets with score >= threshold are traded.
    strategy:
        'underdog' or 'favorite'.
    stake_cents:
        Fixed dollar-equivalent stake per trade (in cents).

    Returns a list of Trade objects, one per qualifying market.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}")

    trades: list[Trade] = []

    for m in markets:
        if m.get("score", 0.0) < score_threshold:
            continue

        price_cents = float(m["last_price_cents"])
        market_result = m["result"]  # 'yes' or 'no'

        direction, entry_price = _choose_side(price_cents, strategy)

        if entry_price <= 0 or entry_price >= 100:
            continue

        contracts = stake_cents / entry_price
        won = direction == market_result
        profit_cents = contracts * (100.0 - entry_price) if won else contracts * (-entry_price)
        roi = profit_cents / stake_cents

        trades.append(
            Trade(
                ticker=m["ticker"],
                category=m.get("category", ""),
                score=m["score"],
                direction=direction,
                entry_price_cents=entry_price,
                market_result=market_result,
                won=won,
                contracts=contracts,
                stake_cents=stake_cents,
                profit_cents=profit_cents,
                roi=roi,
                score_price_deviation=m.get("score_price_deviation", 0.0),
                score_volume_factor=m.get("score_volume_factor", 0.0),
                score_time_factor=m.get("score_time_factor", 0.0),
            )
        )

    return trades


def _choose_side(price_cents: float, strategy: str) -> tuple[str, float]:
    """Return (direction, entry_price_cents) for a given strategy."""
    yes_price = price_cents
    no_price = 100.0 - price_cents

    if strategy == "underdog":
        # Buy whichever side is cheaper (< 50¢)
        if yes_price <= no_price:
            return "yes", yes_price
        else:
            return "no", no_price
    else:  # favorite
        # Buy whichever side is more expensive (closer to 100¢ = market favourite)
        if yes_price >= no_price:
            return "yes", yes_price
        else:
            return "no", no_price
