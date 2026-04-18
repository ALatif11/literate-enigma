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

Position sizing
---------------
Two modes:

Fixed stake (default)
    Every trade risks the same number of cents.  Simple but ignores edge size.

Kelly Criterion
    Sizes each trade proportionally to the estimated edge.  Requires a bankroll.

    The model probability p_model is derived from the market's score:
      p_model = entry_price/100 + score × (1 − entry_price/100)

    Intuition: at score=0 we agree with the market (no edge); at score=1 we
    are maximally confident the cheap side wins.  The score interpolates
    linearly between those extremes.

    Kelly fraction:
      b      = (100 − entry) / entry          # net odds (profit per $1 risked)
      f*     = (p_model × b − q) / b          # full Kelly
      stake  = bankroll × kelly_fraction × f* # fractional Kelly

    Use kelly_fraction=0.25 (quarter-Kelly) as a conservative default.
    Full Kelly (1.0) is theoretically optimal but extremely volatile in
    practice; most professionals use 0.25–0.50.
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_STRATEGIES = ("underdog", "favorite")


@dataclass
class Trade:
    ticker: str
    category: str
    score: float
    direction: str           # 'yes' or 'no'
    entry_price_cents: float
    market_result: str       # 'yes' or 'no'
    won: bool
    contracts: float
    stake_cents: float
    profit_cents: float
    roi: float               # profit / stake
    p_model: float           # model's estimated win probability

    # Sub-scores for reporting
    score_price_deviation: float = 0.0
    score_volume_factor: float = 0.0
    score_time_factor: float = 0.0


def _model_probability(score: float, entry_price_cents: float) -> float:
    """Estimate win probability from the opportunity score and entry price.

    The market implies p_win = entry_price / 100.  Our score captures how
    confident we are that the true probability is higher.  We interpolate
    linearly: score=0 → agree with market, score=1 → certain to win.
    """
    p_market = entry_price_cents / 100.0
    return p_market + score * (1.0 - p_market)


def _kelly_stake(
    p_model: float,
    entry_price_cents: float,
    bankroll_cents: float,
    kelly_fraction: float,
) -> float:
    """Return the fractional-Kelly stake in cents.

    f* = (p * b - q) / b   where b = net odds (profit per unit risked)
    stake = bankroll * kelly_fraction * max(0, f*)
    """
    b = (100.0 - entry_price_cents) / entry_price_cents
    q = 1.0 - p_model
    f_star = (p_model * b - q) / b
    return bankroll_cents * kelly_fraction * max(0.0, f_star)


def simulate_trades(
    markets: list[dict],
    score_threshold: float = 0.4,
    strategy: str = "underdog",
    stake_cents: float = 100.0,
    bankroll_cents: float = 0.0,
    kelly_fraction: float = 0.25,
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
        Fixed stake per trade in cents.  Used when bankroll_cents=0.
    bankroll_cents:
        Starting bankroll for Kelly sizing.  When > 0, overrides stake_cents
        and sizes each trade via the fractional Kelly formula.
    kelly_fraction:
        Fraction of full Kelly to apply (0.25 = quarter-Kelly).  Only used
        when bankroll_cents > 0.

    Returns a list of Trade objects, one per qualifying market.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}")

    use_kelly = bankroll_cents > 0
    running_bankroll = bankroll_cents

    trades: list[Trade] = []

    for m in markets:
        if m.get("score", 0.0) < score_threshold:
            continue

        price_cents = float(m["last_price_cents"])
        market_result = m["result"]

        direction, entry_price = _choose_side(price_cents, strategy)

        if entry_price <= 0 or entry_price >= 100:
            continue

        p_model = _model_probability(m["score"], entry_price)

        if use_kelly:
            trade_stake = _kelly_stake(
                p_model, entry_price, running_bankroll, kelly_fraction
            )
            if trade_stake < 1.0:
                continue
        else:
            trade_stake = stake_cents

        contracts = trade_stake / entry_price
        won = direction == market_result
        profit_cents = contracts * (100.0 - entry_price) if won else contracts * (-entry_price)
        roi = profit_cents / trade_stake

        if use_kelly:
            running_bankroll += profit_cents

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
                stake_cents=trade_stake,
                profit_cents=profit_cents,
                roi=roi,
                p_model=p_model,
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
        if yes_price <= no_price:
            return "yes", yes_price
        else:
            return "no", no_price
    else:  # favorite
        if yes_price >= no_price:
            return "yes", yes_price
        else:
            return "no", no_price
