"""Scoring engine — assigns an opportunity score to each resolved market.

Score components (each normalized to [0, 1]):
  1. price_deviation  — |last_price_cents - 50| / 50
                        How far the closing price strayed from the fair-coin 50¢
                        midpoint.  A market closing at 10¢ or 90¢ gets 0.80.

  2. volume_factor    — log1p(volume) / log1p(max_volume)
                        Higher traded volume means the price signal is more
                        reliable (liquid markets are harder to be systematically
                        mispriced).  Normalised log scale so outlier volumes
                        don't dominate.

  3. time_factor      — log1p(duration_hours) / log1p(max_duration_hours)
                        Longer-running markets that still closed far from 50¢
                        represent a more persistent, significant deviation than
                        a market that was only open for minutes.

Final score = 0.50 * price_deviation
            + 0.30 * volume_factor
            + 0.20 * time_factor

All weights sum to 1, so the final score is also in [0, 1].
"""

from __future__ import annotations

import math
from typing import Sequence


WEIGHT_PRICE = 0.50
WEIGHT_VOLUME = 0.30
WEIGHT_TIME = 0.20


def _safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def score_markets(markets: list[dict]) -> list[dict]:
    """Compute opportunity scores for a list of market dicts.

    Each market must have: last_price_cents, volume, duration_hours.
    Returns the same list with a new 'score' key added to each dict.
    """
    if not markets:
        return markets

    max_log_volume = max(_safe_log1p(m["volume"]) for m in markets) or 1.0
    max_log_duration = max(_safe_log1p(m["duration_hours"]) for m in markets) or 1.0

    for m in markets:
        price_cents = float(m["last_price_cents"])
        price_deviation = abs(price_cents - 50.0) / 50.0

        volume_factor = _safe_log1p(m["volume"]) / max_log_volume

        time_factor = _safe_log1p(m["duration_hours"]) / max_log_duration

        m["score"] = (
            WEIGHT_PRICE * price_deviation
            + WEIGHT_VOLUME * volume_factor
            + WEIGHT_TIME * time_factor
        )

        # Store sub-scores for transparency
        m["score_price_deviation"] = price_deviation
        m["score_volume_factor"] = volume_factor
        m["score_time_factor"] = time_factor

    return markets


def filter_by_threshold(
    markets: list[dict],
    threshold: float,
) -> list[dict]:
    """Return only markets whose score meets the threshold."""
    return [m for m in markets if m.get("score", 0.0) >= threshold]
