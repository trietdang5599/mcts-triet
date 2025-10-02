"""Negotiation evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Optional

from dyna_gym.data_utils.craigslist import extract_final_price


@dataclass
class NegotiationRecord:
    """Generated negotiation annotated with metadata."""

    text: str
    plain_text: Optional[str]
    reward: float
    buyer_target: Optional[float]
    seller_target: Optional[float]
    turn_count: Optional[int]


def _is_success(text: str) -> bool:
    """Heuristic success: dialogue explicitly mentions a deal."""
    lowered = text.lower()
    return "deal" in lowered and "no deal" not in lowered


def compute_negotiation_metrics(records: Iterable[NegotiationRecord]) -> dict:
    """Compute success-rate style metrics for generated negotiations."""
    records = list(records)
    if not records:
        return {
            "count": 0,
            "success_rate": 0.0,
            "success_within_limits_pct": 0.0,
            "average_turns": 0.0,
            "average_reward": 0.0,
        }

    successes: List[bool] = []
    success_within_limits: List[bool] = []
    turns: List[int] = []
    rewards: List[float] = []

    for record in records:
        rewards.append(record.reward)
        text = record.plain_text or record.text
        successes.append(_is_success(text))
        if record.turn_count is not None:
            turns.append(record.turn_count)
        elif record.text:
            turns.append(record.text.count("<buyer>") + record.text.count("<seller>"))

        price = extract_final_price(text)
        if price is not None and record.buyer_target is not None and record.seller_target is not None:
            lower = min(record.buyer_target, record.seller_target)
            upper = max(record.buyer_target, record.seller_target)
            success_within_limits.append(lower <= price <= upper)
        else:
            success_within_limits.append(False)

    return {
        "count": len(records),
        "success_rate": sum(successes) / len(successes),
        "success_within_limits_pct": sum(success_within_limits) / len(success_within_limits),
        "average_turns": mean(turns) if turns else 0.0,
        "average_reward": mean(rewards) if rewards else 0.0,
    }
