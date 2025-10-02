"""Helpers for working with the Craigslist Bargains negotiation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import json
import re


CRAIGSLIST_SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<buyer>",
        "<seller>",
        "<context>",
        "</context>",
        "<deal>",
        "<system>",
    ]
}


@dataclass
class NegotiationTurn:
    """A single turn in the negotiation."""

    speaker: str
    utterance: str
    intent: Optional[str] = None
    price: Optional[float] = None


@dataclass
class NegotiationExample:
    """A negotiation episode parsed from the Craigslist Bargains dataset."""

    scenario_id: str
    title: str
    description: str
    category: Optional[str]
    list_price: Optional[float]
    buyer_target: Optional[float]
    seller_target: Optional[float]
    turns: List[NegotiationTurn]

    @property
    def deal_price(self) -> Optional[float]:
        """Return the first accepted price if a deal was reached."""
        for turn in self.turns:
            if turn.intent and turn.intent.lower() in {"accept", "agree", "deal"} and turn.price is not None:
                return turn.price
        return None

    @property
    def last_offer(self) -> Optional[float]:
        """Return the most recent offer made by either party."""
        for turn in reversed(self.turns):
            if turn.price is not None:
                return turn.price
        return None


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split())


def _iter_raw_examples(data_dir: Path, split: str) -> Iterator[dict]:
    data_path = data_dir / "raw" / f"{split}.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find split '{split}' at {data_path}")
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_craigslist_split(data_dir: Path | str, split: str) -> List[NegotiationExample]:
    """Load and parse a craigslist_bargains data split into structured examples."""
    data_dir = Path(data_dir)
    examples: List[NegotiationExample] = []

    for idx, raw in enumerate(_iter_raw_examples(data_dir, split)):
        roles: List[str] = [role.lower() for role in raw.get("agent_info", {}).get("Role", [])]
        targets: List[Optional[float]] = raw.get("agent_info", {}).get("Target", [])
        buyer_target = _safe_float(targets[roles.index("buyer")] if "buyer" in roles else None)
        seller_target = _safe_float(targets[roles.index("seller")] if "seller" in roles else None)

        items = raw.get("items", {})
        title_candidates = items.get("Title", []) or [""]
        description_candidates = items.get("Description", []) or [""]
        category_candidates = items.get("Category", [])
        price_candidates = items.get("Price", [])

        turns: List[NegotiationTurn] = []
        utterances: Iterable[str] = raw.get("utterance", [])
        intents: Iterable[str] = raw.get("dialogue_acts", {}).get("intent", [])
        prices: Iterable[Optional[float]] = raw.get("dialogue_acts", {}).get("price", [])
        agent_turn: Iterable[int] = raw.get("agent_turn", [])

        # Pad intents/prices if necessary so zipping works safely
        intents = list(intents)
        prices = list(prices)
        agent_turn = list(agent_turn)
        utterances = list(utterances)

        for turn_idx, utter in enumerate(utterances):
            role_idx = agent_turn[turn_idx] if turn_idx < len(agent_turn) else 0
            speaker = roles[role_idx] if role_idx < len(roles) else f"agent_{role_idx}"
            intent = intents[turn_idx] if turn_idx < len(intents) else None
            price_val = prices[turn_idx] if turn_idx < len(prices) else None
            price = _safe_float(price_val)
            turns.append(
                NegotiationTurn(
                    speaker=speaker,
                    utterance=_normalise_text(utter),
                    intent=intent,
                    price=price,
                )
            )

        list_price = _safe_float(price_candidates[0] if price_candidates else None)
        category = category_candidates[0].lower() if category_candidates else None

        example = NegotiationExample(
            scenario_id=f"{split}_{idx:05d}",
            title=_normalise_text(title_candidates[0]),
            description=_normalise_text(description_candidates[0]),
            category=category,
            list_price=list_price,
            buyer_target=buyer_target,
            seller_target=seller_target,
            turns=turns,
        )
        examples.append(example)

    return examples


def render_dialogue(
    example: NegotiationExample,
    include_outcome: bool = True,
    max_turns: Optional[int] = None,
) -> str:
    """Format an example into a single training string with role tags."""
    context_lines = [
        "<context>",
        f"Listing title: {example.title}",
    ]
    if example.category:
        context_lines.append(f"Category: {example.category}")
    if example.list_price is not None:
        context_lines.append(f"List price: ${example.list_price:,.2f}")
    if example.buyer_target is not None:
        context_lines.append(f"Buyer target: ${example.buyer_target:,.2f}")
    if example.seller_target is not None:
        context_lines.append(f"Seller target: ${example.seller_target:,.2f}")
    if example.description:
        context_lines.append(f"Description: {example.description}")
    context_lines.append("</context>")

    dialogue_lines: List[str] = []
    total_turns = len(example.turns) if max_turns is None else min(max_turns, len(example.turns))

    for turn in example.turns[:total_turns]:
        prefix = "<buyer>" if turn.speaker == "buyer" else "<seller>"
        dialogue_lines.append(f"{prefix} {turn.utterance}")

    if include_outcome:
        deal_price = example.deal_price
        if deal_price is not None:
            summary = f"Deal reached at ${deal_price:,.2f}."
        else:
            last_offer = example.last_offer
            if last_offer is not None:
                summary = f"No deal. Last offer was ${last_offer:,.2f}."
            else:
                summary = "No deal reached."
        dialogue_lines.append(f"<deal> {summary}")

    return "\n".join(context_lines + dialogue_lines)


PRICE_PATTERN = re.compile(r"(?i)(?:final price|deal(?: reached)? at)\s*\$?([0-9]+(?:\.[0-9]+)?)")


def extract_final_price(text: str) -> Optional[float]:
    """Extract a final price from generated dialogue, if explicitly stated."""
    match = PRICE_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
