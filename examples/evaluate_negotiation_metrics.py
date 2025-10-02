"""Compute negotiation metrics (success rate, SL%, average turns) from generated dialogues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyna_gym.evaluation.metrics import NegotiationRecord, compute_negotiation_metrics


def load_records(path: Path) -> List[NegotiationRecord]:
    records: List[NegotiationRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                NegotiationRecord(
                    text=payload.get("text", ""),
                    plain_text=payload.get("plain_text"),
                    reward=float(payload.get("reward", 0.0)),
                    buyer_target=payload.get("buyer_target"),
                    seller_target=payload.get("seller_target"),
                    turn_count=payload.get("turn_count"),
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/craigslist-mcts/mcts_dialogues.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    metrics = compute_negotiation_metrics(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
