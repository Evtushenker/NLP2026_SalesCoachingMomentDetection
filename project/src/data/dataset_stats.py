"""Print simple label statistics for a labeled CSV dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    print(f"rows: {len(rows)}")
    for label in LABEL_COLUMNS:
        counts = Counter(row[label] for row in rows)
        positives = counts.get("1", 0)
        negatives = counts.get("0", 0)
        print(f"{label}: positives={positives} negatives={negatives}")

    recommendation_counts = Counter(row.get("recommendation_type", "") for row in rows)
    print("recommendation_type:")
    for key, value in recommendation_counts.most_common():
        print(f"  {key or '<empty>'}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

