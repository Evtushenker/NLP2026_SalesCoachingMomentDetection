"""Generate recommendations for a labeled CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .rules import LABEL_COLUMNS, recommend
except ImportError:
    from rules import LABEL_COLUMNS, recommend


OUTPUT_COLUMNS = [
    "id",
    "text",
    *LABEL_COLUMNS,
    "recommendation_type",
    "action",
    "rationale",
    "confidence",
]


def require_columns(rows: list[dict[str, str]], columns: list[str]) -> None:
    if not rows:
        raise ValueError("Input file contains no rows.")
    missing = sorted(set(columns).difference(rows[0].keys()))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    require_columns(rows, ["id", "text", *LABEL_COLUMNS])

    output_rows = []
    for row in rows[: args.limit]:
        labels = {label: row[label] for label in LABEL_COLUMNS}
        recommendation = recommend(labels, text=row["text"])
        output_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                **labels,
                "recommendation_type": recommendation.recommendation_type,
                "action": recommendation.action,
                "rationale": recommendation.rationale,
                "confidence": f"{recommendation.confidence:.3f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"rows={len(output_rows)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
