"""Create focused binary-task datasets from gold-labeled sales windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "id",
    "source",
    "dialogue_id",
    "window_start_turn",
    "window_end_turn",
    "speaker_window",
    "text",
    "recommendation_type",
    "actionable_coaching_needed",
    "unresolved_customer_concern",
    "missing_next_step_action",
    "missing_value_anchor",
    "early_presentation_risk",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    output_rows = []
    for row in rows:
        recommendation_type = row["recommendation_type"]
        output_rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "dialogue_id": row["dialogue_id"],
                "window_start_turn": row["window_start_turn"],
                "window_end_turn": row["window_end_turn"],
                "speaker_window": row["speaker_window"],
                "text": row["text"],
                "recommendation_type": recommendation_type,
                "actionable_coaching_needed": int(recommendation_type != "no_action"),
                "unresolved_customer_concern": int(
                    recommendation_type == "handle_objection"
                ),
                "missing_next_step_action": int(recommendation_type == "fix_next_step"),
                "missing_value_anchor": int(recommendation_type == "anchor_value"),
                "early_presentation_risk": int(
                    recommendation_type == "avoid_early_presentation"
                ),
                "notes": row.get("notes", ""),
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

