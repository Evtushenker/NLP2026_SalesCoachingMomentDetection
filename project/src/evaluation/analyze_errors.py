"""Create compact error-analysis artifacts for gold test predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]


def as_int(value: str) -> int:
    return int(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--cases-output", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()

    with args.predictions.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    summary: dict[str, object] = {"rows": len(rows), "model_prefix": args.model_prefix}
    label_summary = {}
    cases = []

    for label in LABEL_COLUMNS:
        pred_column = f"{args.model_prefix}_{label}"
        gold_column = f"gold_{label}"
        counts = Counter()
        false_positive_examples = []
        false_negative_examples = []
        for row in rows:
            gold = as_int(row[gold_column])
            pred = as_int(row[pred_column])
            if gold == 1 and pred == 1:
                counts["tp"] += 1
            elif gold == 0 and pred == 0:
                counts["tn"] += 1
            elif gold == 0 and pred == 1:
                counts["fp"] += 1
                false_positive_examples.append(row)
            elif gold == 1 and pred == 0:
                counts["fn"] += 1
                false_negative_examples.append(row)

        label_summary[label] = dict(counts)
        for error_type, examples in (
            ("false_positive", false_positive_examples[:2]),
            ("false_negative", false_negative_examples[:2]),
        ):
            for example in examples:
                cases.append(
                    {
                        "id": example["id"],
                        "label": label,
                        "error_type": error_type,
                        "gold": example[gold_column],
                        "prediction": example[pred_column],
                        "text": example["text"],
                    }
                )

    exact_failures = []
    for row in rows:
        wrong_labels = [
            label
            for label in LABEL_COLUMNS
            if as_int(row[f"gold_{label}"]) != as_int(row[f"{args.model_prefix}_{label}"])
        ]
        if wrong_labels:
            exact_failures.append(
                {
                    "id": row["id"],
                    "wrong_label_count": len(wrong_labels),
                    "wrong_labels": wrong_labels,
                    "text": row["text"],
                }
            )
    exact_failures.sort(key=lambda item: item["wrong_label_count"], reverse=True)

    summary["labels"] = label_summary
    summary["exact_failure_count"] = len(exact_failures)
    summary["exact_match_count"] = len(rows) - len(exact_failures)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = cases[: args.max_cases]
    with args.cases_output.open("w", newline="", encoding="utf-8") as target:
        fieldnames = ["id", "label", "error_type", "gold", "prediction", "text"]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_cases)

    print(f"summary_output={args.summary_output}")
    print(f"cases_output={args.cases_output}")
    print(f"exact_failures={len(exact_failures)}")
    for label, counts in label_summary.items():
        print(f"{label}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

