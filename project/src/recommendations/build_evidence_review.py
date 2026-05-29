"""Build a sample for reviewing extracted prediction evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .evidence import evidence_is_copied_from_input, extract_evidence
except ImportError:
    from evidence import evidence_is_copied_from_input, extract_evidence


OUTPUT_COLUMNS = [
    "id",
    "dialogue_id",
    "window_start_turn",
    "window_end_turn",
    "text",
    "target",
    "model",
    "gold",
    "prediction",
    "score",
    "evidence_text",
    "speaker",
    "turn_index",
    "reason",
    "matched_pattern",
    "method",
    "confidence_hint",
    "evidence_copied_from_input",
    "is_evidence_correct",
]


def build_review_sample(
    predictions: pd.DataFrame,
    target: str,
    model: str,
    limit: int,
) -> pd.DataFrame:
    filtered = predictions[
        (predictions["target"] == target)
        & (predictions["model"] == model)
        & (predictions["prediction"].astype(int) == 1)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No predicted-positive rows for target={target} model={model}.")

    filtered = filtered.sort_values(
        by=["score", "id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(limit)

    rows = []
    for _, row in filtered.iterrows():
        evidence = extract_evidence(str(row["text"]), target)
        copied = evidence_is_copied_from_input(str(row["text"]), evidence)
        has_evidence = bool(evidence.evidence_text)
        rows.append(
            {
                "id": row.get("id", ""),
                "dialogue_id": row.get("dialogue_id", ""),
                "window_start_turn": row.get("window_start_turn", ""),
                "window_end_turn": row.get("window_end_turn", ""),
                "text": row["text"],
                "target": target,
                "model": model,
                "gold": int(row["gold"]),
                "prediction": int(row["prediction"]),
                "score": float(row["score"]),
                "evidence_text": evidence.evidence_text,
                "speaker": evidence.speaker,
                "turn_index": evidence.turn_index,
                "reason": evidence.reason,
                "matched_pattern": evidence.matched_pattern,
                "method": evidence.method,
                "confidence_hint": evidence.confidence_hint,
                "evidence_copied_from_input": int(copied),
                "is_evidence_correct": int(has_evidence and copied),
            }
        )

    sample = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(sample) < limit:
        raise ValueError(
            f"Only {len(sample)} predicted-positive rows were available; requested {limit}."
        )
    return sample


def summarize(sample: pd.DataFrame) -> dict[str, float | int]:
    rows = int(len(sample))
    fallback_count = int((sample["method"] == "fallback_no_match").sum())
    copied_count = int(sample["evidence_copied_from_input"].sum())
    correct_count = int(sample["is_evidence_correct"].sum())
    return {
        "rows": rows,
        "fallback_count": fallback_count,
        "copied_count": copied_count,
        "evidence_accuracy": float(correct_count / rows) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="unresolved_customer_concern")
    parser.add_argument("--model", default="open_concern_rule")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required_columns = {"target", "model", "prediction", "gold", "score", "text"}
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Predictions file is missing columns: {missing_columns}")

    sample = build_review_sample(
        predictions=predictions,
        target=args.target,
        model=args.model,
        limit=args.limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    summary = summarize(sample)
    print(f"rows={summary['rows']}")
    print(f"fallback_count={summary['fallback_count']}")
    print(f"copied_count={summary['copied_count']}")
    print(f"evidence_accuracy={summary['evidence_accuracy']:.3f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
