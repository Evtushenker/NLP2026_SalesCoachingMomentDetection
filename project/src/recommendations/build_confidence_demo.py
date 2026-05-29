"""Build an integrated prediction -> evidence -> recommendation demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .confidence import assign_confidence, thresholds_summary
    from .evidence import evidence_is_copied_from_input, extract_evidence
    from .rules import LABEL_COLUMNS, recommend
except ImportError:
    from confidence import assign_confidence, thresholds_summary
    from evidence import evidence_is_copied_from_input, extract_evidence
    from rules import LABEL_COLUMNS, recommend


FALLBACK_RECOMMENDATION_TYPE = "manual_review"
FALLBACK_RECOMMENDATION = (
    "Manual review recommended: the model score is too low for a specific coaching "
    "recommendation."
)

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
    "score_type",
    "confidence_level",
    "review_required",
    "fallback_used",
    "recommendation_type",
    "recommendation",
    "recommendation_rationale",
    "evidence_text",
    "evidence_speaker",
    "evidence_turn_index",
    "evidence_reason",
    "evidence_method",
    "evidence_matched_pattern",
    "evidence_copied_from_input",
    "decision_reason",
]


def labels_for_recommendation(
    target: str,
    prediction: int,
    evidence_method: str,
) -> dict[str, int]:
    labels = {label: 1 for label in LABEL_COLUMNS}
    if int(prediction) == 0:
        return labels

    if target == "unresolved_customer_concern":
        labels["objection_handling"] = 0
        return labels

    if target == "actionable_coaching_needed":
        if evidence_method == "latest_open_client_concern":
            labels["objection_handling"] = 0
        elif evidence_method in {"next_step_need_pattern", "manager_action_pattern"}:
            labels["next_step_fixed"] = 0
        else:
            labels["needs_discovery"] = 0
    return labels


def build_recommendation(
    row: pd.Series,
    fallback_used: bool,
    evidence_method: str,
) -> tuple[str, str, str]:
    if fallback_used:
        return (
            FALLBACK_RECOMMENDATION_TYPE,
            FALLBACK_RECOMMENDATION,
            "Low-confidence or negative prediction; specific advice is suppressed.",
        )

    labels = labels_for_recommendation(
        target=str(row["target"]),
        prediction=int(row["prediction"]),
        evidence_method=evidence_method,
    )
    recommendation = recommend(labels, text=str(row["text"]))
    return (
        recommendation.recommendation_type,
        recommendation.action,
        recommendation.rationale,
    )


def _select_group(
    predictions: pd.DataFrame,
    target: str,
    model: str,
    prediction: int,
    count: int,
    ascending: bool,
) -> pd.DataFrame:
    group = predictions[
        (predictions["target"] == target)
        & (predictions["model"] == model)
        & (predictions["prediction"].astype(int) == prediction)
    ].copy()
    return group.sort_values(
        by=["score", "id"],
        ascending=[ascending, True],
        kind="mergesort",
    ).head(count)


def select_demo_rows(predictions: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive.")

    preferred_counts = [
        (
            "unresolved_customer_concern",
            "open_concern_rule",
            1,
            max(1, round(limit * 0.4)),
            False,
        ),
        (
            "actionable_coaching_needed",
            "tfidf_logreg",
            1,
            max(1, round(limit * 0.4)),
            False,
        ),
        (
            "actionable_coaching_needed",
            "tfidf_logreg",
            0,
            max(1, limit - (2 * max(1, round(limit * 0.4)))),
            True,
        ),
    ]

    parts = [
        _select_group(predictions, target, model, prediction, count, ascending)
        for target, model, prediction, count, ascending in preferred_counts
    ]
    selected = pd.concat(parts, ignore_index=True)

    if len(selected) < limit:
        selected_keys = set(
            zip(selected["target"], selected["model"], selected["source_row_index"])
        )
        preferred = predictions[
            (
                (predictions["target"] == "unresolved_customer_concern")
                & (predictions["model"] == "open_concern_rule")
            )
            | (
                (predictions["target"] == "actionable_coaching_needed")
                & (predictions["model"] == "tfidf_logreg")
            )
        ].copy()
        preferred = preferred[
            ~preferred.apply(
                lambda row: (
                    row["target"],
                    row["model"],
                    row["source_row_index"],
                )
                in selected_keys,
                axis=1,
            )
        ]
        selected = pd.concat(
            [
                selected,
                preferred.sort_values(
                    by=["prediction", "score", "id"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).head(limit - len(selected)),
            ],
            ignore_index=True,
        )

    if len(selected) < limit:
        raise ValueError(f"Only {len(selected)} rows were available; requested {limit}.")
    return selected.head(limit)


def build_demo(predictions: pd.DataFrame, limit: int) -> pd.DataFrame:
    selected = select_demo_rows(predictions, limit)
    output_rows = []
    for _, row in selected.iterrows():
        decision = assign_confidence(
            score=float(row["score"]),
            model=str(row["model"]),
            prediction=int(row["prediction"]),
            target=str(row["target"]),
        )
        evidence = extract_evidence(str(row["text"]), str(row["target"]))
        copied = evidence_is_copied_from_input(str(row["text"]), evidence)
        recommendation_type, recommendation_text, recommendation_rationale = (
            build_recommendation(row, decision.fallback_used, evidence.method)
        )

        output_rows.append(
            {
                "id": row.get("id", ""),
                "dialogue_id": row.get("dialogue_id", ""),
                "window_start_turn": row.get("window_start_turn", ""),
                "window_end_turn": row.get("window_end_turn", ""),
                "text": row["text"],
                "target": row["target"],
                "model": row["model"],
                "gold": int(row["gold"]),
                "prediction": int(row["prediction"]),
                "score": float(row["score"]),
                "score_type": decision.score_type,
                "confidence_level": decision.confidence_level,
                "review_required": int(decision.review_required),
                "fallback_used": int(decision.fallback_used),
                "recommendation_type": recommendation_type,
                "recommendation": recommendation_text,
                "recommendation_rationale": recommendation_rationale,
                "evidence_text": evidence.evidence_text,
                "evidence_speaker": evidence.speaker,
                "evidence_turn_index": evidence.turn_index,
                "evidence_reason": evidence.reason,
                "evidence_method": evidence.method,
                "evidence_matched_pattern": evidence.matched_pattern,
                "evidence_copied_from_input": int(copied),
                "decision_reason": decision.decision_reason,
            }
        )
    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)


def summarize(demo: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(demo)),
        "confidence_counts": demo["confidence_level"].value_counts().to_dict(),
        "fallback_count": int(demo["fallback_used"].sum()),
        "review_required_count": int(demo["review_required"].sum()),
        "specific_recommendation_count": int((demo["fallback_used"] == 0).sum()),
        "low_confidence_specific_advice_count": int(
            (
                (demo["confidence_level"] == "low")
                & (demo["fallback_used"].astype(int) == 0)
            ).sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required_columns = {
        "source_row_index",
        "id",
        "target",
        "model",
        "gold",
        "prediction",
        "score",
        "text",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Predictions file is missing columns: {missing_columns}")

    demo = build_demo(predictions, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(args.output, index=False)

    summary = summarize(demo)
    print(f"rows={summary['rows']}")
    print(f"confidence_counts={summary['confidence_counts']}")
    print(f"fallback_count={summary['fallback_count']}")
    print(f"review_required_count={summary['review_required_count']}")
    print(f"specific_recommendation_count={summary['specific_recommendation_count']}")
    print(
        "low_confidence_specific_advice_count="
        f"{summary['low_confidence_specific_advice_count']}"
    )
    thresholds = thresholds_summary()
    print(
        "thresholds="
        f"medium>={thresholds['medium_threshold']:.2f}, "
        f"high>={thresholds['high_threshold']:.2f}"
    )
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
