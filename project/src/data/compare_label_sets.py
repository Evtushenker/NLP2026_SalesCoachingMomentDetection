"""Compare two labeled CSV files by row id."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]


def load_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return {row["id"]: row for row in csv.DictReader(source)}


def as_int(row: dict[str, str], column: str) -> int:
    return int(row[column])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    predicted = load_by_id(args.predicted)
    gold = load_by_id(args.gold)
    common_ids = sorted(set(predicted) & set(gold))

    if not common_ids:
        raise ValueError("No common ids between files.")

    result: dict[str, object] = {
        "rows_predicted": len(predicted),
        "rows_gold": len(gold),
        "rows_compared": len(common_ids),
        "labels": {},
        "recommendation_type": {},
    }

    for label in LABEL_COLUMNS:
        y_pred = [as_int(predicted[row_id], label) for row_id in common_ids]
        y_gold = [as_int(gold[row_id], label) for row_id in common_ids]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_gold,
            y_pred,
            average="binary",
            zero_division=0,
        )
        result["labels"][label] = {
            "accuracy": sum(a == b for a, b in zip(y_pred, y_gold)) / len(common_ids),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cohen_kappa": cohen_kappa_score(y_gold, y_pred),
            "predicted_positive": sum(y_pred),
            "gold_positive": sum(y_gold),
        }

    pred_matrix = [[as_int(predicted[row_id], label) for label in LABEL_COLUMNS] for row_id in common_ids]
    gold_matrix = [[as_int(gold[row_id], label) for label in LABEL_COLUMNS] for row_id in common_ids]
    exact_matches = sum(pred == gold for pred, gold in zip(pred_matrix, gold_matrix))
    result["multi_label"] = {
        "exact_match_ratio": exact_matches / len(common_ids),
        "micro_f1": f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0),
        "macro_f1": f1_score(gold_matrix, pred_matrix, average="macro", zero_division=0),
    }

    pred_rec = [predicted[row_id].get("recommendation_type", "") for row_id in common_ids]
    gold_rec = [gold[row_id].get("recommendation_type", "") for row_id in common_ids]
    rec_matches = sum(a == b for a, b in zip(pred_rec, gold_rec))
    result["recommendation_type"] = {
        "accuracy": rec_matches / len(common_ids),
        "predicted_counts": dict(Counter(pred_rec)),
        "gold_counts": dict(Counter(gold_rec)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"rows_compared={len(common_ids)}")
    print(
        "multi_label "
        f"micro_f1={result['multi_label']['micro_f1']:.3f} "
        f"macro_f1={result['multi_label']['macro_f1']:.3f} "
        f"exact={result['multi_label']['exact_match_ratio']:.3f}"
    )
    for label, metrics in result["labels"].items():
        print(
            f"{label}: acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"kappa={metrics['cohen_kappa']:.3f} "
            f"pred_pos={metrics['predicted_positive']} "
            f"gold_pos={metrics['gold_positive']}"
        )
    print(f"recommendation_accuracy={result['recommendation_type']['accuracy']:.3f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
