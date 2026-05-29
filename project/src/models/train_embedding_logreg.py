"""Train SentenceTransformer embeddings + Logistic Regression baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    return data["text"].tolist(), data[LABEL_COLUMNS].astype(int).to_numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match_ratio": float(np.all(y_true == y_pred, axis=1).mean()),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABEL_COLUMNS)
        },
    }


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray) -> tuple[list[float], float]:
    thresholds: list[float] = []
    predictions = np.zeros_like(y_true)
    for label_index in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.1, 0.9, 17):
            label_pred = (scores[:, label_index] >= threshold).astype(int)
            label_f1 = f1_score(y_true[:, label_index], label_pred, zero_division=0)
            if label_f1 > best_f1:
                best_f1 = label_f1
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
        predictions[:, label_index] = (scores[:, label_index] >= best_threshold).astype(int)
    return thresholds, float(f1_score(y_true, predictions, average="macro", zero_division=0))


def apply_thresholds(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    threshold_array = np.asarray(thresholds).reshape(1, -1)
    return (scores >= threshold_array).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_x, train_y = load_split(args.splits_dir / "train.csv")
    validation_x, validation_y = load_split(args.splits_dir / "validation.csv")
    test_x, test_y = load_split(args.splits_dir / "test.csv")

    encoder = SentenceTransformer(args.model_name)
    train_embeddings = encoder.encode(
        train_x,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    validation_embeddings = encoder.encode(
        validation_x,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    test_embeddings = encoder.encode(
        test_x,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    classifier.fit(train_embeddings, train_y)

    validation_scores = classifier.predict_proba(validation_embeddings)
    thresholds, validation_macro_f1 = tune_thresholds(validation_y, validation_scores)
    validation_pred = apply_thresholds(validation_scores, thresholds)
    validation_pred_fixed = (validation_scores >= 0.5).astype(int)

    test_scores = classifier.predict_proba(test_embeddings)
    test_pred = apply_thresholds(test_scores, thresholds)
    test_pred_fixed = (test_scores >= 0.5).astype(int)

    results = {
        "model": args.model_name,
        "classifier": "OneVsRest LogisticRegression",
        "thresholds": dict(zip(LABEL_COLUMNS, thresholds)),
        "validation_threshold_tuning_macro_f1": validation_macro_f1,
        "validation": evaluate(validation_y, validation_pred),
        "test": evaluate(test_y, test_pred),
        "validation_fixed_0_5": evaluate(validation_y, validation_pred_fixed),
        "test_fixed_0_5": evaluate(test_y, test_pred_fixed),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.predictions_output:
        test_data = pd.read_csv(args.splits_dir / "test.csv", encoding="utf-8-sig")
        rows = []
        for row_index, row in test_data.iterrows():
            output_row = {
                "id": row["id"],
                "dialogue_id": row["dialogue_id"],
                "text": row["text"],
            }
            for label_index, label in enumerate(LABEL_COLUMNS):
                output_row[f"gold_{label}"] = int(test_y[row_index, label_index])
                output_row[f"embedding_logreg_{label}"] = int(
                    test_pred[row_index, label_index]
                )
                output_row[f"embedding_logreg_fixed_0_5_{label}"] = int(
                    test_pred_fixed[row_index, label_index]
                )
            rows.append(output_row)
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"output={args.output}")
    if args.predictions_output:
        print(f"predictions_output={args.predictions_output}")
    print(
        "validation "
        f"micro_f1={results['validation']['micro_f1']:.3f} "
        f"macro_f1={results['validation']['macro_f1']:.3f} "
        f"exact={results['validation']['exact_match_ratio']:.3f}"
    )
    print(
        "test "
        f"micro_f1={results['test']['micro_f1']:.3f} "
        f"macro_f1={results['test']['macro_f1']:.3f} "
        f"exact={results['test']['exact_match_ratio']:.3f}"
    )
    print(
        "test_fixed_0_5 "
        f"micro_f1={results['test_fixed_0_5']['micro_f1']:.3f} "
        f"macro_f1={results['test_fixed_0_5']['macro_f1']:.3f} "
        f"exact={results['test_fixed_0_5']['exact_match_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
