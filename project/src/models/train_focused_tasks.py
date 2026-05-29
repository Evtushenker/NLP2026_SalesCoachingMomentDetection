"""Evaluate focused binary sales-coaching tasks with grouped CV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TARGETS = [
    "unresolved_customer_concern",
    "actionable_coaching_needed",
]

CONCERN_PATTERNS = [
    r"\bconcern",
    r"\bworried\b",
    r"\bworry\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bskeptical\b",
    r"\bhesitant\b",
    r"\bexpensive\b",
    r"\bcost\b",
    r"\bprice\b",
    r"\bpricing\b",
    r"\bbudget\b",
    r"\bdowntime\b",
    r"\blearning curve\b",
    r"\bquality\b",
    r"\bfit\b",
    r"\bsizing\b",
    r"\breturn",
    r"\bshipping\b",
    r"\bdifficult\b",
    r"\bhassle\b",
]

NEXT_STEP_NEED_PATTERNS = [
    r"\bnext step",
    r"\bwhat would be the next",
    r"\binterested in finding out more",
    r"\bi'?ll go with\b",
    r"\bthat sounds good\b",
    r"\bthat sounds pretty good\b",
]


def has_open_client_concern_at_end(text: str) -> int:
    turns: list[tuple[str, str]] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        speaker, utterance = line.split(":", 1)
        turns.append((speaker.strip().lower(), utterance.strip()))
    if not turns:
        return 0

    last_client_index = None
    for index, (speaker, _) in enumerate(turns):
        if speaker == "client":
            last_client_index = index
    if last_client_index is None:
        return 0

    last_client_text = turns[last_client_index][1]
    has_concern = any(
        re.search(pattern, last_client_text, flags=re.IGNORECASE)
        for pattern in CONCERN_PATTERNS
    )
    has_later_manager = any(speaker == "manager" for speaker, _ in turns[last_client_index + 1 :])
    return int(has_concern and not has_later_manager)


def action_needed_rule(text: str) -> int:
    if has_open_client_concern_at_end(text):
        return 1
    return int(
        any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NEXT_STEP_NEED_PATTERNS)
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def aggregate(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = fold_metrics[0].keys()
    return {key: mean_std([fold[key] for fold in fold_metrics]) for key in keys}


def require_columns(data: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def metadata_value(row: pd.Series, key: str, default: Any = "") -> Any:
    return row[key] if key in row.index else default


def prediction_record(
    fold_index: int,
    row_index: int,
    row: pd.Series,
    target: str,
    model_name: str,
    gold: int,
    prediction: int,
    score: float,
    text: str,
) -> dict[str, object]:
    return {
        "fold": int(fold_index),
        "source_row_index": int(row_index),
        "id": metadata_value(row, "id", row_index),
        "dialogue_id": metadata_value(row, "dialogue_id"),
        "window_start_turn": int(metadata_value(row, "window_start_turn", -1)),
        "window_end_turn": int(metadata_value(row, "window_end_turn", -1)),
        "recommendation_type": metadata_value(row, "recommendation_type"),
        "target": target,
        "model": model_name,
        "gold": int(gold),
        "prediction": int(prediction),
        "score": float(score),
        "text": text,
    }


def evaluate_model_cv(
    name: str,
    x: pd.Series,
    y: np.ndarray,
    groups: pd.Series,
    metadata: pd.DataFrame,
    target: str,
    n_splits: int,
) -> tuple[dict, list[dict]]:
    group_kfold = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    predictions = []

    for fold_index, (train_idx, test_idx) in enumerate(group_kfold.split(x, y, groups)):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if name == "majority":
            model = DummyClassifier(strategy="most_frequent")
        elif name == "tfidf_logreg":
            model = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                    (
                        "clf",
                        LogisticRegression(max_iter=1000, class_weight="balanced"),
                    ),
                ]
            )
        elif name == "tfidf_linear_svm":
            model = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                    ("clf", LinearSVC(class_weight="balanced")),
                ]
            )
        elif name == "open_concern_rule":
            if target == "unresolved_customer_concern":
                y_pred = np.asarray(
                    [has_open_client_concern_at_end(text) for text in x_test]
                )
            else:
                y_pred = np.asarray([action_needed_rule(text) for text in x_test])
            y_score = y_pred.astype(float)
            fold_metrics.append(metrics(y_test, y_pred))
            for row_index, pred, score in zip(test_idx, y_pred, y_score):
                row = metadata.iloc[row_index]
                predictions.append(
                    prediction_record(
                        fold_index=fold_index,
                        row_index=row_index,
                        row=row,
                        target=target,
                        model_name=name,
                        gold=int(y[row_index]),
                        prediction=int(pred),
                        score=float(score),
                        text=x.iloc[row_index],
                    )
                )
            continue
        else:
            raise ValueError(f"Unknown model: {name}")

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(x_test)
        else:
            y_score = y_pred.astype(float)
        fold_metrics.append(metrics(y_test, y_pred))
        for row_index, pred, score in zip(test_idx, y_pred, y_score):
            row = metadata.iloc[row_index]
            predictions.append(
                prediction_record(
                    fold_index=fold_index,
                    row_index=row_index,
                    row=row,
                    target=target,
                    model_name=name,
                    gold=int(y[row_index]),
                    prediction=int(pred),
                    score=float(score),
                    text=x.iloc[row_index],
                )
            )

    return aggregate(fold_metrics), predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-output", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(args.input, encoding="utf-8-sig")
    require_columns(data, ["text", "dialogue_id", *TARGETS], args.input)
    x = data["text"]
    groups = data["dialogue_id"]
    n_splits = min(args.splits, groups.nunique())
    if n_splits < 2:
        raise ValueError("Grouped cross-validation requires at least two dialogue groups.")

    results: dict[str, object] = {
        "rows": int(len(data)),
        "groups": int(groups.nunique()),
        "n_splits": int(n_splits),
        "targets": {},
    }
    all_predictions: list[dict] = []

    for target in TARGETS:
        y = data[target].astype(int).to_numpy()
        target_results = {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "models": {},
        }
        for model_name in [
            "majority",
            "open_concern_rule",
            "tfidf_logreg",
            "tfidf_linear_svm",
        ]:
            model_result, predictions = evaluate_model_cv(
                model_name,
                x,
                y,
                groups,
                data,
                target,
                n_splits,
            )
            target_results["models"][model_name] = model_result
            all_predictions.extend(predictions)
        results["targets"][target] = target_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    predictions_df = pd.DataFrame(all_predictions)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.predictions_output, index=False)

    print(f"output={args.output}")
    print(f"predictions_output={args.predictions_output}")
    for target, target_result in results["targets"].items():
        print(target, f"positive={target_result['positive']}", f"negative={target_result['negative']}")
        for model_name, model_result in target_result["models"].items():
            print(
                f"  {model_name}: "
                f"f1={model_result['f1']['mean']:.3f}+/-{model_result['f1']['std']:.3f} "
                f"precision={model_result['precision']['mean']:.3f} "
                f"recall={model_result['recall']['mean']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
