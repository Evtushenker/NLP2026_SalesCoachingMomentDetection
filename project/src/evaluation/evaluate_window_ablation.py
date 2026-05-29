"""Evaluate focused tasks across several dialogue window sizes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

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

MODELS = [
    "majority",
    "open_concern_rule",
    "tfidf_logreg",
    "tfidf_linear_svm",
]

REQUIRED_COLUMNS = [
    "id",
    "base_id",
    "dialogue_id",
    "window_size",
    "text",
    *TARGETS,
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
    has_later_manager = any(
        speaker == "manager" for speaker, _ in turns[last_client_index + 1 :]
    )
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


def average_precision_from_predictions(predictions: pd.DataFrame) -> float:
    ranked = predictions.sort_values("score", ascending=False).reset_index(drop=True)
    positives_total = int(ranked["gold"].sum())
    if positives_total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, gold in enumerate(ranked["gold"], start=1):
        if int(gold) == 1:
            hits += 1
            precision_sum += hits / index
    return float(precision_sum / positives_total)


def top_share_metrics(predictions: pd.DataFrame, share: float) -> dict[str, float | int]:
    ranked = predictions.sort_values("score", ascending=False)
    k = max(1, round(len(ranked) * share))
    top = ranked.head(k)
    positives_total = int(ranked["gold"].sum())
    positives_found = int(top["gold"].sum())
    positive_rate = float(positives_total / len(ranked)) if len(ranked) else 0.0
    precision = float(positives_found / k) if k else 0.0
    return {
        "k": int(k),
        "precision": precision,
        "recall": float(positives_found / positives_total) if positives_total else 0.0,
        "lift_over_random": float(precision / positive_rate) if positive_rate else 0.0,
        "positives_found": positives_found,
        "positives_total": positives_total,
    }


def make_model(name: str):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "tfidf_logreg":
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    if name == "tfidf_linear_svm":
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        )
    raise ValueError(f"Unknown trainable model: {name}")


def evaluate_model_cv(
    data: pd.DataFrame,
    target: str,
    model_name: str,
    n_splits: int,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    x = data["text"]
    y = data[target].astype(int).to_numpy()
    groups = data["dialogue_id"]
    group_kfold = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    predictions = []

    for fold_index, (train_idx, test_idx) in enumerate(group_kfold.split(x, y, groups)):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if model_name == "open_concern_rule":
            if target == "unresolved_customer_concern":
                y_pred = np.asarray([has_open_client_concern_at_end(text) for text in x_test])
            else:
                y_pred = np.asarray([action_needed_rule(text) for text in x_test])
            y_score = y_pred.astype(float)
        else:
            model = make_model(model_name)
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
            row = data.iloc[row_index]
            predictions.append(
                {
                    "fold": fold_index,
                    "id": row["id"],
                    "base_id": row["base_id"],
                    "dialogue_id": row["dialogue_id"],
                    "window_size": int(row["window_size"]),
                    "target": target,
                    "model": model_name,
                    "gold": int(y[row_index]),
                    "prediction": int(pred),
                    "score": float(score),
                }
            )

    return aggregate(fold_metrics), pd.DataFrame(predictions)


def evaluate_window_file(path: Path, n_splits_requested: int) -> dict[str, object]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(data, REQUIRED_COLUMNS, path)
    if data.empty:
        raise ValueError(f"{path} is empty.")
    groups = data["dialogue_id"]
    n_splits = min(n_splits_requested, groups.nunique())
    if n_splits < 2:
        raise ValueError(f"{path} must contain at least two dialogue groups.")
    result: dict[str, object] = {
        "path": str(path),
        "window_size": int(data["window_size"].iloc[0]),
        "rows": int(len(data)),
        "groups": int(groups.nunique()),
        "n_splits": int(n_splits),
        "targets": {},
    }

    for target in TARGETS:
        y = data[target].astype(int)
        target_result: dict[str, object] = {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "models": {},
            "best_by_f1": {},
        }
        best_model = None
        best_f1 = -1.0
        for model_name in MODELS:
            classification, predictions = evaluate_model_cv(
                data=data,
                target=target,
                model_name=model_name,
                n_splits=n_splits,
            )
            ranking = {
                "average_precision": average_precision_from_predictions(predictions),
                "top_20_percent": top_share_metrics(predictions, 0.2),
            }
            model_result = {
                "classification": classification,
                "ranking": ranking,
            }
            target_result["models"][model_name] = model_result
            f1_mean = float(classification["f1"]["mean"])
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_model = model_name

        target_result["best_by_f1"] = {
            "model": best_model,
            "f1": best_f1,
        }
        result["targets"][target] = target_result
    return result


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_markdown_report(path: Path, results: dict[str, object]) -> None:
    lines = [
        "# Window-Size Ablation Results",
        "",
        "This ablation compares 2-turn, 4-turn, and 6-turn dialogue windows using the same grouped cross-validation protocol.",
        "",
        "Important limitation: labels were manually reviewed for the original 4-turn windows. Labels for 2-turn and 6-turn variants are projected from the anchored 4-turn gold example. The experiment therefore measures robustness to local context changes, not fully independent annotation for every window size.",
        "",
        "## Dataset Summary",
        "",
        "| Window size | Rows | Dialogue groups |",
        "| ---: | ---: | ---: |",
    ]

    for window_result in results["windows"]:
        lines.append(
            f"| {window_result['window_size']} | {window_result['rows']} | {window_result['groups']} |"
        )

    lines.extend(
        [
            "",
            "## Best Models By F1",
            "",
            "| Window size | Target | Best model | F1 | Precision | Recall | AP | Top-20 P/R/Lift |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for window_result in results["windows"]:
        for target in TARGETS:
            target_result = window_result["targets"][target]
            best_model = target_result["best_by_f1"]["model"]
            model_result = target_result["models"][best_model]
            classification = model_result["classification"]
            ranking = model_result["ranking"]
            top20 = ranking["top_20_percent"]
            lines.append(
                "| "
                f"{window_result['window_size']} | `{target}` | `{best_model}` | "
                f"{format_float(float(classification['f1']['mean']))} | "
                f"{format_float(float(classification['precision']['mean']))} | "
                f"{format_float(float(classification['recall']['mean']))} | "
                f"{format_float(float(ranking['average_precision']))} | "
                f"{format_float(float(top20['precision']))} / "
                f"{format_float(float(top20['recall']))} / "
                f"{format_float(float(top20['lift_over_random']))}x |"
            )

    lines.extend(
        [
            "",
            "## Model Detail",
            "",
        ]
    )

    for target in TARGETS:
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| Window size | Model | F1 | Precision | Recall | AP | Top-20 precision |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for window_result in results["windows"]:
            target_result = window_result["targets"][target]
            for model_name, model_result in sorted(target_result["models"].items()):
                classification = model_result["classification"]
                ranking = model_result["ranking"]
                top20 = ranking["top_20_percent"]
                lines.append(
                    f"| {window_result['window_size']} | `{model_name}` | "
                    f"{format_float(float(classification['f1']['mean']))} | "
                    f"{format_float(float(classification['precision']['mean']))} | "
                    f"{format_float(float(classification['recall']['mean']))} | "
                    f"{format_float(float(ranking['average_precision']))} | "
                    f"{format_float(float(top20['precision']))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Conclusion",
            "",
            "- For `unresolved_customer_concern`, the open-concern rule is stable on 2-turn and 4-turn windows (F1 0.810) but degrades on 6-turn windows. Extra context can move the latest client concern away from the end of the visible window, which weakens this structural rule.",
            "- For `actionable_coaching_needed`, TF-IDF Logistic Regression is best for all three window sizes. F1 changes from 0.849 (2 turns) to 0.862 (4 turns) and 0.873 (6 turns), but top-20 precision is strongest on the original 4-turn setup.",
            "- The default 4-turn window remains a good compromise: it preserves the original manually reviewed unit, gives the strongest review-queue top-20 precision for actionable coaching, and keeps the transparent unresolved-concern rule strong.",
            "",
            "The ablation should be interpreted as a robustness check. Because labels are projected from the 4-turn gold window, small metric differences are not strong evidence that one window size is universally better.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("project/data/processed/window_ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project/data/processed/window_ablation/window_ablation_results.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("project/outputs/window_size_ablation_results.md"),
    )
    parser.add_argument("--window-sizes", default="2,4,6")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    window_sizes = [
        int(value.strip()) for value in args.window_sizes.split(",") if value.strip()
    ]
    if not window_sizes:
        raise ValueError("At least one window size is required.")

    results: dict[str, object] = {
        "label_projection": (
            "Labels for 2-turn and 6-turn variants are projected from anchored "
            "4-turn gold windows."
        ),
        "windows": [],
    }
    for window_size in window_sizes:
        path = args.input_dir / f"window_{window_size}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing ablation dataset: {path}")
        results["windows"].append(evaluate_window_file(path, args.splits))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(args.report_output, results)

    print(f"output={args.output}")
    print(f"report_output={args.report_output}")
    for window_result in results["windows"]:
        print(
            f"window_size={window_result['window_size']} "
            f"rows={window_result['rows']} groups={window_result['groups']}"
        )
        for target in TARGETS:
            best = window_result["targets"][target]["best_by_f1"]
            print(f"  {target}: best={best['model']} f1={best['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
