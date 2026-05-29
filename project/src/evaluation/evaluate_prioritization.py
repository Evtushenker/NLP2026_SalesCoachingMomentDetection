"""Evaluate how well model scores prioritize windows for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["target", "model", "gold", "score"]


def require_columns(data: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def precision_recall_at_k(data: pd.DataFrame, k: int, requested_share: float) -> dict[str, float]:
    ranked = data.sort_values("score", ascending=False).head(k)
    positives_total = int(data["gold"].sum())
    positives_found = int(ranked["gold"].sum())
    rows = len(data)
    random_precision = float(positives_total / rows) if rows else 0.0
    precision = float(positives_found / k) if k else 0.0
    return {
        "k": int(k),
        "requested_review_share": float(requested_share),
        "precision_at_k": precision,
        "recall_at_k": float(positives_found / positives_total)
        if positives_total
        else 0.0,
        "random_precision": random_precision,
        "lift_over_random": float(precision / random_precision)
        if random_precision
        else 0.0,
        "positives_found": positives_found,
        "positives_total": positives_total,
        "review_share": float(k / rows) if rows else 0.0,
    }


def average_precision(data: pd.DataFrame) -> float:
    ranked = data.sort_values("score", ascending=False).reset_index(drop=True)
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


def evaluate(data: pd.DataFrame, review_shares: list[float]) -> dict[str, object]:
    result: dict[str, object] = {}
    for (target, model), group in data.groupby(["target", "model"]):
        key = f"{target}::{model}"
        rows = int(len(group))
        positives = int(group["gold"].sum())
        group_result = {
            "rows": rows,
            "positive": positives,
            "random_precision": float(positives / rows) if rows else 0.0,
            "average_precision": average_precision(group),
            "cutoffs": [],
        }
        for share in review_shares:
            k = max(1, round(len(group) * share))
            group_result["cutoffs"].append(precision_recall_at_k(group, k, share))
        result[key] = group_result
    return result


def best_models_by_target(result: dict[str, object]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for key, metrics in result.items():
        target, model = key.split("::", 1)
        current = best.get(target)
        average_precision_value = float(metrics["average_precision"])
        if current is None or average_precision_value > float(current["average_precision"]):
            best[target] = {
                "model": model,
                "average_precision": average_precision_value,
            }
    return best


def format_float(value: float) -> str:
    return f"{value:.3f}"


def format_share(share: float) -> str:
    return f"{share:.0%}"


def write_markdown_report(
    path: Path,
    result: dict[str, object],
    review_shares: list[float],
    predictions_path: Path,
) -> None:
    best = best_models_by_target(result)
    lines = [
        "# Prioritization Results",
        "",
        "This report evaluates how well out-of-fold model scores rank dialogue windows for human review.",
        "",
        "Input predictions:",
        "",
        f"```text\n{predictions_path}\n```",
        "",
        "Metrics:",
        "",
        "- `Average Precision` summarizes ranking quality across all thresholds.",
        "- `precision@top-k` is the share of reviewed windows that are true positives.",
        "- `recall@top-k` is the share of all positives found in the reviewed subset.",
        "- `lift over random` is `precision@top-k / positive_rate`; values above 1 mean the queue is better than random review.",
        "",
        "## Best Ranking Models",
        "",
        "| Target | Best ranking model | Average Precision |",
        "| --- | --- | ---: |",
    ]

    for target in sorted(best):
        entry = best[target]
        lines.append(
            f"| `{target}` | `{entry['model']}` | {format_float(float(entry['average_precision']))} |"
        )

    for target in sorted(best):
        cutoff_headers = [
            f"Top {format_share(float(share))} P/R/Lift" for share in review_shares
        ]
        lines.extend(
            [
                "",
                f"## `{target}`",
                "",
                "| Model | Positive rate | Average Precision | "
                + " | ".join(cutoff_headers)
                + " |",
                "| --- | ---: | ---: | "
                + " | ".join("---:" for _ in review_shares)
                + " |",
            ]
        )
        rows = []
        for key, metrics in result.items():
            row_target, model = key.split("::", 1)
            if row_target != target:
                continue
            cutoff_by_share = {
                round(float(cutoff["requested_review_share"]), 3): cutoff
                for cutoff in metrics["cutoffs"]
            }
            cells = []
            for share in review_shares:
                cutoff = cutoff_by_share[round(float(share), 3)]
                cells.append(
                    (
                        f"{format_float(float(cutoff['precision_at_k']))} / "
                        f"{format_float(float(cutoff['recall_at_k']))} / "
                        f"{format_float(float(cutoff['lift_over_random']))}x"
                    )
                )
            rows.append(
                (
                    float(metrics["average_precision"]),
                    (
                        f"| `{model}` | {format_float(float(metrics['random_precision']))} "
                        f"| {format_float(float(metrics['average_precision']))} "
                        f"| {' | '.join(cells)} |"
                    ),
                )
            )
        for _, row in sorted(rows, reverse=True):
            lines.append(row)

    interpretation_lines = []
    preferred_share = 0.2 if 0.2 in review_shares else review_shares[0]
    for target in sorted(best):
        model = str(best[target]["model"])
        metrics = result[f"{target}::{model}"]
        cutoff = min(
            metrics["cutoffs"],
            key=lambda item: abs(float(item["requested_review_share"]) - preferred_share),
        )
        interpretation_lines.append(
            (
                f"- For `{target}`, `{model}` is the strongest ranking model by Average Precision. "
                f"At top {format_share(float(cutoff['requested_review_share']))}, it finds "
                f"{int(cutoff['positives_found'])} positive windows out of "
                f"{int(cutoff['k'])} reviewed, which is "
                f"{format_float(float(cutoff['lift_over_random']))}x better than random review."
            )
        )

    lines.extend(
        [
            "",
            "## Practical Interpretation",
            "",
            "For the current 100-window gold sample, each review budget corresponds to the same number of reviewed windows as its percentage value.",
            "",
            *interpretation_lines,
            "",
            "These results support the product framing: the system is most useful as a prioritization queue for a sales coach, not as a fully automatic call scorer.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("project/outputs/prioritization_results.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--review-shares",
        default="0.1,0.2,0.3",
        help="Comma-separated review budget shares.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    require_columns(data, REQUIRED_COLUMNS, args.predictions)
    if data.empty:
        raise ValueError("Predictions file is empty.")

    review_shares = [
        float(value.strip())
        for value in args.review_shares.split(",")
        if value.strip()
    ]
    if not review_shares:
        raise ValueError("At least one review share is required.")
    invalid_shares = [share for share in review_shares if share <= 0 or share > 1]
    if invalid_shares:
        raise ValueError(f"Review shares must be in the interval (0, 1]: {invalid_shares}")
    result = evaluate(data, review_shares)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(args.report_output, result, review_shares, args.predictions)

    print(f"output={args.output}")
    print(f"report_output={args.report_output}")
    for key, metrics in result.items():
        print(
            f"{key}: AP={metrics['average_precision']:.3f} "
            f"positive={metrics['positive']}/{metrics['rows']} "
            f"random_precision={metrics['random_precision']:.3f}"
        )
        for cutoff in metrics["cutoffs"]:
            print(
                f"  top {cutoff['review_share']:.0%}: "
                f"P@k={cutoff['precision_at_k']:.3f} "
                f"R@k={cutoff['recall_at_k']:.3f} "
                f"lift={cutoff['lift_over_random']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
