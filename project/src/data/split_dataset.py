"""Create train/validation/test splits grouped by dialogue_id."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    by_dialogue: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_id"]].append(row)

    dialogue_ids = sorted(by_dialogue)
    random.Random(args.seed).shuffle(dialogue_ids)

    train_cut = int(len(dialogue_ids) * args.train_ratio)
    validation_cut = train_cut + int(len(dialogue_ids) * args.validation_ratio)

    split_dialogues = {
        "train": set(dialogue_ids[:train_cut]),
        "validation": set(dialogue_ids[train_cut:validation_cut]),
        "test": set(dialogue_ids[validation_cut:]),
    }

    for split_name, split_ids in split_dialogues.items():
        split_rows = [
            row
            for dialogue_id in sorted(split_ids)
            for row in by_dialogue[dialogue_id]
        ]
        write_csv(args.output_dir / f"{split_name}.csv", split_rows, fieldnames)
        print(f"{split_name}: dialogues={len(split_ids)} rows={len(split_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

