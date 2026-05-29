"""Build anchored window-size ablation datasets from the gold sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "id",
    "base_id",
    "source",
    "dialogue_id",
    "window_size",
    "window_start_turn",
    "window_end_turn",
    "speaker_window",
    "text",
    "recommendation_type",
    "actionable_coaching_needed",
    "unresolved_customer_concern",
    "label_projection_note",
]

GOLD_REQUIRED_COLUMNS = [
    "id",
    "source",
    "dialogue_id",
    "window_start_turn",
    "window_end_turn",
    "recommendation_type",
    "actionable_coaching_needed",
    "unresolved_customer_concern",
]

CANDIDATE_REQUIRED_COLUMNS = [
    "dialogue_id",
    "window_start_turn",
    "text",
]


def require_columns(data: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_window_turns(row: pd.Series) -> list[tuple[int, str, str]]:
    start_turn = int(row["window_start_turn"])
    parsed_turns = []
    for offset, line in enumerate(str(row["text"]).splitlines()):
        if ":" not in line:
            continue
        speaker, utterance = line.split(":", 1)
        parsed_turns.append(
            (
                start_turn + offset,
                speaker.strip().lower(),
                utterance.strip(),
            )
        )
    return parsed_turns


def build_dialogue_turns(candidate_windows: pd.DataFrame) -> dict[str, dict[int, tuple[str, str]]]:
    dialogue_turns: dict[str, dict[int, tuple[str, str]]] = {}
    for _, row in candidate_windows.iterrows():
        dialogue_id = str(row["dialogue_id"])
        turn_map = dialogue_turns.setdefault(dialogue_id, {})
        for turn_index, speaker, utterance in parse_window_turns(row):
            previous = turn_map.get(turn_index)
            current = (speaker, utterance)
            if previous is not None and previous != current:
                raise ValueError(
                    f"Conflicting text for dialogue={dialogue_id} turn={turn_index}."
                )
            turn_map[turn_index] = current
    return dialogue_turns


def choose_window_bounds(
    base_start: int,
    base_end: int,
    window_size: int,
    min_turn: int,
    max_turn: int,
) -> tuple[int, int] | None:
    available_turns = max_turn - min_turn + 1
    if available_turns < window_size:
        return None

    if window_size == base_end - base_start + 1:
        start = base_start
    elif window_size < base_end - base_start + 1:
        start = base_end - window_size + 1
    else:
        extra = window_size - (base_end - base_start + 1)
        left_extra = extra // 2
        start = base_start - left_extra

    start = max(min_turn, min(start, max_turn - window_size + 1))
    end = start + window_size - 1
    return start, end


def make_ablation_row(
    gold_row: pd.Series,
    turn_map: dict[int, tuple[str, str]],
    window_size: int,
) -> dict[str, object] | None:
    base_start = int(gold_row["window_start_turn"])
    base_end = int(gold_row["window_end_turn"])
    min_turn = min(turn_map)
    max_turn = max(turn_map)
    bounds = choose_window_bounds(
        base_start=base_start,
        base_end=base_end,
        window_size=window_size,
        min_turn=min_turn,
        max_turn=max_turn,
    )
    if bounds is None:
        return None

    start, end = bounds
    missing_turns = [turn_index for turn_index in range(start, end + 1) if turn_index not in turn_map]
    if missing_turns:
        return None

    window = [turn_map[turn_index] for turn_index in range(start, end + 1)]
    text = "\n".join(f"{speaker}: {utterance}" for speaker, utterance in window)
    speaker_window = ",".join(speaker for speaker, _ in window)
    base_id = str(gold_row["id"])
    dialogue_id = str(gold_row["dialogue_id"])
    return {
        "id": f"{base_id}_ws{window_size}",
        "base_id": base_id,
        "source": gold_row["source"],
        "dialogue_id": dialogue_id,
        "window_size": int(window_size),
        "window_start_turn": int(start),
        "window_end_turn": int(end),
        "speaker_window": speaker_window,
        "text": text,
        "recommendation_type": gold_row["recommendation_type"],
        "actionable_coaching_needed": int(gold_row["actionable_coaching_needed"]),
        "unresolved_customer_concern": int(gold_row["unresolved_customer_concern"]),
        "label_projection_note": (
            "Gold labels are projected from the anchored 4-turn reviewed window."
        ),
    }


def build_ablation_frames(
    gold: pd.DataFrame,
    candidate_windows: pd.DataFrame,
    window_sizes: list[int],
) -> dict[int, pd.DataFrame]:
    dialogue_turns = build_dialogue_turns(candidate_windows)
    frames: dict[int, pd.DataFrame] = {}
    for window_size in window_sizes:
        rows = []
        for _, gold_row in gold.iterrows():
            dialogue_id = str(gold_row["dialogue_id"])
            if dialogue_id not in dialogue_turns:
                continue
            row = make_ablation_row(gold_row, dialogue_turns[dialogue_id], window_size)
            if row is not None:
                rows.append(row)
        frames[window_size] = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("project/data/samples/gwenshap_focused_gold_tasks.csv"),
    )
    parser.add_argument(
        "--candidate-windows",
        type=Path,
        default=Path("project/data/processed/gwenshap_candidate_windows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project/data/processed/window_ablation"),
    )
    parser.add_argument(
        "--window-sizes",
        default="2,4,6",
        help="Comma-separated window sizes to build.",
    )
    args = parser.parse_args()

    gold = pd.read_csv(args.gold, encoding="utf-8-sig")
    candidate_windows = pd.read_csv(args.candidate_windows, encoding="utf-8-sig")
    window_sizes = [
        int(value.strip()) for value in args.window_sizes.split(",") if value.strip()
    ]
    if not window_sizes:
        raise ValueError("At least one window size is required.")
    invalid_window_sizes = [window_size for window_size in window_sizes if window_size <= 0]
    if invalid_window_sizes:
        raise ValueError(f"Window sizes must be positive: {invalid_window_sizes}")
    require_columns(gold, GOLD_REQUIRED_COLUMNS, args.gold)
    require_columns(candidate_windows, CANDIDATE_REQUIRED_COLUMNS, args.candidate_windows)
    if gold.empty:
        raise ValueError(f"{args.gold} is empty.")
    if candidate_windows.empty:
        raise ValueError(f"{args.candidate_windows} is empty.")

    frames = build_ablation_frames(gold, candidate_windows, window_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for window_size, frame in frames.items():
        output_path = args.output_dir / f"window_{window_size}.csv"
        if frame.empty:
            raise ValueError(f"No ablation rows were built for window_size={window_size}.")
        frame.to_csv(output_path, index=False)
        print(
            f"window_size={window_size} rows={len(frame)} "
            f"groups={frame['dialogue_id'].nunique() if len(frame) else 0} "
            f"output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
