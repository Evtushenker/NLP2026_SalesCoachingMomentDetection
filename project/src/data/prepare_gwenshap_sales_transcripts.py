"""Download and window the gwenshap/sales-transcripts dataset.

The script intentionally uses only the Python standard library so that the
initial data preparation works before the ML environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlopen


DATASET_ID = "gwenshap/sales-transcripts"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
RAW_BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"

OUTPUT_COLUMNS = [
    "id",
    "source",
    "dialogue_id",
    "window_start_turn",
    "window_end_turn",
    "speaker_window",
    "text",
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
    "recommendation_type",
    "notes",
]

TURN_RE = re.compile(
    r"^(?:\*\*)?(?P<speaker>Sales Rep|Sales Representative|Representative|Customer|Client|Prospect)(?:\*\*)?:\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)


def fetch_json(url: str) -> dict:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def list_transcript_files() -> list[str]:
    metadata = fetch_json(API_URL)
    siblings = metadata.get("siblings", [])
    files = [
        item["rfilename"]
        for item in siblings
        if item.get("rfilename", "").startswith("data/transcripts/")
        and item.get("rfilename", "").endswith("_transcript.txt")
    ]
    return sorted(files)


def normalize_speaker(speaker: str) -> str:
    normalized = speaker.strip().lower()
    if normalized in {"sales rep", "sales representative", "representative"}:
        return "manager"
    if normalized in {"customer", "client", "prospect"}:
        return "client"
    return normalized.replace(" ", "_")


def parse_turns(transcript: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TURN_RE.match(line)
        if not match:
            continue
        speaker = normalize_speaker(match.group("speaker"))
        text = match.group("text").strip()
        turns.append((speaker, text))
    return turns


def make_windows(
    turns: list[tuple[str, str]],
    dialogue_id: str,
    window_size: int,
    stride: int,
) -> Iterable[dict[str, str]]:
    if len(turns) < window_size:
        return
    for start in range(0, len(turns) - window_size + 1, stride):
        end = start + window_size
        window = turns[start:end]
        text = "\n".join(f"{speaker}: {utterance}" for speaker, utterance in window)
        speakers = ",".join(speaker for speaker, _ in window)
        yield {
            "id": f"gwenshap_{dialogue_id}_w{start + 1:03d}_{end:03d}",
            "source": "public_synthetic",
            "dialogue_id": dialogue_id,
            "window_start_turn": str(start + 1),
            "window_end_turn": str(end),
            "speaker_window": speakers,
            "text": text,
            "needs_discovery": "",
            "value_articulation": "",
            "objection_handling": "",
            "next_step_fixed": "",
            "early_presentation_before_discovery": "",
            "recommendation_type": "",
            "notes": "",
        }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("project/data/raw/gwenshap_sales_transcripts/transcripts"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("project/data/processed/gwenshap_candidate_windows.csv"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("project/data/samples/gwenshap_sample_100_windows.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    transcript_files = list_transcript_files()
    if not transcript_files:
        print("No transcript files found.", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    for remote_path in transcript_files:
        url = f"{RAW_BASE_URL}/{quote(remote_path)}"
        transcript = fetch_text(url)
        local_path = args.raw_dir / Path(remote_path).name
        local_path.write_text(transcript, encoding="utf-8")

        dialogue_id = Path(remote_path).stem.replace("_transcript", "")
        turns = parse_turns(transcript)
        all_rows.extend(
            make_windows(
                turns=turns,
                dialogue_id=dialogue_id,
                window_size=args.window_size,
                stride=args.stride,
            )
        )

    write_csv(args.candidate_output, all_rows)
    write_csv(args.sample_output, all_rows[: args.sample_size])

    print(f"Downloaded transcripts: {len(transcript_files)}")
    print(f"Candidate windows: {len(all_rows)}")
    print(f"Candidate output: {args.candidate_output}")
    print(f"Sample output: {args.sample_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
