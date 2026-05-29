"""Create an initial weakly labeled version of sales dialogue windows.

These labels are not a substitute for final human validation. They are a
practical first pass for bootstrapping annotation, class-balance checks, and
baseline development.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]


QUESTION_PATTERNS = [
    r"\bwhat\b.*\?",
    r"\bwhich\b.*\?",
    r"\bhow\b.*\?",
    r"\bwhen\b.*\?",
    r"\bwhy\b.*\?",
    r"\bmay i ask\b",
    r"\bcan you tell\b",
    r"\bcould you tell\b",
    r"\blooking for\b",
    r"\bwhat.*looking\b",
    r"\bwhat.*important\b",
    r"\bwhat.*specific\b",
    r"\bwhat.*concerns\b",
    r"\bwhat.*challenge\b",
    r"\bwhat.*budget\b",
    r"\bwhat.*timeline\b",
    r"\bwhat.*size\b",
    r"\bwhat.*features\b",
    r"\bwhat.*criteria\b",
    r"\bwhat.*goals\b",
    r"\bwhat.*needs\b",
    r"\bwhat.*requirement\b",
    r"\bwhat.*pain\b",
    r"\bwhat.*issue\b",
    r"\bwhat.*problem\b",
]

VALUE_PATTERNS = [
    r"\bour\b.*\b(can|helps?|offers?|provides?|designed|built|allows?|enables?)\b",
    r"\bwe\b.*\b(can|help|offer|provide|specialize|ensure|support)\b",
    r"\bbenefit\b",
    r"\bvalue\b",
    r"\bsolution\b",
    r"\bfeature\b",
    r"\bhigh-quality\b",
    r"\bflexible return\b",
    r"\bdiscount\b",
    r"\btestimonial\b",
    r"\bcase stud",
    r"\bsecurity\b",
    r"\bscalable\b",
    r"\bintegrat",
    r"\beasy to\b",
    r"\bstreamline\b",
    r"\bautomate\b",
    r"\bsave\b.*\btime\b",
    r"\breduce\b.*\bcost\b",
    r"\bincrease\b.*\b(revenue|conversion|efficiency|productivity)\b",
]

OBJECTION_PATTERNS = [
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
    r"\btoo much\b",
    r"\bissue\b",
    r"\bproblem\b",
    r"\bdifficult\b",
    r"\bmisleading\b",
    r"\bquality\b",
    r"\bsecurity\b",
    r"\bprivacy\b",
    r"\bintegration\b",
    r"\bimplementation\b",
    r"\bfit\b",
    r"\bsize\b",
    r"\breturn\b",
]

OBJECTION_RESPONSE_PATTERNS = [
    r"\bi understand\b",
    r"\bi hear you\b",
    r"\bthat'?s a valid\b",
    r"\bvalid point\b",
    r"\bcompletely understand\b",
    r"\bto address\b",
    r"\bwe can\b",
    r"\bwe offer\b",
    r"\bwe provide\b",
    r"\bi can send\b",
    r"\bi'll send\b",
    r"\blet'?s compare\b",
    r"\bpolicy\b",
    r"\bguarantee\b",
    r"\btrial\b",
    r"\bdemo\b",
    r"\btestimonial\b",
]

NEXT_STEP_PATTERNS = [
    r"\bi'?ll send\b",
    r"\bi will send\b",
    r"\bi can send\b",
    r"\bfollow up\b",
    r"\bschedule\b",
    r"\bbook\b.*\b(call|demo|meeting)\b",
    r"\bset up\b.*\b(call|demo|meeting)\b",
    r"\bnext step\b",
    r"\bafter our call\b",
    r"\bright after\b",
    r"\btoday\b",
    r"\btomorrow\b",
    r"\bnext week\b",
    r"\bemail you\b",
    r"\bsend you\b",
    r"\breach out\b",
]


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def split_turns(text: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        speaker, utterance = line.split(":", 1)
        turns.append((speaker.strip().lower(), utterance.strip()))
    return turns


def manager_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(utterance for speaker, utterance in turns if speaker == "manager")


def client_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(utterance for speaker, utterance in turns if speaker == "client")


def has_client_objection_before_manager_response(turns: list[tuple[str, str]]) -> bool:
    seen_objection = False
    for speaker, utterance in turns:
        if speaker == "client" and contains_any(utterance, OBJECTION_PATTERNS):
            seen_objection = True
        elif speaker == "manager" and seen_objection:
            if contains_any(utterance, OBJECTION_RESPONSE_PATTERNS) or contains_any(
                utterance, VALUE_PATTERNS
            ):
                return True
    return False


def early_presentation(turns: list[tuple[str, str]]) -> bool:
    discovered = False
    for speaker, utterance in turns:
        if speaker == "manager" and contains_any(utterance, QUESTION_PATTERNS):
            discovered = True
        if speaker == "client" and len(utterance.split()) > 6:
            if contains_any(utterance, OBJECTION_PATTERNS) or contains_any(
                utterance,
                [
                    r"\bneed\b",
                    r"\blooking for\b",
                    r"\bwant\b",
                    r"\bprefer\b",
                    r"\bimportant\b",
                ],
            ):
                discovered = True
        if speaker == "manager" and contains_any(utterance, VALUE_PATTERNS):
            if not discovered:
                return True
    return False


def choose_recommendation(labels: dict[str, int]) -> str:
    if labels["early_presentation_before_discovery"]:
        return "avoid_early_presentation"
    if not labels["needs_discovery"]:
        return "ask_needs"
    if not labels["objection_handling"]:
        return "handle_objection"
    if not labels["value_articulation"]:
        return "anchor_value"
    if not labels["next_step_fixed"]:
        return "fix_next_step"
    return "no_action"


def weak_label(text: str) -> dict[str, int | str]:
    turns = split_turns(text)
    m_text = manager_text(turns)
    c_text = client_text(turns)

    labels: dict[str, int] = {
        "needs_discovery": int(contains_any(m_text, QUESTION_PATTERNS)),
        "value_articulation": int(contains_any(m_text, VALUE_PATTERNS)),
        "objection_handling": int(has_client_objection_before_manager_response(turns)),
        "next_step_fixed": int(contains_any(m_text, NEXT_STEP_PATTERNS)),
        "early_presentation_before_discovery": int(early_presentation(turns)),
    }

    # If there is no customer objection in the window, do not recommend objection
    # handling merely because the binary label is 0.
    recommendation = choose_recommendation(labels)
    if recommendation == "handle_objection" and not contains_any(c_text, OBJECTION_PATTERNS):
        recommendation = "no_action" if labels["needs_discovery"] else "ask_needs"

    return {**labels, "recommendation_type": recommendation}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    for row in rows:
        labels = weak_label(row["text"])
        for key, value in labels.items():
            row[key] = str(value)
        note = row.get("notes", "").strip()
        row["notes"] = f"{note} weak_label=v1".strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

