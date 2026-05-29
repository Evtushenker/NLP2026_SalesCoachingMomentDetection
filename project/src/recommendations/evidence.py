"""Extract text evidence for sales-coaching predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
    r"\bconfusing\b",
    r"\bproblem",
    r"\bissue",
]

NEXT_STEP_NEED_PATTERNS = [
    r"\bnext step",
    r"\bwhat would be the next",
    r"\binterested in finding out more",
    r"\bi'?ll go with\b",
    r"\bthat sounds good\b",
    r"\bthat sounds pretty good\b",
]

MANAGER_ACTION_PATTERNS = [
    r"\bfeel free to reach out\b",
    r"\blet me know\b",
    r"\bwe can discuss\b",
    r"\bi can send\b",
    r"\bi'?ll send\b",
    r"\bfollow up\b",
]


@dataclass(frozen=True)
class Turn:
    speaker: str
    utterance: str
    line_text: str
    turn_index: int


@dataclass(frozen=True)
class Evidence:
    target: str
    evidence_text: str
    speaker: str
    turn_index: int
    reason: str
    matched_pattern: str
    method: str
    confidence_hint: str


def parse_turns(text: str) -> list[Turn]:
    """Parse one speaker-tagged window into turns with 1-based indices."""

    turns: list[Turn] = []
    for line in text.splitlines():
        raw_line = line.strip()
        if ":" not in raw_line:
            continue
        speaker, utterance = raw_line.split(":", 1)
        speaker = speaker.strip().lower()
        utterance = utterance.strip()
        if not speaker or not utterance:
            continue
        turns.append(
            Turn(
                speaker=speaker,
                utterance=utterance,
                line_text=f"{speaker}: {utterance}",
                turn_index=len(turns) + 1,
            )
        )
    return turns


def _first_matching_pattern(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return ""


def _fallback(target: str, reason: str) -> Evidence:
    return Evidence(
        target=target,
        evidence_text="",
        speaker="",
        turn_index=-1,
        reason=reason,
        matched_pattern="",
        method="fallback_no_match",
        confidence_hint="low",
    )


def _extract_open_concern(turns: list[Turn], target: str) -> Evidence | None:
    for turn_position in range(len(turns) - 1, -1, -1):
        turn = turns[turn_position]
        if turn.speaker != "client":
            continue
        matched_pattern = _first_matching_pattern(turn.utterance, CONCERN_PATTERNS)
        if not matched_pattern:
            continue
        has_later_manager = any(
            later_turn.speaker == "manager" for later_turn in turns[turn_position + 1 :]
        )
        if has_later_manager:
            continue
        return Evidence(
            target=target,
            evidence_text=turn.line_text,
            speaker=turn.speaker,
            turn_index=turn.turn_index,
            reason=(
                "Latest client concern has no later manager response in the visible window."
            ),
            matched_pattern=matched_pattern,
            method="latest_open_client_concern",
            confidence_hint="high",
        )
    return None


def _extract_actionable(turns: list[Turn], target: str) -> Evidence | None:
    open_concern = _extract_open_concern(turns, target)
    if open_concern is not None:
        return open_concern

    for turn in reversed(turns):
        matched_pattern = _first_matching_pattern(turn.utterance, NEXT_STEP_NEED_PATTERNS)
        if matched_pattern:
            return Evidence(
                target=target,
                evidence_text=turn.line_text,
                speaker=turn.speaker,
                turn_index=turn.turn_index,
                reason="Turn signals that a concrete next step may be needed.",
                matched_pattern=matched_pattern,
                method="next_step_need_pattern",
                confidence_hint="medium",
            )

    for turn in reversed(turns):
        if turn.speaker != "manager":
            continue
        matched_pattern = _first_matching_pattern(turn.utterance, MANAGER_ACTION_PATTERNS)
        if matched_pattern:
            return Evidence(
                target=target,
                evidence_text=turn.line_text,
                speaker=turn.speaker,
                turn_index=turn.turn_index,
                reason="Manager turn contains a soft follow-up phrase that may need review.",
                matched_pattern=matched_pattern,
                method="manager_action_pattern",
                confidence_hint="medium",
            )
    return None


def extract_evidence(text: str, target: str) -> Evidence:
    """Return the input turn that best explains a prediction target."""

    turns = parse_turns(text)
    if not turns:
        return _fallback(target, "No speaker-tagged turns were found.")

    if target == "unresolved_customer_concern":
        evidence = _extract_open_concern(turns, target)
        if evidence is not None:
            return evidence
        return _fallback(
            target,
            "No open client concern was found in the visible window.",
        )

    if target == "actionable_coaching_needed":
        evidence = _extract_actionable(turns, target)
        if evidence is not None:
            return evidence
        return _fallback(
            target,
            "No explicit evidence turn was found for actionable coaching.",
        )

    return _fallback(target, f"Unsupported evidence target: {target}.")


def evidence_is_copied_from_input(text: str, evidence: Evidence) -> bool:
    return evidence.evidence_text == "" or evidence.evidence_text in text
