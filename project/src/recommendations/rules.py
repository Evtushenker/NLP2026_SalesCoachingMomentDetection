"""Template-based micro-coaching recommendations."""

from __future__ import annotations

import re
from dataclasses import dataclass


LABEL_COLUMNS = [
    "needs_discovery",
    "value_articulation",
    "objection_handling",
    "next_step_fixed",
    "early_presentation_before_discovery",
]

OPEN_OBJECTION_PATTERNS = [
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
]


@dataclass(frozen=True)
class Recommendation:
    recommendation_type: str
    action: str
    rationale: str
    evidence_labels: list[str]
    confidence: float


def _label_value(labels: dict[str, int | float | str], key: str) -> int:
    value = labels.get(key, 0)
    if isinstance(value, str):
        return int(float(value)) if value.strip() else 0
    return int(value)


def _has_open_client_objection_at_end(text: str | None) -> bool:
    if not text:
        return False
    turns = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        speaker, utterance = line.split(":", 1)
        turns.append((speaker.strip().lower(), utterance.strip()))
    if not turns:
        return False

    last_client_index = None
    for index, (speaker, _) in enumerate(turns):
        if speaker == "client":
            last_client_index = index
    if last_client_index is None:
        return False

    last_client_text = turns[last_client_index][1]
    has_objection = any(
        re.search(pattern, last_client_text, flags=re.IGNORECASE)
        for pattern in OPEN_OBJECTION_PATTERNS
    )
    has_later_manager_response = any(
        speaker == "manager" for speaker, _ in turns[last_client_index + 1 :]
    )
    return has_objection and not has_later_manager_response


def recommend(
    labels: dict[str, int | float | str],
    probabilities: dict[str, float] | None = None,
    text: str | None = None,
) -> Recommendation:
    """Map predicted labels to one actionable coaching recommendation."""

    values = {label: _label_value(labels, label) for label in LABEL_COLUMNS}
    probabilities = probabilities or {}

    if _has_open_client_objection_at_end(text):
        return Recommendation(
            recommendation_type="handle_objection",
            action=(
                "Address the customer's latest concern before moving on: restate it, "
                "answer with a specific fact or policy, and check whether it is resolved."
            ),
            rationale=(
                "The window ends with a customer concern that has not yet received a "
                "manager response in the visible context."
            ),
            evidence_labels=["objection_handling"],
            confidence=float(1.0 - probabilities.get("objection_handling", 0.0)),
        )

    if values["early_presentation_before_discovery"]:
        return Recommendation(
            recommendation_type="avoid_early_presentation",
            action=(
                "Before presenting the solution, ask one or two discovery questions "
                "about the customer's goal, constraints, budget, or decision criteria."
            ),
            rationale=(
                "The fragment looks like the manager moved into presentation before "
                "the customer's need was clear."
            ),
            evidence_labels=["early_presentation_before_discovery"],
            confidence=float(probabilities.get("early_presentation_before_discovery", 1.0)),
        )

    if not values["needs_discovery"]:
        return Recommendation(
            recommendation_type="ask_needs",
            action=(
                "Ask a concrete discovery question before continuing: what the "
                "customer needs, what matters most, and what constraints exist."
            ),
            rationale=(
                "The fragment does not show enough needs discovery to make the next "
                "recommendation or value statement specific."
            ),
            evidence_labels=["needs_discovery"],
            confidence=float(1.0 - probabilities.get("needs_discovery", 0.0)),
        )

    if not values["value_articulation"]:
        return Recommendation(
            recommendation_type="anchor_value",
            action=(
                "Connect the offer to the customer's stated need and support it with "
                "a fact, guarantee, example, case, or measurable benefit."
            ),
            rationale=(
                "The conversation contains discovery, but the value is not clearly "
                "anchored to the customer's situation."
            ),
            evidence_labels=["value_articulation"],
            confidence=float(1.0 - probabilities.get("value_articulation", 0.0)),
        )

    if not values["objection_handling"]:
        return Recommendation(
            recommendation_type="handle_objection",
            action=(
                "If the customer expresses doubt, restate the concern, answer with a "
                "specific fact or policy, and check whether the concern is resolved."
            ),
            rationale=(
                "No complete objection-handling pattern is visible in this fragment."
            ),
            evidence_labels=["objection_handling"],
            confidence=float(1.0 - probabilities.get("objection_handling", 0.0)),
        )

    if not values["next_step_fixed"]:
        return Recommendation(
            recommendation_type="fix_next_step",
            action=(
                "Close with a specific next step: owner, action, channel, and time."
            ),
            rationale=(
                "The fragment does not clearly fix what happens next and when."
            ),
            evidence_labels=["next_step_fixed"],
            confidence=float(1.0 - probabilities.get("next_step_fixed", 0.0)),
        )

    return Recommendation(
        recommendation_type="no_action",
        action="No immediate coaching action is required for this fragment.",
        rationale="The key MVP elements are present in the visible context.",
        evidence_labels=[],
        confidence=1.0,
    )
