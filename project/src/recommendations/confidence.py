"""Confidence and fallback policy for coaching recommendations."""

from __future__ import annotations

from dataclasses import dataclass


HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.50


@dataclass(frozen=True)
class ConfidenceDecision:
    score: float
    score_type: str
    confidence_level: str
    fallback_used: bool
    review_required: bool
    decision_reason: str


def assign_confidence(
    score: float,
    model: str,
    prediction: int,
    target: str,
) -> ConfidenceDecision:
    """Map a model score to high/medium/low confidence behavior."""

    prediction = int(prediction)
    score = float(score)

    if model == "open_concern_rule":
        if prediction == 1:
            return ConfidenceDecision(
                score=score,
                score_type="deterministic_rule_score",
                confidence_level="high",
                fallback_used=False,
                review_required=False,
                decision_reason=(
                    "The structural open-concern rule fired; deterministic rule-positive "
                    "examples are treated as high confidence."
                ),
            )
        return ConfidenceDecision(
            score=score,
            score_type="deterministic_rule_score",
            confidence_level="low",
            fallback_used=True,
            review_required=True,
            decision_reason=(
                "The structural open-concern rule did not fire, so no specific coaching "
                "recommendation is shown."
            ),
        )

    if model == "tfidf_logreg":
        score_type = "logistic_regression_probability_estimate_uncalibrated"
        if prediction == 0:
            return ConfidenceDecision(
                score=score,
                score_type=score_type,
                confidence_level="low",
                fallback_used=True,
                review_required=True,
                decision_reason=(
                    "The TF-IDF logistic regression score is below the positive decision "
                    "threshold; use manual review instead of specific advice."
                ),
            )
        if score >= HIGH_THRESHOLD:
            return ConfidenceDecision(
                score=score,
                score_type=score_type,
                confidence_level="high",
                fallback_used=False,
                review_required=False,
                decision_reason=(
                    f"The positive TF-IDF logistic regression score is at least {HIGH_THRESHOLD:.2f}."
                ),
            )
        if score >= MEDIUM_THRESHOLD:
            return ConfidenceDecision(
                score=score,
                score_type=score_type,
                confidence_level="medium",
                fallback_used=False,
                review_required=True,
                decision_reason=(
                    f"The positive TF-IDF logistic regression score is between "
                    f"{MEDIUM_THRESHOLD:.2f} and {HIGH_THRESHOLD:.2f}; show advice but "
                    "keep it marked for review."
                ),
            )
        return ConfidenceDecision(
            score=score,
            score_type=score_type,
            confidence_level="low",
            fallback_used=True,
            review_required=True,
            decision_reason=(
                f"The TF-IDF logistic regression score is below {MEDIUM_THRESHOLD:.2f}; "
                "avoid specific advice."
            ),
        )

    if model == "tfidf_linear_svm":
        abs_score = abs(score)
        if prediction == 1 and abs_score >= 1.0:
            return ConfidenceDecision(
                score=score,
                score_type="raw_linear_svm_decision_score",
                confidence_level="high",
                fallback_used=False,
                review_required=False,
                decision_reason=(
                    "The raw Linear SVM margin is large; it is not a calibrated probability."
                ),
            )
        if prediction == 1:
            return ConfidenceDecision(
                score=score,
                score_type="raw_linear_svm_decision_score",
                confidence_level="medium",
                fallback_used=False,
                review_required=True,
                decision_reason=(
                    "The raw Linear SVM margin is positive but small; it is not a calibrated "
                    "probability."
                ),
            )

    return ConfidenceDecision(
        score=score,
        score_type="unknown_score",
        confidence_level="low",
        fallback_used=True,
        review_required=True,
        decision_reason=(
            f"No confidence policy is defined for target={target} model={model}; "
            "manual review is required."
        ),
    )


def thresholds_summary() -> dict[str, float]:
    return {
        "high_threshold": HIGH_THRESHOLD,
        "medium_threshold": MEDIUM_THRESHOLD,
    }
