from __future__ import annotations

import json
from typing import Any


REVIEW_PROMPT_VERSION = "definition_review_v2"
REVIEW_MAX_TOKENS = 3000

REVIEW_VERDICTS = ("pass", "minor_issues", "major_issues")
REVIEW_SCORE_FIELDS = (
    "accuracy",
    "explanations",
    "examples",
    "usage_notes",
    "chinese_quality",
)
REVIEW_ISSUE_KINDS = (
    "hallucination",
    "mistranslation",
    "unfaithful_to_source",
    "unnatural_example",
    "restated_usage_note",
    "priority_error",
    "language_mixing",
    "other",
)


def build_review_system_prompt() -> str:
    return """
You are a bilingual lexicographer auditing one machine-generated entry of an
English-Chinese learner's dictionary. You receive the curated source skeleton
(the Wiktionary glosses the generator was given) and the generated entry.
Judge the generated content strictly against the source and against learner
usefulness. Return exactly one JSON object and nothing else.

Calibration — read this before judging:
- The dictionary's explanation style is DELIBERATE synthesis: explanations are
  anchored to an entry-wide memory hook and paraphrase the source glosses in
  learner-friendly language instead of translating them word for word. Do not
  flag paraphrase, reorganization, or hook-anchored framing as unfaithful.
  Only flag unfaithful_to_source when a meaning is actually changed, invented,
  or entirely lost.
- Partial coverage of a multi-part gloss, thin-but-correct explanations of
  marginal senses, and disagreements of taste about priority markings are
  minor observations, never major.

The JSON object must contain:
- scores: object with integer scores from 1 (unusable) to 5 (excellent) for
  exactly these keys:
  - accuracy: are the explanations faithful to the source glosses, with no
    invented meanings and no dropped core meaning?
  - explanations: are they clear, concrete, and genuinely explanatory rather
    than mechanical translations of the gloss?
  - examples: are the English sentences natural and idiomatic, and are the
    Chinese translations faithful and natural?
  - usage_notes: do they give real usage substance (patterns, collocations,
    register, pitfalls) instead of restating the meaning?
  - chinese_quality: is the Chinese fluent, natural, and free of stray
    English words mixed into prose?
- priority_ok: boolean — are the core/common/rare markings sensible for a
  learner (core = the everyday heart of the word)?
- verdict: exactly one of "pass", "minor_issues", "major_issues".
  major_issues is reserved for content that would actively mislead a learner:
  a factual error, a wrong translation, a hallucinated meaning, or an example
  demonstrating incorrect language. Anything real but non-misleading is
  minor_issues. pass means you would ship it as is.
- issues: array (possibly empty) of objects, each with:
  - kind: one of "hallucination", "mistranslation", "unfaithful_to_source",
    "unnatural_example", "restated_usage_note", "priority_error",
    "language_mixing", "other"
  - location: short pointer such as a sense_id or field name
  - note: one concise sentence describing the problem

Judge only what is present; do not penalize the entry for information the
source does not contain. Be strict about factual faithfulness and translation
correctness, and calibrate scores so that 4 means minor polish possible and
5 means nothing to fix. Output valid JSON only.
""".strip()


def build_review_user_prompt(
    *,
    source_payload: dict[str, Any],
    generated_payload: dict[str, Any],
) -> str:
    body = {
        "source_skeleton": source_payload,
        "generated_entry": generated_payload,
    }
    return (
        "Entry under review (JSON):\n"
        + json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def validate_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Review payload must be a JSON object")
    if "�" in json.dumps(payload, ensure_ascii=False):
        raise ValueError("Review payload contains U+FFFD replacement characters")

    required = {"scores", "priority_ok", "verdict", "issues"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Review payload is missing required keys: {sorted(missing)}")

    scores = payload["scores"]
    if not isinstance(scores, dict):
        raise ValueError("Review scores must be an object")
    normalized_scores: dict[str, int] = {}
    for field in REVIEW_SCORE_FIELDS:
        if field not in scores:
            raise ValueError(f"Review scores missing {field}")
        value = scores[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"Review score {field} must be an integer between 1 and 5")
        normalized_scores[field] = value

    if not isinstance(payload["priority_ok"], bool):
        raise ValueError("Review priority_ok must be a boolean")

    verdict = str(payload["verdict"] or "").strip().lower()
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"Review verdict must be one of {list(REVIEW_VERDICTS)}")

    raw_issues = payload["issues"]
    if raw_issues is None:
        raw_issues = []
    if not isinstance(raw_issues, list):
        raise ValueError("Review issues must be an array")
    issues: list[dict[str, str]] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            raise ValueError("Each review issue must be an object")
        kind = str(item.get("kind") or "").strip()
        if kind not in REVIEW_ISSUE_KINDS:
            kind = "other"
        note = str(item.get("note") or "").strip()
        if not note:
            raise ValueError("Review issues need a non-empty note")
        issues.append(
            {
                "kind": kind,
                "location": str(item.get("location") or "").strip(),
                "note": note,
            }
        )

    return {
        "scores": normalized_scores,
        "priority_ok": payload["priority_ok"],
        "verdict": verdict,
        "issues": issues,
    }
