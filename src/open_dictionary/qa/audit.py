from __future__ import annotations

import json
import re
from typing import Any

from psycopg import sql

from open_dictionary.config import RuntimeSettings
from open_dictionary.db.connection import get_connection


AUDIT_HEURISTICS_VERSION = "definitions_audit_v1"

MIXED_LANGUAGE_PATTERN = re.compile(r"[一-鿿] ?(?:idea|thing|word|concept|root|sense)")
RESTATE_OPENERS = ("指", "表示", "意为", "即")
MIN_EXPLANATION_CHARS = 12
MAX_CORE_SENSES = 5


def audit_response_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run soft-quality heuristics over one validated response payload.

    Returns a list of finding dicts; an empty list means no findings. These
    are advisory signals, deliberately separate from the blocking contract
    validation that already ran at generation time.
    """
    findings: list[dict[str, Any]] = []
    meanings = [
        (group, meaning)
        for group in payload.get("pos_groups", [])
        for meaning in group.get("meanings", [])
    ]

    core_count = sum(1 for _, meaning in meanings if meaning.get("priority") == "core")
    if core_count > MAX_CORE_SENSES:
        findings.append({"check": "core_over_budget", "detail": core_count})

    for group, meaning in meanings:
        priority = meaning.get("priority")
        if priority in ("core", "common") and not meaning.get("examples"):
            findings.append(
                {
                    "check": "missing_example",
                    "detail": f"{group.get('pos_group_id')}/{meaning.get('sense_id')} ({priority})",
                }
            )
        explanation = meaning.get("learner_explanation") or ""
        if len(explanation) < MIN_EXPLANATION_CHARS:
            findings.append(
                {
                    "check": "short_explanation",
                    "detail": f"{group.get('pos_group_id')}/{meaning.get('sense_id')}",
                }
            )
        usage_note = meaning.get("usage_note")
        if usage_note and usage_note.startswith(RESTATE_OPENERS) and not _looks_pattern_first(usage_note):
            findings.append(
                {
                    "check": "restate_style_usage_note",
                    "detail": f"{group.get('pos_group_id')}/{meaning.get('sense_id')}: {usage_note[:60]}",
                }
            )

    hook = (payload.get("memory_hook") or "").strip()
    summary = (payload.get("headword_summary") or "").strip()
    if hook and hook == summary:
        findings.append({"check": "hook_equals_summary", "detail": hook[:60]})

    mixed = MIXED_LANGUAGE_PATTERN.search(json.dumps(payload, ensure_ascii=False))
    if mixed:
        findings.append({"check": "mixed_language", "detail": mixed.group(0)})

    return findings


def _looks_pattern_first(usage_note: str) -> bool:
    # Pattern-first notes routinely quote a template in the opening clause,
    # e.g. 表示进行中的动作时，用“be doing …”。Quoted latin content early in
    # the note is a strong signal the note is structural, not a re-explanation.
    head = usage_note[:40]
    return bool(re.search(r"[“\"'][^”\"']*[A-Za-z]", head))


def audit_definitions(
    settings: RuntimeSettings,
    *,
    llm_table: str = "llm.entry_enrichments",
    prompt_version_like: str = "%",
    bucket_size: int = 1000,
    sample_size: int = 200,
) -> dict[str, Any]:
    """Build an advisory quality report over generated definitions.

    Bucket statistics cover every row (grouped by enrichment_id ranges so a
    long run can be reviewed batch by batch); content heuristics run over a
    random sample of succeeded rows.
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be a positive integer")
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")

    llm_identifier = sql.Identifier(*[part for part in llm_table.split(".") if part])

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT
                        (enrichment_id / %s)::bigint AS bucket,
                        count(*) FILTER (WHERE status = 'succeeded') AS succeeded,
                        count(*) FILTER (WHERE status = 'failed') AS failed,
                        sum(retries) AS retries,
                        count(*) FILTER (
                            WHERE generation_metadata->>'used_compact_retry_prompt' = 'true'
                        ) AS compact_used,
                        count(*) FILTER (
                            WHERE generation_metadata->>'sharded' = 'true'
                        ) AS sharded,
                        avg((generation_metadata->'usage'->>'prompt_tokens')::int) AS avg_prompt_tokens,
                        avg((generation_metadata->'usage'->>'completion_tokens')::int) AS avg_completion_tokens
                    FROM {}
                    WHERE prompt_version LIKE %s
                    GROUP BY bucket
                    ORDER BY bucket
                    """
                ).format(llm_identifier),
                (bucket_size, prompt_version_like),
            )
            buckets = [
                {
                    "bucket": int(row[0]),
                    "succeeded": row[1],
                    "failed": row[2],
                    "retries": int(row[3] or 0),
                    "compact_used": row[4],
                    "sharded": row[5],
                    "avg_prompt_tokens": round(float(row[6]), 1) if row[6] is not None else None,
                    "avg_completion_tokens": round(float(row[7]), 1) if row[7] is not None else None,
                }
                for row in cursor.fetchall()
            ]

            cursor.execute(
                sql.SQL(
                    """
                    SELECT entry_id::text, response_payload
                    FROM {}
                    WHERE status = 'succeeded'
                      AND prompt_version LIKE %s
                    ORDER BY random()
                    LIMIT %s
                    """
                ).format(llm_identifier),
                (prompt_version_like, sample_size),
            )
            sample_rows = cursor.fetchall()

            cursor.execute(
                sql.SQL(
                    """
                    SELECT entry_id::text, left(error, 200)
                    FROM {}
                    WHERE status = 'failed'
                      AND prompt_version LIKE %s
                    ORDER BY enrichment_id DESC
                    LIMIT 10
                    """
                ).format(llm_identifier),
                (prompt_version_like,),
            )
            recent_failures = [
                {"entry_id": row[0], "error": row[1]} for row in cursor.fetchall()
            ]

    finding_counts: dict[str, int] = {}
    flagged_entries: list[dict[str, Any]] = []
    for entry_id, payload in sample_rows:
        findings = audit_response_payload(payload)
        if findings:
            flagged_entries.append({"entry_id": entry_id, "findings": findings})
            for finding in findings:
                finding_counts[finding["check"]] = finding_counts.get(finding["check"], 0) + 1

    return {
        "heuristics_version": AUDIT_HEURISTICS_VERSION,
        "prompt_version_like": prompt_version_like,
        "bucket_size": bucket_size,
        "buckets": buckets,
        "totals": {
            "succeeded": sum(bucket["succeeded"] for bucket in buckets),
            "failed": sum(bucket["failed"] for bucket in buckets),
            "retries": sum(bucket["retries"] for bucket in buckets),
            "compact_used": sum(bucket["compact_used"] for bucket in buckets),
        },
        "sample": {
            "size": len(sample_rows),
            "flagged": len(flagged_entries),
            "finding_counts": finding_counts,
            "flagged_entries": flagged_entries[:20],
        },
        "recent_failures": recent_failures,
    }
