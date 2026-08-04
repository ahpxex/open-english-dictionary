from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from open_dictionary.config import RuntimeSettings
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.client import LiteLLMClient
from open_dictionary.llm.config import load_llm_settings
from open_dictionary.llm.prompt import build_generation_source_payload
from open_dictionary.pipeline import ProgressCallback, ThrottledProgressReporter, complete_run, fail_run, start_run
from open_dictionary.stages.llm_enrich.stage import _call_with_retries

from .judge import (
    REVIEW_MAX_TOKENS,
    REVIEW_PROMPT_VERSION,
    build_review_system_prompt,
    build_review_user_prompt,
    validate_review_payload,
)
from .sampling import SAMPLING_METHOD_VERSION, SampledEntry, allocate_stratified_sample


QUALITY_REVIEW_STAGE = "definitions.review"
PERSIST_COMMIT_INTERVAL = 25


@dataclass(frozen=True)
class QualityReviewResult:
    run_id: UUID
    sampled: int
    reviewed: int
    failed: int
    skipped_existing: int


def run_quality_review_stage(
    *,
    settings: RuntimeSettings,
    env_file: str = ".env",
    llm_table: str = "llm.entry_enrichments",
    review_table: str = "llm.quality_reviews",
    generation_prompt_version_like: str = "%",
    sample_size: int = 1000,
    seed: str = "review-v1",
    max_workers: int = 8,
    max_retries: int = 3,
    client: LiteLLMClient | None = None,
    parent_run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QualityReviewResult:
    llm_settings = load_llm_settings(env_file=env_file)
    judge_client = client or LiteLLMClient(llm_settings)

    population = _load_population(
        settings,
        llm_table=llm_table,
        generation_prompt_version_like=generation_prompt_version_like,
    )
    sampled = allocate_stratified_sample(
        population,
        sample_size=sample_size,
        seed=seed,
    )
    payloads = {str(item["entry_id"]): item for item in population}

    already_reviewed = _load_existing_review_keys(
        settings,
        review_table=review_table,
        seed=seed,
    )
    pending = [
        entry
        for entry in sampled
        if (entry.entry_id, payloads[entry.entry_id]["input_hash"]) not in already_reviewed
    ]

    with get_connection(settings) as conn:
        run_id = start_run(
            conn,
            stage=QUALITY_REVIEW_STAGE,
            config={
                "review_table": review_table,
                "definitions_table": llm_table,
                "generation_prompt_version_like": generation_prompt_version_like,
                "review_prompt_version": REVIEW_PROMPT_VERSION,
                "sampling_method": SAMPLING_METHOD_VERSION,
                "sample_size": sample_size,
                "sampled": len(sampled),
                "skipped_existing": len(sampled) - len(pending),
                "seed": seed,
                "judge_models": list(llm_settings.models),
                "max_workers": max_workers,
                "max_retries": max_retries,
            },
            parent_run_id=parent_run_id,
        )

    reviewed = 0
    failed = 0
    system_prompt = build_review_system_prompt()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _review_one_entry,
                    entry=entry,
                    payload_row=payloads[entry.entry_id],
                    judge_client=judge_client,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                ): entry
                for entry in pending
            }

            with get_connection(settings) as conn:
                reporter = ThrottledProgressReporter(progress_callback, stage=QUALITY_REVIEW_STAGE)
                pending_writes = 0
                for future in concurrent.futures.as_completed(future_map):
                    entry = future_map[future]
                    payload_row = payloads[entry.entry_id]
                    try:
                        record = future.result()
                    except Exception as exc:
                        failed += 1
                        _persist_review(
                            conn,
                            review_table=review_table,
                            run_id=run_id,
                            entry=entry,
                            seed=seed,
                            input_hash=payload_row["input_hash"],
                            judge_model=getattr(exc, "model", None) or llm_settings.models[0],
                            status="failed",
                            verdict=None,
                            review=None,
                            raw_response=None,
                            error=str(exc),
                            generation_metadata=getattr(exc, "generation_metadata", None) or {},
                        )
                    else:
                        reviewed += 1
                        _persist_review(
                            conn,
                            review_table=review_table,
                            run_id=run_id,
                            entry=entry,
                            seed=seed,
                            input_hash=payload_row["input_hash"],
                            judge_model=record["judge_model"],
                            status="succeeded",
                            verdict=record["review"]["verdict"],
                            review=record["review"],
                            raw_response=record["raw_response"],
                            error=None,
                            generation_metadata=record["generation_metadata"],
                        )
                    pending_writes += 1
                    if pending_writes >= PERSIST_COMMIT_INTERVAL:
                        conn.commit()
                        pending_writes = 0
                    reporter.report(
                        event="review_progress",
                        reviewed=reviewed,
                        failed=failed,
                        queued=len(pending),
                    )
                if pending_writes:
                    conn.commit()

                complete_run(
                    conn,
                    run_id=run_id,
                    stats={
                        "sampled": len(sampled),
                        "reviewed": reviewed,
                        "failed": failed,
                        "skipped_existing": len(sampled) - len(pending),
                        "review_prompt_version": REVIEW_PROMPT_VERSION,
                        "seed": seed,
                    },
                )

        return QualityReviewResult(
            run_id=run_id,
            sampled=len(sampled),
            reviewed=reviewed,
            failed=failed,
            skipped_existing=len(sampled) - len(pending),
        )
    except Exception as exc:
        with get_connection(settings) as conn:
            fail_run(conn, run_id=run_id, error=str(exc))
        raise


def _review_one_entry(
    *,
    entry: SampledEntry,
    payload_row: dict[str, Any],
    judge_client: LiteLLMClient,
    system_prompt: str,
    max_retries: int,
) -> dict[str, Any]:
    source_payload = build_generation_source_payload(payload_row["curated_payload"])
    user_prompt = build_review_user_prompt(
        source_payload=source_payload,
        generated_payload=payload_row["response_payload"],
    )
    call = _call_with_retries(
        scope="review",
        llm_client=judge_client,
        primary_system_prompt=system_prompt,
        compact_system_prompt=system_prompt,
        user_prompt=user_prompt,
        validate=validate_review_payload,
        max_retries=max_retries,
    )
    return {
        "review": call.validated,
        "judge_model": call.generation.model,
        "raw_response": call.generation.content,
        "generation_metadata": call.as_metadata(),
    }


def _load_population(
    settings: RuntimeSettings,
    *,
    llm_table: str,
    generation_prompt_version_like: str,
) -> list[dict[str, Any]]:
    from psycopg import sql

    llm_identifier = sql.Identifier(*[part for part in llm_table.split(".") if part])
    query = sql.SQL(
        """
        SELECT c.entry_id::text, c.word, c.entry_flags, c.payload,
               latest.response_payload, latest.input_hash,
               (SELECT count(*) FROM jsonb_array_elements(c.payload->'pos_groups') g,
                       jsonb_array_elements(g->'senses') s) AS sense_count
        FROM curated.entries c
        JOIN (
            SELECT DISTINCT ON (entry_id) entry_id, response_payload, input_hash
            FROM {}
            WHERE status = 'succeeded' AND prompt_version LIKE %s
            ORDER BY entry_id, created_at DESC
        ) latest ON latest.entry_id = c.entry_id
        """
    ).format(llm_identifier)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (generation_prompt_version_like,))
            return [
                {
                    "entry_id": row[0],
                    "word": row[1],
                    "entry_flags": list(row[2] or []),
                    "curated_payload": row[3],
                    "response_payload": row[4],
                    "input_hash": row[5],
                    "sense_count": row[6],
                }
                for row in cursor.fetchall()
            ]


def _load_existing_review_keys(
    settings: RuntimeSettings,
    *,
    review_table: str,
    seed: str,
) -> set[tuple[str, str]]:
    from psycopg import sql

    review_identifier = sql.Identifier(*[part for part in review_table.split(".") if part])
    query = sql.SQL(
        """
        SELECT entry_id::text, enrichment_input_hash
        FROM {}
        WHERE status = 'succeeded'
          AND review_prompt_version = %s
          AND sample_seed = %s
        """
    ).format(review_identifier)
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (REVIEW_PROMPT_VERSION, seed))
            return {(row[0], row[1]) for row in cursor.fetchall()}


def _persist_review(
    conn,
    *,
    review_table: str,
    run_id: UUID,
    entry: SampledEntry,
    seed: str,
    input_hash: str,
    judge_model: str,
    status: str,
    verdict: str | None,
    review: dict[str, Any] | None,
    raw_response: str | None,
    error: str | None,
    generation_metadata: dict[str, Any],
) -> None:
    from psycopg import sql

    review_identifier = sql.Identifier(*[part for part in review_table.split(".") if part])
    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                INSERT INTO {} (
                    run_id, entry_id, enrichment_input_hash, review_prompt_version,
                    judge_model, sample_seed, stratum, sampling_weight,
                    status, verdict, scores, issues, raw_response, error,
                    generation_metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            ).format(review_identifier),
            (
                run_id,
                entry.entry_id,
                input_hash,
                REVIEW_PROMPT_VERSION,
                judge_model,
                seed,
                entry.stratum,
                entry.sampling_weight,
                status,
                verdict,
                Jsonb(review["scores"]) if review else None,
                Jsonb(review["issues"]) if review else None,
                raw_response,
                error,
                Jsonb(generation_metadata),
            ),
        )

