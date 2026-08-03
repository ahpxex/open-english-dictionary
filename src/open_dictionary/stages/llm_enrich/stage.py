from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from open_dictionary.config import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE, LanguageSpec, normalize_language_spec
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.client import LLMClientError, LLMGenerationResult, LiteLLMClient
from open_dictionary.llm.config import load_llm_settings
from open_dictionary.llm.prompt import (
    CHUNK_SENSE_BUDGET,
    COMPACT_RETRY_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    PROMPT_VERSION,
    SHARD_SENSE_THRESHOLD,
    PromptBundle,
    build_chunk_user_prompt,
    build_enrichment_request_payload,
    build_overview_digest,
    build_overview_user_prompt,
    build_prompt_bundle,
    build_user_prompt,
    compute_request_hash,
)
from open_dictionary.pipeline import ProgressCallback, ThrottledProgressReporter, complete_run, emit_progress, fail_run, start_run, update_run_config

from .schema import (
    build_expected_generation_targets,
    validate_enrichment_chunk,
    validate_enrichment_payload,
    validate_overview_payload,
)


LLM_ENRICH_STAGE = "definitions.generate"
PERSIST_COMMIT_INTERVAL = 25
RETRY_TEMPERATURE = 0.3
# When every deployment in the provider pool is cooling down, litellm refuses
# the call for ~cooldown_time seconds; short content-level retry sleeps would
# land inside that same window and burn the whole retry budget.
POOL_COOLDOWN_BACKOFF_SECONDS = 35.0


@dataclass(frozen=True)
class LLMEnrichResult:
    run_id: UUID
    processed: int
    succeeded: int
    failed: int


class EnrichmentError(RuntimeError):
    """Terminal failure of one entry after all content-level retries.

    Carries the model attribution and operational metadata of the failed
    generation so the stage can persist a truthful failure row.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None,
        generation_metadata: dict[str, Any],
    ):
        super().__init__(message)
        self.model = model
        self.generation_metadata = generation_metadata


def run_llm_enrich_stage(
    *,
    settings: RuntimeSettings,
    env_file: str = ".env",
    source_table: str = "curated.entries",
    target_table: str = "llm.entry_enrichments",
    prompt_version: str = PROMPT_VERSION,
    definition_language: LanguageSpec | dict[str, Any] = DEFAULT_DEFINITION_LANGUAGE,
    limit_entries: int | None = None,
    max_workers: int = 4,
    max_retries: int = 3,
    recompute_existing: bool = False,
    client: LiteLLMClient | None = None,
    parent_run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LLMEnrichResult:
    llm_settings = load_llm_settings(env_file=env_file)
    llm_client = client or LiteLLMClient(llm_settings)
    language = normalize_language_spec(definition_language)
    prompt_bundle = build_prompt_bundle(
        prompt_version=prompt_version,
        definition_language=language,
    )

    with get_connection(settings) as conn:
        run_id = start_run(
            conn,
            stage=LLM_ENRICH_STAGE,
            config={
                "source_table": source_table,
                "target_table": target_table,
                "prompt_template_version": prompt_bundle.template_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": language.as_dict(),
                "models": list(llm_settings.models),
                "providers": [
                    {
                        "model": provider.model,
                        "api_base": provider.api_base,
                        "rpm": provider.rpm,
                    }
                    for provider in llm_settings.providers
                ],
                "limit_entries": limit_entries,
                "max_workers": max_workers,
                "max_retries": max_retries,
                "recompute_existing": recompute_existing,
            },
            parent_run_id=parent_run_id,
        )
        ensure_prompt_version(conn, prompt_bundle=prompt_bundle)

    processed = 0
    succeeded = 0
    failed = 0

    try:
        items = list(
            iter_curated_entries(
                settings,
                source_table=source_table,
                target_table=target_table,
                prompt_bundle=prompt_bundle,
                models=llm_settings.models,
                recompute_existing=recompute_existing,
                limit_entries=limit_entries,
            )
        )
        source_run_ids = sorted(
            {
                str(item["source_run_id"])
                for item in items
                if item.get("source_run_id") is not None
            }
        )
        with get_connection(settings) as conn:
            update_run_config(
                conn,
                run_id=run_id,
                config_updates={
                    "queued_entries": len(items),
                    "source_run_ids": source_run_ids,
                },
            )
        emit_progress(
            progress_callback,
            stage=LLM_ENRICH_STAGE,
            event="generate_start",
            queued_entries=len(items),
            max_workers=max_workers,
            max_retries=max_retries,
            prompt_version=prompt_bundle.resolved_prompt_version,
            prompt_template_version=prompt_bundle.template_version,
            definition_language_code=language.code,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    enrich_one_entry,
                    entry=item,
                    llm_client=llm_client,
                    prompt_bundle=prompt_bundle,
                    max_retries=max_retries,
                ): item
                for item in items
            }

            with get_connection(settings) as conn:
                reporter = ThrottledProgressReporter(progress_callback, stage=LLM_ENRICH_STAGE)
                pending_writes = 0
                for future in concurrent.futures.as_completed(future_map):
                    processed += 1
                    try:
                        record = future.result()
                    except Exception as exc:
                        failed += 1
                        if isinstance(exc, EnrichmentError):
                            failed_model = exc.model or llm_settings.models[0]
                            failure_metadata = exc.generation_metadata
                        else:
                            failed_model = llm_settings.models[0]
                            failure_metadata = {"error_type": type(exc).__name__}
                        persist_enrichment_failure(
                            conn,
                            target_table=target_table,
                            run_id=run_id,
                            entry_id=future_map[future]["entry_id"],
                            model=failed_model,
                            prompt_version=prompt_bundle.resolved_prompt_version,
                            definition_language=language,
                            input_hash=future_map[future]["input_hash"],
                            request_payload=future_map[future]["request_payload"],
                            retries=max_retries,
                            error=str(exc),
                            generation_metadata=failure_metadata,
                        )
                    else:
                        succeeded += 1
                        persist_enrichment_success(
                            conn,
                            target_table=target_table,
                            run_id=run_id,
                            record=record,
                        )
                    pending_writes += 1
                    if pending_writes >= PERSIST_COMMIT_INTERVAL:
                        conn.commit()
                        pending_writes = 0
                    reporter.report(
                        event="generate_progress",
                        processed=processed,
                        queued_entries=len(items),
                        succeeded=succeeded,
                        failed=failed,
                    )

                if pending_writes:
                    conn.commit()

                complete_run(
                    conn,
                    run_id=run_id,
                    stats={
                        "processed": processed,
                        "succeeded": succeeded,
                        "failed": failed,
                        "models": list(llm_settings.models),
                        "prompt_template_version": prompt_bundle.template_version,
                        "prompt_version": prompt_bundle.resolved_prompt_version,
                        "definition_language": language.as_dict(),
                        "source_run_ids": source_run_ids,
                    },
                )
                emit_progress(
                    progress_callback,
                    stage=LLM_ENRICH_STAGE,
                    event="generate_complete",
                    processed=processed,
                    queued_entries=len(items),
                    succeeded=succeeded,
                    failed=failed,
                    prompt_version=prompt_bundle.resolved_prompt_version,
                    prompt_template_version=prompt_bundle.template_version,
                    definition_language_code=language.code,
                )

        return LLMEnrichResult(run_id=run_id, processed=processed, succeeded=succeeded, failed=failed)
    except Exception as exc:
        with get_connection(settings) as conn:
            fail_run(conn, run_id=run_id, error=str(exc))
        raise


def ensure_prompt_version(conn, *, prompt_bundle: PromptBundle) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO llm.prompt_versions (
                prompt_version,
                prompt_text,
                output_contract,
                definition_language_code,
                definition_language_name,
                prompt_bundle
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (prompt_version) DO NOTHING
            """,
            (
                prompt_bundle.resolved_prompt_version,
                prompt_bundle.system_prompt,
                Jsonb(prompt_bundle.output_contract),
                prompt_bundle.definition_language.code,
                prompt_bundle.definition_language.name,
                Jsonb(prompt_bundle.as_metadata()),
            ),
        )
    conn.commit()


def iter_curated_entries(
    settings: RuntimeSettings,
    *,
    source_table: str,
    target_table: str,
    prompt_bundle: PromptBundle,
    models: Sequence[str],
    recompute_existing: bool,
    limit_entries: int | None,
):
    source_identifier = identifier_from_dotted(source_table)
    existing_success_hashes = (
        load_existing_success_hashes(
            settings,
            target_table=target_table,
            prompt_bundle=prompt_bundle,
            models=models,
        )
        if not recompute_existing
        else {}
    )
    query = sql.SQL(
        """
        SELECT e.run_id, e.entry_id, e.payload
        FROM {} AS e
        """
    ).format(source_identifier)
    query += sql.SQL(" ORDER BY e.lang_code, e.normalized_word")
    params: list[Any] = []

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            yielded = 0
            for source_run_id, entry_id, payload in cursor.fetchall():
                request_payload = build_enrichment_request_payload(
                    payload,
                    prompt_bundle=prompt_bundle,
                )
                input_hash = compute_input_hash(request_payload)
                if not recompute_existing and input_hash in existing_success_hashes.get(str(entry_id), set()):
                    continue
                yield {
                    "source_run_id": source_run_id,
                    "entry_id": entry_id,
                    "payload": payload,
                    "request_payload": request_payload,
                    "input_hash": input_hash,
                }
                yielded += 1
                if limit_entries is not None and yielded >= limit_entries:
                    break


def count_pending_entries(
    settings: RuntimeSettings,
    *,
    source_table: str,
    target_table: str,
    prompt_bundle: PromptBundle,
    models: Sequence[str],
    recompute_existing: bool,
) -> int:
    return sum(
        1
        for _ in iter_curated_entries(
            settings,
            source_table=source_table,
            target_table=target_table,
            prompt_bundle=prompt_bundle,
            models=models,
            recompute_existing=recompute_existing,
            limit_entries=None,
        )
    )


@dataclass(frozen=True)
class GenerationCall:
    scope: str
    validated: Any
    generation: LLMGenerationResult
    attempts: int
    used_compact_retry_prompt: bool
    temperature: float

    def as_metadata(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "model": self.generation.model,
            "api_base": self.generation.api_base,
            "attempts": self.attempts,
            "used_compact_retry_prompt": self.used_compact_retry_prompt,
            "temperature": self.temperature,
            "usage": {
                "prompt_tokens": self.generation.prompt_tokens,
                "completion_tokens": self.generation.completion_tokens,
                "total_tokens": self.generation.total_tokens,
            },
        }


def enrich_one_entry(
    *,
    entry: dict[str, Any],
    llm_client: LiteLLMClient,
    prompt_bundle: PromptBundle,
    max_retries: int,
) -> dict[str, Any]:
    request_payload = entry["request_payload"]
    generation_source_payload = request_payload["entry"]
    expected_pos_targets = build_expected_generation_targets(generation_source_payload)
    total_senses = sum(len(target["sense_ids"]) for target in expected_pos_targets)

    if total_senses > SHARD_SENSE_THRESHOLD:
        validated, calls, core_senses = _generate_sharded(
            llm_client=llm_client,
            prompt_bundle=prompt_bundle,
            source_payload=generation_source_payload,
            expected_pos_targets=expected_pos_targets,
            max_retries=max_retries,
        )
        raw_response = json.dumps(validated, ensure_ascii=False, sort_keys=True)
        generation_metadata = {
            "sharded": True,
            "core_senses": core_senses,
            "chunk_count": len(calls) - 1,
            "calls": [call.as_metadata() for call in calls],
            "usage": {
                "prompt_tokens": _sum_usage(calls, "prompt_tokens"),
                "completion_tokens": _sum_usage(calls, "completion_tokens"),
                "total_tokens": _sum_usage(calls, "total_tokens"),
            },
            "attempts": sum(call.attempts for call in calls),
        }
        model = _majority_model(calls)
        retries = sum(call.attempts - 1 for call in calls)
    else:
        call = _call_with_retries(
            scope="entry",
            llm_client=llm_client,
            primary_system_prompt=prompt_bundle.system_prompt,
            compact_system_prompt=prompt_bundle.compact_retry_system_prompt,
            user_prompt=build_user_prompt(generation_source_payload),
            validate=lambda payload: validate_enrichment_payload(
                payload,
                expected_pos_targets=expected_pos_targets,
            ),
            max_retries=max_retries,
        )
        validated = call.validated
        raw_response = call.generation.content
        generation_metadata = {
            "api_base": call.generation.api_base,
            "usage": call.as_metadata()["usage"],
            "attempts": call.attempts,
            "used_compact_retry_prompt": call.used_compact_retry_prompt,
            "temperature": call.temperature,
        }
        model = call.generation.model
        retries = call.attempts - 1

    return {
        "entry_id": entry["entry_id"],
        "model": model,
        "prompt_version": prompt_bundle.resolved_prompt_version,
        "prompt_template_version": prompt_bundle.template_version,
        "definition_language": prompt_bundle.definition_language,
        "input_hash": entry["input_hash"],
        "request_payload": request_payload,
        "response_payload": validated,
        "raw_response": raw_response,
        "retries": retries,
        "generation_metadata": generation_metadata,
    }


def _call_with_retries(
    *,
    scope: str,
    llm_client: LiteLLMClient,
    primary_system_prompt: str,
    compact_system_prompt: str,
    user_prompt: str,
    validate,
    max_retries: int,
) -> GenerationCall:
    last_error: Exception | None = None
    last_generation: LLMGenerationResult | None = None

    for attempt in range(1, max_retries + 1):
        # The compact prompt is a last resort: retry at full quality first so
        # a transient failure does not permanently degrade the entry. Middle
        # retries add sampling temperature because at temperature 0 an
        # identical request fails identically — the retry must explore.
        use_compact_retry_prompt = max_retries > 1 and attempt == max_retries
        system_prompt = compact_system_prompt if use_compact_retry_prompt else primary_system_prompt
        max_tokens = COMPACT_RETRY_MAX_TOKENS if use_compact_retry_prompt else DEFAULT_MAX_TOKENS
        temperature = 0.0 if attempt == 1 or use_compact_retry_prompt else RETRY_TEMPERATURE
        last_generation = None
        try:
            generation = llm_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_generation = generation
            validated = validate(json.loads(generation.content))
            return GenerationCall(
                scope=scope,
                validated=validated,
                generation=generation,
                attempts=attempt,
                used_compact_retry_prompt=use_compact_retry_prompt,
                temperature=temperature,
            )
        except (json.JSONDecodeError, ValueError, LLMClientError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(_retry_delay(exc, attempt))

    assert last_error is not None
    if isinstance(last_error, LLMClientError) and last_error.model is not None:
        failed_model = last_error.model
    elif last_generation is not None:
        failed_model = last_generation.model
    else:
        failed_model = None
    raise EnrichmentError(
        f"{scope}: {last_error}",
        model=failed_model,
        generation_metadata={
            "scope": scope,
            "attempts": max_retries,
            "error_type": type(last_error).__name__,
            "last_model": failed_model,
            "last_api_base": last_generation.api_base if last_generation is not None else None,
        },
    ) from last_error


def _retry_delay(error: Exception, attempt: int) -> float:
    if isinstance(error, LLMClientError) and "No deployments available" in str(error):
        return POOL_COOLDOWN_BACKOFF_SECONDS
    return min(0.5 * attempt, 2.0)


def _generate_sharded(
    *,
    llm_client: LiteLLMClient,
    prompt_bundle: PromptBundle,
    source_payload: dict[str, Any],
    expected_pos_targets: list[dict[str, Any]],
    max_retries: int,
) -> tuple[dict[str, Any], list[GenerationCall], list[dict[str, str]]]:
    all_sense_keys = {
        (target["pos_group_id"], sense_id)
        for target in expected_pos_targets
        for sense_id in target["sense_ids"]
    }
    overview_call = _call_with_retries(
        scope="overview",
        llm_client=llm_client,
        primary_system_prompt=prompt_bundle.overview_system_prompt,
        compact_system_prompt=prompt_bundle.overview_system_prompt,
        user_prompt=build_overview_user_prompt(build_overview_digest(source_payload)),
        validate=lambda payload: validate_overview_payload(
            payload,
            valid_sense_keys=all_sense_keys,
        ),
        max_retries=max_retries,
    )
    overview_fields = overview_call.validated
    core_senses = overview_fields["core_senses"]

    calls: list[GenerationCall] = [overview_call]
    chunk_group_lists: list[list[dict[str, Any]]] = []
    chunks = plan_generation_chunks(source_payload, budget=CHUNK_SENSE_BUDGET)
    total_senses = sum(
        len(group.get("meanings") or [])
        for group in source_payload.get("pos_groups", [])
    )
    for chunk_index, chunk_groups in enumerate(chunks, start=1):
        chunk_payload = {
            "entry_context": {
                "headword": source_payload.get("headword"),
                "headword_language": source_payload.get("headword_language"),
                "definition_language": source_payload.get("definition_language"),
                "headword_summary": overview_fields["headword_summary"],
                "memory_hook": overview_fields["memory_hook"],
                "core_senses": core_senses,
                "part": {"index": chunk_index, "of": len(chunks)},
                "total_senses_in_entry": total_senses,
            },
            "pos_groups": chunk_groups,
        }
        chunk_targets = build_expected_generation_targets({"pos_groups": chunk_groups})
        chunk_call = _call_with_retries(
            scope=f"chunk_{chunk_index}",
            llm_client=llm_client,
            primary_system_prompt=prompt_bundle.chunk_system_prompt,
            compact_system_prompt=prompt_bundle.compact_chunk_system_prompt,
            user_prompt=build_chunk_user_prompt(chunk_payload),
            validate=lambda payload, targets=chunk_targets: validate_enrichment_chunk(
                payload,
                expected_pos_targets=targets,
            ),
            max_retries=max_retries,
        )
        calls.append(chunk_call)
        chunk_group_lists.append(chunk_call.validated)

    assembled = assemble_sharded_payload(
        overview_fields=overview_fields,
        chunk_group_lists=chunk_group_lists,
        source_payload=source_payload,
        core_senses=core_senses,
    )
    validated = validate_enrichment_payload(
        assembled,
        expected_pos_targets=expected_pos_targets,
    )
    return validated, calls, core_senses


def plan_generation_chunks(
    source_payload: dict[str, Any],
    *,
    budget: int,
) -> list[list[dict[str, Any]]]:
    """Pack pos groups into chunks of at most `budget` senses.

    Whole groups are kept together whenever they fit the budget; only groups
    larger than the budget itself are split into sense slices, each in its own
    chunk. Skeleton order is preserved throughout.
    """
    if budget <= 0:
        raise ValueError("budget must be a positive integer")

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_count = 0

    def close_current() -> None:
        nonlocal current, current_count
        if current:
            chunks.append(current)
            current = []
            current_count = 0

    for group in source_payload.get("pos_groups", []):
        meanings = group.get("meanings") or []
        if len(meanings) <= budget:
            if current and current_count + len(meanings) > budget:
                close_current()
            current.append(dict(group))
            current_count += len(meanings)
        else:
            close_current()
            for start in range(0, len(meanings), budget):
                slice_group = dict(group)
                slice_group["meanings"] = meanings[start : start + budget]
                chunks.append([slice_group])
    close_current()
    return chunks


def assemble_sharded_payload(
    *,
    overview_fields: dict[str, Any],
    chunk_group_lists: list[list[dict[str, Any]]],
    source_payload: dict[str, Any],
    core_senses: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge validated chunk outputs back into one full payload.

    When a group was split across chunks, the slice containing its first
    senses provides the group-level summary and usage note, and the meanings
    are concatenated in skeleton order. The overview call nominates the
    entry-wide core senses with full visibility, so its decision is enforced
    here: nominated senses become core, and no other sense may stay core.
    """
    groups_by_id: dict[str, dict[str, Any]] = {}
    for chunk_groups in chunk_group_lists:
        for group in chunk_groups:
            group_id = group["pos_group_id"]
            existing = groups_by_id.get(group_id)
            if existing is None:
                merged = dict(group)
                merged["meanings"] = list(group["meanings"])
                groups_by_id[group_id] = merged
            else:
                existing["meanings"].extend(group["meanings"])

    ordered_groups = []
    for skeleton_group in source_payload.get("pos_groups", []):
        group_id = skeleton_group.get("pos_group_id")
        if group_id not in groups_by_id:
            raise ValueError(f"Sharded generation produced no output for pos_group_id {group_id}")
        ordered_groups.append(groups_by_id[group_id])

    if core_senses is not None:
        nominated = {(item["pos_group_id"], item["sense_id"]) for item in core_senses}
        for group in ordered_groups:
            group_id = group["pos_group_id"]
            for meaning in group["meanings"]:
                if (group_id, meaning["sense_id"]) in nominated:
                    meaning["priority"] = "core"
                elif meaning["priority"] == "core":
                    meaning["priority"] = "common"

    return {
        "headword_summary": overview_fields["headword_summary"],
        "memory_hook": overview_fields["memory_hook"],
        "study_notes": overview_fields["study_notes"],
        "etymology_note": overview_fields["etymology_note"],
        "pos_groups": ordered_groups,
    }


def _sum_usage(calls: list[GenerationCall], field: str) -> int | None:
    values = [getattr(call.generation, field) for call in calls]
    if any(value is None for value in values):
        return None
    return sum(values)


def _majority_model(calls: list[GenerationCall]) -> str:
    counts: dict[str, int] = {}
    for call in calls:
        counts[call.generation.model] = counts.get(call.generation.model, 0) + 1
    return max(counts, key=lambda model: (counts[model], -list(counts).index(model)))


def persist_enrichment_success(conn, *, target_table: str, run_id: UUID, record: dict[str, Any]) -> None:
    target_identifier = identifier_from_dotted(target_table)
    definition_language = normalize_language_spec(record["definition_language"])
    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                INSERT INTO {} (
                    run_id,
                    entry_id,
                    model,
                    prompt_version,
                    definition_language_code,
                    definition_language_name,
                    input_hash,
                    status,
                    request_payload,
                    response_payload,
                    raw_response,
                    retries,
                    generation_metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'succeeded', %s, %s, %s, %s, %s)
                """
            ).format(target_identifier),
            (
                run_id,
                record["entry_id"],
                record["model"],
                record["prompt_version"],
                definition_language.code,
                definition_language.name,
                record["input_hash"],
                Jsonb(record["request_payload"]),
                Jsonb(record["response_payload"]),
                record["raw_response"],
                record["retries"],
                Jsonb(record.get("generation_metadata") or {}),
            ),
        )


def persist_enrichment_failure(
    conn,
    *,
    target_table: str,
    run_id: UUID,
    entry_id: str,
    model: str,
    prompt_version: str,
    definition_language: LanguageSpec | dict[str, Any],
    input_hash: str,
    request_payload: dict[str, Any],
    retries: int,
    error: str,
    generation_metadata: dict[str, Any] | None = None,
) -> None:
    target_identifier = identifier_from_dotted(target_table)
    language = normalize_language_spec(definition_language)
    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                INSERT INTO {} (
                    run_id,
                    entry_id,
                    model,
                    prompt_version,
                    definition_language_code,
                    definition_language_name,
                    input_hash,
                    status,
                    request_payload,
                    error,
                    retries,
                    generation_metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'failed', %s, %s, %s, %s)
                """
            ).format(target_identifier),
            (
                run_id,
                entry_id,
                model,
                prompt_version,
                language.code,
                language.name,
                input_hash,
                Jsonb(request_payload),
                error,
                retries,
                Jsonb(generation_metadata or {}),
            ),
        )


def compute_input_hash(payload: dict[str, Any]) -> str:
    return compute_request_hash(payload)


def load_existing_success_hashes(
    settings: RuntimeSettings,
    *,
    target_table: str,
    prompt_bundle: PromptBundle,
    models: Sequence[str],
) -> dict[str, set[str]]:
    target_identifier = identifier_from_dotted(target_table)
    query = sql.SQL(
        """
        SELECT entry_id::text, input_hash
        FROM {}
        WHERE status = 'succeeded'
          AND model = ANY(%s)
          AND prompt_version = %s
          AND definition_language_code = %s
        """
    ).format(target_identifier)
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    list(models),
                    prompt_bundle.resolved_prompt_version,
                    prompt_bundle.definition_language.code,
                ),
            )
            rows = cursor.fetchall()

    result: dict[str, set[str]] = {}
    for entry_id, input_hash in rows:
        result.setdefault(str(entry_id), set()).add(str(input_hash))
    return result


def identifier_from_dotted(qualified_name: str) -> sql.Identifier:
    parts = [segment.strip() for segment in qualified_name.split(".") if segment.strip()]
    if not parts:
        raise ValueError("Identifier name cannot be empty")
    return sql.Identifier(*parts)
