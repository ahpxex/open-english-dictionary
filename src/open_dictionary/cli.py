"""Command-line entry point for the rewrite pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg
from psycopg import sql

from .config import load_settings
from .contracts import DEFAULT_DEFINITION_LANGUAGE, normalize_language_spec
from .db.bootstrap import LATEST_FOUNDATION_VERSION, apply_foundation
from .db.connection import get_connection
from .llm.config import load_llm_settings
from .llm.prompt import PROMPT_VERSION, build_prompt_bundle
from .pipeline import complete_run, fail_run, start_run
from .sources.wiktionary import (
    DEFAULT_WIKTIONARY_SOURCE_URL,
    download_wiktionary_dump,
    extract_wiktionary_dump,
)
from .stages.curated_build import (
    CURATED_BUILD_STAGE,
    DEFAULT_CURATED_TABLE,
    run_curated_build_stage,
)
from .stages.curated_build.word_selection import build_word_selection_rule
from .qa.audit import audit_definitions
from .qa.review_stage import QUALITY_REVIEW_STAGE, run_quality_review_stage
from .stages.export_distribution_jsonl import (
    DISTRIBUTION_SCHEMA_VERSION,
    EXPORT_DISTRIBUTION_JSONL_STAGE,
    run_export_distribution_jsonl_stage,
    validate_distribution_jsonl_file,
)
from .stages.export_distribution_sqlite import (
    EXPORT_DISTRIBUTION_SQLITE_STAGE,
    SQLITE_SCHEMA_VERSION,
    run_export_distribution_sqlite_stage,
)
from .stages.export_jsonl import (
    EXPORT_AUDIT_JSONL_STAGE,
    run_export_audit_jsonl_stage,
)
from .stages.llm_enrich import LLM_ENRICH_STAGE, count_pending_entries, run_llm_enrich_stage
from .stages.raw_ingest import DEFAULT_RAW_TABLE, RAW_INGEST_STAGE, run_raw_ingest_stage


def _add_database_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file containing the database URL (default: .env).",
    )
    parser.add_argument(
        "--database-url-var",
        default="DATABASE_URL",
        help="Environment variable name holding the connection string.",
    )


def _add_definition_language_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--definition-language-code",
        help=f"BCP 47-style language code for generated definitions (default: {DEFAULT_DEFINITION_LANGUAGE.code}).",
    )
    parser.add_argument(
        "--definition-language-name",
        help=f"Human-readable name for the definition language (default: {DEFAULT_DEFINITION_LANGUAGE.name}).",
    )


def _add_word_selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--top-words",
        type=int,
        help=(
            "Select only headwords whose wordfreq Zipf frequency reaches the "
            "top-N token threshold (word_selection_v1). Applies to groups in "
            "the selection language; other languages pass through unfiltered."
        ),
    )
    parser.add_argument(
        "--top-words-lang",
        default="en",
        help="wordfreq language for the top-words selection rule (default: %(default)s).",
    )
    parser.add_argument(
        "--phrase-min-zipf",
        type=float,
        help=(
            "Stricter Zipf boundary for multiword headwords, compensating for "
            "wordfreq's optimistic combined-token phrase estimates. Omit to "
            "hold phrases to the same boundary as single words."
        ),
    )


def _make_progress_callback():
    def progress_callback(event: dict[str, object]) -> None:
        stage = event.get("stage", "unknown")
        name = event.get("event", "progress")
        details = " ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"stage", "event"} and value is not None
        )
        message = f"[progress] stage={stage} event={name}"
        if details:
            message += f" {details}"
        print(message, file=sys.stderr, flush=True)

    return progress_callback


def _print_command_result(command: str, **payload: object) -> None:
    print(
        json.dumps(
            {
                "command": command,
                "status": "succeeded",
                **payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _get_settings(args: argparse.Namespace):
    try:
        return load_settings(
            env_file=getattr(args, "env_file", ".env"),
            database_url_var=getattr(args, "database_url_var", "DATABASE_URL"),
        )
    except RuntimeError as exc:
        args._parser.error(str(exc))


def _identifier_from_dotted(qualified_name: str) -> sql.Identifier:
    parts = [segment.strip() for segment in qualified_name.split(".") if segment.strip()]
    if not parts:
        raise ValueError("Identifier name cannot be empty")
    return sql.Identifier(*parts)


def _get_definition_language(args: argparse.Namespace):
    code = getattr(args, "definition_language_code", None)
    name = getattr(args, "definition_language_name", None)
    if (code is None) != (name is None):
        args._parser.error(
            "Provide both --definition-language-code and --definition-language-name together, "
            "or omit both to use the default definition language."
        )
    try:
        return normalize_language_spec(
            {
                "code": code or DEFAULT_DEFINITION_LANGUAGE.code,
                "name": name or DEFAULT_DEFINITION_LANGUAGE.name,
            }
        )
    except ValueError as exc:
        args._parser.error(str(exc))


def _count_pending_llm_entries(
    settings,
    *,
    source_table: str,
    target_table: str,
    prompt_bundle,
    models: list[str],
    recompute_existing: bool,
) -> int:
    return count_pending_entries(
        settings,
        source_table=source_table,
        target_table=target_table,
        prompt_bundle=prompt_bundle,
        models=models,
        recompute_existing=recompute_existing,
    )


def _fetch_recent_failed_enrichments(
    settings,
    *,
    target_table: str,
    prompt_version: str,
    definition_language_code: str,
    models: list[str] | None,
    limit: int,
) -> list[tuple[str, str]]:
    query = sql.SQL(
        """
        SELECT entry_id::text, left(error, 300)
        FROM {target_table}
        WHERE status = 'failed'
          AND prompt_version = %s
          AND definition_language_code = %s
          {model_filter}
        ORDER BY enrichment_id DESC
        LIMIT %s
        """
    ).format(
        target_table=_identifier_from_dotted(target_table),
        model_filter=sql.SQL("AND model = ANY(%s)") if models else sql.SQL(""),
    )
    params: list[object] = [prompt_version, definition_language_code]
    if models:
        params.append(list(models))
    params.append(limit)
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        destination = download_wiktionary_dump(
            args.output,
            url=args.url,
            overwrite=args.overwrite,
        )
    except RuntimeError as exc:  # pragma: no cover - network failure guard
        args._parser.error(str(exc))
    except OSError as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "fetch-snapshot",
        output_path=str(destination),
        source_url=args.url,
    )
    return 0


def _cmd_db_init(args: argparse.Namespace) -> int:
    settings = _get_settings(args)

    with get_connection(settings) as conn:
        applied_versions = apply_foundation(conn)

    _print_command_result(
        "init-db",
        latest_foundation_version=LATEST_FOUNDATION_VERSION,
        applied_versions=applied_versions,
        already_current=not applied_versions,
    )
    return 0


def _get_word_selection(args: argparse.Namespace):
    top_words = getattr(args, "top_words", None)
    if top_words is None:
        return None
    try:
        return build_word_selection_rule(
            lang=args.top_words_lang,
            top_n=top_words,
            phrase_min_zipf=getattr(args, "phrase_min_zipf", None),
        )
    except ValueError as exc:
        args._parser.error(str(exc))


def _cmd_curated_build(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    word_selection = _get_word_selection(args)

    try:
        result = run_curated_build_stage(
            settings=settings,
            source_table=args.source_table,
            target_table=args.target_table,
            relations_table=args.relations_table,
            triage_table=args.triage_table,
            lang_codes=args.lang_codes,
            limit_groups=args.limit_groups,
            replace_existing=args.replace_existing,
            word_selection=word_selection,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError) as exc:
        args._parser.error(f"Database error: {exc}")

    _print_command_result(
        "assemble-entries",
        stage=CURATED_BUILD_STAGE,
        run_id=str(result.run_id),
        groups_processed=result.groups_processed,
        groups_filtered_out=result.groups_filtered_out,
        entries_written=result.entries_written,
        relations_written=result.relations_written,
        triage_written=result.triage_written,
        word_selection=(
            word_selection.as_metadata() if word_selection is not None else None
        ),
    )
    return 0


def _cmd_review_definitions(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    model_env_file = args.model_env_file or args.env_file

    try:
        result = run_quality_review_stage(
            settings=settings,
            env_file=model_env_file,
            llm_table=args.definitions_table,
            review_table=args.review_table,
            generation_prompt_version_like=args.prompt_version_like,
            sample_size=args.sample_size,
            seed=args.seed,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        args._parser.error(str(exc))

    from .qa.judge import REVIEW_PROMPT_VERSION as _review_prompt_version

    report = _summarize_reviews(
        settings,
        review_table=args.review_table,
        seed=args.seed,
        review_prompt_version=_review_prompt_version,
    )
    _print_command_result(
        "review-definitions",
        stage=QUALITY_REVIEW_STAGE,
        run_id=str(result.run_id),
        sampled=result.sampled,
        reviewed=result.reviewed,
        failed=result.failed,
        skipped_existing=result.skipped_existing,
        report=report,
    )
    return 0


def _summarize_reviews(settings, *, review_table: str, seed: str, review_prompt_version: str) -> dict:
    review_identifier = _identifier_from_dotted(review_table)
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT verdict,
                           count(*),
                           sum(sampling_weight),
                           avg((scores->>'accuracy')::int),
                           avg((scores->>'examples')::int)
                    FROM {}
                    WHERE status = 'succeeded' AND sample_seed = %s
                      AND review_prompt_version = %s
                    GROUP BY verdict
                    """
                ).format(review_identifier),
                (seed, review_prompt_version),
            )
            verdict_rows = cursor.fetchall()
            cursor.execute(
                sql.SQL(
                    """
                    SELECT issue->>'kind', count(*)
                    FROM {}, jsonb_array_elements(issues) issue
                    WHERE status = 'succeeded' AND sample_seed = %s
                      AND review_prompt_version = %s
                    GROUP BY 1 ORDER BY 2 DESC
                    """
                ).format(review_identifier),
                (seed, review_prompt_version),
            )
            issue_rows = cursor.fetchall()
            cursor.execute(
                sql.SQL(
                    """
                    SELECT stratum,
                           count(*),
                           count(*) FILTER (WHERE verdict = 'major_issues')
                    FROM {}
                    WHERE status = 'succeeded' AND sample_seed = %s
                      AND review_prompt_version = %s
                    GROUP BY stratum ORDER BY 3 DESC, stratum
                    """
                ).format(review_identifier),
                (seed, review_prompt_version),
            )
            stratum_rows = cursor.fetchall()

    total_weight = sum(float(row[2]) for row in verdict_rows) or 1.0
    return {
        "verdicts": {
            row[0]: {
                "count": row[1],
                "weighted_share": round(float(row[2]) / total_weight, 4),
                "avg_accuracy": round(float(row[3]), 2) if row[3] is not None else None,
                "avg_examples": round(float(row[4]), 2) if row[4] is not None else None,
            }
            for row in verdict_rows
        },
        "issue_kinds": {row[0]: row[1] for row in issue_rows},
        "strata": [
            {"stratum": row[0], "reviewed": row[1], "major_issues": row[2]}
            for row in stratum_rows
        ],
    }


def _cmd_audit_definitions(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    try:
        report = audit_definitions(
            settings,
            llm_table=args.definitions_table,
            prompt_version_like=args.prompt_version_like,
            bucket_size=args.bucket_size,
            sample_size=args.sample_size,
        )
    except (psycopg.Error, ValueError) as exc:
        args._parser.error(str(exc))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_llm_enrich(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    definition_language = _get_definition_language(args)
    model_env_file = args.model_env_file or args.env_file
    prompt_bundle = build_prompt_bundle(
        prompt_version=args.prompt_version,
        definition_language=definition_language,
    )

    try:
        result = run_llm_enrich_stage(
            settings=settings,
            env_file=model_env_file,
            source_table=args.source_table,
            target_table=args.target_table,
            prompt_version=args.prompt_version,
            definition_language=definition_language,
            limit_entries=args.limit_entries,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            recompute_existing=args.recompute_existing,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "generate-definitions",
        stage=LLM_ENRICH_STAGE,
        run_id=str(result.run_id),
        processed=result.processed,
        succeeded=result.succeeded,
        failed=result.failed,
        prompt_template_version=args.prompt_version,
        prompt_version=prompt_bundle.resolved_prompt_version,
        definition_language=definition_language.as_dict(),
        max_workers=args.max_workers,
    )
    return 0


def _cmd_export_audit_jsonl(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    definition_language = _get_definition_language(args)
    prompt_bundle = (
        build_prompt_bundle(
            prompt_version=args.prompt_version,
            definition_language=definition_language,
        )
        if args.prompt_version is not None
        else None
    )

    try:
        result = run_export_audit_jsonl_stage(
            settings=settings,
            output_path=args.output,
            curated_table=args.curated_table,
            llm_table=args.llm_table,
            artifact_table=args.artifact_table,
            models=args.model,
            prompt_version=args.prompt_version,
            definition_language=definition_language,
            include_unenriched=args.include_unenriched,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "export-audit",
        stage=EXPORT_AUDIT_JSONL_STAGE,
        run_id=str(result.run_id),
        entry_count=result.entry_count,
        output_path=str(result.output_path),
        output_sha256=result.output_sha256,
        prompt_template_version=args.prompt_version,
        prompt_version=prompt_bundle.resolved_prompt_version if prompt_bundle is not None else None,
        definition_language=definition_language.as_dict(),
    )
    return 0


def _cmd_export_distribution_jsonl(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    definition_language = _get_definition_language(args)
    prompt_versions = args.prompt_version or [PROMPT_VERSION]
    prompt_bundles = [
        build_prompt_bundle(prompt_version=version, definition_language=definition_language)
        for version in prompt_versions
    ]

    try:
        result = run_export_distribution_jsonl_stage(
            settings=settings,
            output_path=args.output,
            curated_table=args.curated_table,
            llm_table=args.llm_table,
            artifact_table=args.artifact_table,
            models=args.model,
            prompt_versions=args.prompt_version,
            definition_language=definition_language,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "export-distribution",
        stage=EXPORT_DISTRIBUTION_JSONL_STAGE,
        schema_version=DISTRIBUTION_SCHEMA_VERSION,
        run_id=str(result.run_id),
        entry_count=result.entry_count,
        output_path=str(result.output_path),
        output_sha256=result.output_sha256,
        prompt_template_versions=prompt_versions,
        prompt_versions=[bundle.resolved_prompt_version for bundle in prompt_bundles],
        definition_language=definition_language.as_dict(),
    )
    return 0


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    model_env_file = args.model_env_file or args.env_file
    worker_tiers = args.worker_tiers or [args.max_workers]
    progress_callback = _make_progress_callback()
    definition_language = _get_definition_language(args)
    word_selection = _get_word_selection(args)
    prompt_bundle = build_prompt_bundle(
        prompt_version=args.prompt_version,
        definition_language=definition_language,
    )
    workflow_run_id = None
    raw_result = None
    curated_result = None
    llm_result = None
    distribution_result = None
    distribution_validation = None
    distribution_sqlite_result = None
    audit_result = None

    try:
        export_models = args.model or list(load_llm_settings(env_file=model_env_file).models)
        if not args.skip_init_db:
            with get_connection(settings) as conn:
                apply_foundation(conn)

        with get_connection(settings) as conn:
            workflow_run_id = start_run(
                conn,
                stage="workflow.run",
                config={
                    "archive_path": str(args.archive_path) if args.archive_path is not None else None,
                    "source_url": (
                        args.source_url or DEFAULT_WIKTIONARY_SOURCE_URL
                        if args.archive_path is None
                        else None
                    ),
                    "workdir": str(args.workdir),
                    "skip_init_db": args.skip_init_db,
                    "raw_table": args.raw_table,
                    "curated_table": args.curated_table,
                    "relations_table": args.relations_table,
                    "triage_table": args.triage_table,
                    "definitions_table": args.llm_table,
                    "artifact_table": args.artifact_table,
                    "lang_codes": args.lang_codes or [],
                    "limit_groups": args.limit_groups,
                    "limit_entries": args.limit_entries,
                    "word_selection": (
                        word_selection.as_metadata() if word_selection is not None else None
                    ),
                    "worker_tiers": worker_tiers,
                    "max_retries": args.max_retries,
                    "recompute_existing": args.recompute_existing,
                    "distribution_output": str(args.distribution_output),
                    "distribution_sqlite_output": (
                        str(args.distribution_sqlite_output)
                        if args.distribution_sqlite_output is not None
                        else None
                    ),
                    "audit_output": str(args.audit_output) if args.audit_output is not None else None,
                    "validate_distribution": args.validate_distribution,
                    "skip_distribution_export": args.skip_distribution_export,
                    "prompt_template_version": args.prompt_version,
                    "prompt_version": prompt_bundle.resolved_prompt_version,
                    "definition_language": definition_language.as_dict(),
                },
            )

        raw_result = run_raw_ingest_stage(
            settings=settings,
            workdir=args.workdir,
            source_url=(
                args.source_url or DEFAULT_WIKTIONARY_SOURCE_URL
                if args.archive_path is None
                else None
            ),
            archive_path=args.archive_path,
            target_table=args.raw_table,
            overwrite_download=args.overwrite_download,
            parent_run_id=workflow_run_id,
            progress_callback=progress_callback,
        )

        curated_result = run_curated_build_stage(
            settings=settings,
            source_table=args.raw_table,
            target_table=args.curated_table,
            relations_table=args.relations_table,
            triage_table=args.triage_table,
            lang_codes=args.lang_codes,
            limit_groups=args.limit_groups,
            replace_existing=args.replace_existing_curated,
            word_selection=word_selection,
            parent_run_id=workflow_run_id,
            progress_callback=progress_callback,
        )

        llm_attempts: list[dict[str, object]] = []
        if args.limit_entries is not None:
            llm_result = run_llm_enrich_stage(
                settings=settings,
                env_file=model_env_file,
                source_table=args.curated_table,
                target_table=args.llm_table,
                prompt_version=args.prompt_version,
                definition_language=definition_language,
                limit_entries=args.limit_entries,
                max_workers=worker_tiers[0],
                max_retries=args.max_retries,
                recompute_existing=args.recompute_existing,
                parent_run_id=workflow_run_id,
                progress_callback=progress_callback,
            )
            llm_attempts.append(
                {
                    "workers": worker_tiers[0],
                    "processed": llm_result.processed,
                    "succeeded": llm_result.succeeded,
                    "failed": llm_result.failed,
                    "remaining_entries": None,
                }
            )
            if llm_result.failed:
                raise RuntimeError(
                    f"Definition generation left {llm_result.failed} failed entries in the limited run. "
                    "Resolve them or rerun with a safer concurrency before export."
                )
        else:
            remaining_entries = _count_pending_llm_entries(
                settings,
                source_table=args.curated_table,
                target_table=args.llm_table,
                prompt_bundle=prompt_bundle,
                models=export_models,
                recompute_existing=args.recompute_existing,
            )
            for workers in worker_tiers:
                if remaining_entries == 0:
                    break
                progress_callback(
                    {
                        "stage": "workflow.run",
                        "event": "definitions_attempt_start",
                        "workers": workers,
                        "remaining_entries": remaining_entries,
                        "definition_language_code": definition_language.code,
                    }
                )
                llm_result = run_llm_enrich_stage(
                    settings=settings,
                    env_file=model_env_file,
                    source_table=args.curated_table,
                    target_table=args.llm_table,
                    prompt_version=args.prompt_version,
                    definition_language=definition_language,
                    limit_entries=None,
                    max_workers=workers,
                    max_retries=args.max_retries,
                    recompute_existing=args.recompute_existing,
                    parent_run_id=workflow_run_id,
                    progress_callback=progress_callback,
                )
                remaining_entries = _count_pending_llm_entries(
                    settings,
                    source_table=args.curated_table,
                    target_table=args.llm_table,
                    prompt_bundle=prompt_bundle,
                    models=export_models,
                    recompute_existing=args.recompute_existing,
                )
                llm_attempts.append(
                    {
                        "workers": workers,
                        "processed": llm_result.processed,
                        "succeeded": llm_result.succeeded,
                        "failed": llm_result.failed,
                        "remaining_entries": remaining_entries,
                    }
                )
                progress_callback(
                    {
                        "stage": "workflow.run",
                        "event": "definitions_attempt_complete",
                        **llm_attempts[-1],
                    }
                )

            if remaining_entries != 0:
                failed_examples = _fetch_recent_failed_enrichments(
                    settings,
                    target_table=args.llm_table,
                    prompt_version=prompt_bundle.resolved_prompt_version,
                    definition_language_code=definition_language.code,
                    models=export_models,
                    limit=10,
                )
                raise RuntimeError(
                    f"Definition generation still has {remaining_entries} unresolved entries after worker tiers {worker_tiers}. "
                    f"Recent failed examples: {failed_examples}"
                )

        assert llm_result is not None

        if not args.skip_distribution_export:
            distribution_result = run_export_distribution_jsonl_stage(
                settings=settings,
                output_path=args.distribution_output,
                curated_table=args.curated_table,
                llm_table=args.llm_table,
                artifact_table=args.artifact_table,
                models=export_models,
                prompt_versions=[args.prompt_version],
                definition_language=definition_language,
                parent_run_id=workflow_run_id,
                progress_callback=progress_callback,
            )
            if args.validate_distribution:
                distribution_validation = validate_distribution_jsonl_file(
                    args.distribution_output,
                    progress_callback=progress_callback,
                )
            else:
                distribution_validation = None
        else:
            distribution_validation = None

        if args.distribution_sqlite_output is not None:
            distribution_sqlite_result = run_export_distribution_sqlite_stage(
                settings=settings,
                output_path=args.distribution_sqlite_output,
                curated_table=args.curated_table,
                llm_table=args.llm_table,
                artifact_table=args.artifact_table,
                models=export_models,
                prompt_versions=[args.prompt_version],
                definition_language=definition_language,
                parent_run_id=workflow_run_id,
                progress_callback=progress_callback,
            )

        if args.audit_output is not None:
            audit_result = run_export_audit_jsonl_stage(
                settings=settings,
                output_path=args.audit_output,
                curated_table=args.curated_table,
                llm_table=args.llm_table,
                artifact_table=args.artifact_table,
                models=export_models,
                prompt_version=args.prompt_version,
                definition_language=definition_language,
                include_unenriched=args.include_unenriched_audit,
                parent_run_id=workflow_run_id,
                progress_callback=progress_callback,
            )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        if workflow_run_id is not None:
            with get_connection(settings) as conn:
                fail_run(
                    conn,
                    run_id=workflow_run_id,
                    error=str(exc),
                    stats={
                        "source_run_id": str(raw_result.run_id) if raw_result is not None else None,
                        "entries_run_id": str(curated_result.run_id) if curated_result is not None else None,
                        "definitions_run_id": str(llm_result.run_id) if llm_result is not None else None,
                        "distribution_export_run_id": (
                            str(distribution_result.run_id) if distribution_result is not None else None
                        ),
                        "distribution_sqlite_export_run_id": (
                            str(distribution_sqlite_result.run_id)
                            if distribution_sqlite_result is not None
                            else None
                        ),
                        "audit_export_run_id": str(audit_result.run_id) if audit_result is not None else None,
                    },
                )
        args._parser.error(str(exc))

    if workflow_run_id is not None:
        with get_connection(settings) as conn:
            complete_run(
                conn,
                run_id=workflow_run_id,
                stats={
                    "source_run_id": str(raw_result.run_id),
                    "snapshot_id": str(raw_result.snapshot_id),
                    "entries_run_id": str(curated_result.run_id),
                    "definitions_run_id": str(llm_result.run_id),
                    "distribution_export_run_id": (
                        str(distribution_result.run_id) if distribution_result is not None else None
                    ),
                    "distribution_sqlite_export_run_id": (
                        str(distribution_sqlite_result.run_id)
                        if distribution_sqlite_result is not None
                        else None
                    ),
                    "audit_export_run_id": str(audit_result.run_id) if audit_result is not None else None,
                },
            )

    summary = {
        "workflow_run_id": str(workflow_run_id) if workflow_run_id is not None else None,
        "db_initialized": not args.skip_init_db,
        "source": {
            "run_id": str(raw_result.run_id),
            "snapshot_id": str(raw_result.snapshot_id),
            "rows_loaded": raw_result.rows_loaded,
            "anomalies_logged": raw_result.anomalies_logged,
            "resumed_from_run_id": (
                str(raw_result.resumed_from_run_id) if raw_result.resumed_from_run_id is not None else None
            ),
        },
        "entries": {
            "run_id": str(curated_result.run_id),
            "entries_written": curated_result.entries_written,
            "groups_filtered_out": curated_result.groups_filtered_out,
            "relations_written": curated_result.relations_written,
            "triage_written": curated_result.triage_written,
            "word_selection": (
                word_selection.as_metadata() if word_selection is not None else None
            ),
        },
        "definitions": {
            "run_id": str(llm_result.run_id),
            "processed": llm_result.processed,
            "succeeded": llm_result.succeeded,
            "failed": llm_result.failed,
            "prompt_template_version": args.prompt_version,
            "prompt_version": prompt_bundle.resolved_prompt_version,
            "definition_language": definition_language.as_dict(),
            "worker_tiers": worker_tiers,
            "attempts": llm_attempts,
        },
        "distribution_export": (
            {
                "run_id": str(distribution_result.run_id),
                "entry_count": distribution_result.entry_count,
                "output_path": str(distribution_result.output_path),
                "output_sha256": distribution_result.output_sha256,
                "prompt_template_version": args.prompt_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": definition_language.as_dict(),
                "validated_entry_count": (
                    distribution_validation.entry_count
                    if distribution_validation is not None
                    else None
                ),
            }
            if distribution_result is not None
            else None
        ),
        "distribution_sqlite_export": (
            {
                "run_id": str(distribution_sqlite_result.run_id),
                "entry_count": distribution_sqlite_result.entry_count,
                "output_path": str(distribution_sqlite_result.output_path),
                "output_sha256": distribution_sqlite_result.output_sha256,
                "prompt_template_version": args.prompt_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": definition_language.as_dict(),
                "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
            }
            if distribution_sqlite_result is not None
            else None
        ),
        "audit_export": (
            {
                "run_id": str(audit_result.run_id),
                "entry_count": audit_result.entry_count,
                "output_path": str(audit_result.output_path),
                "output_sha256": audit_result.output_sha256,
                "prompt_template_version": args.prompt_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": definition_language.as_dict(),
            }
            if audit_result is not None
            else None
        ),
    }
    _print_command_result("run", **summary)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _cmd_pipeline_run(args)


def _cmd_export_distribution_sqlite(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()
    definition_language = _get_definition_language(args)
    prompt_versions = args.prompt_version or [PROMPT_VERSION]
    prompt_bundles = [
        build_prompt_bundle(prompt_version=version, definition_language=definition_language)
        for version in prompt_versions
    ]

    try:
        result = run_export_distribution_sqlite_stage(
            settings=settings,
            output_path=args.output,
            curated_table=args.curated_table,
            llm_table=args.llm_table,
            artifact_table=args.artifact_table,
            models=args.model,
            prompt_versions=args.prompt_version,
            definition_language=definition_language,
            progress_callback=progress_callback,
        )
    except (psycopg.Error, ValueError, RuntimeError) as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "export-distribution-sqlite",
        stage=EXPORT_DISTRIBUTION_SQLITE_STAGE,
        schema_version=DISTRIBUTION_SCHEMA_VERSION,
        sqlite_schema_version=SQLITE_SCHEMA_VERSION,
        run_id=str(result.run_id),
        entry_count=result.entry_count,
        output_path=str(result.output_path),
        output_sha256=result.output_sha256,
        prompt_template_versions=prompt_versions,
        prompt_versions=[bundle.resolved_prompt_version for bundle in prompt_bundles],
        definition_language=definition_language.as_dict(),
    )
    return 0


def _cmd_validate_distribution_jsonl(args: argparse.Namespace) -> int:
    try:
        result = validate_distribution_jsonl_file(
            args.input,
            progress_callback=_make_progress_callback(),
        )
    except ValueError as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "validate-distribution",
        schema_version=DISTRIBUTION_SCHEMA_VERSION,
        output_path=str(result.output_path),
        entry_count=result.entry_count,
    )
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    try:
        output = extract_wiktionary_dump(
            args.input,
            args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, IsADirectoryError) as exc:
        args._parser.error(str(exc))
    except OSError as exc:
        args._parser.error(str(exc))

    _print_command_result(
        "unpack-snapshot",
        input_path=str(args.input),
        output_path=str(output),
    )
    return 0


def _cmd_raw_ingest(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    progress_callback = _make_progress_callback()

    try:
        result = run_raw_ingest_stage(
            settings=settings,
            workdir=args.workdir,
            source_url=(
                args.source_url or DEFAULT_WIKTIONARY_SOURCE_URL
                if args.archive_path is None
                else None
            ),
            archive_path=args.archive_path,
            target_table=args.target_table,
            overwrite_download=args.overwrite_download,
            progress_callback=progress_callback,
        )
    except FileNotFoundError as exc:
        args._parser.error(str(exc))
    except RuntimeError as exc:  # pragma: no cover - network failure guard
        args._parser.error(str(exc))
    except (psycopg.Error, ValueError) as exc:
        args._parser.error(f"Database error: {exc}")

    _print_command_result(
        "ingest-snapshot",
        stage=RAW_INGEST_STAGE,
        run_id=str(result.run_id),
        snapshot_id=str(result.snapshot_id),
        rows_loaded=result.rows_loaded,
        anomalies_logged=result.anomalies_logged,
        snapshot_preexisting=result.snapshot_preexisting,
        resumed_from_run_id=(
            str(result.resumed_from_run_id) if result.resumed_from_run_id is not None else None
        ),
        archive_path=str(result.archive_path),
        archive_sha256=result.archive_sha256,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tracked rewrite pipeline for Wiktionary-derived dictionary builds.",
    )
    subparsers = parser.add_subparsers(dest="command")

    db_init_parser = subparsers.add_parser(
        "init-db",
        help="Apply the rewrite foundation schemas and tables.",
    )
    _add_database_options(db_init_parser)
    db_init_parser.set_defaults(func=_cmd_db_init, _parser=db_init_parser)

    curated_build_parser = subparsers.add_parser(
        "assemble-entries",
        help="Assemble word-centric learner entries from raw Wiktionary records.",
    )
    curated_build_parser.add_argument(
        "--source-table",
        default=DEFAULT_RAW_TABLE,
        help="Source raw table to read from (default: %(default)s).",
    )
    curated_build_parser.add_argument(
        "--target-table",
        default=DEFAULT_CURATED_TABLE,
        help="Target assembled-entries table (default: %(default)s).",
    )
    curated_build_parser.add_argument(
        "--relations-table",
        default="curated.entry_relations",
        help="Target table for normalized relations (default: %(default)s).",
    )
    curated_build_parser.add_argument(
        "--triage-table",
        default="curated.triage_queue",
        help="Target table for triage items (default: %(default)s).",
    )
    curated_build_parser.add_argument(
        "--lang-codes",
        nargs="+",
        help="Optional subset of language codes to process.",
    )
    curated_build_parser.add_argument(
        "--limit-groups",
        type=int,
        help="Optional limit on grouped headwords for debugging.",
    )
    curated_build_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Clear curated output tables before rebuilding them.",
    )
    _add_word_selection_options(curated_build_parser)
    _add_database_options(curated_build_parser)
    curated_build_parser.set_defaults(func=_cmd_curated_build, _parser=curated_build_parser)

    audit_parser = subparsers.add_parser(
        "audit-definitions",
        help="Advisory quality report over generated definitions (read-only).",
    )
    audit_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        help="Generated-definitions table to audit (default: %(default)s).",
    )
    audit_parser.add_argument(
        "--prompt-version-like",
        default="%",
        help="SQL LIKE filter on prompt_version (default: all).",
    )
    audit_parser.add_argument(
        "--bucket-size",
        type=int,
        default=1000,
        help="Rows per health bucket keyed by enrichment_id (default: %(default)s).",
    )
    audit_parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Random succeeded rows to run content heuristics over (default: %(default)s).",
    )
    _add_database_options(audit_parser)
    audit_parser.set_defaults(func=_cmd_audit_definitions, _parser=audit_parser)

    review_parser = subparsers.add_parser(
        "review-definitions",
        help="LLM-judged quality review over a stratified sample of generated entries.",
    )
    review_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        help="Generated-definitions table (default: %(default)s).",
    )
    review_parser.add_argument(
        "--review-table",
        default="llm.quality_reviews",
        help="Review results table (default: %(default)s).",
    )
    review_parser.add_argument(
        "--prompt-version-like",
        default="%",
        help="SQL LIKE filter on the generation prompt_version (default: all).",
    )
    review_parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Stratified sample size (default: %(default)s).",
    )
    review_parser.add_argument(
        "--seed",
        default="review-v1",
        help="Deterministic sampling seed; same seed resumes the same sample (default: %(default)s).",
    )
    review_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent judge calls (default: %(default)s).",
    )
    review_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per judge call (default: %(default)s).",
    )
    review_parser.add_argument(
        "--model-env-file",
        help="Env file with the judge provider pool; defaults to --env-file.",
    )
    _add_database_options(review_parser)
    review_parser.set_defaults(func=_cmd_review_definitions, _parser=review_parser)

    llm_enrich_parser = subparsers.add_parser(
        "generate-definitions",
        help="Generate learner-facing definitions from assembled entries.",
    )
    llm_enrich_parser.add_argument(
        "--source-table",
        default="curated.entries",
        help="Source assembled-entries table (default: %(default)s).",
    )
    llm_enrich_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        dest="target_table",
        help="Target generated-definitions table (default: %(default)s).",
    )
    llm_enrich_parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help="Prompt version identifier to persist with this run (default: %(default)s).",
    )
    llm_enrich_parser.add_argument(
        "--limit-entries",
        type=int,
        help="Optional limit on entries processed during this run.",
    )
    llm_enrich_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads used for model calls (default: %(default)s).",
    )
    llm_enrich_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per entry before marking definition generation as failed (default: %(default)s).",
    )
    llm_enrich_parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Regenerate definitions even if successful rows already exist.",
    )
    llm_enrich_parser.add_argument(
        "--model-env-file",
        dest="model_env_file",
        help="Optional env file for model credentials. Defaults to --env-file.",
    )
    _add_definition_language_options(llm_enrich_parser)
    _add_database_options(llm_enrich_parser)
    llm_enrich_parser.set_defaults(func=_cmd_llm_enrich, _parser=llm_enrich_parser)

    export_audit_jsonl_parser = subparsers.add_parser(
        "export-audit",
        help="Export the current merged entries+definitions audit artifact as JSONL.",
    )
    export_audit_jsonl_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/export/audit.jsonl"),
        help="Output JSONL path (default: %(default)s).",
    )
    export_audit_jsonl_parser.add_argument(
        "--curated-table",
        default="curated.entries",
        help="Source assembled-entries table (default: %(default)s).",
    )
    export_audit_jsonl_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        dest="llm_table",
        help="Source generated-definitions table (default: %(default)s).",
    )
    export_audit_jsonl_parser.add_argument(
        "--artifact-table",
        default="export.artifacts",
        help="Export artifact metadata table (default: %(default)s).",
    )
    export_audit_jsonl_parser.add_argument(
        "--model",
        action="append",
        help=(
            "Optional model filter when choosing the latest successful definition run. "
            "Repeat to accept definitions from several models."
        ),
    )
    export_audit_jsonl_parser.add_argument(
        "--prompt-version",
        help="Optional prompt version filter when choosing the latest successful definition run.",
    )
    export_audit_jsonl_parser.add_argument(
        "--include-unenriched",
        action="store_true",
        help="Include assembled entries even when no successful definition row exists.",
    )
    _add_definition_language_options(export_audit_jsonl_parser)
    _add_database_options(export_audit_jsonl_parser)
    export_audit_jsonl_parser.set_defaults(
        func=_cmd_export_audit_jsonl,
        _parser=export_audit_jsonl_parser,
    )

    export_distribution_jsonl_parser = subparsers.add_parser(
        "export-distribution",
        help="Export the learner-facing distribution JSONL artifact.",
    )
    export_distribution_jsonl_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/export/distribution.jsonl"),
        help="Output JSONL path (default: %(default)s).",
    )
    export_distribution_jsonl_parser.add_argument(
        "--curated-table",
        default="curated.entries",
        help="Source assembled-entries table (default: %(default)s).",
    )
    export_distribution_jsonl_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        dest="llm_table",
        help="Source generated-definitions table (default: %(default)s).",
    )
    export_distribution_jsonl_parser.add_argument(
        "--artifact-table",
        default="export.artifacts",
        help="Export artifact metadata table (default: %(default)s).",
    )
    export_distribution_jsonl_parser.add_argument(
        "--model",
        action="append",
        help=(
            "Optional model filter when choosing the latest successful definition run. "
            "Repeat to accept definitions from several models."
        ),
    )
    export_distribution_jsonl_parser.add_argument(
        "--prompt-version",
        action="append",
        help=(
            "Prompt version accepted for distribution export; repeat to give a "
            "preference order (first wins per entry). Defaults to the current "
            f"prompt version {PROMPT_VERSION}."
        ),
    )
    _add_definition_language_options(export_distribution_jsonl_parser)
    _add_database_options(export_distribution_jsonl_parser)
    export_distribution_jsonl_parser.set_defaults(
        func=_cmd_export_distribution_jsonl,
        _parser=export_distribution_jsonl_parser,
    )

    export_distribution_sqlite_parser = subparsers.add_parser(
        "export-distribution-sqlite",
        help="Export the learner-facing distribution artifact as SQLite.",
    )
    export_distribution_sqlite_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/export/distribution.sqlite"),
        help="Output SQLite path (default: %(default)s).",
    )
    export_distribution_sqlite_parser.add_argument(
        "--curated-table",
        default="curated.entries",
        help="Source assembled-entries table (default: %(default)s).",
    )
    export_distribution_sqlite_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        dest="llm_table",
        help="Source generated-definitions table (default: %(default)s).",
    )
    export_distribution_sqlite_parser.add_argument(
        "--artifact-table",
        default="export.artifacts",
        help="Export artifact metadata table (default: %(default)s).",
    )
    export_distribution_sqlite_parser.add_argument(
        "--model",
        action="append",
        help=(
            "Optional model filter when choosing the latest successful definition run. "
            "Repeat to accept definitions from several models."
        ),
    )
    export_distribution_sqlite_parser.add_argument(
        "--prompt-version",
        action="append",
        help=(
            "Prompt version accepted for distribution export; repeat to give a "
            "preference order (first wins per entry). Defaults to the current "
            f"prompt version {PROMPT_VERSION}."
        ),
    )
    _add_definition_language_options(export_distribution_sqlite_parser)
    _add_database_options(export_distribution_sqlite_parser)
    export_distribution_sqlite_parser.set_defaults(
        func=_cmd_export_distribution_sqlite,
        _parser=export_distribution_sqlite_parser,
    )

    validate_distribution_jsonl_parser = subparsers.add_parser(
        "validate-distribution",
        help="Validate every row of a learner-facing distribution JSONL file.",
    )
    validate_distribution_jsonl_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a distribution JSONL file to validate.",
    )
    validate_distribution_jsonl_parser.set_defaults(
        func=_cmd_validate_distribution_jsonl,
        _parser=validate_distribution_jsonl_parser,
    )

    download_parser = subparsers.add_parser(
        "fetch-snapshot",
        help="Download a Wiktionary snapshot archive for local inspection or staged ingest.",
    )
    download_parser.add_argument(
        "--url",
        default=DEFAULT_WIKTIONARY_SOURCE_URL,
        help="Source URL for the Wiktionary dump (default: official raw dataset).",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw-wiktextract-data.jsonl.gz"),
        help="Where to store the downloaded archive (default: data/raw-wiktextract-data.jsonl.gz).",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing archive if it already exists.",
    )
    download_parser.set_defaults(func=_cmd_download, _parser=download_parser)

    extract_parser = subparsers.add_parser(
        "unpack-snapshot",
        help="Extract a snapshot archive to a plain JSONL file for local inspection.",
    )
    extract_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw-wiktextract-data.jsonl.gz"),
        help="Path to the .jsonl.gz archive (default: data/raw-wiktextract-data.jsonl.gz).",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw-wiktextract-data.jsonl"),
        help="Where to write the decompressed JSONL file (default: data/raw-wiktextract-data.jsonl).",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the extracted JSONL if it already exists.",
    )
    extract_parser.set_defaults(func=_cmd_extract, _parser=extract_parser)

    raw_ingest_parser = subparsers.add_parser(
        "ingest-snapshot",
        help="Run the tracked raw Wiktionary ingestion stage into the rewrite tables.",
    )
    raw_ingest_parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/raw"),
        help="Working directory for archive and extracted JSONL files (default: %(default)s).",
    )
    source_group = raw_ingest_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-url",
        help="Source URL for the Wiktionary snapshot (default: official raw dataset).",
    )
    source_group.add_argument(
        "--archive-path",
        type=Path,
        help="Use an existing local .jsonl or .jsonl.gz archive instead of downloading.",
    )
    raw_ingest_parser.add_argument(
        "--target-table",
        default=DEFAULT_RAW_TABLE,
        help="Destination raw table for imported entries (default: %(default)s).",
    )
    raw_ingest_parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download or overwrite an acquired archive in workdir.",
    )
    _add_database_options(raw_ingest_parser)
    raw_ingest_parser.set_defaults(func=_cmd_raw_ingest, _parser=raw_ingest_parser)

    pipeline_run_parser = subparsers.add_parser(
        "run",
        help="Run the staged source -> entries -> definitions -> distribution workflow from one command.",
    )
    pipeline_run_parser.add_argument(
        "--skip-init-db",
        action="store_true",
        help="Assume the database foundation is already applied.",
    )
    pipeline_run_parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/raw"),
        help="Working directory for snapshot acquisition and ingest (default: %(default)s).",
    )
    pipeline_source_group = pipeline_run_parser.add_mutually_exclusive_group()
    pipeline_source_group.add_argument(
        "--source-url",
        help="Source URL for the Wiktionary snapshot (default: official raw dataset).",
    )
    pipeline_source_group.add_argument(
        "--archive-path",
        type=Path,
        help="Use an existing local .jsonl or .jsonl.gz archive instead of downloading.",
    )
    pipeline_run_parser.add_argument(
        "--raw-table",
        default=DEFAULT_RAW_TABLE,
        help="Destination raw table for imported entries (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download or overwrite an acquired archive in workdir.",
    )
    pipeline_run_parser.add_argument(
        "--curated-table",
        default=DEFAULT_CURATED_TABLE,
        help="Target assembled-entries table (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--relations-table",
        default="curated.entry_relations",
        help="Target table for normalized curated relations (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--triage-table",
        default="curated.triage_queue",
        help="Target table for curated triage rows (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--lang-codes",
        nargs="+",
        help="Optional subset of language codes to process.",
    )
    pipeline_run_parser.add_argument(
        "--limit-groups",
        type=int,
        help="Optional limit on grouped headwords during curated build.",
    )
    pipeline_run_parser.add_argument(
        "--replace-existing-curated",
        action="store_true",
        help="Clear curated output tables before rebuilding them.",
    )
    pipeline_run_parser.add_argument(
        "--definitions-table",
        default="llm.entry_enrichments",
        dest="llm_table",
        help="Target generated-definitions table (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help="Prompt version identifier used for definition generation and export (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--model-env-file",
        dest="model_env_file",
        help="Optional env file for model credentials. Defaults to --env-file.",
    )
    pipeline_run_parser.add_argument(
        "--limit-entries",
        type=int,
        help="Optional limit on definition-generation entries.",
    )
    pipeline_run_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Default number of concurrent model worker threads when --worker-tiers is not set (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--worker-tiers",
        nargs="+",
        type=int,
        help="Adaptive model worker tiers, retried in order until no unresolved entries remain. Example: --worker-tiers 50 12 4 1",
    )
    pipeline_run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per entry during definition generation (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Regenerate definitions even if successful rows already exist.",
    )
    pipeline_run_parser.add_argument(
        "--artifact-table",
        default="export.artifacts",
        help="Export artifact metadata table (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--model",
        action="append",
        help=(
            "Optional model filter for the export stages. Repeat to accept "
            "definitions from several models. Defaults to every model in the "
            "configured provider pool."
        ),
    )
    pipeline_run_parser.add_argument(
        "--skip-distribution-export",
        action="store_true",
        help="Run through definition generation but skip the learner-facing distribution export.",
    )
    pipeline_run_parser.add_argument(
        "--distribution-output",
        type=Path,
        default=Path("data/export/distribution.jsonl"),
        help="Output path for the learner-facing distribution export (default: %(default)s).",
    )
    pipeline_run_parser.add_argument(
        "--distribution-sqlite-output",
        type=Path,
        help="Optional output path for the learner-facing distribution SQLite export. Omit to skip SQLite export.",
    )
    pipeline_run_parser.add_argument(
        "--validate-distribution",
        action="store_true",
        help="Validate every distribution JSONL row after export.",
    )
    pipeline_run_parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional output path for the merged audit artifact. Omit to skip audit export.",
    )
    pipeline_run_parser.add_argument(
        "--include-unenriched-audit",
        action="store_true",
        help="When writing the optional audit artifact, include entries without successful definition rows.",
    )
    _add_word_selection_options(pipeline_run_parser)
    _add_definition_language_options(pipeline_run_parser)
    _add_database_options(pipeline_run_parser)
    pipeline_run_parser.set_defaults(func=_cmd_run, _parser=pipeline_run_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)

    args = parser.parse_args(argv_list)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    return func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry guard
    sys.exit(main())


__all__ = ["main"]
