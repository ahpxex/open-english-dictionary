from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg import sql

from open_dictionary.config import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE, LanguageSpec, normalize_language_spec
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.prompt import PROMPT_VERSION, build_pos_group_id, build_prompt_bundle
from open_dictionary.pipeline import ProgressCallback, ThrottledProgressReporter, complete_run, emit_progress, fail_run, start_run, update_run_config
from open_dictionary.stages.export_distribution_jsonl.schema import DISTRIBUTION_SCHEMA_VERSION, validate_distribution_document
from open_dictionary.stages.export_jsonl.stage import ExportJSONLResult, iter_curated_rows, load_matching_enrichment_candidates, record_export_artifact, select_matching_enrichment, write_jsonl_atomic


EXPORT_DISTRIBUTION_JSONL_STAGE = "distribution.export"


def run_export_distribution_jsonl_stage(
    *,
    settings: RuntimeSettings,
    output_path: Path,
    curated_table: str = "curated.entries",
    llm_table: str = "llm.entry_enrichments",
    artifact_table: str = "export.artifacts",
    models: Sequence[str] | None = None,
    prompt_versions: Sequence[str] | None = None,
    definition_language: LanguageSpec | dict[str, Any] = DEFAULT_DEFINITION_LANGUAGE,
    parent_run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExportJSONLResult:
    language = normalize_language_spec(definition_language)
    versions = list(prompt_versions) if prompt_versions else [PROMPT_VERSION]
    prompt_bundles = [
        build_prompt_bundle(prompt_version=version, definition_language=language)
        for version in versions
    ]
    prompt_bundle = prompt_bundles[0]

    with get_connection(settings) as conn:
        run_id = start_run(
            conn,
            stage=EXPORT_DISTRIBUTION_JSONL_STAGE,
            config={
                "output_path": str(output_path),
                "curated_table": curated_table,
                "definitions_table": llm_table,
                "artifact_table": artifact_table,
                "models": list(models) if models else None,
                "prompt_template_versions": [bundle.template_version for bundle in prompt_bundles],
                "prompt_versions": [bundle.resolved_prompt_version for bundle in prompt_bundles],
                "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                "artifact_role": "distribution",
                "definition_language": language.as_dict(),
            },
            parent_run_id=parent_run_id,
        )

    try:
        emit_progress(
            progress_callback,
            stage=EXPORT_DISTRIBUTION_JSONL_STAGE,
            event="export_start",
            models=list(models) if models else None,
            prompt_versions=[bundle.resolved_prompt_version for bundle in prompt_bundles],
            definition_language_code=language.code,
        )
        records = list(
            iter_distribution_records(
                settings=settings,
                curated_table=curated_table,
                llm_table=llm_table,
                models=models,
                prompt_bundles=prompt_bundles,
                progress_callback=progress_callback,
            )
        )
        skipped_unenriched = sum(
            record.get("skipped_unenriched") or 0 for record in records
        )
        records = [record for record in records if record.get("skipped_unenriched") is None]
        documents = [record["document"] for record in records if record["document"] is not None]
        skipped_entries_without_meanings = sum(1 for record in records if record["document"] is None)
        output_sha256 = write_jsonl_atomic(output_path, documents)
        emit_progress(
            progress_callback,
            stage=EXPORT_DISTRIBUTION_JSONL_STAGE,
            event="export_complete",
            entry_count=len(documents),
            skipped_entries_without_meanings=skipped_entries_without_meanings,
            output_path=str(output_path),
            output_sha256=output_sha256,
        )
        curated_run_ids = sorted({record["curated_run_id"] for record in records if record["curated_run_id"]})
        llm_run_ids = sorted({record["llm_run_id"] for record in records if record["llm_run_id"]})

        with get_connection(settings) as conn:
            update_run_config(
                conn,
                run_id=run_id,
                config_updates={
                    "curated_run_ids": curated_run_ids,
                    "definition_run_ids": llm_run_ids,
                },
            )
            record_export_artifact(
                conn,
                artifact_table=artifact_table,
                run_id=run_id,
                artifact_type="distribution_jsonl",
                output_path=output_path,
                output_sha256=output_sha256,
                entry_count=len(documents),
                metadata={
                    "curated_table": curated_table,
                    "definitions_table": llm_table,
                    "models": list(models) if models else None,
                    "prompt_template_versions": [bundle.template_version for bundle in prompt_bundles],
                    "prompt_versions": [bundle.resolved_prompt_version for bundle in prompt_bundles],
                    "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                    "artifact_role": "distribution",
                    "definition_language": language.as_dict(),
                    "curated_run_ids": curated_run_ids,
                    "definition_run_ids": llm_run_ids,
                    "skipped_entries_without_meanings": skipped_entries_without_meanings,
                    "skipped_unenriched": skipped_unenriched,
                },
            )
            complete_run(
                conn,
                run_id=run_id,
                stats={
                    "output_path": str(output_path),
                    "entry_count": len(documents),
                    "output_sha256": output_sha256,
                    "curated_run_ids": curated_run_ids,
                    "definition_run_ids": llm_run_ids,
                    "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                    "definition_language": language.as_dict(),
                    "skipped_entries_without_meanings": skipped_entries_without_meanings,
                    "skipped_unenriched": skipped_unenriched,
                },
            )

        return ExportJSONLResult(
            run_id=run_id,
            output_path=Path(output_path),
            entry_count=len(documents),
            output_sha256=output_sha256,
        )
    except Exception as exc:
        with get_connection(settings) as conn:
            fail_run(conn, run_id=run_id, error=str(exc))
        raise


def iter_distribution_records(
    *,
    settings: RuntimeSettings,
    curated_table: str,
    llm_table: str,
    models: Sequence[str] | None,
    prompt_bundle=None,
    prompt_bundles=None,
    progress_callback: ProgressCallback | None = None,
):
    bundles = list(prompt_bundles) if prompt_bundles else [prompt_bundle]
    if not bundles or bundles[0] is None:
        raise ValueError("iter_distribution_records needs at least one prompt bundle")
    candidates = load_matching_enrichment_candidates(
        settings=settings,
        llm_table=llm_table,
        models=models,
        prompt_bundles=bundles,
    )
    reporter = ThrottledProgressReporter(progress_callback, stage=EXPORT_DISTRIBUTION_JSONL_STAGE)
    processed = 0
    exported = 0
    skipped_unenriched = 0
    for curated_run_id, entry_id, _lang_code, _normalized_word, _word, curated_payload in iter_curated_rows(
        settings=settings,
        curated_table=curated_table,
    ):
        current_enrichment = select_matching_enrichment(
            curated_payload=curated_payload,
            entry_id=entry_id,
            prompt_bundles=bundles,
            candidates=candidates,
        )
        if current_enrichment is None:
            # Entries without a matching enrichment (persistent generation
            # failures) are excluded from the distribution artifact and
            # counted explicitly; they remain queued for future passes.
            processed += 1
            skipped_unenriched += 1
            continue
        if (
            current_enrichment["definition_language_code"] != bundles[0].definition_language.code
            or (current_enrichment["definition_language_name"] or "").strip()
            != bundles[0].definition_language.name
        ):
            raise ValueError(
                "Selected enrichment does not match the requested definition language contract: "
                f"expected {bundles[0].definition_language.as_dict()}, "
                f"got {{'code': {current_enrichment['definition_language_code']!r}, "
                f"'name': {current_enrichment['definition_language_name']!r}}}"
            )
        document = build_distribution_document(
            curated_payload=curated_payload,
            llm_payload=current_enrichment["response_payload"],
            definition_language=bundles[0].definition_language,
        )
        if document is not None:
            validate_distribution_document(document)
            exported += 1
        processed += 1
        reporter.report(
            event="export_progress",
            processed_entries=processed,
            exported_entries=exported,
        )
        yield {
            "curated_run_id": str(curated_run_id) if curated_run_id is not None else None,
            "llm_run_id": current_enrichment["run_id"],
            "document": document,
            "skipped_unenriched": None,
        }
    yield {
        "curated_run_id": None,
        "llm_run_id": None,
        "document": None,
        "skipped_unenriched": skipped_unenriched,
    }
    reporter.report(
        event="export_progress",
        force=True,
        processed_entries=processed,
        exported_entries=exported,
        skipped_unenriched=skipped_unenriched,
    )


def build_distribution_document(
    *,
    curated_payload: dict[str, Any],
    llm_payload: dict[str, Any],
    definition_language: LanguageSpec | dict[str, Any],
) -> dict[str, Any] | None:
    language = normalize_language_spec(definition_language)
    if not isinstance(curated_payload, dict):
        raise ValueError("Curated payload must be a JSON object")
    if not isinstance(llm_payload, dict):
        raise ValueError("LLM payload must be a JSON object")

    llm_group_lookup = {
        str(group["pos_group_id"]): group
        for group in llm_payload.get("pos_groups", [])
        if isinstance(group, dict) and group.get("pos_group_id")
    }

    distribution_pos_groups = []
    for curated_group in curated_payload.get("pos_groups", []):
        pos_group_id = build_pos_group_id(
            pos=curated_group.get("pos"),
            etymology_id=curated_group.get("etymology_id"),
        )
        llm_group = llm_group_lookup.get(pos_group_id)
        if llm_group is None:
            raise ValueError(f"LLM payload is missing generated fields for pos_group_id {pos_group_id}")
        distribution_pos_group = build_distribution_pos_group(
            curated_group=curated_group,
            llm_group=llm_group,
            pos_group_id=pos_group_id,
        )
        if distribution_pos_group is not None:
            distribution_pos_groups.append(distribution_pos_group)

    if not distribution_pos_groups:
        return None

    return {
        "schema_version": DISTRIBUTION_SCHEMA_VERSION,
        "entry_id": curated_payload["entry_id"],
        "headword": curated_payload["word"],
        "normalized_headword": curated_payload["normalized_word"],
        "headword_language": {
            "code": curated_payload["lang_code"],
            "name": curated_payload["lang"],
        },
        "definition_language": language.as_dict(),
        "entry_type": derive_entry_type(
            curated_payload.get("entry_flags") or [],
            all_groups_proper_name=_all_groups_proper_name(curated_payload),
        ),
        "headword_summary": llm_payload["headword_summary"],
        "memory_hook": llm_payload["memory_hook"],
        "study_notes": llm_payload["study_notes"],
        "etymology_note": llm_payload["etymology_note"],
        "etymologies": [
            {
                "etymology_id": group.get("etymology_id"),
                "text": group.get("etymology_text"),
                "pos_members": group.get("member_pos") or [],
            }
            for group in curated_payload.get("etymology_groups", [])
        ],
        "pos_groups": distribution_pos_groups,
    }


def build_distribution_pos_group(
    *,
    curated_group: dict[str, Any],
    llm_group: dict[str, Any],
    pos_group_id: str,
) -> dict[str, Any] | None:
    llm_meaning_lookup = {
        str(meaning["sense_id"]): meaning
        for meaning in llm_group.get("meanings", [])
        if isinstance(meaning, dict) and meaning.get("sense_id")
    }

    meanings = []
    for curated_meaning in curated_group.get("senses", []):
        sense_id = str(curated_meaning.get("sense_id"))
        llm_meaning = llm_meaning_lookup.get(sense_id)
        if llm_meaning is None:
            raise ValueError(
                f"LLM payload is missing generated meaning fields for pos_group_id {pos_group_id} sense_id {sense_id}"
            )
        meanings.append(
            {
                "sense_id": sense_id,
                "priority": llm_meaning.get("priority"),
                "short_gloss": llm_meaning.get("short_gloss"),
                "learner_explanation": llm_meaning.get("learner_explanation"),
                "usage_note": llm_meaning.get("usage_note"),
                "labels": curated_meaning.get("tags") or [],
                "topics": curated_meaning.get("topics") or [],
                "examples": [
                    {
                        "text": example.get("text"),
                        "translation": example.get("translation"),
                    }
                    for example in llm_meaning.get("examples") or []
                ],
            }
        )

    if not meanings:
        return None

    return {
        "pos": curated_group.get("pos"),
        "etymology_id": curated_group.get("etymology_id"),
        "proper_name": "entry_type:proper_name" in (curated_group.get("pos_flags") or []),
        "summary": llm_group.get("summary"),
        "usage_note": llm_group.get("usage_note"),
        "forms": select_distribution_forms(curated_group.get("forms", [])),
        "pronunciations": select_distribution_pronunciations(
            curated_group.get("pronunciations", [])
        ),
        "meanings": meanings,
        "relations": [
            {
                "type": relation.get("relation_type"),
                "word": relation.get("target_word"),
                "lang_code": relation.get("target_lang_code"),
            }
            for relation in curated_group.get("relations", [])
            if relation.get("relation_type") in {"derived_term", "related_term", "synonym", "antonym", "descendant"}
        ],
    }


def derive_entry_type(entry_flags: list[str], *, all_groups_proper_name: bool = False) -> str:
    flag_set = set(entry_flags)
    if "entry_type:proverb" in flag_set:
        return "proverb"
    if "entry_type:affix" in flag_set:
        return "affix"
    if all_groups_proper_name and "entry_type:proper_name" in flag_set:
        return "proper_name"
    return "standard"


def _all_groups_proper_name(curated_payload: dict[str, Any]) -> bool:
    groups = curated_payload.get("pos_groups") or []
    if not groups:
        return False
    return all(
        "entry_type:proper_name" in (group.get("pos_flags") or [])
        for group in groups
    )


# Distribution packaging rules (user-approved 2026-08-03): the learner artifact
# carries only inflection forms and at most one US plus one UK pronunciation.
# The curated layer keeps the full source data for audit and future products.
INFLECTION_FORM_TAGS = frozenset(
    {"plural", "comparative", "superlative", "past", "participle", "present", "third-person", "singular"}
)
US_PRONUNCIATION_TAGS = frozenset({"General-American", "US"})
UK_PRONUNCIATION_TAGS = frozenset({"Received-Pronunciation", "UK"})


def select_distribution_forms(curated_forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for form in curated_forms:
        tags = set(form.get("tags") or [])
        if not tags & INFLECTION_FORM_TAGS:
            continue
        selected.append(
            {
                "text": form.get("form"),
                "tags": sorted(tags),
                "roman": form.get("roman"),
            }
        )
    return selected


def select_distribution_pronunciations(
    curated_pronunciations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def first_ipa_with(tag_set: frozenset[str]) -> dict[str, Any] | None:
        for item in curated_pronunciations:
            if item.get("ipa") and set(item.get("tags") or []) & tag_set:
                return item
        return None

    def as_document(item: dict[str, Any], variety: str | None) -> dict[str, Any]:
        return {
            "ipa": item.get("ipa"),
            "text": item.get("pronunciation_text"),
            "tags": [variety] if variety else [],
        }

    selected = []
    us = first_ipa_with(US_PRONUNCIATION_TAGS)
    uk = first_ipa_with(UK_PRONUNCIATION_TAGS)
    if us is not None:
        selected.append(as_document(us, "US"))
    if uk is not None:
        selected.append(as_document(uk, "UK"))
    if not selected:
        for item in curated_pronunciations:
            if item.get("ipa") or item.get("pronunciation_text"):
                selected.append(as_document(item, None))
                break
    return selected
