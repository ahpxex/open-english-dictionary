from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from open_dictionary.config import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE, LanguageSpec, normalize_language_spec
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.prompt import PROMPT_VERSION, build_prompt_bundle
from open_dictionary.pipeline import ProgressCallback, ThrottledProgressReporter, complete_run, emit_progress, fail_run, start_run, update_run_config
from open_dictionary.sources.wiktionary.artifacts import install_atomic_file, prepare_atomic_path, sha256_file
from open_dictionary.stages.export_distribution_jsonl.schema import DISTRIBUTION_SCHEMA_VERSION
from open_dictionary.stages.export_distribution_jsonl.stage import iter_distribution_records
from open_dictionary.stages.export_jsonl.stage import record_export_artifact


EXPORT_DISTRIBUTION_SQLITE_STAGE = "distribution.sqlite_export"
SQLITE_SCHEMA_VERSION = "distribution_sqlite_v1"


@dataclass(frozen=True)
class ExportSQLiteResult:
    run_id: UUID
    output_path: Path
    entry_count: int
    output_sha256: str


@dataclass(frozen=True)
class SQLiteWriteResult:
    entry_count: int
    skipped_entries_without_meanings: int
    output_sha256: str
    curated_run_ids: list[str]
    definition_run_ids: list[str]


def run_export_distribution_sqlite_stage(
    *,
    settings: RuntimeSettings,
    output_path: Path,
    curated_table: str = "curated.entries",
    llm_table: str = "llm.entry_enrichments",
    artifact_table: str = "export.artifacts",
    models: Sequence[str] | None = None,
    prompt_version: str = PROMPT_VERSION,
    definition_language: LanguageSpec | dict[str, Any] = DEFAULT_DEFINITION_LANGUAGE,
    parent_run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExportSQLiteResult:
    language = normalize_language_spec(definition_language)
    prompt_bundle = build_prompt_bundle(
        prompt_version=prompt_version,
        definition_language=language,
    )

    with get_connection(settings) as conn:
        run_id = start_run(
            conn,
            stage=EXPORT_DISTRIBUTION_SQLITE_STAGE,
            config={
                "output_path": str(output_path),
                "curated_table": curated_table,
                "definitions_table": llm_table,
                "artifact_table": artifact_table,
                "models": list(models) if models else None,
                "prompt_template_version": prompt_bundle.template_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
                "artifact_role": "distribution",
                "definition_language": language.as_dict(),
            },
            parent_run_id=parent_run_id,
        )

    try:
        emit_progress(
            progress_callback,
            stage=EXPORT_DISTRIBUTION_SQLITE_STAGE,
            event="export_start",
            models=list(models) if models else None,
            prompt_version=prompt_bundle.resolved_prompt_version,
            prompt_template_version=prompt_bundle.template_version,
            definition_language_code=language.code,
        )
        result = write_distribution_sqlite_atomic(
            output_path=output_path,
            records=iter_distribution_records(
                settings=settings,
                curated_table=curated_table,
                llm_table=llm_table,
                models=models,
                prompt_bundle=prompt_bundle,
                progress_callback=None,
            ),
            metadata={
                "artifact_role": "distribution",
                "distribution_schema_version": DISTRIBUTION_SCHEMA_VERSION,
                "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
                "definition_language": language.as_dict(),
                "models": list(models) if models else None,
                "prompt_template_version": prompt_bundle.template_version,
                "prompt_version": prompt_bundle.resolved_prompt_version,
            },
            progress_callback=progress_callback,
        )
        emit_progress(
            progress_callback,
            stage=EXPORT_DISTRIBUTION_SQLITE_STAGE,
            event="export_complete",
            entry_count=result.entry_count,
            skipped_entries_without_meanings=result.skipped_entries_without_meanings,
            output_path=str(output_path),
            output_sha256=result.output_sha256,
        )

        with get_connection(settings) as conn:
            update_run_config(
                conn,
                run_id=run_id,
                config_updates={
                    "curated_run_ids": result.curated_run_ids,
                    "definition_run_ids": result.definition_run_ids,
                },
            )
            record_export_artifact(
                conn,
                artifact_table=artifact_table,
                run_id=run_id,
                artifact_type="distribution_sqlite",
                output_path=output_path,
                output_sha256=result.output_sha256,
                entry_count=result.entry_count,
                metadata={
                    "curated_table": curated_table,
                    "definitions_table": llm_table,
                    "models": list(models) if models else None,
                    "prompt_template_version": prompt_bundle.template_version,
                    "prompt_version": prompt_bundle.resolved_prompt_version,
                    "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                    "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
                    "artifact_role": "distribution",
                    "definition_language": language.as_dict(),
                    "curated_run_ids": result.curated_run_ids,
                    "definition_run_ids": result.definition_run_ids,
                    "skipped_entries_without_meanings": result.skipped_entries_without_meanings,
                },
            )
            complete_run(
                conn,
                run_id=run_id,
                stats={
                    "output_path": str(output_path),
                    "entry_count": result.entry_count,
                    "output_sha256": result.output_sha256,
                    "curated_run_ids": result.curated_run_ids,
                    "definition_run_ids": result.definition_run_ids,
                    "schema_version": DISTRIBUTION_SCHEMA_VERSION,
                    "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
                    "definition_language": language.as_dict(),
                    "skipped_entries_without_meanings": result.skipped_entries_without_meanings,
                },
            )

        return ExportSQLiteResult(
            run_id=run_id,
            output_path=Path(output_path),
            entry_count=result.entry_count,
            output_sha256=result.output_sha256,
        )
    except Exception as exc:
        with get_connection(settings) as conn:
            fail_run(conn, run_id=run_id, error=str(exc))
        raise


def write_distribution_sqlite_atomic(
    *,
    output_path: Path,
    records: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> SQLiteWriteResult:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = prepare_atomic_path(output_path)
    temp_path.unlink(missing_ok=True)

    reporter = ThrottledProgressReporter(progress_callback, stage=EXPORT_DISTRIBUTION_SQLITE_STAGE)
    curated_run_ids: set[str] = set()
    definition_run_ids: set[str] = set()
    entry_count = 0
    skipped_entries_without_meanings = 0
    processed_records = 0
    connection: sqlite3.Connection | None = None

    try:
        connection = sqlite3.connect(temp_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        _initialize_distribution_sqlite(connection)

        for record in records:
            processed_records += 1
            curated_run_id = record.get("curated_run_id")
            if curated_run_id:
                curated_run_ids.add(str(curated_run_id))
            definition_run_id = record.get("llm_run_id")
            if definition_run_id:
                definition_run_ids.add(str(definition_run_id))

            document = record.get("document")
            if document is None:
                skipped_entries_without_meanings += 1
                reporter.report(
                    event="export_progress",
                    processed_entries=processed_records,
                    exported_entries=entry_count,
                    skipped_entries_without_meanings=skipped_entries_without_meanings,
                )
                continue

            _insert_distribution_document(connection, document)
            entry_count += 1
            reporter.report(
                event="export_progress",
                processed_entries=processed_records,
                exported_entries=entry_count,
                skipped_entries_without_meanings=skipped_entries_without_meanings,
            )

        _insert_metadata(
            connection,
            metadata={
                **metadata,
                "entry_count": entry_count,
                "skipped_entries_without_meanings": skipped_entries_without_meanings,
                "curated_run_ids": sorted(curated_run_ids),
                "definition_run_ids": sorted(definition_run_ids),
            },
        )
        integrity_status = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_status is None or integrity_status[0] != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity_status}")
        connection.commit()
        connection.close()
        connection = None
        install_atomic_file(temp_path, output_path)
        output_sha256 = sha256_file(output_path)
        return SQLiteWriteResult(
            entry_count=entry_count,
            skipped_entries_without_meanings=skipped_entries_without_meanings,
            output_sha256=output_sha256,
            curated_run_ids=sorted(curated_run_ids),
            definition_run_ids=sorted(definition_run_ids),
        )
    except Exception:
        if connection is not None:
            connection.close()
        temp_path.unlink(missing_ok=True)
        raise


def _initialize_distribution_sqlite(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE entries (
            entry_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            headword TEXT NOT NULL,
            normalized_headword TEXT NOT NULL,
            headword_language_code TEXT NOT NULL,
            headword_language_name TEXT NOT NULL,
            definition_language_code TEXT NOT NULL,
            definition_language_name TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            headword_summary TEXT NOT NULL,
            memory_hook TEXT NOT NULL,
            etymology_note TEXT,
            study_notes_json TEXT NOT NULL,
            document_json TEXT NOT NULL
        );
        CREATE INDEX entries_lookup_idx
        ON entries (headword_language_code, normalized_headword);

        CREATE TABLE entry_study_notes (
            entry_id TEXT NOT NULL,
            note_index INTEGER NOT NULL,
            note_text TEXT NOT NULL,
            PRIMARY KEY (entry_id, note_index),
            FOREIGN KEY (entry_id) REFERENCES entries(entry_id) ON DELETE CASCADE
        );

        CREATE TABLE etymologies (
            entry_id TEXT NOT NULL,
            etymology_id TEXT NOT NULL,
            etymology_index INTEGER NOT NULL,
            text TEXT,
            pos_members_json TEXT NOT NULL,
            PRIMARY KEY (entry_id, etymology_id),
            FOREIGN KEY (entry_id) REFERENCES entries(entry_id) ON DELETE CASCADE
        );

        CREATE TABLE pos_groups (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            pos TEXT NOT NULL,
            etymology_id TEXT,
            summary TEXT NOT NULL,
            usage_note TEXT,
            PRIMARY KEY (entry_id, pos_group_index),
            FOREIGN KEY (entry_id) REFERENCES entries(entry_id) ON DELETE CASCADE
        );

        CREATE TABLE pos_group_forms (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            form_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            roman TEXT,
            PRIMARY KEY (entry_id, pos_group_index, form_index),
            FOREIGN KEY (entry_id, pos_group_index) REFERENCES pos_groups(entry_id, pos_group_index) ON DELETE CASCADE
        );

        CREATE TABLE pos_group_pronunciations (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            pronunciation_index INTEGER NOT NULL,
            ipa TEXT,
            text TEXT,
            tags_json TEXT NOT NULL,
            PRIMARY KEY (entry_id, pos_group_index, pronunciation_index),
            FOREIGN KEY (entry_id, pos_group_index) REFERENCES pos_groups(entry_id, pos_group_index) ON DELETE CASCADE
        );

        CREATE TABLE pos_group_relations (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            relation_index INTEGER NOT NULL,
            type TEXT NOT NULL,
            word TEXT NOT NULL,
            lang_code TEXT,
            PRIMARY KEY (entry_id, pos_group_index, relation_index),
            FOREIGN KEY (entry_id, pos_group_index) REFERENCES pos_groups(entry_id, pos_group_index) ON DELETE CASCADE
        );

        CREATE TABLE meanings (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            sense_id TEXT NOT NULL,
            meaning_index INTEGER NOT NULL,
            priority TEXT NOT NULL CHECK (priority IN ('core', 'common', 'rare')),
            short_gloss TEXT,
            learner_explanation TEXT NOT NULL,
            usage_note TEXT,
            labels_json TEXT NOT NULL,
            topics_json TEXT NOT NULL,
            PRIMARY KEY (entry_id, pos_group_index, sense_id),
            FOREIGN KEY (entry_id, pos_group_index) REFERENCES pos_groups(entry_id, pos_group_index) ON DELETE CASCADE
        );
        CREATE INDEX meanings_lookup_idx ON meanings (entry_id, pos_group_index, meaning_index);
        CREATE INDEX meanings_priority_idx ON meanings (entry_id, priority);

        CREATE TABLE meaning_examples (
            entry_id TEXT NOT NULL,
            pos_group_index INTEGER NOT NULL,
            sense_id TEXT NOT NULL,
            example_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            translation TEXT NOT NULL,
            PRIMARY KEY (entry_id, pos_group_index, sense_id, example_index),
            FOREIGN KEY (entry_id, pos_group_index, sense_id) REFERENCES meanings(entry_id, pos_group_index, sense_id) ON DELETE CASCADE
        );

        """
    )
    connection.execute("PRAGMA user_version = 1")


def _insert_distribution_document(connection: sqlite3.Connection, document: dict[str, Any]) -> None:
    entry_id = str(document["entry_id"])
    document_json = _json(document)

    connection.execute(
        """
        INSERT INTO entries (
            entry_id,
            schema_version,
            headword,
            normalized_headword,
            headword_language_code,
            headword_language_name,
            definition_language_code,
            definition_language_name,
            entry_type,
            headword_summary,
            memory_hook,
            etymology_note,
            study_notes_json,
            document_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry_id,
            document["schema_version"],
            document["headword"],
            document["normalized_headword"],
            document["headword_language"]["code"],
            document["headword_language"]["name"],
            document["definition_language"]["code"],
            document["definition_language"]["name"],
            document["entry_type"],
            document["headword_summary"],
            document["memory_hook"],
            document["etymology_note"],
            _json(document["study_notes"]),
            document_json,
        ),
    )

    connection.executemany(
        """
        INSERT INTO entry_study_notes (entry_id, note_index, note_text)
        VALUES (?, ?, ?)
        """,
        [
            (entry_id, note_index, note_text)
            for note_index, note_text in enumerate(document["study_notes"], start=1)
        ],
    )

    connection.executemany(
        """
        INSERT INTO etymologies (
            entry_id,
            etymology_id,
            etymology_index,
            text,
            pos_members_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                entry_id,
                etymology["etymology_id"],
                etymology_index,
                etymology.get("text"),
                _json(etymology.get("pos_members") or []),
            )
            for etymology_index, etymology in enumerate(document["etymologies"], start=1)
        ],
    )

    for pos_group_index, pos_group in enumerate(document["pos_groups"], start=1):
        connection.execute(
            """
            INSERT INTO pos_groups (
                entry_id,
                pos_group_index,
                pos,
                etymology_id,
                summary,
                usage_note
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                pos_group_index,
                pos_group["pos"],
                pos_group.get("etymology_id"),
                pos_group["summary"],
                pos_group.get("usage_note"),
            ),
        )

        connection.executemany(
            """
            INSERT INTO pos_group_forms (
                entry_id,
                pos_group_index,
                form_index,
                text,
                tags_json,
                roman
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry_id,
                    pos_group_index,
                    form_index,
                    form["text"],
                    _json(form.get("tags") or []),
                    form.get("roman"),
                )
                for form_index, form in enumerate(pos_group["forms"], start=1)
            ],
        )

        connection.executemany(
            """
            INSERT INTO pos_group_pronunciations (
                entry_id,
                pos_group_index,
                pronunciation_index,
                ipa,
                text,
                tags_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry_id,
                    pos_group_index,
                    pronunciation_index,
                    pronunciation.get("ipa"),
                    pronunciation.get("text"),
                    _json(pronunciation.get("tags") or []),
                )
                for pronunciation_index, pronunciation in enumerate(pos_group["pronunciations"], start=1)
            ],
        )

        connection.executemany(
            """
            INSERT INTO pos_group_relations (
                entry_id,
                pos_group_index,
                relation_index,
                type,
                word,
                lang_code
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry_id,
                    pos_group_index,
                    relation_index,
                    relation["type"],
                    relation["word"],
                    relation.get("lang_code"),
                )
                for relation_index, relation in enumerate(pos_group["relations"], start=1)
            ],
        )

        for meaning_index, meaning in enumerate(pos_group["meanings"], start=1):
            sense_id = meaning["sense_id"]
            connection.execute(
                """
                INSERT INTO meanings (
                    entry_id,
                    pos_group_index,
                    sense_id,
                    meaning_index,
                    priority,
                    short_gloss,
                    learner_explanation,
                    usage_note,
                    labels_json,
                    topics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    pos_group_index,
                    sense_id,
                    meaning_index,
                    meaning["priority"],
                    meaning.get("short_gloss"),
                    meaning["learner_explanation"],
                    meaning.get("usage_note"),
                    _json(meaning.get("labels") or []),
                    _json(meaning.get("topics") or []),
                ),
            )

            connection.executemany(
                """
                INSERT INTO meaning_examples (
                    entry_id,
                    pos_group_index,
                    sense_id,
                    example_index,
                    text,
                    translation
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry_id,
                        pos_group_index,
                        sense_id,
                        example_index,
                        example["text"],
                        example["translation"],
                    )
                    for example_index, example in enumerate(meaning["examples"], start=1)
                ],
            )



def _insert_metadata(connection: sqlite3.Connection, *, metadata: dict[str, Any]) -> None:
    connection.executemany(
        """
        INSERT INTO metadata (key, value_json)
        VALUES (?, ?)
        """,
        [
            (key, _json(value))
            for key, value in sorted(metadata.items())
        ],
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
