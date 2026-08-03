from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.prompt import PROMPT_VERSION, build_enrichment_request_payload, build_pos_group_id, build_prompt_bundle, compute_request_hash
from open_dictionary.pipeline.runs import start_run
from open_dictionary.stages.export_distribution_sqlite import stage as sqlite_stage


ENGLISH_DEFINITION_LANGUAGE = {
    "code": "en",
    "name": "English",
}


def seed_curated_entry(
    conn,
    *,
    word: str = "sophisticated",
    lang: str = "English",
    lang_code: str = "en",
) -> str:
    entry_id = str(uuid4())
    run_id = start_run(conn, stage="entries.assemble")
    payload = {
        "entry_id": entry_id,
        "word": word,
        "normalized_word": word.casefold(),
        "lang": lang,
        "lang_code": lang_code,
        "entry_flags": [],
        "source_summary": {
            "raw_record_count": 1,
            "raw_snapshot_ids": ["snapshot-1"],
            "raw_run_ids": ["run-1"],
            "raw_record_refs": [
                {
                    "snapshot_id": "snapshot-1",
                    "run_id": "run-1",
                    "raw_record_id": 1,
                    "source_line": 1,
                    "pos": "adj",
                }
            ],
        },
        "etymology_groups": [
            {
                "etymology_id": "et1",
                "etymology_text": "From Latin.",
                "etymology_flags": [],
                "member_pos": ["adj"],
                "source_refs": [{"raw_record_id": 1}],
            }
        ],
        "pos_groups": [
            {
                "pos": "adj",
                "pos_flags": [],
                "etymology_id": "et1",
                "senses": [
                    {
                        "sense_id": "s1",
                        "gloss": "complex or refined",
                        "raw_gloss": None,
                        "tags": ["formal"],
                        "qualifier": None,
                        "topics": ["style"],
                        "examples": [
                            {
                                "text": "a sophisticated system",
                                "translation": "一套复杂精密的系统",
                                "type": "example",
                                "ref": None,
                                "example_flags": [],
                            }
                        ],
                        "relations": [
                            {
                                "relation_type": "form_of",
                                "target_word": "sophisticate",
                                "target_lang_code": "en",
                                "relation_flags": [],
                                "source_scope": "sense",
                            }
                        ],
                        "sense_flags": [],
                    }
                ],
                "forms": [
                    {
                        "form": "more sophisticated",
                        "tags": ["comparative"],
                        "roman": None,
                        "form_flags": [],
                    }
                ],
                "pronunciations": [
                    {
                        "ipa": "/səˈfɪstɪkeɪtɪd/",
                        "pronunciation_text": None,
                        "audio_url": None,
                        "audio_format": None,
                        "tags": [],
                        "pronunciation_flags": [],
                    }
                ],
                "relations": [
                    {
                        "relation_type": "synonym",
                        "target_word": "refined",
                        "target_lang_code": "en",
                        "relation_flags": [],
                        "source_scope": "entry",
                    }
                ],
            }
        ],
    }
    with conn.cursor() as cursor:
        cursor.execute(
            """
            insert into curated.entries (
                run_id, entry_id, lang_code, normalized_word, word, payload, entry_flags, source_summary
            ) values (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
            """,
            (
                run_id,
                entry_id,
                lang_code,
                word.casefold(),
                word,
                json.dumps(payload),
                payload["entry_flags"],
                json.dumps(payload["source_summary"]),
            ),
        )
    return entry_id


def seed_llm_enrichment(
    conn,
    *,
    entry_id: str,
    payload: dict | None = None,
    model: str = "test-model",
    prompt_version: str = PROMPT_VERSION,
    definition_language: dict | None = None,
    input_hash: str | None = None,
) -> None:
    definition_language = definition_language or DEFAULT_DEFINITION_LANGUAGE.as_dict()
    prompt_bundle = build_prompt_bundle(
        prompt_version=prompt_version,
        definition_language=definition_language,
    )
    with conn.cursor() as cursor:
        cursor.execute("select payload from curated.entries where entry_id = %s", (entry_id,))
        curated_payload = cursor.fetchone()[0]
    resolved_input_hash = input_hash or compute_request_hash(
        build_enrichment_request_payload(curated_payload, prompt_bundle=prompt_bundle)
    )
    run_id = start_run(conn, stage="definitions.generate")
    with conn.cursor() as cursor:
        cursor.execute(
            """
            insert into llm.prompt_versions (
                prompt_version,
                prompt_text,
                output_contract,
                definition_language_code,
                definition_language_name,
                prompt_bundle
            )
            values (%s, %s, '{}'::jsonb, %s, %s, %s::jsonb)
            on conflict (prompt_version) do nothing
            """,
            (
                prompt_bundle.resolved_prompt_version,
                "prompt text",
                definition_language["code"],
                definition_language["name"],
                json.dumps(prompt_bundle.as_metadata()),
            ),
        )
        cursor.execute(
            """
            insert into llm.entry_enrichments (
                run_id, entry_id, model, prompt_version, definition_language_code, definition_language_name, input_hash, status,
                request_payload, response_payload, raw_response, error, retries
            ) values (
                %s, %s, %s, %s, %s, %s, %s, 'succeeded',
                '{}'::jsonb, %s, %s, null, 0
            )
            """,
            (
                run_id,
                entry_id,
                model,
                prompt_bundle.resolved_prompt_version,
                definition_language["code"],
                definition_language["name"],
                resolved_input_hash,
                json.dumps(payload or {}),
                json.dumps(payload or {}),
            ),
        )


def sqlite_metadata(connection: sqlite3.Connection) -> dict[str, object]:
    return {
        key: json.loads(value_json)
        for key, value_json in connection.execute("select key, value_json from metadata")
    }


def test_run_export_distribution_sqlite_stage_writes_output_and_manifest(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "distribution.sqlite"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            payload={
                "headword_summary": "整体说明。",
                "memory_hook": "一句帮助记忆的主线。",
                "study_notes": ["学习提示。"],
                "etymology_note": None,
                "pos_groups": [
                    {
                        "pos_group_id": build_pos_group_id(pos="adj", etymology_id="et1"),
                        "pos": "adj",
                        "summary": "形容词整体说明。",
                        "usage_note": None,
                        "meanings": [
                            {
                                "sense_id": "s1",
                                "priority": "core",
                                "short_gloss": "复杂精密的",
                                "learner_explanation": "详细解释。",
                                "usage_note": None,
                                "examples": [
                                    {"text": "A generated example sentence.", "translation": "一条生成的例句。"}
                                ],
                            }
                        ],
                    }
                ],
            },
        )
        conn.commit()

    result = sqlite_stage.run_export_distribution_sqlite_stage(
        settings=settings,
        output_path=output,
    )

    with sqlite3.connect(output) as connection:
        entry_row = connection.execute(
            """
            select headword, normalized_headword, definition_language_code, document_json
            from entries
            """
        ).fetchone()
        meanings_count = connection.execute("select count(*) from meanings").fetchone()[0]
        examples_count = connection.execute("select count(*) from meaning_examples").fetchone()[0]
        example_row = connection.execute("select text, translation from meaning_examples").fetchone()
        metadata = sqlite_metadata(connection)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select artifact_type, metadata->>'sqlite_schema_version', metadata->'curated_run_ids', metadata->'definition_run_ids'
                from export.artifacts
                where run_id = %s
                """,
                (result.run_id,),
            )
            artifact_type, sqlite_schema_version, curated_run_ids, definition_run_ids = cursor.fetchone()

    assert result.entry_count == 1
    assert entry_row[0] == "sophisticated"
    assert entry_row[1] == "sophisticated"
    assert entry_row[2] == "zh-Hans"
    assert json.loads(entry_row[3])["schema_version"] == "distribution_entry_v5"
    assert meanings_count == 1
    assert examples_count == 1
    assert example_row == ("A generated example sentence.", "一条生成的例句。")
    assert metadata["entry_count"] == 1
    assert metadata["distribution_schema_version"] == "distribution_entry_v5"
    assert metadata["sqlite_schema_version"] == "distribution_sqlite_v1"
    assert artifact_type == "distribution_sqlite"
    assert sqlite_schema_version == "distribution_sqlite_v1"
    assert curated_run_ids
    assert definition_run_ids


def test_run_export_distribution_sqlite_stage_selects_requested_definition_language(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "distribution-en.sqlite"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            definition_language=DEFAULT_DEFINITION_LANGUAGE.as_dict(),
            payload={
                "headword_summary": "中文整体说明。",
                "memory_hook": "一句帮助记忆的主线。",
                "study_notes": [],
                "etymology_note": None,
                "pos_groups": [
                    {
                        "pos_group_id": build_pos_group_id(pos="adj", etymology_id="et1"),
                        "pos": "adj",
                        "summary": "中文词性说明。",
                        "usage_note": None,
                        "meanings": [
                            {
                                "sense_id": "s1",
                                "priority": "core",
                                "short_gloss": "复杂",
                                "learner_explanation": "中文解释。",
                                "usage_note": None,
                            }
                        ],
                    }
                ],
            },
        )
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            definition_language=ENGLISH_DEFINITION_LANGUAGE,
            payload={
                "headword_summary": "English overall summary.",
                "memory_hook": "一句帮助记忆的主线。",
                "study_notes": ["English study note."],
                "etymology_note": None,
                "pos_groups": [
                    {
                        "pos_group_id": build_pos_group_id(pos="adj", etymology_id="et1"),
                        "pos": "adj",
                        "summary": "English adjective summary.",
                        "usage_note": None,
                        "meanings": [
                            {
                                "sense_id": "s1",
                                "priority": "core",
                                "short_gloss": "refined",
                                "learner_explanation": "Detailed English explanation.",
                                "usage_note": None,
                            }
                        ],
                    }
                ],
            },
        )
        conn.commit()

    result = sqlite_stage.run_export_distribution_sqlite_stage(
        settings=settings,
        output_path=output,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )

    with sqlite3.connect(output) as connection:
        row = connection.execute(
            """
            select definition_language_code, headword_summary
            from entries
            """
        ).fetchone()
        metadata = sqlite_metadata(connection)

    assert result.entry_count == 1
    assert row == ("en", "English overall summary.")
    assert metadata["definition_language"] == ENGLISH_DEFINITION_LANGUAGE
