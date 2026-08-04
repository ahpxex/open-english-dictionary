from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.prompt import PROMPT_VERSION, build_enrichment_request_payload, build_pos_group_id, build_prompt_bundle, compute_request_hash
from open_dictionary.pipeline.runs import start_run
from open_dictionary.stages.export_distribution_jsonl.schema import validate_distribution_document
from open_dictionary.stages.export_distribution_jsonl import stage as distribution_stage


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
    entry_flags: list[str] | None = None,
    pos_groups: list[dict] | None = None,
    etymology_groups: list[dict] | None = None,
) -> str:
    entry_id = str(uuid4())
    run_id = start_run(conn, stage="entries.assemble")
    payload = {
        "entry_id": entry_id,
        "word": word,
        "normalized_word": word.casefold(),
        "lang": lang,
        "lang_code": lang_code,
        "entry_flags": entry_flags or [],
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
        "etymology_groups": etymology_groups
        or [
            {
                "etymology_id": "et1",
                "etymology_text": "From Latin.",
                "etymology_flags": [],
                "member_pos": ["adj"],
                "source_refs": [{"raw_record_id": 1}],
            }
        ],
        "pos_groups": pos_groups
        or [
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
    status: str = "succeeded",
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
                %s, %s, %s, %s, %s, %s, %s, %s,
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
                status,
                json.dumps(payload or {}),
                json.dumps(payload or {}),
            ),
        )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_distribution_document_merges_curated_and_llm_fields() -> None:
    curated_payload = {
        "entry_id": "entry-1",
        "word": "sophisticated",
        "normalized_word": "sophisticated",
        "lang": "English",
        "lang_code": "en",
        "entry_flags": [],
        "etymology_groups": [
            {
                "etymology_id": "et1",
                "etymology_text": "From Latin.",
                "member_pos": ["adj"],
            }
        ],
        "pos_groups": [
            {
                "pos": "adj",
                "etymology_id": "et1",
                "forms": [{"form": "more sophisticated", "tags": ["comparative"], "roman": None}],
                "pronunciations": [{"ipa": "/x/", "pronunciation_text": None, "audio_url": None, "tags": []}],
                "relations": [{"relation_type": "synonym", "target_word": "refined", "target_lang_code": "en"}],
                "senses": [
                    {
                        "sense_id": "s1",
                        "tags": ["formal"],
                        "topics": ["style"],
                        "examples": [{"text": "a sophisticated system", "translation": "一套复杂精密的系统", "ref": None, "type": "example"}],
                        "relations": [{"relation_type": "form_of", "target_word": "sophisticate", "target_lang_code": "en"}],
                    }
                ],
            }
        ],
    }
    llm_payload = {
        "headword_summary": "这是一个面向中文学习者的整体说明。",
        "memory_hook": "一句帮助记忆的主线。",
        "study_notes": ["不要机械翻译成“复杂的”。"],
        "etymology_note": "带有成熟、精细的语感。",
        "pos_groups": [
            {
                "pos_group_id": build_pos_group_id(pos="adj", etymology_id="et1"),
                "pos": "adj",
                "summary": "形容词整体说明。",
                "usage_note": "要结合搭配理解。",
                "meanings": [
                    {
                        "sense_id": "s1",
                        "priority": "core",
                        "short_gloss": "复杂精密的；老练的",
                        "learner_explanation": "这里是详细的中文自然语言解释。",
                        "usage_note": "这是该义项的用法说明。",
                        "examples": [
                            {"text": "She has sophisticated taste in music.", "translation": "她的音乐品味很老练。"}
                        ],
                    }
                ],
            }
        ],
    }

    document = distribution_stage.build_distribution_document(
        curated_payload=curated_payload,
        llm_payload=llm_payload,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )

    assert document["schema_version"] == "distribution_entry_v5"
    assert document["headword"] == "sophisticated"
    assert document["definition_language"]["code"] == "zh-Hans"
    assert "entries" not in document
    assert "definitions" not in document
    assert document["pos_groups"][0]["meanings"][0]["learner_explanation"] == "这里是详细的中文自然语言解释。"
    meaning = document["pos_groups"][0]["meanings"][0]
    assert meaning["examples"] == [
        {"text": "She has sophisticated taste in music.", "translation": "她的音乐品味很老练。"}
    ]
    assert "citations" not in meaning
    assert "relations" not in meaning
    assert "pos_group_id" not in document["pos_groups"][0]
    assert validate_distribution_document(document) == document


def test_build_distribution_document_distinguishes_same_pos_across_etymologies() -> None:
    curated_payload = {
        "entry_id": "entry-1",
        "word": "bank",
        "normalized_word": "bank",
        "lang": "English",
        "lang_code": "en",
        "entry_flags": [],
        "etymology_groups": [
            {"etymology_id": "et1", "etymology_text": "land edge", "member_pos": ["noun"]},
            {"etymology_id": "et2", "etymology_text": "financial institution", "member_pos": ["noun"]},
        ],
        "pos_groups": [
            {
                "pos": "noun",
                "etymology_id": "et1",
                "forms": [],
                "pronunciations": [],
                "relations": [],
                "senses": [{"sense_id": "s1", "tags": [], "topics": [], "examples": [], "relations": []}],
            },
            {
                "pos": "noun",
                "etymology_id": "et2",
                "forms": [],
                "pronunciations": [],
                "relations": [],
                "senses": [{"sense_id": "s1", "tags": [], "topics": [], "examples": [], "relations": []}],
            },
        ],
    }
    llm_payload = {
        "headword_summary": "bank 有两个常见来源。",
        "memory_hook": "一句帮助记忆的主线。",
        "study_notes": [],
        "etymology_note": None,
        "pos_groups": [
            {
                "pos_group_id": build_pos_group_id(pos="noun", etymology_id="et1"),
                "pos": "noun",
                "summary": "河岸义项说明。",
                "usage_note": None,
                "meanings": [
                    {"sense_id": "s1", "priority": "core", "short_gloss": "河岸", "learner_explanation": "与河流边缘有关。", "usage_note": None}
                ],
            },
            {
                "pos_group_id": build_pos_group_id(pos="noun", etymology_id="et2"),
                "pos": "noun",
                "summary": "金融机构义项说明。",
                "usage_note": None,
                "meanings": [
                    {"sense_id": "s1", "priority": "core", "short_gloss": "银行", "learner_explanation": "与金融机构有关。", "usage_note": None}
                ],
            },
        ],
    }

    document = distribution_stage.build_distribution_document(
        curated_payload=curated_payload,
        llm_payload=llm_payload,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )

    assert document["definition_language"]["code"] == "en"
    assert document["pos_groups"][0]["summary"] == "河岸义项说明。"
    assert document["pos_groups"][1]["summary"] == "金融机构义项说明。"


def test_run_export_distribution_jsonl_stage_writes_output_and_manifest(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "distribution.jsonl"
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
                            }
                        ],
                    }
                ],
            },
        )
        conn.commit()

    result = distribution_stage.run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output,
    )

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select artifact_type, metadata->>'schema_version', metadata->'curated_run_ids', metadata->'definition_run_ids' from export.artifacts where run_id = %s",
                (result.run_id,),
            )
            artifact_type, schema_version, curated_run_ids, definition_run_ids = cursor.fetchone()

    rows = read_jsonl(output)

    assert result.entry_count == 1
    assert rows[0]["schema_version"] == "distribution_entry_v5"
    assert rows[0]["headword"] == "sophisticated"
    assert artifact_type == "distribution_jsonl"
    assert schema_version == "distribution_entry_v5"
    assert curated_run_ids
    assert definition_run_ids


def test_run_export_distribution_jsonl_stage_selects_requested_definition_language(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "distribution-en.jsonl"
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

    result = distribution_stage.run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )

    rows = read_jsonl(output)

    assert result.entry_count == 1
    assert rows[0]["definition_language"]["code"] == "en"
    assert rows[0]["headword_summary"] == "English overall summary."


def test_distribution_export_excludes_stale_enrichment_payloads(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "distribution.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            input_hash="stale-hash",
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
                            }
                        ],
                    }
                ],
            },
        )
        conn.commit()

    result = distribution_stage.run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output,
    )

    # A stale enrichment no longer matches the current curated payload, so the
    # entry is excluded from the artifact and counted, never silently exported.
    assert result.entry_count == 0
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select metadata->>'skipped_unenriched' from export.artifacts where run_id = %s",
                (result.run_id,),
            )
            assert cursor.fetchone()[0] == "1"


def test_distribution_export_skips_unenriched_entries_with_accounting(
    temp_database_url: str,
    tmp_path: Path,
) -> None:
    # Persistent generation failures must not block the artifact: entries
    # without a matching enrichment are excluded and counted explicitly.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        enriched_id = seed_curated_entry(conn, word="alpha")
        seed_curated_entry(conn, word="beta")
        seed_llm_enrichment(
            conn,
            entry_id=enriched_id,
            payload={
                "headword_summary": "整体说明。",
                "memory_hook": "一句帮助记忆的主线。",
                "study_notes": [],
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
                                "short_gloss": None,
                                "learner_explanation": "详细解释。",
                                "usage_note": None,
                            }
                        ],
                    }
                ],
            },
        )
        conn.commit()

    output = tmp_path / "distribution.jsonl"
    result = distribution_stage.run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output,
    )

    assert result.entry_count == 1
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select metadata->>'skipped_unenriched' from export.artifacts where run_id = %s",
                (result.run_id,),
            )
            skipped = cursor.fetchone()[0]
    assert skipped == "1"
