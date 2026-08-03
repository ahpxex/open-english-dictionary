from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.prompt import build_enrichment_request_payload, build_prompt_bundle, compute_request_hash
from open_dictionary.pipeline.runs import start_run
from open_dictionary.stages.export_jsonl import stage as export_stage


ENGLISH_DEFINITION_LANGUAGE = {
    "code": "en",
    "name": "English",
}


def seed_curated_entry(
    conn,
    *,
    word: str,
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
            "raw_record_refs": [{"snapshot_id": "snapshot-1", "run_id": "run-1", "raw_record_id": 1, "source_line": 1, "pos": "noun"}],
        },
        "etymology_groups": [],
        "pos_groups": [{"pos": "noun", "pos_flags": [], "etymology_id": None, "senses": [], "forms": [], "pronunciations": [], "relations": []}],
    }
    with conn.cursor() as cursor:
        cursor.execute(
            """
            insert into curated.entries (
                run_id, entry_id, lang_code, normalized_word, word, payload, entry_flags, source_summary
            ) values (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
            """,
            (run_id, entry_id, lang_code, word.casefold(), word, json.dumps(payload), [], json.dumps(payload["source_summary"])),
        )
    return entry_id


def seed_llm_enrichment(
    conn,
    *,
    entry_id: str,
    model: str = "test-model",
    prompt_version: str = "prompt-v1",
    definition_language: dict | None = None,
    created_offset: int = 0,
    payload: dict | None = None,
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
                prompt_version, prompt_text, output_contract,
                definition_language_code, definition_language_name, prompt_bundle
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
                request_payload, response_payload, raw_response, error, retries, created_at, updated_at
            ) values (
                %s, %s, %s, %s, %s, %s, %s, %s,
                '{}'::jsonb, %s, %s, null, 0,
                now() + (%s || ' seconds')::interval,
                now() + (%s || ' seconds')::interval
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
                json.dumps(payload or {"headword_summary": "headword summary"}),
                json.dumps(payload or {"headword_summary": "headword summary"}),
                created_offset,
                created_offset,
            ),
        )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_write_jsonl_atomic_creates_output_file(tmp_path: Path) -> None:
    # This case covers the basic export write path.
    output = tmp_path / "out.jsonl"
    digest = export_stage.write_jsonl_atomic(output, [{"word": "cat"}])

    assert output.exists()
    assert digest
    assert read_jsonl(output) == [{"word": "cat"}]


def test_write_jsonl_atomic_overwrites_existing_output(tmp_path: Path) -> None:
    # This case ensures rebuilds replace stale artifacts instead of appending to them.
    output = tmp_path / "out.jsonl"
    output.write_text('{"old":true}\n', encoding="utf-8")

    export_stage.write_jsonl_atomic(output, [{"word": "cat"}])

    assert read_jsonl(output) == [{"word": "cat"}]


def test_identifier_from_dotted_rejects_blank_names() -> None:
    # This case protects all SQL builders in the export stage.
    with pytest.raises(ValueError, match="Identifier name cannot be empty"):
        export_stage.identifier_from_dotted(" . ")


def test_iter_export_documents_includes_llm_payload_when_present(temp_database_url: str) -> None:
    # This case verifies the merged audit document shape on the happy path.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id)
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))

    assert len(docs) == 1
    assert docs[0]["definitions"] is not None
    assert docs[0]["entries"]["word"] == "cat"


def test_iter_export_documents_can_include_unenriched_entries(temp_database_url: str) -> None:
    # This case ensures export can still emit raw curated output before LLM is finished.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))

    assert len(docs) == 1
    assert docs[0]["definitions"] is None


def test_iter_export_documents_can_exclude_unenriched_entries(temp_database_url: str) -> None:
    # This case supports a strict audit export mode that excludes unenriched rows.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=False,
    ))

    assert docs == []


def test_iter_export_documents_filters_by_model(temp_database_url: str) -> None:
    # This case makes model-specific export selection explicit.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id, model="model-a")
        seed_llm_enrichment(conn, entry_id=entry_id, model="model-b")
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=["model-b"],
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=False,
    ))

    assert docs[0]["definitions"]["model"] == "model-b"


def test_iter_export_documents_filters_by_prompt_version(temp_database_url: str) -> None:
    # This case keeps prompt-versioned exports reproducible.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version="prompt-b",
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id, prompt_version="prompt-a")
        seed_llm_enrichment(conn, entry_id=entry_id, prompt_version="prompt-b")
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=prompt_bundle,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=False,
    ))

    assert docs[0]["definitions"]["prompt_version"] == prompt_bundle.resolved_prompt_version


def test_iter_export_documents_filters_by_definition_language(temp_database_url: str) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            definition_language=DEFAULT_DEFINITION_LANGUAGE.as_dict(),
            payload={"headword_summary": "中文摘要"},
        )
        seed_llm_enrichment(
            conn,
            entry_id=entry_id,
            definition_language=ENGLISH_DEFINITION_LANGUAGE,
            payload={"headword_summary": "English summary"},
        )
        conn.commit()

    docs = list(
        export_stage.iter_export_documents(
            settings=settings,
            curated_table="curated.entries",
            llm_table="llm.entry_enrichments",
            models=None,
            prompt_bundle=None,
            definition_language=ENGLISH_DEFINITION_LANGUAGE,
            include_unenriched=False,
        )
    )

    assert docs[0]["definitions"]["definition_language"]["code"] == "en"
    assert docs[0]["definitions"]["payload"]["headword_summary"] == "English summary"


def test_iter_export_documents_selects_latest_successful_enrichment(temp_database_url: str) -> None:
    # This case ensures the export stage picks the newest successful enrichment rather than an older stale row.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id, created_offset=0, payload={"headword_summary": "old"})
        seed_llm_enrichment(conn, entry_id=entry_id, created_offset=5, payload={"headword_summary": "new"})
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=False,
    ))

    assert docs[0]["definitions"]["payload"]["headword_summary"] == "new"


def test_iter_export_documents_ignores_failed_enrichments(temp_database_url: str) -> None:
    # This case prevents broken LLM rows from leaking into the audit artifact.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id, status="failed", payload={"headword_summary": "bad"})
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))

    assert docs[0]["definitions"] is None


def test_iter_export_documents_orders_by_lang_and_normalized_word(temp_database_url: str) -> None:
    # This case makes export ordering deterministic.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="zeta", lang_code="en")
        seed_curated_entry(conn, word="alpha", lang_code="en")
        seed_curated_entry(conn, word="bonjour", lang_code="fr", lang="French")
        conn.commit()

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))

    assert [(doc["lang_code"], doc["normalized_word"]) for doc in docs] == [
        ("en", "alpha"),
        ("en", "zeta"),
        ("fr", "bonjour"),
    ]


def test_record_export_artifact_persists_metadata(temp_database_url: str, tmp_path: Path) -> None:
    # This case verifies the export artifact manifest table.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        run_id = start_run(conn, stage="audit.export")
        export_stage.record_export_artifact(
            conn,
            artifact_table="export.artifacts",
            run_id=run_id,
            artifact_type="audit_jsonl",
            output_path=tmp_path / "out.jsonl",
            output_sha256="sha",
            entry_count=3,
            metadata={"model": "test-model"},
        )
        with conn.cursor() as cursor:
            cursor.execute("select artifact_type, entry_count, metadata->>'model' from export.artifacts")
            artifact_type, entry_count, model = cursor.fetchone()

    assert artifact_type == "audit_jsonl"
    assert entry_count == 3
    assert model == "test-model"


def test_run_export_jsonl_stage_writes_output_and_manifest(temp_database_url: str, tmp_path: Path) -> None:
    # This case covers the end-to-end audit export path with both curated and LLM content present.
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "audit.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        seed_llm_enrichment(conn, entry_id=entry_id)
        conn.commit()

    result = export_stage.run_export_jsonl_stage(settings=settings, output_path=output)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select metadata->'curated_run_ids', metadata->'definition_run_ids' from export.artifacts where run_id = %s",
                (result.run_id,),
            )
            curated_run_ids, definition_run_ids = cursor.fetchone()

    assert result.entry_count == 1
    assert output.exists()
    assert read_jsonl(output)[0]["word"] == "cat"
    assert curated_run_ids
    assert definition_run_ids


def test_run_export_jsonl_stage_can_emit_only_enriched_entries(temp_database_url: str, tmp_path: Path) -> None:
    # This case exercises the strict audit export path where unenriched entries are suppressed.
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "audit.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    result = export_stage.run_export_jsonl_stage(
        settings=settings,
        output_path=output,
        include_unenriched=False,
    )

    assert result.entry_count == 0
    assert read_jsonl(output) == []


def test_run_export_jsonl_stage_records_pipeline_run_success(temp_database_url: str, tmp_path: Path) -> None:
    # This case verifies audit export runs are visible in pipeline metadata.
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "audit.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    result = export_stage.run_export_jsonl_stage(settings=settings, output_path=output)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select status from meta.pipeline_runs where run_id = %s", (result.run_id,))
            status = cursor.fetchone()[0]

    assert status == "succeeded"


def test_run_export_jsonl_stage_marks_run_failed_when_write_crashes(
    temp_database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case ensures audit export failures do not leave ambiguous running pipeline rows.
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "audit.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    monkeypatch.setattr(export_stage, "write_jsonl_atomic", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    with pytest.raises(RuntimeError, match="disk full"):
        export_stage.run_export_jsonl_stage(settings=settings, output_path=output)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select status, error from meta.pipeline_runs where stage = 'audit.export' order by started_at desc limit 1"
            )
            status, error = cursor.fetchone()

    assert status == "failed"
    assert "disk full" in error


def test_run_export_jsonl_stage_sha_changes_with_content(temp_database_url: str, tmp_path: Path) -> None:
    # This case makes the audit artifact hash meaningful by verifying different exports produce different digests.
    settings = RuntimeSettings(database_url=temp_database_url)
    output_one = tmp_path / "one.jsonl"
    output_two = tmp_path / "two.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    first = export_stage.run_export_jsonl_stage(settings=settings, output_path=output_one)

    with get_connection(settings) as conn:
        seed_curated_entry(conn, word="dog")
        conn.commit()

    second = export_stage.run_export_jsonl_stage(settings=settings, output_path=output_two)

    assert first.output_sha256 != second.output_sha256


def test_export_document_contains_expected_top_level_keys(temp_database_url: str) -> None:
    # This case locks in the merged audit document contract for downstream consumers.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    document = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))[0]

    assert set(document.keys()) == {"entry_id", "lang_code", "normalized_word", "word", "entries", "definitions"}


def test_export_document_uses_null_llm_when_missing(temp_database_url: str) -> None:
    # This case makes the missing-enrichment shape explicit in the audit JSONL.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    document = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))[0]

    assert document["definitions"] is None


def test_export_document_embeds_curated_payload_verbatim(temp_database_url: str) -> None:
    # This case protects curated content from being silently reshaped during audit export.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        conn.commit()

    document = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))[0]

    assert document["entries"]["word"] == "cat"
    assert document["entries"]["normalized_word"] == "cat"


def test_write_jsonl_atomic_produces_one_line_per_document(tmp_path: Path) -> None:
    # This case ensures the audit export helper always emits valid JSON Lines rather than one giant JSON blob.
    output = tmp_path / "out.jsonl"
    export_stage.write_jsonl_atomic(output, [{"word": "cat"}, {"word": "dog"}])

    assert output.read_text(encoding="utf-8").count("\n") == 2


def test_iter_export_documents_returns_empty_when_no_curated_rows_exist(temp_database_url: str) -> None:
    # This case protects the empty-database audit export path.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)

    docs = list(export_stage.iter_export_documents(
        settings=settings,
        curated_table="curated.entries",
        llm_table="llm.entry_enrichments",
        models=None,
        prompt_bundle=None,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
        include_unenriched=True,
    ))

    assert docs == []


def test_record_export_artifact_tracks_entry_count(temp_database_url: str, tmp_path: Path) -> None:
    # This case verifies the manifest count field independently from the stage wrapper.
    settings = RuntimeSettings(database_url=temp_database_url)
    with get_connection(settings) as conn:
        apply_foundation(conn)
        run_id = start_run(conn, stage="audit.export")
        export_stage.record_export_artifact(
            conn,
            artifact_table="export.artifacts",
            run_id=run_id,
            artifact_type="audit_jsonl",
            output_path=tmp_path / "out.jsonl",
            output_sha256="sha",
            entry_count=7,
            metadata={},
        )
        with conn.cursor() as cursor:
            cursor.execute("select entry_count from export.artifacts")
            entry_count = cursor.fetchone()[0]

    assert entry_count == 7


def test_run_export_jsonl_stage_manifest_matches_file_line_count(temp_database_url: str, tmp_path: Path) -> None:
    # This case ensures audit artifact metadata and emitted content stay in sync.
    settings = RuntimeSettings(database_url=temp_database_url)
    output = tmp_path / "audit.jsonl"
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        seed_curated_entry(conn, word="dog")
        conn.commit()

    result = export_stage.run_export_jsonl_stage(settings=settings, output_path=output)

    assert result.entry_count == len(read_jsonl(output))
