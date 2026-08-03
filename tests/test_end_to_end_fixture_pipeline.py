from __future__ import annotations

import json
from pathlib import Path

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.client import LLMGenerationResult
from open_dictionary.stages.export_distribution_jsonl.schema import validate_distribution_document
from open_dictionary.stages.curated_build.stage import run_curated_build_stage
from open_dictionary.stages.export_distribution_jsonl.stage import run_export_distribution_jsonl_stage
from open_dictionary.stages.export_jsonl.stage import run_export_jsonl_stage
from open_dictionary.stages.llm_enrich.stage import run_llm_enrich_stage
from open_dictionary.stages.raw_ingest.stage import run_raw_ingest_stage


ENGLISH_DEFINITION_LANGUAGE = {
    "code": "en",
    "name": "English",
}


class FixtureAwareFakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMGenerationResult:
        self.calls += 1
        marker = "Generated-field source payload (JSON):\n"
        payload = json.loads(user_prompt.split(marker, 1)[1])
        definition_language_code = payload.get("definition_language", {}).get("code")
        pos_groups = []
        for group in payload.get("pos_groups", []):
            if definition_language_code == "en":
                summary = f"{group['pos']} overview."
                learner_explanation = "{sense_id} detailed English explanation."
                usage_note = None
            else:
                summary = f"{group['pos']} 的整体说明。"
                learner_explanation = "{sense_id} 的详细中文解释。"
                usage_note = None
            pos_groups.append(
                {
                    "pos_group_id": group["pos_group_id"],
                    "pos": group["pos"],
                    "summary": summary,
                    "usage_note": None,
                    "meanings": [
                        {
                            "sense_id": meaning["sense_id"],
                            "priority": "core",
                            "short_gloss": f"{meaning['sense_id']} short gloss",
                            "learner_explanation": learner_explanation.format(sense_id=meaning["sense_id"]),
                            "usage_note": usage_note,
                            "examples": [
                                {
                                    "text": f"Example sentence for {meaning['sense_id']}.",
                                    "translation": f"{meaning['sense_id']} 的例句翻译。",
                                }
                            ],
                        }
                        for meaning in group.get("meanings", [])
                    ],
                }
            )
        if definition_language_code == "en":
            response = {
                "headword_summary": f"Overall English summary for {payload['headword']}.",
                "memory_hook": "一句帮助记忆的主线。",
                "etymology_note": None,
                "study_notes": [f"Study the usage of {payload['headword']} in context."],
                "pos_groups": pos_groups,
            }
        else:
            response = {
                "headword_summary": f"{payload['headword']} 的整体中文说明。",
                "memory_hook": "一句帮助记忆的主线。",
                "etymology_note": None,
                "study_notes": [f"学习 {payload['headword']} 时要注意语境。"],
                "pos_groups": pos_groups,
            }
        return LLMGenerationResult(
            content=json.dumps(response, ensure_ascii=False),
            model="test-model",
            api_base="http://localhost:3888/v1",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )


def test_fixture_pipeline_runs_end_to_end_with_fake_llm(tmp_path: Path, temp_database_url: str) -> None:
    # This case drives the full rewritten pipeline over the repository fixture:
    # raw ingest -> curated build -> llm enrich -> audit jsonl export.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings(database_url=temp_database_url)
    fixture_path = Path("fixtures/wiktionary/raw.jsonl")
    output_path = tmp_path / "audit.jsonl"
    fake_client = FixtureAwareFakeLLMClient()

    with get_connection(settings) as conn:
        apply_foundation(conn)

    raw_result = run_raw_ingest_stage(
        settings=settings,
        workdir=tmp_path / "raw-workdir",
        archive_path=fixture_path,
    )
    curated_result = run_curated_build_stage(settings=settings)
    llm_result = run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=fake_client,
        max_workers=4,
    )
    export_result = run_export_jsonl_stage(
        settings=settings,
        output_path=output_path,
        include_unenriched=False,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert raw_result.rows_loaded == 1000
    assert curated_result.entries_written > 0
    assert llm_result.succeeded == curated_result.entries_written
    assert export_result.entry_count == curated_result.entries_written
    assert len(lines) == export_result.entry_count
    assert fake_client.calls == llm_result.processed


def test_fixture_pipeline_export_rows_contain_entries_and_definitions_sections(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    # This case verifies the shape of the merged audit documents produced by the
    # rewritten end-to-end fixture pipeline.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings(database_url=temp_database_url)
    fixture_path = Path("fixtures/wiktionary/raw.jsonl")
    output_path = tmp_path / "audit.jsonl"

    with get_connection(settings) as conn:
        apply_foundation(conn)

    run_raw_ingest_stage(
        settings=settings,
        workdir=tmp_path / "raw-workdir",
        archive_path=fixture_path,
    )
    run_curated_build_stage(settings=settings, limit_groups=25)
    run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=FixtureAwareFakeLLMClient(),
        max_workers=2,
    )
    run_export_jsonl_stage(
        settings=settings,
        output_path=output_path,
        include_unenriched=False,
    )

    first_doc = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert "entries" in first_doc
    assert "definitions" in first_doc
    assert "pos_groups" in first_doc["entries"]
    assert "headword_summary" in first_doc["definitions"]["payload"]


def test_fixture_pipeline_distribution_export_rows_have_learner_facing_shape(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings(database_url=temp_database_url)
    fixture_path = Path("fixtures/wiktionary/raw.jsonl")
    output_path = tmp_path / "distribution.jsonl"

    with get_connection(settings) as conn:
        apply_foundation(conn)

    run_raw_ingest_stage(
        settings=settings,
        workdir=tmp_path / "raw-workdir",
        archive_path=fixture_path,
    )
    run_curated_build_stage(settings=settings, limit_groups=25)
    run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=FixtureAwareFakeLLMClient(),
        max_workers=2,
    )
    run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output_path,
    )

    first_doc = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_doc["schema_version"] == "distribution_entry_v4"
    assert "entries" not in first_doc
    assert "definitions" not in first_doc
    assert "headword_summary" in first_doc
    assert "definition_language" in first_doc
    assert "learner_explanation" in first_doc["pos_groups"][0]["meanings"][0]


def test_fixture_pipeline_distribution_export_supports_english_definition_language(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings(database_url=temp_database_url)
    fixture_path = Path("fixtures/wiktionary/raw.jsonl")
    output_path = tmp_path / "distribution-en.jsonl"

    with get_connection(settings) as conn:
        apply_foundation(conn)

    run_raw_ingest_stage(
        settings=settings,
        workdir=tmp_path / "raw-workdir",
        archive_path=fixture_path,
    )
    run_curated_build_stage(settings=settings, limit_groups=25)
    run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=FixtureAwareFakeLLMClient(),
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
        max_workers=2,
    )
    run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output_path,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )

    first_doc = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_doc["definition_language"] == ENGLISH_DEFINITION_LANGUAGE
    assert first_doc["headword_summary"].startswith("Overall English summary")
    assert "English explanation" in first_doc["pos_groups"][0]["meanings"][0]["learner_explanation"]


def test_fixture_pipeline_distribution_export_validates_all_rows_on_full_fixture(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings(database_url=temp_database_url)
    fixture_path = Path("fixtures/wiktionary/raw.jsonl")
    output_path = tmp_path / "distribution-full.jsonl"

    with get_connection(settings) as conn:
        apply_foundation(conn)

    raw_result = run_raw_ingest_stage(
        settings=settings,
        workdir=tmp_path / "raw-workdir",
        archive_path=fixture_path,
    )
    curated_result = run_curated_build_stage(settings=settings)
    llm_result = run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=FixtureAwareFakeLLMClient(),
        max_workers=4,
    )
    export_result = run_export_distribution_jsonl_stage(
        settings=settings,
        output_path=output_path,
    )

    documents = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert raw_result.rows_loaded == 1000
    assert curated_result.entries_written == 742
    assert llm_result.succeeded == 742
    assert 0 < export_result.entry_count <= 742
    assert len(documents) == export_result.entry_count
    for document in documents:
        validate_distribution_document(document)
