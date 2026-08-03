from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.client import LLMClientError, LLMGenerationResult, LiteLLMClient
from open_dictionary.llm.config import LLMProviderSettings, LLMSettings, load_llm_settings
from open_dictionary.llm.prompt import (
    PROMPT_VERSION,
    build_enrichment_request_payload,
    build_prompt_bundle,
    build_generation_source_payload,
    build_pos_group_id,
    build_user_prompt,
    compute_request_hash,
)
from open_dictionary.pipeline.runs import start_run
from open_dictionary.stages.llm_enrich import stage as llm_stage
from open_dictionary.stages.llm_enrich.schema import validate_enrichment_payload


class FakeLLMClient:
    def __init__(self, responses, *, model: str = "test-model"):
        self._responses = list(responses)
        self._model = model
        self.calls = 0
        self.max_tokens_seen: list[int | None] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMGenerationResult:
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMGenerationResult(
            content=response,
            model=self._model,
            api_base="http://127.0.0.1:3888/v1",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )


ENGLISH_DEFINITION_LANGUAGE = {
    "code": "en",
    "name": "English",
}


def valid_payload(
    pos: str = "noun",
    sense_ids: list[str] | None = None,
    etymology_id: str | None = None,
) -> dict:
    sense_ids = ["s1"] if sense_ids is None else sense_ids
    return {
        "headword_summary": "一个对中文学习者友好的整体说明。",
        "memory_hook": "一句帮助记忆的主线。",
        "study_notes": ["Note one", "Note two"],
        "etymology_note": "一个简短的词源说明。",
        "pos_groups": [
            {
                "pos_group_id": build_pos_group_id(pos=pos, etymology_id=etymology_id),
                "pos": pos,
                "summary": "这个词性的整体说明。",
                "usage_note": "Usage note.",
                "meanings": [
                    {
                        "sense_id": sense_id,
                        "priority": "core",
                        "short_gloss": f"{sense_id} short gloss",
                        "learner_explanation": f"{sense_id} 的详细自然语言解释。",
                        "usage_note": f"{sense_id} usage note.",
                        "examples": [
                            {"text": f"An example sentence for {sense_id}.", "translation": f"{sense_id} 的例句翻译。"}
                        ],
                    }
                    for sense_id in sense_ids
                ],
            }
        ],
    }


def seed_curated_entry(conn, *, word: str = "cat", lang_code: str = "en", pos: str = "noun") -> str:
    entry_id = str(uuid4())
    run_id = start_run(conn, stage="entries.assemble")
    payload = {
        "entry_id": entry_id,
        "word": word,
        "normalized_word": word,
        "lang": "English",
        "lang_code": lang_code,
        "entry_flags": [],
        "source_summary": {
            "raw_record_count": 1,
            "raw_snapshot_ids": ["snapshot-1"],
            "raw_run_ids": ["run-1"],
            "raw_record_refs": [{"snapshot_id": "snapshot-1", "run_id": "run-1", "raw_record_id": 1, "source_line": 1, "pos": pos}],
        },
        "etymology_groups": [],
        "pos_groups": [
            {
                "pos": pos,
                "pos_flags": [],
                "etymology_id": None,
                "senses": [
                    {
                        "sense_id": "s1",
                        "gloss": f"{word} gloss",
                        "raw_gloss": None,
                        "tags": [],
                        "qualifier": None,
                        "topics": [],
                        "examples": [],
                        "relations": [],
                        "sense_flags": [],
                    }
                ],
                "forms": [],
                "pronunciations": [],
                "relations": [],
            }
        ],
    }
    with conn.cursor() as cursor:
        cursor.execute(
            """
            insert into curated.entries (
                run_id, entry_id, lang_code, normalized_word, word, payload, entry_flags, source_summary
            ) values (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                entry_id,
                lang_code,
                word,
                word,
                json.dumps(payload),
                [],
                json.dumps(payload["source_summary"]),
            ),
        )
    return entry_id


def test_load_llm_settings_reads_values_from_env_file(tmp_path: Path) -> None:
    # This case verifies that the LLM stage can bootstrap itself from the repository .env.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://127.0.0.1:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )

    settings = load_llm_settings(env_file=env_file)

    assert len(settings.providers) == 1
    assert settings.providers[0].api_base == "http://127.0.0.1:3888/v1"
    assert settings.providers[0].api_key == "EMPTY"
    assert settings.providers[0].model == "test-model"
    assert settings.models == ("test-model",)


def test_load_llm_settings_raises_when_api_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This case protects the stage from starting with half-configured model credentials.
    monkeypatch.delenv("LLM_API", raising=False)
    monkeypatch.delenv("LLM_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="LLM_API"):
        load_llm_settings(env_file=env_file)


def test_load_llm_settings_reads_provider_pool_array(monkeypatch: pytest.MonkeyPatch) -> None:
    # This case verifies the multi-provider pool configuration contract.
    monkeypatch.setenv(
        "LLM_PROVIDERS",
        json.dumps(
            [
                {"api": "http://one.example/v1", "model": "model-one", "key": "key-one", "rpm": 120},
                {"api": "http://two.example/v1", "model": "model-two", "key": "key-two"},
            ]
        ),
    )

    settings = load_llm_settings(env_file=None)

    assert len(settings.providers) == 2
    assert settings.providers[0].api_base == "http://one.example/v1"
    assert settings.providers[0].model == "model-one"
    assert settings.providers[0].rpm == 120
    assert settings.providers[1].api_base == "http://two.example/v1"
    assert settings.providers[1].rpm is None
    assert settings.models == ("model-one", "model-two")


def test_load_llm_settings_provider_pool_loads_from_multiline_env_file(tmp_path: Path) -> None:
    # This case pins the dotenv format users will actually write: a quoted
    # multiline JSON array inside the model env file.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDERS='[\n"
        '  {"api": "http://one.example/v1", "model": "model-one", "key": "key-one"},\n'
        '  {"api": "http://two.example/v1", "model": "model-two", "key": "key-two", "rpm": 60}\n'
        "]'\n",
        encoding="utf-8",
    )

    settings = load_llm_settings(env_file=env_file)

    assert settings.models == ("model-one", "model-two")
    assert settings.providers[1].rpm == 60


def test_load_llm_settings_provider_pool_takes_precedence_over_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case pins the resolution order when both configuration styles are present.
    monkeypatch.setenv("LLM_API", "http://legacy.example/v1")
    monkeypatch.setenv("LLM_KEY", "legacy-key")
    monkeypatch.setenv("LLM_MODEL", "legacy-model")
    monkeypatch.setenv(
        "LLM_PROVIDERS",
        json.dumps([{"api": "http://pool.example/v1", "model": "pool-model", "key": "pool-key"}]),
    )

    settings = load_llm_settings(env_file=None)

    assert settings.models == ("pool-model",)


def test_load_llm_settings_rejects_incomplete_provider_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case prevents silently running with a half-configured provider entry.
    monkeypatch.setenv(
        "LLM_PROVIDERS",
        json.dumps([{"api": "http://one.example/v1", "key": "key-one"}]),
    )

    with pytest.raises(RuntimeError, match=r"LLM_PROVIDERS\[1\] is missing the required field 'model'"):
        load_llm_settings(env_file=None)


def test_load_llm_settings_rejects_unknown_provider_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case turns field typos into startup errors instead of silently dropped settings.
    monkeypatch.setenv(
        "LLM_PROVIDERS",
        json.dumps(
            [{"api": "http://one.example/v1", "model": "model-one", "key": "key-one", "apikey": "oops"}]
        ),
    )

    with pytest.raises(RuntimeError, match=r"LLM_PROVIDERS\[1\] contains unknown fields: apikey"):
        load_llm_settings(env_file=None)


def test_load_llm_settings_rejects_invalid_providers_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case keeps malformed pool configuration loud and immediate.
    monkeypatch.setenv("LLM_PROVIDERS", "not-json")

    with pytest.raises(RuntimeError, match="LLM_PROVIDERS is not valid JSON"):
        load_llm_settings(env_file=None)


def test_load_llm_settings_rejects_non_integer_rpm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This case keeps rate-limit configuration explicit and validated.
    monkeypatch.setenv(
        "LLM_PROVIDERS",
        json.dumps(
            [{"api": "http://one.example/v1", "model": "model-one", "key": "key-one", "rpm": "fast"}]
        ),
    )

    with pytest.raises(RuntimeError, match=r"LLM_PROVIDERS\[1\] field 'rpm' must be an integer"):
        load_llm_settings(env_file=None)


def test_build_user_prompt_embeds_curated_payload() -> None:
    # This case verifies that prompt rendering keeps the full curated structure available to the model.
    prompt = build_user_prompt(
        build_generation_source_payload(
            {"word": "cat", "lang": "English", "lang_code": "en", "pos_groups": []},
            definition_language=ENGLISH_DEFINITION_LANGUAGE,
        )
    )

    assert "Generated-field source payload" in prompt
    assert '"headword":"cat"' in prompt
    assert '"definition_language":{' in prompt
    assert '"code":"en"' in prompt


def test_build_prompt_bundle_resolves_language_specific_prompt_version() -> None:
    bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )

    assert bundle.template_version == PROMPT_VERSION
    assert bundle.resolved_prompt_version.endswith("__deflang__en")
    assert "English (en) is the required definition language" in bundle.system_prompt


def test_validate_enrichment_payload_accepts_valid_shape() -> None:
    # This case locks in the baseline enrichment contract.
    payload = valid_payload("noun")

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["headword_summary"] == payload["headword_summary"]


def test_validate_enrichment_payload_rejects_missing_keys() -> None:
    # This case prevents silently accepting incomplete LLM output.
    with pytest.raises(ValueError, match="missing required keys"):
        validate_enrichment_payload(
            {"headword_summary": "x"},
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_empty_headword_summary() -> None:
    # This case forces the model output to contain meaningful top-level text.
    payload = valid_payload()
    payload["headword_summary"] = "   "

    with pytest.raises(ValueError, match="headword_summary"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_invalid_study_notes() -> None:
    # This case guards against malformed list fields in the generated JSON.
    payload = valid_payload()
    payload["study_notes"] = ["good", 123]

    with pytest.raises(ValueError, match="study_notes"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_unexpected_pos() -> None:
    # This case prevents the model from inventing part-of-speech groups that do not exist in curated data.
    payload = valid_payload("verb")

    with pytest.raises(ValueError, match="unexpected pos_group_id"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_unexpected_sense_id() -> None:
    # This case prevents the model from inventing meanings outside the curated entry skeleton.
    payload = valid_payload("noun", sense_ids=["s2"])

    with pytest.raises(ValueError, match="unexpected sense_id"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_missing_sense_id() -> None:
    # This case ensures every curated sense receives a generated explanation.
    payload = valid_payload("noun", sense_ids=[])

    with pytest.raises(ValueError, match="missing sense_ids"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_coerces_string_study_notes() -> None:
    # This case captures a common model drift where a one-item list becomes a bare string.
    payload = valid_payload()
    payload["study_notes"] = "Single note"

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["study_notes"] == ["Single note"]


def test_validate_enrichment_payload_coerces_null_study_notes_to_empty_list() -> None:
    # This case handles compact fallback generations that choose null instead of [] for optional notes.
    payload = valid_payload()
    payload["study_notes"] = None

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["study_notes"] == []


def test_validate_enrichment_payload_coerces_usage_note_lists() -> None:
    # This case captures the exact schema drift seen during a real model smoke run.
    payload = valid_payload()
    payload["pos_groups"][0]["usage_note"] = ["First note.", "Second note."]
    payload["pos_groups"][0]["meanings"][0]["usage_note"] = ["Meaning note one.", "Meaning note two."]

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["pos_groups"][0]["usage_note"] == "First note. Second note."
    assert validated["pos_groups"][0]["meanings"][0]["usage_note"] == "Meaning note one. Meaning note two."


def test_compute_input_hash_is_stable() -> None:
    # This case ensures repeated enrichment over the same request payload gets the same hash.
    payload = {"entry": {"word": "cat"}, "prompt_version": "v1"}

    first = llm_stage.compute_input_hash(payload)
    second = llm_stage.compute_input_hash({"prompt_version": "v1", "entry": {"word": "cat"}})

    assert first == second


def test_enrich_one_entry_succeeds_with_fake_client() -> None:
    # This case covers the happy-path single-entry enrichment flow.
    client = FakeLLMClient([json.dumps(valid_payload("noun"))])
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    request_entry = build_generation_source_payload(
        {
            "entry_id": "entry-1",
            "word": "cat",
            "normalized_word": "cat",
            "lang": "English",
            "lang_code": "en",
            "entry_flags": [],
            "etymology_groups": [],
            "pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}],
        }
    )
    entry = {
        "entry_id": "entry-1",
        "payload": {"pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}]},
        "request_payload": {"entry": request_entry},
        "input_hash": "hash",
    }

    record = llm_stage.enrich_one_entry(
        entry=entry,
        llm_client=client,
        prompt_bundle=prompt_bundle,
        max_retries=2,
    )

    assert record["entry_id"] == "entry-1"
    assert record["model"] == "test-model"
    assert record["retries"] == 0
    assert record["generation_metadata"]["usage"]["completion_tokens"] == 200
    assert client.calls == 1


def test_enrich_one_entry_retries_before_success() -> None:
    # This case verifies that transient model failures are retried instead of immediately failing the stage.
    client = FakeLLMClient([ValueError("bad"), json.dumps(valid_payload("noun"))])
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    request_entry = build_generation_source_payload(
        {
            "entry_id": "entry-1",
            "word": "cat",
            "normalized_word": "cat",
            "lang": "English",
            "lang_code": "en",
            "entry_flags": [],
            "etymology_groups": [],
            "pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}],
        }
    )
    entry = {
        "entry_id": "entry-1",
        "payload": {"pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}]},
        "request_payload": {"entry": request_entry},
        "input_hash": "hash",
    }

    record = llm_stage.enrich_one_entry(
        entry=entry,
        llm_client=client,
        prompt_bundle=prompt_bundle,
        max_retries=2,
    )

    assert record["retries"] == 1
    assert client.calls == 2
    assert client.max_tokens_seen == [llm_stage.DEFAULT_MAX_TOKENS, llm_stage.COMPACT_RETRY_MAX_TOKENS]


def test_enrich_one_entry_raises_after_max_retries() -> None:
    # This case ensures persistent invalid outputs bubble up as failures.
    client = FakeLLMClient([ValueError("bad"), ValueError("still bad")])
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    request_entry = build_generation_source_payload(
        {
            "entry_id": "entry-1",
            "word": "cat",
            "normalized_word": "cat",
            "lang": "English",
            "lang_code": "en",
            "entry_flags": [],
            "etymology_groups": [],
            "pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}],
        }
    )
    entry = {
        "entry_id": "entry-1",
        "payload": {"pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}]},
        "request_payload": {"entry": request_entry},
        "input_hash": "hash",
    }

    with pytest.raises(llm_stage.EnrichmentError, match="still bad"):
        llm_stage.enrich_one_entry(
            entry=entry,
            llm_client=client,
            prompt_bundle=prompt_bundle,
            max_retries=2,
        )


def test_litellm_client_wraps_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # This case ensures transport-level failures enter the stage retry path instead of escaping as raw exceptions.
    client = LiteLLMClient(
        LLMSettings(
            providers=(
                LLMProviderSettings(
                    api_base="http://127.0.0.1:3888/v1",
                    api_key="EMPTY",
                    model="test-model",
                ),
            )
        )
    )

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(client._router, "completion", raise_timeout)

    with pytest.raises(LLMClientError, match="timed out"):
        client.generate_json(system_prompt="system", user_prompt="user", max_tokens=100)


def test_ensure_prompt_version_is_idempotent(temp_database_url: str) -> None:
    # This case prevents duplicate prompt metadata rows for the same version string.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version="v-test",
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        with conn.cursor() as cursor:
            cursor.execute(
                "select count(*) from llm.prompt_versions where prompt_version = %s",
                (prompt_bundle.resolved_prompt_version,),
            )
            count = cursor.fetchone()[0]

    assert count == 1


def test_iter_curated_entries_skips_existing_successes(temp_database_url: str) -> None:
    # This case verifies that reruns do not keep sending already-enriched entries back to the model.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        with conn.cursor() as cursor:
            cursor.execute("select payload from curated.entries where entry_id = %s", (entry_id,))
            curated_payload = cursor.fetchone()[0]
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into meta.pipeline_runs (run_id, stage, status, config)
                values (gen_random_uuid(), 'definitions.generate', 'succeeded', '{}'::jsonb)
                returning run_id
                """
            )
            run_id = cursor.fetchone()[0]
        llm_stage.persist_enrichment_success(
            conn,
            target_table="llm.entry_enrichments",
            run_id=run_id,
            record={
                "entry_id": entry_id,
                "model": "test-model",
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": DEFAULT_DEFINITION_LANGUAGE,
                "input_hash": compute_request_hash(
                    build_enrichment_request_payload(curated_payload, prompt_bundle=prompt_bundle)
                ),
                "request_payload": build_enrichment_request_payload(curated_payload, prompt_bundle=prompt_bundle),
                "response_payload": valid_payload(),
                "raw_response": json.dumps(valid_payload()),
                "retries": 0,
            },
        )

    items = list(
        llm_stage.iter_curated_entries(
            settings,
            source_table="curated.entries",
            target_table="llm.entry_enrichments",
            prompt_bundle=prompt_bundle,
            models=["test-model"],
            recompute_existing=False,
            limit_entries=None,
        )
    )

    assert items == []


def test_iter_curated_entries_does_not_skip_stale_successes(temp_database_url: str) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into meta.pipeline_runs (run_id, stage, status, config)
                values (gen_random_uuid(), 'definitions.generate', 'succeeded', '{}'::jsonb)
                returning run_id
                """
            )
            run_id = cursor.fetchone()[0]
        llm_stage.persist_enrichment_success(
            conn,
            target_table="llm.entry_enrichments",
            run_id=run_id,
            record={
                "entry_id": entry_id,
                "model": "test-model",
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": DEFAULT_DEFINITION_LANGUAGE,
                "input_hash": "stale-hash",
                "request_payload": {"entry": "payload"},
                "response_payload": valid_payload(),
                "raw_response": json.dumps(valid_payload()),
                "retries": 0,
            },
        )

    items = list(
        llm_stage.iter_curated_entries(
            settings,
            source_table="curated.entries",
            target_table="llm.entry_enrichments",
            prompt_bundle=prompt_bundle,
            models=["test-model"],
            recompute_existing=False,
            limit_entries=None,
        )
    )

    assert len(items) == 1


def test_iter_curated_entries_recompute_existing_returns_entries(temp_database_url: str) -> None:
    # This case covers the explicit rebuild path for LLM output regeneration.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn)

    items = list(
        llm_stage.iter_curated_entries(
            settings,
            source_table="curated.entries",
            target_table="llm.entry_enrichments",
            prompt_bundle=prompt_bundle,
            models=["test-model"],
            recompute_existing=True,
            limit_entries=None,
        )
    )

    assert len(items) == 1


def test_persist_enrichment_success_stores_success_row(temp_database_url: str) -> None:
    # This case ensures successful generations are durably persisted for export.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        run_id = start_run(conn, stage="definitions.generate")
        llm_stage.persist_enrichment_success(
            conn,
            target_table="llm.entry_enrichments",
            run_id=run_id,
            record={
                "entry_id": entry_id,
                "model": "test-model",
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": DEFAULT_DEFINITION_LANGUAGE,
                "input_hash": "hash",
                "request_payload": {"entry": "payload"},
                "response_payload": valid_payload(),
                "raw_response": json.dumps(valid_payload()),
                "retries": 0,
            },
        )
        with conn.cursor() as cursor:
            cursor.execute("select status, model from llm.entry_enrichments")
            status, model = cursor.fetchone()

    assert status == "succeeded"
    assert model == "test-model"


def test_persist_enrichment_failure_stores_error_row(temp_database_url: str) -> None:
    # This case ensures failed generations are visible for debugging and retries.
    settings = RuntimeSettings(database_url=temp_database_url)
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        run_id = start_run(conn, stage="definitions.generate")
        llm_stage.persist_enrichment_failure(
            conn,
            target_table="llm.entry_enrichments",
            run_id=run_id,
            entry_id=entry_id,
            model="test-model",
            prompt_version=prompt_bundle.resolved_prompt_version,
            definition_language=DEFAULT_DEFINITION_LANGUAGE,
            input_hash="hash",
            request_payload={"entry": "payload"},
            retries=3,
            error="boom",
        )
        with conn.cursor() as cursor:
            cursor.execute("select status, error from llm.entry_enrichments")
            status, error = cursor.fetchone()

    assert status == "failed"
    assert error == "boom"


def test_run_llm_enrich_stage_records_upstream_curated_run_ids(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://127.0.0.1:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )

    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "select run_id::text from curated.entries where entry_id = %s",
                (entry_id,),
            )
            curated_run_id = cursor.fetchone()[0]

    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=FakeLLMClient([json.dumps(valid_payload("noun"))]),
        max_workers=1,
    )

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select config->'source_run_ids', stats->'source_run_ids'
                from meta.pipeline_runs
                where run_id = %s
                """,
                (result.run_id,),
            )
            source_run_ids, stats_source_run_ids = cursor.fetchone()

    assert source_run_ids == [curated_run_id]
    assert stats_source_run_ids == [curated_run_id]


def test_run_llm_enrich_stage_processes_curated_entries_with_fake_client(tmp_path: Path, temp_database_url: str) -> None:
    # This case drives the entire stage end to end without a live model service.
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")
    settings = RuntimeSettings(database_url=temp_database_url)
    client = FakeLLMClient([json.dumps(valid_payload())])

    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn)

    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=client,
        max_workers=1,
    )

    assert result.processed == 1
    assert result.succeeded == 1
    assert result.failed == 0


def test_run_llm_enrich_stage_supports_non_default_definition_language(
    tmp_path: Path,
    temp_database_url: str,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")
    settings = RuntimeSettings(database_url=temp_database_url)
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "headword_summary": "Overall learner-facing summary.",
                    "memory_hook": "一句帮助记忆的主线。",
                    "study_notes": ["Keep register in mind."],
                    "etymology_note": "Short etymology note.",
                    "pos_groups": [
                        {
                            "pos_group_id": build_pos_group_id(pos="noun", etymology_id=None),
                            "pos": "noun",
                            "summary": "Noun summary.",
                            "usage_note": None,
                            "meanings": [
                                {
                                    "sense_id": "s1",
                                    "priority": "core",
                                    "short_gloss": "cat",
                                    "learner_explanation": "A domestic feline animal.",
                                    "usage_note": None,
                                }
                            ],
                        }
                    ],
                }
            )
        ]
    )

    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn)

    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=client,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
        max_workers=1,
    )

    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=ENGLISH_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select definition_language_code, definition_language_name, prompt_version
                from llm.entry_enrichments
                """
            )
            stored_language_code, stored_language_name, stored_prompt_version = cursor.fetchone()

    assert result.succeeded == 1
    assert stored_language_code == "en"
    assert stored_language_name == "English"
    assert stored_prompt_version == prompt_bundle.resolved_prompt_version


def test_run_llm_enrich_stage_records_failed_entries(tmp_path: Path, temp_database_url: str) -> None:
    # This case verifies the stage keeps going and stores failures when the model output never validates.
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")
    settings = RuntimeSettings(database_url=temp_database_url)
    client = FakeLLMClient([ValueError("bad output")])

    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn)

    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=client,
        max_workers=1,
        max_retries=1,
    )

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select count(*) from llm.entry_enrichments where status = 'failed'")
            failed_count = cursor.fetchone()[0]

    assert result.failed == 1
    assert failed_count == 1


def test_run_llm_enrich_stage_respects_limit_entries(tmp_path: Path, temp_database_url: str) -> None:
    # This case supports debugger-friendly partial LLM runs over larger curated tables.
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")
    settings = RuntimeSettings(database_url=temp_database_url)
    client = FakeLLMClient([json.dumps(valid_payload()), json.dumps(valid_payload())])

    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")
        seed_curated_entry(conn, word="dog")

    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=client,
        max_workers=1,
        limit_entries=1,
    )

    assert result.processed == 1
    assert client.calls == 1


def test_run_llm_enrich_stage_recompute_existing_enriches_again(tmp_path: Path, temp_database_url: str) -> None:
    # This case verifies the explicit rebuild mode for enrichment rows.
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=test-model\n", encoding="utf-8")
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        seed_curated_entry(conn, word="cat")

    first_client = FakeLLMClient([json.dumps(valid_payload())])
    second_client = FakeLLMClient([json.dumps(valid_payload())])

    llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=first_client,
        max_workers=1,
    )
    result = llm_stage.run_llm_enrich_stage(
        settings=settings,
        env_file=str(env_file),
        client=second_client,
        max_workers=1,
        recompute_existing=True,
    )

    assert result.processed == 1
    assert second_client.calls == 1


def test_validate_enrichment_payload_rejects_missing_memory_hook() -> None:
    # This case pins the v4 contract: every entry must carry a memory hook.
    payload = valid_payload()
    del payload["memory_hook"]

    with pytest.raises(ValueError, match="memory_hook"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_invalid_priority() -> None:
    # This case keeps the priority enum closed so clients can rely on it.
    payload = valid_payload()
    payload["pos_groups"][0]["meanings"][0]["priority"] = "important"

    with pytest.raises(ValueError, match="priority"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_normalizes_priority_case() -> None:
    # This case tolerates harmless model drift like "Core" without widening the enum.
    payload = valid_payload()
    payload["pos_groups"][0]["meanings"][0]["priority"] = " Core "

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["pos_groups"][0]["meanings"][0]["priority"] == "core"


def test_validate_enrichment_payload_defaults_missing_examples_to_empty_list() -> None:
    # This case keeps the compact retry path valid: examples may be absent.
    payload = valid_payload()
    del payload["pos_groups"][0]["meanings"][0]["examples"]

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["pos_groups"][0]["meanings"][0]["examples"] == []


def test_validate_enrichment_payload_rejects_example_without_translation() -> None:
    # This case pins the v5 contract: generated examples must be bilingual pairs.
    payload = valid_payload()
    payload["pos_groups"][0]["meanings"][0]["examples"] = [{"text": "Only source side."}]

    with pytest.raises(ValueError, match="examples.translation"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def test_validate_enrichment_payload_rejects_replacement_characters() -> None:
    # This case turns corrupted model output (U+FFFD) into a retryable failure
    # instead of silently persisting broken text.
    payload = valid_payload()
    payload["pos_groups"][0]["meanings"][0]["usage_note"] = "若表示�某物打结"

    with pytest.raises(ValueError, match="replacement characters"):
        validate_enrichment_payload(
            payload,
            expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
        )


def _sharded_source_payload(n_verb_senses: int = 90, n_noun_senses: int = 10) -> dict:
    return build_generation_source_payload(
        {
            "entry_id": "entry-big",
            "word": "set",
            "normalized_word": "set",
            "lang": "English",
            "lang_code": "en",
            "entry_flags": [],
            "etymology_groups": [],
            "pos_groups": [
                {
                    "pos": "verb",
                    "etymology_id": None,
                    "senses": [
                        {"sense_id": f"v{i}", "gloss": f"verb gloss {i}"}
                        for i in range(1, n_verb_senses + 1)
                    ],
                },
                {
                    "pos": "noun",
                    "etymology_id": None,
                    "senses": [
                        {"sense_id": f"n{i}", "gloss": f"noun gloss {i}"}
                        for i in range(1, n_noun_senses + 1)
                    ],
                },
            ],
        }
    )


def test_plan_generation_chunks_keeps_small_groups_whole_and_splits_giants() -> None:
    # This case pins the chunk planner: whole groups pack together, only
    # over-budget groups split, and skeleton order is preserved.
    source = _sharded_source_payload(n_verb_senses=90, n_noun_senses=10)

    chunks = llm_stage.plan_generation_chunks(source, budget=40)

    sizes = [sum(len(g["meanings"]) for g in chunk) for chunk in chunks]
    assert sizes == [40, 40, 10, 10]
    assert [g["pos"] for chunk in chunks for g in chunk] == ["verb", "verb", "verb", "noun"]
    assert all(
        sum(len(g["meanings"]) for g in chunk) <= 40 for chunk in chunks
    )
    all_ids = [m["sense_id"] for chunk in chunks for g in chunk for m in g["meanings"]]
    assert all_ids == [f"v{i}" for i in range(1, 91)] + [f"n{i}" for i in range(1, 11)]


class ShardAwareFakeLLMClient:
    """Answers overview and chunk prompts with contract-valid payloads."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int | None = None) -> LLMGenerationResult:
        if user_prompt.startswith("Entry digest"):
            self.calls.append("overview")
            response = {
                "core_senses": [
                    {"pos_group_id": "verb|_", "sense_id": "v1"},
                    {"pos_group_id": "noun|_", "sense_id": "n1"},
                ],
                "headword_summary": "整体说明。",
                "memory_hook": "一句记忆主线。",
                "study_notes": [],
                "etymology_note": None,
            }
        else:
            marker = "Partial-entry source payload (JSON):\n"
            payload = json.loads(user_prompt.split(marker, 1)[1])
            self.calls.append("chunk")
            groups = []
            for group in payload["pos_groups"]:
                groups.append(
                    {
                        "pos_group_id": group["pos_group_id"],
                        "pos": group["pos"],
                        "summary": f"{group['pos']} 概述。",
                        "usage_note": None,
                        "meanings": [
                            {
                                "sense_id": m["sense_id"],
                                "priority": "core" if m["sense_id"] == "v2" else "common",
                                "short_gloss": None,
                                "learner_explanation": f"{m['sense_id']} 的解释。",
                                "usage_note": None,
                                "examples": [
                                    {"text": f"Example {m['sense_id']}.", "translation": f"{m['sense_id']} 例句。"}
                                ],
                            }
                            for m in group["meanings"]
                        ],
                    }
                )
            response = {"pos_groups": groups}
        return LLMGenerationResult(
            content=json.dumps(response, ensure_ascii=False),
            model="test-model",
            api_base="http://127.0.0.1:3888/v1",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )


def test_enrich_one_entry_shards_large_entries_and_assembles_full_payload() -> None:
    # This case drives the engineered long-entry path end to end: overview
    # call, chunked generation, deterministic assembly, full-skeleton check.
    client = ShardAwareFakeLLMClient()
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    source = _sharded_source_payload(n_verb_senses=90, n_noun_senses=10)
    entry = {
        "entry_id": "entry-big",
        "payload": {},
        "request_payload": {"entry": source},
        "input_hash": "hash-big",
    }

    record = llm_stage.enrich_one_entry(
        entry=entry,
        llm_client=client,
        prompt_bundle=prompt_bundle,
        max_retries=2,
    )

    expected_chunks = len(llm_stage.plan_generation_chunks(source, budget=llm_stage.CHUNK_SENSE_BUDGET))
    assert client.calls == ["overview"] + ["chunk"] * expected_chunks
    payload = record["response_payload"]
    assert payload["memory_hook"] == "一句记忆主线。"
    verb_group = payload["pos_groups"][0]
    assert len(verb_group["meanings"]) == 90
    assert verb_group["summary"] == "verb 概述。"
    priorities = {
        m["sense_id"]: m["priority"]
        for group in payload["pos_groups"]
        for m in group["meanings"]
    }
    assert priorities["v1"] == "core"
    assert priorities["n1"] == "core"
    assert priorities["v2"] == "common"
    assert sum(1 for value in priorities.values() if value == "core") == 2
    assert record["generation_metadata"]["core_senses"] == [
        {"pos_group_id": "verb|_", "sense_id": "v1"},
        {"pos_group_id": "noun|_", "sense_id": "n1"},
    ]
    assert record["generation_metadata"]["sharded"] is True
    assert record["generation_metadata"]["chunk_count"] == expected_chunks
    assert record["generation_metadata"]["usage"]["completion_tokens"] == (expected_chunks + 1) * 100
    assert record["retries"] == 0
    assert record["model"] == "test-model"


def test_enrich_one_entry_small_entries_stay_single_call() -> None:
    # This case protects the fast path: small entries never shard.
    client = FakeLLMClient([json.dumps(valid_payload("noun"))])
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    request_entry = build_generation_source_payload(
        {
            "entry_id": "entry-1",
            "word": "cat",
            "normalized_word": "cat",
            "lang": "English",
            "lang_code": "en",
            "entry_flags": [],
            "etymology_groups": [],
            "pos_groups": [{"pos": "noun", "etymology_id": None, "senses": [{"sense_id": "s1"}]}],
        }
    )
    entry = {
        "entry_id": "entry-1",
        "payload": {},
        "request_payload": {"entry": request_entry},
        "input_hash": "hash",
    }

    record = llm_stage.enrich_one_entry(
        entry=entry,
        llm_client=client,
        prompt_bundle=prompt_bundle,
        max_retries=2,
    )

    assert client.calls == 1
    assert "sharded" not in record["generation_metadata"]


def test_assemble_sharded_payload_enforces_pair_keyed_nominations() -> None:
    # Regression: sense_ids repeat across pos groups (every group restarts at
    # s1), so nominations must bind to (pos_group_id, sense_id) pairs — an
    # id-only match once inflated 4 nominations into 17 core senses.
    overview = {
        "headword_summary": "整体说明。",
        "memory_hook": "主线。",
        "study_notes": [],
        "etymology_note": None,
    }
    def meaning(sense_id, priority):
        return {
            "sense_id": sense_id,
            "priority": priority,
            "short_gloss": None,
            "learner_explanation": "解释。",
            "usage_note": None,
            "examples": [],
        }
    chunk_groups = [[
        {"pos_group_id": "verb|_", "pos": "verb", "summary": "V。", "usage_note": None,
         "meanings": [meaning("s1", "core"), meaning("s2", "core")]},
        {"pos_group_id": "noun|_", "pos": "noun", "summary": "N。", "usage_note": None,
         "meanings": [meaning("s1", "common")]},
    ]]
    source = {"pos_groups": [
        {"pos_group_id": "verb|_", "meanings": [{"sense_id": "s1"}, {"sense_id": "s2"}]},
        {"pos_group_id": "noun|_", "meanings": [{"sense_id": "s1"}]},
    ]}

    assembled = llm_stage.assemble_sharded_payload(
        overview_fields=overview,
        chunk_group_lists=chunk_groups,
        source_payload=source,
        core_senses=[{"pos_group_id": "verb|_", "sense_id": "s1"}],
    )

    priorities = {
        (g["pos_group_id"], m["sense_id"]): m["priority"]
        for g in assembled["pos_groups"] for m in g["meanings"]
    }
    assert priorities[("verb|_", "s1")] == "core"
    assert priorities[("verb|_", "s2")] == "common"
    assert priorities[("noun|_", "s1")] == "common"


def test_validate_pos_groups_normalizes_translated_pos_echo() -> None:
    # Regression: models occasionally translate the redundant pos echo
    # ("noun" -> "名词") at temperature 0; alignment is carried by
    # pos_group_id, so the echo is normalized instead of failing the call.
    payload = valid_payload("noun")
    payload["pos_groups"][0]["pos"] = "名词"

    validated = validate_enrichment_payload(
        payload,
        expected_pos_targets=[{"pos_group_id": build_pos_group_id(pos="noun", etymology_id=None), "pos": "noun", "sense_ids": ["s1"]}],
    )

    assert validated["pos_groups"][0]["pos"] == "noun"
