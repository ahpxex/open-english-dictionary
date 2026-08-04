from __future__ import annotations

import json

import pytest

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.llm.client import LLMGenerationResult
from open_dictionary.qa.judge import validate_review_payload
from open_dictionary.qa.review_stage import run_quality_review_stage
from open_dictionary.qa.sampling import allocate_stratified_sample, build_stratum

from test_llm_enrich import FakeLLMClient, seed_curated_entry, valid_payload
from open_dictionary.llm.prompt import PROMPT_VERSION, build_enrichment_request_payload, build_prompt_bundle, compute_request_hash
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.stages.llm_enrich import stage as llm_stage


def _valid_review(verdict: str = "pass", issues: list | None = None) -> dict:
    return {
        "scores": {
            "accuracy": 5,
            "explanations": 4,
            "examples": 4,
            "usage_notes": 4,
            "chinese_quality": 5,
        },
        "priority_ok": True,
        "verdict": verdict,
        "issues": issues or [],
    }


def test_build_stratum_classifies_kinds_and_bands() -> None:
    assert build_stratum("water", [], 1) == "word|1"
    assert build_stratum("give up", [], 5) == "phrase|2-9"
    assert build_stratum("DNS", [], 3) == "abbreviation|2-9"
    assert build_stratum("London", ["entry_type:proper_name"], 2) == "proper_name|2-9"
    assert build_stratum("set", [], 98) == "word|sharded"


def test_allocate_stratified_sample_is_deterministic_and_weighted() -> None:
    population = (
        [{"entry_id": f"w{i}", "word": f"word{i}", "entry_flags": [], "sense_count": 1} for i in range(900)]
        + [{"entry_id": f"p{i}", "word": f"two words{i}", "entry_flags": [], "sense_count": 3} for i in range(90)]
        + [{"entry_id": f"g{i}", "word": f"giant{i}", "entry_flags": [], "sense_count": 40} for i in range(10)]
    )

    first = allocate_stratified_sample(population, sample_size=100, seed="s1")
    second = allocate_stratified_sample(population, sample_size=100, seed="s1")
    shuffled = allocate_stratified_sample(list(reversed(population)), sample_size=100, seed="s1")

    assert [e.entry_id for e in first] == [e.entry_id for e in second]
    assert [e.entry_id for e in first] == [e.entry_id for e in shuffled]

    by_stratum: dict[str, list] = {}
    for entry in first:
        by_stratum.setdefault(entry.stratum, []).append(entry)
    # every rare stratum gets full coverage or the floor, and weights invert
    # the sampling fraction so global estimates stay unbiased
    assert len(by_stratum["word|sharded"]) == 10
    assert by_stratum["word|sharded"][0].sampling_weight == 1.0
    assert len(by_stratum["phrase|2-9"]) >= 25
    total_weight = sum(e.sampling_weight for e in first)
    assert abs(total_weight - len(population)) < 1e-6


def test_validate_review_payload_normalizes_and_rejects() -> None:
    validated = validate_review_payload(_valid_review(issues=[{"kind": "weird", "note": "note text"}]))
    assert validated["issues"][0]["kind"] == "other"

    with pytest.raises(ValueError, match="verdict"):
        validate_review_payload({**_valid_review(), "verdict": "excellent"})
    with pytest.raises(ValueError, match="score"):
        validate_review_payload({**_valid_review(), "scores": {**_valid_review()["scores"], "accuracy": 7}})


def test_run_quality_review_stage_persists_reviews_and_resumes(
    tmp_path, temp_database_url: str
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API=http://localhost:3888/v1\nLLM_KEY=EMPTY\nLLM_MODEL=judge-model\n",
        encoding="utf-8",
    )

    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )
    with get_connection(settings) as conn:
        apply_foundation(conn)
        entry_id = seed_curated_entry(conn, word="cat")
        with conn.cursor() as cursor:
            cursor.execute("select payload from curated.entries where entry_id = %s", (entry_id,))
            curated_payload = cursor.fetchone()[0]
        llm_stage.ensure_prompt_version(conn, prompt_bundle=prompt_bundle)
        from open_dictionary.pipeline.runs import start_run
        gen_run = start_run(conn, stage="definitions.generate")
        request_payload = build_enrichment_request_payload(curated_payload, prompt_bundle=prompt_bundle)
        llm_stage.persist_enrichment_success(
            conn,
            target_table="llm.entry_enrichments",
            run_id=gen_run,
            record={
                "entry_id": entry_id,
                "model": "test-model",
                "prompt_version": prompt_bundle.resolved_prompt_version,
                "definition_language": DEFAULT_DEFINITION_LANGUAGE,
                "input_hash": compute_request_hash(request_payload),
                "request_payload": request_payload,
                "response_payload": valid_payload(),
                "raw_response": json.dumps(valid_payload()),
                "retries": 0,
            },
        )
        conn.commit()

    judge = FakeLLMClient([json.dumps(_valid_review("minor_issues", [{"kind": "unnatural_example", "location": "s1", "note": "stiff"}]))], model="judge-model")
    result = run_quality_review_stage(
        settings=settings,
        env_file=str(env_file),
        sample_size=10,
        seed="test-seed",
        max_workers=1,
        client=judge,
    )

    assert result.reviewed == 1
    assert result.failed == 0

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select verdict, stratum, sampling_weight, scores->>'accuracy', issues->0->>'kind' from llm.quality_reviews"
            )
            verdict, stratum, weight, accuracy, issue_kind = cursor.fetchone()
    assert verdict == "minor_issues"
    assert stratum == "word|1"
    assert float(weight) == 1.0
    assert accuracy == "5"
    assert issue_kind == "unnatural_example"

    # same seed resumes: nothing left to review
    rerun = run_quality_review_stage(
        settings=settings,
        env_file=str(env_file),
        sample_size=10,
        seed="test-seed",
        max_workers=1,
        client=FakeLLMClient([], model="judge-model"),
    )
    assert rerun.reviewed == 0
    assert rerun.skipped_existing == 1
