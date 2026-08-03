from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from open_dictionary import cli
from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE
from open_dictionary.llm.prompt import PROMPT_VERSION, build_prompt_bundle


@pytest.mark.parametrize(
    ("argv", "patches", "expected_command", "expected_checks"),
    [
        (
            ["init-db"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "get_connection": None,
                "apply_foundation": lambda conn: ["20260403_curated_lineage_v2"],
            },
            "init-db",
            {"applied_versions": ["20260403_curated_lineage_v2"]},
        ),
        (
            ["fetch-snapshot", "--output", "data/raw/sample.jsonl.gz"],
            {
                "download_wiktionary_dump": lambda output, **kwargs: Path(output),
            },
            "fetch-snapshot",
            {"output_path": "data/raw/sample.jsonl.gz"},
        ),
        (
            ["unpack-snapshot", "--input", "data/raw/sample.jsonl.gz", "--output", "data/raw/sample.jsonl"],
            {
                "extract_wiktionary_dump": lambda input_path, output, **kwargs: Path(output),
            },
            "unpack-snapshot",
            {"output_path": "data/raw/sample.jsonl"},
        ),
        (
            ["ingest-snapshot", "--archive-path", "fixtures/wiktionary/raw.jsonl"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_raw_ingest_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    snapshot_id=uuid4(),
                    archive_path=Path("fixtures/wiktionary/raw.jsonl"),
                    rows_loaded=1000,
                    anomalies_logged=0,
                    archive_sha256="sha",
                    snapshot_preexisting=False,
                    resumed_from_run_id=None,
                ),
            },
            "ingest-snapshot",
            {"rows_loaded": 1000},
        ),
        (
            ["assemble-entries"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_curated_build_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    groups_processed=742,
                    groups_filtered_out=0,
                    entries_written=742,
                    relations_written=10,
                    triage_written=1,
                ),
            },
            "assemble-entries",
            {"entries_written": 742},
        ),
        (
            ["generate-definitions"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_llm_enrich_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    processed=742,
                    succeeded=742,
                    failed=0,
                ),
            },
            "generate-definitions",
            {"succeeded": 742},
        ),
        (
            ["export-audit", "--output", "data/export/audit.jsonl"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_export_audit_jsonl_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    entry_count=742,
                    output_path=Path("data/export/audit.jsonl"),
                    output_sha256="audit-sha",
                ),
            },
            "export-audit",
            {"entry_count": 742},
        ),
        (
            ["export-distribution", "--output", "data/export/distribution.jsonl"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_export_distribution_jsonl_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    entry_count=741,
                    output_path=Path("data/export/distribution.jsonl"),
                    output_sha256="dist-sha",
                ),
            },
            "export-distribution",
            {"entry_count": 741},
        ),
        (
            ["export-distribution-sqlite", "--output", "data/export/distribution.sqlite"],
            {
                "load_settings": lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
                "run_export_distribution_sqlite_stage": lambda **kwargs: SimpleNamespace(
                    run_id=uuid4(),
                    entry_count=741,
                    output_path=Path("data/export/distribution.sqlite"),
                    output_sha256="sqlite-sha",
                ),
            },
            "export-distribution-sqlite",
            {"entry_count": 741, "sqlite_schema_version": "distribution_sqlite_v1"},
        ),
    ],
)
def test_cli_commands_emit_consistent_json_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    patches: dict[str, object],
    expected_command: str,
    expected_checks: dict[str, object],
) -> None:
    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    for name, value in patches.items():
        if value is None and name == "get_connection":
            monkeypatch.setattr(cli, "get_connection", lambda settings: DummyConnection())
        else:
            monkeypatch.setattr(cli, name, value)

    exit_code = cli.main(argv)

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["command"] == expected_command
    assert summary["status"] == "succeeded"
    for key, value in expected_checks.items():
        assert summary[key] == value


def test_pipeline_run_executes_stages_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}
    workflow_run_id = uuid4()
    prompt_bundle = build_prompt_bundle(
        prompt_version=PROMPT_VERSION,
        definition_language=DEFAULT_DEFINITION_LANGUAGE,
    )

    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_raw(**kwargs):
        calls["source"] = kwargs
        kwargs["progress_callback"]({"stage": "source.ingest", "event": "acquire_complete", "rows_loaded": 0})
        return SimpleNamespace(
            run_id=uuid4(),
            snapshot_id=uuid4(),
            rows_loaded=1000,
            anomalies_logged=0,
            snapshot_preexisting=False,
            archive_sha256="sha",
            resumed_from_run_id=None,
        )

    def fake_curated(**kwargs):
        calls["entries"] = kwargs
        kwargs["progress_callback"]({"stage": "entries.assemble", "event": "build_progress", "groups_processed": 10})
        return SimpleNamespace(
            run_id=uuid4(),
            groups_processed=742,
            groups_filtered_out=0,
            entries_written=742,
            relations_written=10,
            triage_written=1,
        )

    def fake_llm(**kwargs):
        calls["definitions"] = kwargs
        kwargs["progress_callback"]({"stage": "definitions.generate", "event": "generate_progress", "processed": 10})
        return SimpleNamespace(
            run_id=uuid4(),
            processed=742,
            succeeded=742,
            failed=0,
        )

    def fake_distribution(**kwargs):
        calls["distribution"] = kwargs
        kwargs["progress_callback"]({"stage": "distribution.export", "event": "export_progress", "processed_entries": 10})
        return SimpleNamespace(
            run_id=uuid4(),
            entry_count=741,
            output_path=Path("data/export/distribution.jsonl"),
            output_sha256="dist-sha",
        )

    def fake_distribution_sqlite(**kwargs):
        calls["distribution_sqlite"] = kwargs
        return SimpleNamespace(
            run_id=uuid4(),
            entry_count=741,
            output_path=Path("data/export/distribution.sqlite"),
            output_sha256="sqlite-sha",
        )

    def fake_audit(**kwargs):
        calls["audit"] = kwargs
        return SimpleNamespace(
            run_id=uuid4(),
            entry_count=742,
            output_path=Path("data/export/audit.jsonl"),
            output_sha256="audit-sha",
        )

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
    )
    monkeypatch.setattr(
        cli,
        "load_llm_settings",
        lambda **kwargs: SimpleNamespace(models=("test-model",)),
    )
    monkeypatch.setattr(cli, "get_connection", lambda settings: DummyConnection())
    monkeypatch.setattr(cli, "start_run", lambda conn, **kwargs: workflow_run_id)
    monkeypatch.setattr(cli, "complete_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(cli, "fail_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(cli, "run_raw_ingest_stage", fake_raw)
    monkeypatch.setattr(cli, "run_curated_build_stage", fake_curated)
    monkeypatch.setattr(cli, "run_llm_enrich_stage", fake_llm)
    monkeypatch.setattr(cli, "run_export_distribution_jsonl_stage", fake_distribution)
    monkeypatch.setattr(cli, "run_export_distribution_sqlite_stage", fake_distribution_sqlite)
    monkeypatch.setattr(cli, "run_export_audit_jsonl_stage", fake_audit)
    pending_counts = iter([742, 0])
    monkeypatch.setattr(cli, "_count_pending_llm_entries", lambda *args, **kwargs: next(pending_counts))
    monkeypatch.setattr(
        cli,
        "validate_distribution_jsonl_file",
        lambda path, **kwargs: SimpleNamespace(output_path=Path(path), entry_count=741),
    )

    exit_code = cli.main(
        [
            "run",
            "--skip-init-db",
            "--archive-path",
            "fixtures/wiktionary/raw.jsonl",
            "--max-workers",
            "50",
            "--validate-distribution",
            "--distribution-output",
            "data/export/distribution.jsonl",
            "--distribution-sqlite-output",
            "data/export/distribution.sqlite",
            "--audit-output",
            "data/export/audit.jsonl",
            "--model-env-file",
            "/tmp/llm.env",
        ]
    )

    assert exit_code == 0
    assert calls["source"]["archive_path"] == Path("fixtures/wiktionary/raw.jsonl")
    assert calls["definitions"]["max_workers"] == 50
    assert calls["definitions"]["env_file"] == "/tmp/llm.env"
    assert calls["definitions"]["definition_language"] == DEFAULT_DEFINITION_LANGUAGE
    assert calls["distribution"]["output_path"] == Path("data/export/distribution.jsonl")
    assert calls["distribution"]["definition_language"] == DEFAULT_DEFINITION_LANGUAGE
    assert calls["distribution_sqlite"]["output_path"] == Path("data/export/distribution.sqlite")
    assert calls["distribution_sqlite"]["definition_language"] == DEFAULT_DEFINITION_LANGUAGE
    assert calls["audit"]["output_path"] == Path("data/export/audit.jsonl")

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["command"] == "run"
    assert summary["status"] == "succeeded"
    assert summary["workflow_run_id"] == str(workflow_run_id)
    assert summary["definitions"]["failed"] == 0
    assert summary["definitions"]["prompt_version"] == prompt_bundle.resolved_prompt_version
    assert summary["definitions"]["definition_language"]["code"] == DEFAULT_DEFINITION_LANGUAGE.code
    assert summary["distribution_export"]["entry_count"] == 741
    assert summary["distribution_export"]["validated_entry_count"] == 741
    assert summary["distribution_export"]["prompt_version"] == prompt_bundle.resolved_prompt_version
    assert summary["distribution_sqlite_export"]["entry_count"] == 741
    assert summary["distribution_sqlite_export"]["sqlite_schema_version"] == "distribution_sqlite_v1"
    assert summary["audit_export"]["entry_count"] == 742
    assert "[progress] stage=source.ingest event=acquire_complete" in captured.err
    assert "[progress] stage=definitions.generate event=generate_progress" in captured.err


def test_pipeline_run_stops_when_llm_has_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
    )
    monkeypatch.setattr(
        cli,
        "load_llm_settings",
        lambda **kwargs: SimpleNamespace(models=("test-model",)),
    )
    monkeypatch.setattr(cli, "get_connection", lambda settings: DummyConnection())
    monkeypatch.setattr(cli, "start_run", lambda conn, **kwargs: uuid4())
    monkeypatch.setattr(cli, "complete_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(cli, "fail_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "run_raw_ingest_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            snapshot_id=uuid4(),
            rows_loaded=1000,
            anomalies_logged=0,
            snapshot_preexisting=False,
            archive_sha256="sha",
            resumed_from_run_id=None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_curated_build_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            groups_processed=742,
            groups_filtered_out=0,
            entries_written=742,
            relations_written=10,
            triage_written=1,
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_llm_enrich_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            processed=742,
            succeeded=733,
            failed=9,
        ),
    )
    pending_counts = iter([742, 9, 9, 9, 9])
    monkeypatch.setattr(cli, "_count_pending_llm_entries", lambda *args, **kwargs: next(pending_counts))
    monkeypatch.setattr(cli, "_fetch_recent_failed_enrichments", lambda *args, **kwargs: [("entry-1", "timed out")])

    with pytest.raises(SystemExit):
        cli.main(
            [
                "run",
                "--skip-init-db",
                "--archive-path",
                "fixtures/wiktionary/raw.jsonl",
            ]
        )


def test_pipeline_run_retries_with_worker_tiers_until_pending_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    workflow_run_id = uuid4()

    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
    )
    monkeypatch.setattr(
        cli,
        "load_llm_settings",
        lambda **kwargs: SimpleNamespace(models=("test-model",)),
    )
    monkeypatch.setattr(cli, "get_connection", lambda settings: DummyConnection())
    monkeypatch.setattr(cli, "start_run", lambda conn, **kwargs: workflow_run_id)
    monkeypatch.setattr(cli, "complete_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(cli, "fail_run", lambda conn, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "run_raw_ingest_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            snapshot_id=uuid4(),
            rows_loaded=1000,
            anomalies_logged=0,
            snapshot_preexisting=False,
            archive_sha256="sha",
            resumed_from_run_id=None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_curated_build_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            groups_processed=742,
            groups_filtered_out=0,
            entries_written=742,
            relations_written=10,
            triage_written=1,
        ),
    )

    def fake_llm(**kwargs):
        calls.append(kwargs["max_workers"])
        if kwargs["max_workers"] == 50:
            return SimpleNamespace(run_id=uuid4(), processed=742, succeeded=731, failed=11)
        return SimpleNamespace(run_id=uuid4(), processed=11, succeeded=11, failed=0)

    monkeypatch.setattr(cli, "run_llm_enrich_stage", fake_llm)
    pending_counts = iter([742, 11, 0])
    monkeypatch.setattr(cli, "_count_pending_llm_entries", lambda *args, **kwargs: next(pending_counts))
    monkeypatch.setattr(
        cli,
        "run_export_distribution_jsonl_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            entry_count=741,
            output_path=Path("data/export/distribution.jsonl"),
            output_sha256="dist-sha",
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_export_distribution_sqlite_stage",
        lambda **kwargs: SimpleNamespace(
            run_id=uuid4(),
            entry_count=741,
            output_path=Path("data/export/distribution.sqlite"),
            output_sha256="sqlite-sha",
        ),
    )
    monkeypatch.setattr(
        cli,
        "validate_distribution_jsonl_file",
        lambda path, **kwargs: SimpleNamespace(output_path=Path(path), entry_count=741),
    )

    exit_code = cli.main(
        [
            "run",
            "--skip-init-db",
            "--archive-path",
            "fixtures/wiktionary/raw.jsonl",
            "--worker-tiers",
            "50",
            "12",
            "--validate-distribution",
            "--distribution-sqlite-output",
            "data/export/distribution.sqlite",
        ]
    )

    assert exit_code == 0
    assert calls == [50, 12]
    summary = json.loads(capsys.readouterr().out)
    assert summary["command"] == "run"
    assert summary["status"] == "succeeded"
    assert summary["workflow_run_id"] == str(workflow_run_id)
    assert summary["definitions"]["attempts"][0]["workers"] == 50
    assert summary["definitions"]["attempts"][1]["workers"] == 12
    assert summary["definitions"]["attempts"][1]["remaining_entries"] == 0
    assert summary["distribution_sqlite_export"]["entry_count"] == 741


def test_validate_distribution_jsonl_command_reads_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "distribution.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "distribution_entry_v4",
                "entry_id": "entry-1",
                "headword": "barra",
                "normalized_headword": "barra",
                "headword_language": {"code": "aa", "name": "Afar"},
                "definition_language": {"code": "en", "name": "English"},
                "entry_type": "standard",
                "headword_summary": "Overall summary.",
                "memory_hook": "一句帮助记忆的主线。",
                "study_notes": [],
                "etymology_note": None,
                "etymologies": [{"etymology_id": "et1", "text": None, "pos_members": ["noun"]}],
                "pos_groups": [
                    {
                        "pos": "noun",
                        "etymology_id": "et1",
                        "summary": "Noun summary.",
                        "usage_note": None,
                        "forms": [],
                        "pronunciations": [{"ipa": "/x/", "text": None, "tags": []}],
                        "meanings": [
                            {
                                "sense_id": "s1",
                                "priority": "core",
                                "short_gloss": "woman",
                                "learner_explanation": "Detailed explanation.",
                                "usage_note": None,
                                "labels": [],
                                "topics": [],
                                "examples": [],
                            }
                        ],
                        "relations": [],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-distribution", "--input", str(output_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["command"] == "validate-distribution"
    assert summary["status"] == "succeeded"
    assert summary["entry_count"] == 1
    assert "[progress] stage=distribution.validate event=validate_start" in captured.err
    assert "[progress] stage=distribution.validate event=validate_complete" in captured.err


def test_llm_enrich_cli_accepts_custom_definition_language(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_run_llm_enrich_stage(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            run_id=uuid4(),
            processed=1,
            succeeded=1,
            failed=0,
        )

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda **kwargs: RuntimeSettings(database_url="postgresql://example/test"),
    )
    monkeypatch.setattr(cli, "run_llm_enrich_stage", fake_run_llm_enrich_stage)

    exit_code = cli.main(
        [
            "generate-definitions",
            "--definition-language-code",
            "fr",
            "--definition-language-name",
            "French",
        ]
    )

    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured_kwargs["definition_language"].code == "fr"
    assert captured_kwargs["definition_language"].name == "French"
    assert summary["definition_language"] == {"code": "fr", "name": "French"}
    assert summary["prompt_version"].endswith("__deflang__fr")
