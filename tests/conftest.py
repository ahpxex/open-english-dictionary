from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest


def current_user() -> str:
    return os.environ.get("USER", "postgres")


@pytest.fixture(autouse=True)
def isolated_llm_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider-pool variables from the developer's real environment must not
    # leak into tests that configure the legacy single-provider variables.
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("LLM_RPM", raising=False)


def admin_dsn() -> str:
    return f"postgresql://{current_user()}@localhost:5432/postgres"


@pytest.fixture
def sample_record_lines() -> list[str]:
    return [
        json.dumps(
            {
                "word": "cat",
                "lang": "English",
                "lang_code": "en",
                "pos": "noun",
                "senses": [{"glosses": ["A domesticated feline."]}],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "word": "run",
                "lang": "English",
                "lang_code": "en",
                "pos": "verb",
                "senses": [{"glosses": ["To move swiftly on foot."]}],
            },
            ensure_ascii=False,
        ),
    ]


@pytest.fixture
def plain_jsonl_path(tmp_path: Path, sample_record_lines: list[str]) -> Path:
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(sample_record_lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def gzip_jsonl_path(tmp_path: Path, sample_record_lines: list[str]) -> Path:
    path = tmp_path / "sample.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(sample_record_lines))
        handle.write("\n")
    return path


@pytest.fixture
def anomaly_jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "anomaly.jsonl"
    rows = [
        json.dumps(
            {
                "word": "cat",
                "lang": "English",
                "lang_code": "en",
                "pos": "noun",
                "senses": [{"glosses": ["A domesticated feline."]}],
            },
            ensure_ascii=False,
        ),
        '{"word": "broken"',
        '["not", "an", "object"]',
        json.dumps(
            {
                "word": "orange",
                "lang": "French",
                "lang_code": "fr",
                "pos": "adj",
                "senses": [{"glosses": ["orange (orange-coloured)"]}],
            },
            ensure_ascii=False,
        ),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def temp_database_url() -> str:
    db_name = f"open_dictionary_test_{uuid4().hex[:12]}"
    admin_conn = psycopg.connect(admin_dsn(), autocommit=True)
    try:
        with admin_conn.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        admin_conn.close()

    yield f"postgresql://{current_user()}@localhost:5432/{db_name}"

    admin_conn = psycopg.connect(admin_dsn(), autocommit=True)
    try:
        with admin_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s
                  AND pid <> pg_backend_pid()
                """,
                (db_name,),
            )
            cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        admin_conn.close()
