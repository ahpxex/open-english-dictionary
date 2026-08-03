from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import psycopg


@dataclass(frozen=True)
class Migration:
    version: str
    statements: tuple[str, ...]


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(
        version="20260327_foundation_v1",
        statements=(
            "CREATE SCHEMA IF NOT EXISTS meta",
            "CREATE SCHEMA IF NOT EXISTS raw",
            "CREATE SCHEMA IF NOT EXISTS curated",
            "CREATE SCHEMA IF NOT EXISTS llm",
            "CREATE SCHEMA IF NOT EXISTS export",
            """
            CREATE TABLE IF NOT EXISTS meta.pipeline_runs (
                run_id UUID PRIMARY KEY,
                stage TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
                parent_run_id UUID REFERENCES meta.pipeline_runs(run_id),
                config JSONB NOT NULL DEFAULT '{}'::jsonb,
                stats JSONB NOT NULL DEFAULT '{}'::jsonb,
                error TEXT,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS meta.source_snapshots (
                snapshot_id UUID PRIMARY KEY,
                run_id UUID,
                source_name TEXT NOT NULL,
                source_url TEXT,
                archive_path TEXT NOT NULL,
                extracted_path TEXT,
                archive_sha256 TEXT NOT NULL,
                archive_size_bytes BIGINT NOT NULL,
                downloaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw.wiktionary_entries (
                id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES meta.pipeline_runs(run_id),
                snapshot_id UUID NOT NULL REFERENCES meta.source_snapshots(snapshot_id),
                source_line BIGINT NOT NULL,
                source_byte_offset BIGINT NOT NULL,
                word TEXT,
                lang TEXT,
                lang_code TEXT,
                pos TEXT,
                payload JSONB NOT NULL,
                inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (run_id, source_line)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_entries_snapshot_id_idx
            ON raw.wiktionary_entries (snapshot_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_entries_word_idx
            ON raw.wiktionary_entries (word)
            """,
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_entries_lang_code_idx
            ON raw.wiktionary_entries (lang_code)
            """,
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_entries_pos_idx
            ON raw.wiktionary_entries (pos)
            """,
        ),
    ),
    Migration(
        version="20260327_raw_ingest_rewrite_v2",
        statements=(
            """
            ALTER TABLE meta.source_snapshots
            ALTER COLUMN run_id DROP NOT NULL
            """,
            """
            ALTER TABLE meta.source_snapshots
            ALTER COLUMN source_url DROP NOT NULL
            """,
            """
            ALTER TABLE meta.source_snapshots
            ADD COLUMN IF NOT EXISTS acquisition_mode TEXT
            """,
            """
            ALTER TABLE meta.source_snapshots
            ADD COLUMN IF NOT EXISTS compression TEXT
            """,
            """
            ALTER TABLE meta.source_snapshots
            ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            """,
            """
            ALTER TABLE raw.wiktionary_entries
            ADD COLUMN IF NOT EXISTS lang TEXT
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS meta_source_snapshots_source_name_sha256_uidx
            ON meta.source_snapshots (source_name, archive_sha256)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS raw_wiktionary_entries_snapshot_line_uidx
            ON raw.wiktionary_entries (snapshot_id, source_line)
            """,
            """
            CREATE TABLE IF NOT EXISTS meta.stage_checkpoints (
                run_id UUID NOT NULL REFERENCES meta.pipeline_runs(run_id),
                stage_name TEXT NOT NULL,
                checkpoint_key TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, stage_name, checkpoint_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw.wiktionary_ingest_anomalies (
                anomaly_id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES meta.pipeline_runs(run_id),
                snapshot_id UUID NOT NULL REFERENCES meta.source_snapshots(snapshot_id),
                source_line BIGINT NOT NULL,
                source_byte_offset BIGINT NOT NULL,
                anomaly_type TEXT NOT NULL,
                detail TEXT NOT NULL,
                raw_payload TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_ingest_anomalies_snapshot_id_idx
            ON raw.wiktionary_ingest_anomalies (snapshot_id)
            """,
        ),
    ),
    Migration(
        version="20260327_curated_v1",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS curated.entries (
                entry_id UUID PRIMARY KEY,
                lang_code TEXT NOT NULL,
                normalized_word TEXT NOT NULL,
                word TEXT NOT NULL,
                payload JSONB NOT NULL,
                entry_flags TEXT[] NOT NULL DEFAULT '{}',
                source_summary JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS curated_entries_lang_word_uidx
            ON curated.entries (lang_code, normalized_word)
            """,
            """
            CREATE TABLE IF NOT EXISTS curated.entry_relations (
                relation_id BIGSERIAL PRIMARY KEY,
                entry_id UUID NOT NULL REFERENCES curated.entries(entry_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                target_word TEXT NOT NULL,
                target_lang_code TEXT,
                source_scope TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS curated_entry_relations_entry_id_idx
            ON curated.entry_relations (entry_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS curated.triage_queue (
                triage_id BIGSERIAL PRIMARY KEY,
                lang_code TEXT,
                word TEXT,
                reason_code TEXT NOT NULL,
                severity TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                raw_record_refs JSONB NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
        ),
    ),
    Migration(
        version="20260327_llm_v1",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS llm.prompt_versions (
                prompt_version TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                output_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS llm.entry_enrichments (
                enrichment_id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES meta.pipeline_runs(run_id),
                entry_id UUID NOT NULL REFERENCES curated.entries(entry_id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL REFERENCES llm.prompt_versions(prompt_version),
                input_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
                request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                response_payload JSONB,
                raw_response TEXT,
                error TEXT,
                retries INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS llm_entry_enrichments_entry_id_idx
            ON llm.entry_enrichments (entry_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS llm_entry_enrichments_status_idx
            ON llm.entry_enrichments (status)
            """,
        ),
    ),
    Migration(
        version="20260327_export_v1",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS export.artifacts (
                artifact_id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES meta.pipeline_runs(run_id),
                artifact_type TEXT NOT NULL,
                output_path TEXT NOT NULL,
                output_sha256 TEXT NOT NULL,
                entry_count INTEGER NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
        ),
    ),
    Migration(
        version="20260403_curated_lineage_v2",
        statements=(
            """
            ALTER TABLE curated.entries
            ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES meta.pipeline_runs(run_id)
            """,
            """
            ALTER TABLE curated.entry_relations
            ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES meta.pipeline_runs(run_id)
            """,
            """
            ALTER TABLE curated.triage_queue
            ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES meta.pipeline_runs(run_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS curated_entries_run_id_idx
            ON curated.entries (run_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS curated_entry_relations_run_id_idx
            ON curated.entry_relations (run_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS curated_triage_queue_run_id_idx
            ON curated.triage_queue (run_id)
            """,
        ),
    ),
    Migration(
        version="20260404_multilingual_definition_language_v3",
        statements=(
            """
            ALTER TABLE llm.prompt_versions
            ADD COLUMN IF NOT EXISTS definition_language_code TEXT NOT NULL DEFAULT 'zh-Hans'
            """,
            """
            ALTER TABLE llm.prompt_versions
            ADD COLUMN IF NOT EXISTS definition_language_name TEXT NOT NULL DEFAULT 'Chinese (Simplified)'
            """,
            """
            ALTER TABLE llm.prompt_versions
            ADD COLUMN IF NOT EXISTS prompt_bundle JSONB NOT NULL DEFAULT '{}'::jsonb
            """,
            """
            CREATE INDEX IF NOT EXISTS llm_prompt_versions_definition_language_code_idx
            ON llm.prompt_versions (definition_language_code)
            """,
            """
            ALTER TABLE llm.entry_enrichments
            ADD COLUMN IF NOT EXISTS definition_language_code TEXT NOT NULL DEFAULT 'zh-Hans'
            """,
            """
            ALTER TABLE llm.entry_enrichments
            ADD COLUMN IF NOT EXISTS definition_language_name TEXT NOT NULL DEFAULT 'Chinese (Simplified)'
            """,
            """
            CREATE INDEX IF NOT EXISTS llm_entry_enrichments_definition_language_code_idx
            ON llm.entry_enrichments (definition_language_code)
            """,
            """
            CREATE INDEX IF NOT EXISTS llm_entry_enrichments_lookup_idx
            ON llm.entry_enrichments (
                entry_id,
                model,
                prompt_version,
                definition_language_code,
                status,
                created_at DESC
            )
            """,
        ),
    ),
    Migration(
        version="20260409_curated_build_perf_v4",
        statements=(
            """
            CREATE INDEX IF NOT EXISTS raw_wiktionary_entries_grouping_idx
            ON raw.wiktionary_entries (
                lang_code,
                (lower(coalesce(word, payload->>'word', ''))),
                id
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS curated_triage_queue_lang_word_idx
            ON curated.triage_queue (lang_code, word)
            """,
        ),
    ),
    Migration(
        version="20260802_llm_generation_metadata_v5",
        statements=(
            """
            ALTER TABLE llm.entry_enrichments
            ADD COLUMN IF NOT EXISTS generation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            """,
        ),
    ),
)


LATEST_FOUNDATION_VERSION: Final[str] = MIGRATIONS[-1].version


def apply_foundation(conn: psycopg.Connection) -> list[str]:
    applied_versions: list[str] = []

    with conn.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS meta")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS meta.schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

    conn.commit()

    for migration in MIGRATIONS:
        if _is_applied(conn, migration.version):
            continue

        with conn.cursor() as cursor:
            for statement in migration.statements:
                cursor.execute(statement)
            cursor.execute(
                "INSERT INTO meta.schema_migrations (version) VALUES (%s)",
                (migration.version,),
            )
        conn.commit()
        applied_versions.append(migration.version)

    return applied_versions


def _is_applied(conn: psycopg.Connection, version: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM meta.schema_migrations WHERE version = %s",
            (version,),
        )
        return cursor.fetchone() is not None
