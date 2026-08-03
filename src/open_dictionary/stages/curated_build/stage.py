from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from open_dictionary.config import RuntimeSettings
from open_dictionary.db.connection import get_connection
from open_dictionary.pipeline import ProgressCallback, ThrottledProgressReporter, emit_progress, complete_run, fail_run, start_run

from .transform import CuratedBuildOutput, TriageItem, build_curated_entry
from .word_selection import WordSelectionRule


CURATED_BUILD_STAGE = "entries.assemble"
DEFAULT_RAW_SOURCE_TABLE = "raw.wiktionary_entries"
DEFAULT_CURATED_TABLE = "curated.entries"
DEFAULT_RELATIONS_TABLE = "curated.entry_relations"
DEFAULT_TRIAGE_TABLE = "curated.triage_queue"
DEFAULT_PERSIST_BATCH_SIZE = 1000


@dataclass(frozen=True)
class CuratedBuildResult:
    run_id: UUID
    groups_processed: int
    groups_filtered_out: int
    entries_written: int
    relations_written: int
    triage_written: int


def run_curated_build_stage(
    *,
    settings: RuntimeSettings,
    source_table: str = DEFAULT_RAW_SOURCE_TABLE,
    target_table: str = DEFAULT_CURATED_TABLE,
    relations_table: str = DEFAULT_RELATIONS_TABLE,
    triage_table: str = DEFAULT_TRIAGE_TABLE,
    lang_codes: list[str] | None = None,
    limit_groups: int | None = None,
    replace_existing: bool = False,
    word_selection: WordSelectionRule | None = None,
    parent_run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CuratedBuildResult:
    with get_connection(settings) as conn:
        source_run_ids, source_snapshot_ids = _resolve_raw_lineage(
            conn,
            source_table=source_table,
            lang_codes=lang_codes,
        )

    with get_connection(settings) as conn:
        run_id = start_run(
            conn,
            stage=CURATED_BUILD_STAGE,
            config={
                "source_table": source_table,
                "target_table": target_table,
                "relations_table": relations_table,
                "triage_table": triage_table,
                "lang_codes": lang_codes or [],
                "limit_groups": limit_groups,
                "replace_existing": replace_existing,
                "word_selection": (
                    word_selection.as_metadata() if word_selection is not None else None
                ),
                "source_run_ids": source_run_ids,
                "source_snapshot_ids": source_snapshot_ids,
            },
            parent_run_id=parent_run_id,
        )

    groups_processed = 0
    groups_filtered_out = 0
    entries_written = 0
    relations_written = 0
    triage_written = 0
    pending_outputs: list[CuratedBuildOutput] = []
    pending_groups = 0
    pending_entries = 0
    pending_relations = 0
    pending_triage = 0

    def group_is_selected(group_key: tuple[str, str]) -> bool:
        if word_selection is None:
            return True
        lang_code, normalized_word = group_key
        if lang_code != word_selection.lang:
            return True
        return word_selection.accepts(normalized_word)

    try:
        with get_connection(settings) as conn:
            reporter = ThrottledProgressReporter(progress_callback, stage=CURATED_BUILD_STAGE)
            emit_progress(
                progress_callback,
                stage=CURATED_BUILD_STAGE,
                event="build_start",
                lang_codes=lang_codes or [],
                limit_groups=limit_groups,
                replace_existing=replace_existing,
                word_selection=(
                    word_selection.as_metadata() if word_selection is not None else None
                ),
            )
            if replace_existing:
                _reset_outputs(conn, target_table=target_table, relations_table=relations_table, triage_table=triage_table)

            current_rows: list[dict[str, Any]] = []
            current_key: tuple[str, str] | None = None

            def flush_pending_outputs(*, force: bool = False) -> None:
                nonlocal groups_processed
                nonlocal entries_written
                nonlocal relations_written
                nonlocal triage_written
                nonlocal pending_groups
                nonlocal pending_entries
                nonlocal pending_relations
                nonlocal pending_triage

                if not pending_outputs:
                    return

                persist_curated_outputs(
                    conn,
                    run_id=run_id,
                    outputs=pending_outputs,
                    target_table=target_table,
                    relations_table=relations_table,
                    triage_table=triage_table,
                )

                groups_processed += pending_groups
                entries_written += pending_entries
                relations_written += pending_relations
                triage_written += pending_triage

                pending_outputs.clear()
                pending_groups = 0
                pending_entries = 0
                pending_relations = 0
                pending_triage = 0

                reporter.report(
                    event="build_progress",
                    force=force,
                    groups_processed=groups_processed,
                    entries_written=entries_written,
                    relations_written=relations_written,
                    triage_written=triage_written,
                )

            for raw_row in iter_groupable_rows(conn, source_table=source_table, lang_codes=lang_codes):
                next_key = (
                    str(raw_row.get("lang_code") or "_"),
                    str(raw_row.get("normalized_word") or "_"),
                )
                if current_key is None:
                    current_key = next_key
                if next_key != current_key:
                    if group_is_selected(current_key):
                        output = build_curated_entry(current_rows)
                        pending_outputs.append(output)
                        pending_groups += 1
                        pending_entries += 1 if output.entry is not None else 0
                        pending_relations += len(output.relations)
                        pending_triage += len(output.triage_items)
                        if len(pending_outputs) >= DEFAULT_PERSIST_BATCH_SIZE:
                            flush_pending_outputs()
                    else:
                        groups_filtered_out += 1
                    reporter.report(
                        event="build_progress",
                        groups_processed=groups_processed + pending_groups,
                        groups_filtered_out=groups_filtered_out,
                        entries_written=entries_written + pending_entries,
                        relations_written=relations_written + pending_relations,
                        triage_written=triage_written + pending_triage,
                    )
                    if limit_groups is not None and (groups_processed + pending_groups) >= limit_groups:
                        current_rows = []
                        current_key = next_key
                        break
                    current_rows = [raw_row]
                    current_key = next_key
                else:
                    current_rows.append(raw_row)

            if current_rows and (limit_groups is None or groups_processed < limit_groups):
                if current_key is not None and group_is_selected(current_key):
                    output = build_curated_entry(current_rows)
                    pending_outputs.append(output)
                    pending_groups += 1
                    pending_entries += 1 if output.entry is not None else 0
                    pending_relations += len(output.relations)
                    pending_triage += len(output.triage_items)
                else:
                    groups_filtered_out += 1

            flush_pending_outputs(force=True)

            complete_run(
                conn,
                run_id=run_id,
                stats={
                    "groups_processed": groups_processed,
                    "groups_filtered_out": groups_filtered_out,
                    "entries_written": entries_written,
                    "relations_written": relations_written,
                    "triage_written": triage_written,
                    "word_selection": (
                        word_selection.as_metadata() if word_selection is not None else None
                    ),
                    "source_run_ids": source_run_ids,
                    "source_snapshot_ids": source_snapshot_ids,
                },
            )
            emit_progress(
                progress_callback,
                stage=CURATED_BUILD_STAGE,
                event="build_complete",
                groups_processed=groups_processed,
                groups_filtered_out=groups_filtered_out,
                entries_written=entries_written,
                relations_written=relations_written,
                triage_written=triage_written,
            )

        return CuratedBuildResult(
            run_id=run_id,
            groups_processed=groups_processed,
            groups_filtered_out=groups_filtered_out,
            entries_written=entries_written,
            relations_written=relations_written,
            triage_written=triage_written,
        )
    except Exception as exc:
        with get_connection(settings) as conn:
            fail_run(conn, run_id=run_id, error=str(exc))
        raise


def iter_groupable_rows(conn, *, source_table: str, lang_codes: list[str] | None):
    table_identifier = _identifier_from_dotted(source_table)
    query = sql.SQL(
        """
        SELECT
            id,
            snapshot_id,
            run_id,
            source_line,
            word,
            lang,
            lang_code,
            pos,
            payload,
            lower(coalesce(word, payload->>'word', '')) AS normalized_word
        FROM {}
        """
    ).format(table_identifier)

    params: list[Any] = []
    if lang_codes:
        query += sql.SQL(" WHERE lang_code = ANY(%s)")
        params.append(lang_codes)

    query += sql.SQL(" ORDER BY lang_code, normalized_word, id")

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        columns = [desc.name for desc in cursor.description]
        for row in cursor:
            yield dict(zip(columns, row, strict=False))


def persist_curated_output(
    conn,
    *,
    run_id: UUID,
    output: CuratedBuildOutput,
    target_table: str,
    relations_table: str,
    triage_table: str,
) -> int:
    persist_curated_outputs(
        conn,
        run_id=run_id,
        outputs=[output],
        target_table=target_table,
        relations_table=relations_table,
        triage_table=triage_table,
    )
    return 1 if output.entry is not None else 0


def persist_curated_outputs(
    conn,
    *,
    run_id: UUID,
    outputs: list[CuratedBuildOutput],
    target_table: str,
    relations_table: str,
    triage_table: str,
) -> None:
    if not outputs:
        return

    clear_group_triage_batch(conn, triage_table=triage_table, outputs=outputs)

    entries = [output.entry for output in outputs if output.entry is not None]
    if entries:
        upsert_entries(conn, target_table=target_table, run_id=run_id, entries=entries)
        replace_relations_batch(conn, relations_table=relations_table, run_id=run_id, outputs=outputs)

    triage_items = [item for output in outputs for item in output.triage_items]
    if triage_items:
        insert_triage_items(conn, triage_table=triage_table, run_id=run_id, items=triage_items)

    conn.commit()


def clear_group_triage_batch(conn, *, triage_table: str, outputs: list[CuratedBuildOutput]) -> None:
    identities = sorted(
        {
            (lang_code, word)
            for output in outputs
            for lang_code, word in [triage_group_identity(output)]
            if lang_code and word
        }
    )
    if not identities:
        return

    table_identifier = _identifier_from_dotted(triage_table)
    values_sql = sql.SQL(", ").join(sql.SQL("(%s, %s)") for _ in identities)
    params: list[Any] = []
    for lang_code, word in identities:
        params.extend((lang_code, word))

    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                DELETE FROM {} AS triage
                USING (VALUES {}) AS doomed(lang_code, word)
                WHERE triage.lang_code = doomed.lang_code
                  AND triage.word = doomed.word
                """
            ).format(table_identifier, values_sql),
            params,
        )


def triage_group_identity(output: CuratedBuildOutput) -> tuple[str | None, str | None]:
    if output.entry is not None:
        return output.entry["lang_code"], output.entry["normalized_word"]
    if output.triage_items:
        return output.triage_items[0].lang_code, output.triage_items[0].word
    return None, None


def upsert_entries(conn, *, target_table: str, run_id: UUID, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return

    table_identifier = _identifier_from_dotted(target_table)
    values_sql = sql.SQL(", ").join(sql.SQL("(%s, %s, %s, %s, %s, %s, %s, %s)") for _ in entries)
    query = sql.SQL(
        """
        INSERT INTO {} (
            run_id,
            entry_id,
            lang_code,
            normalized_word,
            word,
            payload,
            entry_flags,
            source_summary
        ) VALUES {}
        ON CONFLICT (entry_id)
        DO UPDATE SET
            run_id = EXCLUDED.run_id,
            lang_code = EXCLUDED.lang_code,
            normalized_word = EXCLUDED.normalized_word,
            word = EXCLUDED.word,
            payload = EXCLUDED.payload,
            entry_flags = EXCLUDED.entry_flags,
            source_summary = EXCLUDED.source_summary,
            updated_at = NOW()
        """
    ).format(table_identifier, values_sql)

    params: list[Any] = []
    for entry in entries:
        params.extend(
            (
                run_id,
                entry["entry_id"],
                entry["lang_code"],
                entry["normalized_word"],
                entry["word"],
                Jsonb(entry),
                entry["entry_flags"],
                Jsonb(entry["source_summary"]),
            )
        )
    with conn.cursor() as cursor:
        cursor.execute(query, params)


def replace_relations_batch(
    conn,
    *,
    relations_table: str,
    run_id: UUID,
    outputs: list[CuratedBuildOutput],
) -> None:
    entry_ids = sorted(
        {
            output.entry["entry_id"]
            for output in outputs
            if output.entry is not None
        }
    )
    if not entry_ids:
        return

    table_identifier = _identifier_from_dotted(relations_table)
    with conn.cursor() as cursor:
        delete_values_sql = sql.SQL(", ").join(sql.SQL("(%s::uuid)") for _ in entry_ids)
        cursor.execute(
            sql.SQL(
                """
                DELETE FROM {} AS relations
                USING (VALUES {}) AS doomed(entry_id)
                WHERE relations.entry_id = doomed.entry_id
                """
            ).format(table_identifier, delete_values_sql),
            entry_ids,
        )

        relations = [relation for output in outputs for relation in output.relations]
        if not relations:
            return

        values_sql = sql.SQL(", ").join(
            sql.SQL("(%s::uuid, %s::uuid, %s::text, %s::text, %s::text, %s::text, %s::jsonb)")
            for _ in relations
        )
        query = sql.SQL(
            """
            INSERT INTO {} (
                run_id,
                entry_id,
                relation_type,
                target_word,
                target_lang_code,
                source_scope,
                payload
            ) VALUES {}
            """
        ).format(table_identifier, values_sql)
        params: list[Any] = []
        for relation in relations:
            params.extend(
                (
                    run_id,
                    relation["entry_id"],
                    relation["relation_type"],
                    relation["target_word"],
                    relation.get("target_lang_code"),
                    relation["source_scope"],
                    Jsonb(relation["payload"]),
                )
            )
        cursor.execute(query, params)


def insert_triage_items(conn, *, triage_table: str, run_id: UUID, items: list[TriageItem]) -> None:
    if not items:
        return

    table_identifier = _identifier_from_dotted(triage_table)
    values_sql = sql.SQL(", ").join(sql.SQL("(%s, %s, %s, %s, %s, %s, %s, %s)") for _ in items)
    query = sql.SQL(
        """
        INSERT INTO {} (
            run_id,
            lang_code,
            word,
            reason_code,
            severity,
            suggested_action,
            raw_record_refs,
            payload
        ) VALUES {}
        """
    ).format(table_identifier, values_sql)

    params: list[Any] = []
    for item in items:
        params.extend(
            (
                run_id,
                item.lang_code,
                item.word,
                item.reason_code,
                item.severity,
                item.suggested_action,
                Jsonb(item.raw_record_refs),
                Jsonb(item.payload),
            )
        )

    with conn.cursor() as cursor:
        cursor.execute(query, params)


def _reset_outputs(conn, *, target_table: str, relations_table: str, triage_table: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(_identifier_from_dotted(target_table)))
        cursor.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(_identifier_from_dotted(relations_table)))
        cursor.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(_identifier_from_dotted(triage_table)))
    conn.commit()


def _identifier_from_dotted(qualified_name: str) -> sql.Identifier:
    parts = [segment.strip() for segment in qualified_name.split(".") if segment.strip()]
    if not parts:
        raise ValueError("Identifier name cannot be empty")
    return sql.Identifier(*parts)


def _resolve_raw_lineage(
    conn,
    *,
    source_table: str,
    lang_codes: list[str] | None,
) -> tuple[list[str], list[str]]:
    table_identifier = _identifier_from_dotted(source_table)
    query = sql.SQL(
        """
        SELECT
            array_remove(array_agg(DISTINCT run_id::text), NULL),
            array_remove(array_agg(DISTINCT snapshot_id::text), NULL)
        FROM {}
        """
    ).format(table_identifier)
    params: list[Any] = []
    if lang_codes:
        query += sql.SQL(" WHERE lang_code = ANY(%s)")
        params.append(lang_codes)

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        source_run_ids, source_snapshot_ids = cursor.fetchone()

    return sorted(source_run_ids or []), sorted(source_snapshot_ids or [])
