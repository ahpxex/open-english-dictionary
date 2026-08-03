from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from open_dictionary.config.settings import RuntimeSettings
from open_dictionary.db.bootstrap import apply_foundation
from open_dictionary.db.connection import get_connection
from open_dictionary.stages.curated_build import stage as curated_stage_module
from open_dictionary.stages.curated_build.stage import run_curated_build_stage
from open_dictionary.stages.raw_ingest.stage import run_raw_ingest_stage


def insert_raw_row(
    conn,
    *,
    word: str,
    lang: str,
    lang_code: str,
    pos: str,
    source_line: int,
    gloss: str,
) -> None:
    run_id = uuid4()
    snapshot_id = uuid4()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            insert into meta.pipeline_runs (run_id, stage, status, config)
            values (%s, %s, %s, '{}'::jsonb)
            """,
            (run_id, "test.raw_seed", "succeeded"),
        )
        cursor.execute(
            """
            insert into meta.source_snapshots (
                snapshot_id, run_id, source_name, source_url, archive_path,
                archive_sha256, archive_size_bytes, acquisition_mode, compression, metadata
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb)
            """,
            (
                snapshot_id,
                run_id,
                "wiktionary",
                None,
                f"/tmp/{word}.jsonl",
                f"sha-{word}-{source_line}",
                1,
                "register_local",
                "plain",
            ),
        )
        cursor.execute(
            """
            insert into raw.wiktionary_entries (
                run_id, snapshot_id, source_line, source_byte_offset,
                word, lang, lang_code, pos, payload
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                run_id,
                snapshot_id,
                source_line,
                source_line,
                word,
                lang,
                lang_code,
                pos,
                (
                    '{"word":"%s","lang":"%s","lang_code":"%s","pos":"%s",'
                    '"senses":[{"glosses":["%s"]}]}'
                    % (word, lang, lang_code, pos, gloss)
                ),
            ),
        )


def test_curated_build_stage_creates_entries_relations_and_triage(
    anomaly_jsonl_path: Path,
    temp_database_url: str,
) -> None:
    # This case runs the first two stages together and verifies that curated
    # entries, relation rows, and triage items all land in the database.
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)

    run_raw_ingest_stage(
        settings=settings,
        workdir=anomaly_jsonl_path.parent,
        archive_path=anomaly_jsonl_path,
    )
    with get_connection(settings) as conn:
        insert_raw_row(
            conn,
            word="倦",
            lang="Japanese",
            lang_code="ja",
            pos="character",
            source_line=99,
            gloss="in fatigue",
        )
        conn.commit()
    result = run_curated_build_stage(settings=settings)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select count(*) from curated.entries")
            entries = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.entries where run_id is not null")
            entries_with_run_id = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.entry_relations")
            relations = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.entry_relations where run_id is not null")
            relations_with_run_id = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.triage_queue")
            triage = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.triage_queue where run_id is not null")
            triage_with_run_id = cursor.fetchone()[0]
            cursor.execute(
                """
                select config->'source_run_ids', stats->'source_run_ids', config->'source_snapshot_ids'
                from meta.pipeline_runs
                where run_id = %s
                """,
                (result.run_id,),
            )
            source_run_ids, stats_source_run_ids, source_snapshot_ids = cursor.fetchone()

    assert result.entries_written == entries
    assert result.relations_written == relations
    assert result.triage_written == triage
    assert entries >= 1
    assert triage >= 1
    assert entries_with_run_id == entries
    assert relations_with_run_id == relations
    assert triage_with_run_id == triage
    assert source_run_ids
    assert stats_source_run_ids
    assert source_snapshot_ids


def test_curated_build_stage_filters_by_lang_codes(
    gzip_jsonl_path: Path,
    temp_database_url: str,
) -> None:
    # This case protects targeted local rebuilds where only one language slice
    # should be materialized.
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="cat", lang="English", lang_code="en", pos="noun", source_line=1, gloss="cat")
        insert_raw_row(conn, word="chien", lang="French", lang_code="fr", pos="noun", source_line=2, gloss="dog")
        conn.commit()

    result = run_curated_build_stage(settings=settings, lang_codes=["fr"])

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select lang_code, word from curated.entries order by word")
            rows = cursor.fetchall()

    assert result.entries_written == 1
    assert rows == [("fr", "chien")]


def test_curated_build_stage_limit_groups_stops_after_requested_count(
    temp_database_url: str,
) -> None:
    # This case supports debugger-friendly partial builds over a larger raw table.
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="alpha", lang="English", lang_code="en", pos="noun", source_line=1, gloss="alpha")
        insert_raw_row(conn, word="beta", lang="English", lang_code="en", pos="noun", source_line=2, gloss="beta")
        conn.commit()

    result = run_curated_build_stage(settings=settings, limit_groups=1)

    assert result.groups_processed == 1


def test_curated_build_stage_replace_existing_resets_outputs(
    temp_database_url: str,
) -> None:
    # This case ensures rebuild mode behaves deterministically instead of appending
    # duplicate output rows across repeated curated runs.
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="alpha", lang="English", lang_code="en", pos="noun", source_line=1, gloss="alpha")
        conn.commit()

    run_curated_build_stage(settings=settings)
    run_curated_build_stage(settings=settings, replace_existing=True)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select count(*) from curated.entries")
            entries = cursor.fetchone()[0]

    assert entries == 1


def test_curated_build_stage_rerun_does_not_duplicate_group_triage(
    temp_database_url: str,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="倦", lang="Japanese", lang_code="ja", pos="character", source_line=1, gloss="in fatigue")
        conn.commit()

    first = run_curated_build_stage(settings=settings)
    second = run_curated_build_stage(settings=settings)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select count(*) from curated.triage_queue")
            triage_count = cursor.fetchone()[0]

    assert first.triage_written == 1
    assert second.triage_written == 1
    assert triage_count == 1


def test_curated_build_stage_batches_writes_without_losing_results(
    temp_database_url: str,
    monkeypatch,
) -> None:
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="alpha", lang="English", lang_code="en", pos="noun", source_line=1, gloss="alpha")
        insert_raw_row(conn, word="beta", lang="English", lang_code="en", pos="noun", source_line=2, gloss="beta")
        insert_raw_row(conn, word="倦", lang="Japanese", lang_code="ja", pos="character", source_line=3, gloss="in fatigue")
        conn.commit()

    monkeypatch.setattr(curated_stage_module, "DEFAULT_PERSIST_BATCH_SIZE", 2)

    first = run_curated_build_stage(settings=settings)
    second = run_curated_build_stage(settings=settings)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select count(*) from curated.entries")
            entry_count = cursor.fetchone()[0]
            cursor.execute("select count(*) from curated.triage_queue")
            triage_count = cursor.fetchone()[0]
            cursor.execute("select array_agg(word order by word) from curated.entries")
            words = cursor.fetchone()[0]

    assert first.groups_processed == 3
    assert second.groups_processed == 3
    assert entry_count == 2
    assert triage_count == 1
    assert words == ["alpha", "beta"]


def test_word_selection_rule_unifies_words_phrases_and_proper_nouns() -> None:
    # This case pins the approved word_selection_v1 semantics: one Zipf
    # threshold applies to single words, phrases, and proper nouns alike.
    from open_dictionary.stages.curated_build.word_selection import build_word_selection_rule

    rule = build_word_selection_rule(lang="en", top_n=40000)

    assert rule.rule_version == "word_selection_v2_phrase_threshold"
    assert rule.min_zipf > 0
    assert rule.accepts("water")
    assert rule.accepts("London")
    assert rule.accepts("boxing glove")
    assert rule.accepts("Peugeot")
    assert not rule.accepts("hemidemisemiquaver")
    assert not rule.accepts("adjectitious")
    assert not rule.accepts("")
    assert not rule.accepts("   ")


def test_word_selection_rule_holds_phrases_to_stricter_boundary() -> None:
    # This case pins the approved v2 clause: multiword headwords must clear
    # phrase_min_zipf while single words keep the top-N boundary.
    from open_dictionary.stages.curated_build.word_selection import build_word_selection_rule

    rule = build_word_selection_rule(lang="en", top_n=40000, phrase_min_zipf=4.5)

    assert rule.accepts("encrypt")
    assert rule.accepts("high school")
    assert not rule.accepts("boxing glove")
    assert not rule.accepts("go and boil your head")
    assert rule.as_metadata()["phrase_min_zipf"] == 4.5


def test_word_selection_rule_threshold_scales_with_top_n() -> None:
    # This case verifies the boundary really is derived from the top-N cutoff.
    from open_dictionary.stages.curated_build.word_selection import build_word_selection_rule

    strict = build_word_selection_rule(lang="en", top_n=1000)
    broad = build_word_selection_rule(lang="en", top_n=40000)

    assert strict.min_zipf > broad.min_zipf
    assert strict.accepts("water")
    assert not strict.accepts("encrypt")
    assert broad.accepts("encrypt")


def test_curated_build_stage_applies_word_selection_to_matching_language_only(
    temp_database_url: str,
) -> None:
    # This case verifies selection filters rare headwords in the rule language
    # while leaving other languages untouched.
    from open_dictionary.stages.curated_build.word_selection import build_word_selection_rule

    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="water", lang="English", lang_code="en", pos="noun", source_line=1, gloss="water")
        insert_raw_row(conn, word="adjectitious", lang="English", lang_code="en", pos="adj", source_line=2, gloss="added")
        insert_raw_row(conn, word="grelot", lang="French", lang_code="fr", pos="noun", source_line=3, gloss="small bell")
        conn.commit()

    result = run_curated_build_stage(
        settings=settings,
        word_selection=build_word_selection_rule(lang="en", top_n=40000),
    )

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select array_agg(word order by word) from curated.entries")
            words = cursor.fetchone()[0]
            cursor.execute(
                "select config->'word_selection'->>'rule_version' from meta.pipeline_runs where run_id = %s",
                (result.run_id,),
            )
            recorded_rule = cursor.fetchone()[0]

    assert words == ["grelot", "water"]
    assert result.groups_processed == 2
    assert result.groups_filtered_out == 1
    assert recorded_rule == "word_selection_v2_phrase_threshold"


def test_curated_build_stage_keeps_lower_distinct_headwords_separate(
    temp_database_url: str,
) -> None:
    # Regression: SQL groups the stream by lower(word) while the transform
    # previously recomputed identity with casefold(), which conflates pairs
    # like ß/ss and crashed batched upserts with duplicate entry_ids.
    settings = RuntimeSettings(database_url=temp_database_url)

    with get_connection(settings) as conn:
        apply_foundation(conn)
        insert_raw_row(conn, word="Maße", lang="English", lang_code="en", pos="noun", source_line=1, gloss="measures")
        insert_raw_row(conn, word="Masse", lang="English", lang_code="en", pos="noun", source_line=2, gloss="mass")
        conn.commit()

    result = run_curated_build_stage(settings=settings)

    with get_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select normalized_word, count(distinct entry_id) from curated.entries group by 1 order by 1"
            )
            rows = cursor.fetchall()

    assert result.entries_written == 2
    assert rows == [("masse", 1), ("maße", 1)]
