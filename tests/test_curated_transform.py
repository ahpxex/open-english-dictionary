from __future__ import annotations

import copy

from open_dictionary.stages.curated_build import transform as curated


def make_raw_row(
    *,
    row_id: int = 1,
    word: str = "encrypt",
    lang: str = "English",
    lang_code: str = "en",
    pos: str = "verb",
    snapshot_id: str = "snapshot-1",
    run_id: str = "run-1",
    source_line: int = 10,
    payload_overrides: dict | None = None,
) -> dict:
    payload = {
        "word": word,
        "lang": lang,
        "lang_code": lang_code,
        "pos": pos,
        "senses": [{"glosses": ["To conceal information by means of a code or cipher."]}],
        "forms": [{"form": f"{word}s", "tags": ["present", "third-person"]}],
        "sounds": [{"ipa": "/ɪnˈkɹɪpt/"}],
        "derived": [{"word": "encryption"}],
        "related": [{"word": "cipher"}],
        "etymology_text": "From en- + -crypt.",
    }
    if payload_overrides:
        payload.update(payload_overrides)
    return {
        "id": row_id,
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "source_line": source_line,
        "word": word,
        "lang": lang,
        "lang_code": lang_code,
        "pos": pos,
        "payload": payload,
    }


def test_classify_raw_row_keeps_core_pos() -> None:
    # This case protects the default lexical categories from accidental triage.
    decision, triage = curated.classify_raw_row(make_raw_row(pos="noun"))

    assert decision == "keep"
    assert triage is None


def test_classify_raw_row_keeps_name_entries_with_proper_name_flag() -> None:
    # User-approved 2026-08-04: proper-name groups are kept and flagged so
    # selected headwords retain senses like DNS = Domain Name System.
    decision, triage = curated.classify_raw_row(make_raw_row(pos="name"))

    assert decision == "keep_with_flag"
    assert triage is None


def test_pos_flags_for_name_rows_carry_proper_name_flag() -> None:
    assert curated.pos_flags_for_row(make_raw_row(pos="name")) == {"entry_type:proper_name"}


def test_classify_raw_row_triages_relation_only_pos() -> None:
    # This case locks in the default handling for romanization-style records.
    decision, triage = curated.classify_raw_row(make_raw_row(pos="romanization"))

    assert decision == "triage"
    assert triage is not None
    assert triage.suggested_action == "convert_to_relation"


def test_classify_raw_row_triages_dropped_pos() -> None:
    # This case makes the out-of-scope `character` policy explicit.
    decision, triage = curated.classify_raw_row(make_raw_row(pos="character"))

    assert decision == "triage"
    assert triage is not None
    assert triage.suggested_action == "drop"


def test_classify_raw_row_triages_unknown_pos() -> None:
    # This case forces unsupported POS labels to surface as rule updates.
    decision, triage = curated.classify_raw_row(make_raw_row(pos="mystery-pos"))

    assert decision == "triage"
    assert triage is not None
    assert triage.reason_code == "unknown_pos"


def test_classify_raw_row_triages_relation_dominant_form_entry() -> None:
    # This case ensures plural-of / form-of style rows no longer survive as
    # first-class dictionary entries in the tightened curation scope.
    row = make_raw_row(
        pos="adj",
        payload_overrides={
            "senses": [
                {
                    "glosses": ["feminine plural of eterocrono"],
                    "form_of": [{"word": "eterocrono"}],
                    "tags": ["feminine", "plural"],
                }
            ]
        },
    )

    decision, triage = curated.classify_raw_row(row)

    assert decision == "triage"
    assert triage is not None
    assert triage.reason_code == "derived_form_entry"
    assert triage.suggested_action == "convert_to_relation"


def test_classify_raw_row_triages_relation_dominant_alternative_entry() -> None:
    # This case ensures alternative-form records are downgraded out of the main export set.
    row = make_raw_row(
        pos="adj",
        payload_overrides={
            "senses": [
                {
                    "glosses": ["alternative form of чэфы (čɛfə)"],
                    "alt_of": [{"word": "чэфы"}],
                    "tags": ["alternative"],
                }
            ]
        },
    )

    decision, triage = curated.classify_raw_row(row)

    assert decision == "triage"
    assert triage is not None
    assert triage.reason_code == "derived_form_entry"


def test_classify_raw_row_triages_missing_lexical_identity() -> None:
    # This case protects the curated layer from building entries with no stable word key.
    row = make_raw_row(word="", payload_overrides={"word": "   "})
    decision, triage = curated.classify_raw_row(row)

    assert decision == "triage"
    assert triage is not None
    assert triage.reason_code == "missing_lexical_identity"


def test_collect_entry_flags_accumulates_keep_with_flag_pos_values() -> None:
    # This case ensures only still-kept flagged POS values propagate into entry flags.
    rows = [make_raw_row(pos="proverb"), make_raw_row(row_id=2, pos="suffix")]

    flags = curated.collect_entry_flags(rows)

    assert flags == {"entry_type:proverb", "entry_type:affix"}


def test_build_source_summary_collects_provenance_references() -> None:
    # This case verifies that curated entries retain raw run and snapshot provenance.
    rows = [
        make_raw_row(row_id=1, snapshot_id="snapshot-1", run_id="run-1", source_line=10),
        make_raw_row(row_id=2, snapshot_id="snapshot-2", run_id="run-2", source_line=11),
    ]

    summary = curated.build_source_summary(rows)

    assert summary["raw_record_count"] == 2
    assert summary["raw_snapshot_ids"] == ["snapshot-1", "snapshot-2"]
    assert summary["raw_run_ids"] == ["run-1", "run-2"]
    assert summary["raw_record_refs"][0]["raw_record_id"] == 1


def test_build_etymology_groups_splits_distinct_etymologies() -> None:
    # This case ensures multiple origin paths remain separately addressable within one entry.
    rows = [
        make_raw_row(row_id=1, payload_overrides={"etymology_text": "From Latin."}),
        make_raw_row(row_id=2, pos="noun", payload_overrides={"etymology_text": "From Greek."}),
    ]

    groups, lookup = curated.build_etymology_groups(rows)

    assert len(groups) == 2
    assert lookup[1] != lookup[2]


def test_build_etymology_groups_flags_missing_etymology() -> None:
    # This case makes missing etymology explicit instead of silently collapsing it.
    rows = [make_raw_row(payload_overrides={"etymology_text": None})]

    groups, _lookup = curated.build_etymology_groups(rows)

    assert groups[0]["etymology_flags"] == ["missing_etymology"]


def test_normalize_examples_keeps_text_translation_type_and_ref() -> None:
    # This case verifies that example objects survive in a compact but still useful form.
    examples = [
        {
            "text": "She quoted the manual.",
            "translation": "她引用了手册。",
            "type": "quote",
            "ref": "Example ref",
        }
    ]

    normalized = curated.normalize_examples(examples)

    assert normalized == [
        {
            "text": "She quoted the manual.",
            "translation": "她引用了手册。",
            "type": "quote",
            "ref": "Example ref",
            "example_flags": ["quote"],
        }
    ]


def test_normalize_examples_marks_machine_translation_when_english_field_is_used() -> None:
    # This case covers source examples that only expose an English rendering instead of a translation field.
    normalized = curated.normalize_examples(
        [{"text": "text", "english": "English translation", "type": "example"}]
    )

    assert normalized[0]["translation"] == "English translation"
    assert "machine_translation" in normalized[0]["example_flags"]


def test_normalize_forms_dedupes_duplicate_forms() -> None:
    # This case keeps morphological tables from multiplying identical rows.
    forms = [
        {"form": "cats", "tags": ["plural"]},
        {"form": "cats", "tags": ["plural"]},
    ]

    normalized, flags = curated.normalize_forms(forms)

    assert len(normalized) == 1
    assert flags == []


def test_normalize_forms_trims_extreme_form_sets() -> None:
    # This case covers pathological records such as names with huge declension tables.
    forms = [{"form": f"f{i}", "tags": ["generated"]} for i in range(curated.MAX_FORMS + 5)]

    normalized, flags = curated.normalize_forms(forms)

    assert len(normalized) == curated.MAX_FORMS
    assert flags == ["trimmed_form_set"]
    assert "trimmed_form_set" in normalized[0]["form_flags"]


def test_normalize_pronunciations_dedupes_duplicate_items() -> None:
    # This case prevents repeated IPA/audio rows from bloating the curated payload.
    sounds = [
        {"ipa": "/cat/"},
        {"ipa": "/cat/"},
    ]

    normalized, flags = curated.normalize_pronunciations(sounds)

    assert len(normalized) == 1
    assert flags == []


def test_normalize_pronunciations_trims_large_audio_sets() -> None:
    # This case caps pronunciation-heavy entries so they remain product-manageable.
    sounds = [{"ipa": f"/p{i}/", "ogg_url": f"https://example.com/{i}.ogg"} for i in range(curated.MAX_PRONUNCIATIONS + 3)]

    normalized, flags = curated.normalize_pronunciations(sounds)

    assert len(normalized) == curated.MAX_PRONUNCIATIONS
    assert flags == ["trimmed_audio_set"]
    assert "trimmed_audio_set" in normalized[0]["pronunciation_flags"]


def test_normalize_top_level_relations_extracts_supported_relation_types() -> None:
    # This case locks in the conversion of raw relation buckets into normalized relation rows.
    payload = {
        "derived": [{"word": "encryption"}],
        "related": [{"word": "cipher"}],
        "synonyms": [{"word": "encode"}],
        "antonyms": [{"word": "decode"}],
    }

    relations = curated.normalize_top_level_relations(payload)

    assert {item["relation_type"] for item in relations} == {
        "derived_term",
        "related_term",
        "synonym",
        "antonym",
    }


def test_normalize_sense_relations_extracts_form_and_alt_relations() -> None:
    # This case ensures sense-local structural relations survive normalization.
    sense = {
        "form_of": [{"word": "run"}],
        "alt_of": [{"word": "encrypt"}],
    }

    relations = curated.normalize_sense_relations(sense)

    assert {item["relation_type"] for item in relations} == {"form_of", "alternative_of"}


def test_normalize_senses_triages_missing_glosses() -> None:
    # This case verifies that empty senses become triage items rather than silently disappearing.
    row = make_raw_row(payload_overrides={"senses": [{"tags": ["figurative"]}]})
    senses, triage = curated.normalize_senses(row, row["payload"])

    assert senses == []
    assert len(triage) == 1
    assert triage[0].reason_code == "missing_gloss"


def test_normalize_senses_sets_figurative_and_archaic_flags() -> None:
    # This case keeps important semantic usage markers visible in the curated payload.
    row = make_raw_row(
        payload_overrides={
            "senses": [
                {
                    "glosses": ["A figurative gloss."],
                    "tags": ["figurative", "obsolete"],
                }
            ]
        }
    )

    senses, triage = curated.normalize_senses(row, row["payload"])

    assert triage == []
    assert senses[0]["sense_flags"] == ["archaic", "figurative"]


def test_build_pos_groups_merges_rows_by_pos_and_etymology() -> None:
    # This case verifies the word-centric contract: multiple raw rows of the same
    # POS and etymology collapse into one POS group.
    row1 = make_raw_row(row_id=1, pos="verb")
    row2 = make_raw_row(row_id=2, pos="verb", source_line=11, payload_overrides={"derived": [{"word": "reencrypt"}]})
    etymology_groups, lookup = curated.build_etymology_groups([row1, row2])

    pos_groups, relations = curated.build_pos_groups([row1, row2], lookup, [])

    assert len(pos_groups) == 1
    assert pos_groups[0]["pos"] == "verb"
    assert len(relations) >= 2


def test_build_curated_entry_merges_multiple_pos_into_single_entry() -> None:
    # This case is the direct statement of the one-word-one-entry rule.
    verb_row = make_raw_row(row_id=1, pos="verb")
    noun_row = make_raw_row(row_id=2, pos="noun", payload_overrides={"senses": [{"glosses": ["An act of encryption."]}]})

    output = curated.build_curated_entry([verb_row, noun_row])

    assert output.entry is not None
    assert output.entry["word"] == "encrypt"
    assert {group["pos"] for group in output.entry["pos_groups"]} == {"noun", "verb"}


def test_build_curated_entry_returns_only_triage_when_all_rows_are_out_of_scope() -> None:
    # This case ensures the system can represent "no curated entry" cleanly when
    # every raw row falls outside the V1 product boundary.
    rows = [make_raw_row(pos="character", word="倦", lang="Japanese", lang_code="ja")]

    output = curated.build_curated_entry(rows)

    assert output.entry is None
    assert output.relations == []
    assert len(output.triage_items) == 1


def test_build_curated_entry_returns_only_triage_for_relation_dominant_form_entries() -> None:
    # This case verifies that grammatical-form-only entries stop at triage and
    # do not leak into the main curated export.
    rows = [
        make_raw_row(
            pos="adj",
            payload_overrides={
                "senses": [
                    {
                        "glosses": ["feminine plural of eterocrono"],
                        "form_of": [{"word": "eterocrono"}],
                    }
                ]
            },
        )
    ]

    output = curated.build_curated_entry(rows)

    assert output.entry is None
    assert output.relations == []
    assert any(item.reason_code == "derived_form_entry" for item in output.triage_items)


def test_build_entry_hash_is_stable_for_equivalent_payloads() -> None:
    # This case prevents accidental hash churn when the same logical entry is rebuilt.
    entry = {
        "entry_id": "id",
        "word": "encrypt",
        "normalized_word": "encrypt",
        "lang": "English",
        "lang_code": "en",
        "entry_flags": [],
        "source_summary": {},
        "etymology_groups": [],
        "pos_groups": [],
    }

    first = curated.build_entry_hash(entry)
    second = curated.build_entry_hash(copy.deepcopy(entry))

    assert first == second
