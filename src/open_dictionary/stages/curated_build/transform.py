from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5


KEEP_POS = {"noun", "verb", "adj", "adv", "pron", "num", "prep", "phrase", "intj", "det"}
KEEP_WITH_FLAG_POS = {
    "proverb": "entry_type:proverb",
    "suffix": "entry_type:affix",
    "prefix": "entry_type:affix",
}
TRIAGE_WITH_FLAG_POS = {
    "name": ("entry_type:proper_name", "defer"),
}
RELATION_POS = {
    "romanization": "convert_to_relation",
    "soft-redirect": "convert_to_relation",
    "hard-redirect": "convert_to_relation",
}
DROP_POS = {"character", "symbol"}

MAX_FORMS = 48
MAX_PRONUNCIATIONS = 8
MAX_RELATIONS_PER_TYPE = 24


@dataclass(frozen=True)
class TriageItem:
    lang_code: str | None
    word: str | None
    reason_code: str
    severity: str
    suggested_action: str
    raw_record_refs: list[dict[str, Any]]
    payload: dict[str, Any]


@dataclass(frozen=True)
class CuratedBuildOutput:
    entry: dict[str, Any] | None
    relations: list[dict[str, Any]]
    triage_items: list[TriageItem]


def build_curated_entry(raw_rows: list[dict[str, Any]]) -> CuratedBuildOutput:
    triage_items: list[TriageItem] = []
    kept_rows: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        decision, triage = classify_raw_row(raw_row)
        if triage is not None:
            triage_items.append(triage)
        if decision in {"keep", "keep_with_flag"}:
            kept_rows.append(raw_row)

    if not kept_rows:
        return CuratedBuildOutput(entry=None, relations=[], triage_items=triage_items)

    first = kept_rows[0]
    lang_code = normalize_text(first.get("lang_code")) or "_"
    # The entry identity key must be exactly the stream grouping key computed
    # by the reader's SQL (lower(word)). Recomputing it here with a different
    # Unicode normalization would let two SQL groups collide on one entry_id.
    normalized_word = (
        normalize_nullable_text(first.get("normalized_word"))
        or normalize_word(first.get("word"))
        or "_"
    )
    word = select_display_word(kept_rows)
    lang = select_display_lang(kept_rows)
    entry_id = str(uuid5(NAMESPACE_URL, f"{lang_code}|{normalized_word}"))

    entry_flags = sorted(collect_entry_flags(kept_rows))
    source_summary = build_source_summary(kept_rows)
    etymology_groups, etymology_lookup = build_etymology_groups(kept_rows)
    pos_groups, relations = build_pos_groups(kept_rows, etymology_lookup, triage_items)

    entry = {
        "entry_id": entry_id,
        "word": word,
        "normalized_word": normalized_word,
        "lang": lang,
        "lang_code": lang_code,
        "entry_flags": entry_flags,
        "source_summary": source_summary,
        "etymology_groups": etymology_groups,
        "pos_groups": pos_groups,
    }

    normalized_relations = [
        {
            "entry_id": entry_id,
            "relation_type": relation["relation_type"],
            "target_word": relation["target_word"],
            "target_lang_code": relation.get("target_lang_code"),
            "source_scope": relation["source_scope"],
            "payload": relation,
        }
        for relation in relations
    ]

    return CuratedBuildOutput(
        entry=entry,
        relations=normalized_relations,
        triage_items=triage_items,
    )


def classify_raw_row(raw_row: dict[str, Any]) -> tuple[str, TriageItem | None]:
    payload = decode_payload(raw_row)
    word = normalize_word(payload.get("word"))
    lang_code = normalize_text(payload.get("lang_code"))
    pos = normalize_text(payload.get("pos"))
    ref = [build_raw_record_ref(raw_row)]

    if not word or not lang_code:
        return (
            "triage",
            TriageItem(
                lang_code=lang_code,
                word=word,
                reason_code="missing_lexical_identity",
                severity="high",
                suggested_action="defer",
                raw_record_refs=ref,
                payload={"pos": pos},
            ),
        )

    dominant_relation = detect_relation_dominant_entry(payload)
    if dominant_relation is not None:
        return (
            "triage",
            TriageItem(
                lang_code=lang_code,
                word=word,
                reason_code="derived_form_entry",
                severity="medium",
                suggested_action="convert_to_relation",
                raw_record_refs=ref,
                payload={"pos": pos, "dominant_relation": dominant_relation},
            ),
        )

    if pos in KEEP_POS:
        return "keep", None

    if pos in KEEP_WITH_FLAG_POS:
        return "keep_with_flag", None

    if pos in TRIAGE_WITH_FLAG_POS:
        flag, action = TRIAGE_WITH_FLAG_POS[pos]
        return (
            "triage",
            TriageItem(
                lang_code=lang_code,
                word=word,
                reason_code="record_type_out_of_scope",
                severity="medium",
                suggested_action=action,
                raw_record_refs=ref,
                payload={"pos": pos, "entry_flag": flag},
            ),
        )

    if pos in RELATION_POS:
        return (
            "triage",
            TriageItem(
                lang_code=lang_code,
                word=word,
                reason_code="record_type_out_of_scope",
                severity="medium",
                suggested_action=RELATION_POS[pos],
                raw_record_refs=ref,
                payload={"pos": pos},
            ),
        )

    if pos in DROP_POS:
        return (
            "triage",
            TriageItem(
                lang_code=lang_code,
                word=word,
                reason_code="record_type_out_of_scope",
                severity="low",
                suggested_action="drop",
                raw_record_refs=ref,
                payload={"pos": pos},
            ),
        )

    return (
        "triage",
        TriageItem(
            lang_code=lang_code,
            word=word,
            reason_code="unknown_pos",
            severity="high",
            suggested_action="requires_rule_update",
            raw_record_refs=ref,
            payload={"pos": pos},
        ),
    )


def collect_entry_flags(raw_rows: list[dict[str, Any]]) -> set[str]:
    flags: set[str] = set()
    for raw_row in raw_rows:
        pos = normalize_text(raw_row.get("pos"))
        flag = KEEP_WITH_FLAG_POS.get(pos)
        if flag:
            flags.add(flag)
    return flags


def detect_relation_dominant_entry(payload: dict[str, Any]) -> str | None:
    senses = payload.get("senses")
    if not isinstance(senses, list) or not senses:
        return None

    dominant_types: set[str] = set()
    lexical_sense_found = False

    for sense in senses:
        if not isinstance(sense, dict):
            continue
        relations = normalize_sense_relations(sense)
        relation_types = {relation["relation_type"] for relation in relations}
        gloss = first_non_empty_text(sense.get("glosses")) or first_non_empty_text(sense.get("raw_glosses"))

        if relation_types & {"form_of", "alternative_of"}:
            dominant_types |= relation_types & {"form_of", "alternative_of"}
            if gloss and not looks_like_derived_form_gloss(gloss):
                lexical_sense_found = True
        else:
            lexical_sense_found = True

    if lexical_sense_found or not dominant_types:
        return None

    if "form_of" in dominant_types:
        return "form_of"
    if "alternative_of" in dominant_types:
        return "alternative_of"
    return None


def build_source_summary(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot_ids = sorted({str(raw_row["snapshot_id"]) for raw_row in raw_rows if raw_row.get("snapshot_id")})
    run_ids = sorted({str(raw_row["run_id"]) for raw_row in raw_rows if raw_row.get("run_id")})
    raw_record_refs = [build_raw_record_ref(raw_row) for raw_row in raw_rows]
    return {
        "raw_record_count": len(raw_rows),
        "raw_snapshot_ids": snapshot_ids,
        "raw_run_ids": run_ids,
        "raw_record_refs": raw_record_refs,
    }


def build_etymology_groups(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    groups: dict[str, dict[str, Any]] = {}
    row_lookup: dict[int, str] = {}

    for raw_row in raw_rows:
        payload = decode_payload(raw_row)
        etymology_text = normalize_nullable_text(payload.get("etymology_text"))
        etymology_key = etymology_text or "__missing__"
        etymology_id = f"et{len(groups) + 1}" if etymology_key not in groups else groups[etymology_key]["etymology_id"]
        if etymology_key not in groups:
            groups[etymology_key] = {
                "etymology_id": etymology_id,
                "etymology_text": etymology_text,
                "etymology_flags": ["missing_etymology"] if etymology_text is None else [],
                "member_pos": set(),
                "source_refs": [],
            }
        groups[etymology_key]["member_pos"].add(normalize_text(raw_row.get("pos")) or "_")
        groups[etymology_key]["source_refs"].append({"raw_record_id": raw_row["id"]})
        row_lookup[int(raw_row["id"])] = etymology_id

    etymology_groups = []
    for group in groups.values():
        etymology_groups.append(
            {
                "etymology_id": group["etymology_id"],
                "etymology_text": group["etymology_text"],
                "etymology_flags": sorted(group["etymology_flags"]),
                "member_pos": sorted(pos for pos in group["member_pos"] if pos != "_"),
                "source_refs": group["source_refs"],
            }
        )

    etymology_groups.sort(key=lambda item: item["etymology_id"])
    return etymology_groups, row_lookup


def build_pos_groups(
    raw_rows: list[dict[str, Any]],
    etymology_lookup: dict[int, str],
    triage_items: list[TriageItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        payload = decode_payload(raw_row)
        pos = normalize_text(raw_row.get("pos")) or "_"
        etymology_id = etymology_lookup.get(int(raw_row["id"]))
        group_key = (pos, etymology_id)

        if group_key not in groups:
            groups[group_key] = {
                "pos": pos,
                "pos_flags": sorted(pos_flags_for_row(raw_row)),
                "etymology_id": etymology_id,
                "senses": [],
                "forms": [],
                "pronunciations": [],
                "relations": [],
            }

        group = groups[group_key]
        forms, form_flags = normalize_forms(payload.get("forms") or [])
        pronunciations, pronunciation_flags = normalize_pronunciations(payload.get("sounds") or [])
        group["forms"].extend(forms)
        group["pronunciations"].extend(pronunciations)
        if form_flags:
            group["pos_flags"] = sorted(set(group["pos_flags"]) | set(form_flags))
        if pronunciation_flags:
            group["pos_flags"] = sorted(set(group["pos_flags"]) | set(pronunciation_flags))

        raw_relations = normalize_top_level_relations(payload)
        group["relations"].extend(raw_relations)
        relations.extend(raw_relations)

        senses, sense_triage = normalize_senses(raw_row, payload)
        group["senses"].extend(senses)
        triage_items.extend(sense_triage)

    pos_groups = []
    for (pos, etymology_id), group in groups.items():
        group["forms"] = dedupe_forms(group["forms"])
        group["pronunciations"] = dedupe_pronunciations(group["pronunciations"])
        deduped_relations = dedupe_relations(group["relations"])
        group["relations"] = deduped_relations
        group["senses"] = dedupe_senses(group["senses"])
        pos_groups.append(group)

    pos_groups.sort(key=lambda item: (item["pos"], item["etymology_id"] or ""))
    return pos_groups, dedupe_relations(relations)


def normalize_senses(raw_row: dict[str, Any], payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[TriageItem]]:
    results: list[dict[str, Any]] = []
    triage_items: list[TriageItem] = []
    for index, sense in enumerate(payload.get("senses") or [], start=1):
        if not isinstance(sense, dict):
            continue
        gloss = first_non_empty_text(sense.get("glosses")) or first_non_empty_text(sense.get("raw_glosses"))
        if gloss is None:
            triage_items.append(
                TriageItem(
                    lang_code=normalize_text(payload.get("lang_code")),
                    word=normalize_word(payload.get("word")),
                    reason_code="missing_gloss",
                    severity="medium",
                    suggested_action="defer",
                    raw_record_refs=[build_raw_record_ref(raw_row)],
                    payload={"sense_index": index, "pos": normalize_text(payload.get("pos"))},
                )
            )
            continue
        raw_gloss = first_non_empty_text(sense.get("raw_glosses"))
        sense_relations = normalize_sense_relations(sense)
        results.append(
            {
                "sense_id": f"s{index}",
                "gloss": gloss,
                "raw_gloss": raw_gloss,
                "tags": normalize_string_list(sense.get("tags")),
                "qualifier": normalize_nullable_text(sense.get("qualifier")),
                "topics": normalize_string_list(sense.get("topics")),
                "examples": normalize_examples(sense.get("examples") or []),
                "relations": sense_relations,
                "sense_flags": normalize_sense_flags(sense),
            }
        )
    return results, triage_items


def normalize_examples(examples: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        text = normalize_nullable_text(example.get("text"))
        if not text:
            continue
        translation = normalize_nullable_text(example.get("translation") or example.get("english"))
        example_type = normalize_nullable_text(example.get("type"))
        ref = normalize_nullable_text(example.get("ref"))
        flags: list[str] = []
        if example_type == "quote":
            flags.append("quote")
        if translation and example.get("english"):
            flags.append("machine_translation")
        normalized.append(
            {
                "text": text,
                "translation": translation,
                "type": example_type,
                "ref": ref,
                "example_flags": flags,
            }
        )
    return normalized


def normalize_forms(forms: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    flags: set[str] = set()
    for form in forms:
        if not isinstance(form, dict):
            continue
        surface = normalize_nullable_text(form.get("form"))
        if not surface:
            continue
        normalized.append(
            {
                "form": surface,
                "tags": normalize_string_list(form.get("tags")),
                "roman": normalize_nullable_text(form.get("roman")),
                "form_flags": [],
            }
        )

    deduped = dedupe_forms(normalized)
    if len(deduped) > MAX_FORMS:
        flags.add("trimmed_form_set")
        deduped = deduped[:MAX_FORMS]
        for form in deduped:
            form["form_flags"] = sorted(set(form["form_flags"]) | {"trimmed_form_set"})
    return deduped, sorted(flags)


def normalize_pronunciations(sounds: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    flags: set[str] = set()
    for sound in sounds:
        if not isinstance(sound, dict):
            continue
        ipa = normalize_nullable_text(sound.get("ipa"))
        pronunciation_text = normalize_nullable_text(sound.get("zh_pron") or sound.get("other") or sound.get("enpr"))
        audio_url = normalize_nullable_text(sound.get("ogg_url") or sound.get("mp3_url") or sound.get("wav_url"))
        audio_format = None
        if normalize_nullable_text(sound.get("ogg_url")):
            audio_format = "ogg"
        elif normalize_nullable_text(sound.get("mp3_url")):
            audio_format = "mp3"
        elif normalize_nullable_text(sound.get("wav_url")):
            audio_format = "wav"
        if not any((ipa, pronunciation_text, audio_url)):
            continue
        normalized.append(
            {
                "ipa": ipa,
                "pronunciation_text": pronunciation_text,
                "audio_url": audio_url,
                "audio_format": audio_format,
                "tags": normalize_string_list(sound.get("tags")),
                "pronunciation_flags": ["preferred_audio"] if audio_url else [],
            }
        )

    deduped = dedupe_pronunciations(normalized)
    if len(deduped) > MAX_PRONUNCIATIONS:
        flags.add("trimmed_audio_set")
        deduped = deduped[:MAX_PRONUNCIATIONS]
        for item in deduped:
            item["pronunciation_flags"] = sorted(set(item["pronunciation_flags"]) | {"trimmed_audio_set"})
    return deduped, sorted(flags)


def normalize_top_level_relations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    relation_fields = {
        "derived": "derived_term",
        "related": "related_term",
        "synonyms": "synonym",
        "antonyms": "antonym",
    }
    for field, relation_type in relation_fields.items():
        for relation in payload.get(field) or []:
            item = normalize_relation_item(relation)
            if item is None:
                continue
            relations.append(
                {
                    "relation_type": relation_type,
                    "target_word": item["target_word"],
                    "target_lang_code": item.get("target_lang_code"),
                    "relation_flags": item.get("relation_flags", []),
                    "source_scope": "entry",
                }
            )
    return cap_relations(relations)


def normalize_sense_relations(sense: dict[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for field, relation_type in (("form_of", "form_of"), ("alt_of", "alternative_of"), ("compound_of", "compound_of")):
        for relation in sense.get(field) or []:
            item = normalize_relation_item(relation)
            if item is None:
                continue
            relations.append(
                {
                    "relation_type": relation_type,
                    "target_word": item["target_word"],
                    "target_lang_code": item.get("target_lang_code"),
                    "relation_flags": item.get("relation_flags", []),
                    "source_scope": "sense",
                }
            )
    return dedupe_relations(relations)


def normalize_relation_item(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        target_word = normalize_nullable_text(value.get("word"))
        target_lang_code = normalize_nullable_text(value.get("lang_code"))
    else:
        target_word = normalize_nullable_text(value)
        target_lang_code = None
    if not target_word:
        return None
    return {
        "target_word": target_word,
        "target_lang_code": target_lang_code,
        "relation_flags": [],
    }


def normalize_sense_flags(sense: dict[str, Any]) -> list[str]:
    flags: set[str] = set()
    tags = normalize_string_list(sense.get("tags"))
    if "figuratively" in tags or "figurative" in tags:
        flags.add("figurative")
    if "archaic" in tags or "obsolete" in tags:
        flags.add("archaic")
    return sorted(flags)


def pos_flags_for_row(raw_row: dict[str, Any]) -> set[str]:
    pos = normalize_text(raw_row.get("pos"))
    flag = KEEP_WITH_FLAG_POS.get(pos)
    return {flag} if flag else set()


def select_display_word(raw_rows: list[dict[str, Any]]) -> str:
    for raw_row in raw_rows:
        word = normalize_nullable_text(raw_row.get("word")) or normalize_nullable_text(decode_payload(raw_row).get("word"))
        if word:
            return word
    return "_"


def select_display_lang(raw_rows: list[dict[str, Any]]) -> str:
    for raw_row in raw_rows:
        lang = normalize_nullable_text(raw_row.get("lang")) or normalize_nullable_text(decode_payload(raw_row).get("lang"))
        if lang:
            return lang
    return "_"


def build_raw_record_ref(raw_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": str(raw_row.get("snapshot_id")),
        "run_id": str(raw_row.get("run_id")),
        "raw_record_id": raw_row.get("id"),
        "source_line": raw_row.get("source_line"),
        "pos": normalize_nullable_text(raw_row.get("pos")),
    }


def normalize_word(value: Any) -> str | None:
    text = normalize_nullable_text(value)
    if text is None:
        return None
    return text.casefold()


def normalize_text(value: Any) -> str | None:
    text = normalize_nullable_text(value)
    if text is None:
        return None
    return text.casefold()


def normalize_nullable_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def looks_like_derived_form_gloss(gloss: str) -> bool:
    normalized = gloss.casefold()
    markers = (
        "alternative form of",
        "alternative spelling of",
        "alternative case form of",
        "plural of",
        "singular of",
        "feminine plural of",
        "masculine plural of",
        "feminine singular of",
        "masculine singular of",
        "dative plural of",
        "genitive plural of",
        "accusative plural of",
        "nominative plural of",
        "inflected infinitive of",
        "infinitive of",
        "participle of",
        "past participle of",
        "present participle of",
        "comparative of",
        "superlative of",
    )
    return any(marker in normalized for marker in markers)


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_nullable_text(item)
        if text:
            result.append(text)
    return dedupe_preserve_order(result)


def first_non_empty_text(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            text = normalize_nullable_text(item)
            if text:
                return text
        return None
    return normalize_nullable_text(value)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def dedupe_forms(forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for form in forms:
        key = (form["form"], tuple(form["tags"]), form.get("roman"))
        if key in seen:
            continue
        seen.add(key)
        result.append(form)
    return result


def dedupe_pronunciations(pronunciations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for item in pronunciations:
        key = (
            item.get("ipa"),
            item.get("pronunciation_text"),
            item.get("audio_url"),
            tuple(item.get("tags", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for relation in relations:
        key = (
            relation["relation_type"],
            relation["target_word"],
            relation.get("target_lang_code"),
            relation["source_scope"],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return cap_relations(result)


def dedupe_senses(senses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for sense in senses:
        key = (
            sense["gloss"],
            tuple(sense["tags"]),
            sense.get("qualifier"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(sense)
    return result


def cap_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_type_count: dict[str, int] = defaultdict(int)
    result: list[dict[str, Any]] = []
    for relation in relations:
        relation_type = relation["relation_type"]
        if per_type_count[relation_type] >= MAX_RELATIONS_PER_TYPE:
            continue
        per_type_count[relation_type] += 1
        result.append(relation)
    return result


def decode_payload(raw_row: dict[str, Any]) -> dict[str, Any]:
    payload = raw_row.get("payload")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError("Raw row payload must be a dict or JSON string")


def build_entry_hash(entry: dict[str, Any]) -> str:
    canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
