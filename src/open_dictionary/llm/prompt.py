from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE, LanguageSpec, normalize_language_spec


PROMPT_VERSION = "curated_v1_distribution_fields_v13"
# Generous ceilings: sharding keeps every call at or below the chunk budget,
# so these are safety nets rather than working limits. A tight cap truncates
# mid-JSON and needlessly demotes entries to the compact fallback.
DEFAULT_MAX_TOKENS = 16000
COMPACT_RETRY_MAX_TOKENS = 8000

# Entries with more senses than this are generated in shards: one overview
# call for the entry-level fields plus chunked pos-group calls, assembled and
# validated against the full skeleton afterwards. 28 is evidence-based: a
# 40-sense single call repeatedly failed into the compact fallback, while
# chunks of <=28 senses generated reliably at full quality.
SHARD_SENSE_THRESHOLD = 28
CHUNK_SENSE_BUDGET = 28

MEANING_PRIORITIES = ("core", "common", "rare")

OUTPUT_CONTRACT: dict[str, Any] = {
    "type": "object",
    "required": [
        "headword_summary",
        "memory_hook",
        "study_notes",
        "etymology_note",
        "pos_groups",
    ],
}


@dataclass(frozen=True)
class PromptBundle:
    template_version: str
    resolved_prompt_version: str
    definition_language: LanguageSpec
    system_prompt: str
    compact_retry_system_prompt: str
    overview_system_prompt: str
    chunk_system_prompt: str
    compact_chunk_system_prompt: str
    output_contract: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "template_version": self.template_version,
            "resolved_prompt_version": self.resolved_prompt_version,
            "definition_language": self.definition_language.as_dict(),
            "system_prompt": self.system_prompt,
            "compact_retry_system_prompt": self.compact_retry_system_prompt,
            "overview_system_prompt": self.overview_system_prompt,
            "chunk_system_prompt": self.chunk_system_prompt,
            "compact_chunk_system_prompt": self.compact_chunk_system_prompt,
            "output_contract": self.output_contract,
        }


def build_prompt_bundle(
    *,
    prompt_version: str = PROMPT_VERSION,
    definition_language: LanguageSpec | dict[str, Any] = DEFAULT_DEFINITION_LANGUAGE,
) -> PromptBundle:
    language = normalize_language_spec(definition_language)
    return PromptBundle(
        template_version=prompt_version,
        resolved_prompt_version=resolve_prompt_version(
            prompt_version=prompt_version,
            definition_language=language,
        ),
        definition_language=language,
        system_prompt=build_system_prompt(language),
        compact_retry_system_prompt=build_compact_retry_system_prompt(language),
        overview_system_prompt=build_overview_system_prompt(language),
        chunk_system_prompt=build_chunk_system_prompt(language),
        compact_chunk_system_prompt=build_compact_chunk_system_prompt(language),
        output_contract=OUTPUT_CONTRACT,
    )


def resolve_prompt_version(
    *,
    prompt_version: str,
    definition_language: LanguageSpec | dict[str, Any],
) -> str:
    language = normalize_language_spec(definition_language)
    code_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", language.code)
    return f"{prompt_version}__deflang__{code_suffix}"


def build_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    language_label = f"{language.name} ({language.code})"
    return f"""
You are writing a learner's dictionary entry in {language.name} from curated
Wiktionary data ({language_label} is the required definition language; follow
its standard register and orthography). Return exactly one JSON object and
nothing else.

Produce a learnable entry, not a sense-by-sense translation: the learner needs
one memorable thread and a clear signal of which senses matter. You generate
explanatory fields only — never structural data such as forms or pronunciations.

The JSON object must contain:
- headword_summary: non-empty summary of the whole headword
- memory_hook: one memorable thread in {language.name} connecting the main
  senses, never null. Express the single mental image or core concept in
  natural {language.name} words and show how the main senses grow out of it;
  when the senses genuinely split, give the clearest split.
- study_notes: entry-level learning-strategy and pitfall reminders only
  (false friends, meanings learners wrongly assume, learning order). Never
  repeat collocation, register, or grammar content that belongs in a
  usage_note. Use [] when there is nothing beyond the usage notes.
- etymology_note: short note or null
- pos_groups: exactly the groups of the input skeleton, each containing:
  - pos_group_id and pos: copied verbatim from the input, never translated
    (write "verb", not a translation of it)
  - summary: non-empty summary of this part of speech
  - usage_note: string or null
  - meanings: exactly the sense_id values of the input skeleton, each with:
    - sense_id: copied verbatim
    - priority: "core", "common", or "rare". core = the few senses that carry
      the memory hook, usually 1-3 in the whole entry; common = genuinely
      useful in ordinary reading and conversation; rare = technical, archaic,
      dialectal, or marginal (clients may hide rare senses — never mark a
      sense rare merely because it is hard to explain).
    - short_gloss: short cue string or null
    - learner_explanation: plain {language.name} explanation anchored to the
      memory hook where possible. core and common senses must stand alone;
      rare senses may be one tight sentence pointing back to the core idea.
      Never copy the source gloss mechanically and never invent facts.
    - usage_note: answers "how do I use it", never restates the meaning. Open
      with a concrete sentence pattern or collocation template, then cover
      register, grammar traps, and mistakes {language.name} speakers typically
      make. 2-4 substantial sentences, otherwise null.
    - examples: core senses need 1-2, common exactly 1, rare []. Each item is
      {{"text": one natural everyday sentence in the headword language showing
      the typical pattern, "translation": its natural {language.name}
      rendering}}. Write fresh sentences; never copy source quotations.

Every natural-language field must be in {language.name}, written as natural
prose: never mix stray headword-language words into it (the headword itself,
quoted patterns, and technical terms are the only exceptions). Do not add,
omit, rename, or translate any pos_group_id, pos, or sense_id. When uncertain,
stay conservative. Output valid JSON only.
""".strip()


def build_compact_retry_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    return f"""
You are generating compact learner-facing dictionary explanations in {language.name}.
Return exactly one complete JSON object and nothing else.

Hard constraints:
- every generated natural-language field must be in {language.name}
- follow the orthography/register implied by `{language.code}`
- keep every field short
- headword_summary must be exactly one sentence
- memory_hook must be exactly one sentence, never null
- study_notes must be [] or a one-item string array, never null
- pos_groups[].summary must be exactly one sentence
- meanings[].priority must be exactly one of "core", "common", "rare",
  with at most three "core" senses in the whole entry
- meanings[].learner_explanation must be exactly one sentence
- meanings[].examples must be [] in this compact mode
- use null instead of long commentary when uncertain
- do not use quoted example phrases inside the generated strings
- do not invent or rename pos_group_id, pos, or sense_id
- output valid JSON only

Required JSON keys:
- headword_summary
- memory_hook
- study_notes
- etymology_note
- pos_groups

Each pos_groups item must contain:
- pos_group_id
- pos
- summary
- usage_note
- meanings

Each meanings item must contain:
- sense_id
- priority
- short_gloss
- learner_explanation
- usage_note
- examples
""".strip()


def build_overview_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    return f"""
You are writing the entry-level fields of a learner's dictionary entry.
The full entry is generated separately in parts; you only see a digest of the
headword's parts of speech and source glosses, and you only produce the fields
that need a view of the whole entry.
Every generated natural-language field must be written in {language.name}.
Follow the standard written register and orthography implied by the language tag `{language.code}`.

Return exactly one JSON object and nothing else.

The JSON object must contain exactly these keys:
- core_senses: array of 1-4 objects {{"pos_group_id", "sense_id"}} copied
  verbatim from the digest — the entry's most essential everyday senses, the
  ones a learner must know first. sense_id values repeat across groups, so
  always give the pair. Pick with the whole entry in view and be strict.
- headword_summary: non-empty learner-facing summary of the whole headword in {language.name}
- memory_hook: one memorable thread in {language.name} that connects the headword's
  main senses — the single mental image or core concept a learner should keep.
  When senses radiate from one root idea, name that idea and show how the main
  senses grow out of it. When they genuinely do not, give the clearest split.
  Never null.
- study_notes: array of short entry-level study notes in {language.name} —
  learning strategy and pitfall reminders only (false friends, meanings
  learners wrongly assume, ordering advice). Use [] when there is nothing
  worth saying.
- etymology_note: short note in {language.name} or null

Requirements:
- write natural {language.name} prose; never mix stray headword-language
  words into it (the headword itself and quoted terms are the exceptions)
- never invent facts; stay conservative when the digest is thin
- output valid JSON only
""".strip()


def build_chunk_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    return f"""
You are writing one part of a large learner's dictionary entry in
{language.name} (follow the standard register and orthography of
`{language.code}`). The entry-level summary and memory hook were generated
separately and appear in the input as entry_context: anchor your explanations
to that memory hook so the whole entry reads as one coherent piece. Return
exactly one JSON object and nothing else.

The JSON object must contain exactly one key, pos_groups: exactly the groups
of the input skeleton, each containing:
- pos_group_id and pos: copied verbatim from the input, never translated
  (write "verb", not a translation of it)
- summary: non-empty summary of this part of speech as a whole, even when the
  input contains only part of its senses
- usage_note: string or null
- meanings: exactly the sense_id values of the input skeleton, each with:
  - sense_id: copied verbatim
  - priority: "core", "common", or "rare". The entry-wide core senses were
    already chosen with full visibility and appear in
    entry_context.core_senses as {{"pos_group_id", "sense_id"}} pairs: mark
    exactly those senses core (the ones present in this part) and never mark
    any other sense core. For the rest, common = genuinely useful in ordinary
    usage; rare = technical, archaic, dialectal, or marginal (never mark a
    sense rare merely because it is hard to explain).
  - short_gloss: short cue string or null
  - learner_explanation: plain {language.name} explanation anchored to the
    entry_context memory hook where possible; core and common senses must
    stand alone, rare senses may be one tight sentence. Never copy the source
    gloss mechanically and never invent facts.
  - usage_note: answers "how do I use it", never restates the meaning. Open
    with a concrete pattern or collocation template, then register, grammar
    traps, and typical mistakes; 2-4 substantial sentences, otherwise null.
  - examples: core senses need 1-2, common exactly 1, rare []. Each item is
    {{"text": one natural everyday sentence in the headword language,
    "translation": its natural {language.name} rendering}}. Write fresh
    sentences; never copy source quotations.

Every natural-language field must be in {language.name}, written as natural
prose: never mix stray headword-language words into it (the headword itself,
quoted patterns, and technical terms are the only exceptions). Do not add,
omit, rename, or translate any pos_group_id, pos, or sense_id. Output valid
JSON only.
""".strip()


def build_compact_chunk_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    return f"""
You are generating one compact part of a learner's dictionary entry in {language.name}.
Return exactly one complete JSON object and nothing else.

Hard constraints:
- every generated natural-language field must be in {language.name}
- follow the orthography/register implied by `{language.code}`
- keep every field short
- the JSON object must contain exactly one key: pos_groups
- pos_groups[].summary must be exactly one sentence
- meanings[].priority must be exactly one of "core", "common", "rare"
- meanings[].learner_explanation must be exactly one sentence
- meanings[].examples must be [] in this compact mode
- use null instead of long commentary when uncertain
- do not invent or rename pos_group_id, pos, or sense_id
- output valid JSON only

Each pos_groups item must contain:
- pos_group_id
- pos
- summary
- usage_note
- meanings

Each meanings item must contain:
- sense_id
- priority
- short_gloss
- learner_explanation
- usage_note
- examples
""".strip()


def build_generation_source_payload(
    entry_payload: dict[str, Any],
    *,
    definition_language: LanguageSpec | dict[str, Any] = DEFAULT_DEFINITION_LANGUAGE,
) -> dict[str, Any]:
    language = normalize_language_spec(definition_language)
    pos_groups = []
    for group in entry_payload.get("pos_groups", []):
        pos = group.get("pos")
        etymology_id = group.get("etymology_id")
        senses = []
        for sense in group.get("senses", []):
            source_examples = sense.get("examples") or []
            first_example_text = None
            for example in source_examples:
                if example.get("text"):
                    first_example_text = example["text"]
                    break
            sense_fields = {
                "sense_id": sense.get("sense_id"),
                "gloss": sense.get("gloss"),
                "qualifier": sense.get("qualifier"),
                "labels": sense.get("tags") or None,
                "topics": sense.get("topics") or None,
                "source_example": first_example_text,
            }
            senses.append({key: value for key, value in sense_fields.items() if value is not None})
        pos_groups.append(
            {
                "pos_group_id": build_pos_group_id(pos=pos, etymology_id=etymology_id),
                "pos": pos,
                "etymology_id": etymology_id,
                "meanings": senses,
            }
        )

    return {
        "entry_id": entry_payload.get("entry_id"),
        "headword": entry_payload.get("word"),
        "normalized_headword": entry_payload.get("normalized_word"),
        "headword_language": {
            "code": entry_payload.get("lang_code"),
            "name": entry_payload.get("lang"),
        },
        "definition_language": language.as_dict(),
        "entry_flags": entry_payload.get("entry_flags") or [],
        "etymologies": [
            {
                "etymology_id": group.get("etymology_id"),
                "text": group.get("etymology_text"),
                "pos_members": group.get("member_pos") or [],
            }
            for group in entry_payload.get("etymology_groups", [])
        ],
        "pos_groups": pos_groups,
    }


def build_enrichment_request_payload(
    entry_payload: dict[str, Any],
    *,
    prompt_bundle: PromptBundle,
) -> dict[str, Any]:
    return {
        "entry": build_generation_source_payload(
            entry_payload,
            definition_language=prompt_bundle.definition_language,
        ),
        "prompt_template_version": prompt_bundle.template_version,
        "prompt_version": prompt_bundle.resolved_prompt_version,
        "definition_language": prompt_bundle.definition_language.as_dict(),
    }


def compute_request_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_user_prompt(entry_payload: dict[str, Any]) -> str:
    # The literal word "JSON" must appear in the user message: OpenAI-style
    # backends reject response_format json_object when the checked message
    # text never mentions json, and some relays only check the user turn.
    return (
        "Generated-field source payload (JSON):\n"
        + json.dumps(entry_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def build_overview_digest(entry_source_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "headword": entry_source_payload.get("headword"),
        "headword_language": entry_source_payload.get("headword_language"),
        "definition_language": entry_source_payload.get("definition_language"),
        "etymologies": entry_source_payload.get("etymologies") or [],
        "pos_groups": [
            {
                "pos_group_id": group.get("pos_group_id"),
                "pos": group.get("pos"),
                "senses": [
                    {"sense_id": meaning.get("sense_id"), "gloss": meaning.get("gloss")}
                    for meaning in group.get("meanings", [])
                    if meaning.get("sense_id")
                ],
            }
            for group in entry_source_payload.get("pos_groups", [])
        ],
    }


def build_overview_user_prompt(digest: dict[str, Any]) -> str:
    return (
        "Entry digest for the entry-level fields (JSON):\n"
        + json.dumps(digest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def build_chunk_user_prompt(chunk_payload: dict[str, Any]) -> str:
    return (
        "Partial-entry source payload (JSON):\n"
        + json.dumps(chunk_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def build_pos_group_id(*, pos: Any, etymology_id: Any) -> str:
    pos_text = str(pos or "").strip() or "_"
    etymology_text = str(etymology_id or "").strip() or "_"
    return f"{pos_text}|{etymology_text}"


DEFINITION_LANGUAGE = DEFAULT_DEFINITION_LANGUAGE.as_dict()
SYSTEM_PROMPT = build_system_prompt(DEFAULT_DEFINITION_LANGUAGE)
COMPACT_RETRY_SYSTEM_PROMPT = build_compact_retry_system_prompt(DEFAULT_DEFINITION_LANGUAGE)
