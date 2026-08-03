from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from open_dictionary.contracts import DEFAULT_DEFINITION_LANGUAGE, LanguageSpec, normalize_language_spec


PROMPT_VERSION = "curated_v1_distribution_fields_v8"
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
You are writing a learner's dictionary entry from curated Wiktionary data.
The headword language may vary between entries.
The required definition language for this run is {language_label}.
Every generated natural-language field must be written in {language.name}.
Follow the standard written register and orthography implied by the language tag `{language.code}`.

Return exactly one JSON object and nothing else.

You are only responsible for the generated explanatory fields.
Do not repeat or invent deterministic structural fields such as forms, pronunciations, provenance, or relation tables.

Your goal is a learnable entry, not a mechanical sense-by-sense translation of
the source. A learner cannot memorize dozens of parallel senses; they need one
thread to hold onto and a clear signal about which senses matter.

The JSON object must contain:
- headword_summary: non-empty learner-facing summary of the whole headword in {language.name}
- memory_hook: one memorable thread in {language.name} that connects the headword's
  main senses — the single mental image or core concept a learner should keep.
  When senses radiate from one root idea, name that idea and show how the main
  senses grow out of it. When they genuinely do not, give the clearest split
  (for example "两条主线：河岸的岸 / 存钱的银行"). Never null.
- study_notes: array of short study notes in {language.name}. Study notes are
  entry-level learning strategy and pitfall reminders only (false friends,
  meanings learners wrongly assume, ordering advice). Never repeat collocation,
  register, or grammar information that belongs in a usage_note. Use [] when
  there is nothing beyond the usage notes.
- etymology_note: short note in {language.name} or null
- pos_groups: array with exactly the same pos values as the input skeleton
  - pos_group_id
  - pos
  - summary: non-empty summary for this part of speech in {language.name}
  - usage_note: {language.name} string or null
  - meanings: array with exactly the same sense_id values as the input skeleton
    - sense_id
    - priority: exactly one of "core", "common", "rare"
    - short_gloss: short cue string in {language.name} or null
    - learner_explanation: natural-language explanation in {language.name}
    - usage_note: {language.name} string or null
    - examples: array of {{"text", "translation"}} objects

How to assign priority:
- "core": the senses a learner must know first — the ones that carry the memory
  hook. Be strict: usually 1 to 3 senses across the whole entry, only more when
  the headword genuinely has more independent everyday meanings.
- "common": genuinely useful in ordinary reading and conversation, learned after
  the core senses.
- "rare": technical, archaic, dialectal, or marginal senses. Clients may hide
  these by default, so never mark a sense "rare" merely because it is hard to
  explain.

How to write learner_explanation:
- explain the sense in plain {language.name}, anchored to the memory hook where
  possible, so related senses read as extensions of one idea rather than
  isolated definitions
- for "core" and "common" senses, be concrete enough to stand alone
- for "rare" senses, one tight sentence is enough; point back to the core idea
  when that helps ("由『打结』引申的航海用法")
- never copy the source gloss mechanically, and never invent facts; when
  uncertain, stay conservative

How to write examples:
- "core" senses must have 1-2 examples; "common" senses must have exactly 1;
  "rare" senses get an empty array
- "text" is one natural, everyday sentence in the headword language that shows
  the sense's typical collocation or sentence pattern — the kind of sentence a
  learner could reuse. Keep it short and self-contained.
- "translation" renders that sentence in {language.name}, natural rather than
  word-for-word
- write fresh sentences; do not copy quotations from the source payload

How to write usage_note (both the sense level and the pos-group level):
- a usage_note answers "how do I use it", never "what does it mean" — do not
  restate or paraphrase the learner_explanation
- when present, open with a concrete sentence pattern or collocation template
  (for example "give up doing sth，而不是 give up to do sth"), then cover
  register (口语/书面/正式), grammar traps, and the mistakes speakers of
  {language.name} typically make (false friends, easily-confused words)
- write 2-4 full sentences with real substance, not a vague label like
  "常用于口语"
- only include one when you have something concrete beyond the explanation;
  otherwise use null

Hard requirements:
- copy pos_group_id, pos, and sense_id values verbatim from the input, in
  their original language and spelling; never translate them (write "verb",
  not a translation of it)
- do not invent or rename pos values
- do not invent or rename pos_group_id values
- do not invent or rename sense_id values
- do not omit any pos group from the input
- do not omit any sense_id from the input
- short_gloss is only a helper field; learner_explanation is the main field
- if the headword language and definition language happen to be the same, still paraphrase the curated source instead of copying it mechanically
- output valid JSON only

Required output shape:
{{
  "headword_summary": "<non-empty summary in {language.name}>",
  "memory_hook": "<one memorable thread in {language.name}, never null>",
  "study_notes": ["<short study note in {language.name}>"],
  "etymology_note": "<short etymology note in {language.name} or null>",
  "pos_groups": [
    {{
      "pos_group_id": "<exactly copied from input>",
      "pos": "<exactly copied from input>",
      "summary": "<non-empty part-of-speech summary in {language.name}>",
      "usage_note": "<usage note in {language.name} or null>",
      "meanings": [
        {{
          "sense_id": "<exactly copied from input>",
          "priority": "<core | common | rare>",
          "short_gloss": "<short cue in {language.name} or null>",
          "learner_explanation": "<explanation in {language.name}>",
          "usage_note": "<usage note in {language.name} or null>",
          "examples": [
            {{
              "text": "<one everyday sentence in the headword language>",
              "translation": "<its {language.name} translation>"
            }}
          ]
        }}
      ]
    }}
  ]
}}
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
- never invent facts; stay conservative when the digest is thin
- output valid JSON only
""".strip()


def build_chunk_system_prompt(definition_language: LanguageSpec | dict[str, Any]) -> str:
    language = normalize_language_spec(definition_language)
    return f"""
You are writing one part of a large learner's dictionary entry.
The entry-level fields (summary and memory hook) were already generated and
are included in the input as entry_context; anchor your explanations to that
memory hook so the whole entry reads as one coherent piece.
Every generated natural-language field must be written in {language.name}.
Follow the standard written register and orthography implied by the language tag `{language.code}`.

Return exactly one JSON object and nothing else.

The JSON object must contain exactly one key:
- pos_groups: array with exactly the same pos values as the input skeleton
  - pos_group_id
  - pos
  - summary: non-empty summary for this part of speech in {language.name}.
    When the input contains only part of a group's senses, still summarize the
    part of speech as a whole.
  - usage_note: {language.name} string or null
  - meanings: array with exactly the same sense_id values as the input skeleton
    - sense_id
    - priority: exactly one of "core", "common", "rare"
    - short_gloss: short cue string in {language.name} or null
    - learner_explanation: natural-language explanation in {language.name}
    - usage_note: {language.name} string or null
    - examples: array of {{"text", "translation"}} objects

How to assign priority:
- "core": the senses that carry the entry's memory hook. Be strict: at most a
  few in the whole entry, so mark a sense "core" only when it clearly belongs
  to the everyday heart of the word.
- "common": genuinely useful in ordinary reading and conversation.
- "rare": technical, archaic, dialectal, or marginal senses. Clients may hide
  these by default, so never mark a sense "rare" merely because it is hard to
  explain.

How to write learner_explanation:
- explain the sense in plain {language.name}, anchored to the entry_context
  memory hook where possible
- for "core" and "common" senses, be concrete enough to stand alone
- for "rare" senses, one tight sentence is enough
- never copy the source gloss mechanically, and never invent facts

How to write examples:
- "core" senses must have 1-2 examples; "common" senses must have exactly 1;
  "rare" senses get an empty array
- "text" is one natural, everyday sentence in the headword language showing
  the sense's typical collocation or pattern; "translation" renders it in
  {language.name}, natural rather than word-for-word
- write fresh sentences; do not copy quotations from the source payload

How to write usage_note (both the sense level and the pos-group level):
- a usage_note answers "how do I use it", never "what does it mean" — do not
  restate or paraphrase the learner_explanation
- when present, open with a concrete sentence pattern or collocation template,
  then cover register, grammar traps, and the mistakes speakers of
  {language.name} typically make
- write 2-4 full sentences with real substance; otherwise use null

Hard requirements:
- copy pos_group_id, pos, and sense_id values verbatim from the input, in
  their original language and spelling; never translate them (write "verb",
  not a translation of it)
- do not invent or rename pos values
- do not invent or rename pos_group_id values
- do not invent or rename sense_id values
- do not omit any pos group from the input
- do not omit any sense_id from the input
- output valid JSON only
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
            senses.append(
                {
                    "sense_id": sense.get("sense_id"),
                    "gloss": sense.get("gloss"),
                    "raw_gloss": sense.get("raw_gloss"),
                    "qualifier": sense.get("qualifier"),
                    "labels": sense.get("tags") or [],
                    "topics": sense.get("topics") or [],
                    "examples": [
                        {
                            "text": example.get("text"),
                            "translation": example.get("translation"),
                            "type": example.get("type"),
                            "ref": example.get("ref"),
                        }
                        for example in sense.get("examples", [])
                    ],
                }
            )
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
        + json.dumps(entry_payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def build_overview_digest(entry_source_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "headword": entry_source_payload.get("headword"),
        "headword_language": entry_source_payload.get("headword_language"),
        "definition_language": entry_source_payload.get("definition_language"),
        "etymologies": entry_source_payload.get("etymologies") or [],
        "pos_groups": [
            {
                "pos": group.get("pos"),
                "glosses": [
                    meaning.get("gloss")
                    for meaning in group.get("meanings", [])
                    if meaning.get("gloss")
                ],
            }
            for group in entry_source_payload.get("pos_groups", [])
        ],
    }


def build_overview_user_prompt(digest: dict[str, Any]) -> str:
    return (
        "Entry digest for the entry-level fields (JSON):\n"
        + json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True)
    )


def build_chunk_user_prompt(chunk_payload: dict[str, Any]) -> str:
    return (
        "Partial-entry source payload (JSON):\n"
        + json.dumps(chunk_payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def build_pos_group_id(*, pos: Any, etymology_id: Any) -> str:
    pos_text = str(pos or "").strip() or "_"
    etymology_text = str(etymology_id or "").strip() or "_"
    return f"{pos_text}|{etymology_text}"


DEFINITION_LANGUAGE = DEFAULT_DEFINITION_LANGUAGE.as_dict()
SYSTEM_PROMPT = build_system_prompt(DEFAULT_DEFINITION_LANGUAGE)
COMPACT_RETRY_SYSTEM_PROMPT = build_compact_retry_system_prompt(DEFAULT_DEFINITION_LANGUAGE)
