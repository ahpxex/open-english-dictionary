# Curation Rulebook

## Status

This document defines the default curation rules for the first rewrite of
Open Dictionary.

The source layer is Wiktionary / Wiktextract raw data.
The target layer is a curated, word-centric dictionary representation.

This is a production rulebook, not a loose set of ideas.
When curation behavior changes, this document must be updated.

## Core Principle

The curated layer is **not** a mirror of Wiktionary.
It is a controlled dictionary product built from Wiktionary data.

The curation objective is:

- preserve useful lexical information
- remove source-site noise
- normalize fragmented records into a stable contract
- support one headword per curated entry
- keep enough provenance to explain or revisit curation decisions later

## Entry Model

### Canonical unit

- One headword equals one curated entry.
- The primary grouping key is `(lang_code, normalized_word)`.
- Part of speech is **not** the top-level identity of the entry.
- Different parts of speech are represented as subgroups under the same entry.

### Implications

- If the same headword appears multiple times with different `pos`, they should
  usually be merged into one curated entry with multiple `pos_groups`.
- If the same headword appears multiple times with multiple etymologies, the
  curated entry should keep separate etymology groups when the distinction is
  semantically meaningful.
- If multiple raw records are obvious duplicates, they should be collapsed.

## Decision Classes

Every raw record or raw field must end up in one of these classes:

- `keep`: carry into the curated entry in normalized form
- `keep_with_flag`: keep, but mark as special or lower-confidence
- `relation`: do not keep as core content; convert into a relation or link
- `drop`: discard from curated output
- `triage`: send to agent-managed review flow for record-level handling

Important:

- `triage` does **not** mean waiting for manual user review
- `triage` is owned by the agent/system
- only rule-level ambiguity should be escalated to the user
- record-level ambiguity should be handled by agent triage queues or exception
  buckets

## POS Policy

### Keep in V1

These are core lexical categories and should be included in the curated layer:

- `noun`
- `verb`
- `adj`
- `adv`
- `pron`
- `num`
- `prep`
- `phrase`
- `intj`
- `det`

### Keep with flag in V1

These may be useful, but should be explicitly marked as special:

- `proverb`
- `suffix`
- `prefix`

Suggested flags:

- `entry_type: proverb`
- `entry_type: affix`

These records should not be silently mixed into ordinary lexical entries.

### Convert to relation in V1

These should generally not become top-level dictionary entries:

- relation-dominant `form_of` entries
- relation-dominant `alternative_of` entries
- `romanization`
- `soft-redirect`
- `hard-redirect`

Suggested handling:

- represent them as aliases, redirect edges, alternate-surface mappings, or
  subordinate form relations
- attach them to a canonical headword when possible

### Triage with flag in V1

These should not be part of the default main export, but they are not
automatically discarded:

- `name`

Suggested handling:

- preserve them through triage or separate buckets
- attach `entry_type: proper_name`
- only promote them into main export through later rule tightening

### Drop by default in V1

These are outside the first product boundary:

- `character`
- `symbol`

Reason:

- they belong to character dictionaries, writing-system references, or symbolic
  reference products more than to the current headword-centric lexical product

### Triage in V1

Any rare or unknown `pos` value not covered above must go to triage.

Examples:

- custom Wiktionary POS labels
- malformed POS labels
- language-specific structural POS values that do not cleanly map into the rule
  set

## Top-Level Field Policy

### Keep

These are primary fields for curated construction:

- `word`
- `lang`
- `lang_code`
- `pos`
- `senses`
- `forms`
- `sounds`
- `etymology_text`

### Keep with normalization

These are useful but must be normalized and often reduced:

- `derived`
- `related`
- `synonyms`
- `antonyms`
- `translations`
- `descendants`
- `hyphenations`
- `hyphenation`

Notes:

- `translations` should not dominate the curated model in V1
- `descendants` are usually too large and should be aggressively reduced or
  deferred
- `sounds` should be reduced to a compact curated pronunciation representation
- `forms` should be normalized into a stable inflection list rather than copied
  verbatim

### Keep only for internal provenance or debugging

These may help explain where curated values came from, but should not usually
  appear in the end-user model:

- `etymology_templates`
- `head_templates`
- `inflection_templates`
- `wikipedia`
- `original_title`
- `title`

### Drop

These should not be part of the curated product model:

- source maintenance categories
- template expansion residue not needed for semantics
- redirect-only scaffolding once canonical links are resolved

In practice this means the following fields are dropped from end-user output by
default:

- `categories`
- raw template scaffolding fields that have already been semantically absorbed

## Sense-Level Field Policy

### Keep

These are sense-defining:

- `glosses`
- `examples`
- `tags`
- `qualifier`
- `topics`
- `form_of`
- `alt_of`

### Keep with caution

These are useful but often noisy:

- `raw_glosses`
- `links`
- `categories`
- `raw_tags`
- `attestations`

Default policy:

- prefer `glosses` over `raw_glosses`
- keep `raw_glosses` only if they preserve meaning missing from `glosses`
- convert `links` into optional metadata only if they prove useful later
- do not expose sense-level `categories` in curated output
- do not expose `attestations` in V1 unless explicitly required

### Convert to relation

- `form_of`
- `alt_of`
- `compound_of`

These should usually be expressed as normalized relations:

- `form_of`
- `alternative_of`
- `compound_of`

### Triage

Sense rows must go to triage when:

- they have no usable `glosses`
- they contain only template residue
- they contain conflicting sense-level structural hints
- they look like malformed redirects or parser artifacts

## Forms Policy

### Keep

Keep forms when they represent genuine inflection or alternate written forms.

Expected fields:

- `form`
- `tags`
- optionally `roman`

### Normalize

Forms must be normalized into:

- the surface form
- a normalized tag set
- optional transliteration or ruby details when meaningful

### Drop or trim

Forms should be reduced when they are clearly excessive or product-irrelevant.

Examples:

- giant name declension tables
- forms that only restate trivial spelling variants without user value
- duplicated forms with the same normalized tag set

### Triage

Send to triage if:

- a record has an extreme number of forms
- forms dominate the record while lexical meaning is weak
- forms conflict across raw duplicates

## Sounds Policy

### Keep

Keep pronunciation data, but in reduced form.

Useful fields include:

- `ipa`
- language-specific pronunciation strings such as `zh_pron`
- `audio`
- `ogg_url`
- `mp3_url`
- `tags`
- `note`

### Normalize

The curated layer should eventually collapse sounds into a smaller model:

- phonetic representations
- language/region tags
- one or a few preferred audio files

### Drop or trim

Do not carry dozens of redundant audio variants into the main curated payload
unless there is a strong product reason.

### Triage

Send to triage if:

- sound data is huge relative to the lexical value of the entry
- the pronunciation fields are structurally irregular
- the entry is mostly pronunciation metadata with minimal lexical content

## Translation Policy

### V1 default

Translations are not core to the first curated dictionary contract.

They may be preserved internally or reduced later, but they should not shape the
core entry model in V1.

### Reason

- translation lists are often massive
- translation quality and coverage are uneven
- the first product goal is a curated headword dictionary, not a translation
  matrix

### Triage

Translation-heavy entries may be triaged if:

- the lexical core is weak and the translation payload is dominant
- translation structure carries meaning that might justify future retention

## Relation Policy

The curated layer should prefer normalized relation edges over raw dump fields.

Potential normalized relation types:

- `derived_term`
- `related_term`
- `synonym`
- `antonym`
- `descendant`
- `form_of`
- `alternative_of`
- `redirect_to`
- `romanization_of`

Relations should be deduplicated and normalized by `(lang_code, word, relation_type)`.

## Merge Policy

### Merge by default

Raw records should be merged into one curated entry when:

- `lang_code` matches
- normalized `word` matches
- the records are not obviously pure redirects

### Keep separate subgroups when needed

The curated entry may still contain separate internal groups for:

- different parts of speech
- separate etymologies
- incompatible sense clusters

### Do not split top-level entry without strong reason

The rulebook is intentionally word-centric.
The system should resist turning every `(word, pos)` pair into a separate top-level row.

## Edge Cases

### 1. Same word, many parts of speech

Default:

- merge into one entry
- store `pos_groups`

### 2. Same word, many etymologies

Default:

- keep one entry
- preserve multiple etymology groups if the senses actually depend on them

### 3. Records with only `form_of` or `alt_of`

Default:

- do not treat as fully independent lexical content
- attach as relations or reduced subordinate content

### 4. Extremely large `forms`

Default:

- keep only meaningful normalized forms
- send pathological cases to triage

### 5. Extremely large `sounds`

Default:

- trim to a compact pronunciation representation
- preserve the raw source link in provenance if needed

### 6. Extremely large `translations`

Default:

- do not include in the main curated structure
- optionally store separately for future use

### 7. Missing `word`, `lang_code`, or `senses`

Default:

- if the entry cannot support a stable curated identity, send to triage or drop
- if it is recoverable from nearby fields, agent triage may repair it

### 8. `name`

Default:

- triage by default with `entry_type: proper_name`
- do not silently mix with ordinary lexical nouns
- only promote selected subclasses later if product scope explicitly expands

### 9. `character`

Default:

- drop from V1 main product
- optionally preserve for future character-dictionary work in a side bucket

### 10. `romanization`

Default:

- convert to relation when a canonical target can be inferred
- otherwise triage

### 11. Redirects

Default:

- do not keep as top-level entries
- convert to alias or redirect relations

### 12. Language-specific pronunciation-heavy entries

Default:

- keep lexical content first
- compress pronunciation aggressively

### 13. Source parser anomalies

Examples:

- malformed tags
- empty sense objects
- duplicated sound objects
- duplicated forms with conflicting tags

Default:

- normalize when deterministic
- triage when not deterministic

### 14. Relation-dominant grammatical forms

Examples:

- plural-of entries
- case-form entries
- participle-of entries
- alternative-form entries

Default:

- do not export as main dictionary entries in V1
- convert into relations or subordinate forms where possible
- triage when relation conversion is not deterministic

## Triage Policy

### Ownership

Record-level triage is owned by the agent/system, not by the user.

The user should only be asked to decide:

- product-level policy changes
- editorial rules that affect many records
- deliberate scope changes

The user should **not** be asked to manually inspect individual raw records as a
normal workflow.

### What triage means

Triage means one of:

- resolve automatically using deterministic fallback rules
- place into a structured exception bucket for later agent processing
- exclude from curated output while preserving provenance

### Triage triggers

Send a record to triage when:

- `pos` is unknown or unsupported
- the record has no stable lexical identity
- multiple raw records conflict in a way not covered by merge rules
- the record is structurally valid but product relevance is unclear
- normalization would otherwise require inventing a new editorial rule

## V1 Curated Schema

The V1 curated schema is a formal contract, not a placeholder.

The system may store this contract either:

- as one primary JSONB payload per curated entry, plus helper metadata columns
- or as a normalized relational model with a materialized JSONB view

Either storage strategy is acceptable.
The logical contract below is mandatory.

### Top-level curated entry

Each curated entry must contain the following top-level fields:

- `entry_id`
- `word`
- `normalized_word`
- `lang`
- `lang_code`
- `entry_flags`
- `source_summary`
- `etymology_groups`
- `pos_groups`

### Top-level field definitions

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `entry_id` | string | yes | Stable entry identifier for the curated layer |
| `word` | string | yes | Canonical display headword |
| `normalized_word` | string | yes | Normalized grouping key used for merge decisions |
| `lang` | string | yes | Human-readable language name |
| `lang_code` | string | yes | Machine-readable language code |
| `entry_flags` | array of strings | yes | Entry-level flags such as `proper_name`, `affix`, `proverb`, `triaged` |
| `source_summary` | object | yes | Minimal provenance summary for the curated entry |
| `etymology_groups` | array of objects | yes | Distinct etymology clusters under the same headword |
| `pos_groups` | array of objects | yes | Part-of-speech groupings under the same headword |

### `source_summary`

`source_summary` must contain:

- `raw_record_count`
- `raw_snapshot_ids`
- `raw_run_ids`
- `raw_record_ids` or an equivalent provenance pointer list

Minimum contract:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `raw_record_count` | integer | yes | Number of raw records merged into this entry |
| `raw_snapshot_ids` | array of strings | yes | Snapshot ids contributing to this entry |
| `raw_run_ids` | array of strings | yes | Raw ingest run ids contributing to this entry |
| `raw_record_refs` | array of objects | yes | Stable references to contributing raw records |

Each `raw_record_ref` should include:

- `snapshot_id`
- `raw_record_id`
- `source_line`
- `pos`

### `etymology_groups`

An etymology group represents a distinct origin path worth preserving.

Each `etymology_group` must contain:

- `etymology_id`
- `etymology_text`
- `etymology_flags`
- `member_pos`
- `source_refs`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `etymology_id` | string | yes | Stable identifier within the entry |
| `etymology_text` | string or null | yes | Normalized etymology text |
| `etymology_flags` | array of strings | yes | Flags such as `missing_etymology`, `merged_etymology`, `conflicted_etymology` |
| `member_pos` | array of strings | yes | POS values attached to this etymology group |
| `source_refs` | array of objects | yes | Raw source references feeding this etymology group |

### `pos_groups`

Each `pos_group` must contain:

- `pos`
- `pos_flags`
- `etymology_id`
- `senses`
- `forms`
- `pronunciations`
- `relations`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `pos` | string | yes | Canonical POS label |
| `pos_flags` | array of strings | yes | Group-level flags such as `proper_name`, `derived_from_form_only`, `triaged_pos` |
| `etymology_id` | string or null | yes | Owning etymology group identifier |
| `senses` | array of objects | yes | Normalized lexical senses |
| `forms` | array of objects | yes | Normalized forms relevant to this POS |
| `pronunciations` | array of objects | yes | Reduced pronunciation payload |
| `relations` | array of objects | yes | Normalized semantic or structural relations |

### `senses`

Each normalized sense must contain:

- `sense_id`
- `gloss`
- `raw_gloss`
- `tags`
- `qualifier`
- `topics`
- `examples`
- `relations`
- `sense_flags`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `sense_id` | string | yes | Stable identifier within the POS group |
| `gloss` | string | yes | Preferred normalized gloss |
| `raw_gloss` | string or null | yes | Optional original gloss when useful |
| `tags` | array of strings | yes | Normalized sense tags |
| `qualifier` | string or null | yes | Preserved qualifier text |
| `topics` | array of strings | yes | Normalized topical labels |
| `examples` | array of objects | yes | Curated examples |
| `relations` | array of objects | yes | Sense-local relations such as `form_of` or `alternative_of` |
| `sense_flags` | array of strings | yes | Flags such as `figurative`, `archaic`, `triaged_sense` |

### `examples`

Each example must contain:

- `text`
- `translation`
- `type`
- `ref`
- `example_flags`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `text` | string | yes | Source example text |
| `translation` | string or null | yes | Any provided translation |
| `type` | string or null | yes | Example/quote label when present |
| `ref` | string or null | yes | Citation text when present |
| `example_flags` | array of strings | yes | Flags such as `quote`, `machine_translation`, `trimmed` |

### `forms`

Each normalized form must contain:

- `form`
- `tags`
- `roman`
- `form_flags`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `form` | string | yes | Surface form |
| `tags` | array of strings | yes | Normalized form tags |
| `roman` | string or null | yes | Optional romanization |
| `form_flags` | array of strings | yes | Flags such as `alternative_spelling`, `trimmed_form_set` |

### `pronunciations`

Each normalized pronunciation must contain:

- `ipa`
- `pronunciation_text`
- `audio_url`
- `audio_format`
- `tags`
- `pronunciation_flags`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `ipa` | string or null | yes | IPA string if available |
| `pronunciation_text` | string or null | yes | Alternate pronunciation text such as `zh_pron` |
| `audio_url` | string or null | yes | Preferred audio URL |
| `audio_format` | string or null | yes | Audio format such as `ogg` or `mp3` |
| `tags` | array of strings | yes | Region or context tags |
| `pronunciation_flags` | array of strings | yes | Flags such as `preferred_audio`, `trimmed_audio_set` |

### `relations`

Each normalized relation must contain:

- `relation_type`
- `target_word`
- `target_lang_code`
- `relation_flags`
- `source_scope`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `relation_type` | string | yes | One of the normalized relation types |
| `target_word` | string | yes | Target surface form |
| `target_lang_code` | string or null | yes | Target language code when known |
| `relation_flags` | array of strings | yes | Flags such as `inferred`, `sense_local`, `source_noisy` |
| `source_scope` | string | yes | `entry`, `pos_group`, or `sense` |

### Required relation types in V1

The implementation must support at least these normalized relation types:

- `derived_term`
- `related_term`
- `synonym`
- `antonym`
- `form_of`
- `alternative_of`
- `redirect_to`
- `romanization_of`

`descendant` may exist in storage but does not need to be promoted into the
main curated payload in V1.

## V1 Triage Schema

Triage is part of the curated contract.
It is not an informal side channel.

The system must be able to persist triaged records or triaged decisions with at
least:

- `triage_id`
- `lang_code`
- `word`
- `reason_code`
- `severity`
- `raw_record_refs`
- `suggested_action`
- `status`

### Required triage reason codes in V1

- `unknown_pos`
- `missing_lexical_identity`
- `missing_gloss`
- `conflicting_merge`
- `oversized_forms`
- `oversized_sounds`
- `record_type_out_of_scope`
- `derived_form_entry`
- `parser_anomaly`

### Required triage actions in V1

- `drop`
- `keep_with_flag`
- `convert_to_relation`
- `defer`
- `requires_rule_update`

## V1 Database Shape

The minimum recommended PostgreSQL shape for the curated layer is:

- `curated.entries`
- `curated.entry_relations`
- `curated.triage_queue`

### `curated.entries`

Recommended columns:

- `entry_id UUID PRIMARY KEY`
- `lang_code TEXT NOT NULL`
- `normalized_word TEXT NOT NULL`
- `word TEXT NOT NULL`
- `payload JSONB NOT NULL`
- `entry_flags TEXT[] NOT NULL DEFAULT '{}'`
- `source_summary JSONB NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Recommended unique key:

- `(lang_code, normalized_word)`

### `curated.entry_relations`

Recommended columns:

- `relation_id BIGSERIAL PRIMARY KEY`
- `entry_id UUID NOT NULL REFERENCES curated.entries(entry_id)`
- `relation_type TEXT NOT NULL`
- `target_word TEXT NOT NULL`
- `target_lang_code TEXT`
- `source_scope TEXT NOT NULL`
- `payload JSONB NOT NULL DEFAULT '{}'::jsonb`

This table is optional if all relations stay embedded inside `payload`, but
keeping it normalized is recommended for later graph features.

### `curated.triage_queue`

Recommended columns:

- `triage_id BIGSERIAL PRIMARY KEY`
- `lang_code TEXT`
- `word TEXT`
- `reason_code TEXT NOT NULL`
- `severity TEXT NOT NULL`
- `suggested_action TEXT NOT NULL`
- `status TEXT NOT NULL DEFAULT 'open'`
- `raw_record_refs JSONB NOT NULL`
- `payload JSONB NOT NULL DEFAULT '{}'::jsonb`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

## Example Curated Entry Shape

The following is illustrative, not exhaustive:

```json
{
  "entry_id": "uuid",
  "word": "encrypt",
  "normalized_word": "encrypt",
  "lang": "English",
  "lang_code": "en",
  "entry_flags": [],
  "source_summary": {
    "raw_record_count": 1,
    "raw_snapshot_ids": ["uuid"],
    "raw_run_ids": ["uuid"],
    "raw_record_refs": [
      {
        "snapshot_id": "uuid",
        "raw_record_id": 123,
        "source_line": 456,
        "pos": "verb"
      }
    ]
  },
  "etymology_groups": [
    {
      "etymology_id": "et1",
      "etymology_text": "From en- + -crypt ...",
      "etymology_flags": [],
      "member_pos": ["verb"],
      "source_refs": [{"raw_record_id": 123}]
    }
  ],
  "pos_groups": [
    {
      "pos": "verb",
      "pos_flags": [],
      "etymology_id": "et1",
      "senses": [
        {
          "sense_id": "s1",
          "gloss": "To conceal information by means of a code or cipher.",
          "raw_gloss": null,
          "tags": [],
          "qualifier": null,
          "topics": [],
          "examples": [],
          "relations": [],
          "sense_flags": []
        }
      ],
      "forms": [],
      "pronunciations": [],
      "relations": [
        {
          "relation_type": "derived_term",
          "target_word": "encryption",
          "target_lang_code": "en",
          "relation_flags": [],
          "source_scope": "entry"
        }
      ]
    }
  ]
}
```

This is still a word-centric model, not a Wiktionary mirror.

## Headword Selection Rule (word_selection_v2, user-approved 2026-08-03)

Full-snapshot builds do not curate every headword in the source language.
The approved selection rule for English builds is:

- Rule id: `word_selection_v2_phrase_threshold`.
- Frequency source: the `wordfreq` Python package (version recorded per run).
- Single-word boundary: the Zipf frequency of the top-N-th most frequent
  token, with N = 40,000 approved for the English learner dictionary.
  Single words and proper nouns are selected iff
  `zipf_frequency(headword, lang) >= boundary`.
- Phrase boundary: multiword headwords must clear `phrase_min_zipf`, with
  4.5 approved for the English learner dictionary. wordfreq scores phrases
  with a combined-token estimate that systematically overstates phrase
  frequency, so phrases are held to this stricter boundary. (v1 applied one
  unified boundary; it admitted roughly 124k long-tail phrases and was
  superseded the next day.)
- The rule only filters groups whose `lang_code` equals the rule language.
  Groups in other languages pass through unfiltered.
- Every curated run records `rule_version`, `lang`, `top_n`, the computed
  `min_zipf` boundary, `phrase_min_zipf`, the wordfreq package version, and
  the count of filtered-out groups in its run metadata.

Operationally the approved configuration is `--top-words 40000
--phrase-min-zipf 4.5` on `assemble-entries` or `run`. Omitting
`--top-words` disables selection entirely; no implicit filtering is
allowed.

## What this rulebook intentionally excludes

This document does not yet define:

- the LLM prompt contract
- export formats
- scoring and prioritization logic

Those belong to later stages.

## Operational Rule

If implementation behavior and this rulebook conflict, the rulebook wins.

If the rulebook proves insufficient for a recurring pattern, update the
rulebook before encoding new long-lived curation behavior.
