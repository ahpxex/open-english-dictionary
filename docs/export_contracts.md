# Export Contracts

## Status

Open Dictionary currently has three implemented export artifacts:

- `audit_jsonl`
- `distribution_jsonl`
- `distribution_sqlite`

The audit artifact is internal.
The distribution artifacts are the learner-facing final formats.

Each artifact records upstream run lineage in `export.artifacts.metadata`,
including `curated_run_ids` and `llm_run_ids`.

The project must keep these export classes distinct:

- audit export: preserves stage boundaries for debugging and reproducibility
- distribution export: final product contract for downstream clients

## Current Audit JSONL

The current implemented export is a merged row of:

- top-level entry identity fields
- `curated` payload
- `llm` payload

Purpose:

- inspect the current curated structure
- inspect the current LLM response payload
- debug export selection rules
- compare prompt/model variants
- retain a replayable artifact outside PostgreSQL

Non-goals:

- this format is not meant to be stable for end-user products
- this format must not be treated as the final dictionary contract

## Distribution JSONL

The implemented final JSONL contract is `distribution_entry_v4`.

Each row represents one learner-facing dictionary entry and does not expose
internal pipeline-stage wrappers such as `curated` and `llm`.

### Required top-level structure

```json
{
  "schema_version": "distribution_entry_v4",
  "entry_id": "string",
  "headword": "string",
  "normalized_headword": "string",
  "headword_language": {
    "code": "string",
    "name": "string"
  },
  "definition_language": {
    "code": "string",
    "name": "string"
  },
  "entry_type": "standard",
  "headword_summary": "string",
  "memory_hook": "string",
  "study_notes": ["string"],
  "etymology_note": "string or null",
  "etymologies": [],
  "pos_groups": []
}
```

### Pos-group identity rule

Inside the pipeline, generation alignment uses a composite `pos_group_id`
derived from `(pos, etymology_id)`. That identifier is internal and is NOT
exported: distribution documents identify a pos group by the `(pos,
etymology_id)` pair directly, and clients must never parse a composite id
string.

Each distribution `pos_group` row therefore contains:

- `pos`
- `etymology_id`
- `summary` and optional `usage_note` (generated)

### Meaning-level rule

The final contract must not collapse meaning content back into short traditional
glosses.

Every meaning row should have:

- required `sense_id` traceable to the curated source sense
- required `priority`, exactly one of `core`, `common`, `rare`; clients may
  hide `rare` meanings by default
- optional `short_gloss` for compact indexing or quick scanning
- required `learner_explanation` as the main definition-language explanation
- optional `usage_note`
- `examples`: generated bilingual pairs `{"text", "translation"}` for core and
  common meanings. Source quotations are NOT part of the distribution
  contract; they remain available in the audit export. Sense-level relation
  edges are also excluded (they were empty for 85% of senses).

The learner explanation is the product field.
The short gloss is only a helper field.

Rows with no distributable meanings are excluded from `distribution_jsonl`
entirely. The export metadata records the number of skipped entries under
`skipped_entries_without_meanings`.

## Distribution SQLite

The implemented SQLite artifact stores the same learner-facing
`distribution_entry_v4` content under a SQLite packaging schema
`distribution_sqlite_v1`.

Goals:

- preserve the exact product row via `entries.document_json`
- support downstream querying without reparsing JSONL
- keep distribution lineage and schema metadata inside the artifact itself

The SQLite artifact currently includes:

- `metadata`
- `entries`
- `entry_study_notes`
- `etymologies`
- `pos_groups`
- `pos_group_forms`
- `pos_group_pronunciations`
- `pos_group_relations`
- `meanings`
- `meaning_examples` (generated bilingual examples)
- `meaning_relations`

The SQLite export must not invent a second product contract.
It is a packaging of the same `distribution_entry_v4` semantics, not a new
editorial model.

### Language rule

The final contract explicitly separates:

- `headword_language`
- `definition_language`

This avoids conflating source-language identity with explanation-language
identity.

## Historical Note

Earlier merged `curated + llm` JSONL rows are still exported as `audit_jsonl`
for debugging and replay. They are not the final product contract.

## What Must Stay Out Of Distribution Rows

The following belong in PostgreSQL metadata or artifact manifests, not in final
distribution entry rows:

- `run_id`
- `snapshot_id`
- `raw_record_refs`
- `raw_run_ids`
- `raw_snapshot_ids`
- `model`
- `prompt_version`
- raw LLM response wrappers
- stage wrapper keys such as `curated` and `llm`

## How To Enforce The Contract In The LLM Workflow

The distribution contract should not be enforced by prompt wording alone.
It must be enforced at multiple layers.

### 1. Keep structure deterministic outside the model

The model should not be responsible for inventing the outer entry shape.

Deterministic code should own:

- entry identity fields
- language fields
- etymology grouping
- part-of-speech grouping
- forms
- pronunciations
- normalized relation edges
- example source text and references

The model should only generate learner-facing explanatory fields.

### 2. Give the model stable IDs from curated input

The LLM input should include stable identifiers from curated data, such as:

- `entry_id`
- `etymology_id`
- `pos_group_id`
- `sense_id`

The model output must attach explanatory content back to those IDs instead of
inventing a free-form structure.

### 3. Narrow the LLM output contract

The next LLM contract should generate only the fields that genuinely require
generation, for example:

- `headword_summary`
- `memory_hook`
- `study_notes`
- `etymology_note`
- `pos_groups[].summary`
- `pos_groups[].usage_note`
- `meanings[].priority`
- `meanings[].learner_explanation`
- `meanings[].usage_note`
- `meanings[].examples`

It should not generate:

- forms
- pronunciations
- provenance
- distribution wrappers

### 4. Validate against the curated shape

Schema validation must check more than JSON shape.
It must also verify alignment with the curated source row.

Examples:

- no extra `pos_group_id` values beyond curated input
- no missing required sense IDs
- no unknown sense IDs
- no extra etymology IDs
- no empty learner explanations

### 5. Assemble the final distribution row in deterministic code

The export layer should merge:

- structural fields from curated data
- explanatory fields from validated LLM data

The model should never emit the final distribution row directly.

### 6. Keep auditability separate

Prompt version, model, retries, failures, input hash, and raw response stay in:

- `llm.*` tables
- `meta.pipeline_runs`
- export artifact manifests

They do not belong in final dictionary rows.

### 7. Test the contract explicitly

Tests for the implemented distribution export assert:

- no `curated` key
- no `llm` key
- no provenance leakage
- presence of `headword_language`
- presence of `definition_language`
- presence of `learner_explanation`
- product rows remain reproducible from stored curated and llm runs


## Distribution Packaging Rules (user-approved 2026-08-03)

- `forms` carries inflection forms only (tags intersecting plural /
  comparative / superlative / past / participle / present / third-person /
  singular). Alternative spellings and abbreviations stay in curated data.
- `pronunciations` carries at most one US IPA (General-American / US tags)
  and one UK IPA (Received-Pronunciation / UK tags); when neither exists, the
  first available pronunciation is used untagged. `audio_url` is not part of
  the contract because source audio coverage is unreliable (~35%).
- Entries whose sense count exceeds the shard threshold are generated in
  parts (overview call plus chunked pos-group calls) and assembled
  deterministically; the assembled payload passes the same full-skeleton
  validation as single-call entries, so the distribution contract is
  unaffected by sharding.
