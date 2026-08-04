from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


SAMPLING_METHOD_VERSION = "stratified_v1"

# Strata are motivated by the failure modes observed during the first full
# production run: abbreviations, sharded giants, phrases, and proper names
# each fail differently, and several of them are far too rare for a simple
# random sample to cover with any statistical power.
MIN_STRATUM_SAMPLE = 25

ABBREVIATION_PATTERN = re.compile(r"^[A-Z0-9./&'-]{2,6}$")


@dataclass(frozen=True)
class SampledEntry:
    entry_id: str
    word: str
    stratum: str
    sampling_weight: float


def classify_entry_kind(word: str, entry_flags: list[str]) -> str:
    if "entry_type:proper_name" in entry_flags:
        return "proper_name"
    if "entry_type:affix" in entry_flags or "entry_type:proverb" in entry_flags:
        return "affix_or_proverb"
    if " " in word:
        return "phrase"
    if ABBREVIATION_PATTERN.match(word):
        return "abbreviation"
    return "word"


def classify_size_band(sense_count: int) -> str:
    if sense_count <= 1:
        return "1"
    if sense_count <= 9:
        return "2-9"
    if sense_count <= 28:
        return "10-28"
    return "sharded"


def build_stratum(word: str, entry_flags: list[str], sense_count: int) -> str:
    return f"{classify_entry_kind(word, entry_flags)}|{classify_size_band(sense_count)}"


def sample_key(seed: str, entry_id: str) -> str:
    return hashlib.md5(f"{seed}|{entry_id}".encode("utf-8")).hexdigest()


def allocate_stratified_sample(
    population: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: str,
) -> list[SampledEntry]:
    """Deterministic stratified sample with proportional allocation.

    Every stratum gets at least MIN_STRATUM_SAMPLE entries (or its full
    population when smaller); the remainder is allocated proportionally.
    Each sampled entry carries weight = stratum_population / stratum_sample
    so global rates can be estimated without bias despite the floors.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")

    strata: dict[str, list[dict[str, Any]]] = {}
    for item in population:
        stratum = build_stratum(
            item["word"],
            item.get("entry_flags") or [],
            item["sense_count"],
        )
        strata.setdefault(stratum, []).append(item)

    floors = {
        name: min(MIN_STRATUM_SAMPLE, len(members))
        for name, members in strata.items()
    }
    remaining = max(0, sample_size - sum(floors.values()))
    total_population = sum(len(members) for members in strata.values())

    allocations: dict[str, int] = dict(floors)
    if remaining and total_population:
        for name, members in sorted(strata.items()):
            extra_capacity = len(members) - allocations[name]
            proportional = round(remaining * len(members) / total_population)
            allocations[name] += max(0, min(extra_capacity, proportional))

    sampled: list[SampledEntry] = []
    for name in sorted(strata):
        members = strata[name]
        take = min(allocations[name], len(members))
        if take <= 0:
            continue
        ordered = sorted(members, key=lambda item: sample_key(seed, str(item["entry_id"])))
        weight = len(members) / take
        for item in ordered[:take]:
            sampled.append(
                SampledEntry(
                    entry_id=str(item["entry_id"]),
                    word=item["word"],
                    stratum=name,
                    sampling_weight=weight,
                )
            )
    return sampled
