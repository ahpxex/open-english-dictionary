from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORD_SELECTION_RULE_VERSION = "word_selection_v2_phrase_threshold"


@dataclass(frozen=True)
class WordSelectionRule:
    """User-approved headword selection rule for curated builds.

    Single-token headwords (words and proper nouns alike) are selected iff
    their wordfreq Zipf frequency reaches the frequency of the top_n-th most
    frequent token for `lang`. Multiword headwords are scored with wordfreq's
    combined-token estimate, which systematically overstates how common a
    phrase is, so phrases may be held to the stricter `phrase_min_zipf`
    boundary; when it is None, phrases share the single-word boundary.
    """

    lang: str
    top_n: int
    min_zipf: float
    wordfreq_version: str
    phrase_min_zipf: float | None = None
    rule_version: str = WORD_SELECTION_RULE_VERSION

    def accepts(self, headword: str) -> bool:
        text = headword.strip()
        if not text:
            return False
        from wordfreq import zipf_frequency

        threshold = self.min_zipf
        if self.phrase_min_zipf is not None and len(text.split()) > 1:
            threshold = self.phrase_min_zipf
        return zipf_frequency(text, self.lang) >= threshold

    def as_metadata(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "lang": self.lang,
            "top_n": self.top_n,
            "min_zipf": self.min_zipf,
            "phrase_min_zipf": self.phrase_min_zipf,
            "wordfreq_version": self.wordfreq_version,
        }


def build_word_selection_rule(
    *,
    lang: str,
    top_n: int,
    phrase_min_zipf: float | None = None,
) -> WordSelectionRule:
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if phrase_min_zipf is not None and phrase_min_zipf <= 0:
        raise ValueError("phrase_min_zipf must be positive when provided")
    from importlib.metadata import version as package_version

    from wordfreq import top_n_list, zipf_frequency

    tokens = top_n_list(lang, top_n)
    if not tokens:
        raise ValueError(f"wordfreq has no frequency data for language {lang!r}")
    if len(tokens) < top_n:
        raise ValueError(
            f"wordfreq only provides {len(tokens)} tokens for language {lang!r}, "
            f"fewer than the requested top_n={top_n}"
        )
    return WordSelectionRule(
        lang=lang,
        top_n=top_n,
        min_zipf=zipf_frequency(tokens[-1], lang),
        wordfreq_version=package_version("wordfreq"),
        phrase_min_zipf=phrase_min_zipf,
    )
