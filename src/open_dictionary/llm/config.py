from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROVIDERS_ENV_VAR = "LLM_PROVIDERS"
PROVIDER_REQUIRED_FIELDS = ("api", "model", "key")
PROVIDER_ALLOWED_FIELDS = frozenset((*PROVIDER_REQUIRED_FIELDS, "rpm"))


@dataclass(frozen=True)
class LLMProviderSettings:
    api_base: str
    api_key: str
    model: str
    rpm: int | None = None


@dataclass(frozen=True)
class LLMSettings:
    providers: tuple[LLMProviderSettings, ...]

    @property
    def models(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for provider in self.providers:
            seen.setdefault(provider.model)
        return tuple(seen)


def load_llm_settings(*, env_file: str | Path | None = ".env") -> LLMSettings:
    env_path = Path(env_file) if env_file else None
    if env_path is not None:
        load_dotenv(env_path, override=True)

    raw_providers = os.getenv(PROVIDERS_ENV_VAR)
    if raw_providers is not None and raw_providers.strip():
        return LLMSettings(providers=_parse_providers_json(raw_providers))
    return LLMSettings(providers=(_load_legacy_provider(),))


def _parse_providers_json(raw: str) -> tuple[LLMProviderSettings, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Environment variable {PROVIDERS_ENV_VAR} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Environment variable {PROVIDERS_ENV_VAR} must be a JSON array of provider objects."
        )
    if not parsed:
        raise RuntimeError(
            f"Environment variable {PROVIDERS_ENV_VAR} must contain at least one provider."
        )

    providers: list[LLMProviderSettings] = []
    for position, entry in enumerate(parsed, start=1):
        providers.append(_parse_provider_entry(entry, position=position))
    return tuple(providers)


def _parse_provider_entry(entry: Any, *, position: int) -> LLMProviderSettings:
    location = f"{PROVIDERS_ENV_VAR}[{position}]"
    if not isinstance(entry, dict):
        raise RuntimeError(f"{location} must be a JSON object.")

    unknown_fields = sorted(set(entry) - PROVIDER_ALLOWED_FIELDS)
    if unknown_fields:
        raise RuntimeError(
            f"{location} contains unknown fields: {', '.join(unknown_fields)}. "
            f"Allowed fields: {', '.join(sorted(PROVIDER_ALLOWED_FIELDS))}."
        )
    for field in PROVIDER_REQUIRED_FIELDS:
        if field not in entry:
            raise RuntimeError(f"{location} is missing the required field '{field}'.")

    api_base = entry["api"]
    model = entry["model"]
    api_key = entry["key"]
    if not isinstance(api_base, str) or not api_base.strip():
        raise RuntimeError(f"{location} field 'api' must be a non-empty string.")
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(f"{location} field 'model' must be a non-empty string.")
    if not isinstance(api_key, str):
        raise RuntimeError(f"{location} field 'key' must be a string.")

    rpm = entry.get("rpm")
    if rpm is not None:
        if isinstance(rpm, bool) or not isinstance(rpm, int):
            raise RuntimeError(f"{location} field 'rpm' must be an integer.")
        if rpm <= 0:
            raise RuntimeError(f"{location} field 'rpm' must be a positive integer.")

    return LLMProviderSettings(
        api_base=api_base,
        api_key=api_key,
        model=model,
        rpm=rpm,
    )


def _load_legacy_provider() -> LLMProviderSettings:
    api_base = os.getenv("LLM_API")
    api_key = os.getenv("LLM_KEY")
    model = os.getenv("LLM_MODEL")
    if not api_base:
        raise RuntimeError(
            "No LLM provider is configured. Set LLM_PROVIDERS to a JSON array "
            'of provider objects ([{"api": ..., "model": ..., "key": ..., "rpm": ...}]), '
            "or the legacy LLM_API / LLM_MODEL / LLM_KEY variables. "
            "Environment variable LLM_API is not set."
        )
    if api_key is None:
        raise RuntimeError("Environment variable LLM_KEY is not set.")
    if not model:
        raise RuntimeError("Environment variable LLM_MODEL is not set.")
    return LLMProviderSettings(
        api_base=api_base,
        api_key=api_key,
        model=model,
        rpm=_parse_legacy_rpm(os.getenv("LLM_RPM")),
    )


def _parse_legacy_rpm(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        rpm = int(raw)
    except ValueError as exc:
        raise RuntimeError("Environment variable LLM_RPM must be an integer.") from exc
    if rpm <= 0:
        raise RuntimeError("Environment variable LLM_RPM must be a positive integer.")
    return rpm
