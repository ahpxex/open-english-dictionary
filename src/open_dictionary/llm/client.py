from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from .config import LLMProviderSettings, LLMSettings


GENERATION_MODEL_GROUP = "open-dictionary-generation"
REQUEST_TIMEOUT_SECONDS = 180
TRANSIENT_RETRIES = 3
COOLDOWN_SECONDS = 30


class LLMClientError(RuntimeError):
    def __init__(self, message: str, *, model: str | None = None):
        super().__init__(message)
        self.model = model


@dataclass(frozen=True)
class LLMGenerationResult:
    content: str
    model: str
    api_base: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class LiteLLMClient:
    """Load-balancing client over one or more OpenAI-compatible providers.

    All providers are registered as deployments of a single litellm Router
    model group. Concurrent callers are spread across the pool (weighted by
    each provider's optional rpm limit), rate-limited or failing deployments
    are cooled down while requests retry on the remaining providers, and each
    result reports which provider actually served the request so callers can
    persist truthful generation provenance.
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        transient_retries: int = TRANSIENT_RETRIES,
        cooldown_seconds: int = COOLDOWN_SECONDS,
    ):
        if not settings.providers:
            raise ValueError("LLMSettings must contain at least one provider")
        # litellm imports lazily so importing this module stays cheap for
        # CLI startup and for tests that only use fake clients.
        import litellm
        from litellm.router import Router

        litellm.telemetry = False
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

        self._providers = settings.providers
        self._deployments_by_id = {
            self._deployment_id(index): provider
            for index, provider in enumerate(settings.providers)
        }
        self._router = Router(
            model_list=[
                {
                    "model_name": GENERATION_MODEL_GROUP,
                    "litellm_params": {
                        "model": f"openai/{provider.model}",
                        "api_base": provider.api_base,
                        "api_key": provider.api_key,
                        "timeout": timeout,
                        **({"rpm": provider.rpm} if provider.rpm is not None else {}),
                    },
                    "model_info": {"id": self._deployment_id(index)},
                }
                for index, provider in enumerate(settings.providers)
            ],
            num_retries=transient_retries,
            cooldown_time=cooldown_seconds,
            allowed_fails=3,
            enable_pre_call_checks=any(
                provider.rpm is not None for provider in settings.providers
            ),
        )

    @property
    def providers(self) -> tuple[LLMProviderSettings, ...]:
        return self._providers

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMGenerationResult:
        request: dict[str, Any] = {
            "model": GENERATION_MODEL_GROUP,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        try:
            response = self._router.completion(**request)
        except Exception as exc:
            raise LLMClientError(
                f"LLM request failed: {exc}",
                model=self._model_from_exception(exc),
            ) from exc

        provider = self._provider_from_response(response)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMClientError(
                "LLM response did not contain chat completion content",
                model=provider.model if provider is not None else None,
            ) from exc
        if not isinstance(content, str) or not content:
            raise LLMClientError(
                "LLM response did not contain chat completion content",
                model=provider.model if provider is not None else None,
            )

        if provider is None:
            raise LLMClientError(
                "LLM response could not be attributed to a configured provider"
            )
        usage = getattr(response, "usage", None)
        return LLMGenerationResult(
            content=content,
            model=provider.model,
            api_base=provider.api_base,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def _deployment_id(index: int) -> str:
        return f"provider-{index + 1}"

    def _provider_from_response(self, response: Any) -> LLMProviderSettings | None:
        hidden_params = getattr(response, "_hidden_params", None) or {}
        deployment_id = hidden_params.get("model_id")
        if deployment_id in self._deployments_by_id:
            return self._deployments_by_id[deployment_id]
        return self._provider_from_model_name(getattr(response, "model", None))

    def _model_from_exception(self, exc: Exception) -> str | None:
        provider = self._provider_from_model_name(getattr(exc, "model", None))
        return provider.model if provider is not None else None

    def _provider_from_model_name(self, raw_model: Any) -> LLMProviderSettings | None:
        if not isinstance(raw_model, str) or not raw_model:
            return None
        candidates = {raw_model}
        prefix, _, remainder = raw_model.partition("/")
        if prefix == "openai" and remainder:
            candidates.add(remainder)
        for provider in self._providers:
            if provider.model in candidates:
                return provider
        return None
