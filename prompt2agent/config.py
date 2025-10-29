"""Configuration helpers for routing models through LiteLLM/OpenRouter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agents.models.interface import ModelProvider

from dotenv import load_dotenv

try:
    from agents.extensions.models.litellm_provider import LitellmProvider
except ImportError as exc:  # pragma: no cover - depends on optional extra
    LitellmProvider = None  # type: ignore[assignment]
    _LITELLM_IMPORT_ERROR = exc
else:  # pragma: no branch - executed when optional dependency is available
    _LITELLM_IMPORT_ERROR = None

load_dotenv()

ENV_OPENROUTER_KEY = "OPENROUTER_API_KEY"
ENV_LITELLM_PROVIDER = "LITELLM_PROVIDER"
ENV_MODEL_ID = "MODEL_ID"
ENV_OPENROUTER_BASE_URL = "OPENROUTER_BASE_URL"
ENV_OPENAI_DEFAULT_MODEL = "OPENAI_DEFAULT_MODEL"

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderConfig:
    """Represents the model routing configuration for an agent workflow."""

    model_id: str
    provider: ModelProvider
    base_url: Optional[str]

    def build(self, *, workflow_name: str | None = None) -> "RunConfig":
        """Return a :class:`agents.run.RunConfig` bound to this provider.

        Parameters
        ----------
        workflow_name:
            Optional name for tracing/observability purposes.
        """

        from agents.run import RunConfig  # Local import to avoid cycle during module import.

        return RunConfig(
            model=self.model_id,
            model_provider=self.provider,
            workflow_name=workflow_name or "Prompt-driven workflow",
        )


class EnvironmentValidationError(RuntimeError):
    """Raised when the runtime configuration is incomplete."""


def ensure_provider_config() -> ProviderConfig:
    """Validate environment variables and construct a LiteLLM/OpenRouter provider.

    The function checks for the required OpenRouter + LiteLLM variables and synchronises the
    SDK's expectations (e.g. ``OPENAI_DEFAULT_MODEL``) so that downstream agent invocations
    transparently route through LiteLLM.
    """

    api_key = os.getenv(ENV_OPENROUTER_KEY)
    if not api_key:
        raise EnvironmentValidationError(
            "Missing OpenRouter credentials. Set OPENROUTER_API_KEY before running workflows."
        )

    provider_name = (os.getenv(ENV_LITELLM_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    if provider_name != DEFAULT_PROVIDER:
        raise EnvironmentValidationError(
            f"Unsupported LiteLLM provider '{provider_name}'. Configure LITELLM_PROVIDER=openrouter."
        )

    def _clean_model(value: str | None) -> str:
        if not value:
            return ""
        return value.strip().lstrip("=")

    model_id = _clean_model(os.getenv(ENV_MODEL_ID))
    if not model_id:
        model_id = _clean_model(os.getenv(ENV_OPENAI_DEFAULT_MODEL))

    if not model_id:
        raise EnvironmentValidationError(
            "MODEL_ID is required to select the default model. Set MODEL_ID or OPENAI_DEFAULT_MODEL."
        )

    if not model_id.startswith(f"{DEFAULT_PROVIDER}/"):
        model_id = f"{DEFAULT_PROVIDER}/{model_id.lstrip('/')}"

    base_url = (os.getenv(ENV_OPENROUTER_BASE_URL) or DEFAULT_OPENROUTER_BASE_URL).strip() or DEFAULT_OPENROUTER_BASE_URL

    # Normalise environment variables so both LiteLLM and the Agents SDK resolve the same defaults.
    os.environ[ENV_OPENROUTER_KEY] = api_key
    os.environ[ENV_LITELLM_PROVIDER] = DEFAULT_PROVIDER
    os.environ[ENV_MODEL_ID] = model_id
    os.environ.setdefault(ENV_OPENAI_DEFAULT_MODEL, model_id)
    os.environ.setdefault(ENV_OPENROUTER_BASE_URL, base_url)

    if LitellmProvider is None:
        raise EnvironmentValidationError(
            "LiteLLM integration is unavailable. Install the optional dependency with "
            "`pip install 'openai-agents[litellm]'` to enable model routing."
        ) from _LITELLM_IMPORT_ERROR

    provider = LitellmProvider()
    return ProviderConfig(model_id=model_id, provider=provider, base_url=base_url)


__all__ = [
    "EnvironmentValidationError",
    "ProviderConfig",
    "ensure_provider_config",
]
