"""Async model adapter for OpenRouter API."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import httpx

from prompt2agent.config import DEFAULT_MODEL, OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL
from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)


class ModelAdapter:
    """Adapter that proxies chat completions to OpenRouter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV, "")
        self.timeout = timeout

    async def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: MutableMapping[str, str] | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> str:
        """Execute a chat completion request using OpenRouter."""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export the key before running commands."
            )

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/openai-agents-python",
            "X-Title": "prompt2agent-poc",
        }
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        logger.debug("Dispatching chat completion: %s", payload)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        logger.debug("Received response: %s", data)

        choices: List[Dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from OpenRouter response")
        message: Mapping[str, Any] = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("Empty content received from OpenRouter")
        return str(content)
