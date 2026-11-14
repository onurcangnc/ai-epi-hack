"""DeepSeek backend for the verifier call layer.

Only the chat-completions endpoint is used. Errors are translated into the
small exception hierarchy below so the client can decide what is retryable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class BackendError(Exception):
    """Base class for backend failures (non-retryable unless subclassed)."""


class RateLimitError(BackendError):
    """HTTP 429."""


class ServerError(BackendError):
    """HTTP 5xx."""


class BackendTimeout(BackendError):
    """The request exceeded the client timeout."""


@dataclass(frozen=True)
class BackendResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    finish_reason: str | None = None


@dataclass(frozen=True)
class ThinkingConfig:
    enabled: bool
    budget_tokens: int | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "enabled" if self.enabled else "disabled",
        }
        if self.enabled and self.budget_tokens is not None:
            # Unverified against the live API; probe_api.py is meant to check it.
            payload["budget_tokens"] = self.budget_tokens
        return payload


class DeepSeekBackend:
    """Thin HTTP client for https://api.deepseek.com/chat/completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> BackendResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = thinking.payload()

        started = time.monotonic()
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise BackendTimeout(f"request timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise BackendError(f"request failed: {exc}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        if response.status_code == 429:
            raise RateLimitError(f"rate limited: {response.text[:500]}")
        if response.status_code >= 500:
            raise ServerError(f"server error {response.status_code}: {response.text[:500]}")
        if response.status_code >= 400:
            raise BackendError(
                f"request rejected {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"malformed API response: {exc}") from exc

        usage = data.get("usage") or {}
        return BackendResult(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason")
            if isinstance(choice, dict)
            else None,
        )
