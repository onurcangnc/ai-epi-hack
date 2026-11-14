"""Verifier call layer (DeepSeek only).

Wires the frozen prompts, DeepSeek backend, strict parser, in-memory cache,
retry/backoff and JSONL logging together. Everything that can vary between
calls is recorded per log row so results stay auditable.

Configuration lives under ``models.deepseek`` in config.yaml. Values that the
human has not set are left null and raise ``ConfigError`` naming the key, so a
run can never silently use an invented default.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_ROOT / "scripts"))

from common import VERDICTS  # noqa: E402,F401  (imported for the stated vocabulary)
from backend import (  # noqa: E402
    BackendError,
    BackendTimeout,
    RateLimitError,
    ServerError,
    ThinkingConfig,
)
from escaping import escape_tags  # noqa: E402
from parsing import ParseError, parse_response  # noqa: E402
from prompts import PromptBundle, load_prompts, render_user  # noqa: E402


class ConfigError(ValueError):
    """Raised when model config is missing or still null."""


def _non_null(value: Any, key: str) -> Any:
    if value is None:
        raise ConfigError(f"config key 'models.deepseek.{key}' is null; set it before running")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    top_p: float | None
    max_tokens: int | None
    thinking_enabled: bool
    thinking_budget_tokens: int | None
    concurrency: int
    max_retries: int
    price_table: dict[str, float] | None

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.output_config(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def output_config(self) -> dict[str, Any]:
        """The settings that can change model output (cache/log identity)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "thinking": {
                "enabled": self.thinking_enabled,
                "budget_tokens": self.thinking_budget_tokens,
            },
        }

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> "ModelConfig":
        if not isinstance(section, dict):
            raise ConfigError("config key 'models.deepseek' must be a mapping")
        base_url = _non_null(section.get("base_url"), "base_url")
        model = _non_null(section.get("model"), "model")
        api_key_env = _non_null(section.get("api_key_env"), "api_key_env")
        temperature = _non_null(section.get("temperature"), "temperature")
        concurrency = _non_null(section.get("concurrency"), "concurrency")
        max_retries = _non_null(section.get("max_retries"), "max_retries")

        if not isinstance(base_url, str) or not base_url:
            raise ConfigError("config key 'models.deepseek.base_url' must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ConfigError("config key 'models.deepseek.model' must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ConfigError("config key 'models.deepseek.api_key_env' must be a non-empty string")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ConfigError("config key 'models.deepseek.temperature' must be a number")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ConfigError("config key 'models.deepseek.concurrency' must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigError("config key 'models.deepseek.max_retries' must be a non-negative integer")

        thinking = section.get("thinking")
        if not isinstance(thinking, dict):
            raise ConfigError("config key 'models.deepseek.thinking' must be a mapping")
        if "enabled" not in thinking or thinking["enabled"] is None:
            raise ConfigError("config key 'models.deepseek.thinking.enabled' is null; set it before running")
        if not isinstance(thinking["enabled"], bool):
            raise ConfigError("config key 'models.deepseek.thinking.enabled' must be a boolean")
        budget = thinking.get("budget_tokens")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
        ):
            raise ConfigError("config key 'models.deepseek.thinking.budget_tokens' must be a positive integer or null")

        top_p = section.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1
        ):
            raise ConfigError("config key 'models.deepseek.top_p' must be a number in [0, 1] or null")
        max_tokens = section.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise ConfigError("config key 'models.deepseek.max_tokens' must be a positive integer or null")

        price_table = section.get("price_table")
        if price_table is not None:
            if not isinstance(price_table, dict) or set(price_table) != {
                "input_per_mtok",
                "output_per_mtok",
            }:
                raise ConfigError(
                    "config key 'models.deepseek.price_table' must be "
                    "{input_per_mtok, output_per_mtok} or null"
                )

        return cls(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            temperature=float(temperature),
            top_p=None if top_p is None else float(top_p),
            max_tokens=max_tokens,
            thinking_enabled=thinking["enabled"],
            thinking_budget_tokens=budget,
            concurrency=concurrency,
            max_retries=max_retries,
            price_table=price_table,
        )


def load_model_config(config_path: Path) -> ModelConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    section = (config.get("models") or {}).get("deepseek")
    if section is None:
        raise ConfigError("config key 'models.deepseek' is missing")
    return ModelConfig.from_dict(section)


def api_key_from_env(config: ModelConfig) -> str:
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ConfigError(f"environment variable {config.api_key_env} is not set")
    return key


def _backoff_seconds(attempt: int) -> float:
    return min(30.0, 0.5 * (2 ** (attempt - 1)))


class VerifierClient:
    """One verifier call = prompt render, backend call(s), parse, cache, log."""

    def __init__(
        self,
        config: ModelConfig,
        backend: Any,
        prompts: PromptBundle,
        log_path: Path,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.backend = backend
        self.prompts = prompts
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Cache / logging helpers
    # ------------------------------------------------------------------ #
    def _cache_key(self, user_prompt: str, sample_idx: int) -> str:
        payload = [
            self.config.model,
            self.config.config_hash,
            self.prompts.system_sha256,
            _sha256(user_prompt),
            sample_idx,
        ]
        return _sha256(json.dumps(payload))

    def _cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        table = self.config.price_table
        if table is None or input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * table["input_per_mtok"]
            + output_tokens * table["output_per_mtok"]
        ) / 1_000_000

    def _log(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")

    def _row(
        self,
        *,
        claim_id: str,
        condition: str | None,
        sample_idx: int,
        user_prompt_sha256: str,
        escapes: list[dict[str, str]],
        outcome: str,
        raw_response: str | None,
        parsed: dict[str, Any] | None,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: float | None,
        attempt_idx: int,
        total_attempts: int,
        first_attempt_outcome: str,
        used_attempt_idx: int | None,
        cached: bool,
        attempts: int,
        is_result: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "claim_id": claim_id,
            "condition": condition,
            "model": self.config.model,
            "sample_idx": sample_idx,
            "config_hash": self.config.config_hash,
            "system_prompt_sha256": self.prompts.system_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "outcome": outcome,
            "verdict": parsed["verdict"] if parsed else None,
            "confidence": parsed["confidence"] if parsed else None,
            "rationale": parsed["rationale"] if parsed else None,
            "evidence_concerns": parsed["evidence_concerns"] if parsed else None,
            "raw_response": raw_response,
            "tokens": {"input": input_tokens, "output": output_tokens},
            "latency_ms": latency_ms,
            "attempt_idx": attempt_idx,
            "total_attempts": total_attempts,
            "first_attempt_outcome": first_attempt_outcome,
            "used_attempt_idx": used_attempt_idx,
            "is_result": is_result,
            "cached": cached,
            "attempts": attempts,
            "escapes": escapes,
            "error": error,
        }
        cost = self._cost(input_tokens, output_tokens)
        if cost is not None:
            row["cost"] = cost
        return row

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(
        self,
        *,
        claim_id: str,
        claim: str,
        evidence_sentences: list[dict[str, Any]],
        sample_idx: int = 0,
        condition: str | None = None,
    ) -> dict[str, Any]:
        """Run one verifier call (with retries/cache) and log the result."""
        claim_text, claim_tags = escape_tags(claim)
        evidence_raw = " ".join(sentence["text"] for sentence in evidence_sentences)
        evidence_text, evidence_tags = escape_tags(evidence_raw)
        escapes = [
            {"claim_id": claim_id, "field": "claim", "tag": tag} for tag in claim_tags
        ]
        escapes += [
            {"claim_id": claim_id, "field": "evidence", "tag": tag}
            for tag in evidence_tags
        ]

        user_prompt = render_user(self.prompts.user_template, claim_text, evidence_text)
        user_prompt_sha256 = _sha256(user_prompt)
        key = self._cache_key(user_prompt, sample_idx)

        with self._lock:
            cached_entry = self._cache.get(key)
        if cached_entry is not None:
            row = self._row(
                claim_id=claim_id,
                condition=condition,
                sample_idx=sample_idx,
                user_prompt_sha256=user_prompt_sha256,
                escapes=escapes,
                outcome="ok",
                raw_response=cached_entry["raw"],
                parsed=cached_entry["parsed"],
                input_tokens=cached_entry["input_tokens"],
                output_tokens=cached_entry["output_tokens"],
                latency_ms=cached_entry["latency_ms"],
                attempt_idx=cached_entry["attempt_idx"],
                total_attempts=cached_entry["total_attempts"],
                first_attempt_outcome=cached_entry["first_attempt_outcome"],
                used_attempt_idx=cached_entry["attempt_idx"],
                cached=True,
                attempts=cached_entry["attempts"],
                is_result=True,
            )
            self._log(row)
            return row

        attempts = 0
        first_attempt_outcome: str | None = None
        thinking = ThinkingConfig(
            enabled=self.config.thinking_enabled,
            budget_tokens=self.config.thinking_budget_tokens,
        )
        while True:
            attempt_idx = attempts
            call_error: Exception | None = None
            retryable = False
            try:
                result = self.backend.complete(
                    self.prompts.system,
                    user_prompt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    thinking=thinking,
                )
            except (RateLimitError, ServerError, BackendTimeout) as exc:
                call_error = exc
                retryable = True
            except BackendError as exc:
                call_error = exc

            if call_error is not None:
                outcome = "api_error"
                if first_attempt_outcome is None:
                    first_attempt_outcome = outcome
                is_final = (not retryable) or attempts >= self.config.max_retries
                row = self._row(
                    claim_id=claim_id,
                    condition=condition,
                    sample_idx=sample_idx,
                    user_prompt_sha256=user_prompt_sha256,
                    escapes=escapes,
                    outcome=outcome,
                    raw_response=None,
                    parsed=None,
                    input_tokens=None,
                    output_tokens=None,
                    latency_ms=None,
                    attempt_idx=attempt_idx,
                    total_attempts=attempt_idx + 1,
                    first_attempt_outcome=first_attempt_outcome,
                    used_attempt_idx=attempt_idx if is_final else None,
                    cached=False,
                    attempts=attempts,
                    is_result=is_final,
                    error=str(call_error),
                )
                self._log(row)
                if is_final:
                    return row
                attempts += 1
                self._sleep(_backoff_seconds(attempts))
                continue

            try:
                parsed = parse_response(result.text)
            except ParseError:
                outcome = "parse_failure"
                if first_attempt_outcome is None:
                    first_attempt_outcome = outcome
                is_final = attempts >= self.config.max_retries
                row = self._row(
                    claim_id=claim_id,
                    condition=condition,
                    sample_idx=sample_idx,
                    user_prompt_sha256=user_prompt_sha256,
                    escapes=escapes,
                    outcome=outcome,
                    raw_response=result.text,
                    parsed=None,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    attempt_idx=attempt_idx,
                    total_attempts=attempt_idx + 1,
                    first_attempt_outcome=first_attempt_outcome,
                    used_attempt_idx=attempt_idx if is_final else None,
                    cached=False,
                    attempts=attempts,
                    is_result=is_final,
                )
                self._log(row)
                if is_final:
                    return row
                attempts += 1
                continue

            outcome = "ok"
            if first_attempt_outcome is None:
                first_attempt_outcome = outcome
            row = self._row(
                claim_id=claim_id,
                condition=condition,
                sample_idx=sample_idx,
                user_prompt_sha256=user_prompt_sha256,
                escapes=escapes,
                outcome=outcome,
                raw_response=result.text,
                parsed=parsed,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                attempt_idx=attempt_idx,
                total_attempts=attempt_idx + 1,
                first_attempt_outcome=first_attempt_outcome,
                used_attempt_idx=attempt_idx,
                cached=False,
                attempts=attempts,
                is_result=True,
            )
            with self._lock:
                self._cache[key] = {
                    "parsed": parsed,
                    "raw": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": result.latency_ms,
                    "attempt_idx": attempt_idx,
                    "total_attempts": attempt_idx + 1,
                    "first_attempt_outcome": first_attempt_outcome,
                    "attempts": attempts,
                }
            self._log(row)
            return row

    def run_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run independent verify() calls with bounded concurrency."""
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(self.verify, **item) for item in items]
            return [future.result() for future in futures]
