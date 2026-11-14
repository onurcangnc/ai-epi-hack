"""Verifier call layer (DeepSeek only).

Wires the frozen prompts, DeepSeek backend, strict parser, persistent cache,
configurable retry/backoff and JSONL logging together. Everything that can vary
between calls is recorded per log row so results stay auditable.

Configuration lives under ``models.deepseek`` in config.yaml (retry policy
included). The cache directory comes from ``cache.dir``. Values that the human
has not set are left null and raise ``ConfigError`` naming the key, so a run can
never silently use an invented default.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
from prompts import PromptBundle, load_prompts, render_evidence, render_user  # noqa: E402

# The only outcomes a retry policy may name.
RETRYABLE_OUTCOMES = frozenset({"api_error", "parse_failure"})


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
    retry_on: tuple[str, ...]
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
        retry_on = _non_null(section.get("retry_on"), "retry_on")

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

        if not isinstance(retry_on, list) or not retry_on:
            raise ConfigError(
                "config key 'models.deepseek.retry_on' must be a non-empty list"
            )
        for outcome in retry_on:
            if not isinstance(outcome, str):
                raise ConfigError(
                    "config key 'models.deepseek.retry_on' must contain strings"
                )
        unknown = sorted(set(retry_on) - RETRYABLE_OUTCOMES)
        if unknown:
            raise ConfigError(
                "config key 'models.deepseek.retry_on' has unknown entries "
                f"{unknown}; allowed: {sorted(RETRYABLE_OUTCOMES)}"
            )
        retry_on = tuple(dict.fromkeys(retry_on))

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
            retry_on=retry_on,
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


def load_cache_dir(config_path: Path) -> Path:
    """Resolve ``cache.dir`` against the directory containing the config file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    section = config.get("cache")
    if not isinstance(section, dict):
        raise ConfigError("config key 'cache' must be a mapping")
    directory = section.get("dir")
    if not isinstance(directory, str) or not directory:
        raise ConfigError("config key 'cache.dir' must be a non-empty string")
    path = Path(directory)
    return path if path.is_absolute() else config_path.resolve().parent / path


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
        cache_dir: Path | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.backend = backend
        self.prompts = prompts
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._sleep = sleep
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    # ------------------------------------------------------------------ #
    # Cache / logging helpers
    # ------------------------------------------------------------------ #
    def _load_disk_cache(self) -> None:
        assert self.cache_dir is not None
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(f"unreadable cache entry {path}: {exc}") from exc
            key = payload.pop("key", None)
            if not isinstance(key, str) or not key:
                raise ConfigError(f"cache entry {path} has no 'key'")
            self._cache[key] = payload

    def _disk_store(self, key: str, entry: dict[str, Any]) -> None:
        assert self.cache_dir is not None
        target = self.cache_dir / f"{key}.json"
        temporary = self.cache_dir / f".{key}.json.tmp"
        temporary.write_text(
            json.dumps({"key": key, **entry}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

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
        finish_reason: str | None = None,
        order_seed: Any = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "claim_id": claim_id,
            "condition": condition,
            "model": self.config.model,
            "sample_idx": sample_idx,
            "order_seed": order_seed,
            "config_hash": self.config.config_hash,
            "system_prompt_sha256": self.prompts.system_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "outcome": outcome,
            "verdict": parsed["verdict"] if parsed else None,
            "confidence": parsed["confidence"] if parsed else None,
            "rationale": parsed["rationale"] if parsed else None,
            "evidence_concerns": parsed["evidence_concerns"] if parsed else None,
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "finish_reason_last": finish_reason,
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
        order_seed: Any = None,
    ) -> dict[str, Any]:
        """Run one verifier call (with retries/cache) and log the result."""
        claim_text, claim_tags = escape_tags(claim)
        evidence_raw = render_evidence(evidence_sentences)
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
                finish_reason=cached_entry.get("finish_reason"),
                order_seed=order_seed,
            )
            self._log(row)
            return row

        attempts = 0
        first_attempt_outcome: str | None = None
        retry_api = "api_error" in self.config.retry_on
        retry_parse = "parse_failure" in self.config.retry_on
        thinking = ThinkingConfig(
            enabled=self.config.thinking_enabled,
            budget_tokens=self.config.thinking_budget_tokens,
        )
        while True:
            attempt_idx = attempts
            call_error: Exception | None = None
            retryable_type = False
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
                retryable_type = True
            except BackendError as exc:
                call_error = exc

            if call_error is not None:
                outcome = "api_error"
                if first_attempt_outcome is None:
                    first_attempt_outcome = outcome
                retryable = retryable_type and retry_api
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
                    finish_reason=None,
                    order_seed=order_seed,
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
                # A parse failure is an outcome, not a transport accident: it is
                # retried only when the config explicitly lists it.
                is_final = (not retry_parse) or attempts >= self.config.max_retries
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
                    finish_reason=result.finish_reason,
                    order_seed=order_seed,
                )
                self._log(row)
                if is_final:
                    return row
                attempts += 1
                continue

            outcome = "ok"
            if first_attempt_outcome is None:
                first_attempt_outcome = outcome
            entry = {
                "parsed": parsed,
                "raw": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
                "finish_reason": result.finish_reason,
                "attempt_idx": attempt_idx,
                "total_attempts": attempt_idx + 1,
                "first_attempt_outcome": first_attempt_outcome,
                "attempts": attempts,
            }
            with self._lock:
                self._cache[key] = entry
                if self.cache_dir is not None:
                    self._disk_store(key, entry)
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
                finish_reason=result.finish_reason,
                order_seed=order_seed,
            )
            self._log(row)
            return row

    def run_batch(
        self, items: list[dict[str, Any]], order_seed: Any
    ) -> list[dict[str, Any]]:
        """Run verify() calls in a seeded shuffle, logging order_seed in every row.

        The shuffle spans all conditions in ``items``; passing the same seed
        reproduces the same execution order.
        """
        ordered = [dict(item) for item in items]
        random.Random(order_seed).shuffle(ordered)
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [
                pool.submit(self.verify, **{**item, "order_seed": order_seed})
                for item in ordered
            ]
            return [future.result() for future in futures]
