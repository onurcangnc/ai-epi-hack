"""Tests that pin the 11 findings of the callayer-v2 review (2026-11-14).

Tests named ``test_fNN_*`` fail on callayer-v2 by design; each one turns green
when finding NN is fixed. Tests with ``guard`` in their name pass on callayer-v2
and protect behavior the fixes must keep.

Where the review left a design choice open, the test docstring names the
contract the test picks, so the choice can be changed in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter

import verifier_client as vc
from backend import BackendError, BackendResult, BackendTimeout, DeepSeekBackend
from common import VERDICTS
from escaping import escape_tags
from parsing import ParseError, parse_response
from prompts import load_prompts, render_evidence

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"

VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'
REFUTED = '{"verdict": "REFUTED", "confidence": 61, "rationale": "no", "evidence_concerns": []}'
CALL: dict[str, Any] = {
    "claim_id": "c_1",
    "claim": "Drug X lowers blood pressure.",
    "evidence_sentences": [
        {"idx": 0, "text": "Drug X was tested in 200 adults."},
        {"idx": 1, "text": "Systolic pressure fell by 12 mmHg."},
    ],
    "sample_idx": 0,
    "condition": "C0",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def model_config(**overrides: Any) -> vc.ModelConfig:
    section: dict[str, Any] = {
        "base_url": "https://api.example.test",
        "model": "deepseek-test",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 512,
        "thinking": {"enabled": False, "budget_tokens": None},
        "concurrency": 1,
        "max_retries": 2,
        "retry_on": ["api_error"],
        "price_table": None,
    }
    for key, value in overrides.items():
        if key == "thinking":
            section["thinking"] = {**section["thinking"], **value}
        else:
            section[key] = value
    return vc.ModelConfig.from_dict(section)


class ScriptedBackend:
    """Replaces the HTTP backend. A reply is a str, a (text, finish_reason) tuple or an exception."""

    def __init__(self, replies: list[Any], delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, **kwargs: Any) -> BackendResult:
        with self._lock:
            self.calls.append({"system": system, "user": user, **kwargs})
            if not self.replies:
                raise AssertionError("backend called more often than the test allows")
            reply = self.replies.pop(0)
        if self.delay:
            time.sleep(self.delay)
        if isinstance(reply, BaseException):
            raise reply
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        return BackendResult(
            text=text,
            input_tokens=523,
            output_tokens=84,
            latency_ms=900.0,
            finish_reason=finish_reason,
        )


class KeyedBackend:
    """Times out on every call whose claim is listed; answers every other call with VALID."""

    def __init__(self, failing_claims: set[str]) -> None:
        self.failing = {f"<claim>\n{claim}\n</claim>" for claim in failing_claims}
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, **kwargs: Any) -> BackendResult:
        with self._lock:
            self.calls.append(user)
        if any(block in user for block in self.failing):
            raise BackendTimeout("read timed out")
        return BackendResult(
            text=VALID, input_tokens=523, output_tokens=84, latency_ms=900.0, finish_reason="stop"
        )


def completion_body(
    content: Any = VALID,
    finish_reason: str | None = "stop",
    *,
    model: str = "deepseek-flash",
    fingerprint: str = "fp_test",
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
    """An OpenAI-format chat completion shaped like DeepSeek's."""
    usage: dict[str, Any] = {
        "prompt_tokens": 523,
        "completion_tokens": 84,
        "total_tokens": 607,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 523,
    }
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "id": "chatcmpl-review",
        "object": "chat.completion",
        "created": 1763130000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "system_fingerprint": fingerprint,
    }


def http_reply(
    status: int = 200, body: Any = None, headers: dict[str, str] | None = None
) -> tuple[int, str, dict[str, str]]:
    payload = completion_body() if body is None else body
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return (status, raw, headers or {})


class CannedTransport(BaseAdapter):
    """Serves queued HTTP replies through the real requests stack, with no network."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__()
        self.replies = list(replies)
        self.sent: list[requests.PreparedRequest] = []

    def send(self, request, **kwargs):  # noqa: ANN001 - requests adapter signature
        self.sent.append(request)
        if not self.replies:
            raise AssertionError("more HTTP requests than the test allows")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        status, raw, headers = reply
        response = requests.Response()
        response.status_code = status
        response._content = raw.encode("utf-8")
        response.headers.update({"Content-Type": "application/json", **headers})
        response.encoding = "utf-8"
        response.url = request.url
        response.request = request
        return response

    def close(self) -> None:
        pass


def http_backend(
    replies: list[Any], *, key: str = "sk-test-key"
) -> tuple[DeepSeekBackend, CannedTransport]:
    transport = CannedTransport(replies)
    session = requests.Session()
    session.trust_env = False  # no proxy or registry lookups from the test machine
    session.mount("https://", transport)
    backend = DeepSeekBackend("https://api.example.test", key, "deepseek-test", session=session)
    return backend, transport


def make_client(
    tmp_path: Path,
    backend: Any,
    *,
    config: vc.ModelConfig | None = None,
    log_name: str = "calls.jsonl",
    prompts_dir: Path = PROMPTS,
    sleeps: list[float] | None = None,
) -> vc.VerifierClient:
    return vc.VerifierClient(
        config or model_config(),
        backend,
        load_prompts(prompts_dir),
        tmp_path / log_name,
        cache_dir=tmp_path / "cache",
        sleep=sleeps.append if sleeps is not None else (lambda seconds: None),
    )


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def result_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_log(path) if row.get("is_result")]


def leaves(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield (key path, value) for every scalar inside nested dicts and lists."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, path + (str(index),))
    else:
        yield path, value


def items_for(count: int) -> list[dict[str, Any]]:
    return [{**CALL, "claim_id": f"c_{index}", "claim": f"Claim {index}."} for index in range(count)]


def exception_chain(exc: BaseException) -> list[BaseException]:
    """The exception plus everything it was raised from, as a traceback would print it."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 10:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


# --------------------------------------------------------------------------- #
# Finding 1 (blocks the study): C0' is replayed from C0's cache entries.
# Break caught: a cache identity that ignores the condition.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param({"condition": "C0"}, {"condition": "C0p"}, id="C0-then-C0p"),
        pytest.param(
            {"condition": "C_closed", "evidence_sentences": []},
            {"condition": "M1", "evidence_sentences": []},
            id="C_closed-then-empty-M1",
        ),
    ],
)
def test_f01_another_condition_never_reuses_a_cached_sample(tmp_path, first, second):
    """Contract: the cache identity includes the condition. Giving C0p its own
    sample_idx range in the runner would leave this client unsafe and this test red."""
    backend = ScriptedBackend([VALID, REFUTED])
    client = make_client(tmp_path, backend)
    client.verify(**{**CALL, **first})
    row = client.verify(**{**CALL, **second})
    assert len(backend.calls) == 2, "second condition was answered from the first one's cache entry"
    assert row["cached"] is False
    assert row["verdict"] == "REFUTED"


def test_f01_run_batch_draws_every_c0p_sample_fresh(tmp_path):
    items = [
        {**CALL, "claim_id": f"c_{claim}", "claim": f"Claim {claim}.", "condition": condition,
         "sample_idx": sample}
        for claim in range(6)
        for condition in ("C0", "C0p")
        for sample in range(5)
    ]
    backend = ScriptedBackend([VALID] * len(items), delay=0.02)
    client = make_client(tmp_path, backend, config=model_config(concurrency=4))
    rows = client.run_batch(items, order_seed=7)
    replays = len(items) - len(backend.calls)
    assert replays == 0, f"{replays} of {len(items)} samples were cache replays, not API draws"
    assert [row for row in rows if row["cached"]] == []


def test_f01_guard_request_body_holds_no_condition_ids_or_labels(tmp_path):
    """Passes on callayer-v2. Fails if the fix for finding 1 moves the condition, the
    opaque id or a label into the request instead of into the cache key."""
    backend, transport = http_backend([http_reply()])
    make_client(tmp_path, backend).verify(
        claim_id="c_9f3a1b2c4d",
        claim="Drug X lowers blood pressure.",
        evidence_sentences=[
            {"idx": 7, "text": "  Systolic pressure fell by 12 mmHg.\n", "label": "SUPPORT",
             "gold_idx": [7]}
        ],
        sample_idx=3,
        condition="M2",
    )
    raw_body = transport.sent[0].body.decode("utf-8")
    body = json.loads(raw_body)
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "thinking"}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == (
        "<claim>\nDrug X lowers blood pressure.\n</claim>\n\n"
        "<evidence>\nSystolic pressure fell by 12 mmHg.\n</evidence>\n"
    )
    assert "c_9f3a1b2c4d" not in raw_body
    assert "M2" not in raw_body


# --------------------------------------------------------------------------- #
# Finding 2 (blocks the study): infrastructure failures become counted results.
# Breaks caught: a transient failure recorded without a retry; an account
# failure recorded as a result; Retry-After ignored.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(requests.ConnectionError("connection reset by peer"), id="connection-reset"),
        pytest.param(
            requests.exceptions.ChunkedEncodingError("response ended prematurely"),
            id="chunked-encoding",
        ),
        pytest.param(
            (200, "<html><body>502 Bad Gateway</body></html>", {"Content-Type": "text/html"}),
            id="html-body-200",
        ),
        pytest.param((200, json.dumps({"choices": []}), {}), id="empty-choices-200"),
    ],
)
def test_f02_transient_failures_are_retried_before_they_become_results(tmp_path, failure):
    """Goes through run_batch, the runner entry point: a retry inside verify() and a
    batch-level resume pass both satisfy it."""
    backend, transport = http_backend([failure, http_reply(), http_reply()])
    make_client(tmp_path, backend).run_batch([CALL], order_seed=1)
    outcomes = [row["outcome"] for row in result_rows(tmp_path / "calls.jsonl")]
    assert len(transport.sent) == 2, "a transient failure became the unit's result without a retry"
    assert outcomes == ["ok"]


@pytest.mark.parametrize(
    "status, message",
    [(401, "Authentication Fails"), (402, "Insufficient Balance"), (403, "Forbidden")],
)
def test_f02_account_failures_abort_instead_of_becoming_results(tmp_path, status, message):
    """Contract: 401/402/403 make verify() raise an exception that is, or is chained from,
    a BackendError, and no result row is logged. How a 400 is recorded is left open."""
    body = json.dumps({"error": {"message": message, "type": "authentication_error"}})
    backend, _ = http_backend([(status, body, {})] * 3)
    client = make_client(tmp_path, backend)
    with pytest.raises(Exception) as excinfo:
        client.verify(**CALL)
    chain = exception_chain(excinfo.value)
    assert any(isinstance(exc, BackendError) for exc in chain), chain
    assert result_rows(tmp_path / "calls.jsonl") == []


def test_f02_rate_limit_waits_at_least_retry_after(tmp_path):
    rate_limited = (429, json.dumps({"error": {"message": "Rate limit reached"}}), {"Retry-After": "20"})
    backend, _ = http_backend([rate_limited, http_reply()])
    sleeps: list[float] = []
    row = make_client(tmp_path, backend, sleeps=sleeps).verify(**CALL)
    assert row["outcome"] == "ok"
    assert sleeps and sleeps[0] >= 20, f"slept {sleeps} after Retry-After: 20"


# --------------------------------------------------------------------------- #
# Finding 3 (blocks the study): the API key reaches the JSONL log.
# Break caught: requests.InvalidHeader echoes "Bearer <key>" into row["error"].
# --------------------------------------------------------------------------- #
SECRET = "sk-review-0123456789abcdef"


@pytest.mark.parametrize(
    "raw", [SECRET + "\r", SECRET + "\r\n", SECRET + "\n"], ids=["CR", "CRLF", "LF"]
)
def test_f03_env_key_with_a_line_break_is_rejected_or_cleaned(monkeypatch, raw):
    """Contract: api_key_from_env raises ConfigError or returns the key without the break."""
    monkeypatch.setenv("REVIEW_DEEPSEEK_KEY", raw)
    config = model_config(api_key_env="REVIEW_DEEPSEEK_KEY")
    try:
        key = vc.api_key_from_env(config)
    except vc.ConfigError:
        return
    assert key == SECRET


def test_f03_malformed_key_never_reaches_the_log_or_an_exception(tmp_path):
    log = tmp_path / "calls.jsonl"
    try:
        backend, _ = http_backend([http_reply()] * 3, key=SECRET + "\r")
        surfaced = json.dumps(make_client(tmp_path, backend).verify(**CALL))
    except Exception as exc:  # an abort is acceptable if no exception in its chain shows the key
        surfaced = " | ".join(f"{type(link).__name__}: {link!r}" for link in exception_chain(exc))
    logged = log.read_text(encoding="utf-8") if log.exists() else ""
    assert SECRET not in logged, "API key written to the call log"
    assert SECRET not in surfaced, "API key surfaced in the returned row or an exception"


# --------------------------------------------------------------------------- #
# Finding 4 (biases results): non-string content or pathological JSON raises out
# of verify() and leaves no row. Break caught: a reply the parser cannot handle
# escaping as an exception instead of becoming a logged failure.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content, finish_reason",
    [
        pytest.param(None, "length", id="null-content-length"),
    ],
)
def test_f04_non_string_content_becomes_a_logged_failure(tmp_path, content, finish_reason):
    backend, _ = http_backend([http_reply(body=completion_body(content, finish_reason))] * 3)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] != "ok"
    assert row["verdict"] is None
    assert len(result_rows(tmp_path / "calls.jsonl")) == 1


# The "null-content-stop" and "non-text-parts" cases moved out of the test above
# when the contract changed on purpose: finding 5 of the callayer-v3 review was
# accepted, so a reply the provider sent with no text at all is infrastructure,
# not the model's non-compliance. It carries outcome "api_error" and stays
# resume-eligible instead of being frozen as a parse failure.
#
# Everything the old cases protected is kept: verify() does not raise, exactly
# one row is logged for the call, the outcome is not "ok", and the raw reply
# survives. "Exactly one row" is now counted over every logged row rather than
# over result rows, because is_result is False here - which also pins that an
# absent reply is recorded once and handed to a resume pass, not retried inside
# the call. The "null-content-length" case above is unchanged: a provider cut
# with finish_reason "length" stays a truncated result row.
ABSENT_TEXT = [
    pytest.param(None, "null", id="null-content-stop"),
    pytest.param(
        [{"type": "image_url", "image_url": {"url": "x"}}], "image_url", id="non-text-parts"
    ),
]


@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
def test_f04_absent_text_is_an_infrastructure_failure(tmp_path, content, raw_marker):
    backend, _ = http_backend([http_reply(body=completion_body(content, "stop"))] * 3)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "api_error", f"absent text recorded as {row['outcome']!r}"
    assert row["is_result"] is False
    assert row["outcome"] != "ok"
    assert row["verdict"] is None
    assert isinstance(row["raw_response"], str) and raw_marker in row["raw_response"], (
        f"the paid reply was dropped instead of recorded: {row['raw_response']!r}"
    )
    assert len(read_log(tmp_path / "calls.jsonl")) == 1


@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
def test_f04_absent_text_leaves_one_result_row_after_the_resume_passes(
    tmp_path, content, raw_marker
):
    """run_batch level: however many passes retry an absent reply, the unit ends with
    exactly one result row under one cache key, so its failure stays in the k=5
    denominator (SPEC 6) instead of vanishing from it or being counted twice."""
    backend, _ = http_backend([http_reply(body=completion_body(content, "stop"))] * 12)
    make_client(tmp_path, backend).run_batch([CALL], order_seed=1)
    per_key = Counter(
        entry["cache_key"] for entry in read_log(tmp_path / "calls.jsonl") if entry["is_result"]
    )
    assert list(per_key.values()) == [1], f"result rows per cache key: {dict(per_key)}"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            '{"verdict": "REFUTED", "confidence": 1' + "0" * 5000
            + ', "rationale": "r", "evidence_concerns": []}',
            id="5001-digit-integer",
        ),
        pytest.param(
            '{"verdict": "REFUTED", "confidence": 50, "rationale": "r", "evidence_concerns": [], "x": '
            + "[" * 100_000 + "]" * 100_000 + "}",
            id="nesting-depth-100000",
        ),
    ],
)
def test_f04_pathological_json_is_a_parse_error(raw):
    with pytest.raises(ParseError):
        parse_response(raw)


def test_f04_run_batch_returns_every_row_when_one_reply_is_null(tmp_path):
    items = items_for(10)
    replies = (
        [http_reply(), http_reply(), http_reply(body=completion_body(None, "stop"))]
        + [http_reply()] * 11
    )
    backend, _ = http_backend(replies)
    rows = make_client(tmp_path, backend).run_batch(items, order_seed=3)
    assert sorted(row["claim_id"] for row in rows) == sorted(item["claim_id"] for item in items)


# --------------------------------------------------------------------------- #
# Finding 5 (biases results): reruns resample failures and log replays as new
# results. Breaks caught: a recorded result drawn again on rerun; a second
# result row for one (claim, condition, sample_idx); a unit left with none.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "first_reply, outcome",
    [
        pytest.param(VALID, "ok", id="ok"),
        pytest.param("Here is my answer: " + VALID, "parse_failure", id="parse_failure"),
    ],
)
def test_f05_a_rerun_neither_redraws_nor_duplicates_a_recorded_result(tmp_path, first_reply, outcome):
    """SPEC 7: the first result row is the result and parse failures are not retried.
    Runs the batch twice on one log, the way a resume after a crash would."""
    make_client(tmp_path, ScriptedBackend([first_reply])).run_batch([CALL], order_seed=1)
    rerun_backend = ScriptedBackend([VALID])
    make_client(tmp_path, rerun_backend).run_batch([CALL], order_seed=1)
    assert rerun_backend.calls == [], "a recorded result was drawn again on rerun"
    assert [row["outcome"] for row in result_rows(tmp_path / "calls.jsonl")] == [outcome]


def test_f05_a_rerun_leaves_one_result_row_after_an_api_error(tmp_path):
    """Contract: one result row per unit. Whether a final API error stays the result or
    is re-called on rerun is the SPEC's call; either answer passes."""
    config = model_config(max_retries=0)
    first_backend = ScriptedBackend([BackendTimeout("read timed out")] * 3)  # any resume passes
    make_client(tmp_path, first_backend, config=config).run_batch([CALL], order_seed=1)
    make_client(tmp_path, ScriptedBackend([VALID]), config=config).run_batch([CALL], order_seed=1)
    outcomes = [row["outcome"] for row in result_rows(tmp_path / "calls.jsonl")]
    assert len(outcomes) == 1, f"result rows for one unit: {outcomes}"


def test_f05_guard_every_unit_ends_with_exactly_one_result_row(tmp_path):
    """Passes on callayer-v2. Two of six units time out on every attempt; each unit must
    still end with exactly one result row, so its failures stay in the k=5 denominator."""
    backend = KeyedBackend({"Claim 1.", "Claim 3."})
    make_client(tmp_path, backend, config=model_config(max_retries=0)).run_batch(
        items_for(6), order_seed=11
    )
    per_unit = Counter(row["claim_id"] for row in result_rows(tmp_path / "calls.jsonl"))
    assert per_unit == {f"c_{index}": 1 for index in range(6)}


# --------------------------------------------------------------------------- #
# Finding 6 (biases results): finish_reason never changes the outcome.
# Break caught: a server-side interruption or a provider cut counted as the
# model's own parse failure.
# --------------------------------------------------------------------------- #
def test_f06_insufficient_system_resource_is_retried_before_it_becomes_a_result(tmp_path):
    """DeepSeek documents this finish_reason as a server-side interruption. Goes through
    run_batch so an in-call retry and a resume pass both satisfy it."""
    interrupted = ('{"verdict": "SUPP', "insufficient_system_resource")
    backend = ScriptedBackend([interrupted, VALID, VALID])
    make_client(tmp_path, backend).run_batch([CALL], order_seed=1)
    outcomes = [row["outcome"] for row in result_rows(tmp_path / "calls.jsonl")]
    assert len(backend.calls) == 2, "a server-side interruption became the unit's result"
    assert outcomes == ["ok"]


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(("", "content_filter"), id="content-filter-empty"),
        pytest.param(('{"verdict": "REFUTED", "confidence": 9', "length"), id="length-cut-json"),
    ],
)
def test_f06_provider_cut_replies_are_not_model_parse_failures(tmp_path, reply):
    """Contract: these replies get their own outcome. SPEC 6 then has to count that
    outcome among the 5, like parse failures and API errors."""
    row = make_client(tmp_path, ScriptedBackend([reply] * 3)).verify(**CALL)
    assert row["outcome"] not in {"ok", "parse_failure"}, row["outcome"]
    assert row["finish_reason"] == reply[1]


# --------------------------------------------------------------------------- #
# Finding 7 (minor): cache identity gaps beyond the condition.
# Breaks caught: replay across endpoints or retry policies; a stale cached parse
# replayed unchecked; the serving model snapshot never recorded.
# --------------------------------------------------------------------------- #
def test_f07_another_base_url_does_not_replay_this_cache(tmp_path):
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    config = model_config(base_url="https://proxy.example.test/v1")
    row = make_client(tmp_path, backend, config=config).verify(**CALL)
    assert len(backend.calls) == 1, "another endpoint was served from this cache"
    assert row["verdict"] == "REFUTED"


def test_f07_strict_run_does_not_replay_a_parse_retried_sample(tmp_path):
    lax = model_config(retry_on=["api_error", "parse_failure"])
    make_client(tmp_path, ScriptedBackend(["Sure: " + VALID, VALID]), config=lax).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    row = make_client(tmp_path, backend).verify(**CALL)
    assert len(backend.calls) == 1, "strict run replayed a sample drawn after a parse-failure retry"
    assert row["verdict"] == "REFUTED"


def test_f07_cache_hit_revalidates_the_stored_reply(tmp_path):
    """The edited entry stands in for one written by an older, looser parser."""
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    entry_path = next((tmp_path / "cache").glob("*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["raw"] = '{"verdict": "supported", "confidence": "80"}'
    entry["parsed"] = {"verdict": "supported", "confidence": "80", "rationale": "ok",
                       "evidence_concerns": []}
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    try:
        client = make_client(tmp_path, ScriptedBackend([VALID]), log_name="calls2.jsonl")
        row = client.verify(**CALL)
    except vc.ConfigError:
        return  # refusing the stale entry loudly is acceptable
    assert row["verdict"] in (None, *VERDICTS), f"replayed verdict {row['verdict']!r}"
    assert row["confidence"] is None or type(row["confidence"]) is int


def test_f07_rows_record_the_model_snapshot_that_served_the_call(tmp_path):
    body = completion_body(model="deepseek-flash-2026-11-01", fingerprint="fp_3a9c1e")
    backend, _ = http_backend([http_reply(body=body)])
    row = make_client(tmp_path, backend).verify(**CALL)
    values = [value for _, value in leaves(row)]
    assert "deepseek-flash-2026-11-01" in values, "served model id not recorded"
    assert "fp_3a9c1e" in values, "system_fingerprint not recorded"


@pytest.mark.parametrize(
    "before, after",
    [
        pytest.param({}, {"model": "deepseek-other"}, id="model"),
        pytest.param({}, {"temperature": 1.0}, id="temperature"),
        pytest.param({}, {"top_p": 0.9}, id="top_p"),
        pytest.param({}, {"max_tokens": 256}, id="max_tokens"),
        pytest.param({}, {"thinking": {"enabled": True}}, id="thinking"),
        pytest.param(
            {"thinking": {"enabled": True}},
            {"thinking": {"enabled": True, "budget_tokens": 1024}},
            id="thinking-budget",
        ),
    ],
)
def test_f07_guard_output_settings_stay_in_the_cache_identity(tmp_path, before, after):
    """Passes on callayer-v2. Guards the key rewrite that findings 1 and 7 need."""
    make_client(tmp_path, ScriptedBackend([VALID]), config=model_config(**before)).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    make_client(tmp_path, backend, config=model_config(**after)).verify(**CALL)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("changed", ["verifier_system.txt", "verifier_user.txt"])
def test_f07_guard_prompt_bytes_stay_in_the_cache_identity(tmp_path, changed):
    """Passes on callayer-v2. Guards the same key rewrite for both prompt files."""
    variant = tmp_path / "prompts_variant"
    variant.mkdir()
    for name in ("verifier_system.txt", "verifier_user.txt"):
        data = (PROMPTS / name).read_bytes()
        (variant / name).write_bytes(data + b"\n" if name == changed else data)
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    make_client(tmp_path, backend, prompts_dir=variant).verify(**CALL)
    assert len(backend.calls) == 1


# --------------------------------------------------------------------------- #
# Finding 8 (minor): cache races and failed replaces lose paid replies.
# Breaks caught: two API calls for one unit; a reply dropped from the log when
# the cache write fails or a second writer interleaves.
# --------------------------------------------------------------------------- #
def test_f08_concurrent_calls_for_one_unit_reach_the_api_once(tmp_path):
    backend = ScriptedBackend([VALID, REFUTED], delay=0.2)
    client = make_client(tmp_path, backend, config=model_config(concurrency=2))
    barrier = threading.Barrier(2)
    rows: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        try:
            rows.append(client.verify(**CALL))
        except BaseException as exc:  # a worker thread cannot fail the test itself
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(backend.calls) == 1, "one unit was sent to the API twice"
    assert {row["raw_response"] for row in rows} == {VALID}


def test_f08_reply_is_logged_even_when_the_cache_write_fails(tmp_path, monkeypatch):
    client = make_client(tmp_path, ScriptedBackend([VALID]))

    def locked(src, dst):  # what a OneDrive or antivirus handle causes on Windows
        raise PermissionError(13, "Access is denied", str(dst))

    monkeypatch.setattr(os, "replace", locked)
    try:
        client.verify(**CALL)
    except OSError:
        pass  # failing loudly is acceptable; losing the paid reply is not
    assert [row["raw_response"] for row in read_log(tmp_path / "calls.jsonl")] == [VALID]


def test_f08_interleaved_cache_writers_both_keep_their_rows(tmp_path, monkeypatch):
    """Two runner processes on one key: writer B lands between A's temp write and A's replace."""
    writer_a = make_client(tmp_path, ScriptedBackend([VALID]), log_name="a.jsonl")
    writer_b = make_client(tmp_path, ScriptedBackend([VALID]), log_name="b.jsonl")
    real_replace = os.replace
    pending = [writer_b]

    def interleave(src, dst):
        if pending:
            pending.pop().verify(**CALL)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", interleave)
    try:
        writer_a.verify(**CALL)
    except OSError:
        pass
    assert len(read_log(tmp_path / "b.jsonl")) == 1
    assert len(read_log(tmp_path / "a.jsonl")) == 1, "writer A's paid reply was lost"


# --------------------------------------------------------------------------- #
# Finding 9 (minor): the parser is lenient where SPEC 4 says strict.
# Breaks caught: duplicate keys silently resolved to the last value; fence
# variants handled unequally.
# --------------------------------------------------------------------------- #
BARE = '{"verdict": "REFUTED", "confidence": 40, "rationale": "r", "evidence_concerns": []}'
BARE_PARSED = {"verdict": "REFUTED", "confidence": 40, "rationale": "r", "evidence_concerns": []}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            '{"verdict": "SUPPORTED", "verdict": "REFUTED", "confidence": 80, "rationale": "r", '
            '"evidence_concerns": []}',
            id="duplicate-verdict",
        ),
        pytest.param(
            '{"verdict": "SUPPORTED", "confidence": 99, "confidence": 80, "rationale": "r", '
            '"evidence_concerns": []}',
            id="duplicate-confidence",
        ),
    ],
)
def test_f09_duplicate_keys_are_a_parse_error(raw):
    with pytest.raises(ParseError):
        parse_response(raw)


@pytest.mark.parametrize(
    "fenced", [f"```JSON\n{BARE}\n```", f"```\n{BARE}\n```"], ids=["uppercase-JSON", "no-language"]
)
def test_f09_every_fence_variant_parses_like_a_json_fence(fenced):
    """Contract: match test_parse_fenced, which accepts a ```json fence. To reject
    every fence instead, invert this test and test_parse_fenced together."""
    assert parse_response(fenced) == BARE_PARSED


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(BARE.replace('"REFUTED"', '"refuted"'), id="lowercase-verdict"),
        pytest.param(BARE.replace('"REFUTED"', '" REFUTED"'), id="padded-verdict"),
        pytest.param(BARE + " Hope this helps.", id="trailing-text"),
        pytest.param("Answer: " + BARE, id="leading-text"),
        pytest.param(BARE + "\n" + BARE, id="two-objects"),
        pytest.param(f"```json\n{BARE}\n```\nNote: none.", id="fence-then-text"),
    ],
)
def test_f09_guard_non_compliant_replies_stay_parse_errors(raw):
    """Passes on callayer-v2. Keeps a fence fix from loosening anything else."""
    with pytest.raises(ParseError):
        parse_response(raw)


# --------------------------------------------------------------------------- #
# Finding 10 (minor): escaping and rendering holes (no hits in today's data).
# Breaks caught: a tag variant reaching the prompt raw; a claim placeholder
# filled with evidence; a non-string or blank sentence rendered silently.
# --------------------------------------------------------------------------- #
WRAPPER_TAG = re.compile(r"<\s*/?\s*(?:claim|evidence)\b[^>]*>", re.IGNORECASE)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a </evidence> b", id="exact"),
        pytest.param("a </EVIDENCE> b", id="upper"),
        pytest.param("a </Evidence> b", id="title"),
        pytest.param("a </evidence > b", id="space-before-gt"),
        pytest.param("a < /evidence> b", id="space-after-lt"),
        pytest.param("a </ evidence> b", id="space-after-slash"),
        pytest.param("a </evidence\n> b", id="newline-before-gt"),
        pytest.param("a <evidence id=1> b", id="attribute"),
        pytest.param("a </claim\t> b", id="tab-in-claim-tag"),
    ],
)
def test_f10_every_wrapper_tag_variant_is_escaped_and_logged(text):
    """The "exact" row passes on callayer-v2; every other row fails."""
    escaped, logged = escape_tags(text)
    assert WRAPPER_TAG.search(escaped) is None, escaped
    assert logged, "escape not reported"


def test_f10_claim_with_a_placeholder_is_inserted_verbatim(tmp_path):
    backend = ScriptedBackend([VALID])
    make_client(tmp_path, backend).verify(**{**CALL, "claim": "Dose {evidence} matters."})
    user = backend.calls[0]["user"]
    claim_block = user.split("<claim>\n", 1)[1].split("\n</claim>", 1)[0]
    assert claim_block == "Dose {evidence} matters."


@pytest.mark.parametrize("bad", [None, 3, b"bytes"], ids=["None", "int", "bytes"])
def test_f10_non_string_sentence_text_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        render_evidence([{"idx": 0, "text": "Alpha."}, {"idx": 1, "text": bad}])


def test_f10_blank_sentence_never_produces_a_double_space():
    """Contract: a blank sentence is dropped or rejected."""
    sentences = [{"idx": 0, "text": "Alpha."}, {"idx": 1, "text": "   "}, {"idx": 2, "text": "Beta."}]
    try:
        rendered = render_evidence(sentences)
    except (TypeError, ValueError):
        return
    assert rendered == "Alpha. Beta."


# --------------------------------------------------------------------------- #
# Finding 11 (minor): ordering and audit-trail gaps.
# Breaks caught: an unseeded or ambiguous shuffle; results misaligned with their
# inputs; facts an audit needs missing from the row.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [None, "7", 7.0], ids=["None", "str", "float"])
def test_f11_run_batch_requires_an_integer_order_seed(tmp_path, seed):
    client = make_client(tmp_path, ScriptedBackend([VALID]))
    with pytest.raises((TypeError, ValueError)):
        client.run_batch([CALL], order_seed=seed)


def test_f11_run_batch_returns_rows_in_input_order(tmp_path):
    rows = make_client(tmp_path, ScriptedBackend([VALID] * 6)).run_batch(items_for(6), order_seed=11)
    assert [row["claim_id"] for row in rows] == ["c_0", "c_1", "c_2", "c_3", "c_4", "c_5"]


def test_f11_execution_order_depends_on_the_seed_not_the_input_order(tmp_path):
    orders = []
    for name, batch in (("forward", items_for(6)), ("reversed", items_for(6)[::-1])):
        directory = tmp_path / name
        directory.mkdir()
        backend = ScriptedBackend([VALID] * 6)
        make_client(directory, backend).run_batch(batch, order_seed=11)
        orders.append([call["user"] for call in backend.calls])
    assert orders[0] == orders[1]


def as_utc(value: Any) -> datetime | None:
    """A timezone-aware ISO string or epoch seconds/milliseconds as a UTC datetime."""
    if isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return stamp if stamp.tzinfo is not None else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1e9:
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, timezone.utc)
    return None


def test_f11_rows_carry_a_utc_timestamp(tmp_path):
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    row = make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    after = datetime.now(timezone.utc) + timedelta(seconds=1)
    stamps = [as_utc(value) for _, value in leaves(row)]
    assert any(s is not None and before <= s <= after for s in stamps), "no UTC timestamp in the row"


def test_f11_error_rows_record_the_http_status_as_a_number(tmp_path):
    overloaded = (503, json.dumps({"error": {"message": "Server overloaded"}}), {})
    backend, _ = http_backend([overloaded] * 3)
    row = make_client(tmp_path, backend).verify(**CALL)
    numbers = [value for _, value in leaves(row) if type(value) is int]
    assert 503 in numbers, "HTTP status appears only inside the error text"


def test_f11_rows_record_reasoning_tokens(tmp_path):
    backend, _ = http_backend([http_reply(body=completion_body(reasoning_tokens=37))])
    row = make_client(tmp_path, backend).verify(**CALL)
    reasoning = [
        value for path, value in leaves(row)
        if type(value) is int and any("reasoning" in part for part in path)
    ]
    assert reasoning == [37], "thinking OFF cannot be checked per response"


def test_f11_rows_keep_the_rendered_user_prompt(tmp_path):
    backend = ScriptedBackend([VALID])
    row = make_client(tmp_path, backend).verify(**CALL)
    assert backend.calls[0]["user"] in [value for _, value in leaves(row)]


def test_f11_rows_record_the_user_template_hash(tmp_path):
    template_sha256 = hashlib.sha256((PROMPTS / "verifier_user.txt").read_bytes()).hexdigest()
    row = make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    assert template_sha256 in [value for _, value in leaves(row)]
