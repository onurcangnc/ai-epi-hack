"""Tests that pin the 8 contracts of the callayer-v3 review (2026-11-15).

Tests named ``test_cNN_*`` fail on callayer-v3 by design; each one turns green
when contract NN is satisfied. Tests with ``guard`` in their name pass on
callayer-v3 and protect behavior the fixes must keep - including the cheap wrong
fixes: "always raise StateError", "treat every 4xx as blocked", "escape every
angle bracket", "put something volatile in the cache key".

Where a contract leaves a design choice open, the test docstring names the
choice it picks, so the choice can be changed in one place.

Nothing here edits src/, SPEC.md or another test file. ``StateError`` is
resolved lazily through ``vc`` so that its absence fails only contract 2 instead
of breaking collection of the whole module. Source is ASCII only: the
zero-width and no-break characters contract 8 needs are written as escapes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter

import verifier_client as vc
from backend import BackendResult, DeepSeekBackend
from prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
M1_PATH = ROOT / "data" / "conditions" / "M1.jsonl"

VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'
REFUTED = '{"verdict": "REFUTED", "confidence": 61, "rationale": "no", "evidence_concerns": []}'

# One marker list for every test in this file; the config key it stands for is
# models.deepseek.blocked_body_markers (contract 3).
MARKERS = ["Content Exists Risk", "content_filter"]

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
        "blocked_body_markers": list(MARKERS),
    }
    for key, value in overrides.items():
        if key == "thinking":
            section["thinking"] = {**section["thinking"], **value}
        else:
            section[key] = value
    return vc.ModelConfig.from_dict(section)


def state_error() -> type[BaseException]:
    """The exception class contract 2 introduces, resolved at call time."""
    exc = getattr(vc, "StateError", None)
    assert exc is not None, (
        "contract 2: src/verifier_client.py defines no StateError, so run_batch "
        "cannot report a cache the log does not account for"
    )
    assert isinstance(exc, type) and issubclass(exc, Exception), f"StateError is {exc!r}"
    return exc


class ScriptedBackend:
    """Replaces the HTTP backend. A reply is a str, a (text, finish_reason) tuple or an exception."""

    def __init__(self, replies: list[Any], timeout: float = 120.0) -> None:
        self.replies = list(replies)
        self.timeout = timeout
        self.calls: list[str] = []

    def complete(self, system: str, user: str, **kwargs: Any) -> BackendResult:
        self.calls.append(user)
        if not self.replies:
            raise AssertionError("backend called more often than the test allows")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        return BackendResult(
            text=text, input_tokens=523, output_tokens=84, latency_ms=900.0,
            finish_reason=finish_reason, http_status=200, model="deepseek-test",
        )


class KeyedBackend:
    """Times out on every call whose claim is listed; answers every other call with VALID."""

    timeout = 120.0

    def __init__(self, failing_claims: set[str]) -> None:
        self.failing = {f"<claim>\n{claim}\n</claim>" for claim in failing_claims}
        self.calls: list[str] = []

    def complete(self, system: str, user: str, **kwargs: Any) -> BackendResult:
        self.calls.append(user)
        if any(block in user for block in self.failing):
            raise vc.BackendTimeout("read timed out")
        return BackendResult(
            text=VALID, input_tokens=523, output_tokens=84, latency_ms=900.0,
            finish_reason="stop", http_status=200,
        )


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
        status, raw, headers = self.replies.pop(0)
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


def completion_body(content: Any = VALID, finish_reason: str | None = "stop") -> str:
    return json.dumps(
        {
            "id": "chatcmpl-v3",
            "object": "chat.completion",
            "created": 1763130000,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 523, "completion_tokens": 84},
            "system_fingerprint": "fp_v3",
        }
    )


def ok_reply() -> tuple[int, str, dict[str, str]]:
    return (200, completion_body(), {})


def error_reply(status: int, message: str) -> tuple[int, str, dict[str, str]]:
    return (status, json.dumps({"error": {"message": message, "type": "invalid_request_error"}}), {})


def http_backend(
    replies: list[Any], *, timeout: float = 120.0
) -> tuple[DeepSeekBackend, CannedTransport]:
    transport = CannedTransport(replies)
    session = requests.Session()
    session.trust_env = False  # no proxy or registry lookups from the test machine
    session.mount("https://", transport)
    return (
        DeepSeekBackend(
            "https://api.example.test", "sk-test-key", "deepseek-test",
            timeout=timeout, session=session,
        ),
        transport,
    )


def make_client(
    tmp_path: Path,
    backend: Any,
    *,
    config: vc.ModelConfig | None = None,
    log_name: str = "calls.jsonl",
    cache_name: str = "cache",
    sleeps: list[float] | None = None,
) -> vc.VerifierClient:
    return vc.VerifierClient(
        config or model_config(),
        backend,
        load_prompts(PROMPTS),
        tmp_path / log_name,
        cache_dir=tmp_path / cache_name,
        sleep=sleeps.append if sleeps is not None else (lambda seconds: None),
    )


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def result_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_log(path) if row.get("is_result")]


def write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# Deliberately not sorted, so "input order" is distinguishable from "sorted order".
INPUT_ORDER = ("c_4", "c_2", "c_5", "c_0", "c_3", "c_1")


def units(ids: tuple[str, ...] = INPUT_ORDER) -> list[dict[str, Any]]:
    return [{**CALL, "claim_id": cid, "claim": f"Claim {cid}."} for cid in ids]


# --------------------------------------------------------------------------- #
# Contract 1 (blocks the study): run_batch returns one row per input unit, in
# input order, on a fresh run and on both kinds of resume; a unit that is
# already recorded comes back as its recorded result row.
# --------------------------------------------------------------------------- #
def test_c01_guard_fresh_run_returns_every_unit_in_input_order(tmp_path):
    """Passes on callayer-v3. Keeps the resume fix from reordering a fresh run."""
    items = units()
    rows = make_client(tmp_path, ScriptedBackend([VALID] * 6)).run_batch(items, order_seed=3)
    assert [row["claim_id"] for row in rows] == list(INPUT_ORDER)
    assert len(rows) == len(items)


def test_c01_full_resume_returns_every_unit(tmp_path):
    """Contract: a run_batch whose units are all recorded still returns one row each,
    in input order, drawing nothing. Returning [] and leaving the caller to re-read
    the log would leave every runner silently writing a truncated results file."""
    items = units()
    make_client(tmp_path, ScriptedBackend([VALID] * 6)).run_batch(items, order_seed=3)
    backend = ScriptedBackend([])  # any API call is an error here, not just a waste
    rows = make_client(tmp_path, backend).run_batch(items, order_seed=3)
    assert backend.calls == [], "a recorded unit was drawn again on resume"
    assert [row["claim_id"] for row in rows] == list(INPUT_ORDER)


def test_c01_partial_resume_returns_recorded_and_new_units_together(tmp_path):
    items = units()
    recorded, fresh = items[:3], items[3:]
    make_client(tmp_path, ScriptedBackend([VALID] * 3)).run_batch(recorded, order_seed=3)
    backend = ScriptedBackend([REFUTED] * 3)
    rows = make_client(tmp_path, backend).run_batch(items, order_seed=3)
    assert len(backend.calls) == len(fresh), "resume redrew a recorded unit or skipped a new one"
    assert [row["claim_id"] for row in rows] == list(INPUT_ORDER)


def test_c01_recorded_units_come_back_as_their_recorded_result_row(tmp_path):
    """Contract: the row returned for an already-recorded unit is the logged result
    row itself (same cache_key, same timestamp, is_result true), not a fresh cache
    replay. SPEC 7 makes the first result row the row analyzed."""
    items = units()
    first = make_client(tmp_path, ScriptedBackend([VALID] * 6)).run_batch(items, order_seed=3)
    assert len(first) == 6
    logged = {row["cache_key"]: row for row in result_rows(tmp_path / "calls.jsonl")}
    rows = make_client(tmp_path, ScriptedBackend([])).run_batch(items, order_seed=3)
    assert len(rows) == 6
    for row in rows:
        recorded = logged[row["cache_key"]]
        assert row["is_result"] is True, f"{row['claim_id']} came back as a non-result row"
        assert row["timestamp"] == recorded["timestamp"], "not the recorded row"
        assert row["outcome"] == recorded["outcome"]
        assert row["raw_response"] == recorded["raw_response"]


def test_c01_one_row_per_unit_even_when_a_unit_fails_for_good(tmp_path):
    """Two of six units fail on every attempt; each must still contribute exactly
    one returned row, so its failures stay in the k=5 denominator."""
    items = units()
    backend = KeyedBackend({"Claim c_2.", "Claim c_5."})
    rows = make_client(tmp_path, backend, config=model_config(max_retries=0)).run_batch(
        items, order_seed=3
    )
    assert [row["claim_id"] for row in rows] == list(INPUT_ORDER)
    outcomes = {row["claim_id"]: row["outcome"] for row in rows}
    assert outcomes["c_2"] == "api_error" and outcomes["c_5"] == "api_error"
    assert all(outcomes[cid] == "ok" for cid in ("c_0", "c_1", "c_3", "c_4"))


# --------------------------------------------------------------------------- #
# Contract 2 (blocks the study): a cache the log does not account for is a
# StateError, raised before any API call. Break caught: a warm cache plus a
# fresh log replaying the whole run with is_result false, so the run produces no
# analyzable row at all.
# --------------------------------------------------------------------------- #
def warm_cache(tmp_path: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Populate tmp_path/cache and tmp_path/calls.jsonl with a complete recorded run."""
    make_client(tmp_path, ScriptedBackend([VALID] * len(items))).run_batch(items, order_seed=3)
    return result_rows(tmp_path / "calls.jsonl")


def test_c02_populated_cache_with_a_missing_log_is_a_state_error(tmp_path):
    exc = state_error()
    items = units()
    warm_cache(tmp_path, items)
    backend = ScriptedBackend([])
    client = make_client(tmp_path, backend, log_name="fresh.jsonl")
    with pytest.raises(exc):
        client.run_batch(items, order_seed=3)
    assert backend.calls == [], "StateError must be raised before any API call"
    assert read_log(tmp_path / "fresh.jsonl") == [], "rows were written before the check"


def test_c02_populated_cache_with_an_empty_log_is_a_state_error(tmp_path):
    exc = state_error()
    items = units()
    warm_cache(tmp_path, items)
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    backend = ScriptedBackend([])
    client = make_client(tmp_path, backend, log_name="empty.jsonl")
    with pytest.raises(exc):
        client.run_batch(items, order_seed=3)
    assert backend.calls == []


def test_c02_cache_key_without_a_result_row_is_a_state_error(tmp_path):
    """The log names every cache key but records none of them as a result, which is
    what a log rotated mid-run, or a log of replays, looks like."""
    exc = state_error()
    items = units()
    recorded = warm_cache(tmp_path, items)
    write_log(
        tmp_path / "replays.jsonl",
        [{**row, "is_result": False, "cached": True} for row in recorded],
    )
    backend = ScriptedBackend([])
    client = make_client(tmp_path, backend, log_name="replays.jsonl")
    with pytest.raises(exc):
        client.run_batch(items, order_seed=3)
    assert backend.calls == []


def test_c02_guard_a_fresh_run_with_no_cache_does_not_raise(tmp_path):
    """Passes on callayer-v3. Blocks the cheap fix of raising StateError whenever a
    log is missing: a first run has no log and no cache and must proceed."""
    items = units()
    rows = make_client(tmp_path, ScriptedBackend([VALID] * 6)).run_batch(items, order_seed=3)
    assert len(rows) == 6


def test_c02_guard_a_cache_fully_covered_by_result_rows_does_not_raise(tmp_path):
    """Passes on callayer-v3. A normal resume on its own log is not a StateError."""
    items = units()
    warm_cache(tmp_path, items)
    client = make_client(tmp_path, ScriptedBackend([]))
    client.run_batch(items, order_seed=3)  # must not raise


def test_c02_guard_a_partly_done_run_on_its_own_log_does_not_raise(tmp_path):
    """Passes on callayer-v3. Three units recorded, three never attempted: the cache
    holds only keys the log accounts for, so the run continues."""
    items = units()
    make_client(tmp_path, ScriptedBackend([VALID] * 3)).run_batch(items[:3], order_seed=3)
    client = make_client(tmp_path, ScriptedBackend([VALID] * 3))
    client.run_batch(items, order_seed=3)  # must not raise


# --------------------------------------------------------------------------- #
# Contract 3 (biases results): a 400 the provider uses for content refusal is an
# outcome; every other non-auth 4xx is infrastructure, and a run of them aborts.
# Break caught: a config-level 4xx recorded as a counted api_error result row
# for every unit in the run.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        pytest.param("Content Exists Risk", id="deepseek-risk-wording"),
        pytest.param("rejected by content_filter", id="marker-inside-a-longer-body"),
    ],
)
def test_c03_marked_400_is_a_blocked_result_row(tmp_path, message):
    """Contract: the marker list comes from models.deepseek.blocked_body_markers and
    is matched as a substring of the response body. The outcome is "blocked", which
    SPEC 6 already counts among the 5 and gives no verdict."""
    backend, transport = http_backend([error_reply(400, message)] * 3)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "blocked", f"marked 400 recorded as {row['outcome']!r}"
    assert row["is_result"] is True
    assert row["verdict"] is None
    assert row["http_status"] == 400
    assert len(transport.sent) == 1, "a refusal is an outcome, not something to retry"


@pytest.mark.parametrize(
    "status, message",
    [
        pytest.param(400, "Invalid max_tokens value", id="400-bad-parameter"),
        pytest.param(404, "Model Not Exist", id="404-unknown-model"),
        pytest.param(422, "Unprocessable Entity", id="422-schema"),
    ],
)
def test_c03_unmarked_non_auth_4xx_is_never_a_result_row(tmp_path, status, message):
    """Contract: an unmarked 4xx is infrastructure, so it carries outcome "api_error"
    and is_result false even when verify() is called outside run_batch. SPEC 7 only
    lets an api_error become a result row after the retry budget AND both resume
    passes, which a bare verify() has not exhausted."""
    backend, _ = http_backend([error_reply(status, message)] * 4)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "api_error"
    assert row["is_result"] is False, (
        f"HTTP {status} became a counted result row; a config-level 4xx would fill "
        "every cell of the run with api_error results"
    )
    assert row["http_status"] == status


def test_c03_third_consecutive_unmarked_4xx_aborts_the_run(tmp_path):
    """Contract: three consecutive unmarked non-auth 4xx outcomes raise RunAborted,
    the same way 401/402/403 do. The count is over consecutive units, not the run."""
    items = units(("c_0", "c_1", "c_2"))
    backend, transport = http_backend([error_reply(400, "Invalid max_tokens value")] * 12)
    client = make_client(tmp_path, backend)
    with pytest.raises(vc.RunAborted):
        client.run_batch(items, order_seed=3)
    assert len(transport.sent) <= 9, "aborted later than the third consecutive failure"


def test_c03_a_success_resets_the_consecutive_4xx_counter(tmp_path):
    """Contract: "consecutive", not "cumulative" - two failures, a success, then two
    more failures must not abort, so a flaky endpoint does not kill a long run.
    Also pins that an unmarked 4xx costs one request, not a retry storm."""
    items = units(("c_0", "c_1", "c_2", "c_3", "c_4"))
    bad = error_reply(400, "Invalid max_tokens value")
    backend, _ = http_backend([bad, bad, ok_reply(), bad, bad])
    client = make_client(tmp_path, backend)
    rows = client.run_batch(items, order_seed=3, resume_passes=0)
    assert len(rows) == 5
    assert sum(1 for row in rows if row["outcome"] == "api_error") == 4


def test_c03_guard_auth_failures_still_abort(tmp_path):
    """Passes on callayer-v3. The 4xx rework must not loosen 401/402/403."""
    backend, _ = http_backend([error_reply(402, "Insufficient Balance")] * 3)
    client = make_client(tmp_path, backend)
    with pytest.raises(vc.RunAborted):
        client.verify(**CALL)
    assert result_rows(tmp_path / "calls.jsonl") == []


def test_c03_guard_a_500_is_still_retried_in_call(tmp_path):
    """Passes on callayer-v3. Blocks the cheap fix of routing every non-200 through
    the new abort counter: a 5xx is retryable and must stay so."""
    overloaded = (503, json.dumps({"error": {"message": "Server overloaded"}}), {})
    backend, transport = http_backend([overloaded, ok_reply()])
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "ok"
    assert len(transport.sent) == 2


# --------------------------------------------------------------------------- #
# Contract 4 (biases results): a reply with no text at all is the provider's
# failure; a reply with empty text is the model's. Break caught: content null
# scored as the model's parse failure and therefore never resumed.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="null-content"),
        pytest.param([{"type": "image_url", "image_url": {"url": "x"}}], id="non-text-parts"),
    ],
)
def test_c04_absent_text_with_finish_stop_is_an_api_error(tmp_path, content):
    """Contract: no string content means the provider returned no reply, so the
    outcome is api_error and the unit stays resume-eligible. Scoring it as
    parse_failure both freezes it (parse failures are not retried) and moves a
    provider glitch into the model's own non-compliance rate."""
    backend, _ = http_backend([(200, completion_body(content, "stop"), {})] * 4)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "api_error", f"absent text recorded as {row['outcome']!r}"
    assert row["verdict"] is None
    assert row["is_result"] is False, "an api_error from a bare verify() is resume-eligible"


def test_c04_absent_text_is_retried_by_a_resume_pass(tmp_path):
    backend, transport = http_backend(
        [(200, completion_body(None, "stop"), {}), ok_reply(), ok_reply()]
    )
    rows = make_client(tmp_path, backend).run_batch([CALL], order_seed=1)
    assert len(transport.sent) == 2, "a null reply was never resumed"
    assert [row["outcome"] for row in rows] == ["ok"]
    assert [row["outcome"] for row in result_rows(tmp_path / "calls.jsonl")] == ["ok"]


def test_c04_guard_empty_string_content_stays_a_parse_failure(tmp_path):
    """Passes on callayer-v3. An empty string IS a reply: the model answered with
    nothing, which is its own non-compliance and must not be retried away."""
    backend, transport = http_backend([(200, completion_body("", "stop"), {})] * 4)
    row = make_client(tmp_path, backend).verify(**CALL)
    assert row["outcome"] == "parse_failure"
    assert row["raw_response"] == ""
    assert row["is_result"] is True
    assert len(transport.sent) == 1, "a parse failure is an outcome and is not retried"


# --------------------------------------------------------------------------- #
# Contract 5 (biases results): the provider's Retry-After is obeyed in full.
# Break caught: a 60 s cap that retries into the same rate limit and burns the
# retry budget until the unit becomes an api_error result row.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seconds", ["900", "900.0"], ids=["integer", "float"])
def test_c05_retry_after_is_obeyed_in_full(tmp_path, seconds):
    rate_limited = (429, json.dumps({"error": {"message": "Rate limit reached"}}),
                    {"Retry-After": seconds})
    backend, _ = http_backend([rate_limited, ok_reply()])
    sleeps: list[float] = []
    row = make_client(tmp_path, backend, sleeps=sleeps).verify(**CALL)
    assert row["outcome"] == "ok"
    assert sleeps, "no wait at all before the retry"
    assert sleeps[0] >= 900, f"slept {sleeps} after Retry-After: {seconds}"


def test_c05_guard_backoff_without_a_retry_after_header_stays_bounded(tmp_path):
    """Passes on callayer-v3. Lifting the Retry-After cap must not uncap the plain
    exponential backoff, which no server asked for."""
    overloaded = (503, json.dumps({"error": {"message": "Server overloaded"}}), {})
    backend, _ = http_backend([overloaded, overloaded, ok_reply()])
    sleeps: list[float] = []
    row = make_client(tmp_path, backend, sleeps=sleeps).verify(**CALL)
    assert row["outcome"] == "ok"
    assert sleeps and max(sleeps) <= 60.0, f"unbounded backoff: {sleeps}"


# --------------------------------------------------------------------------- #
# Contract 6 (minor): the cache identity covers retry depth and the request
# timeout. Break caught: a run with a different retry budget or a shorter
# timeout replaying samples it would have drawn differently.
# --------------------------------------------------------------------------- #
def test_c06_cache_key_changes_with_max_retries(tmp_path):
    backend = ScriptedBackend([])
    base = make_client(tmp_path, backend, config=model_config(), cache_name="k1")
    other = make_client(tmp_path, backend, config=model_config(max_retries=9), cache_name="k2")
    assert base.cache_key("C0", "prompt", 0) != other.cache_key("C0", "prompt", 0)


def test_c06_cache_key_changes_with_the_backend_timeout(tmp_path):
    """Contract: the timeout is output-affecting (it decides whether a slow reply is
    kept or becomes an api_error), so it belongs in the cache identity. Reading it
    from the config or from the backend both satisfy this."""
    slow, _ = http_backend([], timeout=120.0)
    quick, _ = http_backend([], timeout=5.0)
    base = make_client(tmp_path, slow, cache_name="k1")
    other = make_client(tmp_path, quick, cache_name="k2")
    assert base.cache_key("C0", "prompt", 0) != other.cache_key("C0", "prompt", 0)


def test_c06_a_deeper_retry_budget_does_not_replay_this_cache(tmp_path):
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    make_client(tmp_path, backend, config=model_config(max_retries=9)).verify(**CALL)
    assert len(backend.calls) == 1, "a run with a different max_retries replayed this cache"


def test_c06_a_shorter_timeout_does_not_replay_this_cache(tmp_path):
    slow, _ = http_backend([ok_reply()], timeout=120.0)
    make_client(tmp_path, slow).verify(**CALL)
    quick, transport = http_backend([(200, completion_body(REFUTED), {})], timeout=5.0)
    row = make_client(tmp_path, quick).verify(**CALL)
    assert len(transport.sent) == 1, "a run with a shorter timeout replayed this cache"
    assert row["verdict"] == "REFUTED"


def test_c06_guard_output_settings_stay_in_the_cache_identity(tmp_path):
    """Passes on callayer-v3. Guards the key rewrite contract 6 needs."""
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    backend = ScriptedBackend([REFUTED])
    make_client(tmp_path, backend, config=model_config(temperature=1.0)).verify(**CALL)
    assert len(backend.calls) == 1


def test_c06_guard_an_identical_run_still_replays(tmp_path):
    """Passes on callayer-v3. Blocks the cheap fix of putting something volatile in
    the key: the same configuration and the same backend must still hit the cache."""
    make_client(tmp_path, ScriptedBackend([VALID])).verify(**CALL)
    backend = ScriptedBackend([])
    row = make_client(tmp_path, backend, log_name="second.jsonl").verify(**CALL)
    assert backend.calls == []
    assert row["cached"] is True


# --------------------------------------------------------------------------- #
# Contract 7 (minor, coverage): the no-leak payload check over M1, the only
# condition file whose manipulation_meta carries the gold indices.
# --------------------------------------------------------------------------- #
WIRE_FIELDS = {"model", "messages", "temperature", "max_tokens", "thinking"}
# Checked against the user message. Bare "condition" is excluded on purpose: it
# occurs as ordinary prose in 9 M1 abstracts, so only its JSON-key form is a leak.
META_TOKENS = (
    "manipulation_meta", "removed_idx", "expected_verdict", "gold_idx",
    '"idx"', "NOT_ENOUGH_INFO", "SUPPORTED", "REFUTED",
)
# Checked against the whole request body, where a bare word cannot collide.
META_KEYS = ('"condition"', '"manipulation_meta"', '"claim_id"', '"sample_idx"')


def m1_rows() -> list[dict[str, Any]]:
    if not M1_PATH.is_file():
        pytest.skip(f"{M1_PATH} not built")
    return [
        json.loads(line)
        for line in M1_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_c07_guard_m1_rows_send_nothing_but_claim_and_evidence_text(tmp_path):
    """Passes on callayer-v3. Every M1 row carries removed_idx (its gold indices) and
    expected_verdict next to the verifier-facing fields; this pins that none of it,
    and nothing else, reaches the request body. The expected user message is rebuilt
    here from the row rather than through src/prompts.py, so a change to rendering
    or to escaping shows up as a failure rather than agreeing with itself."""
    rows = m1_rows()
    backend, transport = http_backend([ok_reply()] * len(rows))
    client = vc.VerifierClient(
        model_config(), backend, load_prompts(PROMPTS), tmp_path / "m1.jsonl", cache_dir=None
    )
    for row in rows:
        client.verify(
            claim_id=row["id"],
            claim=row["claim"],
            evidence_sentences=row["evidence_sentences"],
            sample_idx=0,
            condition=row["condition"],
        )
    assert len(transport.sent) == len(rows)
    for row, request in zip(rows, transport.sent):
        raw = request.body.decode("utf-8")
        payload = json.loads(raw)
        assert set(payload) == WIRE_FIELDS, f"extra wire field for {row['id']}"
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        evidence = " ".join(s["text"].strip() for s in row["evidence_sentences"])
        user_message = payload["messages"][1]["content"]
        assert user_message == (
            f"<claim>\n{row['claim']}\n</claim>\n\n<evidence>\n{evidence}\n</evidence>\n"
        ), f"user message is not exactly claim + evidence text for {row['id']}"
        assert row["id"] not in raw, f"opaque id leaked for {row['id']}"
        for token in META_TOKENS:
            assert token not in user_message, f"{token} reached the user message for {row['id']}"
        for token in META_KEYS:
            assert token not in raw, f"{token} reached the request body for {row['id']}"
        for index in row["manipulation_meta"]["removed_idx"]:
            assert f'"idx": {index}' not in raw
            assert f'"idx":{index}' not in raw


def test_c07_guard_verify_refuses_a_whole_m1_row(tmp_path):
    """Passes on callayer-v3. The keyword-only signature is the structural guard that
    stops a runner from splatting a condition row, meta included, into the call."""
    row = m1_rows()[0]
    client = vc.VerifierClient(
        model_config(), ScriptedBackend([VALID]), load_prompts(PROMPTS),
        tmp_path / "m1.jsonl", cache_dir=None,
    )
    with pytest.raises(TypeError):
        client.verify(**row)


# --------------------------------------------------------------------------- #
# Contract 8 (minor): wrapper-tag escaping, checked against a regex written here
# rather than the one in src/escaping.py, so the test cannot agree with a wrong
# definition. Break caught: separator characters the module's own pattern does
# not consider part of a tag.
# --------------------------------------------------------------------------- #
# Contract: anything a reader would take for a wrapper tag - the bracket, then
# any run of whitespace, slashes, backslashes or zero-width characters, then the
# tag name - must be escaped. To narrow the contract, delete rows from the
# parametrize list below rather than widening this pattern.
ZERO_WIDTH = "​‌‍﻿"  # ZWSP, ZWNJ, ZWJ, BOM
OWN_WRAPPER_TAG = re.compile(
    "<[\\s/\\\\" + ZERO_WIDTH + "]*(?:claim|evidence)\\b[^>]*>", re.IGNORECASE
)


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
        pytest.param("a </evidence > b", id="nbsp-before-gt"),
        pytest.param("a <//evidence> b", id="double-slash"),
        pytest.param("a <​/evidence> b", id="zero-width-after-lt"),
        pytest.param("a </​evidence> b", id="zero-width-after-slash"),
        pytest.param("a <\\evidence> b", id="backslash"),
        pytest.param("a </\\evidence> b", id="backslash-after-slash"),
    ],
)
def test_c08_every_wrapper_tag_variant_is_escaped_and_logged(text):
    from escaping import escape_tags

    escaped, logged = escape_tags(text)
    assert OWN_WRAPPER_TAG.search(escaped) is None, escaped
    assert logged, "escape not reported, so the audit trail misses it"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a 5 < 7 and 9 > 3 b", id="bare-comparison"),
        pytest.param("a <br> b", id="unrelated-tag"),
        pytest.param("a <evidential> b", id="prefix-of-the-name"),
        pytest.param("claim and evidence are words", id="plain-words"),
    ],
)
def test_c08_guard_text_that_is_not_a_wrapper_tag_is_left_alone(text):
    """Passes on callayer-v3. Blocks the cheap fix of escaping every angle bracket,
    which would silently rewrite the dataset's own prose."""
    from escaping import escape_tags

    escaped, logged = escape_tags(text)
    assert escaped == text
    assert logged == []


def test_c08_guard_no_condition_file_needs_an_escape():
    """Passes on callayer-v3 and records why contract 8 is cheap to get wrong: the
    escape path never fires on today's data, so only a test can protect it."""
    directory = ROOT / "data" / "conditions"
    # Only the condition files; skips.jsonl and m1_overlap_screen.jsonl have
    # their own schema and carry no verifier-facing evidence.
    paths = [directory / f"{code}.jsonl" for code in
             ("C0", "C0p", "C1", "C2", "M1", "M2", "C_closed")]
    if not all(path.is_file() for path in paths):
        pytest.skip(f"{directory} not built")
    offenders = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            texts = [row["claim"]] + [s["text"] for s in row["evidence_sentences"]]
            if any(OWN_WRAPPER_TAG.search(text) for text in texts):
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], offenders


if __name__ == "__main__":  # pragma: no cover - convenience
    sys.exit(pytest.main([__file__, "-q"]))
