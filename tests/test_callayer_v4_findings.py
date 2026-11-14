"""Tests that pin the absent-text result-row contract at callayer-v4 (2026-11-15).

callayer-v4 (commit 8575e36, "B5") made an absent-text reply become a result row
on the last resume pass and kept its raw body. It did that with

    is_result=(is_last_pass and resume_pass > 0)

so the result row depends on a resume pass having happened rather than on being
the last attempt. With resume_passes=0 the only pass is both the first and the
last, the conjunction is False, and the unit ends with no result row at all -
it leaves the k=5 denominator silently, against SPEC section 6.

Tests named ``test_cNN_*`` state the contract and fail on callayer-v4 wherever
it is not met. Tests with ``guard`` in their name pass on callayer-v4 and
protect behavior the fix must keep: a bare verify() still logs a non-result row
(what tests/test_callayer_v2_findings.py requires), and a provider cut with
finish_reason "length" stays a truncated result row that is not retried.

Read-only with respect to src/, SPEC.md and every other test file.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter

import verifier_client as vc
from backend import DeepSeekBackend
from prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"

# Every run_batch setting the contract covers.
PASSES = [pytest.param(0, id="resume_passes=0"),
          pytest.param(1, id="resume_passes=1"),
          pytest.param(2, id="resume_passes=2")]

# A reply the provider sent with no text at all, and the marker that proves the
# raw body survived into the row.
ABSENT_TEXT = [
    pytest.param(None, "null", id="null-content"),
    pytest.param(
        [{"type": "image_url", "image_url": {"url": "x"}}], "image_url", id="non-text-parts"
    ),
]

# A provider cut: finish_reason "length", with and without text.
LENGTH_CUT = [
    pytest.param('{"verdict": "REFUT', '{"verdict": "REFUT', id="partial-json"),
    pytest.param(None, "null", id="null-content"),
]

UNIT_COUNT = 3


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
        "blocked_body_markers": ["content exists risk"],
    }
    section.update(overrides)
    return vc.ModelConfig.from_dict(section)


def completion_body(content: Any, finish_reason: str | None = "stop") -> str:
    """An OpenAI-format chat completion shaped like DeepSeek's."""
    return json.dumps(
        {
            "id": "chatcmpl-v4",
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
            "system_fingerprint": "fp_v4",
        }
    )


class AlwaysTransport(BaseAdapter):
    """Answers every request with one canned reply, recording the unit it was for.

    Unlike a queued transport this never runs out, so the number of requests is
    decided by the client rather than by the test.
    """

    def __init__(self, content: Any, finish_reason: str | None = "stop") -> None:
        super().__init__()
        self.payload = completion_body(content, finish_reason)
        self.claims: list[str] = []

    def send(self, request, **kwargs):  # noqa: ANN001 - requests adapter signature
        sent = json.loads(request.body.decode("utf-8"))
        user = sent["messages"][1]["content"]
        self.claims.append(user.split("<claim>\n", 1)[1].split("\n</claim>", 1)[0])
        response = requests.Response()
        response.status_code = 200
        response._content = self.payload.encode("utf-8")
        response.headers.update({"Content-Type": "application/json"})
        response.encoding = "utf-8"
        response.url = request.url
        response.request = request
        return response

    def close(self) -> None:
        pass


def make_client(tmp_path: Path, transport: AlwaysTransport) -> vc.VerifierClient:
    session = requests.Session()
    session.trust_env = False  # no proxy or registry lookups from the test machine
    session.mount("https://", transport)
    backend = DeepSeekBackend(
        "https://api.example.test", "sk-test-key", "deepseek-test", session=session
    )
    return vc.VerifierClient(
        model_config(),
        backend,
        load_prompts(PROMPTS),
        tmp_path / "calls.jsonl",
        cache_dir=tmp_path / "cache",
        sleep=lambda seconds: None,
    )


def units(count: int = UNIT_COUNT) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": f"u{index}",
            "claim": f"Claim u{index}.",
            "evidence_sentences": [{"idx": 0, "text": f"Sentence for unit {index}."}],
            "sample_idx": 0,
            "condition": "C0",
        }
        for index in range(count)
    ]


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rows_for(log: list[dict[str, Any]], claim_id: str) -> list[dict[str, Any]]:
    return [row for row in log if row["claim_id"] == claim_id]


def paid_calls(transport: AlwaysTransport, item: dict[str, Any]) -> int:
    return Counter(transport.claims)[item["claim"]]


# --------------------------------------------------------------------------- #
# Contract 1: through run_batch, an absent-text reply ends with exactly one
# result row, whatever the pass budget. Break caught: a result row that depends
# on resume_pass > 0, so a zero-pass run leaves the unit out of the denominator.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
@pytest.mark.parametrize("resume_passes", PASSES)
def test_c01_absent_text_ends_with_one_result_row(
    tmp_path, content, raw_marker, resume_passes
):
    """Contract: for resume_passes 0, 1 and 2 alike, every unit ends with exactly
    one result row, produced by the last pass, carrying outcome "api_error" and the
    raw body the provider actually sent. SPEC section 6 needs that row so the
    failure stays among the k=5 samples of its cell."""
    transport = AlwaysTransport(content, "stop")
    items = units()
    make_client(tmp_path, transport).run_batch(
        items, order_seed=7, resume_passes=resume_passes
    )
    log = read_log(tmp_path / "calls.jsonl")
    for item in items:
        mine = rows_for(log, item["claim_id"])
        results = [row for row in mine if row["is_result"]]
        assert len(results) == 1, (
            f"{item['claim_id']} ended with {len(results)} result rows out of "
            f"{len(mine)} logged; the unit leaves the k=5 denominator"
        )
        row = results[0]
        assert row["outcome"] == "api_error", f"{item['claim_id']}: {row['outcome']!r}"
        assert row["verdict"] is None
        assert isinstance(row["raw_response"], str) and raw_marker in row["raw_response"], (
            f"{item['claim_id']}: the paid reply was dropped instead of recorded: "
            f"{row['raw_response']!r}"
        )
        assert row["resume_pass"] == resume_passes, (
            f"{item['claim_id']}: the result row came from pass {row['resume_pass']}, "
            f"not the last pass ({resume_passes})"
        )


@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
@pytest.mark.parametrize("resume_passes", PASSES)
def test_c01_rows_before_the_last_pass_are_not_results(
    tmp_path, content, raw_marker, resume_passes
):
    """Contract: only the last pass may mark a result row. Every earlier pass logs
    its attempt with is_result false so a crash mid-run leaves the unit resumable."""
    transport = AlwaysTransport(content, "stop")
    items = units()
    make_client(tmp_path, transport).run_batch(
        items, order_seed=7, resume_passes=resume_passes
    )
    log = read_log(tmp_path / "calls.jsonl")
    early = [row for row in log if row["resume_pass"] < resume_passes]
    assert len(early) == resume_passes * len(items), (
        f"expected {resume_passes} pre-final row(s) per unit, got {len(early)} in total"
    )
    for row in early:
        assert row["is_result"] is False, (
            f"{row['claim_id']} pass {row['resume_pass']} was marked a result before "
            f"the last pass ({resume_passes})"
        )


@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
@pytest.mark.parametrize("resume_passes", PASSES)
def test_c02_absent_text_costs_one_paid_call_per_pass(
    tmp_path, content, raw_marker, resume_passes
):
    """Contract: resume_passes + 1 calls per unit - one first attempt plus one per
    resume pass. More than that means the reply is also being retried inside the
    call, which would multiply the bill for a provider that never returns text."""
    transport = AlwaysTransport(content, "stop")
    items = units()
    make_client(tmp_path, transport).run_batch(
        items, order_seed=7, resume_passes=resume_passes
    )
    per_unit = {item["claim_id"]: paid_calls(transport, item) for item in items}
    assert per_unit == {item["claim_id"]: resume_passes + 1 for item in items}, per_unit
    assert len(transport.claims) == (resume_passes + 1) * len(items)


@pytest.mark.parametrize("content, raw_marker", ABSENT_TEXT)
def test_c02_guard_bare_verify_logs_a_non_result_row(tmp_path, content, raw_marker):
    """Passes on callayer-v4. verify() outside run_batch has exhausted no resume
    pass, so its row stays out of the results - the contract
    tests/test_callayer_v2_findings.py already pins. The fix for contract 1 must
    not turn this into a result row."""
    transport = AlwaysTransport(content, "stop")
    row = make_client(tmp_path, transport).verify(**units(1)[0])
    assert row["outcome"] == "api_error"
    assert row["is_result"] is False
    assert row["verdict"] is None
    assert isinstance(row["raw_response"], str) and raw_marker in row["raw_response"]
    log = read_log(tmp_path / "calls.jsonl")
    assert len(log) == 1
    assert [entry["is_result"] for entry in log] == [False]
    assert len(transport.claims) == 1


# --------------------------------------------------------------------------- #
# Contract 3: a provider cut is a different thing from an absent reply and keeps
# its own outcome. Break caught: a fix for contract 1 that routes every non-ok
# reply through the resume machinery.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content, raw_marker", LENGTH_CUT)
@pytest.mark.parametrize("resume_passes", PASSES)
def test_c03_guard_length_cut_stays_a_truncated_result_row(
    tmp_path, content, raw_marker, resume_passes
):
    """Passes on callayer-v4. Contract: finish_reason "length" is the model's reply
    cut short, so it is a truncated result row on the first attempt at every pass
    budget, and it is not retried - SPEC section 7 retries API errors only."""
    transport = AlwaysTransport(content, "length")
    items = units()
    make_client(tmp_path, transport).run_batch(
        items, order_seed=7, resume_passes=resume_passes
    )
    log = read_log(tmp_path / "calls.jsonl")
    for item in items:
        mine = rows_for(log, item["claim_id"])
        results = [row for row in mine if row["is_result"]]
        assert len(results) == 1, f"{item['claim_id']}: {len(results)} result rows"
        row = results[0]
        assert row["outcome"] == "truncated", f"{item['claim_id']}: {row['outcome']!r}"
        assert row["finish_reason"] == "length"
        assert row["verdict"] is None
        assert isinstance(row["raw_response"], str) and raw_marker in row["raw_response"]
        assert row["resume_pass"] == 0, "a provider cut was resumed"
        assert paid_calls(transport, item) == 1, "a provider cut was retried"


@pytest.mark.parametrize("content, raw_marker", LENGTH_CUT)
def test_c03_guard_length_cut_at_verify_level(tmp_path, content, raw_marker):
    """Passes on callayer-v4. The same cut through verify(): a result row straight
    away, unlike an absent reply."""
    transport = AlwaysTransport(content, "length")
    row = make_client(tmp_path, transport).verify(**units(1)[0])
    assert row["outcome"] == "truncated"
    assert row["is_result"] is True
    assert row["verdict"] is None
    assert raw_marker in row["raw_response"]
    assert len(transport.claims) == 1


if __name__ == "__main__":  # pragma: no cover - convenience
    sys.exit(pytest.main([__file__, "-q"]))
