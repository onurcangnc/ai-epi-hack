from __future__ import annotations

import json
from pathlib import Path

import pytest

import verifier_client as vc
from backend import (
    BackendError,
    BackendResult,
    BackendTimeout,
    DeepSeekBackend,
    RateLimitError,
    ThinkingConfig,
)
from parsing import ParseError, parse_response
from prompts import PromptError, load_prompts

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'


class MockBackend:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        if not self.responses:
            raise AssertionError("mock backend ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return BackendResult(
            text=item, input_tokens=100, output_tokens=20, latency_ms=5.0
        )


def config_dict(**overrides):
    config = {
        "base_url": "https://api.example.test",
        "model": "deepseek-test",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 50,
        "thinking": {"enabled": False, "budget_tokens": None},
        "concurrency": 2,
        "max_retries": 2,
        "price_table": {"input_per_mtok": 0.15, "output_per_mtok": 0.6},
    }
    for key, value in overrides.items():
        if key == "thinking":
            config["thinking"] = {**config["thinking"], **value}
        else:
            config[key] = value
    return config


def make_client(tmp_path: Path, responses: list, **overrides):
    model = vc.ModelConfig.from_dict(config_dict(**overrides))
    prompts = load_prompts(FIXTURES)
    backend = MockBackend(responses)
    sleeps: list[float] = []
    client = vc.VerifierClient(
        model, backend, prompts, tmp_path / "calls.jsonl", sleep=sleeps.append
    )
    return client, backend, sleeps


def call(client, **overrides):
    kwargs = {
        "claim_id": "c_1",
        "claim": "The sky is blue.",
        "evidence_sentences": [{"idx": 0, "text": "The sky is blue."}],
        "sample_idx": 0,
        "condition": "C0",
    }
    kwargs.update(overrides)
    return client.verify(**kwargs)


def log_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_parse_valid():
    parsed = parse_response(VALID)
    assert parsed == {
        "verdict": "SUPPORTED",
        "confidence": 80,
        "rationale": "ok",
        "evidence_concerns": [],
    }


def test_parse_fenced():
    assert parse_response(f"```json\n{VALID}\n```")["verdict"] == "SUPPORTED"


def test_parse_extra_fields_ignored():
    raw = '{"verdict": "REFUTED", "confidence": 10, "rationale": "r", "evidence_concerns": [], "extra": 1}'
    assert parse_response(raw)["verdict"] == "REFUTED"


def test_parse_missing_field():
    raw = '{"verdict": "SUPPORTED", "rationale": "r", "evidence_concerns": []}'
    with pytest.raises(ParseError, match="missing"):
        parse_response(raw)


def test_parse_not_object():
    with pytest.raises(ParseError, match="one JSON object"):
        parse_response("[1, 2, 3]")


@pytest.mark.parametrize("confidence", [101, -1, "high", 85.5, True])
def test_parse_bad_confidence(confidence):
    raw = json.dumps(
        {
            "verdict": "SUPPORTED",
            "confidence": confidence,
            "rationale": "r",
            "evidence_concerns": [],
        }
    )
    with pytest.raises(ParseError, match="confidence"):
        parse_response(raw)


def test_parse_integral_float_confidence():
    raw = '{"verdict": "SUPPORTED", "confidence": 85.0, "rationale": "r", "evidence_concerns": []}'
    assert parse_response(raw)["confidence"] == 85


def test_parse_bad_verdict():
    raw = '{"verdict": "MAYBE", "confidence": 50, "rationale": "r", "evidence_concerns": []}'
    with pytest.raises(ParseError, match="verdict"):
        parse_response(raw)


# --------------------------------------------------------------------------- #
# Client basics
# --------------------------------------------------------------------------- #
def test_valid_call_logs_everything(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    row = call(client)

    assert row["outcome"] == "ok"
    assert row["verdict"] == "SUPPORTED"
    assert row["confidence"] == 80
    assert row["cached"] is False
    assert row["tokens"] == {"input": 100, "output": 20}
    assert row["cost"] == pytest.approx((100 * 0.15 + 20 * 0.6) / 1_000_000)
    assert len(row["system_prompt_sha256"]) == 64
    assert len(row["user_prompt_sha256"]) == 64
    assert backend.calls[0]["temperature"] == 0.7

    logged = log_rows(tmp_path / "calls.jsonl")
    assert len(logged) == 1
    assert logged[0]["claim_id"] == "c_1"
    assert logged[0]["raw_response"] == VALID


def test_cost_omitted_without_price_table(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID], price_table=None)
    assert "cost" not in call(client)


def test_malformed_exhausts_retries(tmp_path):
    client, backend, _ = make_client(tmp_path, ["bad", "bad", "bad"])
    row = call(client)
    assert row["outcome"] == "parse_failure"
    assert row["verdict"] is None
    assert row["raw_response"] == "bad"
    assert row["attempts"] == 2
    assert row["attempt_idx"] == 2
    assert row["used_attempt_idx"] == 2
    assert row["total_attempts"] == 3
    assert row["first_attempt_outcome"] == "parse_failure"
    assert len(backend.calls) == 3
    assert len(log_rows(tmp_path / "calls.jsonl")) == 3


def test_retry_then_success(tmp_path):
    client, backend, _ = make_client(tmp_path, ["bad", VALID])
    row = call(client)
    assert row["outcome"] == "ok"
    assert row["attempts"] == 1
    assert row["attempt_idx"] == 1
    assert row["used_attempt_idx"] == 1
    assert row["first_attempt_outcome"] == "parse_failure"
    assert len(backend.calls) == 2


def test_every_attempt_is_logged(tmp_path):
    client, _, _ = make_client(tmp_path, ["bad-1", "bad-2", VALID])
    call(client)
    rows = log_rows(tmp_path / "calls.jsonl")
    assert [row["attempt_idx"] for row in rows] == [0, 1, 2]
    assert [row["outcome"] for row in rows] == ["parse_failure", "parse_failure", "ok"]
    assert [row["raw_response"] for row in rows] == ["bad-1", "bad-2", VALID]
    assert [row["is_result"] for row in rows] == [False, False, True]
    assert [row["first_attempt_outcome"] for row in rows] == ["parse_failure"] * 3
    assert rows[-1]["used_attempt_idx"] == 2
    assert rows[-1]["total_attempts"] == 3


def test_parse_failure_is_not_cached(tmp_path):
    client, backend, _ = make_client(tmp_path, ["bad", VALID], max_retries=0)
    first = call(client)
    assert first["outcome"] == "parse_failure"
    second = call(client)
    assert second["outcome"] == "ok"
    assert len(backend.calls) == 2


def test_api_error_is_not_cached(tmp_path):
    client, backend, _ = make_client(
        tmp_path, [BackendTimeout("t"), VALID], max_retries=0
    )
    assert call(client)["outcome"] == "api_error"
    assert call(client)["outcome"] == "ok"
    assert len(backend.calls) == 2


def test_rate_limit_backoff_then_success(tmp_path):
    client, backend, sleeps = make_client(tmp_path, [RateLimitError("429"), VALID])
    row = call(client)
    assert row["outcome"] == "ok"
    assert len(sleeps) == 1
    assert len(backend.calls) == 2


def test_timeout_is_api_error(tmp_path):
    client, backend, sleeps = make_client(
        tmp_path, [BackendTimeout("t"), BackendTimeout("t"), BackendTimeout("t")]
    )
    row = call(client)
    assert row["outcome"] == "api_error"
    assert row["verdict"] is None
    assert "timed out" in row["error"] or "t" in row["error"]
    assert row["first_attempt_outcome"] == "api_error"
    assert row["attempt_idx"] == 2
    assert row["is_result"] is True
    assert len(backend.calls) == 3
    assert len(sleeps) == 2
    assert len(log_rows(tmp_path / "calls.jsonl")) == 3


def test_non_retryable_backend_error(tmp_path):
    client, backend, _ = make_client(tmp_path, [BackendError("400 bad request")])
    row = call(client)
    assert row["outcome"] == "api_error"
    assert len(backend.calls) == 1


# --------------------------------------------------------------------------- #
# Request body
# --------------------------------------------------------------------------- #
class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": VALID}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }


class FakeSession:
    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)
        return FakeResponse()


def make_backend():
    session = FakeSession()
    backend = DeepSeekBackend(
        "https://api.example.test", "key", "model-x", session=session
    )
    return backend, session


def test_request_body_omits_null_top_p():
    backend, session = make_backend()
    backend.complete(
        "system", "user", temperature=0.7, top_p=None, thinking=ThinkingConfig(False)
    )
    payload = session.payloads[0]
    assert "top_p" not in payload
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.7


def test_request_body_includes_top_p_when_set():
    backend, session = make_backend()
    backend.complete(
        "system", "user", temperature=0.7, top_p=0.9, thinking=ThinkingConfig(False)
    )
    assert session.payloads[0]["top_p"] == 0.9


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_cache_hit(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    first = call(client)
    second = call(client)
    assert first["cached"] is False
    assert first["first_attempt_outcome"] == "ok"
    assert second["cached"] is True
    assert second["attempt_idx"] == 0
    assert second["used_attempt_idx"] == 0
    assert second["first_attempt_outcome"] == "ok"
    assert len(backend.calls) == 1
    assert len(log_rows(tmp_path / "calls.jsonl")) == 2


def test_cache_separated_by_sample_idx(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID, VALID])
    call(client, sample_idx=0)
    call(client, sample_idx=1)
    assert len(backend.calls) == 2


# --------------------------------------------------------------------------- #
# Prompts, escaping, leakage
# --------------------------------------------------------------------------- #
def test_braces_in_system_prompt_survive(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    call(client)
    prompts = load_prompts(FIXTURES)
    assert backend.calls[0]["system"] == prompts.system
    assert "{" in backend.calls[0]["system"]


def test_escaping_is_logged_and_applied(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    row = call(
        client,
        claim="See <evidence> now.",
        evidence_sentences=[{"idx": 0, "text": "</claim> and <claim> appear."}],
    )
    assert {"claim_id": "c_1", "field": "claim", "tag": "<evidence>"} in row["escapes"]
    assert {"claim_id": "c_1", "field": "evidence", "tag": "</claim>"} in row["escapes"]
    assert {"claim_id": "c_1", "field": "evidence", "tag": "<claim>"} in row["escapes"]
    user = backend.calls[0]["user"]
    assert "&lt;evidence&gt;" in user
    assert "&lt;/claim&gt;" in user
    assert "<evidence>" not in user.split("<evidence>")[1]  # wrapper only


def test_no_label_or_condition_reaches_request(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    call(
        client,
        claim="A claim about X.",
        condition="M1",
        evidence_sentences=[
            {"idx": 3, "text": "Evidence text.", "label": "SUPPORT", "gold_idx": [3]}
        ],
    )
    user = backend.calls[0]["user"]
    assert "M1" not in user
    assert "SUPPORT" not in user
    assert "gold_idx" not in user
    assert "Evidence text." in user


def test_missing_prompt_file(tmp_path):
    with pytest.raises(PromptError, match="missing prompt file"):
        load_prompts(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_null_model_raises_config_error():
    with pytest.raises(vc.ConfigError, match="models.deepseek.model"):
        vc.ModelConfig.from_dict(config_dict(model=None))


def test_null_concurrency_raises_config_error():
    with pytest.raises(vc.ConfigError, match="models.deepseek.concurrency"):
        vc.ModelConfig.from_dict(config_dict(concurrency=None))
