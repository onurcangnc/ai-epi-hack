from __future__ import annotations

import json
import random
import subprocess
import sys
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
from prompts import PromptError, load_prompts, render_evidence

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'


class MockBackend:
    """Responses may be a str, an Exception, or a (text, finish_reason) tuple."""

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
        if isinstance(item, tuple):
            text, finish_reason = item
        else:
            text, finish_reason = item, "stop"
        return BackendResult(
            text=text,
            input_tokens=100,
            output_tokens=20,
            latency_ms=5.0,
            finish_reason=finish_reason,
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
        "retry_on": ["api_error"],
        "price_table": {"input_per_mtok": 0.15, "output_per_mtok": 0.6},
    }
    for key, value in overrides.items():
        if key == "thinking":
            config["thinking"] = {**config["thinking"], **value}
        else:
            config[key] = value
    return config


def make_client(tmp_path: Path, responses: list, *, use_cache: bool = True, **overrides):
    model = vc.ModelConfig.from_dict(config_dict(**overrides))
    prompts = load_prompts(FIXTURES)
    backend = MockBackend(responses)
    sleeps: list[float] = []
    client = vc.VerifierClient(
        model,
        backend,
        prompts,
        tmp_path / "calls.jsonl",
        cache_dir=(tmp_path / "cache") if use_cache else None,
        sleep=sleeps.append,
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
# Client basics and retry policy
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
    assert row["finish_reason"] == "stop"
    assert row["finish_reason_last"] == "stop"
    assert row["order_seed"] is None
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


def test_parse_failure_is_not_retried(tmp_path):
    """SPEC v1.1: a parse failure is the result, never retried in the main run."""
    client, backend, _ = make_client(tmp_path, ["bad", VALID])
    row = call(client)
    assert row["outcome"] == "parse_failure"
    assert row["verdict"] is None
    assert row["raw_response"] == "bad"
    assert row["attempts"] == 0
    assert row["attempt_idx"] == 0
    assert row["used_attempt_idx"] == 0
    assert row["total_attempts"] == 1
    assert row["first_attempt_outcome"] == "parse_failure"
    assert row["is_result"] is True
    assert len(backend.calls) == 1
    assert len(log_rows(tmp_path / "calls.jsonl")) == 1


def test_retry_on_api_error_then_success(tmp_path):
    client, backend, sleeps = make_client(tmp_path, [BackendTimeout("t"), VALID])
    row = call(client)
    assert row["outcome"] == "ok"
    assert row["attempts"] == 1
    assert row["attempt_idx"] == 1
    assert row["used_attempt_idx"] == 1
    assert row["first_attempt_outcome"] == "api_error"
    assert len(backend.calls) == 2
    assert len(sleeps) == 1


def test_every_api_attempt_is_logged(tmp_path):
    client, _, _ = make_client(
        tmp_path, [BackendTimeout("t0"), BackendTimeout("t1"), VALID]
    )
    call(client)
    rows = log_rows(tmp_path / "calls.jsonl")
    assert [row["attempt_idx"] for row in rows] == [0, 1, 2]
    assert [row["outcome"] for row in rows] == ["api_error", "api_error", "ok"]
    assert [row["raw_response"] for row in rows] == [None, None, VALID]
    assert [row["is_result"] for row in rows] == [False, False, True]
    assert [row["first_attempt_outcome"] for row in rows] == ["api_error"] * 3
    assert [row["finish_reason"] for row in rows] == [None, None, "stop"]
    assert rows[-1]["used_attempt_idx"] == 2
    assert rows[-1]["total_attempts"] == 3


def test_parse_failure_is_not_cached(tmp_path):
    client, backend, _ = make_client(tmp_path, ["bad", VALID], max_retries=0)
    first = call(client)
    assert first["outcome"] == "parse_failure"
    second = call(client)
    assert second["outcome"] == "ok"
    assert second["cached"] is False
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
    assert row["is_result"] is True
    assert len(backend.calls) == 1


def test_retry_policy_can_retry_parse_failure(tmp_path):
    client, backend, _ = make_client(
        tmp_path, ["bad", VALID], retry_on=["api_error", "parse_failure"]
    )
    row = call(client)
    assert row["outcome"] == "ok"
    assert row["attempts"] == 1
    assert row["first_attempt_outcome"] == "parse_failure"
    assert len(backend.calls) == 2


def test_retry_policy_parse_failure_exhausts(tmp_path):
    client, backend, _ = make_client(
        tmp_path,
        ["bad", "bad"],
        retry_on=["api_error", "parse_failure"],
        max_retries=1,
    )
    row = call(client)
    assert row["outcome"] == "parse_failure"
    assert row["attempts"] == 1
    assert row["total_attempts"] == 2
    assert len(backend.calls) == 2


def test_empty_retry_on_rejected():
    with pytest.raises(vc.ConfigError, match="retry_on"):
        vc.ModelConfig.from_dict(config_dict(retry_on=[]))


def test_unknown_retry_on_rejected():
    with pytest.raises(vc.ConfigError, match="unknown entries"):
        vc.ModelConfig.from_dict(config_dict(retry_on=["timeout"]))


# --------------------------------------------------------------------------- #
# finish_reason
# --------------------------------------------------------------------------- #
def test_length_finish_reason_visible_on_result(tmp_path):
    client, _, _ = make_client(tmp_path, [(VALID, "length")])
    row = call(client)
    assert row["outcome"] == "ok"
    assert row["finish_reason"] == "length"
    assert row["finish_reason_last"] == "length"


def test_absent_finish_reason_is_null(tmp_path):
    client, _, _ = make_client(tmp_path, [(VALID, None)])
    row = call(client)
    assert row["finish_reason"] is None
    assert row["finish_reason_last"] is None


def test_cached_row_keeps_finish_reason(tmp_path):
    client, backend, _ = make_client(tmp_path, [(VALID, "length")])
    call(client)
    cached = call(client)
    assert cached["cached"] is True
    assert cached["finish_reason_last"] == "length"
    assert len(backend.calls) == 1


class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": VALID}, "finish_reason": "length"}],
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


def test_backend_extracts_finish_reason():
    backend, _ = make_backend()
    result = backend.complete("system", "user", temperature=0.7)
    assert result.finish_reason == "length"


# --------------------------------------------------------------------------- #
# Request body
# --------------------------------------------------------------------------- #
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


def test_cache_is_written_to_disk(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    call(client)
    entries = list((tmp_path / "cache").glob("*.json"))
    assert len(entries) == 1
    payload = json.loads(entries[0].read_text(encoding="utf-8"))
    assert payload["raw"] == VALID
    assert len(payload["key"]) == 64


def test_cache_persists_across_clients(tmp_path):
    first_client, _, _ = make_client(tmp_path, [VALID])
    call(first_client)

    second_client, second_backend, _ = make_client(tmp_path, [])
    row = call(second_client)
    assert row["cached"] is True
    assert row["outcome"] == "ok"
    assert second_backend.calls == []


def test_second_process_hits_cache(tmp_path):
    first_client, _, _ = make_client(tmp_path, [VALID])
    call(first_client)

    script = tmp_path / "second_process.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "root, cache, fixtures, log = map(Path, sys.argv[1:5])\n"
        "sys.path.insert(0, str(root / 'src'))\n"
        "sys.path.insert(0, str(root / 'scripts'))\n"
        "from verifier_client import ModelConfig, VerifierClient\n"
        "from prompts import load_prompts\n"
        "class Dead:\n"
        "    def complete(self, *a, **k):\n"
        "        raise AssertionError('backend called: cache miss')\n"
        "cfg = ModelConfig.from_dict({\n"
        "    'base_url': 'https://api.example.test',\n"
        "    'model': 'deepseek-test',\n"
        "    'api_key_env': 'DEEPSEEK_API_KEY',\n"
        "    'temperature': 0.7,\n"
        "    'top_p': None,\n"
        "    'max_tokens': 50,\n"
        "    'thinking': {'enabled': False, 'budget_tokens': None},\n"
        "    'concurrency': 1,\n"
        "    'max_retries': 2,\n"
        "    'retry_on': ['api_error'],\n"
        "    'price_table': None,\n"
        "})\n"
        "client = VerifierClient(cfg, Dead(), load_prompts(fixtures), log, cache_dir=cache)\n"
        "row = client.verify(\n"
        "    claim_id='c_1',\n"
        "    claim='The sky is blue.',\n"
        "    evidence_sentences=[{'idx': 0, 'text': 'The sky is blue.'}],\n"
        "    sample_idx=0,\n"
        "    condition='C0',\n"
        ")\n"
        "print(json.dumps({'cached': row['cached'], 'outcome': row['outcome'], 'verdict': row['verdict']}))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(ROOT),
            str(tmp_path / "cache"),
            str(FIXTURES),
            str(tmp_path / "second.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {"cached": True, "outcome": "ok", "verdict": "SUPPORTED"}


# --------------------------------------------------------------------------- #
# Batch ordering
# --------------------------------------------------------------------------- #
def make_items(count: int) -> list[dict]:
    return [
        {
            "claim_id": f"c_{index}",
            "claim": f"Claim {index}.",
            "evidence_sentences": [{"idx": 0, "text": f"Evidence {index}."}],
            "sample_idx": index,
            "condition": "C0" if index % 2 else "C1",
        }
        for index in range(count)
    ]


def test_run_batch_logs_order_seed_in_every_row(tmp_path):
    client, _, _ = make_client(
        tmp_path, [VALID] * 6, concurrency=1, retry_on=["parse_failure"]
    )
    client.run_batch(make_items(6), order_seed=11)
    rows = log_rows(tmp_path / "calls.jsonl")
    assert len(rows) == 6
    assert [row["order_seed"] for row in rows] == [11] * 6


def test_run_batch_shuffle_matches_seed(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID] * 6, concurrency=1)
    items = make_items(6)
    expected = [item["claim_id"] for item in items]
    random.Random(11).shuffle(expected)

    client.run_batch(items, order_seed=11)
    logged = [row["claim_id"] for row in log_rows(tmp_path / "calls.jsonl")]
    assert logged == expected


def test_run_batch_is_reproducible_and_seeded(tmp_path):
    orders = []
    for seed in (11, 12):
        directory = tmp_path / str(seed)
        directory.mkdir()
        client, _, _ = make_client(directory, [VALID] * 6, concurrency=1)
        client.run_batch(make_items(6), order_seed=seed)
        orders.append(
            [row["claim_id"] for row in log_rows(directory / "calls.jsonl")]
        )
    assert orders[0] != orders[1]


# --------------------------------------------------------------------------- #
# Render normalization
# --------------------------------------------------------------------------- #
def test_evidence_is_stripped_and_joined_on_the_client_path(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID])
    call(
        client,
        evidence_sentences=[
            {"idx": 0, "text": "First sentence.\n\n  "},
            {"idx": 1, "text": "   Second sentence.\t"},
        ],
    )
    user = backend.calls[0]["user"]
    assert "First sentence. Second sentence." in user
    assert "First sentence.\n" not in user
    assert "  Second" not in user


def test_render_evidence_strips_newlines_and_trailing_spaces():
    sentences = [
        {"idx": 0, "text": "  Alpha beta.\n"},
        {"idx": 1, "text": "Gamma delta.   "},
        {"idx": 2, "text": "\nEpsilon zeta.\n\n"},
    ]
    assert render_evidence(sentences) == "Alpha beta. Gamma delta. Epsilon zeta."


def test_render_evidence_empty_list():
    assert render_evidence([]) == ""


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


def test_null_retry_on_raises_config_error():
    with pytest.raises(vc.ConfigError, match="models.deepseek.retry_on"):
        vc.ModelConfig.from_dict(config_dict(retry_on=None))


def test_load_cache_dir_resolves_relative(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cache:\n  dir: cache\n", encoding="utf-8")
    assert vc.load_cache_dir(config_path) == tmp_path / "cache"


def test_load_cache_dir_missing_key(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(vc.ConfigError, match="cache"):
        vc.load_cache_dir(config_path)
