"""Tests for the programmatic condition builder (SPEC v1.3.1 §5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import build_conditions as bc
import verifier_client as vc
from backend import BackendResult, DeepSeekBackend
from common import expected_verdict, read_jsonl
from prompts import load_prompts, render_evidence

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("conditions")
    summary = bc.build(CONFIG, force=True, out_dir=out)
    return out, summary


def load(out: Path, code: str):
    return read_jsonl(out / f"{code}.jsonl")


def index_by_id(rows):
    return {row["id"]: row for row in rows}


# --------------------------------------------------------------------------- #
# Row counts and identity
# --------------------------------------------------------------------------- #
def test_row_counts(built):
    out, summary = built
    assert summary["rows"] == {
        "C0": 150,
        "C0p": 150,
        "C1": 150,
        "C2": 150,
        "M1": 100,
        "M2": 150,
        "C_closed": 100,
    }
    assert len(read_jsonl(out / "skips.jsonl")) == 50
    assert len(read_jsonl(out / "m1_overlap_screen.jsonl")) == 100


def test_c0p_is_byte_identical_to_c0(built):
    out, _ = built
    assert (out / "C0.jsonl").read_bytes() == (out / "C0p.jsonl").read_bytes()


def test_hashes_cover_every_condition_file(built):
    out, _ = built
    lines = (out / "HASHES.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    names = {line.split("  ", 1)[1] for line in lines}
    assert names == {f"data/conditions/{code}.jsonl" for code in bc.CODES}


# --------------------------------------------------------------------------- #
# M1
# --------------------------------------------------------------------------- #
def test_m1_keeps_non_gold_sentences_in_order(built):
    out, _ = built
    c0 = index_by_id(load(out, "C0"))
    m1 = index_by_id(load(out, "M1"))
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    assert set(m1) == {cid for cid, row in gold.items() if row["label"] != "NEI"}
    for cid, row in m1.items():
        removed = set(gold[cid]["gold_idx"])
        expected = [s for s in c0[cid]["evidence_sentences"] if s["idx"] not in removed]
        assert row["evidence_sentences"] == expected
        assert all(s["idx"] not in removed for s in row["evidence_sentences"])


def test_m1_skips_are_the_nei_claims(built):
    out, _ = built
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    skips = read_jsonl(out / "skips.jsonl")
    assert {row["id"] for row in skips} == {cid for cid, row in gold.items() if row["label"] == "NEI"}


def test_overlap_screen_covers_every_m1_claim(built):
    out, _ = built
    rows = read_jsonl(out / "m1_overlap_screen.jsonl")
    assert {row["id"] for row in rows} == {row["id"] for row in load(out, "M1")}
    for row in rows:
        assert isinstance(row["flagged"], bool)
        assert isinstance(row["matched_idx"], list)


# --------------------------------------------------------------------------- #
# M2
# --------------------------------------------------------------------------- #
def test_m2_has_exactly_one_inserted_instruction(built):
    out, _ = built
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    for row in load(out, "M2"):
        inserted = [s for s in row["evidence_sentences"] if s["idx"] is None]
        assert len(inserted) == 1
        meta = row["manipulation_meta"]
        assert meta["pushed_verdict"] in bc.VERDICTS
        assert meta["pushed_verdict"] != expected_verdict("M2", gold[row["id"]]["label"])
        assert inserted[0]["text"] == (
            json.loads((ROOT / "prompts" / "m2_templates.json").read_text(encoding="utf-8"))[
                meta["family"]
            ][meta["template_index"]].replace("{pushed_verdict}", meta["pushed_verdict"])
        )
        assert meta["instruction_words"] == len(inserted[0]["text"].split())


def test_m2_latin_square_is_balanced_within_one_per_class(built):
    out, _ = built
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    counts: dict[str, dict[tuple[str, str], int]] = {}
    for row in load(out, "M2"):
        label = gold[row["id"]]["label"]
        key = (row["manipulation_meta"]["family"], row["manipulation_meta"]["position"])
        counts.setdefault(label, {}).setdefault(key, 0)
        counts[label][key] += 1
    for label in bc.LABELS:
        assert len(counts[label]) == 9
        values = list(counts[label].values())
        assert max(values) - min(values) <= 1


def test_nei_pushed_verdicts_are_a_seeded_50_50_split(built):
    out, summary = built
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    nei = [
        row["manipulation_meta"]["pushed_verdict"]
        for row in load(out, "M2")
        if gold[row["id"]]["label"] == "NEI"
    ]
    assert nei.count("SUPPORTED") == 25
    assert nei.count("REFUTED") == 25
    assert summary["nei_pushed"] == {"REFUTED": 25, "SUPPORTED": 25}


# --------------------------------------------------------------------------- #
# C1 / C2
# --------------------------------------------------------------------------- #
def _assert_clean_candidate(claim: str, sentence: str) -> None:
    assert not (set(bc.content_words(sentence)) & set(bc.content_words(claim)))


def test_c1_appends_six_clean_sentences_from_other_documents(built):
    out, _ = built
    c0 = index_by_id(load(out, "C0"))
    key = {row["id"]: row for row in read_jsonl(ROOT / "data" / "key.jsonl")}
    for row in load(out, "C1"):
        inserted = [s for s in row["evidence_sentences"] if s["idx"] is None]
        assert len(inserted) == 6
        assert row["manipulation_meta"]["pool_size"] >= 30
        assert len(row["manipulation_meta"]["source_doc_ids"]) == 6
        for sentence in inserted:
            _assert_clean_candidate(row["claim"], sentence["text"])
        assert all(doc != key[row["id"]]["doc_id"] for doc in row["manipulation_meta"]["source_doc_ids"])
        # Original evidence is preserved verbatim.
        assert row["evidence_sentences"][:-6] == c0[row["id"]]["evidence_sentences"]


def test_c2_single_filler_at_m2_position_with_matching_length(built):
    out, _ = built
    m2 = index_by_id(load(out, "M2"))
    key = {row["id"]: row for row in read_jsonl(ROOT / "data" / "key.jsonl")}
    for row in load(out, "C2"):
        inserted = [s for s in row["evidence_sentences"] if s["idx"] is None]
        assert len(inserted) == 1
        meta = row["manipulation_meta"]
        assert abs(meta["filler_words"] - meta["instruction_words"]) <= 2
        assert meta["source_doc_id"] != key[row["id"]]["doc_id"]
        _assert_clean_candidate(row["claim"], inserted[0]["text"])
        # Same insertion position as the claim's M2 instruction.
        assert meta["m2_position"] == m2[row["id"]]["manipulation_meta"]["position"]
        positions = {
            "start": 0,
            "middle": len(row["evidence_sentences"][:-1]) // 2,
            "end": len(row["evidence_sentences"]) - 1,
        }
        inserted_index = next(
            i for i, s in enumerate(row["evidence_sentences"]) if s["idx"] is None
        )
        assert inserted_index == positions[meta["m2_position"]]


def test_c_closed_is_empty_and_directional_only(built):
    out, _ = built
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    rows = load(out, "C_closed")
    assert len(rows) == 100
    assert all(row["evidence_sentences"] == [] for row in rows)
    assert all(gold[row["id"]]["label"] in ("SUPPORT", "REFUTE") for row in rows)


# --------------------------------------------------------------------------- #
# Determinism and overwrite policy
# --------------------------------------------------------------------------- #
def test_rebuild_is_byte_deterministic(built, tmp_path):
    _, _ = built
    other = tmp_path / "conditions2"
    bc.build(CONFIG, force=True, out_dir=other)
    for code in bc.CODES:
        assert (other / f"{code}.jsonl").read_bytes() == (built[0] / f"{code}.jsonl").read_bytes()
    for name in ("skips.jsonl", "m1_overlap_screen.jsonl", "HASHES.txt", "conditions_meta.json"):
        assert (other / name).read_bytes() == (built[0] / name).read_bytes()


def test_overwrite_requires_force(built):
    out, _ = built
    with pytest.raises(bc.DataError, match="refusing to overwrite"):
        bc.build(CONFIG, force=False, out_dir=out)


# --------------------------------------------------------------------------- #
# No leakage into the HTTP payload
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self):
        self.payloads = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)
        return Response()


class Response:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": VALID}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def model_config():
    return vc.ModelConfig.from_dict(
        {
            "base_url": "https://api.example.test",
            "model": "deepseek-test",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.7,
            "top_p": None,
            "max_tokens": 64,
            "thinking": {"enabled": False, "budget_tokens": None},
            "concurrency": 1,
            "max_retries": 2,
            "retry_on": ["api_error"],
            "price_table": None,
        }
    )


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


@pytest.mark.parametrize("code", ["C1", "M2", "C2"])
def test_condition_and_meta_never_reach_the_http_payload(built, tmp_path, code):
    out, _ = built
    row = load(out, code)[0]
    session = Session()
    backend = DeepSeekBackend(
        "https://api.example.test", "sk-test", "deepseek-test", session=session
    )
    client = vc.VerifierClient(
        model_config(), backend, load_prompts(ROOT / "prompts"),
        tmp_path / f"{code}.jsonl", cache_dir=None,
    )
    client.verify(
        claim_id=row["id"],
        claim=row["claim"],
        evidence_sentences=row["evidence_sentences"],
        sample_idx=0,
        condition=row["condition"],
    )
    payload = session.payloads[0]
    # The request body may contain only the wire fields and role/content.
    assert set(_keys(payload)) <= {
        "model", "messages", "role", "content", "temperature", "top_p",
        "max_tokens", "thinking", "type",
    }
    flat = json.dumps(payload, ensure_ascii=False)
    assert row["id"] not in flat
    assert '"condition"' not in flat
    assert '"manipulation_meta"' not in flat
    assert "pushed_verdict" not in flat


def test_render_evidence_accepts_inserted_null_idx():
    assert render_evidence(
        [{"idx": 0, "text": "Alpha."}, {"idx": None, "text": "  Beta.  "}]
    ) == "Alpha. Beta."
