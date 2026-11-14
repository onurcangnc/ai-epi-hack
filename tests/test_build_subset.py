from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import build_subset as bs
from common import DataError, load_config, read_jsonl, sha256_file
from conftest import ROOT, make_claim, write_config, write_jsonl


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def test_load_corpus(tiny_spec):
    corpus = bs.load_corpus(tiny_spec["raw"] / "corpus.jsonl")
    assert len(corpus) == 6
    assert corpus[100]["abstract"] == ["a0", "a1", "a2"]
    assert corpus[105]["abstract"] == []


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_jsonl(
        path,
        [
            {"doc_id": 1, "title": "t", "abstract": ["s"], "structured": False},
            {"doc_id": 1, "title": "t", "abstract": ["s"], "structured": False},
        ],
    )
    with pytest.raises(DataError, match="duplicate"):
        bs.load_corpus(path)


def test_load_corpus_rejects_bad_abstract(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_jsonl(
        path,
        [{"doc_id": 1, "title": "t", "abstract": "not a list", "structured": False}],
    )
    with pytest.raises(DataError, match="abstract"):
        bs.load_corpus(path)


def test_load_claims(tiny_spec):
    claims = bs.load_claims(tiny_spec["raw"] / "claims_train.jsonl", "train")
    assert len(claims) == 7
    assert all(claim["_split"] == "train" for claim in claims)


@pytest.mark.parametrize(
    "bad_claim, match",
    [
        ({"id": "x", "claim": "c", "evidence": {}, "cited_doc_ids": []}, "id"),
        ({"id": 1, "claim": "", "evidence": {}, "cited_doc_ids": []}, "claim"),
        ({"id": 1, "claim": "c", "evidence": [], "cited_doc_ids": []}, "evidence"),
        ({"id": 1, "claim": "c", "evidence": {}, "cited_doc_ids": "7"}, "cited_doc_ids"),
        (
            {
                "id": 1,
                "claim": "c",
                "evidence": {"7": [{"label": "MAYBE", "sentences": [0]}]},
                "cited_doc_ids": [7],
            },
            "label",
        ),
        (
            {
                "id": 1,
                "claim": "c",
                "evidence": {"7": [{"label": "SUPPORT", "sentences": ["0"]}]},
                "cited_doc_ids": [7],
            },
            "sentences",
        ),
    ],
)
def test_load_claims_rejects_invalid(tmp_path, bad_claim, match):
    path = tmp_path / "claims.jsonl"
    write_jsonl(path, [bad_claim])
    with pytest.raises(DataError, match=match):
        bs.load_claims(path, "train")


# --------------------------------------------------------------------------- #
# Labels and documents
# --------------------------------------------------------------------------- #
def test_claim_label():
    assert bs.claim_label(make_claim(1, "train", "SUPPORT", [1])) == "SUPPORT"
    assert bs.claim_label(make_claim(2, "train", "REFUTE", [1])) == "REFUTE"
    assert bs.claim_label(make_claim(3, "train", "NEI", [1])) == "NEI"


def test_claim_label_mixed_raises():
    claim = make_claim(1, "train", "SUPPORT", [1])
    claim["evidence"]["2"] = [{"label": "CONTRADICT", "sentences": [0]}]
    with pytest.raises(DataError, match="mixes"):
        bs.claim_label(claim)


def test_claim_label_unknown_raises():
    claim = make_claim(1, "train", "SUPPORT", [1])
    claim["evidence"]["1"][0]["label"] = "MAYBE"
    with pytest.raises(DataError, match="unknown"):
        bs.claim_label(claim)


def test_associated_doc_id():
    assert bs.associated_doc_id(make_claim(1, "train", "SUPPORT", [7]), "SUPPORT") == 7
    assert bs.associated_doc_id(make_claim(2, "train", "NEI", [9]), "NEI") == 9


def test_associated_doc_id_rejects_multiple():
    with pytest.raises(DataError, match="evidence documents"):
        bs.associated_doc_id(make_claim(1, "train", "SUPPORT", [7, 8]), "SUPPORT")
    with pytest.raises(DataError, match="cited documents"):
        bs.associated_doc_id(make_claim(2, "train", "NEI", [7, 8]), "NEI")


def test_gold_sentence_indices():
    claim = make_claim(1, "train", "SUPPORT", [7], {7: [2, 0]})
    claim["evidence"]["7"].append({"label": "SUPPORT", "sentences": [2, 1]})
    assert bs.gold_sentence_indices(claim, "SUPPORT", 7, abstract_len=3) == [0, 1, 2]
    assert bs.gold_sentence_indices(make_claim(2, "train", "NEI", [7]), "NEI", 7, 3) == []


def test_gold_sentence_indices_out_of_range():
    claim = make_claim(1, "train", "SUPPORT", [7], {7: [3]})
    with pytest.raises(DataError, match="outside abstract"):
        bs.gold_sentence_indices(claim, "SUPPORT", 7, abstract_len=3)


# --------------------------------------------------------------------------- #
# Candidate filtering
# --------------------------------------------------------------------------- #
def test_is_candidate(tiny_spec):
    corpus = bs.load_corpus(tiny_spec["raw"] / "corpus.jsonl")
    claims = {
        claim["id"]: claim
        for claim in bs.load_claims(tiny_spec["raw"] / "claims_train.jsonl", "train")
    }
    assert bs.is_candidate(claims[1], "SUPPORT", corpus)
    assert bs.is_candidate(claims[4], "NEI", corpus)
    assert not bs.is_candidate(claims[6], "SUPPORT", corpus)  # two evidence docs
    assert not bs.is_candidate(claims[8], "REFUTE", corpus)  # doc missing
    assert not bs.is_candidate(
        make_claim(9, "dev", "REFUTE", [105]), "REFUTE", corpus
    )  # empty abstract


def test_count_candidates(tiny_spec):
    corpus = bs.load_corpus(tiny_spec["raw"] / "corpus.jsonl")
    claims = bs.load_claims(tiny_spec["raw"] / "claims_train.jsonl", "train")
    claims += bs.load_claims(tiny_spec["raw"] / "claims_dev.jsonl", "dev")
    assert bs.count_candidates(claims, corpus) == {"SUPPORT": 3, "REFUTE": 1, "NEI": 2}


# --------------------------------------------------------------------------- #
# Sampling and IDs
# --------------------------------------------------------------------------- #
def make_pool(per_label=5):
    claims = []
    corpus = {}
    doc_id = 1000
    for index in range(per_label):
        for label, id_offset in (("SUPPORT", 0), ("REFUTE", 100), ("NEI", 200)):
            corpus[doc_id] = {
                "doc_id": doc_id,
                "title": str(doc_id),
                "abstract": ["s0", "s1"],
                "structured": False,
            }
            split = "dev" if label == "NEI" else "train"
            claims.append(make_claim(id_offset + index, split, label, [doc_id], {doc_id: [0]}))
            doc_id += 1
    return claims, corpus


def test_stratified_sample_counts_and_determinism():
    claims, corpus = make_pool(5)
    per_label = {"SUPPORT": 2, "REFUTE": 1, "NEI": 3}
    selected_a = bs.stratified_sample(claims, corpus, per_label, seed=11)
    selected_b = bs.stratified_sample(claims, corpus, per_label, seed=11)
    assert [claim["id"] for claim in selected_a] == [claim["id"] for claim in selected_b]
    assert len(selected_a) == len({claim["id"] for claim in selected_a}) == 6


def test_stratified_sample_insufficient():
    claims, corpus = make_pool(2)
    with pytest.raises(DataError, match="only 2 candidate"):
        bs.stratified_sample(claims, corpus, {"SUPPORT": 3, "REFUTE": 1, "NEI": 1}, 11)


def test_make_opaque_id():
    opaque = bs.make_opaque_id(7, "train", 42, "c_", 10)
    assert re.fullmatch(r"c_[0-9a-f]{10}", opaque)
    assert opaque == bs.make_opaque_id(7, "train", 42, "c_", 10)
    assert opaque != bs.make_opaque_id(8, "train", 42, "c_", 10)
    assert opaque != bs.make_opaque_id(7, "dev", 42, "c_", 10)


# --------------------------------------------------------------------------- #
# End-to-end build
# --------------------------------------------------------------------------- #
def test_build_end_to_end(tiny_spec):
    root = tiny_spec["root"]
    summary = bs.build(tiny_spec["config"])

    assert summary["selected_counts"] == {"SUPPORT": 2, "REFUTE": 1, "NEI": 2}
    assert summary["candidate_counts"] == {"SUPPORT": 3, "REFUTE": 1, "NEI": 2}

    verifier_input = read_jsonl(root / "data" / "verifier_input.jsonl")
    gold = read_jsonl(root / "data" / "gold.jsonl")
    key = read_jsonl(root / "data" / "key.jsonl")
    assert len(verifier_input) == len(gold) == len(key) == 5

    # No label or gold information in verifier_input.jsonl.
    for record in verifier_input:
        assert set(record) == {"id", "claim", "evidence_sentences"}
        for sentence in record["evidence_sentences"]:
            assert set(sentence) == {"idx", "text"}
    raw_input = (root / "data" / "verifier_input.jsonl").read_text(encoding="utf-8")
    assert '"label"' not in raw_input
    assert '"gold_idx"' not in raw_input

    # Gold indices are consistent with the fixture annotations.
    corpus = {doc["doc_id"]: doc for doc in tiny_spec["corpus"]}
    expected = {1: [1], 2: [0, 2], 10: [0, 1], 3: [2], 4: [], 5: []}
    by_opaque = {record["id"]: record for record in verifier_input}
    key_by_opaque = {record["id"]: record for record in key}
    gold_by_opaque = {record["id"]: record for record in gold}
    for opaque_id, key_record in key_by_opaque.items():
        claim = next(
            claim
            for claim in tiny_spec["claims_train"] + tiny_spec["claims_dev"]
            if claim["id"] == key_record["claim_id"]
        )
        doc = corpus[key_record["doc_id"]]
        assert [s["text"] for s in by_opaque[opaque_id]["evidence_sentences"]] == doc["abstract"]
        assert gold_by_opaque[opaque_id]["gold_idx"] == expected[claim["id"]]
        assert gold_by_opaque[opaque_id]["label"] == bs.claim_label(claim)

    # Opaque ids follow the configured seed and no claim id is exposed.
    for key_record in key:
        assert key_record["id"] == bs.make_opaque_id(
            summary["seed"],
            key_record["split"],
            key_record["claim_id"],
            "c_",
            10,
        )

    # Hashes cover the three data files and match recomputed digests.
    hashes = {}
    for line in (root / "data" / "HASHES.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        hashes[relative] = digest
    assert set(hashes) == {
        "data/verifier_input.jsonl",
        "data/gold.jsonl",
        "data/key.jsonl",
    }
    for relative, digest in hashes.items():
        assert sha256_file(root / relative) == digest


def test_build_is_deterministic(tiny_spec):
    root = tiny_spec["root"]
    bs.build(tiny_spec["config"])
    first = {
        path.name: path.read_bytes()
        for path in (root / "data").iterdir()
        if path.name in {"verifier_input.jsonl", "gold.jsonl", "key.jsonl", "HASHES.txt"}
    }
    bs.build(tiny_spec["config"], force=True)
    second = {
        path.name: path.read_bytes()
        for path in (root / "data").iterdir()
        if path.name in {"verifier_input.jsonl", "gold.jsonl", "key.jsonl", "HASHES.txt"}
    }
    assert first == second


def test_build_fails_when_too_few_candidates(tiny_spec):
    config_path = write_config(
        tiny_spec["root"], {"SUPPORT": 10, "REFUTE": 1, "NEI": 1}
    )
    with pytest.raises(DataError, match="candidate claims for SUPPORT"):
        bs.build(config_path)


# --------------------------------------------------------------------------- #
# Freeze guard
# --------------------------------------------------------------------------- #
def test_build_refuses_without_force(tiny_spec):
    config_path = tiny_spec["config"]
    bs.build(config_path)
    data = tiny_spec["root"] / "data"
    before = {path.name: path.read_bytes() for path in data.iterdir() if path.is_file()}
    with pytest.raises(DataError, match="refusing to overwrite"):
        bs.build(config_path)
    after = {path.name: path.read_bytes() for path in data.iterdir() if path.is_file()}
    assert before == after


def test_force_without_accept_changes_aborts(tiny_spec, capsys):
    config_path = tiny_spec["config"]
    bs.build(config_path)
    data = tiny_spec["root"] / "data"
    before = {path.name: path.read_bytes() for path in data.iterdir() if path.is_file()}
    # A different seed changes the opaque ids, so all three files differ.
    write_config(tiny_spec["root"], {"SUPPORT": 2, "REFUTE": 1, "NEI": 2}, seed=8)
    with pytest.raises(DataError, match="differ"):
        bs.build(config_path, force=True)
    after = {path.name: path.read_bytes() for path in data.iterdir() if path.is_file()}
    assert before == after
    assert "DIFFERENT" in capsys.readouterr().out


def test_identical_rebuild_reports_identical(tiny_spec, capsys):
    config_path = tiny_spec["config"]
    bs.build(config_path)
    capsys.readouterr()
    summary = bs.build(config_path, force=True)
    assert set(summary["comparisons"].values()) == {"identical"}
    assert summary["written"] == []
    assert capsys.readouterr().out.count("identical") == 3


def test_force_accept_changes_overwrites(tiny_spec):
    config_path = tiny_spec["config"]
    bs.build(config_path)
    write_config(tiny_spec["root"], {"SUPPORT": 2, "REFUTE": 1, "NEI": 2}, seed=8)
    summary = bs.build(config_path, force=True, accept_changes=True)
    assert set(summary["comparisons"].values()) == {"DIFFERENT"}
    assert set(summary["written"]) == {"verifier_input", "gold", "key"}


def test_unmanaged_hash_lines_survive_rebuild(tiny_spec):
    config_path = tiny_spec["config"]
    bs.build(config_path)
    hashes_path = tiny_spec["root"] / "data" / "HASHES.txt"
    pilot_line = "abc123  data/pilot_ids.txt"
    hashes_path.write_text(
        hashes_path.read_text(encoding="utf-8") + pilot_line + "\n", encoding="utf-8"
    )
    bs.build(config_path, force=True)
    lines = hashes_path.read_text(encoding="utf-8").splitlines()
    assert pilot_line in lines
    assert len(lines) == 4


def test_merge_hashes_preserves_order_and_unmanaged(tmp_path):
    path = tmp_path / "HASHES.txt"
    path.write_text("a  data/one\nb  data/pilot_ids.txt\nc  data/two\n", encoding="utf-8")
    bs.merge_hashes(path, {"data/one": "A", "data/two": "C", "data/three": "D"})
    assert path.read_text(encoding="utf-8").splitlines() == [
        "A  data/one",
        "b  data/pilot_ids.txt",
        "C  data/two",
        "D  data/three",
    ]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_validation(tmp_path):
    config_path = write_config(tmp_path / "a", {"SUPPORT": 1, "REFUTE": 1, "NEI": 1})
    assert load_config(config_path)["seed"] == 7

    bad_seed = write_config(tmp_path / "b", {"SUPPORT": 1, "REFUTE": 1, "NEI": 1}, seed="x")
    with pytest.raises(DataError, match="seed"):
        load_config(bad_seed)

    bad_labels = write_config(tmp_path / "c", {"SUPPORT": 1, "REFUTE": 1})
    with pytest.raises(DataError, match="per_label"):
        load_config(bad_labels)

    config = load_config(config_path)
    config["subset"]["id_hex_len"] = 3
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataError, match="id_hex_len"):
        load_config(config_path)


def test_config_missing_file(tmp_path):
    with pytest.raises(DataError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# Real SciFact data (skipped when raw files are absent)
# --------------------------------------------------------------------------- #
REAL_RAW = ROOT / "data" / "raw" / "scifact"


@pytest.mark.skipif(
    not (REAL_RAW / "corpus.jsonl").is_file(), reason="SciFact raw data not fetched"
)
def test_real_data_supports_balanced_subset():
    corpus = bs.load_corpus(REAL_RAW / "corpus.jsonl")
    claims = bs.load_claims(REAL_RAW / "claims_train.jsonl", "train")
    claims += bs.load_claims(REAL_RAW / "claims_dev.jsonl", "dev")
    counts = bs.count_candidates(claims, corpus)
    assert counts["SUPPORT"] >= 50
    assert counts["REFUTE"] >= 50
    assert counts["NEI"] >= 50

    per_label = {"SUPPORT": 50, "REFUTE": 50, "NEI": 50}
    selected = bs.stratified_sample(claims, corpus, per_label, seed=20261114)
    labels = [bs.claim_label(claim) for claim in selected]
    assert labels.count("SUPPORT") == 50
    assert labels.count("REFUTE") == 50
    assert labels.count("NEI") == 50
    for claim in selected:
        label = bs.claim_label(claim)
        doc_id = bs.associated_doc_id(claim, label)
        gold = bs.gold_sentence_indices(
            claim, label, doc_id, len(corpus[doc_id]["abstract"])
        )
        if label == "NEI":
            assert gold == []
        else:
            assert gold
