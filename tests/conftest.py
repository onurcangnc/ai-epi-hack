from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_claim(claim_id, split, label, docs, sentences=None):
    sentences = sentences or {}
    record = {"id": claim_id, "claim": f"Claim number {claim_id}.", "_split": split}
    if label == "NEI":
        record["evidence"] = {}
        record["cited_doc_ids"] = list(docs)
    else:
        record["evidence"] = {
            str(doc): [
                {
                    "label": "SUPPORT" if label == "SUPPORT" else "CONTRADICT",
                    "sentences": list(sentences.get(doc, [0])),
                }
            ]
            for doc in docs
        }
        record["cited_doc_ids"] = list(docs)
    return record


def write_config(root: Path, per_label: dict[str, int], seed: int = 7) -> Path:
    config = {
        "seed": seed,
        "data": {
            "source_url": "https://example.invalid/data.tar.gz",
            "source_sha256": "0" * 64,
            "raw_dir": "data/raw/scifact",
            "splits": ["train", "dev"],
            "files": {"corpus": "corpus.jsonl", "claims": "claims_{split}.jsonl"},
            "output": {
                "verifier_input": "data/verifier_input.jsonl",
                "gold": "data/gold.jsonl",
                "key": "data/key.jsonl",
                "hashes": "data/HASHES.txt",
            },
        },
        "subset": {"per_label": per_label, "id_prefix": "c_", "id_hex_len": 10},
        "pilot": {"seed": 7, "per_class": 10},
    }
    path = root / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def tiny_spec(tmp_path: Path) -> dict:
    """Small hand-made SciFact-like dataset with known candidates."""
    root = tmp_path
    raw = root / "data" / "raw" / "scifact"
    corpus = [
        {"doc_id": 100, "title": "Doc 100", "abstract": ["a0", "a1", "a2"], "structured": False},
        {"doc_id": 101, "title": "Doc 101", "abstract": ["b0", "b1", "b2", "b3"], "structured": False},
        {"doc_id": 102, "title": "Doc 102", "abstract": ["c0", "c1"], "structured": False},
        {"doc_id": 103, "title": "Doc 103", "abstract": ["d0"], "structured": False},
        {"doc_id": 104, "title": "Doc 104", "abstract": ["e0", "e1"], "structured": False},
        {"doc_id": 105, "title": "Doc 105", "abstract": [], "structured": False},
    ]
    claims_train = [
        make_claim(1, "train", "SUPPORT", [100], {100: [1]}),
        make_claim(2, "train", "SUPPORT", [100], {100: [0, 2]}),
        make_claim(3, "train", "REFUTE", [101], {101: [2]}),
        make_claim(4, "train", "NEI", [102]),
        make_claim(5, "train", "NEI", [103]),
        make_claim(6, "train", "SUPPORT", [100, 101], {100: [0], 101: [1]}),  # multi-doc
        make_claim(8, "train", "REFUTE", [999], {999: [0]}),  # missing doc
    ]
    claims_dev = [
        make_claim(7, "dev", "NEI", [100, 102]),  # multi cited doc
        make_claim(9, "dev", "REFUTE", [105], {105: [0]}),  # empty abstract
        make_claim(10, "dev", "SUPPORT", [101], {101: [0, 1]}),
    ]
    write_jsonl(raw / "corpus.jsonl", corpus)
    write_jsonl(raw / "claims_train.jsonl", claims_train)
    write_jsonl(raw / "claims_dev.jsonl", claims_dev)
    config_path = write_config(root, {"SUPPORT": 2, "REFUTE": 1, "NEI": 2})
    return {
        "root": root,
        "raw": raw,
        "corpus": corpus,
        "claims_train": claims_train,
        "claims_dev": claims_dev,
        "config": config_path,
    }
