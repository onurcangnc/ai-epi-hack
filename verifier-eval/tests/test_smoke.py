"""TASK A2: the smoke claims must be disjoint from the main 150 and balanced."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_jsonl  # noqa: E402


def test_smoke_claims_are_disjoint_from_main_150() -> None:
    smoke = read_jsonl(ROOT / "smoke" / "claims.jsonl")
    key = read_jsonl(ROOT / "data" / "key.jsonl")

    assert len(smoke) == 6
    assert Counter(record["label"] for record in smoke) == {
        "SUPPORT": 2,
        "REFUTE": 2,
        "NEI": 2,
    }

    main_ids = {record["id"] for record in key}
    main_claims = {(record["split"], record["claim_id"]) for record in key}
    smoke_ids = {record["id"] for record in smoke}
    smoke_claims = {(record["split"], record["claim_id"]) for record in smoke}

    assert len(smoke_ids) == 6
    assert smoke_ids.isdisjoint(main_ids)
    assert smoke_claims.isdisjoint(main_claims)

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert config["seed"] == 20261114
