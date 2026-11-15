from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import select_pilot as sp
from common import DataError, load_config, sha256_file
from conftest import ROOT, write_config, write_jsonl


def make_gold(per_label: int = 15) -> list[dict]:
    records = []
    for label in ("SUPPORT", "REFUTE", "NEI"):
        for index in range(per_label):
            records.append(
                {"id": f"{label.lower()}_{index:02d}", "label": label, "gold_idx": []}
            )
    return records


def class_counts(gold: list[dict], ids: list[str]) -> dict[str, int]:
    labels = {record["id"]: record["label"] for record in gold}
    counts = {label: 0 for label in ("SUPPORT", "REFUTE", "NEI")}
    for opaque_id in ids:
        counts[labels[opaque_id]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_select_is_deterministic_and_balanced():
    gold = make_gold()
    first = sp.select_pilot_ids(gold, 10, 20261114)
    second = sp.select_pilot_ids(gold, 10, 20261114)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 30
    assert class_counts(gold, first) == {"SUPPORT": 10, "REFUTE": 10, "NEI": 10}
    assert set(first) <= {record["id"] for record in gold}


def test_select_is_independent_of_gold_order():
    gold = make_gold()
    assert sp.select_pilot_ids(gold, 10, 20261114) == sp.select_pilot_ids(
        list(reversed(gold)), 10, 20261114
    )


def test_select_insufficient_class():
    with pytest.raises(DataError, match="only 2 claims for SUPPORT"):
        sp.select_pilot_ids(make_gold(2), 10, 1)


def test_load_gold_rejects_bad_records(tmp_path):
    path = tmp_path / "gold.jsonl"
    write_jsonl(path, [{"id": "", "label": "SUPPORT", "gold_idx": []}])
    with pytest.raises(DataError, match="invalid gold id"):
        sp.load_gold(path)

    write_jsonl(
        path,
        [
            {"id": "c_1", "label": "SUPPORT", "gold_idx": []},
            {"id": "c_1", "label": "REFUTE", "gold_idx": []},
        ],
    )
    with pytest.raises(DataError, match="duplicate"):
        sp.load_gold(path)

    write_jsonl(path, [{"id": "c_1", "label": "MAYBE", "gold_idx": []}])
    with pytest.raises(DataError, match="invalid label"):
        sp.load_gold(path)


# --------------------------------------------------------------------------- #
# Planned prompt count
# --------------------------------------------------------------------------- #
def test_planned_prompt_counts():
    gold = make_gold()
    ids = sp.select_pilot_ids(gold, 10, 20261114)
    counts = sp.planned_prompt_counts(gold, ids)
    assert counts["pilot_claims"] == 30
    assert counts["by_class"] == {"SUPPORT": 10, "REFUTE": 10, "NEI": 10}
    assert counts["by_condition"] == {"C0": 30, "M1": 20, "M2": 30}
    assert counts["planned_prompts"] == 80
    assert counts["k"] == 1


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #
def test_upsert_hash_line_appends_and_replaces(tmp_path):
    path = tmp_path / "HASHES.txt"
    path.write_text("aaa  data/verifier_input.jsonl\nbbb  data/gold.jsonl\n", encoding="utf-8")
    sp.upsert_hash_line(path, sp.PILOT_IDS_REL, "ccc")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "aaa  data/verifier_input.jsonl",
        "bbb  data/gold.jsonl",
        "ccc  data/pilot_ids.txt",
    ]
    # Re-running replaces instead of duplicating.
    sp.upsert_hash_line(path, sp.PILOT_IDS_REL, "ddd")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[-1] == "ddd  data/pilot_ids.txt"


# --------------------------------------------------------------------------- #
# End-to-end build
# --------------------------------------------------------------------------- #
def test_build_end_to_end(tmp_path):
    config_path = write_config(tmp_path, {"SUPPORT": 50, "REFUTE": 50, "NEI": 50})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gold = make_gold()
    write_jsonl(data_dir / "gold.jsonl", gold)
    main_files = [
        "verifier_input.jsonl",
        "gold.jsonl",
        "key.jsonl",
    ]
    hashes_lines = []
    for name in main_files:
        path = data_dir / name
        if not path.exists():
            path.write_bytes(b"fixture\n")
        hashes_lines.append(f"{sha256_file(path)}  data/{name}")
    hashes_path = data_dir / "HASHES.txt"
    hashes_path.write_text("\n".join(hashes_lines) + "\n", encoding="utf-8")

    before_main = {name: (data_dir / name).read_bytes() for name in main_files}
    before_hashes = hashes_path.read_text(encoding="utf-8").splitlines()

    summary = sp.build(config_path)

    assert summary["planned_prompts"] == 80
    assert summary["by_class"] == {"SUPPORT": 10, "REFUTE": 10, "NEI": 10}

    ids_path = data_dir / "pilot_ids.txt"
    lines = ids_path.read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines)
    assert len(lines) == 30
    assert set(lines) <= {record["id"] for record in gold}
    assert class_counts(gold, lines) == {"SUPPORT": 10, "REFUTE": 10, "NEI": 10}

    hashes_after = hashes_path.read_text(encoding="utf-8").splitlines()
    assert hashes_after[: len(before_hashes)] == before_hashes  # main lines untouched
    assert hashes_after[-1] == f"{sha256_file(ids_path)}  data/pilot_ids.txt"
    assert len(hashes_after) == len(before_hashes) + 1

    for name in main_files:
        assert (data_dir / name).read_bytes() == before_main[name]

    # Re-running is idempotent and does not duplicate the pilot hash line.
    sp.build(config_path, force=True)
    assert ids_path.read_text(encoding="utf-8") == "\n".join(lines) + "\n"
    assert hashes_path.read_text(encoding="utf-8").splitlines() == hashes_after


def test_pilot_refuses_without_force(tmp_path):
    config_path = write_config(tmp_path, {"SUPPORT": 50, "REFUTE": 50, "NEI": 50})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_jsonl(data_dir / "gold.jsonl", make_gold())
    sp.build(config_path)
    ids_before = (data_dir / "pilot_ids.txt").read_bytes()
    hashes_before = (data_dir / "HASHES.txt").read_bytes()

    with pytest.raises(DataError, match="refusing to overwrite"):
        sp.build(config_path)

    assert (data_dir / "pilot_ids.txt").read_bytes() == ids_before
    assert (data_dir / "HASHES.txt").read_bytes() == hashes_before


def test_build_requires_pilot_config(tmp_path):
    config_path = write_config(tmp_path, {"SUPPORT": 50, "REFUTE": 50, "NEI": 50})
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["pilot"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataError, match="pilot"):
        load_config(config_path)


def test_config_allows_separate_pilot_seed(tmp_path):
    config_path = write_config(tmp_path, {"SUPPORT": 50, "REFUTE": 50, "NEI": 50})
    config = load_config(config_path)
    assert config["pilot"]["seed"] == 7
    assert config["pilot"]["per_class"] == 10


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
GOLD_PATH = ROOT / "data" / "gold.jsonl"
VERIFIER_INPUT_PATH = ROOT / "data" / "verifier_input.jsonl"


@pytest.mark.skipif(not GOLD_PATH.is_file(), reason="Task 1 data not built")
def test_real_pilot_selection():
    gold = sp.load_gold(GOLD_PATH)
    assert len(gold) == 150
    assert "pilot" in load_config(ROOT / "config.yaml")

    ids = sp.select_pilot_ids(gold, 10, 20261114)
    assert class_counts(gold, ids) == {"SUPPORT": 10, "REFUTE": 10, "NEI": 10}
    assert len(ids) == 30

    verifier_ids = {
        json.loads(line)["id"]
        for line in VERIFIER_INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert set(ids) <= verifier_ids

    counts = sp.planned_prompt_counts(gold, ids)
    assert counts["planned_prompts"] == 80
