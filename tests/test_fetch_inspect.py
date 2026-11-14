from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import yaml

import fetch_scifact as fs
import inspect_scifact as ins
from common import DataError
from conftest import write_config


def make_tar(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "scifact.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def test_download_and_verify(tmp_path):
    source = tmp_path / "source.bin"
    payload = b"scifact-bytes"
    source.write_bytes(payload)
    dest = tmp_path / "downloaded.bin"
    fs.download_and_verify(source.as_uri(), dest, hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload

    with pytest.raises(DataError, match="checksum mismatch"):
        fs.download_and_verify(source.as_uri(), dest, "0" * 64)


def test_extract_dataset(tmp_path):
    tar = make_tar(
        tmp_path,
        {
            "data/corpus.jsonl": b"{}\n",
            "data/claims_train.jsonl": b"",
            "data/claims_dev.jsonl": b"x\n",
            "data/unrelated.json": b"no",
        },
    )
    dest = tmp_path / "raw"
    extracted = fs.extract_dataset(
        tar, dest, ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")
    )
    assert sorted(path.name for path in extracted) == [
        "claims_dev.jsonl",
        "claims_train.jsonl",
        "corpus.jsonl",
    ]
    assert (dest / "corpus.jsonl").read_bytes() == b"{}\n"
    assert (dest / "claims_dev.jsonl").read_bytes() == b"x\n"
    assert not (dest / "unrelated.json").exists()


def test_extract_dataset_missing_file(tmp_path):
    tar = make_tar(tmp_path, {"data/corpus.jsonl": b"{}\n"})
    with pytest.raises(DataError, match="missing files"):
        fs.extract_dataset(tar, tmp_path / "raw", ("corpus.jsonl", "claims_dev.jsonl"))


def test_fetch_end_to_end(tmp_path, tiny_spec):
    config_path = write_config(tiny_spec["root"], {"SUPPORT": 2, "REFUTE": 1, "NEI": 2})
    tar = make_tar(
        tmp_path,
        {
            "data/corpus.jsonl": b'{"doc_id": 1, "title": "t", "abstract": ["s"], "structured": false}\n',
            "data/claims_train.jsonl": b"",
            "data/claims_dev.jsonl": b"",
        },
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["source_url"] = tar.as_uri()
    config["data"]["source_sha256"] = hashlib.sha256(tar.read_bytes()).hexdigest()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = fs.fetch(config_path, force=True)
    raw = tiny_spec["root"] / "data" / "raw" / "scifact"
    for name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
        assert (raw / name).is_file()
    assert summary["tarball_sha256"] == config["data"]["source_sha256"]

    # Second run reuses the verified tarball and existing files.
    summary_again = fs.fetch(config_path)
    assert summary_again["extracted_now"] == []


def test_inspect_report(tiny_spec):
    report = ins.build_report(tiny_spec["raw"])
    assert report.startswith("# SciFact data inspection")
    for required in (
        "## (a) License",
        "CC BY 4.0",
        "## (b) Fields observed",
        "## (c) How NEI claims are represented",
        "## (d) Is there an associated abstract for every claim?",
        "Single-document candidates",
        "empty abstract: 1",
    ):
        assert required in report

    stats = ins.analyze(tiny_spec["raw"])
    assert stats["splits"]["train"]["labels"] == {"SUPPORT": 3, "REFUTE": 2, "NEI": 2}
    assert stats["single_doc_totals"] == {"SUPPORT": 3, "REFUTE": 1, "NEI": 2}
    assert stats["empty_abstracts"] == 1
