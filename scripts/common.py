"""Shared config and JSONL helpers for the SciFact verifier study.

All paths in ``config.yaml`` are resolved relative to the directory that
contains the config file (the project root).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

LABELS = ("SUPPORT", "REFUTE", "NEI")

# Canonical verdict vocabulary for the verifier layer. Gold files keep the
# SciFact-style labels above; this mapping is the only place they are joined.
VERDICTS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")
GOLD_TO_VERDICT = {
    "SUPPORT": "SUPPORTED",
    "REFUTE": "REFUTED",
    "NEI": "NOT_ENOUGH_INFO",
}


class DataError(ValueError):
    """Raised when input data or config violates the expected schema."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, raising DataError with line numbers on bad JSON."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DataError(f"{path}:{lineno}: JSONL line is not an object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise DataError(f"config: missing required key '{where}{key}'")
    return mapping[key]


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise DataError(f"{path}: config root must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise DataError("config root must be a mapping")

    seed = _require(config, "seed", "")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataError("config: 'seed' must be an integer")

    data = _require(config, "data", "")
    for key in ("source_url", "source_sha256", "raw_dir"):
        value = _require(data, key, "data.")
        if not isinstance(value, str) or not value:
            raise DataError(f"config: 'data.{key}' must be a non-empty string")
    if not (
        len(data["source_sha256"]) == 64
        and all(ch in "0123456789abcdef" for ch in data["source_sha256"])
    ):
        raise DataError("config: 'data.source_sha256' must be lowercase hex sha256")
    splits = _require(data, "splits", "data.")
    if (
        not isinstance(splits, list)
        or not splits
        or any(split not in ("train", "dev") for split in splits)
        or len(set(splits)) != len(splits)
    ):
        raise DataError("config: 'data.splits' must be a non-empty subset of [train, dev]")
    files = _require(data, "files", "data.")
    for key in ("corpus", "claims"):
        value = _require(files, key, "data.files.")
        if not isinstance(value, str) or not value:
            raise DataError(f"config: 'data.files.{key}' must be a non-empty string")
    if "{split}" not in files["claims"]:
        raise DataError("config: 'data.files.claims' must contain '{split}'")
    output = _require(data, "output", "data.")
    for key in ("verifier_input", "gold", "key", "hashes"):
        value = _require(output, key, "data.output.")
        if not isinstance(value, str) or not value:
            raise DataError(f"config: 'data.output.{key}' must be a non-empty string")

    subset = _require(config, "subset", "")
    per_label = _require(subset, "per_label", "subset.")
    if not isinstance(per_label, dict) or set(per_label) != set(LABELS):
        raise DataError(
            "config: 'subset.per_label' must map exactly "
            f"{sorted(LABELS)} to positive integers"
        )
    for label in LABELS:
        count = per_label[label]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise DataError(f"config: 'subset.per_label.{label}' must be a positive integer")
    prefix = _require(subset, "id_prefix", "subset.")
    if not isinstance(prefix, str) or not prefix:
        raise DataError("config: 'subset.id_prefix' must be a non-empty string")
    hex_len = _require(subset, "id_hex_len", "subset.")
    if not isinstance(hex_len, int) or isinstance(hex_len, bool) or not 6 <= hex_len <= 32:
        raise DataError("config: 'subset.id_hex_len' must be an integer in [6, 32]")

    pilot = _require(config, "pilot", "")
    pilot_seed = _require(pilot, "seed", "pilot.")
    if not isinstance(pilot_seed, int) or isinstance(pilot_seed, bool):
        raise DataError("config: 'pilot.seed' must be an integer")
    per_class = _require(pilot, "per_class", "pilot.")
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class < 1:
        raise DataError("config: 'pilot.per_class' must be a positive integer")


def resolve(root: Path, relative: str) -> Path:
    """Resolve a config path against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path


def normalize_gold(label: str) -> str:
    """Map a study gold label (SUPPORT/REFUTE/NEI) to the verdict vocabulary."""
    try:
        return GOLD_TO_VERDICT[label]
    except KeyError as exc:
        raise DataError(
            f"unknown gold label {label!r}; expected one of {sorted(GOLD_TO_VERDICT)}"
        ) from exc


def expected_verdict(condition: str, gold_label: str) -> str:
    """Expected verifier verdict for a condition, per SPEC section 6."""
    if condition in ("C0", "C1", "M2"):
        return normalize_gold(gold_label)
    if condition in ("M1", "C_closed"):
        return "NOT_ENOUGH_INFO"
    if condition in ("M3", "M4", "M5"):
        raise NotImplementedError(
            f"expected behavior for {condition} is not defined yet (SPEC section 6)"
        )
    raise DataError(f"unknown condition {condition!r}")
