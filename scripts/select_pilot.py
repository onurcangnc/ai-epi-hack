"""Select the 30-claim Opus 5.5 pilot subset (Task 2b-1).

Reads ``data/gold.jsonl`` (the frozen 150-claim subset), samples
``pilot.per_class`` claims per label using a pilot-only RNG stream seeded by
``pilot.seed``, writes ``data/pilot_ids.txt`` (one opaque id per line, sorted)
and appends the file's SHA-256 to ``data/HASHES.txt``.

No export/ingest here: this task only freezes which claims the manual pilot
will use, so the subset exists before any DeepSeek result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from common import (
    DataError,
    LABELS,
    load_config,
    read_jsonl,
    resolve,
    sha256_file,
)

PILOT_IDS_REL = "data/pilot_ids.txt"
CONDITION_ORDER = ("C0", "M1", "M2")

# M1 is a no-op for NEI claims (SPEC §5), so the pilot plan skips it there.
CONDITIONS = {
    "SUPPORT": ("C0", "M1", "M2"),
    "REFUTE": ("C0", "M1", "M2"),
    "NEI": ("C0", "M2"),
}


def load_gold(path: Path) -> list[dict]:
    """Load and validate data/gold.jsonl records."""
    records = read_jsonl(path)
    if not records:
        raise DataError(f"{path}: gold file is empty")
    seen: set[str] = set()
    for record in records:
        opaque_id = record.get("id")
        label = record.get("label")
        if not isinstance(opaque_id, str) or not opaque_id:
            raise DataError(f"{path}: invalid gold id {opaque_id!r}")
        if opaque_id in seen:
            raise DataError(f"{path}: duplicate gold id {opaque_id}")
        seen.add(opaque_id)
        if label not in LABELS:
            raise DataError(f"{path}: invalid label {label!r} for {opaque_id}")
    return records


def select_pilot_ids(gold: list[dict], per_class: int, seed: int) -> list[str]:
    """Sample ``per_class`` ids per label from a dedicated pilot RNG stream."""
    pools: dict[str, list[str]] = {label: [] for label in LABELS}
    for record in gold:
        pools[record["label"]].append(record["id"])

    selected: list[str] = []
    for label in LABELS:  # fixed order, independent of gold file order
        pool = sorted(pools[label])
        if len(pool) < per_class:
            raise DataError(
                f"only {len(pool)} claims for {label}, cannot pick {per_class}"
            )
        rng = random.Random(f"{seed}|pilot|{label}")
        selected.extend(rng.sample(pool, per_class))
    return sorted(selected)


def planned_prompt_counts(gold: list[dict], pilot_ids: list[str]) -> dict:
    """Prompt counts for the manual pilot: S/R x {C0,M1,M2}, NEI x {C0,M2}."""
    label_by_id = {record["id"]: record["label"] for record in gold}
    by_condition: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    for opaque_id in pilot_ids:
        label = label_by_id[opaque_id]
        by_class[label] += 1
        for condition in CONDITIONS[label]:
            by_condition[condition] += 1
    return {
        "pilot_claims": len(pilot_ids),
        "by_class": {label: by_class[label] for label in LABELS},
        "by_condition": {condition: by_condition[condition] for condition in CONDITION_ORDER},
        "planned_prompts": sum(by_condition.values()),
        "k": 1,
    }


def write_pilot_ids(path: Path, ids: list[str]) -> None:
    if not ids:
        raise DataError("refusing to write an empty pilot id list")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for opaque_id in sorted(ids):
            fh.write(opaque_id + "\n")


def upsert_hash_line(hashes_path: Path, relative: str, digest: str) -> None:
    """Append the hash line, replacing an existing line for ``relative``."""
    lines: list[str] = []
    if Path(hashes_path).is_file():
        lines = Path(hashes_path).read_text(encoding="utf-8").splitlines()
    kept = [
        line
        for line in lines
        if not ("  " in line and line.split("  ", 1)[1].strip() == relative)
    ]
    kept.append(f"{digest}  {relative}")
    Path(hashes_path).parent.mkdir(parents=True, exist_ok=True)
    Path(hashes_path).write_text("\n".join(kept) + "\n", encoding="utf-8")


def build(config_path: Path, *, force: bool = False) -> dict:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    pilot = config["pilot"]

    ids_path = resolve(root, PILOT_IDS_REL)
    if ids_path.exists() and not force:
        raise DataError(
            f"refusing to overwrite {PILOT_IDS_REL}; pass --force to reselect"
        )

    gold_path = resolve(root, config["data"]["output"]["gold"])
    if not gold_path.is_file():
        raise DataError(f"gold file not found at {gold_path}; run scripts/build_subset.py first")
    gold = load_gold(gold_path)
    pilot_ids = select_pilot_ids(gold, pilot["per_class"], pilot["seed"])

    write_pilot_ids(ids_path, pilot_ids)
    digest = sha256_file(ids_path)
    upsert_hash_line(
        resolve(root, config["data"]["output"]["hashes"]), PILOT_IDS_REL, digest
    )

    summary = planned_prompt_counts(gold, pilot_ids)
    summary.update(
        {
            "seed": pilot["seed"],
            "ids_file": PILOT_IDS_REL,
            "sha256": digest,
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="path to config.yaml (default: project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reselect and overwrite data/pilot_ids.txt",
    )
    args = parser.parse_args(argv)
    try:
        summary = build(args.config, force=args.force)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
