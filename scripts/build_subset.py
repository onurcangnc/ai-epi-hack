"""Build the stratified SciFact subset used by the verifier-degradation study.

Input: the official SciFact release (claims_train.jsonl, claims_dev.jsonl,
corpus.jsonl) fetched by ``scripts/fetch_scifact.py`` into ``data/raw/scifact``.

Output: three JSONL files plus their SHA-256 hashes.

Policies (also documented in ``data/DATA_REPORT.md``):
- SciFact stores no explicit label. A claim with ``"evidence": {}`` is NEI;
  otherwise the rationale label is ``SUPPORT`` or ``CONTRADICT``, the latter
  mapped to the study's ``REFUTE``.
- Each selected claim has exactly one associated document: the single evidence
  document for SUPPORT/REFUTE, the single cited document for NEI. Claims with
  several associated documents are excluded so that "the associated document"
  is unambiguous and the study seeds control only sampling, not document choice.
- SciFact abstracts are already sentence-split; sentence indices are positions
  in that list. ``gold_idx`` is the sorted union of gold sentence indices for
  the selected document, and ``[]`` for NEI claims (no gold annotation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    LABELS,
    DataError,
    load_config,
    read_jsonl,
    resolve,
    sha256_file,
    write_jsonl,
)

SCIFACT_TO_LABEL = {"SUPPORT": "SUPPORT", "CONTRADICT": "REFUTE"}


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_corpus(path: Path) -> dict[int, dict[str, Any]]:
    corpus: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        doc_id = record.get("doc_id")
        if not isinstance(doc_id, int) or isinstance(doc_id, bool):
            raise DataError(f"{path}: document has invalid 'doc_id': {doc_id!r}")
        if doc_id in corpus:
            raise DataError(f"{path}: duplicate doc_id {doc_id}")
        abstract = record.get("abstract")
        if not isinstance(abstract, list) or not all(
            isinstance(sentence, str) for sentence in abstract
        ):
            raise DataError(f"{path}: document {doc_id} has invalid 'abstract'")
        title = record.get("title")
        if not isinstance(title, str):
            raise DataError(f"{path}: document {doc_id} has invalid 'title'")
        corpus[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "abstract": list(abstract),
            "structured": bool(record.get("structured", False)),
        }
    return corpus


def validate_claim(claim: dict[str, Any], path: Path) -> None:
    claim_id = claim.get("id")
    if not isinstance(claim_id, int) or isinstance(claim_id, bool):
        raise DataError(f"{path}: claim has invalid 'id': {claim_id!r}")
    text = claim.get("claim")
    if not isinstance(text, str) or not text.strip():
        raise DataError(f"{path}: claim {claim_id} has invalid 'claim' text")
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict):
        raise DataError(f"{path}: claim {claim_id} has invalid 'evidence'")
    cited = claim.get("cited_doc_ids")
    if not isinstance(cited, list) or not all(
        isinstance(doc_id, int) and not isinstance(doc_id, bool) for doc_id in cited
    ):
        raise DataError(f"{path}: claim {claim_id} has invalid 'cited_doc_ids'")
    for doc_key, rationales in evidence.items():
        try:
            int(doc_key)
        except (TypeError, ValueError) as exc:
            raise DataError(
                f"{path}: claim {claim_id} has non-numeric evidence key {doc_key!r}"
            ) from exc
        if not isinstance(rationales, list):
            raise DataError(f"{path}: claim {claim_id} evidence {doc_key!r} is not a list")
        for rationale in rationales:
            label = rationale.get("label") if isinstance(rationale, dict) else None
            if label not in SCIFACT_TO_LABEL:
                raise DataError(
                    f"{path}: claim {claim_id} has unknown rationale label {label!r}"
                )
            sentences = rationale.get("sentences")
            if not isinstance(sentences, list) or not all(
                isinstance(idx, int) and not isinstance(idx, bool) for idx in sentences
            ):
                raise DataError(
                    f"{path}: claim {claim_id} has invalid rationale 'sentences'"
                )


def load_claims(path: Path, split: str) -> list[dict[str, Any]]:
    claims = read_jsonl(path)
    for claim in claims:
        validate_claim(claim, path)
        claim["_split"] = split
    return claims


# --------------------------------------------------------------------------- #
# Labels, documents, gold sentences
# --------------------------------------------------------------------------- #
def claim_label(claim: dict[str, Any]) -> str:
    """Return SUPPORT/REFUTE/NEI for a SciFact claim (see module docstring)."""
    evidence = claim["evidence"]
    if not evidence:
        return "NEI"
    mapped = {
        rationale["label"]
        for rationales in evidence.values()
        for rationale in rationales
    }
    unknown = mapped - set(SCIFACT_TO_LABEL)
    if unknown:
        raise DataError(f"claim {claim['id']} has unknown labels: {sorted(unknown)}")
    labels = {SCIFACT_TO_LABEL[label] for label in mapped}
    if len(labels) != 1:
        raise DataError(f"claim {claim['id']} mixes labels: {sorted(labels)}")
    return labels.pop()


def associated_doc_id(claim: dict[str, Any], label: str) -> int:
    """The single document associated with this claim, or DataError."""
    if label == "NEI":
        docs = claim["cited_doc_ids"]
        if len(docs) != 1:
            raise DataError(
                f"claim {claim['id']} is NEI with {len(docs)} cited documents"
            )
        return int(docs[0])
    docs = sorted(int(doc_key) for doc_key in claim["evidence"])
    if len(docs) != 1:
        raise DataError(
            f"claim {claim['id']} has {len(docs)} evidence documents"
        )
    return docs[0]


def gold_sentence_indices(
    claim: dict[str, Any], label: str, doc_id: int, abstract_len: int
) -> list[int]:
    """Sorted union of gold sentence indices for the associated document."""
    if label == "NEI":
        return []
    indices: set[int] = set()
    for rationale in claim["evidence"][str(doc_id)]:
        indices.update(rationale["sentences"])
    for idx in indices:
        if not 0 <= idx < abstract_len:
            raise DataError(
                f"claim {claim['id']}: gold sentence {idx} outside abstract "
                f"of document {doc_id} (length {abstract_len})"
            )
    return sorted(indices)


def is_candidate(claim: dict[str, Any], label: str, corpus: dict[int, dict[str, Any]]) -> bool:
    """True when the claim has exactly one resolvable, non-empty abstract."""
    try:
        doc_id = associated_doc_id(claim, label)
    except DataError:
        return False
    doc = corpus.get(doc_id)
    return bool(doc and doc["abstract"])


def count_candidates(
    claims: list[dict[str, Any]], corpus: dict[int, dict[str, Any]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for claim in claims:
        try:
            label = claim_label(claim)
        except DataError:
            continue
        if is_candidate(claim, label, corpus):
            counts[label] += 1
    return {label: counts.get(label, 0) for label in LABELS}


# --------------------------------------------------------------------------- #
# Sampling and opaque IDs
# --------------------------------------------------------------------------- #
def stratified_sample(
    claims: list[dict[str, Any]],
    corpus: dict[int, dict[str, Any]],
    per_label: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Sample ``per_label`` claims per label, deterministically from ``seed``."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        label = claim_label(claim)
        if is_candidate(claim, label, corpus):
            pools[label].append(claim)

    selected: list[dict[str, Any]] = []
    for label in LABELS:  # fixed order: independent of config key order
        pool = sorted(pools[label], key=lambda claim: (claim["_split"], claim["id"]))
        wanted = per_label[label]
        if len(pool) < wanted:
            raise DataError(
                f"only {len(pool)} candidate claims for {label}, need {wanted}"
            )
        # One RNG stream per label keeps a label's sample stable when another
        # label's pool size changes.
        rng = random.Random(f"{seed}|{label}")
        selected.extend(rng.sample(pool, wanted))
    return selected


def make_opaque_id(
    seed: int, split: str, claim_id: int, prefix: str, hex_len: int
) -> str:
    """Deterministic, content-independent opaque id for a claim."""
    digest = hashlib.sha256(f"{seed}|{split}|{claim_id}".encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:hex_len]}"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def build_records(
    selected: list[dict[str, Any]],
    corpus: dict[int, dict[str, Any]],
    seed: int,
    id_prefix: str,
    id_hex_len: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (verifier_input, gold, key) records in deterministic order."""
    label_order = {label: position for position, label in enumerate(LABELS)}
    ordered = sorted(
        selected,
        key=lambda claim: (
            label_order[claim_label(claim)],
            claim["_split"],
            claim["id"],
        ),
    )

    verifier_input: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for claim in ordered:
        split = claim["_split"]
        claim_id = claim["id"]
        label = claim_label(claim)
        opaque_id = make_opaque_id(seed, split, claim_id, id_prefix, id_hex_len)
        if opaque_id in seen_ids:
            raise DataError(f"opaque id collision for claim {split}/{claim_id}")
        seen_ids.add(opaque_id)

        doc_id = associated_doc_id(claim, label)
        if doc_id not in corpus:
            raise DataError(
                f"claim {split}/{claim_id}: document {doc_id} not in corpus"
            )
        abstract = corpus[doc_id]["abstract"]
        gold_idx = gold_sentence_indices(claim, label, doc_id, len(abstract))

        verifier_input.append(
            {
                "id": opaque_id,
                "claim": claim["claim"],
                "evidence_sentences": [
                    {"idx": index, "text": sentence}
                    for index, sentence in enumerate(abstract)
                ],
            }
        )
        gold.append({"id": opaque_id, "label": label, "gold_idx": gold_idx})
        key.append(
            {
                "id": opaque_id,
                "split": split,
                "claim_id": claim_id,
                "doc_id": doc_id,
            }
        )

    return verifier_input, gold, key


def compute_hashes(base_dir: Path, paths: list[Path]) -> dict[str, str]:
    return {
        Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix(): sha256_file(
            path
        )
        for path in paths
    }


def merge_hashes(path: Path, managed: dict[str, str]) -> None:
    """Rewrite only the managed hash lines, keeping every other line as-is."""
    path = Path(path)
    existing: list[str] = []
    if path.is_file():
        existing = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    written: set[str] = set()
    for line in existing:
        if "  " in line:
            relative = line.split("  ", 1)[1].strip()
            if relative in managed:
                if relative not in written:
                    output.append(f"{managed[relative]}  {relative}")
                    written.add(relative)
                continue
        output.append(line)
    for relative, digest in managed.items():
        if relative not in written:
            output.append(f"{digest}  {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(
    config_path: Path, *, force: bool = False, accept_changes: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)

    data = config["data"]
    subset = config["subset"]
    outputs = {
        "verifier_input": resolve(root, data["output"]["verifier_input"]),
        "gold": resolve(root, data["output"]["gold"]),
        "key": resolve(root, data["output"]["key"]),
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not force:
        raise DataError(
            "refusing to overwrite existing data files: "
            + ", ".join(str(path) for path in existing)
            + "; pass --force (and --accept-changes if they differ)"
        )

    corpus_path = resolve(root, data["raw_dir"]) / data["files"]["corpus"]
    if not corpus_path.is_file():
        raise DataError(
            f"SciFact corpus not found at {corpus_path}; "
            "run `python scripts/fetch_scifact.py` first"
        )

    claims: list[dict[str, Any]] = []
    for split in data["splits"]:
        claims_path = resolve(root, data["raw_dir"]) / data["files"]["claims"].format(
            split=split
        )
        if not claims_path.is_file():
            raise DataError(
                f"SciFact claims not found at {claims_path}; "
                "run `python scripts/fetch_scifact.py` first"
            )
        claims.extend(load_claims(claims_path, split))

    corpus = load_corpus(corpus_path)
    candidate_counts = count_candidates(claims, corpus)
    selected = stratified_sample(claims, corpus, subset["per_label"], config["seed"])
    verifier_input, gold, key = build_records(
        selected,
        corpus,
        config["seed"],
        subset["id_prefix"],
        subset["id_hex_len"],
    )

    with tempfile.TemporaryDirectory(prefix="build_subset_") as tmp:
        tmp_paths = {name: Path(tmp) / path.name for name, path in outputs.items()}
        write_jsonl(tmp_paths["verifier_input"], verifier_input)
        write_jsonl(tmp_paths["gold"], gold)
        write_jsonl(tmp_paths["key"], key)

        comparisons: dict[str, str] = {}
        for name, path in outputs.items():
            if path.exists():
                identical = path.read_bytes() == tmp_paths[name].read_bytes()
                comparisons[name] = "identical" if identical else "DIFFERENT"
            else:
                comparisons[name] = "new"
            print(f"{path.name}: {comparisons[name]}")

        if "DIFFERENT" in comparisons.values() and not accept_changes:
            print(
                "aborting: rebuilt files differ; pass --accept-changes to overwrite",
                file=sys.stderr,
            )
            raise DataError(
                "rebuilt files differ from existing data files; pass --accept-changes"
            )

        written: list[str] = []
        for name, path in outputs.items():
            if comparisons[name] != "identical":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tmp_paths[name].read_bytes())
                written.append(name)

    hashes = compute_hashes(root, list(outputs.values()))
    merge_hashes(resolve(root, data["output"]["hashes"]), hashes)

    label_counts = Counter(record["label"] for record in gold)
    return {
        "seed": config["seed"],
        "candidate_counts": candidate_counts,
        "selected_counts": {label: label_counts[label] for label in LABELS},
        "gold_sentences": sum(len(record["gold_idx"]) for record in gold),
        "outputs": {name: path.relative_to(root).as_posix() for name, path in outputs.items()},
        "comparisons": comparisons,
        "written": written,
        "hashes": hashes,
    }


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
        help="allow rebuilding when data files already exist",
    )
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        help="overwrite when the rebuilt files differ from the existing ones",
    )
    args = parser.parse_args(argv)
    try:
        summary = build(
            args.config, force=args.force, accept_changes=args.accept_changes
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
