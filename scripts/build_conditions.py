"""Build the programmatic conditions for SPEC v1.3.1 section 5.

No model is called. Inputs are read-only; outputs go to data/conditions/.

Design notes
------------
- A condition row carries ``claim`` and ``evidence_sentences`` for the call
  layer, plus ``condition`` and ``manipulation_meta`` which the call layer never
  sends.
- Inserted sentences (M2 instruction, C1/C2 filler) are new elements with
  ``idx: null``; original sentences keep their indices.
- C0p is written as a byte-identical copy of C0. The run layer must pass
  ``condition="C0p"`` when consuming that file; the cache key (not the file
  bytes) is what separates the two conditions at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    LABELS,
    VERDICTS,
    DataError,
    expected_verdict,
    load_config,
    normalize_gold,
    read_jsonl,
    resolve,
    sha256_file,
    write_jsonl,
)

CODES = ("C0", "C0p", "C1", "C2", "M1", "M2", "C_closed")

# One standard English stopword list, used by every content-word filter.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been before
    being below between both but by can't cannot could couldn't did didn't do does doesn't doing
    don't down during each few for from further had hadn't has hasn't have haven't having he he'd
    he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in
    into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up very was
    wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
    who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
    """.split()
)

_WORD = re.compile(r"[a-z]+")


def content_words(text: str) -> list[str]:
    """Lowercase alphabetic tokens of length >= 4 that are not stopwords."""
    return [
        token
        for token in _WORD.findall(text.lower())
        if len(token) >= 4 and token not in STOPWORDS
    ]


def stopwords_sha256() -> str:
    payload = "\n".join(sorted(STOPWORDS)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 with LF endings (no BOM, no CRLF translation)."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def metadata_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "c0p_note": "C0p.jsonl is a byte-identical copy of C0.jsonl; pass condition='C0p' at run time.",
        "stopwords_sha256": stopwords_sha256(),
        "stopwords_count": len(STOPWORDS),
        "m2": config["m2"],
        "c1": config["c1"],
    }


def hash_lines(paths: dict[str, Path]) -> list[str]:
    return [
        f"{sha256_file(paths[code])}  data/conditions/{paths[code].name}"
        for code in CODES
    ]


def read_hash_lines(path: Path) -> list[tuple[str, str]]:
    """Parse a HASHES.txt guard file, tolerating CR and surrounding whitespace."""
    rows: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        digest, relative = line.split(None, 1)
        rows.append((digest.strip(), relative.strip()))
    return rows


def refresh_metadata(config_path: Path, *, out_dir: Path | None = None) -> dict[str, Any]:
    """Rewrite only conditions_meta.json and HASHES.txt from existing .jsonl files."""
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    output = Path(out_dir) if out_dir is not None else root / "data" / "conditions"
    paths = {code: output / f"{code}.jsonl" for code in CODES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise DataError(
            "cannot refresh metadata; missing condition files: "
            + ", ".join(str(path) for path in missing)
        )
    meta_path = output / "conditions_meta.json"
    hashes_path = output / "HASHES.txt"
    write_text_lf(meta_path, json.dumps(metadata_payload(config), indent=2) + "\n")
    write_text_lf(hashes_path, "\n".join(hash_lines(paths)) + "\n")
    return {"meta": str(meta_path), "hashes": str(hashes_path)}


def insert_at(evidence: list[dict[str, Any]], position: str, sentence: dict[str, Any]) -> list[dict[str, Any]]:
    if position == "start":
        index = 0
    elif position == "middle":
        index = len(evidence) // 2
    elif position == "end":
        index = len(evidence)
    else:
        raise DataError(f"unknown position {position!r}")
    return evidence[:index] + [sentence] + evidence[index:]


def load_corpus_sentences(corpus_path: Path) -> list[dict[str, Any]]:
    sentences = []
    for record in read_jsonl(corpus_path):
        doc_id = record["doc_id"]
        for sentence in record["abstract"]:
            text = sentence.strip()
            if not text:
                continue
            sentences.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "words": frozenset(content_words(text)),
                    "word_count": len(text.split()),
                }
            )
    return sentences


def validate_templates(templates: dict[str, list[str]], families: list[str]) -> None:
    if set(templates) != set(families):
        raise DataError(
            "config m2.families does not match prompts/m2_templates.json: "
            f"{sorted(families)} vs {sorted(templates)}"
        )
    for family in families:
        for index, template in enumerate(templates[family]):
            count = template.count("{pushed_verdict}")
            if count != 1:
                raise DataError(
                    f"template {family}[{index}] has {{pushed_verdict}} {count} times, need exactly 1"
                )


def build(config_path: Path, *, force: bool = False, out_dir: Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    output = Path(out_dir) if out_dir is not None else root / "data" / "conditions"

    paths = {code: output / f"{code}.jsonl" for code in CODES}
    extras = {
        "skips": output / "skips.jsonl",
        "overlap": output / "m1_overlap_screen.jsonl",
        "meta": output / "conditions_meta.json",
        "hashes": output / "HASHES.txt",
    }
    existing = [p for p in [*paths.values(), *extras.values()] if p.exists()]
    if existing and not force:
        raise DataError(
            "refusing to overwrite existing condition files: "
            + ", ".join(str(p) for p in existing)
            + "; pass --force to rebuild"
        )

    data = config["data"]
    verifier_rows = read_jsonl(resolve(root, data["output"]["verifier_input"]))
    gold_rows = {row["id"]: row for row in read_jsonl(resolve(root, data["output"]["gold"]))}
    key_rows = {row["id"]: row for row in read_jsonl(resolve(root, data["output"]["key"]))}
    corpus_path = resolve(root, data["raw_dir"]) / data["files"]["corpus"]
    corpus_sentences = load_corpus_sentences(corpus_path)

    ids = sorted(row["id"] for row in verifier_rows)
    if set(ids) != set(gold_rows) or set(ids) != set(key_rows):
        raise DataError("verifier_input, gold and key do not cover the same ids")

    m2 = config["m2"]
    c1_cfg = config["c1"]
    families = list(m2["families"])
    positions = list(m2["positions"])
    seed = int(m2["seed"])
    templates = json.loads((root / "prompts" / "m2_templates.json").read_text(encoding="utf-8"))
    validate_templates(templates, families)
    if len(positions) != 3 or set(positions) != {"start", "middle", "end"}:
        raise DataError("config m2.positions must be exactly [start, middle, end]")

    # Pushed verdicts. SUPPORT/REFUTE are forced to differ from expected; NEI
    # gets a seeded 50/50 SPLIT over the two directional verdicts.
    pushed: dict[str, str] = {}
    for cid in ids:
        label = gold_rows[cid]["label"]
        expected = expected_verdict("M2", label)
        if label == "SUPPORT":
            pushed[cid] = "REFUTED"
        elif label == "REFUTE":
            pushed[cid] = "SUPPORTED"
        elif label == "NEI":
            pushed[cid] = ""  # filled below
        else:
            raise DataError(f"unknown label {label!r} for {cid}")
        if pushed[cid] and pushed[cid] not in VERDICTS:
            raise DataError(f"pushed verdict {pushed[cid]!r} not in VERDICTS")
        if pushed[cid] and pushed[cid] == expected:
            raise DataError(f"pushed verdict equals expected for {cid}")
    nei_ids = sorted(cid for cid in ids if gold_rows[cid]["label"] == "NEI")
    nei_rng = random.Random(f"{seed}|NEI|push")
    nei_rng.shuffle(nei_ids)
    split_at = int(round(len(nei_ids) * float(m2["nei_push_split"])))
    for index, cid in enumerate(nei_ids):
        pushed[cid] = "SUPPORTED" if index < split_at else "REFUTED"
        if pushed[cid] == expected_verdict("M2", "NEI"):
            raise DataError(f"pushed verdict equals expected for NEI {cid}")

    # (family, position) Latin square, balanced within each class.
    assignment: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        class_ids = sorted(cid for cid in ids if gold_rows[cid]["label"] == label)
        rng = random.Random(f"{seed}|M2|{label}")
        rng.shuffle(class_ids)
        family_offset = rng.randrange(len(families))
        position_offset = rng.randrange(len(positions))
        template_offsets = {
            family: random.Random(f"{seed}|tpl|{family}").randrange(len(templates[family]))
            for family in families
        }
        template_counts: Counter[str] = Counter()
        for index, cid in enumerate(class_ids):
            family = families[(index + family_offset) % len(families)]
            position = positions[((index // len(families)) + position_offset) % len(positions)]
            template_index = (
                template_offsets[family] + template_counts[family]
            ) % len(templates[family])
            template_counts[family] += 1
            assignment[cid] = {
                "family": family,
                "template_index": template_index,
                "position": position,
            }

    by_id = {row["id"]: row for row in verifier_rows}
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    c1_pools: list[int] = []
    c2_missing: list[str] = []
    m1_empty: list[str] = []
    empty_rows: list[str] = []
    overlap_rows: list[dict[str, Any]] = []

    for cid in ids:
        base = by_id[cid]
        label = gold_rows[cid]["label"]
        own_doc = key_rows[cid]["doc_id"]
        evidence = [dict(sentence) for sentence in base["evidence_sentences"]]
        claim_words = set(content_words(base["claim"]))

        # C0 / C0p payload.
        rows["C0"].append(
            {
                "id": cid,
                "claim": base["claim"],
                "evidence_sentences": evidence,
                "condition": "C0",
                "manipulation_meta": {},
            }
        )

        # M2: one instruction as its own sentence element.
        meta = assignment[cid]
        template = templates[meta["family"]][meta["template_index"]]
        instruction = template.replace("{pushed_verdict}", pushed[cid])
        instruction_words = len(instruction.split())
        m2_evidence = insert_at(
            [dict(sentence) for sentence in evidence],
            meta["position"],
            {"idx": None, "text": instruction},
        )
        rows["M2"].append(
            {
                "id": cid,
                "claim": base["claim"],
                "evidence_sentences": m2_evidence,
                "condition": "M2",
                "manipulation_meta": {
                    "family": meta["family"],
                    "template_index": meta["template_index"],
                    "position": meta["position"],
                    "pushed_verdict": pushed[cid],
                    "instruction_words": instruction_words,
                },
            }
        )

        # C1 and C2 candidate pools in one pass over other documents.
        clean: list[dict[str, Any]] = []
        sized: list[dict[str, Any]] = []
        for sentence in corpus_sentences:
            if sentence["doc_id"] == own_doc:
                continue
            if sentence["words"] & claim_words:
                continue
            clean.append(sentence)
            if abs(sentence["word_count"] - instruction_words) <= 2:
                sized.append(sentence)
        c1_pools.append(len(clean))
        c1_rng = random.Random(f"{seed}|C1|{cid}")
        chosen = c1_rng.sample(clean, int(c1_cfg["n_sentences"]))
        c1_evidence = [dict(sentence) for sentence in evidence]
        for sentence in chosen:
            c1_evidence.append({"idx": None, "text": sentence["text"]})
        rows["C1"].append(
            {
                "id": cid,
                "claim": base["claim"],
                "evidence_sentences": c1_evidence,
                "condition": "C1",
                "manipulation_meta": {
                    "pool_size": len(clean),
                    "source_doc_ids": [sentence["doc_id"] for sentence in chosen],
                },
            }
        )

        if not sized:
            c2_missing.append(cid)
            filler = None
        else:
            filler = random.Random(f"{seed}|C2|{cid}").choice(sized)
        if filler is not None:
            c2_evidence = insert_at(
                [dict(sentence) for sentence in evidence],
                meta["position"],
                {"idx": None, "text": filler["text"]},
            )
            rows["C2"].append(
                {
                    "id": cid,
                    "claim": base["claim"],
                    "evidence_sentences": c2_evidence,
                    "condition": "C2",
                    "manipulation_meta": {
                        "m2_family": meta["family"],
                        "m2_position": meta["position"],
                        "instruction_words": instruction_words,
                        "filler_words": filler["word_count"],
                        "source_doc_id": filler["doc_id"],
                    },
                }
            )

        # M1 / C_closed: SUPPORT and REFUTE only.
        if label in ("SUPPORT", "REFUTE"):
            removed = set(gold_rows[cid]["gold_idx"])
            remaining = [dict(s) for s in evidence if s["idx"] not in removed]
            if not remaining:
                m1_empty.append(cid)
            matched: list[int] = []
            max_ratio = 0.0
            for sentence in remaining:
                if not claim_words:
                    ratio = 0.0
                else:
                    ratio = len(set(content_words(sentence["text"])) & claim_words) / len(claim_words)
                max_ratio = max(max_ratio, ratio)
                if ratio >= 0.5:
                    matched.append(int(sentence["idx"]))
            overlap_rows.append(
                {
                    "id": cid,
                    "claim": base["claim"],
                    "flagged": bool(matched),
                    "matched_idx": sorted(matched),
                    "max_ratio": round(max_ratio, 4),
                }
            )
            rows["M1"].append(
                {
                    "id": cid,
                    "claim": base["claim"],
                    "evidence_sentences": remaining,
                    "condition": "M1",
                    "manipulation_meta": {
                        "removed_idx": sorted(removed),
                        "expected_verdict": expected_verdict("M1", label),
                    },
                }
            )
            rows["C_closed"].append(
                {
                    "id": cid,
                    "claim": base["claim"],
                    "evidence_sentences": [],
                    "condition": "C_closed",
                    "manipulation_meta": {},
                }
            )
        else:
            empty_rows.append(cid)

    # Stop before writing anything if a hard precondition fails.
    problems = []
    if m1_empty:
        problems.append(f"M1 evidence empty for {len(m1_empty)} claims: {m1_empty[:5]}")
    low_pools = [(cid, size) for cid, size in zip(ids, c1_pools) if size < 30]
    if low_pools:
        problems.append(f"C1 candidate pool < 30 for {len(low_pools)} claims: {low_pools[:5]}")
    if c2_missing:
        problems.append(f"no C2 filler within +-2 words for {len(c2_missing)} claims: {c2_missing[:5]}")
    if problems:
        raise DataError("STOP: " + " | ".join(problems))

    # Write deterministic, id-sorted files.
    for code in CODES:
        rows[code].sort(key=lambda row: row["id"])
    output.mkdir(parents=True, exist_ok=True)
    for code in CODES:
        if code == "C0p":
            continue
        write_jsonl(paths[code], rows[code])
    shutil.copyfile(paths["C0"], paths["C0p"])  # byte-identical by contract
    write_jsonl(extras["skips"], [{"id": cid, "reason": "NEI"} for cid in sorted(empty_rows)])
    write_jsonl(extras["overlap"], sorted(overlap_rows, key=lambda row: row["id"]))

    write_text_lf(extras["meta"], json.dumps(metadata_payload(config), indent=2) + "\n")
    write_text_lf(extras["hashes"], "\n".join(hash_lines(paths)) + "\n")

    family_position: dict[str, dict[str, int]] = {}
    for label in LABELS:
        table: Counter[str] = Counter()
        for cid, meta_row in assignment.items():
            if gold_rows[cid]["label"] == label:
                table[f"{meta_row['family']}|{meta_row['position']}"] += 1
        family_position[label] = dict(sorted(table.items()))

    nei_pushed = Counter(pushed[cid] for cid in nei_ids)
    return {
        "rows": {
            code: len(rows["C0"]) if code == "C0p" else len(rows[code])
            for code in CODES
        },
        "m1_skips": len(empty_rows),
        "m1_empty": len(m1_empty),
        "m1_overlap_flagged": sum(1 for row in overlap_rows if row["flagged"]),
        "c1_pool_min": min(c1_pools),
        "c1_pool_max": max(c1_pools),
        "family_position": family_position,
        "nei_pushed": dict(sorted(nei_pushed.items())),
        "outputs": {code: str(paths[code]) for code in CODES},
        "hashes": str(extras["hashes"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="rewrite only conditions_meta.json and HASHES.txt (never the .jsonl files)",
    )
    args = parser.parse_args(argv)
    try:
        if args.metadata_only:
            summary = refresh_metadata(args.config)
        else:
            summary = build(args.config, force=args.force)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
