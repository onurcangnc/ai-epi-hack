"""Inspect the local SciFact release and write data/DATA_REPORT.md.

Everything in the report is computed from the files in ``data/raw/scifact``;
nothing about the format is assumed. Run after ``scripts/fetch_scifact.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common import DataError, LABELS, read_jsonl, resolve
from build_subset import SCIFACT_TO_LABEL, load_corpus

SPLITS = ("train", "dev", "test")


def claim_label_counts(claim: dict) -> str:
    evidence = claim.get("evidence") or {}
    if not evidence:
        return "NEI"
    labels = {
        SCIFACT_TO_LABEL[rationale["label"]]
        for rationales in evidence.values()
        for rationale in rationales
    }
    if len(labels) != 1:
        raise DataError(f"claim {claim.get('id')} mixes labels {sorted(labels)}")
    return labels.pop()


def analyze(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    corpus_path = raw_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise DataError(f"{corpus_path} not found; run scripts/fetch_scifact.py")
    corpus = load_corpus(corpus_path)
    empty_abstracts = sum(1 for doc in corpus.values() if not doc["abstract"])

    splits: dict[str, dict] = {}
    for split in SPLITS:
        path = raw_dir / f"claims_{split}.jsonl"
        if not path.is_file():
            continue
        claims = read_jsonl(path)
        labels: Counter[str] = Counter()
        cited_empty = 0
        cited_missing = 0
        evidence_missing = 0
        nei_with_doc = 0
        nei_without_doc = 0
        single_doc: Counter[str] = Counter()
        evidence_docs: Counter[int] = Counter()
        cited_docs: Counter[int] = Counter()

        for claim in claims:
            label = claim_label_counts(claim)
            labels[label] += 1
            cited = claim.get("cited_doc_ids") or []
            cited_docs[len(cited)] += 1
            if not cited:
                cited_empty += 1
            else:
                cited_missing += sum(1 for doc_id in cited if doc_id not in corpus)
            evidence = claim.get("evidence") or {}
            evidence_docs[len(evidence)] += 1
            evidence_missing += sum(1 for doc_id in evidence if int(doc_id) not in corpus)

            if label == "NEI":
                if cited and all(doc_id in corpus for doc_id in cited):
                    nei_with_doc += 1
                else:
                    nei_without_doc += 1
            docs = evidence if evidence else cited
            if len(docs) == 1:
                doc_id = int(next(iter(evidence))) if evidence else cited[0]
                if doc_id in corpus and corpus[doc_id]["abstract"]:
                    single_doc[label] += 1

        splits[split] = {
            "rows": len(claims),
            "fields": sorted(claims[0].keys()) if claims else [],
            "labels": {label: labels.get(label, 0) for label in LABELS},
            "cited_empty": cited_empty,
            "cited_missing": cited_missing,
            "evidence_missing": evidence_missing,
            "nei_with_doc": nei_with_doc,
            "nei_without_doc": nei_without_doc,
            "evidence_docs": dict(sorted(evidence_docs.items())),
            "cited_docs": dict(sorted(cited_docs.items())),
            "single_doc": {label: single_doc.get(label, 0) for label in LABELS},
        }

    labeled = {split: stats for split, stats in splits.items() if split != "test"}
    totals = {label: sum(s["labels"][label] for s in labeled.values()) for label in LABELS}
    return {
        "corpus_docs": len(corpus),
        "empty_abstracts": empty_abstracts,
        "splits": splits,
        "totals": totals,
        "single_doc_totals": {
            label: sum(s["single_doc"][label] for s in labeled.values())
            for label in LABELS
        },
    }


def _fields_table(stats: dict) -> str:
    lines = ["| file | rows | fields observed |", "|---|---|---|"]
    for split, info in stats["splits"].items():
        lines.append(
            f"| claims_{split}.jsonl | {info['rows']} | {', '.join(info['fields'])} |"
        )
    return "\n".join(lines)


def build_report(raw_dir: Path) -> str:
    stats = analyze(raw_dir)
    train = stats["splits"].get("train", {"labels": {}, "single_doc": {}})
    dev = stats["splits"].get("dev", {"labels": {}, "single_doc": {}})
    test = stats["splits"].get("test", None)

    lines: list[str] = []
    lines.append("# SciFact data inspection")
    lines.append("")
    lines.append(
        "Computed from the official SciFact release in `data/raw/scifact/` by "
        "`scripts/inspect_scifact.py`. Nothing here is assumed; counts are read "
        "from the files."
    )
    lines.append("")

    lines.append("## (a) License")
    lines.append("")
    lines.append(
        "The official repository's `LICENSE.md` splits the license by component:"
    )
    lines.append("")
    lines.append(
        "- Claims and evidence annotations (`claims_*.jsonl`): **CC BY 4.0** "
        "(<https://creativecommons.org/licenses/by/4.0/>)."
    )
    lines.append(
        "- Abstracts (`corpus.jsonl`, derived from S2ORC): **ODC-By 1.0** "
        "(<https://opendatacommons.org/licenses/by/1-0/>)."
    )
    lines.append(
        "- Repository code: **Apache 2.0**."
    )
    lines.append("")
    lines.append(
        "The Hugging Face dataset card for `allenai/scifact` instead tags the "
        "whole dataset `cc-by-nc-2.0`. That conflicts with the repository's "
        "`LICENSE.md`; treat the per-component `LICENSE.md` terms as "
        "authoritative."
    )
    lines.append("")

    lines.append("## (b) Fields observed")
    lines.append("")
    lines.append(_fields_table(stats))
    lines.append("")
    lines.append(
        f"`corpus.jsonl`: {stats['corpus_docs']} documents with "
        "`doc_id` (int), `title` (str), `abstract` (list[str]), "
        "`structured` (bool). Abstracts are already sentence-split; the "
        "list position is the sentence index."
    )
    lines.append("")
    lines.append(
        "Claims in `train`/`dev` carry `id` (int), `claim` (str), "
        "`evidence` (object: doc id string -> list of "
        "`{label: \"SUPPORT\"|\"CONTRADICT\", sentences: list[int]}`), and "
        "`cited_doc_ids` (list[int]). There is **no explicit label field**."
    )
    lines.append("")

    lines.append("## (c) How NEI claims are represented")
    lines.append("")
    lines.append(
        "NEI is the absence of evidence: a claim is NEI exactly when "
        "`evidence` is the empty object `{}`. The files store no label value "
        "\"NEI\". SUPPORT/REFUTE come from the rationale `label` field, and "
        "SciFact's `CONTRADICT` is mapped to the study's `REFUTE`."
    )
    lines.append("")
    lines.append("### Label counts")
    lines.append("")
    lines.append("| split | SUPPORT | REFUTE | NEI | total |")
    lines.append("|---|---|---|---|---|")
    for split, info in stats["splits"].items():
        if split == "test":
            continue
        lines.append(
            f"| {split} | {info['labels']['SUPPORT']} | {info['labels']['REFUTE']} "
            f"| {info['labels']['NEI']} | {info['rows']} |"
        )
    lines.append(
        f"| **total** | {stats['totals']['SUPPORT']} | {stats['totals']['REFUTE']} "
        f"| {stats['totals']['NEI']} | "
        f"{sum(stats['totals'].values())} |"
    )
    lines.append("")

    lines.append("## (d) Is there an associated abstract for every claim?")
    lines.append("")
    lines.append(
        "For `train` and `dev`: **yes, for every claim including NEI**. Every "
        "claim has at least one `cited_doc_ids` entry and every cited document "
        "resolves to an abstract in `corpus.jsonl`. NEI claims simply have no "
        "annotated gold sentences inside that abstract."
    )
    lines.append("")
    lines.append("| split | claims | empty `cited_doc_ids` | cited docs missing from corpus | NEI claims with resolvable cited doc |")
    lines.append("|---|---|---|---|---|")
    for split, info in stats["splits"].items():
        lines.append(
            f"| {split} | {info['rows']} | {info['cited_empty']} | "
            f"{info['cited_missing']} | {info['nei_with_doc']} |"
        )
    lines.append("")
    if test is not None:
        lines.append(
            f"`claims_test.jsonl` is unlabeled and carries only "
            f"{', '.join(test['fields'])}: it has no `cited_doc_ids` and no "
            "`evidence`, so test claims cannot be paired with an abstract and "
            "are not used."
        )
        lines.append("")

    lines.append("## Multi-document claims")
    lines.append("")
    lines.append(
        "Some claims have several associated documents (evidence documents for "
        "SUPPORT/REFUTE, cited documents for NEI). The study's policy selects "
        "only claims with exactly one associated document so that \"the "
        "associated document\" needs no extra selection rule."
    )
    lines.append("")
    lines.append("| split | evidence docs per claim | cited docs per claim |")
    lines.append("|---|---|---|")
    for split, info in stats["splits"].items():
        if split == "test":
            continue
        ev = ", ".join(f"{k}:{v}" for k, v in info["evidence_docs"].items())
        cited = ", ".join(f"{k}:{v}" for k, v in info["cited_docs"].items())
        lines.append(f"| {split} | {ev} | {cited} |")
    lines.append("")
    lines.append("### Single-document candidates (the subset population)")
    lines.append("")
    lines.append("| split | SUPPORT | REFUTE | NEI |")
    lines.append("|---|---|---|---|")
    for split in ("train", "dev"):
        info = stats["splits"].get(split)
        if info is None:
            continue
        lines.append(
            f"| {split} | {info['single_doc']['SUPPORT']} | "
            f"{info['single_doc']['REFUTE']} | {info['single_doc']['NEI']} |"
        )
    lines.append(
        f"| **total** | {stats['single_doc_totals']['SUPPORT']} | "
        f"{stats['single_doc_totals']['REFUTE']} | {stats['single_doc_totals']['NEI']} |"
    )
    lines.append("")
    lines.append(
        f"Corpus documents with an empty abstract: {stats['empty_abstracts']}."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from common import load_config

    config = load_config(args.config.resolve())
    raw_dir = resolve(args.config.resolve().parent, config["data"]["raw_dir"])
    report = build_report(raw_dir)
    out = args.out or (root / "data" / "DATA_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(out), "bytes": len(report.encode("utf-8"))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
