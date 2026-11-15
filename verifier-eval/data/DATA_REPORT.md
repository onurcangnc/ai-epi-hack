# SciFact data inspection

Computed from the official SciFact release in `data/raw/scifact/` by `scripts/inspect_scifact.py`. Nothing here is assumed; counts are read from the files.

## (a) License

The official repository's `LICENSE.md` splits the license by component:

- Claims and evidence annotations (`claims_*.jsonl`): **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).
- Abstracts (`corpus.jsonl`, derived from S2ORC): **ODC-By 1.0** (<https://opendatacommons.org/licenses/by/1-0/>).
- Repository code: **Apache 2.0**.

The Hugging Face dataset card for `allenai/scifact` instead tags the whole dataset `cc-by-nc-2.0`. That conflicts with the repository's `LICENSE.md`; treat the per-component `LICENSE.md` terms as authoritative.

## (b) Fields observed

| file | rows | fields observed |
|---|---|---|
| claims_train.jsonl | 809 | cited_doc_ids, claim, evidence, id |
| claims_dev.jsonl | 300 | cited_doc_ids, claim, evidence, id |
| claims_test.jsonl | 300 | claim, id |

`corpus.jsonl`: 5183 documents with `doc_id` (int), `title` (str), `abstract` (list[str]), `structured` (bool). Abstracts are already sentence-split; the list position is the sentence index.

Claims in `train`/`dev` carry `id` (int), `claim` (str), `evidence` (object: doc id string -> list of `{label: "SUPPORT"|"CONTRADICT", sentences: list[int]}`), and `cited_doc_ids` (list[int]). There is **no explicit label field**.

## (c) How NEI claims are represented

NEI is the absence of evidence: a claim is NEI exactly when `evidence` is the empty object `{}`. The files store no label value "NEI". SUPPORT/REFUTE come from the rationale `label` field, and SciFact's `CONTRADICT` is mapped to the study's `REFUTE`.

### Label counts

| split | SUPPORT | REFUTE | NEI | total |
|---|---|---|---|---|
| train | 332 | 173 | 304 | 809 |
| dev | 124 | 64 | 112 | 300 |
| **total** | 456 | 237 | 416 | 1109 |

## (d) Is there an associated abstract for every claim?

For `train` and `dev`: **yes, for every claim including NEI**. Every claim has at least one `cited_doc_ids` entry and every cited document resolves to an abstract in `corpus.jsonl`. NEI claims simply have no annotated gold sentences inside that abstract.

| split | claims | empty `cited_doc_ids` | cited docs missing from corpus | NEI claims with resolvable cited doc |
|---|---|---|---|---|
| train | 809 | 0 | 0 | 304 |
| dev | 300 | 0 | 0 | 112 |
| test | 300 | 300 | 0 | 0 |

`claims_test.jsonl` is unlabeled and carries only claim, id: it has no `cited_doc_ids` and no `evidence`, so test claims cannot be paired with an abstract and are not used.

## Multi-document claims

Some claims have several associated documents (evidence documents for SUPPORT/REFUTE, cited documents for NEI). The study's policy selects only claims with exactly one associated document so that "the associated document" needs no extra selection rule.

| split | evidence docs per claim | cited docs per claim |
|---|---|---|
| train | 0:304, 1:468, 2:20, 3:12, 4:5 | 1:732, 2:53, 3:16, 4:7, 5:1 |
| dev | 0:112, 1:178, 2:4, 3:2, 4:3, 5:1 | 1:276, 2:15, 3:4, 4:3, 5:2 |

### Single-document candidates (the subset population)

| split | SUPPORT | REFUTE | NEI |
|---|---|---|---|
| train | 308 | 160 | 284 |
| dev | 117 | 61 | 110 |
| **total** | 425 | 221 | 394 |

Corpus documents with an empty abstract: 0.
