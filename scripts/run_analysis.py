"""Single analysis entry point for a DeepSeek verifier log (SPEC v1.5 §6, §7).

The driver reads ONE explicit log path, fixes the confidence cut points from
that log's C0 rows, assigns the bins, asserts k = 5 result rows per (claim,
condition), and then writes the long table, C0 accuracy, the filter accounting,
the key-cell table, the contrasts, both sensitivity tables, the M1/M2 strata
and the summary. It never globs across logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "scripts", _ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import DataError, load_config  # noqa: E402
from analysis import contrasts, cutpoints, loaders, report, strata, tables  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def assert_complete_cells(samples: list[dict[str, Any]], k: int) -> None:
    """SPEC v1.5 §7: before analysis, 5 result rows per (claim, condition)."""
    counts = Counter((sample["claim_id"], sample["manipulation"]) for sample in samples)
    short = sorted(key for key, count in counts.items() if count != k)
    if short:
        shown = ", ".join(f"{cid}/{cond}={counts[(cid, cond)]}" for cid, cond in short[:5])
        raise DataError(
            f"{len(short)} (claim, condition) cell(s) do not hold {k} result rows: {shown}"
        )


def c0_accuracy_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accuracy = tables.c0_accuracy(samples)
    return [{"class": label, **accuracy[label]} for label in tables.LABELS]


def included_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = tables.included_claims(samples)
    rows = [
        {
            "class": label,
            "included": counts["included"][label],
            "excluded_no_majority": counts["excluded_no_majority"][label],
            "excluded_wrong_majority": counts["excluded_wrong_majority"][label],
            "claims": counts["claims"][label],
        }
        for label in tables.LABELS
    ]
    rows.append(
        {
            "class": "TOTAL",
            "included": sum(counts["included"].values()),
            "excluded_no_majority": counts["excluded_no_majority"].total,
            "excluded_wrong_majority": counts["excluded_wrong_majority"].total,
            "claims": sum(counts["claims"].values()),
        }
    )
    return rows


def key_cell_rows(
    samples: list[dict[str, Any]], points: dict[str, Any]
) -> list[dict[str, Any]]:
    table = tables.key_cell_table(samples, points)
    rows: list[dict[str, Any]] = []
    for condition in sorted(table):
        for label in sorted(table[condition]):
            cell = table[condition][label]
            for bin_name in ("low", "mid", "high"):
                values = cell["bins"][bin_name]
                rows.append(
                    {
                        "condition": condition,
                        "class": label,
                        "bin": bin_name,
                        "n": values["n"],
                        "not_as_expected": values["not_as_expected"],
                        "rate": values["p"],
                        "no_verdict": cell["no_verdict"],
                        "samples": cell["samples"],
                    }
                )
    return rows


def sensitivity_no_verdict_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sensitivity table 1: no_verdict samples counted as not_as_expected."""
    included = tables.included_claim_ids(samples)
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"high_n": 0, "high_not_as_expected": 0, "samples": 0,
                 "with_verdict": 0, "no_verdict": 0, "not_as_expected": 0}
    )
    for sample in samples:
        if sample["claim_id"] not in included:
            continue
        group = groups[(sample["manipulation"], sample["class"])]
        group["samples"] += 1
        if sample["model_verdict"] is None:
            group["no_verdict"] += 1
            continue
        group["with_verdict"] += 1
        if sample["as_expected"] is False:
            group["not_as_expected"] += 1
        if sample.get("bin") == "high":
            group["high_n"] += 1
            if sample["as_expected"] is False:
                group["high_not_as_expected"] += 1
    rows: list[dict[str, Any]] = []
    for (condition, label), group in sorted(groups.items()):
        rows.append(
            {
                "condition": condition,
                "class": label,
                "high_n": group["high_n"],
                "high_not_as_expected": group["high_not_as_expected"],
                "high_rate": (
                    group["high_not_as_expected"] / group["high_n"]
                    if group["high_n"]
                    else None
                ),
                "samples": group["samples"],
                "with_verdict": group["with_verdict"],
                "no_verdict": group["no_verdict"],
                "not_as_expected": group["not_as_expected"],
                "rate_excluding_no_verdict": (
                    group["not_as_expected"] / group["with_verdict"]
                    if group["with_verdict"]
                    else None
                ),
                "not_as_expected_with_no_verdict": (
                    group["not_as_expected"] + group["no_verdict"]
                ),
                "rate_with_no_verdict": (
                    (group["not_as_expected"] + group["no_verdict"]) / group["samples"]
                    if group["samples"]
                    else None
                ),
            }
        )
    return rows


def sensitivity_high_90_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sensitivity table 2: fixed threshold, high = confidence >= 90."""
    included = tables.included_claim_ids(samples)
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"high_n": 0, "high_not_as_expected": 0,
                 "below_n": 0, "below_not_as_expected": 0}
    )
    for sample in samples:
        if sample["claim_id"] not in included or sample["model_verdict"] is None:
            continue
        confidence = sample.get("confidence")
        if confidence is None:
            continue
        group = groups[(sample["manipulation"], sample["class"])]
        if confidence >= cutpoints.SENSITIVITY_HIGH:
            group["high_n"] += 1
            if sample["as_expected"] is False:
                group["high_not_as_expected"] += 1
        else:
            group["below_n"] += 1
            if sample["as_expected"] is False:
                group["below_not_as_expected"] += 1
    rows: list[dict[str, Any]] = []
    for (condition, label), group in sorted(groups.items()):
        rows.append(
            {
                "condition": condition,
                "class": label,
                "threshold": f"high >= {cutpoints.SENSITIVITY_HIGH}",
                "high_n": group["high_n"],
                "high_not_as_expected": group["high_not_as_expected"],
                "high_rate": (
                    group["high_not_as_expected"] / group["high_n"]
                    if group["high_n"]
                    else None
                ),
                "below_n": group["below_n"],
                "below_not_as_expected": group["below_not_as_expected"],
                "below_rate": (
                    group["below_not_as_expected"] / group["below_n"]
                    if group["below_n"]
                    else None
                ),
            }
        )
    return rows


def analyse(
    log_path: Path,
    out_dir: Path,
    config_path: Path,
    *,
    seed: int | None = None,
    n_resamples: int = contrasts.N_RESAMPLES,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    section = config.get("run") or {}
    k = int(section.get("k", 5))
    gold_path = root / "data" / "gold.jsonl"
    conditions_dir = root / "data" / "conditions"
    flags_path = conditions_dir / "flags.jsonl"
    audit_dir = root / "audit"
    samples = loaders.load_samples(
        Path(log_path),
        gold_path,
        conditions_dir,
        flags_path if flags_path.is_file() else None,
    )
    assert_complete_cells(samples, k)
    points = cutpoints.write_cutpoints(samples, out)
    cutpoints.apply_bins(samples, points)
    loaders.write_long_table(samples, out / "long.csv")
    _write_csv(
        out / "c0_accuracy.csv",
        c0_accuracy_rows(samples),
        ["class", "claims", "correct", "accuracy"],
    )
    _write_csv(
        out / "included_claims.csv",
        included_rows(samples),
        ["class", "included", "excluded_no_majority", "excluded_wrong_majority", "claims"],
    )
    _write_csv(
        out / "key_cell.csv",
        key_cell_rows(samples, points),
        ["condition", "class", "bin", "n", "not_as_expected", "rate", "no_verdict", "samples"],
    )
    _write_csv(
        out / "sensitivity_no_verdict.csv",
        sensitivity_no_verdict_rows(samples),
        ["condition", "class", "high_n", "high_not_as_expected", "high_rate", "samples",
         "with_verdict", "no_verdict", "not_as_expected", "rate_excluding_no_verdict",
         "not_as_expected_with_no_verdict", "rate_with_no_verdict"],
    )
    _write_csv(
        out / "sensitivity_high_90.csv",
        sensitivity_high_90_rows(samples),
        ["condition", "class", "threshold", "high_n", "high_not_as_expected", "high_rate",
         "below_n", "below_not_as_expected", "below_rate"],
    )
    analysis_seed = int(config["seed"] if seed is None else seed)
    contrast_rows = contrasts.run_contrasts(samples, analysis_seed, n_resamples=n_resamples)
    contrasts.write_contrasts(contrast_rows, out)
    strata_data = strata.write_strata(
        samples, out, audit_dir=audit_dir if audit_dir.is_dir() else None
    )
    report.write_summary(contrast_rows, strata_data, out)
    return {
        "log": str(Path(log_path)),
        "out": str(out),
        "samples": len(samples),
        "claims": len({sample["claim_id"] for sample in samples}),
        "cutpoints": points,
        "contrasts": len(contrast_rows),
        "seed": analysis_seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="one explicit result log (JSONL)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=_ROOT / "config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=contrasts.N_RESAMPLES)
    args = parser.parse_args(argv)
    try:
        summary = analyse(
            args.log, args.out, args.config,
            seed=args.seed, n_resamples=args.n_resamples,
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
