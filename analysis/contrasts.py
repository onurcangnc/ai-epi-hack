"""Paired bootstrap contrasts (SPEC v1.4 §6), fixture-safe.

The filter is recomputed inside each resample (a claim is included only when its
own C0 majority equals gold); cut points are fixed before the bootstrap.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis import tables

PRIMARY = (("M1_vs_C0p", "M1", "C0p"), ("M2_vs_C2", "M2", "C2"))
SECONDARY = (("M2_vs_C0p", "M2", "C0p"),)
N_RESAMPLES = 10000
ALPHA = 0.05


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _betai(a, b, mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    low = 0.0 if k == 0 else beta_ppf(alpha / 2.0, float(k), float(n - k + 1))
    high = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, float(k + 1), float(n - k))
    return low, high


def _gold_verdict(label: str) -> str:
    return {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "NEI": "NOT_ENOUGH_INFO"}[label]


def c0_inclusion(samples: list[dict[str, Any]]) -> dict[str, bool]:
    gold = {s["claim_id"]: s["class"] for s in samples if s["manipulation"] == "C0"}
    verdicts: dict[str, list[str | None]] = defaultdict(list)
    for s in samples:
        if s["manipulation"] == "C0":
            verdicts[s["claim_id"]].append(s["model_verdict"])
    return {
        claim_id: tables.majority(values) == _gold_verdict(label)
        for claim_id, label in gold.items()
        for values in [verdicts.get(claim_id, [])]
    }


def cell_counts(
    samples: list[dict[str, Any]],
    included: dict[str, bool],
    condition: str,
    label: str,
) -> tuple[int, int]:
    numerator = denominator = 0
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] != label:
            continue
        if not included.get(sample["claim_id"], False):
            continue
        if sample.get("bin") != "high" or sample["model_verdict"] is None:
            continue
        denominator += 1
        if sample["as_expected"] is False:
            numerator += 1
    return numerator, denominator


def _per_claim_counts(samples, included, condition, label):
    per_claim: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] != label:
            continue
        if not included.get(sample["claim_id"], False):
            continue
        if sample.get("bin") != "high" or sample["model_verdict"] is None:
            continue
        num, den = per_claim[sample["claim_id"]]
        per_claim[sample["claim_id"]] = (
            num + (1 if sample["as_expected"] is False else 0),
            den + 1,
        )
    return per_claim


def _bootstrap_delta(
    claims: list[str],
    per_m: dict[str, tuple[int, int]],
    per_c: dict[str, tuple[int, int]],
    seed: int,
    n_resamples: int,
) -> list[float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        num_m = den_m = num_c = den_c = 0
        for _ in range(len(claims)):
            claim = claims[rng.randrange(len(claims))]
            a, b = per_m.get(claim, (0, 0))
            num_m += a
            den_m += b
            a, b = per_c.get(claim, (0, 0))
            num_c += a
            den_c += b
        if den_m and den_c:
            deltas.append(num_m / den_m - num_c / den_c)
    return deltas


def _holm(primary_rows: list[dict[str, Any]]) -> None:
    """Holm correction across the two primary contrasts, within one class."""
    p_values = [(row["p_value"], row) for row in primary_rows if row.get("p_value") is not None]
    p_values.sort(key=lambda item: item[0])
    m = len(p_values)
    previous = 0.0
    for index, (p_value, row) in enumerate(p_values):
        adjusted = min(1.0, max(previous, (m - index) * p_value))
        row["p_holm"] = adjusted
        previous = adjusted


def run_contrasts(
    samples: list[dict[str, Any]],
    seed: int,
    *,
    n_resamples: int = N_RESAMPLES,
) -> list[dict[str, Any]]:
    included = c0_inclusion(samples)
    claim_ids = sorted(included)
    rows: list[dict[str, Any]] = []
    for class_name in tables.LABELS:
        primary_rows: list[dict[str, Any]] = []
        for name, manip, control in PRIMARY + SECONDARY:
            if not any(
                sample["manipulation"] in (manip, control) and sample["class"] == class_name
                for sample in samples
            ):
                continue
            num_m, den_m = cell_counts(samples, included, manip, class_name)
            num_c, den_c = cell_counts(samples, included, control, class_name)
            per_m = _per_claim_counts(samples, included, manip, class_name)
            per_c = _per_claim_counts(samples, included, control, class_name)
            contributing = sorted(set(per_m) | set(per_c))
            row: dict[str, Any] = {
                "contrast": name,
                "class": class_name,
                "manipulation": manip,
                "control": control,
                "n_claims": len(contributing),
                "n_manipulation": den_m,
                "not_as_expected_manipulation": num_m,
                "n_control": den_c,
                "not_as_expected_control": num_c,
                "estimate": (num_m / den_m if den_m else None) - (num_c / den_c if den_c else None)
                if den_m and den_c else None,
                "method": "clopper_pearson" if len(contributing) < 10 else "bootstrap",
                "ci_low": None,
                "ci_high": None,
                "p_value": None,
                "p_holm": None,
                "cp_low_manipulation": None,
                "cp_high_manipulation": None,
                "cp_low_control": None,
                "cp_high_control": None,
            }
            if len(contributing) < 10:
                row["cp_low_manipulation"], row["cp_high_manipulation"] = clopper_pearson(num_m, den_m)
                row["cp_low_control"], row["cp_high_control"] = clopper_pearson(num_c, den_c)
            else:
                deltas = _bootstrap_delta(claim_ids, per_m, per_c, seed, n_resamples)
                if deltas:
                    row["ci_low"] = float(np.percentile(deltas, 2.5))
                    row["ci_high"] = float(np.percentile(deltas, 97.5))
                    below = sum(1 for delta in deltas if delta <= 0.0) / len(deltas)
                    above = sum(1 for delta in deltas if delta >= 0.0) / len(deltas)
                    row["p_value"] = min(1.0, 2.0 * min(below, above))
                row["estimate"] = (num_m / den_m if den_m else 0.0) - (num_c / den_c if den_c else 0.0)
            rows.append(row)
            if name in {primary[0] for primary in PRIMARY}:
                primary_rows.append(row)
        _holm(primary_rows)
    return rows


def write_contrasts(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contrasts.csv"
    columns = [
        "contrast", "class", "manipulation", "control", "n_claims",
        "n_manipulation", "not_as_expected_manipulation", "n_control",
        "not_as_expected_control", "estimate", "method", "ci_low", "ci_high",
        "p_value", "p_holm", "cp_low_manipulation", "cp_high_manipulation",
        "cp_low_control", "cp_high_control",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return path
