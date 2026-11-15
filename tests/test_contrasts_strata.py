"""Fixture-only tests for analysis/contrasts.py, strata.py and report.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import contrasts, report, strata  # noqa: E402


def sample(
    claim_id: str,
    manipulation: str,
    sample_idx: int,
    verdict: str | None,
    expected: str,
    *,
    confidence: int | None = 90,
    bin_name: str | None = "high",
    class_name: str = "SUPPORT",
    as_expected: bool | None = None,
    outcome_column: str | None = None,
    outcome: str = "ok",
    finish_reason: str = "stop",
    raw_response: str | None = None,
    flags: dict | None = None,
    meta: dict | None = None,
) -> dict:
    if as_expected is None:
        as_expected = None if verdict is None else verdict == expected
    if outcome_column is None:
        outcome_column = "no_verdict" if verdict is None else (
            "as_expected" if as_expected else "not_as_expected"
        )
    return {
        "manipulation": manipulation,
        "claim_id": claim_id,
        "class": class_name,
        "sample_idx": sample_idx,
        "model_verdict": verdict,
        "expected_verdict": expected,
        "as_expected": as_expected,
        "confidence": confidence,
        "bin": bin_name,
        "outcome_column": outcome_column,
        "outcome": outcome,
        "finish_reason": finish_reason,
        "raw_response": raw_response,
        "manipulation_meta": meta or {},
        "c0_majority": None,
        "flags": flags or {},
    }


def dataset(claims: int = 20) -> list[dict]:
    rows: list[dict] = []
    for index in range(claims):
        claim = f"c{index}"
        for sample_idx in range(5):
            rows.append(sample(claim, "C0", sample_idx, "SUPPORTED", "SUPPORTED",
                               confidence=90, flags={"structured": True}))
            rows.append(sample(claim, "C0p", sample_idx, "SUPPORTED", "SUPPORTED",
                               confidence=90, flags={"structured": True}))
            rows.append(sample(claim, "C2", sample_idx, "SUPPORTED", "SUPPORTED",
                               confidence=90, flags={"structured": True}))
            rows.append(sample(claim, "M1", sample_idx, "REFUTED", "NOT_ENOUGH_INFO",
                               confidence=90, as_expected=False, outcome_column="changed_to_other",
                               flags={"structured": True}))
            rows.append(sample(claim, "M2", sample_idx, "REFUTED", "SUPPORTED",
                               confidence=90, as_expected=False,
                               outcome_column="followed_injection",
                               meta={"pushed_verdict": "REFUTED"},
                               flags={"structured": True, "family": "direct_command",
                                      "position": "start", "states_confidence_number": True,
                                      "template_confidence_number": 99}))
    return rows


def by_contrast(rows):
    return {(row["contrast"], row["class"]): row for row in rows}


def test_bootstrap_known_effect():
    rows = by_contrast(contrasts.run_contrasts(dataset(20), seed=7, n_resamples=2000))
    row = rows[("M1_vs_C0p", "SUPPORT")]
    assert row["estimate"] == pytest.approx(1.0)
    assert row["method"] == "bootstrap"
    assert row["n_claims"] == 20
    assert row["ci_low"] == pytest.approx(1.0)
    assert row["ci_high"] == pytest.approx(1.0)
    assert row["p_value"] == 0.0


def test_cell_with_four_claims_uses_clopper_pearson():
    rows = by_contrast(contrasts.run_contrasts(dataset(4), seed=7, n_resamples=200))
    row = rows[("M1_vs_C0p", "SUPPORT")]
    assert row["n_claims"] == 4
    assert row["method"] == "clopper_pearson"
    assert row["ci_low"] is None and row["ci_high"] is None
    assert row["cp_low_manipulation"] > 0.5
    assert row["cp_high_control"] < 0.5


def test_holm_only_for_the_two_primary_contrasts():
    rows = by_contrast(contrasts.run_contrasts(dataset(12), seed=3, n_resamples=200))
    assert rows[("M1_vs_C0p", "SUPPORT")]["p_holm"] is not None
    assert rows[("M2_vs_C2", "SUPPORT")]["p_holm"] is not None
    assert rows[("M2_vs_C0p", "SUPPORT")]["p_holm"] is None


def test_m1_strata_by_structured():
    rows = []
    for index in range(5):
        rows.append(sample(f"a{index}", "M1", 0, "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO",
                           as_expected=True, outcome_column="as_expected",
                           flags={"structured": True}))
    for index in range(5):
        rows.append(sample(f"b{index}", "M1", 0, "REFUTED", "NOT_ENOUGH_INFO",
                           as_expected=False, outcome_column="changed_to_other",
                           flags={"structured": False}))
    table = {
        (row["descriptor"], row["value"]): row
        for row in strata.m1_strata(rows)
    }
    assert table[("structured", "True")]["rate"] == 0.0
    assert table[("structured", "False")]["rate"] == 1.0
    assert table[("structured", "True")]["claims"] == 5


def test_spearman_degenerate_when_stability_is_one():
    rows = []
    for index in range(10):
        for sample_idx in range(5):
            rows.append(sample(f"c{index}", "C0", sample_idx, "SUPPORTED", "SUPPORTED",
                               confidence=90 + index))
    table = strata.stability_spearman(rows)
    c0 = [row for row in table if row["manipulation"] == "C0"][0]
    assert c0["degenerate_stability_share"] == 1.0
    assert c0["degenerate"] is True


def test_injection_confidence_share():
    rows = [
        sample("c1", "M2", 0, "REFUTED", "SUPPORTED", as_expected=False,
               outcome_column="followed_injection", confidence=99,
               flags={"states_confidence_number": True, "template_confidence_number": 99},
               meta={"pushed_verdict": "REFUTED"}),
        sample("c1", "M2", 1, "REFUTED", "SUPPORTED", as_expected=False,
               outcome_column="followed_injection", confidence=50,
               flags={"states_confidence_number": True, "template_confidence_number": 99},
               meta={"pushed_verdict": "REFUTED"}),
        sample("c1", "M2", 2, "REFUTED", "SUPPORTED", as_expected=False,
               outcome_column="followed_injection", confidence=70,
               flags={"states_confidence_number": False},
               meta={"pushed_verdict": "REFUTED"}),
    ]
    table = {row["group"]: row for row in strata.injection_confidence(rows)}
    assert table["stated_number"]["followed"] == 2
    assert table["stated_number"]["share"] == 0.5
    assert table["no_number"]["followed"] == 1


def test_pushed_verdict_regex_counts_non_ok_replies():
    rows = [
        sample("c1", "M2", 0, None, "SUPPORTED", outcome="parse_failure",
               raw_response='{"verdict": "REFUTED", "confidence": 9',
               meta={"pushed_verdict": "REFUTED"}),
        sample("c1", "M2", 1, None, "SUPPORTED", outcome="truncated",
               raw_response="no verdict here", meta={"pushed_verdict": "REFUTED"}),
    ]
    table = {(row["manipulation"], row["matched_pushed_regex"]): row for row in strata.pushed_regex_counts(rows)}
    assert table[("M2", True)]["n"] == 1
    assert table[("M2", False)]["n"] == 1


def test_outputs_are_lf_and_summary_exists(tmp_path):
    result = report.write_all(dataset(12), seed=5, out_dir=tmp_path / "out", n_resamples=200)
    assert result["contrasts"]
    expected = (
        "contrasts.csv", "strata_m1.csv", "strata_m2.csv", "injection_confidence.csv",
        "numberless_confidence.csv", "pushed_regex.csv", "spearman.csv", "summary.md",
    )
    for name in expected:
        path = tmp_path / "out" / name
        assert path.is_file(), name
        assert b"\r" not in path.read_bytes(), name
