from __future__ import annotations

import pytest

from common import (
    VERDICTS,
    DataError,
    expected_verdict,
    normalize_gold,
)


def test_verdicts_vocabulary():
    assert VERDICTS == ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


@pytest.mark.parametrize(
    "gold, verdict",
    [("SUPPORT", "SUPPORTED"), ("REFUTE", "REFUTED"), ("NEI", "NOT_ENOUGH_INFO")],
)
def test_normalize_gold(gold, verdict):
    assert normalize_gold(gold) == verdict


def test_normalize_gold_unknown_raises():
    with pytest.raises(DataError, match="unknown gold label"):
        normalize_gold("MAYBE")


@pytest.mark.parametrize(
    "condition, gold, expected",
    [
        ("C0", "SUPPORT", "SUPPORTED"),
        ("C1", "REFUTE", "REFUTED"),
        ("M2", "NEI", "NOT_ENOUGH_INFO"),
        ("M1", "SUPPORT", "NOT_ENOUGH_INFO"),  # M1 case for a SUPPORT claim
        ("C_closed", "REFUTE", "NOT_ENOUGH_INFO"),
    ],
)
def test_expected_verdict(condition, gold, expected):
    assert expected_verdict(condition, gold) == expected


@pytest.mark.parametrize("condition", ["M3", "M4", "M5"])
def test_expected_verdict_pending_raises(condition):
    with pytest.raises(NotImplementedError):
        expected_verdict(condition, "SUPPORT")


def test_expected_verdict_unknown_condition_raises():
    with pytest.raises(DataError, match="unknown condition"):
        expected_verdict("C9", "SUPPORT")
