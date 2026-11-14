from __future__ import annotations

import pytest

from common import (
    VERDICTS,
    DataError,
    expected_verdict,
    normalize_gold,
)

GOLD_VERDICT = [
    ("SUPPORT", "SUPPORTED"),
    ("REFUTE", "REFUTED"),
    ("NEI", "NOT_ENOUGH_INFO"),
]

# C0p is the ASCII spelling of C0' (second C0 run). C0, C0p, C1, C2 and M2 all
# return the normalized gold; M1 and C_closed return NOT_ENOUGH_INFO on the two
# directional labels.
GOLD_CONDITIONS = [
    ("C0", "SUPPORT", "SUPPORTED"),
    ("C0", "REFUTE", "REFUTED"),
    ("C0", "NEI", "NOT_ENOUGH_INFO"),
    ("C0p", "SUPPORT", "SUPPORTED"),
    ("C0p", "REFUTE", "REFUTED"),
    ("C0p", "NEI", "NOT_ENOUGH_INFO"),
    ("C1", "SUPPORT", "SUPPORTED"),
    ("C1", "REFUTE", "REFUTED"),
    ("C1", "NEI", "NOT_ENOUGH_INFO"),
    ("C2", "SUPPORT", "SUPPORTED"),
    ("C2", "REFUTE", "REFUTED"),
    ("C2", "NEI", "NOT_ENOUGH_INFO"),
    ("M2", "SUPPORT", "SUPPORTED"),
    ("M2", "REFUTE", "REFUTED"),
    ("M2", "NEI", "NOT_ENOUGH_INFO"),
]

DIRECTIONAL_CONDITIONS = [
    ("M1", "SUPPORT", "NOT_ENOUGH_INFO"),
    ("M1", "REFUTE", "NOT_ENOUGH_INFO"),
    ("C_closed", "SUPPORT", "NOT_ENOUGH_INFO"),
    ("C_closed", "REFUTE", "NOT_ENOUGH_INFO"),
]


def test_verdicts_vocabulary():
    assert VERDICTS == ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


@pytest.mark.parametrize("gold, verdict", GOLD_VERDICT)
def test_normalize_gold(gold, verdict):
    assert normalize_gold(gold) == verdict


def test_normalize_gold_unknown_raises():
    with pytest.raises(DataError, match="unknown gold label"):
        normalize_gold("MAYBE")


@pytest.mark.parametrize(
    "condition, gold, expected", GOLD_CONDITIONS + DIRECTIONAL_CONDITIONS
)
def test_expected_verdict(condition, gold, expected):
    assert expected_verdict(condition, gold) == expected


@pytest.mark.parametrize("condition", ["M1", "C_closed"])
def test_nei_claims_have_no_row(condition):
    with pytest.raises(DataError, match="no row for NEI"):
        expected_verdict(condition, "NEI")


@pytest.mark.parametrize("condition", ["M1", "C_closed"])
def test_nei_guard_still_rejects_unknown_gold(condition):
    with pytest.raises(DataError, match="unknown gold label"):
        expected_verdict(condition, "MAYBE")


def test_c0_prime_apostrophe_is_not_a_condition():
    # The code spelling is C0p everywhere; the typographic C0' must not slide in.
    with pytest.raises(DataError, match="unknown condition"):
        expected_verdict("C0'", "SUPPORT")


@pytest.mark.parametrize("condition", ["M3", "M4", "M5"])
def test_expected_verdict_pending_raises(condition):
    with pytest.raises(NotImplementedError):
        expected_verdict(condition, "SUPPORT")


def test_expected_verdict_unknown_condition_raises():
    with pytest.raises(DataError, match="unknown condition"):
        expected_verdict("C9", "SUPPORT")
