"""Strict parsing of verifier responses.

Only two things are tolerated beyond a bare JSON object: surrounding
whitespace and a single ```json fenced block around the whole response.
Everything else is a parse failure, and a parse failure never becomes a verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_ROOT / "scripts"))

from common import VERDICTS  # noqa: E402

REQUIRED_FIELDS = ("verdict", "confidence", "rationale", "evidence_concerns")


class ParseError(ValueError):
    """Raised when a model response is not a valid verifier answer."""


def strip_fence(raw: str) -> str:
    """Strip exactly one optional ```json fence; leave everything else alone."""
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json") : -len("```")].strip()
    return text


def _confidence(value: Any) -> int:
    if isinstance(value, bool):
        raise ParseError("confidence must not be a boolean")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        raise ParseError(
            f"confidence must be an integer, got {type(value).__name__}: {value!r}"
        )
    if not 0 <= number <= 100:
        raise ParseError(f"confidence out of range: {number}")
    return number


def parse_response(raw: str) -> dict[str, Any]:
    """Parse and validate one verifier response."""
    text = strip_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"expected one JSON object, got {type(payload).__name__}")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ParseError(f"missing required field(s): {', '.join(missing)}")

    verdict = payload["verdict"]
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        raise ParseError(f"verdict must be one of {VERDICTS}, got {verdict!r}")

    rationale = payload["rationale"]
    if not isinstance(rationale, str):
        raise ParseError(f"rationale must be a string, got {type(rationale).__name__}")

    concerns = payload["evidence_concerns"]
    if not isinstance(concerns, list) or not all(
        isinstance(item, str) for item in concerns
    ):
        raise ParseError("evidence_concerns must be a list of strings")

    return {
        "verdict": verdict,
        "confidence": _confidence(payload["confidence"]),
        "rationale": rationale,
        "evidence_concerns": list(concerns),
    }
