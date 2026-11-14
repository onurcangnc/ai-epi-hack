"""Delimiter escaping for verifier inputs.

The verifier prompt wraps claim/evidence text in XML-like tags. Any literal
tag appearing inside user-controlled text must be escaped before insertion so
that it cannot close the wrapper early. Matching is case-insensitive and
tolerates whitespace inside the brackets (``<CLAIM>``, ``</ Claim >``).
"""

from __future__ import annotations

import re

TAGS = ("<claim>", "</claim>", "<evidence>", "</evidence>")

_TAG_RE = re.compile(r"<\s*(/?)\s*(claim|evidence)\s*>", re.IGNORECASE)


def escape_tag(tag: str) -> str:
    return tag.replace("<", "&lt;").replace(">", "&gt;")


def escape_tags(text: str, tags: tuple[str, ...] = TAGS) -> tuple[str, list[str]]:
    """Escape literal wrapper tags in ``text``.

    Returns the escaped text and the canonical tags that were actually found
    (in TAGS order), so callers can log every escape.
    """
    found: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        closing, name = match.group(1), match.group(2).lower()
        canonical = f"</{name}>" if closing else f"<{name}>"
        if canonical in tags:
            found.add(canonical)
            return escape_tag(match.group(0))
        return match.group(0)

    escaped_text = _TAG_RE.sub(replace, text)
    return escaped_text, [tag for tag in TAGS if tag in found]
