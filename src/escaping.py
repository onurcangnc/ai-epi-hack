"""Delimiter escaping for verifier inputs.

The verifier prompt wraps claim/evidence text in XML-like tags. Any literal
tag appearing inside user-controlled text must be escaped before insertion so
that it cannot close the wrapper early. The pilot export imports this module
later, so the behaviour lives here and nowhere else.
"""

from __future__ import annotations

TAGS = ("<claim>", "</claim>", "<evidence>", "</evidence>")


def escape_tag(tag: str) -> str:
    return tag.replace("<", "&lt;").replace(">", "&gt;")


def escape_tags(text: str, tags: tuple[str, ...] = TAGS) -> tuple[str, list[str]]:
    """Escape literal wrapper tags in ``text``.

    Returns the escaped text and the list of tags that were actually found
    (in TAGS order), so callers can log every escape.
    """
    escaped: list[str] = []
    for tag in tags:
        if tag in text:
            text = text.replace(tag, escape_tag(tag))
            escaped.append(tag)
    return text, escaped
