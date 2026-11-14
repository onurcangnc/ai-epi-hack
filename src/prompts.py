"""Loading and rendering of the frozen verifier prompts.

The prompts are human-owned artifacts: this module never edits them. It only
reads bytes, hashes them, and fills the user template. Substitution is a single
pass over the template, so a ``{evidence}`` sequence inside the claim text is
never substituted a second time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SYSTEM_FILENAME = "verifier_system.txt"
USER_FILENAME = "verifier_user.txt"

_PLACEHOLDER = re.compile(r"\{(claim|evidence)\}")


class PromptError(ValueError):
    """Raised when a required prompt file is missing, unreadable or unfillable."""


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user_template: str
    system_sha256: str
    user_sha256: str


def _read_text(path: Path) -> tuple[bytes, str]:
    if not path.is_file():
        raise PromptError(f"missing prompt file: {path}")
    raw = path.read_bytes()
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptError(f"prompt file is not UTF-8: {path}: {exc}") from exc


def load_prompts(prompt_dir: Path) -> PromptBundle:
    prompt_dir = Path(prompt_dir)
    system_bytes, system = _read_text(prompt_dir / SYSTEM_FILENAME)
    user_bytes, user_template = _read_text(prompt_dir / USER_FILENAME)
    return PromptBundle(
        system=system,
        user_template=user_template,
        system_sha256=hashlib.sha256(system_bytes).hexdigest(),
        user_sha256=hashlib.sha256(user_bytes).hexdigest(),
    )


def render_user(template: str, claim: str, evidence: str) -> str:
    """Fill the user template with placeholders in a single pass."""
    values = {"claim": claim, "evidence": evidence}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def render_evidence(sentences: Iterable[dict[str, Any]]) -> str:
    """Normalise evidence for ALL conditions: strip() each sentence, join with one space.

    Every element must be a mapping with a non-blank string ``text``; anything
    else is a data error rather than silently dropped evidence.
    """
    parts: list[str] = []
    for index, sentence in enumerate(sentences):
        text = sentence.get("text") if isinstance(sentence, dict) else None
        if not isinstance(text, str):
            raise PromptError(
                f"evidence sentence {index} has no string 'text': {type(text).__name__}"
            )
        stripped = text.strip()
        if not stripped:
            raise PromptError(f"evidence sentence {index} is blank")
        parts.append(stripped)
    return " ".join(parts)
