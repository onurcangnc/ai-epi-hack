"""Loading and rendering of the frozen verifier prompts.

The prompts are human-owned artifacts: this module never edits them. It only
reads bytes, hashes them, and fills the user template with ``str.replace``.
``str.format`` must not be used: the system prompt contains literal braces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SYSTEM_FILENAME = "verifier_system.txt"
USER_FILENAME = "verifier_user.txt"


class PromptError(ValueError):
    """Raised when a required prompt file is missing or unreadable."""


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
    """Fill the user template with placeholders; never use str.format."""
    return template.replace("{claim}", claim).replace("{evidence}", evidence)


def render_evidence(sentences: Iterable[dict[str, Any]]) -> str:
    """Normalise evidence for ALL conditions: strip() each sentence, join with one space.

    This is the single render path for every condition (C0, C0p, C1, C2, M1, M2,
    ...). Sentence order is preserved; a sentence that is only whitespace
    collapses to an empty field, exactly as ``strip()`` implies.
    """
    return " ".join(str(sentence["text"]).strip() for sentence in sentences)
