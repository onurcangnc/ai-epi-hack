"""Probe DeepSeek for temperature x thinking support.

NOT RUN by the agent. It makes a small number of trivial calls and prints a
table of which combinations are accepted, silently ignored, or rejected.

Method (per combination):
1. Send a trivial prompt with the target temperature.
   - HTTP 4xx -> "rejected" for the combination.
2. If accepted, send the same prompt with an out-of-range temperature (3.0).
   - If that is also accepted, the API is not enforcing temperature at all in
     this mode -> "silently ignored".
   - If it is rejected, temperature is validated -> "accepted".

This is a model of API behaviour, not a proof of sampling behaviour; the
per-cell notes record what was observed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT / "src", _ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from backend import (  # noqa: E402
    BackendError,
    BackendTimeout,
    DeepSeekBackend,
    RateLimitError,
    ServerError,
    ThinkingConfig,
)
from verifier_client import (  # noqa: E402
    ConfigError,
    api_key_from_env,
    load_model_config,
)

SYSTEM = "You are a test endpoint. Reply with the single word OK."
USER = "Reply OK."


@dataclass
class Row:
    temperature: float
    thinking: str
    call_status: str
    enforcement_probe: str
    conclusion: str
    detail: str


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, ServerError):
        return "server_error"
    if isinstance(exc, BackendTimeout):
        return "timeout"
    if isinstance(exc, BackendError):
        return "rejected"
    raise exc


def run(config_path: Path) -> list[Row]:
    config = load_model_config(config_path)
    backend = DeepSeekBackend(
        base_url=config.base_url,
        api_key=api_key_from_env(config),
        model=config.model,
        timeout=60.0,
    )
    rows: list[Row] = []
    for thinking_enabled in (False, True):
        thinking = ThinkingConfig(enabled=thinking_enabled)
        for temperature in (0.0, 0.7):
            call_status = "accepted"
            detail = ""
            try:
                backend.complete(
                    SYSTEM,
                    USER,
                    temperature=temperature,
                    max_tokens=16,
                    thinking=thinking,
                )
            except Exception as exc:  # noqa: BLE001 - classification is the point
                call_status = _classify_exception(exc)
                detail = str(exc)[:200]

            enforcement = "not_run"
            if call_status == "accepted":
                try:
                    backend.complete(
                        SYSTEM,
                        USER,
                        temperature=3.0,
                        max_tokens=16,
                        thinking=thinking,
                    )
                    enforcement = "accepted"
                except Exception as exc:  # noqa: BLE001
                    enforcement = _classify_exception(exc)

            if call_status != "accepted":
                conclusion = "rejected"
            elif enforcement == "accepted":
                conclusion = "silently ignored"
            elif enforcement == "rejected":
                conclusion = "accepted"
            else:
                conclusion = "unknown"

            rows.append(
                Row(
                    temperature=temperature,
                    thinking="on" if thinking_enabled else "off",
                    call_status=call_status,
                    enforcement_probe=enforcement,
                    conclusion=conclusion,
                    detail=detail,
                )
            )
    return rows


def render_table(rows: list[Row]) -> str:
    lines = [
        "| temperature | thinking | normal call | out-of-range probe | conclusion |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.temperature} | {row.thinking} | {row.call_status} | "
            f"{row.enforcement_probe} | {row.conclusion} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path for the JSON rows",
    )
    args = parser.parse_args(argv)
    try:
        rows = run(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_table(rows))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([asdict(row) for row in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
