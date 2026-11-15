"""Experiment runner for the DeepSeek matrix (SPEC §3, §5, §6, §7).

Dry-run by default. A real run needs --confirm and every frozen-state guard.
The runner never prints prompts, keys or ids, never deletes a log, and writes
only to results/raw/ and the configured cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT / "scripts", _ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from backend import DeepSeekBackend  # noqa: E402
from common import DataError, read_jsonl  # noqa: E402
from escaping import escape_tags  # noqa: E402
from prompts import PromptBundle, PromptError, render_evidence, render_user  # noqa: E402
from verifier_client import (  # noqa: E402
    ConfigError,
    RunAborted,
    StateError,
    VerifierClient,
    api_key_from_env,
    load_cache_dir,
    load_model_config,
    load_prompts,
)

NAIVE_SYSTEM = "verifier_system_naive.txt"
WARNED_LOG = Path("results/raw/deepseek_warned.jsonl")
NAIVE_LOG = Path("results/raw/deepseek_naive.jsonl")
PROGRESS_EVERY = 50
WINDOW = 100
MAX_API_ERRORS_IN_WINDOW = 20


class GuardError(RuntimeError):
    """A frozen-state guard failed; a real run must not start."""


# --------------------------------------------------------------------------- #
# Config and units
# --------------------------------------------------------------------------- #
def run_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("run")
    if not isinstance(section, dict):
        raise ConfigError("config key 'run' is missing")
    for key in ("conditions_core", "conditions_extended", "k", "order_seed", "tag_required", "shards"):
        if key not in section:
            raise ConfigError(f"config key 'run.{key}' is missing")
    if not isinstance(section["k"], int) or section["k"] < 1:
        raise ConfigError("config key 'run.k' must be a positive integer")
    if not isinstance(section["shards"], int) or section["shards"] < 1:
        raise ConfigError("config key 'run.shards' must be a positive integer")
    for key in ("conditions_core", "conditions_extended"):
        if not isinstance(section[key], list) or not all(isinstance(c, str) for c in section[key]):
            raise ConfigError(f"config key 'run.{key}' must be a list of condition codes")
    return section


def resolve_conditions(config: dict[str, Any], tier: str, override: list[str] | None = None) -> list[str]:
    section = run_config(config)
    if tier == "core":
        conditions = list(section["conditions_core"])
    elif tier == "extended":
        conditions = list(section["conditions_extended"])
    elif tier == "naive":
        conditions = ["C0", "M2"]
    else:
        raise ConfigError(f"unknown tier {tier!r}")
    if override is not None:
        if tier == "naive" and set(override) != {"C0", "M2"}:
            raise ConfigError("naive tier runs only C0 and M2")
        conditions = list(override)
    return conditions


def condition_path(root: Path, code: str) -> Path:
    base = root / "data" / "conditions"
    return base / ("C0.jsonl" if code == "C0p" else f"{code}.jsonl")


def build_units(root: Path, config: dict[str, Any], tier: str) -> list[dict[str, Any]]:
    k = int(run_config(config)["k"])
    units: list[dict[str, Any]] = []
    for code in resolve_conditions(config, tier):
        rows = read_jsonl(condition_path(root, code))
        for row in rows:
            evidence = row["evidence_sentences"]
            for sample_idx in range(k):
                units.append(
                    {
                        "claim_id": row["id"],
                        "claim": row["claim"],
                        "evidence_sentences": evidence,
                        "sample_idx": sample_idx,
                        "condition": code if code != "C0p" else "C0p",
                    }
                )
    return units


def ordered_units(units: list[dict[str, Any]], order_seed: int) -> list[dict[str, Any]]:
    if isinstance(order_seed, bool) or not isinstance(order_seed, int):
        raise ConfigError("run.order_seed must be an integer before ordering units")
    ordered = [dict(unit) for unit in units]
    random.Random(order_seed).shuffle(ordered)
    return ordered


def shard_slice(units: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    if count < 1 or not 1 <= index <= count:
        raise ConfigError(f"invalid shard {index}/{count}")
    start = (index - 1) * len(units) // count
    end = index * len(units) // count
    return units[start:end]


def parse_shard(text: str | None, default_count: int) -> tuple[int, int]:
    if text is None:
        return 1, default_count
    match = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not match:
        raise ConfigError(f"invalid --shard {text!r}; expected i/n")
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------- #
# Prompts and planning
# --------------------------------------------------------------------------- #
def prompt_bundle(root: Path, tier: str) -> tuple[PromptBundle, bool]:
    prompts_dir = root / "prompts"
    if tier != "naive":
        return load_prompts(prompts_dir), False
    system_path = prompts_dir / NAIVE_SYSTEM
    user_path = prompts_dir / "verifier_user.txt"
    if not user_path.is_file():
        raise PromptError(f"missing prompt file: {user_path}")
    user_bytes = user_path.read_bytes()
    user_template = user_bytes.decode("utf-8")
    if not system_path.is_file():
        return (
            PromptBundle(
                system="",
                user_template=user_template,
                system_sha256="",
                user_sha256=hashlib.sha256(user_bytes).hexdigest(),
            ),
            True,
        )
    system_bytes = system_path.read_bytes()
    return (
        PromptBundle(
            system=system_bytes.decode("utf-8"),
            user_template=user_template,
            system_sha256=hashlib.sha256(system_bytes).hexdigest(),
            user_sha256=hashlib.sha256(user_bytes).hexdigest(),
        ),
        False,
    )


def log_path_for(root: Path, tier: str) -> Path:
    return root / (NAIVE_LOG if tier == "naive" else WARNED_LOG)


def cache_dir_for(root: Path, config_path: Path, tier: str) -> Path:
    return load_cache_dir(config_path) / tier


def prompt_chars(units: Iterable[dict[str, Any]], bundle: PromptBundle) -> int:
    total = 0
    for unit in units:
        claim_text, _ = escape_tags(unit["claim"])
        evidence_text, _ = escape_tags(render_evidence(unit["evidence_sentences"]))
        prompt = render_user(bundle.user_template, claim_text, evidence_text)
        total += len(bundle.system) + len(prompt)
    return total


def done_keys(log_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not log_path.is_file():
        return done
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("is_result"):
                done.add((str(row["claim_id"]), str(row["condition"]), int(row["sample_idx"])))
    return done


def dry_run(root: Path, config_path: Path, tier: str, shard: str | None = None) -> dict[str, Any]:
    config = load_config_for(config_path)
    section = run_config(config)
    units = build_units(root, config, tier)
    order_seed = section["order_seed"]
    index, count = parse_shard(shard, section["shards"])
    selected = units
    if order_seed is not None:
        selected = shard_slice(ordered_units(units, order_seed), index, count)
    bundle, missing_naive = prompt_bundle(root, tier)
    log_path = log_path_for(root, tier)
    per_condition: dict[str, int] = {}
    for unit in selected:
        per_condition[unit["condition"]] = per_condition.get(unit["condition"], 0) + 1
    done = done_keys(log_path)
    already = sum(
        1 for unit in selected if (unit["claim_id"], unit["condition"], unit["sample_idx"]) in done
    )
    return {
        "tier": tier,
        "shard": f"{index}/{count}",
        "units_per_condition": per_condition,
        "total_calls": len(selected),
        "already_done": already,
        "prompt_chars": None if missing_naive else prompt_chars(selected, bundle),
        "missing_naive_prompt": missing_naive,
        "log": str(log_path),
        "cache_dir": str(cache_dir_for(root, config_path, tier)),
    }


def load_config_for(config_path: Path) -> dict[str, Any]:
    from common import load_config

    return load_config(config_path)


# --------------------------------------------------------------------------- #
# Frozen-state guards
# --------------------------------------------------------------------------- #
def git_tree_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def tag_is_ancestor(root: Path, tag: str) -> bool:
    exists = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", tag],
        capture_output=True, text=True,
    )
    if exists.returncode != 0:
        return False
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", tag, "HEAD"],
        capture_output=True, text=True,
    )
    return ancestor.returncode == 0


def latest_spec_hash(changelog_text: str) -> str | None:
    match = re.search(r"SPEC SHA-256: `([0-9a-f]{64})`", changelog_text)
    return match.group(1) if match else None


def spec_hash_matches(root: Path) -> bool:
    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        return False
    expected = latest_spec_hash(changelog.read_text(encoding="utf-8"))
    actual = hashlib.sha256((root / "SPEC.md").read_bytes()).hexdigest()
    return expected == actual


def read_hash_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        digest, relative = line.split(None, 1)
        rows.append((digest.strip(), relative.strip()))
    return rows


def hash_file_matches(root: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    for digest, relative in read_hash_file(path):
        target = root / relative
        if not target.is_file():
            return False
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            return False
    return True


def hash_files_match(root: Path) -> bool:
    return hash_file_matches(root, root / "data" / "conditions" / "HASHES.txt") and hash_file_matches(
        root, root / "data" / "HASHES.txt"
    )


def prompt_hashes_match(root: Path) -> bool:
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pairs = re.findall(r"`prompts/([^`]+)`\s*\|\s*`([0-9a-f]{64})`", changelog)
    if not pairs:
        return False
    for name, digest in pairs:
        path = root / "prompts" / name
        if not path.is_file():
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def assert_guards(root: Path, config: dict[str, Any]) -> None:
    tag = run_config(config)["tag_required"]
    checks = (
        ("git tree is not clean", git_tree_clean(root)),
        (f"tag {tag!r} is missing or not an ancestor of HEAD", tag_is_ancestor(root, tag)),
        ("SPEC.md hash does not match the latest CHANGELOG value", spec_hash_matches(root)),
        ("HASHES.txt files do not match their files", hash_files_match(root)),
        ("prompt hashes do not match CHANGELOG", prompt_hashes_match(root)),
    )
    errors = [message for message, ok in checks if not ok]
    if errors:
        raise GuardError("; ".join(errors))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def api_error_window_abort(result_rows: list[dict[str, Any]]) -> bool:
    outcomes = [row.get("outcome") for row in result_rows]
    for start in range(0, max(0, len(outcomes) - WINDOW + 1)):
        window = outcomes[start : start + WINDOW]
        if sum(1 for outcome in window if outcome == "api_error") > MAX_API_ERRORS_IN_WINDOW:
            return True
    return False


def execute(
    client: VerifierClient,
    units: list[dict[str, Any]],
    *,
    order_seed: int,
    resume_passes: int = 2,
    progress_every: int = PROGRESS_EVERY,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for offset in range(0, len(units), progress_every):
        chunk = units[offset : offset + progress_every]
        batch = client.run_batch(chunk, order_seed=order_seed, resume_passes=resume_passes)
        rows.extend(batch)
        results.extend(row for row in batch if row.get("is_result"))
        if api_error_window_abort(results):
            raise RunAborted(0, "more than 20 api_error rows in 100 consecutive results")
        counts: dict[str, int] = {}
        finishes: dict[str, int] = {}
        for row in batch:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
            finishes[str(row.get("finish_reason"))] = finishes.get(str(row.get("finish_reason")), 0) + 1
        print(
            json.dumps(
                {"processed": min(offset + len(chunk), len(units)), "outcomes": counts, "finish": finishes}
            ),
            file=sys.stderr,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("core", "extended", "naive"), default="core")
    parser.add_argument("--shard", default=None, help="i/n; default is shard 1/n")
    parser.add_argument("--confirm", action="store_true", help="execute a real run")
    parser.add_argument("--config", type=Path, default=_ROOT / "config.yaml")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    root = config_path.parent
    try:
        config = load_config_for(config_path)
        section = run_config(config)
        index, count = parse_shard(args.shard, section["shards"])
        if not args.confirm:
            print(json.dumps(dry_run(root, config_path, args.tier, args.shard), indent=2))
            return 0
        if section["order_seed"] is None:
            raise ConfigError("config key 'run.order_seed' is null; set it before a real run")
        assert_guards(root, config)
        units = ordered_units(build_units(root, config, args.tier), int(section["order_seed"]))
        shard = shard_slice(units, index, count)
        bundle, missing_naive = prompt_bundle(root, args.tier)
        if missing_naive:
            raise ConfigError(f"missing prompt file: {root / 'prompts' / NAIVE_SYSTEM}")
        model = load_model_config(config_path)
        backend = DeepSeekBackend(
            base_url=model.base_url,
            api_key=api_key_from_env(model),
            model=model.model,
        )
        client = VerifierClient(
            model,
            backend,
            bundle,
            log_path_for(root, args.tier),
            cache_dir=cache_dir_for(root, config_path, args.tier),
        )
        rows = execute(client, shard, order_seed=int(section["order_seed"]))
        print(json.dumps({"tier": args.tier, "shard": f"{index}/{count}", "rows": len(rows)}))
        if count > 1:
            print("health shard complete; later shards need --shard", file=sys.stderr)
        return 0
    except (ConfigError, DataError, GuardError, StateError, RunAborted, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
