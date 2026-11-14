"""TASK A2 official smoke rerun.

Runs 6 claims disjoint from the main 150 x 5 samples x T in {0.7, 1.0} through
the normal VerifierClient: frozen prompts, condition C0, config model/settings,
disk cache in smoke/cache/, JSONL logs in smoke/raw/. No wrappers or monkey
patching; finish_reason comes from the client's own attempt rows.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from backend import DeepSeekBackend  # noqa: E402
from common import read_jsonl  # noqa: E402
from prompts import load_prompts  # noqa: E402
from verifier_client import (  # noqa: E402
    VerifierClient,
    api_key_from_env,
    load_model_config,
)

SMOKE = ROOT / "smoke"
RAW = SMOKE / "raw"
CACHE = SMOKE / "cache"
CLAIMS = read_jsonl(SMOKE / "claims.jsonl")


def run_phase(config, phase: str, temperature: float) -> dict:
    cfg = dataclasses.replace(config, temperature=temperature)
    backend = DeepSeekBackend(
        base_url=cfg.base_url,
        api_key=api_key_from_env(cfg),
        model=cfg.model,
        timeout=120.0,
    )
    prompts = load_prompts(ROOT / "prompts")
    log_path = RAW / f"{phase}.jsonl"
    if log_path.exists():
        log_path.unlink()
    client = VerifierClient(cfg, backend, prompts, log_path, cache_dir=CACHE)

    started = time.monotonic()
    rows = []
    for record in CLAIMS:
        for sample_idx in range(5):
            rows.append(
                client.verify(
                    claim_id=record["id"],
                    claim=record["claim"],
                    evidence_sentences=record["evidence_sentences"],
                    sample_idx=sample_idx,
                    condition="C0",
                )
            )
    elapsed = time.monotonic() - started

    tin = sum(row["tokens"]["input"] or 0 for row in rows)
    tout = sum(row["tokens"]["output"] or 0 for row in rows)
    return {
        "phase": phase,
        "temperature": temperature,
        "calls": len(rows),
        "input": tin,
        "output": tout,
        "total": tin + tout,
        "elapsed_s": round(elapsed, 1),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "cached": sum(1 for row in rows if row["cached"]),
    }


def main() -> int:
    config = load_model_config(ROOT / "config.yaml")
    CACHE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    stats = {}
    for phase, temperature in (("temp_0.7", 0.7), ("temp_1.0", 1.0)):
        stats[phase] = run_phase(config, phase, temperature)
        print(json.dumps(stats[phase], indent=2))
    (SMOKE / "run_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
