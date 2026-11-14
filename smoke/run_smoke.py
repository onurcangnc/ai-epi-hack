"""TASK A2 smoke runner (harness, not project code).

Runs 6 disjoint SciFact claims x 5 samples x 2 temperatures through the real
VerifierClient with the frozen prompts and condition C0. Adds only observation:
captures finish_reason per HTTP call and dumps the client's in-memory cache to
smoke/cache/. Project modules and prompts are imported, never modified.
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
from common import read_jsonl, write_jsonl  # noqa: E402
from prompts import load_prompts  # noqa: E402
from verifier_client import VerifierClient, api_key_from_env, load_model_config  # noqa: E402

SMOKE = ROOT / "smoke"
RAW = SMOKE / "raw"
CACHE = SMOKE / "cache"
CLAIMS = read_jsonl(SMOKE / "claims.jsonl")


def run_phase(config, phase: str, temperature: float):
    cfg = dataclasses.replace(config, temperature=temperature, max_tokens=512)
    backend = DeepSeekBackend(
        base_url=cfg.base_url,
        api_key=api_key_from_env(cfg),
        model=cfg.model,
        timeout=120.0,
    )
    calls: list[dict] = []
    last: dict = {}

    orig_post = backend.session.post

    def post(*args, **kwargs):
        response = orig_post(*args, **kwargs)
        last["status"] = response.status_code
        last["finish_reason"] = None
        if response.status_code == 200:
            try:
                last["finish_reason"] = response.json()["choices"][0].get(
                    "finish_reason"
                )
            except Exception:  # noqa: BLE001
                pass
        return response

    backend.session.post = post

    orig_complete = backend.complete

    def complete(*args, **kwargs):
        last.clear()
        try:
            result = orig_complete(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            calls.append(
                {
                    "ok": False,
                    "http_status": last.get("status"),
                    "finish_reason": None,
                    "input": None,
                    "output": None,
                    "latency_ms": None,
                    "error": type(exc).__name__,
                }
            )
            raise
        calls.append(
            {
                "ok": True,
                "http_status": last.get("status"),
                "finish_reason": last.get("finish_reason"),
                "input": result.input_tokens,
                "output": result.output_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "error": None,
            }
        )
        return result

    backend.complete = complete
    prompts = load_prompts(ROOT / "prompts")
    log_path = RAW / f"{phase}.jsonl"
    if log_path.exists():
        log_path.unlink()
    client = VerifierClient(cfg, backend, prompts, log_path)

    results = []
    for rec in CLAIMS:
        samples = []
        for sample_idx in range(5):
            mark = len(calls)
            row = client.verify(
                claim_id=rec["id"],
                claim=rec["claim"],
                evidence_sentences=rec["evidence_sentences"],
                sample_idx=sample_idx,
                condition="C0",
            )
            samples.append({"sample_idx": sample_idx, "row": row, "calls": calls[mark:]})
        results.append({"claim": rec, "samples": samples})

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_dump = {
        "phase": phase,
        "temperature": temperature,
        "entries": client._cache,
    }
    (CACHE / f"{phase}.cache.json").write_text(
        json.dumps(cache_dump, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results, calls


def summarize(results, calls) -> dict:
    tin = sum(c["input"] or 0 for c in calls)
    tout = sum(c["output"] or 0 for c in calls)
    return {
        "http_calls": len(calls),
        "ok_calls": sum(1 for c in calls if c["ok"]),
        "input": tin,
        "output": tout,
        "total": tin + tout,
        "finish": dict(Counter(c["finish_reason"] for c in calls)),
    }


def main() -> int:
    config = load_model_config(ROOT / "config.yaml")
    all_stats = {}
    all_results = {}
    for phase, temperature in (("temp_0.7", 0.7), ("temp_1.0", 1.0)):
        started = time.monotonic()
        results, calls = run_phase(config, phase, temperature)
        elapsed = time.monotonic() - started
        stats = summarize(results, calls)
        stats["elapsed_s"] = round(elapsed, 1)
        all_stats[phase] = stats
        all_results[phase] = (results, calls)

    print(json.dumps(all_stats, indent=2))
    (SMOKE / "run_stats.json").write_text(
        json.dumps(all_stats, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
