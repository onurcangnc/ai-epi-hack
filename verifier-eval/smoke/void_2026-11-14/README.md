# VOID — TASK A2 smoke run, 2026-11-14

These artifacts are **not** the official smoke result. They were produced before Task B existed
and before the conditions on the smoke run could be applied. Kept only for audit.

- 60 calls (30 at T=0.7, 30 at T=1.0), 36,469 tokens, all `finish_reason=stop`, no retries.
- Decision table still applies to these numbers: all 6 claims show >=2 distinct confidence values
  at T=0.7, so the setting is kept at 0.7; stability is predominantly confidence-driven.
- The official run must be executed only after Task B is merged and committed, using
  `smoke/claims.jsonl` and the same settings, writing fresh `smoke/raw/` and `smoke/cache/`.
- See CHANGELOG entry "TASK A2 smoke run executed before `spec-frozen` (VOID; re-run after Task B)".
