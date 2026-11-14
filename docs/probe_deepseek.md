# Probe — DeepSeek temperature × thinking

- Run (UTC): 2026-11-14T16:21:16Z
- Script: `scripts/probe_api.py`, run once, unmodified
- Model: `deepseek-flash` @ `https://api.deepseek.com` (OpenAI-format `/chat/completions`)
- `config.yaml` → `models.deepseek.max_tokens`: set to `512` for this task; the probe itself passes `max_tokens=16` explicitly, so the config value does not affect this run
- Probe's own definition: a cell is `accepted` when the normal call succeeds and a `temperature=3.0` attempt is rejected; `silently ignored` when the 3.0 attempt is also accepted; `rejected` when the normal call fails.

## Raw table

| temperature | thinking | normal call | out-of-range probe | conclusion |
|---|---|---|---|---|
| 0.0 | off | accepted | rejected | accepted |
| 0.7 | off | accepted | rejected | accepted |
| 0.0 | on | accepted | rejected | accepted |
| 0.7 | on | accepted | rejected | accepted |

## Per-cell observations requested

| temperature | thinking | HTTP verdict (raw table) | T=0.7 varies across 5 identical calls? | Thinking-off reasoning field? |
|---|---|---|---|---|
| 0.0 | off | accepted | n/a (variance only tested at 0.7) | No `reasoning_content`; message keys are `role`, `content` |
| 0.7 | off | accepted | No — 5/5 returned `OK` (1 distinct output) | No `reasoning_content` |
| 0.0 | on | accepted | n/a (variance only tested at 0.7) | n/a — thinking on returns `reasoning_content` |
| 0.7 | on | accepted | Yes — 2 distinct outputs (4× empty, 1× `OK`) | n/a — thinking on returns `reasoning_content` |

Notes on the two non-table columns (these are **not** produced by `probe_api.py`; measured separately against the same endpoint, same prompts, `max_tokens=16`):

- **5 identical calls at T=0.7, thinking off:** outputs were `OK` five times. The prompt (`Reply OK.`) is degenerate, so this shows no variance *on this prompt*; it is not evidence that T=0.7 is deterministic on substantive prompts.
- **5 identical calls at T=0.7, thinking on:** 4 responses had empty `content` and 1 returned `OK`. With `max_tokens=16`, reasoning tokens are counted inside `completion_tokens`, so the budget is consumed by `reasoning_content` (observed `reasoning_tokens: 10` on a 12-token completion) and `content` truncates to empty. This is budget truncation, not evidence of sampling variance.
- **Reasoning field:** thinking off → response `message` keys `['content', 'role']`, no `reasoning_content`. Thinking on → `['content', 'reasoning_content', 'role']`, and `usage.completion_tokens_details.reasoning_tokens` is present. Thinking on also raised `prompt_tokens` from 21 to 46 for the identical messages.
- **Out-of-range probe:** `temperature=3.0` was rejected (HTTP 4xx) in every cell, so the endpoint range-validates temperature. That does not prove it honors 0.7 during sampling.

## Token cost

| Phase | Input | Output | Total | Calls |
|---|---|---|---|---|
| `probe_api.py` (the single mandated run) | 134 | 18 | **152** | 8 HTTP (4 rejected by the 3.0 probe; no usage returned) |
| Supplementary variance / response-shape checks | 402 | 97 | 499 | 12 |
| **All calls made for this task** | **536** | **115** | **651** | 16 billed + 4 rejected |

`price_table` is null in `config.yaml` (peak/off-peak split), so cost in currency is deliberately not computed; token counts above are the measured values.

## Caveats

1. The probe's `conclusion` column is a *range-validation* proxy, not proof that temperature is honored during sampling. Sampling variability on real verifier prompts remains unverified.
2. `thinking: {"type": "enabled"}` is accepted by this endpoint and returns `reasoning_content`; with `max_tokens=16` it cannot emit final content, so thinking-on cells are unusable at that budget (they would register as parse failures downstream).
3. Per the 2026-11-14 §7 amendment, these prompts contain no dataset claim or evidence text, so this run is an infrastructure check, not a "result"; no `spec-frozen` tag was required for it.
4. No project code was modified. Only `config.yaml` (`models.deepseek.max_tokens`) and the new `results/` directory changed.
