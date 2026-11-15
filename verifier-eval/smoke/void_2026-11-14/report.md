> **VOID — not the official smoke result.** Produced before Task B existed; kept for audit only.
> The official run must be executed after Task B is merged and committed.

# Smoke test — sampling behaviour on 6 disjoint SciFact claims

- Run (UTC): 2026-11-14
- Claims: `smoke/claims.jsonl`, 2 SUPPORT / 2 REFUTE / 2 NEI, seed `smoke.seed = 20261115`, opaque ids prefixed `s_`
- Disjointness: 0 overlap with the 150 on both `(split, claim_id)` and opaque id (checked at selection)
- Model/settings: `deepseek-flash`, thinking off, `max_tokens=512`; temperature 0.7 then 1.0
- Client: real `VerifierClient`, frozen prompts, condition C0 (full abstract, no degradation), `sample_idx` 0-4
- Logs: `smoke/raw/`; nothing written under `results/raw/`
- Calls: 30 + 30 = 60, every `finish_reason` = `stop`; no retries, no parse failures

## T = 0.7

| claim | gold | n | verdict | confidence | finish_reason | outcome | tokens in/out | latency ms |
|---|---|---|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | 0 | NOT_ENOUGH_INFO | 80 | stop | ok | 463/93 | 1500 |
| s_8eb99f781d | SUPPORT | 1 | NOT_ENOUGH_INFO | 70 | stop | ok | 463/120 | 1360 |
| s_8eb99f781d | SUPPORT | 2 | NOT_ENOUGH_INFO | 78 | stop | ok | 463/142 | 1484 |
| s_8eb99f781d | SUPPORT | 3 | NOT_ENOUGH_INFO | 75 | stop | ok | 463/130 | 1406 |
| s_8eb99f781d | SUPPORT | 4 | SUPPORTED | 78 | stop | ok | 463/166 | 1813 |
| s_d08782b254 | SUPPORT | 0 | SUPPORTED | 90 | stop | ok | 425/65 | 1156 |
| s_d08782b254 | SUPPORT | 1 | SUPPORTED | 85 | stop | ok | 425/70 | 828 |
| s_d08782b254 | SUPPORT | 2 | SUPPORTED | 88 | stop | ok | 425/54 | 1235 |
| s_d08782b254 | SUPPORT | 3 | SUPPORTED | 88 | stop | ok | 425/58 | 1140 |
| s_d08782b254 | SUPPORT | 4 | SUPPORTED | 88 | stop | ok | 425/71 | 1000 |
| s_98161f2655 | REFUTE | 0 | REFUTED | 90 | stop | ok | 478/66 | 922 |
| s_98161f2655 | REFUTE | 1 | REFUTED | 92 | stop | ok | 478/62 | 1031 |
| s_98161f2655 | REFUTE | 2 | REFUTED | 90 | stop | ok | 478/61 | 1360 |
| s_98161f2655 | REFUTE | 3 | REFUTED | 90 | stop | ok | 478/66 | 1297 |
| s_98161f2655 | REFUTE | 4 | REFUTED | 92 | stop | ok | 478/58 | 1187 |
| s_572de81f83 | REFUTE | 0 | REFUTED | 88 | stop | ok | 808/83 | 1375 |
| s_572de81f83 | REFUTE | 1 | REFUTED | 88 | stop | ok | 808/77 | 1031 |
| s_572de81f83 | REFUTE | 2 | REFUTED | 88 | stop | ok | 808/77 | 1172 |
| s_572de81f83 | REFUTE | 3 | REFUTED | 88 | stop | ok | 808/83 | 1156 |
| s_572de81f83 | REFUTE | 4 | REFUTED | 85 | stop | ok | 808/83 | 1297 |
| s_5a60051d0c | NEI | 0 | NOT_ENOUGH_INFO | 92 | stop | ok | 489/75 | 782 |
| s_5a60051d0c | NEI | 1 | NOT_ENOUGH_INFO | 85 | stop | ok | 489/120 | 1453 |
| s_5a60051d0c | NEI | 2 | NOT_ENOUGH_INFO | 90 | stop | ok | 489/122 | 1593 |
| s_5a60051d0c | NEI | 3 | NOT_ENOUGH_INFO | 85 | stop | ok | 489/116 | 1610 |
| s_5a60051d0c | NEI | 4 | NOT_ENOUGH_INFO | 85 | stop | ok | 489/102 | 937 |
| s_dac9b38245 | NEI | 0 | NOT_ENOUGH_INFO | 78 | stop | ok | 475/67 | 1297 |
| s_dac9b38245 | NEI | 1 | NOT_ENOUGH_INFO | 78 | stop | ok | 475/103 | 1469 |
| s_dac9b38245 | NEI | 2 | NOT_ENOUGH_INFO | 80 | stop | ok | 475/64 | 1015 |
| s_dac9b38245 | NEI | 3 | NOT_ENOUGH_INFO | 82 | stop | ok | 475/78 | 1172 |
| s_dac9b38245 | NEI | 4 | NOT_ENOUGH_INFO | 85 | stop | ok | 475/67 | 1094 |

Per-claim stability:

| claim | gold | verdicts (5) | confidences (5) | distinct (verdict, conf) | distinct conf |
|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, SUPPORTED | 80, 70, 78, 75, 78 | 5 | 4 |
| s_d08782b254 | SUPPORT | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 90, 85, 88, 88, 88 | 3 | 3 |
| s_98161f2655 | REFUTE | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 90, 92, 90, 90, 92 | 2 | 2 |
| s_572de81f83 | REFUTE | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 88, 88, 88, 88, 85 | 2 | 2 |
| s_5a60051d0c | NEI | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 92, 85, 90, 85, 85 | 3 | 3 |
| s_dac9b38245 | NEI | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 78, 80, 82, 85 | 4 | 4 |

## T = 1.0

| claim | gold | n | verdict | confidence | finish_reason | outcome | tokens in/out | latency ms |
|---|---|---|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | 0 | NOT_ENOUGH_INFO | 70 | stop | ok | 463/79 | 1203 |
| s_8eb99f781d | SUPPORT | 1 | NOT_ENOUGH_INFO | 72 | stop | ok | 463/101 | 1344 |
| s_8eb99f781d | SUPPORT | 2 | NOT_ENOUGH_INFO | 70 | stop | ok | 463/98 | 1078 |
| s_8eb99f781d | SUPPORT | 3 | SUPPORTED | 78 | stop | ok | 463/162 | 1875 |
| s_8eb99f781d | SUPPORT | 4 | NOT_ENOUGH_INFO | 78 | stop | ok | 463/103 | 1031 |
| s_d08782b254 | SUPPORT | 0 | SUPPORTED | 90 | stop | ok | 425/62 | 1250 |
| s_d08782b254 | SUPPORT | 1 | SUPPORTED | 90 | stop | ok | 425/71 | 1063 |
| s_d08782b254 | SUPPORT | 2 | SUPPORTED | 92 | stop | ok | 425/64 | 1156 |
| s_d08782b254 | SUPPORT | 3 | SUPPORTED | 90 | stop | ok | 425/54 | 1219 |
| s_d08782b254 | SUPPORT | 4 | SUPPORTED | 92 | stop | ok | 425/55 | 765 |
| s_98161f2655 | REFUTE | 0 | REFUTED | 90 | stop | ok | 478/66 | 969 |
| s_98161f2655 | REFUTE | 1 | REFUTED | 88 | stop | ok | 478/61 | 969 |
| s_98161f2655 | REFUTE | 2 | REFUTED | 90 | stop | ok | 478/66 | 1093 |
| s_98161f2655 | REFUTE | 3 | REFUTED | 92 | stop | ok | 478/63 | 1063 |
| s_98161f2655 | REFUTE | 4 | REFUTED | 92 | stop | ok | 478/62 | 875 |
| s_572de81f83 | REFUTE | 0 | REFUTED | 92 | stop | ok | 808/82 | 1125 |
| s_572de81f83 | REFUTE | 1 | REFUTED | 88 | stop | ok | 808/106 | 1281 |
| s_572de81f83 | REFUTE | 2 | REFUTED | 88 | stop | ok | 808/89 | 1578 |
| s_572de81f83 | REFUTE | 3 | REFUTED | 88 | stop | ok | 808/80 | 1157 |
| s_572de81f83 | REFUTE | 4 | REFUTED | 88 | stop | ok | 808/102 | 1359 |
| s_5a60051d0c | NEI | 0 | NOT_ENOUGH_INFO | 92 | stop | ok | 489/85 | 1250 |
| s_5a60051d0c | NEI | 1 | NOT_ENOUGH_INFO | 90 | stop | ok | 489/103 | 1406 |
| s_5a60051d0c | NEI | 2 | NOT_ENOUGH_INFO | 85 | stop | ok | 489/106 | 1360 |
| s_5a60051d0c | NEI | 3 | NOT_ENOUGH_INFO | 88 | stop | ok | 489/115 | 1468 |
| s_5a60051d0c | NEI | 4 | NOT_ENOUGH_INFO | 85 | stop | ok | 489/106 | 1235 |
| s_dac9b38245 | NEI | 0 | NOT_ENOUGH_INFO | 78 | stop | ok | 475/83 | 1203 |
| s_dac9b38245 | NEI | 1 | NOT_ENOUGH_INFO | 85 | stop | ok | 475/63 | 922 |
| s_dac9b38245 | NEI | 2 | NOT_ENOUGH_INFO | 75 | stop | ok | 475/63 | 1093 |
| s_dac9b38245 | NEI | 3 | NOT_ENOUGH_INFO | 85 | stop | ok | 475/67 | 907 |
| s_dac9b38245 | NEI | 4 | NOT_ENOUGH_INFO | 80 | stop | ok | 475/73 | 1297 |

Per-claim stability:

| claim | gold | verdicts (5) | confidences (5) | distinct (verdict, conf) | distinct conf |
|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, SUPPORTED, NOT_ENOUGH_INFO | 70, 72, 70, 78, 78 | 4 | 3 |
| s_d08782b254 | SUPPORT | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 90, 90, 92, 90, 92 | 2 | 2 |
| s_98161f2655 | REFUTE | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 90, 88, 90, 92, 92 | 3 | 3 |
| s_572de81f83 | REFUTE | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 88, 88, 88, 88 | 2 | 2 |
| s_5a60051d0c | NEI | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 92, 90, 85, 88, 85 | 4 | 4 |
| s_dac9b38245 | NEI | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 85, 75, 85, 80 | 4 | 4 |

## Does the temperature knob do anything?

| claim | gold | T=0.7 distinct pairs | T=1.0 distinct pairs | T=0.7 verdict counts | T=1.0 verdict counts |
|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | 5 | 4 | {'NOT_ENOUGH_INFO': 4, 'SUPPORTED': 1} | {'NOT_ENOUGH_INFO': 4, 'SUPPORTED': 1} |
| s_d08782b254 | SUPPORT | 3 | 2 | {'SUPPORTED': 5} | {'SUPPORTED': 5} |
| s_98161f2655 | REFUTE | 2 | 3 | {'REFUTED': 5} | {'REFUTED': 5} |
| s_572de81f83 | REFUTE | 2 | 2 | {'REFUTED': 5} | {'REFUTED': 5} |
| s_5a60051d0c | NEI | 3 | 4 | {'NOT_ENOUGH_INFO': 5} | {'NOT_ENOUGH_INFO': 5} |
| s_dac9b38245 | NEI | 4 | 4 | {'NOT_ENOUGH_INFO': 5} | {'NOT_ENOUGH_INFO': 5} |

## Reading

- Both temperatures are stochastic on identical prompts: every claim shows at least 2 distinct
  (verdict, confidence) pairs at both settings.
- Raising T from 0.7 to 1.0 produced no detectable increase in dispersion at this n: 19 distinct
  pairs at each temperature, identical per-claim verdict counts, and only 1-2 points of confidence
  movement in both directions. The 0.7 setting is already visibly non-deterministic.
- n = 6 claims x 5 samples, so this cannot establish that the knob does nothing in general. It does
  show that k=5 for the main run is measuring real variance, and that any 0.7-vs-1.0 difference
  claim would need far more samples.
- `s_8eb99f781d` (gold SUPPORT) returns 4/5 NOT_ENOUGH_INFO at both temperatures: a
  gold-disagreement case, not a temperature effect.

## Totals

| phase | calls | input | output | total | finish_reason | retries |
|---|---|---|---|---|---|---|
| T=0.7 | 30 | 15690 | 2599 | 18289 | {'stop': 30} | 0 |
| T=1.0 | 30 | 15690 | 2490 | 18180 | {'stop': 30} | 0 |
| **both** | **60** | **31380** | **5089** | **36469** | stop x60 | 0 |

`price_table` is null, so no currency cost is computed; the token counts above are measured usage.

## Notes

- `finish_reason` was captured with a read-only wrapper around the backend session; it is not stored in the client log schema. All 30+30 values were `stop`, so no output hit the 512-token cap.
- `VerifierClient` has no disk cache: its cache is an in-memory dict and takes no path argument. The task's `smoke/cache/` is populated with a dump of that cache (`temp_0.7.cache.json`, `temp_1.0.cache.json`), not used as a live cache.
- All rows are `outcome: ok`; no parse failures, no API errors.
- Claims were never in the main 150; selection excluded all 150 `(split, claim_id)` pairs before sampling.
