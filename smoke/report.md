# TASK A2 smoke rerun — side by side with the void run

- Rerun: 2026-11-14, normal `VerifierClient` (no wrappers), disk cache `smoke/cache/`, logs `smoke/raw/`
- Same 6 disjoint claims, condition C0, k=5, T=0.7 and T=1.0; temperature unchanged
- Decision table recorded in CHANGELOG before the first call

## T = 0.7

| claim | gold | run | 5 verdicts | 5 confidences | distinct pairs |
|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, SUPPORTED | 80, 70, 78, 75, 78 | 5 |
| s_8eb99f781d | SUPPORT | rerun | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 70, 78, 70, 70, 75 | 3 |
| s_d08782b254 | SUPPORT | void | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 90, 85, 88, 88, 88 | 3 |
| s_d08782b254 | SUPPORT | rerun | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 92, 88, 85, 88, 90 | 4 |
| s_98161f2655 | REFUTE | void | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 90, 92, 90, 90, 92 | 2 |
| s_98161f2655 | REFUTE | rerun | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 92, 95, 95, 92 | 2 |
| s_572de81f83 | REFUTE | void | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 88, 88, 88, 88, 85 | 2 |
| s_572de81f83 | REFUTE | rerun | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 88, 88, 88, 92 | 2 |
| s_5a60051d0c | NEI | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 92, 85, 90, 85, 85 | 3 |
| s_5a60051d0c | NEI | rerun | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 85, 85, 80, 85, 92 | 3 |
| s_dac9b38245 | NEI | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 78, 80, 82, 85 | 4 |
| s_dac9b38245 | NEI | rerun | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 80, 80, 72, 75, 80 | 3 |

## T = 1.0

| claim | gold | run | 5 verdicts | 5 confidences | distinct pairs |
|---|---|---|---|---|---|
| s_8eb99f781d | SUPPORT | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, SUPPORTED, NOT_ENOUGH_INFO | 70, 72, 70, 78, 78 | 4 |
| s_8eb99f781d | SUPPORT | rerun | NOT_ENOUGH_INFO, SUPPORTED, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 70, 70, 78, 78 | 3 |
| s_d08782b254 | SUPPORT | void | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 90, 90, 92, 90, 92 | 2 |
| s_d08782b254 | SUPPORT | rerun | SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED | 92, 90, 88, 90, 95 | 4 |
| s_98161f2655 | REFUTE | void | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 90, 88, 90, 92, 92 | 3 |
| s_98161f2655 | REFUTE | rerun | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 95, 92, 90, 95 | 3 |
| s_572de81f83 | REFUTE | void | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 88, 88, 88, 88 | 2 |
| s_572de81f83 | REFUTE | rerun | REFUTED, REFUTED, REFUTED, REFUTED, REFUTED | 92, 90, 90, 88, 85 | 4 |
| s_5a60051d0c | NEI | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 92, 90, 85, 88, 85 | 4 |
| s_5a60051d0c | NEI | rerun | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 80, 88, 82, 88, 85 | 4 |
| s_dac9b38245 | NEI | void | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 85, 75, 85, 80 | 4 |
| s_dac9b38245 | NEI | rerun | NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO, NOT_ENOUGH_INFO | 78, 78, 75, 85, 85 | 3 |

## Summary

| run | T | claims with >=2 distinct conf | claims with a verdict change | total distinct pairs | finish_reason | input | output |
|---|---|---|---|---|---|---|---|
| void | 0.7 | 6/6 | 1/6 | 19 | {'n/a (pre-Task-B rows)': 30} | 15690 | 2599 |
| void | 1.0 | 6/6 | 1/6 | 19 | {'n/a (pre-Task-B rows)': 30} | 15690 | 2490 |
| rerun | 0.7 | 6/6 | 0/6 | 17 | {'stop': 30} | 15680 | 2561 |
| rerun | 1.0 | 6/6 | 1/6 | 21 | {'stop': 30} | 15680 | 2421 |

## Decision (decision table, rerun)

**keep 0.7** — first row fires: at T=0.7, 6/6 claims show >=2 distinct confidence values (0/6 show a verdict change). This is also the "only confidence varies" case, so stability is confidence-driven. The rerun changed no temperature.

## Totals

Rerun token cost: **36342** (31360 input / 4982 output), 60 calls, 60 result rows, no retries.

`price_table` is null, so no currency cost is computed.
