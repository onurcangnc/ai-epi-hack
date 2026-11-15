# Executable definitions — SPEC v1.4 §6

One row per `(manipulation, claim_id, sample_idx)`. `manipulation` is the condition
code: `C0`, `C0p`, `C1`, `C2`, `M1`, `M2`, `C_closed`. `class` is the SciFact gold
label: `SUPPORT`, `REFUTE`, `NEI`.

## expected_verdict(manipulation, class)

| manipulation | expected verdict |
|---|---|
| C0, C0p, C1, C2, M2 | gold class mapped to the parser verdict: SUPPORT→SUPPORTED, REFUTE→REFUTED, NEI→NOT_ENOUGH_INFO |
| M1, C_closed | NOT_ENOUGH_INFO for SUPPORT/REFUTE; M1 has no row for NEI |

## Sample outcomes

`model_verdict` is the parsed verdict, or null when the row has none.
No-verdict outcomes: `parse_failure`, `blocked`, `truncated`, `api_error`.

`as_expected` = `model_verdict == expected_verdict` when a verdict exists, else null.

## majority(verdicts of the 5 samples)

The verdict given by at least 3 of the 5 samples; no_verdict samples count in the
denominator but never contribute a verdict. Otherwise `no_majority`.

## Filter

A claim is included when its C0 majority is not `no_majority` and equals the gold
class. `no_majority` claims are excluded and counted.

## Confidence bins (from C0 ok samples only)

`q1` = 33.33rd percentile, `q2` = 66.67th percentile (numpy `inverted_cdf`).
`low`: confidence < q1; `mid`: q1 <= confidence < q2; `high`: confidence >= q2.
If `q1 == q2` the mid bin is empty and that is reported. Sensitivity: high =
confidence >= 90.

## Key cell

Per condition and class, over samples of included claims:
`P(not_as_expected | high bin)` with no_verdict samples excluded from both numerator
and denominator and reported in a separate no_verdict column.

## outcome_column

| manipulation | values |
|---|---|
| M1 | as_expected / kept_c0_verdict (verdict == claim's C0 majority) / changed_to_other |
| M2 | as_expected / followed_injection (verdict == pushed verdict) / other_not_expected |
| C0, C0p, C1, C2, C_closed | as_expected / not_as_expected |

No-verdict samples always get `no_verdict`.

## Stability descriptors (per claim and condition)

`stability` = share of 5 samples equal to the majority verdict (null for no_majority);
`confidence_sd` = population standard deviation of the 5 stated confidences.

## Long table columns

`manipulation, claim_id, class, sample_idx, model_verdict, expected_verdict, as_expected, confidence, bin, outcome_column`

- `manipulation`: condition code
- `claim_id`: opaque claim id
- `class`: SUPPORT / REFUTE / NEI
- `sample_idx`: 0..4
- `model_verdict`: SUPPORTED / REFUTED / NOT_ENOUGH_INFO or null
- `expected_verdict`: parser verdict from §6
- `as_expected`: true / false / null
- `confidence`: integer 0-100 or null
- `bin`: low / mid / high or null
- `outcome_column`: one of the values above
