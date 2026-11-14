# CHANGELOG

Changes to `SPEC.md` are recorded here with their reason, per SPEC §7. No
experiment results existed at the time of the change below; only the Task 1
data files.

## 2026-11-14 — §3 Subjects: second API subject replaced by a manual Opus pilot

**Change.** `SPEC.md` §3 was rewritten:

- DeepSeek direct API is now the only subject that runs the full matrix
  (150 claims, thinking off, temperature 0.7, k=5).
- `claude-opus-5-5` moves from "second API subject (decision pending)" to a
  manual Claude Desktop pilot: 30 nested claims (10/10/10), seed-fixed before
  any DeepSeek result exists, conditions C0/M1/M2, k=1.
- Pilot protocol is fixed: fresh private chat per prompt, memory and personal
  preferences off, system prompt pasted as the opening of the first message
  (not a separate system field).
- Temperature is not controllable in Claude Desktop, so stability is not
  measured for Opus; the pilot reports verdict and confidence shifts only and
  is never pooled with the DeepSeek tables.
- Reviewer sessions (code/data/results audit) are separate chats and must say
  so in the writeup if they also reviewed pilot chats of the same model.

**Reason.** The planned second API subject was not available under the
sprint's call-layer protocol: Opus 5.5 is reachable only through Claude
Desktop (no API access/key in this environment), and Claude Desktop does not
expose temperature. Running it as an API subject next to DeepSeek would have
required a different harness and different sampling settings, which breaks
the controlled comparison the spec exists to provide. Separating it into a
small, seed-fixed, hand-run pilot keeps the main table comparable and still
yields an exploratory second-model signal. Because temperature cannot be
fixed, the pilot drops the sampling-stability measure (k=1 could not estimate
it anyway) and reports only verdict and confidence shifts.

**Consequences not yet applied elsewhere.**

- The verifier call layer (Task 2) was specified with an Anthropic API backend
  and a two-backend `probe_api.py`. Subject calls now only need DeepSeek;
  whether to drop the unused Anthropic backend or keep it for the probe is an
  open human decision.
- The 30-claim pilot subset has not been drawn yet. It must be drawn with an
  explicit seed from config and frozen before the first DeepSeek result.

## 2026-11-14 — §8 Known limitations aligned with the new §3

**Change.** §8 now says the main results cover one open-weight model (DeepSeek)
only, and adds an explicit description of the manual Opus 5.5 pilot: n=30
claims, k=1, no temperature control, system prompt pasted into the first
message, descriptive only, no statistical claims, no stability measurement,
never pooled with the DeepSeek tables.

**Reason.** §8 still claimed "Two subjects only" after the §3 amendment moved
Opus out of the API matrix into a hand-run pilot. Without this edit the
writeup could over-claim a two-model comparison and a stability result for
Opus that the design does not produce.

## 2026-11-14 — §6 Definitions rewritten around per-condition expected behavior

**Change.**

- Adds `expected(condition, gold)`: C0/C1/M2 -> gold label;
  M1/C_closed -> NOT_ENOUGH_INFO; M3-M5 pending.
- Replaces the `flip_to_wrong` / `flip_to_right` split with a per-sample
  classification: `as_expected` / `not_as_expected`, the latter split into
  `kept_c0_verdict` / `changed_to_other`.
- Majority-level flip is kept but demoted to a secondary table. The primary
  table is now sample-level: confidence bin x
  (as_expected / kept_c0_verdict / changed_to_other), with the key cell
  "high bin x not_as_expected".
- Confidence bins are computed across all outcomes (not "before any flip
  outcome"), then used for the table.

**Reason.**

1. `flip_to_right` was structurally empty under the old wording: flip analysis
   keeps only claims whose C0 majority equals the gold label, and "right" was
   measured relative to gold, so a flipped verdict could never be "right".
2. M1's correct behavior is NOT_ENOUGH_INFO, but the old `flip_to_wrong` was
   measured against the gold label. A SUPPORT/REFUTE claim that correctly
   answered NEI in M1 was therefore counted as a wrong flip.
3. `expected()` separates what a condition asks for from what the model did,
   so M1 achieving NOT_ENOUGH_INFO is expected behavior, and "right/wrong
   relative to gold" is no longer needed.
4. The sample-level primary table keeps the per-sample confidence signal
   instead of collapsing it before looking at confidence, and ties/parse
   failures stay visible instead of disappearing inside a majority.

No results existed when this change was made; §6 is frozen as of the first
frozen DeepSeek result and any later change gets its own entry. The existing
`analysis/DEFINITIONS.HASH` has not been created yet (Task 5), so it will
hash the new text.


