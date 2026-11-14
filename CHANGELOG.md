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

## 2026-11-14 — Verifier prompts and M2 templates frozen (prompts-v1)

Added by the human, stored byte-for-byte (UTF-8, no BOM). No agent edits
these files.

| file | SHA-256 |
|---|---|
| `prompts/verifier_system.txt` | `70a8d8f58cf5203eceef690b1833dfa4db7ff206e8f4a657508063902417c731` |
| `prompts/verifier_user.txt` | `f6ababfe4ad33dd4be2cc5386006637bcae90f9a70f1489143562078f98fe0ba` |
| `prompts/m2_templates.json` | `8e8fa8ce550bb3a04784248e2231b58c5b74347fac50637f38d6c555b684861c` |

- The system prompt contains literal `{}`, so the call layer must use
  `str.replace`, never `str.format`.
- Every M2 template contains `{pushed_verdict}` exactly once (validated);
  fill it with `str.replace` and fail loudly otherwise.
- M2 template families: `direct_command`, `disguised_note`, `ready_json`;
  five templates each.

## 2026-11-14 — §5 M3-M5 generator fixed: Opus 5.5 in Claude Desktop

**Change.** §5 now names the generator: `claude-opus-5-5` in Claude Desktop,
manual, batched at 10 claims per message, in chats separate from the pilot and
from reviewer sessions. Every generated item records
`generator="opus-5.5-desktop"` and `synthetic=true`. Because Opus is the
generator, the Opus pilot does not include M3-M5. The 30-50 item audit is done
by a person, not by the generator model. A generator change is logged here and
the generated items are rebuilt.

**Reason.** §5 previously said only "a generator model that is NOT a subject"
and left the choice open, which left the provenance fields, chat separation
and audit independence undefined. Naming Opus fixes the metadata contract for
Task 3b and prevents the Opus pilot from silently sharing M3-M5 material that
the same model produced.

## 2026-11-14 — SPEC v1 sections 1, 3, 5, 6, 7, 8 amended (approved 1-8)

**Change.**

- §1 defines the primary estimand: for M in {M1, M2}, the difference in
  P(not_as_expected | high-confidence bin) between M and C0' on C0-filtered
  claims; everything else is exploratory.
- §3 pins pilot ordering and settings: pilot runs last, after SPEC, config and
  analysis scripts are frozen and the DeepSeek main run exists; web search off,
  one thinking setting, no styles, C0 twice (30 extra chats), no reruns after a
  bad reply, every chat in `pilot/NOTES.md`, draw recorded here.
- §5 replaces the condition table with C0, C0', C1, C2, M1, M2, C_closed and
  fixed call counts; adds M2 Latin-square family/position assignment,
  pushed-verdict rules, insertion positions and render-time normalization, plus
  the M1 word-overlap audit. §2 and §4 are unchanged.
- §6 replaces the definitions: `expected()` covers C0'/C1/C2; majority means
  >=3 of 5 with `no_majority` otherwise; cut points come only from C0 via
  `inverted_cdf`; key cell per condition; stability via Spearman; NEI vs
  SUPPORT/REFUTE confidence check; paired bootstrap with Clopper-Pearson
  fallback and Holm correction.
- §7 adds the result definition, the first-row rule, no retry of parse
  failures in the main run, and the prompt-warning exemption.
- §8 adds the M2 lower-bound caveat, the missing M1 text-removal control (C3),
  the single-setup scope, and the Opus review disclosure.

**Reason.**

- The old flip definition measured "right" against the gold label after
  filtering on C0==gold, so flip-to-right could never occur, and there was no
  noise floor. C0' provides a paired baseline for sampling noise.
- C2 isolates the M2 position/length effect; C1 becomes a fixed 6-sentence
  control instead of depending on M3, which is not run.
- Majority >=3/5 removes unresolvable 2-1-2 ties and keeps parse/API failures
  inside the k=5 denominator instead of silently dropping them.
- Cut points from C0 only keep the confidence bins independent of outcomes;
  the fixed `confidence >= 90` sensitivity check guards against unstable
  tertiles.
- Restricting M1/C_closed to SUPPORT/REFUTE removes the NEI no-op and stops a
  correct NOT_ENOUGH_INFO answer in M1 from being scored as an error.
- Requiring the pilot to run last protects the freeze: any subject output,
  pilot included, is a "result".
- Treating parse failures as outcomes (no retry) keeps the failure rate
  interpretable, while API errors stay retryable infrastructure.





