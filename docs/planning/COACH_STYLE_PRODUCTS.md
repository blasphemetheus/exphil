# Coach and style products — the third line

**Written 2026-09-20.** Own planning doc with its own goals and status
ledger. Sits beside `IMITATION_SQUEEZE.md` (better prior) and
`RL_ON_PRIOR.md` (stronger player). This line ships things people use,
built on what the other two lines bank. Detailed design for the coach
stays in `COACH_ROADMAP.md`; the identity pipeline stays in
`STYLE_IDENTITY.md`. This doc is the product ledger over both.

## What exists today (product-relevant inventory)

| Piece | State | Where |
| --- | --- | --- |
| Play as a named profile (`--style-tag X --player-registry …`) | **Works**; D3 passed at n=10; Bradley: C2 "played a little different" | `scripts/play_dolphin.exs`, V3.1-ep3 registry (111 slots) |
| Style fingerprint per game (v1 + v2 timing, 30+ habits) | Works, calibrated (same-player retrieval, NCA metric, ~91 % closed-set at τ0.8) | `ExPhil.Interp.Style{Fingerprint,Timing,Calibration,Metric,Matcher,Clusters}`, `scripts/style_*.exs` |
| Identity registry: 58 % of corpus games conditioned (real tags + pseudo-clusters) | Works; S4 alias merges adjudicated, not yet applied | `data/identity/`, `STYLE_RECONCILIATION_2026-09-17.md` |
| Situations labeler (47 labels), Inspect API, HTML rewind viewer | Shipped 08-10/11 | `ExPhil.Situations`, `ExPhil.Inspect`, `priv/viewer/rewind_viewer.html` |
| Option vocabulary + per-situation corpus stats (1.5 M events) | Shipped | `ExPhil.Options`, situation_stats v2 |
| Value model (COACH F5) | Not built; = RL doc gate R2 | `RL_ON_PRIOR.md` |
| Improoover (.slp moment → bootable savestate) | Recipe known, plumbing spike unscheduled | memory `reference_improoover` |

## Products, ranked by distance from shipping

### S1 — Style report card ("how do you play")
Input: a folder of someone's replays. Output: their fingerprint against
the human pool (z-scores on the identity tells: jump button, c-stick
aerials, wavedash/dashdance rates, L-cancel timing, roll/spotdodge
rates), nearest known players, and which habits are far from the pool.
Everything needed exists; the work is a script and a one-page HTML
render. **Distance: days.** Eval bar: the card for a known player should
name them as nearest at the matcher's accuracy.

### S2 — Play like a specific person ("play like your rival / like Michael")
Two tiers. (a) If they are in the registry (or match a slot at τ ≥ 0.8
via S1), it is the existing `--style-tag` path. (b) If not, a
**per-player fine-tune** of the identity embedding only (the 64-dim slot
vector, trunk frozen) on their replays; with ≥ 20 games that is minutes
of training. Standing example player: Michael ([INFP]). **Distance:
(a) done, (b) ~1 week** (a `scripts/style_finetune_slot.exs` + a D3-style
probe on the result). Eval bar: D3 at n=10 against the person's own
fingerprint.

### S3 — Sparring partner modes
Named-profile bot + the delay rung + stage pin, packaged as a launcher
with a menu (profile, stage, delay, CPU vs Direct). Mostly DEPLOY_KNOBS
plumbing. **Distance: days**, after S1/S2 so the profiles mean something.
Depends on the RL line only for "harder than the prior" difficulty
levels.

### C1 — Options overlay ("chess-move highlights")
Per situation: the options the corpus's players actually take here, with
frequencies, over the rewind viewer. `Situations` + `Options` +
situation_stats exist; needs the option enumerator for the current state
and the overlay render. **Distance: 1–2 weeks.** Eval bar: corpus-stats
proxy (COACH_ROADMAP v0/v0.5).

### C2 — Engine lens ("what does the bot think here")
Policy's option distribution + value at a moment, and per-option value
deltas. Policy half exists (Inspect.counterfactual); the value half is RL
gate R2. **Distance: after R2.**

### C3 — Natural-language replay feedback
Narration over C1/C2 structured facts. **Last**; the phrasing layer is
small once C1/C2 emit facts.

### Curriculum / Improoover reps
Savestate from a moment + scripted or searched scenario director; the sim
makes "randomized starts from this moment" cheap (MELEE_SIM_USES §2).
**After the RL doc's R0 adapter**, since the same adapter serves both.

## Design rules for this line

- Products consume the other lines' artifacts by contract, never by
  private branches: policy = the crowned generalist artifact + registry;
  value = R2's critic; sim = the R0 adapter.
- Every "plays like X" claim is measured with the fingerprint pipeline at
  n ≥ 10 games (the D3 bar), never by impression alone; impressions are
  recorded next to the numbers.
- Tags and costumes are evidence, not ground truth (STYLE_IDENTITY);
  products show confidence, never a bare name, for matched identities.
- Privacy: anonymized (hashed) players get cluster names (C2), never a
  de-anonymized real handle, unless the handle came from their own
  in-game tag.

## Sequencing

1. S1 report card (unblocks S2b eval and is the first shippable).
2. S2b slot fine-tune.
3. S3 launcher.
4. C1 overlay (independent of RL; can interleave with 1–3).
5. C2 when R2 lands; C3 after C1+C2; curriculum after R0.

## Status ledger

| Date | Item | State | Notes |
| --- | --- | --- | --- |
| 2026-09-20 | — | Doc written | S2a (registry profiles) already works: D3 n=10 pass, Bradley played C2 and INFP |
| 2026-09-20 | S1 | not started | All inputs exist (`style_fingerprint.exs`, calibration, metric) |

### S4 — the model's own costume (Bradley 2026-09-21, "a favorite costume developed by training")
Costume is not an input today (mapper drops it, no embedding dims), so a
preference must come from somewhere the model *does* express itself:
(a) per named slot, the matcher's P(costume | entity) is a queryable
preference now (evidence, not truth — sample it); (b) **costume head**:
predict the subject's costume from play (fingerprint-style), then let the
policy play in the sim and pick the costume its own play predicts — a
preference that is genuinely a product of training; (c) costume as an
embedding input (V3.2 pilot, changes the contract). Do (b) with the S1
report card; the head doubles as matcher evidence.
- [ ] S4 costume head + "pick your costume" demo.
