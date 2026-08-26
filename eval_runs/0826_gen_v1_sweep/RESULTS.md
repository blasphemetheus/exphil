# fox_gen_v1 sweep — 2026-08-26 (COMPLETE)

**Headline: fox_gen_v1 plays Melee.** On a quiet machine at temperature
0.5 it approaches ~0.5x/min, converts **35% of openings (35/100 over 30
games)**, and dies 1.7x per 2-minute game vs a CPU. Compare the
drill-era Mewtwo baseline that motivated GOALS.md's Track A gates: 0.17
approaches/min and **5%** conversion. **Decode strategy, not training,
was the difference between a "forever-crouching fox" and a functioning
one** — argmax collapses this model; sampling releases it.

(Overnight status — LIVE RUNG BLOCKED — resolved 12:12 when Bradley
paused the `./permut` job; all live numbers below are from loadavg<1
runs with 0-1.5% stale sends.)

## The run

`fox_gen_v1` — 7,911 master-Fox games (~90M frames, ~340k batches/epoch),
GRU-60 stride 5 batch 256 f32 lr 1e-4 constant, `--stage-internals`
(actually wired — 296-wide, canary-verified), streaming lazy pipeline
(chunk 100), 10 epochs, ~11.4h wall on the 5090. Clean exit; registered
as `reskinned_jab`. Checkpoints:
`checkpoints/fox_gen_v1_20260825_210355{_ep1..10,_best,}_policy/bin`.

Val curve (monotonic, decelerating; best = ep10 5.789):
6.639 / 6.209 / 6.084 / 6.006 / 5.949 / 5.905 / 5.869 / 5.839 / 5.813 / 5.789

Remember the standing law: loss ANTI-correlates with behavior — the val
argmax is not a deploy decision.

## Offline fixture agreement (all 10 epochs, master-game fixture, n=10,259 frames)

ep1..10: .9745 .9737 .9762 .9731 .9751 .9749 .9747 .9750 .9736 .9749

**Two readings:**
1. **The mapping is LEARNED** — 97.4% B/X agreement with master play,
   press rates matching (B 3.8% vs 4.2%), from epoch 1 onward. The
   "trained on garbage / learned to SD" hypothesis is refuted.
2. **The instrument is SATURATED** — flat across epochs, zero ranking
   power (rhymes with the stand-gate saturation in the specialist line).
   Teacher-forced agreement cannot see closed-loop drift (a known 0.93
   offline / collapsed live case exists). Epoch ranking must come from
   the live rung.

## What the 4am live attempts actually measured

- `eval_behavior_pool` smoke (ep10): 0 dmg, 4.5 stocks/min lost, all SD.
  This is the pool's DOCUMENTED artifact for delay-untrained policies
  (sync input loop; header records 0-dmg-sync vs 233-dmg-async for the
  same policy) — NOT a policy verdict, and floored so hard it can't
  even rank.
- `eval_live_protocol` async control (ep10): 51% stale sends, loadavg
  31 → starvation law: discard. (Also produced the fixture agreement
  0.889 side-channel — consistent with the offline sweep.)
- Bradley's 4am impression ("forever-crouching fox, ran offstage 4x")
  was one of these starved runs: a policy fed half-stale inputs
  degenerates exactly like this. It is NOT yet evidence about v1.

## The decode discovery (12:12-12:21, ep10, quiet machine)

| arm | staleness | survived 120s | coach score |
|---|---|---|---|
| deterministic (argmax) | 0.0% | **no — 1932 frames, dead in ~32s** | armed 0.0, conv 0/3, **4 deaths** |
| temperature 0.3 | 1.5% | yes (7331 frames) | armed 0.0, conv 0/0, 1 death, 3 passive |
| temperature 0.5 | 1.5% | yes (7323 frames) | **armed 1.0/min, conv 3/4, 1 death** |

Bradley's blind live impressions match exactly: argmax = "crouches then
runs off the stage four times to the left"; T=0.5 = "grabs, lasers,
shines, rolls, pummels, jab-to-multijab, throws, up-B back to ledge when
knocked off, shine wavedash backward, side-B — much more variance",
later "short hop drilling". **The overnight 'starved harness' diagnosis
was WRONG about the cause** (staleness was 0.0% in the clean argmax run
and it still crouched-and-SD'd) — the real cause is argmax mode
collapse. Starvation was a real confound, just not this one.

## Per-epoch live sweep (T=0.5, 3 games x 120s each, cpu dummy, delay 0)

| epoch | armed/min | conversions | deaths | dropped | passive |
|---|---|---|---|---|---|
| ep1 | 0.49 | 4/14 (29%) | 4 | 9 | 5 |
| ep2 | 0.65 | 2/9 (22%) | 5 | 10 | 6 |
| ep3 | 0.72 | 2/9 (22%) | 6 | 8 | 3 |
| ep4 | 0.33 | 3/7 (43%) | 3 | 11 | 2 |
| ep5 | 0.49 | 5/11 (45%) | 6 | 8 | 8 |
| ep6 | 0.33 | 3/10 (30%) | 6 | 10 | 4 |
| ep7 | 0.65 | 3/15 (20%) | 6 | 9 | 2 |
| ep8 | 0.49 | 4/9 (44%) | **1** | **2** | 7 |
| ep9 | 0.33 | 2/6 (33%) | 5 | 9 | 14 |
| ep10 | 0.51 | **7/10 (70%)** | 9 | 13 | 9 |
| **POOLED** | **0.50** | **35/100 (35%)** | 1.7/run | — | — |

**The sweep CANNOT rank epochs and should not be read as ranking them.**
Within-epoch spread equals between-epoch spread (ep4: 0.0/0.98/0.0;
ep8: 1.47/0.0/0.0), n=3, and the standing law is that <2x differences
are unresolved. ep10's 70% conversion and ep8's 1-death/2-dropped runs
are the kind of numbers that look like signal and are not — at n=3
they're one lucky game each. **Actionable form: epochs 1-10 are
behaviorally indistinguishable on this protocol; pick by principle
(latest, or best val) and spend eval budget on things that DO
separate** (decode arms, interp measurements).

Worth noting anyway: even **epoch 1** already plays (0.49 armed/min,
29% conversion). Whatever the corpus teaches, it teaches fast — 10
epochs of val-loss improvement bought no visible behavior change.

## Interpretability read (INTERP_GEN_V1 G1/G2, full detail in that doc)

- **G1 entropy map**: heads live in different regimes — buttons at 69%
  of uniform entropy vs c-stick at 25%. One global temperature is the
  wrong shape; **per-head T is a correction, not a tweak.** Entropy
  tracks game sense (lowest on `edge_danger`/`shine_cancellable`,
  highest on `pummel_throw_decision`/`being_tech_chased`/shield
  pressure). `disadvantage`+`in_hitstun` >> `advantage`+`combo_active`
  → defense is the thin-supervision pocket (v2 curation target).
  `neutral` < `conversion_open` → **the weakness is finishing, not
  entering** (matches 9-13 dropped punishes/epoch above).
- **G2 history dominance**: ratio **0.391** (current state moves the
  distribution 2.6x more than 50 frames of history); idle-tiled
  stay-mass 0.202 vs active 0.208 = **no idle stickiness**. Exposure
  bias is NOT structural → **v2 does not need scheduled sampling**, and
  the crouch loop is confirmed a decode pathology, not a memory one.

## Prediction scorecard (registered in DIRECTIONS_2026-08-25 §D1)

| Prediction | Verdict |
|---|---|
| No pilot-style idle collapse | **SPLIT — the sharpest lesson of the day.** Under sampling: correct, no collapse, full repertoire. Under argmax: collapse WORSE than the pilot's. The prediction assumed decode was neutral; it is the dominant variable. |
| Mechanical hygiene present | **HELD** — live: shines, wavedashes, up-B ledge recovery, short-hop drilling, l-cancel-shaped play (Bradley's blind impressions). |
| Dithering/averaging at contested points | **HELD, and now MECHANISTIC** — G1 shows buttons at 69% of uniform entropy; argmax over that distribution is exactly the observed crouch. |
| Off-distribution collapse vs CPUs | **PARTIAL** — 2-14 passivity windows/epoch persist at T=0.5, but the bot still approaches and converts. Not collapse; a tail. |
| Offstage SD class present | **HELD** — deaths 1.7/run at T=0.5 (and the argmax arm SD'd 4x in 32s). |
| Execution good, not ms_g19-crisp | **HELD** — shines appear (max chain 2 in sweep scoring) but no multishine chains. |
| Stock table (beginner→pro) | **PENDING** — needs human sessions; CPU numbers can't settle it (g6 law). |

## Next steps

**Immediate (cheap, high value):**
1. **Per-head temperature bracket** — G1 says buttons want colder T
   than sticks. Needs a small decode change (per-head T in Sampling),
   then a 3-arm live bracket. Highest-leverage decode work.
2. **Temperature bracket proper** on a fixed epoch: 0.4 / 0.5 / 0.7 x
   5+ games — the one comparison this protocol CAN resolve if the
   effect is >2x (0.3-vs-0.5 already was).
3. Bradley live look / play-against session on ep10 at T=0.5 (the
   human-gated rung; the g6 law says CPU numbers can invert).

**Program-level (now evidence-backed):**
4. **D2 value model** is the top investment — G1+G2 together say the
   model contains the behavior and holds appropriate uncertainty but
   lacks a SELECTION RULE. Best-of-N with a value model is the direct
   fix, and the coach product (D11/C2) wants the same artifact.
5. **v2 recipe**: multi-delay {2,3} baked in (v1 is delay-0-only and
   undeployable to netplay), defensive/disadvantage curation mix (G1
   finding 3), punish-continuation emphasis (G1 finding 4). **Skip
   scheduled sampling** (G2). Bigger backbone is the open capacity
   question (~3M params on ~90M frames = underfit).
6. Yeti ingest (D3) and OGSwaglord (D4) unchanged; G5 (style
   separability) would tell us how cheap D4 is.

## Side results banked overnight

- `export_epoch_policies.exs` (new): .axon → per-epoch policy .bin with
  correct canaries; guard #6 exercised 10x.
- **Streaming+resume smoke PASSED** (04:57): `--resume` from the 30-file
  smoke's epoch2 .axon restored optimizer step 478 and continued val
  descent (7.934 → 7.848) on the streaming pipeline. Extending
  fox_gen_v1 beyond 10 epochs later = tested capability
  (`--resume checkpoints/fox_gen_v1_20260825_210355_epoch10.axon`,
  embedding cache warm, ~66 min/epoch).
- Ops: ALL live evals invalid under heavy CPU load — check
  `pgrep -fc permut` / loadavg BEFORE any live rung. The pool regime is
  unusable for delay-0-trained policies (floored, no ranking signal).
