# fox_gen_v1 sweep — 2026-08-26 overnight (LIVE RUNG PENDING)

**Status: training + offline evals complete; LIVE behavioral verdict
BLOCKED on machine load** (a ~29-process `./permut` python job has all
cores at ~78% since before the eval window; live harness starved at 51%
stale sends — any live score under this load measures the machine).

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

## Prediction scorecard (registered in DIRECTIONS_2026-08-25 §D1)

| Prediction | Verdict so far |
|---|---|
| No pilot-style idle collapse | **UNDECIDED** — starved harness reproduced idle-like behavior; needs clean run |
| Mechanical hygiene present | Offline: press-rate + agreement consistent with yes; live pending |
| Off-distribution collapse vs CPUs | Pending clean live |
| Offstage SD class present | Pending (starved runs can't count) |
| Stock table (beginner→pro) | Pending human/live sessions |

## Next steps (in order, when machine is quiet)

1. **Clean async control on ep10** — THE discriminator: staleness <10%
   and the crouch/SD pattern persists → real policy flaw (BC drift);
   pattern vanishes → 4am was harness starvation. One run answers it.
2. Per-epoch async ranking (3 games each, cpu dummy, frame-delay 0 —
   v1 is delay-0 trained; the delay guard enforces this) via
   `eval_live_protocol.sh --server`.
3. `coach_report.exs --char fox` on the ranked replays → approaches/min,
   conversion, opener entropy → finish the scorecard.
4. Bradley live look at the top 1-2 (the human-gated step; g6 law).

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
