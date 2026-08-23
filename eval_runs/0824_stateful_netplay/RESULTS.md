# 0824_stateful_netplay — stateful-step netplay rung: NO FLIP (behavior regresses)

The netplay-human rung for the JIT lever 2b (`--stateful-step`), vs
DBTD#411 (Bradley). Verdict up front: **do not flip the script
default.** Latency is fine; chaining is not.

## Canonical scores (ShineChain v3 over bot-side .slp — the only valid chain metric)

| game | arm | shines/min | max chain | sustained (>=2) |
|---|---|---|---|---|
| T115434 | stateful, d4/**id4** | 47.9 | 1 | 0 |
| T115856 | stateful, d4/**id4** | 48.5 | 2 | 1 |
| T120459 | stateful+resync60, d4/**id4** | 69.7 | 1 | 0 |

Reference: windowed netplay record (0822, same checkpoint, d4/**id3**)
= 87.6 shines/min, chain 62.

## Two launch/measurement errors (both already-documented laws)

1. **Untrained delay-id confound**: launched with bare
   `--frame-delay 4` → delay_id=4, UNTRAINED. The 0822 record knobs
   are d4 + `--delay-id-override 3`. So these netplay games cannot
   separate "stateful damps chains" from "id4 damps chains" — both
   arms are contaminated.
2. **Wrong chain metric mid-session**: in-session comparisons used
   qtrace applied-B-press runs (commanded inputs), violating the
   "score chains from replays, never qtrace presses" rule
   (0822_netplay_crown). Those numbers (e.g. "windowed 44 vs
   stateful 7") are void.

## The clean evidence: local d3 A/B (trained id), same morning

eval_runs/0824_stateful_live replays, scored canonically:

| arm | shines/min | max chain | sustained (>=2) |
|---|---|---|---|
| stateful (T041536) | 38.9 | 6 | 2 |
| windowed (T042421) | 61.4 | 8 | 8 |

Same settings, same CPU-3 opponent, trained delay-id — **stateful
still chains ~4x less and shines ~60% as often.** So the stateful
deficit is real independent of the id4 confound. Mechanism: the step
path carries GRU state continuously; training only ever saw 60-frame
windows; behavior is pinned identical for the first 60 frames
(equivalence test) and diverges after. `--stateful-resync 60` did not
rescue netplay chaining, but that arm was id4-confounded — resync at
a trained id is untested.

## Standing verdict + next

- `--stateful-step` stays DEFAULT-OFF for play. It remains correct
  and mandatory for headless probes (GOTCHA #69) and is fine wherever
  chain fidelity is not the metric.
- The 19.9s→1.5s JIT win therefore does NOT ship via stateful for
  live deploys. Next JIT levers for the windowed path: resident
  policy server (JIT_WARMUP.md option 3) or ONNX runway (option 4).
- Open follow-ups: (a) stateful+resync at trained id, local
  chain-scored — does as-if-windowed resync actually restore chains?
  (b) if yes, netplay retest at d4/id3.
- qtrace both netplay arms: not analyzed for lag here (behavior
  already decided the flip); logs retained (g1.log, g2_resync60.log).

## Addendum: resync repair + retest at record knobs (same day)

- **Local d3 vs CPU-3 (eval_runs/0824_resync_local): stateful+resync60
  RESTORES windowed-grade chaining** — 57.2 shines/min, max 9,
  sustained 9 (windowed control: 61.4/9/8; plain stateful: 38.9/6/2).
  Same run validated the new qtrace `act=` canonical chain line
  bit-exact against replay scoring (129 shines, identical histogram).
- Netplay retest at RECORD knobs (d4 + --delay-id-override 3,
  stateful+resync60, g3_resync60_id3.log): lag sharp 6 @ 99.7%; game 1
  max chain 3, game 2 aborted ~25s. Max 3 is INSIDE the windowed
  netplay per-game distribution ({3,1,46,6} / {4,62} / {23,12} — most
  games score 1-6; records are tail events), so n=1 cannot separate
  the arms. Default flip stays open pending tail events on the resync
  arm from organic sessions.
- Identity checks: ms_g19_ep4.bin mtime 2026-08-20, unchanged since
  before the 0822 chain-62 session — today's arm played the identical
  bytes. ep4 remains UNCROWNED (ms_g15 = champion-of-record; the
  ep4-vs-g15 stage-pinned decider is still owed).
- EXPHIL_GPU_MEMORY_FRACTION=0.25 ruled out as a behavior factor:
  identical to the record session's env; caps allocator pool only;
  latency stayed sharp.
