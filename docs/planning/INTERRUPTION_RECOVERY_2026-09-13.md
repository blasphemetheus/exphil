# Human-session inspection and interruption-recovery baseline

## Original session

Source: `eval_runs/local_multishine_20260913_154147/2026-09-Mainline/Game_20260913T154159.slp`.
Fox port1 vs human Fox port2 on Final Destination. Policy association comes from
the launch script/session context, not cryptographic identity inside the replay.

Existing `MultishineBenchmark`/ShineChain v3 scores8,203 playable frames (~137s):

- **Maximum chain27**,124 grounded shine onsets,78 completed cycles,4 stocks lost.
-46 hitstun-onset episodes:9 eventually complete a full cycle,33 are censored by
  another hitstun onset,4 by death. These are episodes, not individual hit counts.
-15 reach the grounded zero-hitstun readiness proxy. Among9 completed episodes,
  ready-to-cycle times are31,289,9,9,8,9,8,8,8frames. Most successful returns are
  quick, but one takes nearly5seconds after readiness.
- The other31 hit episodes never reach that proxy before interruption/death.
  Being airborne or animation-locked is not the same as refusing to act; this
  proxy is conservative and is **not** an exact actionable-frame detector.

This supports the user's observation: strong uninterrupted multishining, but
inconsistent re-entry after interference. Do not label33 renewed interruptions
as33 independent failed decisions or report success-only latencies without them.

## Frozen development benchmark

Same candidate: `eval_runs/0913_no_dropout_fit/round21/candidate.bin`.
SHA256: `404a1f543889ddd4193380745fbc603c6c94bb4a200a3caa7cbca6ddf823182a`.

Before collecting responses, selected the first observed example of each:
no return after reaching readiness (hit502), fast return (hit1389), and slow
return (hit3152). Manifest: `eval_runs/0913_interruption_recovery/manifest.json`.
This is mined development evidence, **not held-out confirmation**.

Two stochastic runs per handoff,360 response frames (6s), temperature1.0,
reaction-delay2, libmelee AF conversion. Both players replay recorded prefix
inputs; policy gets committed input history, then controls port1. Port2 continues
the human's recorded inputs: deterministic/non-reactive, not a new human match.

Scoring uses the existing strict full-cycle metric, not an isolated shine or
the scenario suite's looser two-shine proximity counter. Report disruption time
and readiness separately. Benchmark deadline:60frames from readiness to a full
cycle, with re-hits, deaths, and observation cutoff explicitly censored.

## Live result

**6/6 valid runs**, no drift at handoff beyond tolerance, no timing invalidity,
no errors or truncation. **0/6 completes a strict full cycle** within its360f
response. This is a bounded observation, not a claim all six had a fair60-frame
actionable opportunity.

| Handoff | Run | First readiness | Outcome / observed follow-up |
|---|---:|---:|---|
| hit502 | 1 | never | death |
| hit502 | 2 | never | response cutoff |
| hit1389 | 1 | frame1548 | no cycle over201 subsequent frames |
| hit1389 | 2 | frame1715 | cutoff after only34 subsequent frames |
| hit3152 | 1 | frame3184 | death after172 subsequent frames |
| hit3152 | 2 | frame3205 | no cycle over307 subsequent frames |

Three runs provide at least60frames of observation after readiness; one provides
only34, and two never become ready. Time after first readiness does not guarantee
continuous actionability. Several runs produce single shines or empty hops, but
none passes the stricter full-cycle definition. In the sliced-replay metric,
`end_of_replay` means **end of the response window**, not necessarily game end.

## Interpretation and next order

The prior12/12 proof covered trained cold handoffs. This benchmark introduces
unfamiliar actual hitstun/airborne trajectories **and warmed history**. It is not
a clean ablation proving that all degradation is caused by training coverage.
The original fast human-session return need not repeat under sampled policy
actions and a non-reactive replayed opponent.

1. Add known-good handoff controls under the same committed-history convention,
   plus isolated-hit cases where the opponent stops attacking. Keep this policy
   frozen; distinguish history sensitivity, re-hit pressure, and recovery ability.
2. Validate a recorded teacher through the specific airborne/landing/restart
   failures before accepting its correction labels. Do not blindly label hitstun
   frames as a requirement to shine.
3. Only then add those transitions to supervised fitting; retain the existing
  12/12 multishine regression gate and these interruption cases.

No training, checkpoint replacement, or readiness-threshold relaxation occurred.

## Artifacts and checks

- `eval_runs/0913_interruption_recovery/observed.json`: original replay metrics.
- `live/eval_candidate.json`: drift/timing gates and live observations.
- `live/recovery_v2.json`: strict per-response cycles and censored recovery events.
- `live/eval_candidate/`: generated replays and command traces.
- `scripts/score_interruption_recovery.exs`: offline response scorer.
- `eval_runs/0913_interruption_recovery/run.sh`: source-loaded six-run collection
  and scoring recipe; intentionally refuses to overwrite its output directory.
-10 existing benchmark regression tests pass, including full-cycle, hitstun,
  gap/death/re-hit censoring and real-replay tests. Source hashes retained.
