# Reaction-4 policy control via the proof recipe (2026-09-14)

Closes PIPELINE_PROOF gate 1 ("restore a passing policy control at reaction
4 and explain the failed one"). Root: `eval_runs/0914_delay4_proof/`,
launcher `run.sh` (six recorded steps).

## Question

ep57 (g-line, ids 3/4/5) chains 14 at reaction 2 and 1 at reaction 4, and
the 09-13 sessions could not separate "harness rung 4 is wrong" from "id 4
is badly trained". The delay-2 proof passed every gate, so: the same recipe
with every 2 replaced by 4 (queue depth k+1 = 5, warm prefix 16-1+5 = 20),
played at `--reaction-delay 4`. If it chains, the harness is fine.

## Candidate

`eval_runs/0914_delay4_proof/round21/candidate.bin`
SHA256 `424ae2f1e7fad4bc40efdac2836a7f0d84b70ed4140afeb3e12cf60d68a7d4ef`.
windowed_gru_f32_v1, GRU64x2, AR head, window 16, **queue 5, delay 4**,
prev action on, dropout 0.0. FRESH zero-state/F32 initialization
(`initial.bin`, 737ee298...): the delay-2 initialization is 328 wide (queue
3) and this recipe is 352 wide (queue 5); checkpoint guard #6 refused the
transplant on the first attempt (`progress_attempt1.log`). 21 epochs on the
canonical fixture + 18 clips re-exported at delay 4 (10,603 supervised
targets), final loss 0.0014.

## Pipeline checks (all recorded in `progress.log`)

- Exporter refactor regression: the default export reproduces `clips_v6`
  (18 clips semantically identical; byte compare is NOT stable across
  exports — `scripts/compare_recorded_clips.exs`).
- The four teacher scoreboards validate at delay 4 (`targets/*_d4.json`:
  116/116/116 and 356-target windows, all valid).
- Agent parity at delay 4 with the trained candidate: max |diff| 1.19e-7
  over the first 18 supervised windows of all 18 clips (`clips_parity/`).
- Frozen cold+warm gate: 18/18, minimum early joint probability 0.9986.

## Closed loop at `--reaction-delay 4` (T=1.0, libmelee AF, 2 runs/handoff)

| gate | result |
|---|---|
| cold familiar controls (120f) | **12/12**, chains 13-14 — including sustain-900 (the delay-2 candidate's cold regression) |
| warm familiar controls (120f) | **12/12**, chains 13-14 |
| isolated interruptions, neutral opponent (360f) | **5/6** strict recoveries, 37-39 full cycles, ready-to-cycle 9f; 502 run 1 = ready at +6 then an empty hop and 311 frames CROUCHING (chain 1) |
| interruptions, replay opponent (360f) | **6/6** recover: 39/39, 38/26, 14/16 cycles; ready-to-cycle 9-10f first time, 75-76f after a re-hit |

All 36 runs: valid timing, zero prefix drift, zero errors (no
`dolphin_disconnected` this time, cold included).

## Verdict

**The reaction-4 rung is fine.** A policy trained at delay 4 on the proof
recipe chains 13-14 on every familiar handoff and recovers from
interruptions at reaction 4. ep57's chain-1 at reaction 4 is a property of
that checkpoint's training (its ids 3/4/5 pool: held off-loop labels,
rollout coverage), not of the harness. Gate 1 closed; the g-line's
reaction-4 numbers can be read as training results again.

Two residuals worth carrying:
- **Crouch stall** (1/12 interruption runs): after readiness the policy
  empty-hops and then crouches for 5 s. The prior candidate showed the same
  stall at delay 2 (RECOVERY_CONTROLS: 236 frames in action 40). Rare, but
  it is the failure mode that survives everything else.
- The cold sustain-900 regression did not reproduce at delay 4 (12/12
  cold). It remains a delay-2-candidate finding, not a recipe property.

## Cost

Whole chain 11.5 minutes (export 5 s, train 90 s, parity 10 s, fit 40 s,
live 9 min).
