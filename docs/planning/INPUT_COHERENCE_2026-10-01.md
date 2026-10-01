# Input coherence: the flicker scoreboard and the MinGRU testbed (2026-10-01)

Follows `ENGAGEMENT_SCAN_2026-09-29.md`. Bradley's direction: understand the
flaw before more training; iterate on the cheapest architecture (MinGRU),
then port the fix; small Mamba is the fallback testbed.

## Scoreboard: `scripts/offline_input_coherence.exs`

Held-out expert game states → the live Agent API, sampled as in play (a
prev-action policy feeds back its own output) → compare the emitted
controller stream with the expert's on the same frames. ~3 min per policy,
no Dolphin. 3 validation games, 29,008 frames. Results in
`eval_runs/1001_coherence/*.json`.

| | Mamba v1 ep2 | Mamba v2 prev-action | v2 ablated (zeros) | MinGRU smoke | expert |
| --- | --- | --- | --- | --- | --- |
| A press-edges / min | **119** | 19 | 33 | **123** | 25 |
| B | 73 | 25 | 19 | 125 | 21 |
| X | 108 | 33 | 31 | 76 | 22 |
| Y | 134 | 26 | 31 | 212 | 30 |
| L | 220 | 46 | 32 | 212 | 41 |
| R | 116 | 21 | 30 | 189 | 8 |
| A edges per expert jab-1 episode (n=8) | 2.4 | 0.6 | 1.1 | 1.75 | 0.125 |
| output repeats previous frame | 0.34 | 0.72 | 0.73 | 0.16 | 0.64 |
| change recall (buttons == expert's on change frames) | 0.32 | 0.30 | 0.40 | 0.32 | – |
| hold agreement | 0.67 | 0.54 | 0.68 | 0.57 | – |
| fully neutral share | 0.32 | 0.25 | **0.82** | 0.30 | 0.27 |

Findings:
1. **Flicker is confirmed, general, and large.** Without its own previous
   input the policy presses EVERY button ~5× as often as the expert (A 119
   vs 25/min; R 116 vs 8). Multi-jab was one visible symptom of a
   whole-controller problem.
2. **The prev-action channel fixes the edge rates** (within ~±30 % of
   expert on most buttons) — the channel does what it is for.
3. **It does not improve change recall** (0.30 vs 0.32): the model is no
   better at knowing WHEN to change input; it mostly learned to hold.
4. **This scoreboard cannot see the live freeze** — the states are the
   expert's, so they never show the consequences of the model's own inputs.
   v2 looks healthy here (neutral 0.25) and self-destructs 4.5×/min live.
   It DOES reproduce the ablated mode's idling (0.82 here, 0.84 live).
   Freezing needs a closed-loop measure (sim rollouts or live games).
5. **The flaw is backbone-independent**: a 430k-parameter MinGRU trained for
   25 seconds on 111 games shows the same 5× edge rates.

## MinGRU testbed

`scripts/train_fox_mamba.exs` now accepts `--backbone min_gru` (name is
historical). Smoke: hidden 256×2, window 80, batch 128, 111 train / 12 val
games (`--max-files 400`), 1601 updates at **8–9 ms/update** (Mamba 512×2:
33 ms) — the whole run is ~25 s of training after ~70 s of replay
discovery + parse. val 3.13. No fused custom call for MinGRU yet (pure
graph); not needed at this scale. Checkpoint
`checkpoints/mingru_smoke_20261001/`.

Caveat: the scoreboard used the Mamba split's validation games; with
`--max-files` the testbed has its own split — pass `--split
<ckpt-dir>/split.json` so the games are held out for the model under test.

## Experiment queue on the testbed (each: train ~2 min, score ~3 min)

Baseline = the smoke config. One change at a time, score = press-edge
ratios (flicker) + change recall; then the best 1–2 get a closed-loop check
for freezing before anything is ported to the Mamba.
1. prev-action, dropout 0 / 0.15 / 0.5 — does more dropout raise change
   recall, and where does edge rate break?
2. prev-action + scheduled sampling (verify it is wired in streaming first).
3. press-event (edge) target without a feedback channel.
4. joint button categorical — expected NOT to fix temporal flicker; include
   as a control for that claim.

## Closed-loop evals (same day) — the freeze is now measurable without a human

### Batched agent path (`lib/exphil/agents/agent.ex`)
`batch_init/2` now accepts WINDOWED policies (rolling `{n, window, dim}`
tensor through the same `predict_fn`; cold rows tile their first frame like
`pad_sequence`) and the PREV-ACTION channel (per-row last emitted
controller; `batch_observe(..., controllers: [...])` warms it from recorded
history). Previously the batch path was carried-state GRU only and rejected
prev-action. Parity, deterministic, 150 frames: single path == batch row
150/150 for Mamba v1 and for v2 with the channel.

### `scripts/sim_closed_loop.exs` — policy drives port 1 in the NIF sim, 32 envs × 1800 frames vs an idle opponent (~35 s)

| | Mamba v1 | Mamba v2 prev-action | v2 ablated | MinGRU smoke | (live, Bradley) |
| --- | --- | --- | --- | --- | --- |
| SD / min | 1.25 | **3.75** | 0.44 | 0.44 | v1 1.05 · v2 4.5 · ablate 1.1 |
| damage dealt / min | 75.6 | **13.9** | 4.6 | 85.6 | |
| kills / min | 1.31 | 0.25 | 0.0 | 0.0 | |
| neutral controller share | 0.40 | 0.54 | **0.96** | 0.33 | v1 0.25 · v2 0.37 · ablate 0.84 |
| output repeats previous | 0.49 | 0.77 | 0.95 | 0.15 | |
| self-inflicted offstage → back | 0.77 | 0.56 | 0.70 | 0.88 | |

It reproduces all three live observations: v1 plays and SDs about once a
minute; v2 SDs 3× as often and barely fights; v2-with-zeros stands still.
This is the eval the offline scoreboard could not be.

### `scripts/recovery_drill.exs` — 36 offstage starts, 8 trials each, neutral opponent

Starts are captured in the sim (v1 vs v1; port 1 leaving hitstun airborne and
offstage; kept only if `ExPhil.Melee.Checkmate` says recoverable), with the
previous 79 states + inputs to warm the window and the prev-action channel.

| | recovered | cases never recovered |
| --- | --- | --- |
| neutral controller (floor) | 0.139 | 31/36 |
| Mamba v1 ep2 | **0.267** | 12/36 |
| Mamba v2 prev-action | 0.174 | 16/36 |
| v2 ablated | 0.115 | 27/36 |
| MinGRU smoke | 0.156 | 18/36 |

Ordering matches (v1 > v2 > ablated ≈ doing nothing), and the absolute
level is the bigger finding: **the best imitation policy recovers about a
quarter of positions the static model calls recoverable.** Caveats: the
Checkmate "recoverable" label means a plan exists, not that it is easy; no
expert baseline here (see next); 36 cases.

Two harness bugs caught on the way, both by a validity control: (1) the
first drill mined cases from held-out replays and seeded them with
`Sim.Seed.from_replay` — replaying the EXPERT's own inputs recovered 10 %,
because the seeded sim state was not the replay's state on Fox-vs-Marth /
Falco games on PS/BF (positions differ by tens of units by frame ~900;
`divergence` reports nil). Replay seeding is NOT bit-exact outside the
configuration it was validated on — GOTCHA #137. (2) deaths were missed
when judged by the stock field alone (a dead Fox on the respawn platform
scored as "timeout", then "recovered" once it stepped off) — outcome rule
now includes death/rebirth action states 0–13.

Gate for the experiment queue, per variant: offline press-edge ratios
(flicker) AND closed-loop SD/min + damage/min + neutral share (freezing),
plus the recovery drill. A variant must beat v1-style baselines on flicker
without losing on the closed-loop numbers.
