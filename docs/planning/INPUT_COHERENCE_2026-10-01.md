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

## Queue item 1 result (15:20) — per-frame prev-action dropout does not fix freezing

MinGRU 256×2, 3000-file slice, same split + seed; unit `exphil-coh-queue1`,
outputs `eval_runs/1001_queue/<name>/`. Offline rows feed the policy its OWN
previous output; closed loop = 32 envs × 1800 frames vs idle; drill = 36 × 8.

| | base (no channel) | prev, dropout 0 | prev, dropout 0.15 | prev, dropout 0.5 | expert |
| --- | --- | --- | --- | --- | --- |
| val loss (teacher-forced for prev) | 2.43 | 1.04 | 1.14 | 1.44 | |
| A press edges / min | 100 | 15.5 | 14.8 | 8.7 | 13.2 |
| R press edges / min | 247 | 28.6 | 24.7 | 6.2 | 8.3 |
| change recall | 0.238 | 0.289 | 0.267 | 0.029 | |
| hold agreement | 0.627 | 0.573 | 0.530 | 0.029 | |
| closed loop: damage dealt / min | **56.8** | 10.5 | 20.4 | 0.5 | |
| closed loop: kills / min | 1.06 | 0.13 | 0.19 | 0.0 | |
| closed loop: SD / min | 1.25 | 1.13 | 1.13 | 0.06 (does not move) | |
| closed loop: output repeats previous | 0.29 | 0.62 | 0.62 | 0.83 | 0.76 |
| recovery drill: recovered | **0.278** | 0.167 | 0.146 | 0.104 | |
| recovery drill: cases never recovered | 5/36 | 19/36 | 19/36 | 25/36 | |

Channel zeroed at inference (`coherence_ablate.json`): dropout 0.15 → neutral
0.94; dropout 0.5 → neutral 0.81, A edges 53/min.

Reading:
- **The testbed is faithful.** MinGRU reproduces the Mamba pair: the channel
  brings press edges to expert rates and costs 3–5× damage and ~40 % of
  recoveries. Experiments here should transfer.
- **Change recall barely moves (0.24 → 0.29).** The channel does not teach
  WHEN to change input; it teaches "repeat what I just did", which is right
  on 76 % of frames. Flicker and freezing are the same missing skill under
  two samplers: without the channel each frame is sampled afresh (too many
  edges); with it the policy copies (too few decisions).
- **Per-frame dropout is the wrong tool for a recurrent model.** The mask is
  drawn independently per frame (`data.ex` `slot_at`), so within an 80-frame
  window the model still sees its previous input on most frames and can
  carry it across the gaps in its recurrent state. It never has to act
  without the channel — which is why a dropout-trained model idles when the
  channel is zeroed for a whole window (0.81–0.94 neutral), and why raising
  the rate did not help. To train "play without it" the mask must cover the
  whole window (per-sequence dropout).
- Dropout 0.5 with the channel on latches a non-neutral held input (neutral
  0.005, hold agreement 0.03). Mechanism not established; it is a
  self-feedback effect (teacher-forced val is a normal 1.44).

Consequences for the queue: drop "heavier dropout". Remaining candidates,
each tied to the diagnosis above:
1. per-WINDOW channel dropout (p≈0.3–0.5): the policy must be competent
   both with and without the channel; tests whether the copy shortcut is
   what crowds out the state-driven behaviour.
2. scheduled sampling (feed the model's own sampled previous output during
   training): attacks the train/play mismatch directly; streaming wiring to
   verify first.
3. press-event target, no channel: makes "when to change" the thing the
   loss scores.
4. joint button categorical, control.

## Carried-state Mamba inference (15:25)

`stateful_step: true` now works for Mamba (single + batched paths):
`Edifice.SSM.Mamba.step/3` carries the SSM state and a conv ring buffer per
layer instead of re-running the 80-frame window. On Mamba v1 ep2 (trained
WINDOWED, never saw more than 80 frames from a zero state):

| | windowed | carried |
| --- | --- | --- |
| argmax controller agreement vs windowed, frames 0–79 / 80–299 / 300–1499 | — | 0.988 / 1.000 / 0.993 |
| single-agent median latency | 2.23 ms | 0.92 ms |
| closed loop 32 envs × 3600: SD / min | 1.28 | 1.28 |
| damage dealt / min | 77.0 | 79.1 |
| kills / min | 1.44 | 1.91 |
| batched step | 17 ms | 14 ms |

The windowed-trained model tolerates unbounded carried state: its effective
memory is shorter than the window, so the two modes compute nearly the same
function. Carried inference is therefore safe to use today (2.4× cheaper per
frame); `--stateful-resync` is not needed. It also means carried-state
TRAINING (fused scan kernel taking/returning state, contiguous chunks) would
only pay off if the model learns to use context beyond 80 frames — a
capacity bet, not a prerequisite. Not started. Test:
`test/exphil/agents/agent_stateful_mamba_test.exs`; outputs
`eval_runs/1001_carried/`.

## Queue 2 result (10-01 16:16) — block dropout gives two policies in one, not a fix

MinGRU, prev-action with dropout drawn per 240-frame block
(`--prev-action-dropout-block 240`), each scored with the channel on and
with it zeroed at inference. Unit `exphil-coh-queue2`, 42 min, clean exit.

| | base (queue 1) | blk 0.3, channel on | blk 0.3, zeroed | blk 0.5, channel on | blk 0.5, zeroed | expert |
| --- | --- | --- | --- | --- | --- | --- |
| val loss | 2.43 | 1.56 | | 1.81 | | |
| A press edges / min | 100 | 21.2 | 114 | 13.2 | 92.7 | 13.2 |
| R press edges / min | 247 | 28.9 | 228 | 42.0 | 227 | 8.3 |
| change recall | 0.238 | 0.271 | 0.283 | 0.300 | 0.271 | |
| damage dealt / min | 56.8 | 22.0 | 29.8 | 17.7 | 29.9 | |
| kills / min | 1.06 | 0.25 | 0.25 | 0.38 | 0.56 | |
| SD / min | 1.25 | 1.06 | 0.69 | 1.25 | 0.69 | |
| output repeats previous | 0.29 | 0.69 | 0.28 | 0.65 | 0.31 | 0.76 |
| recovery drill | 0.278 | 0.174 | 0.233 | 0.194 | 0.267 | |

Reading:
- Block dropout did what per-frame dropout could not: with the channel
  zeroed the policy now PLAYS (neutral 0.29–0.45, not 0.81–0.94) and
  recovers like the base model (0.23–0.27).
- But it did not merge the two behaviours. With the channel on it is the
  prev-action policy again (edges at expert rate, damage 18–22, recovery
  0.17–0.19); with it off it is the base policy again (5–10× edges). One set
  of weights holds both modes and the channel selects between them.
- So the low-damage mode is not caused by the channel crowding out
  state-driven skill during training — the skill is there (zeroed rows). It
  is caused by what the policy does when it SEES its own last input: it
  defers to it. That is an inference-time feedback effect, which
  input-side masking cannot reach.
- Change recall is 0.27–0.30 in all six columns, as in queue 1. No variant
  so far has moved the thing that would fix both symptoms.
- The zeroed rows deal about half the base model's damage (30 vs 57); the
  capacity is shared between two modes.

Dropout on the input (per frame or per block) is closed as a fix. What is
left attacks either the feedback itself (scheduled sampling: train on own
sampled outputs) or the objective (press-event target: score the decision
to change).

## Queue 3 result (10-02 11:19) — press/release event button head: flicker fixed without a trunk channel, closed loop mixed

`--button-events` (6ec9fd9f): the AR button head emits 16 logits (press-if-up,
release-if-down); the previous button state selects per button
(`Heads.collapse_button_events/2`), so the usual BCE trains hazards and the
usual sampler draws them. The previous buttons reach ONLY the head; the trunk
sees the prev-action slot zeroed (training: `Loss.policy_forward_inputs/4`
event clause; agent: embeds without the channel, hands the sampler its last
buttons). Sticks are unchanged (no feedback, sampled per frame). MinGRU,
same slice/seed, unit `exphil-coh-queue3`.

| | base | prev-action d0 | **event head** | expert |
| --- | --- | --- | --- | --- |
| val loss | 2.43 | 1.04 (teacher-forced) | 2.30 | |
| A / B / R press edges per min | 100 / 70 / 247 | 15.5 / 22.9 / 28.6 | 18.9 / 16.9 / 29.5 | 13.2 / 11.0 / 8.3 |
| change recall | 0.238 | 0.289 | 0.284 | |
| output repeats previous (offline) | 0.20 | 0.65 | 0.28 | 0.76 |
| damage dealt / min | 56.8 | 10.5 | 23.7 | |
| kills / min | 1.06 | 0.13 | 0.31 | |
| SD / min | 1.25 | 1.13 | **2.81** | |
| offstage episodes / min | 5.5 | 7.7 | **9.9** | |
| recovery drill | 0.278 | 0.167 | **0.309** | |
| drill cases never recovered | 5/36 | 19/36 | 12/36 | |

Reading:
- Button flicker is fixed with NO previous-action input to the trunk: press
  edges are in the expert band on every button (R still 3.5×). So expert
  press rates do not require the trunk to see its own input — a head-level
  connection is enough. This is the part of slippi-ai's design (each head
  component sees its own previous value) that matters for buttons.
- Recovery drill is the best of any variant (0.309).
- Free play got worse than base: damage 24 vs 57, and it leaves the stage
  almost twice as often and self-destructs 2.3× as often. Mechanism NOT
  established. Sticks still change nearly every frame (repeat 0.28 vs expert
  0.76), so the model now pairs coherent button holds with incoherent sticks;
  whether that mismatch is the cause is a hypothesis, not a finding.
- Change recall is 0.28 again.

Not a fix on its own. Open question it raises: give the sticks the same
treatment (head-level previous stick → hold/change hazard) vs scheduled
sampling on the full channel.

## Eval suite v2 (10-02) — distance from the EXPERT, with noise bars

Until now every eval compared variants with each other. New instruments, all
committed, all run by `scripts/coherence_experiment.sh` (`EVALS="coherence
closed_loop recovery calibration fidelity"`, default all):

| Instrument | Script / module | Answers |
| --- | --- | --- |
| Expert reference | `scripts/expert_reference.exs` → `eval_runs/1002_fidelity/expert_fd.json` | `ExPhil.Eval.PlayStats` over 150 expert Fox games on FD (390 min), plus split-half distances = noise floor |
| Fidelity scorecard | `scripts/fidelity_scorecard.exs` | policy plays ITSELF in the sim (32 envs × 3600 frames × 3 seeds); same PlayStats; total-variation distance per histogram + rates vs expert |
| Technique rates | inside the scorecard | L-cancel rate (landing-lag modes), short-hop share (jump-peak modes), tech rate, wavedashes, dashes |
| Calibration | `ExPhil.Eval.Calibration`, `CALIBRATE_ONLY=1 scripts/train_fox_mamba.exs … --resume` | teacher-forced: button ECE, per-head loss, P(down \| previous state) model vs expert |
| Change events ±k | `offline_input_coherence.exs` | press / release / stick-zone events recalled within ±0/2/5 frames, and precision |
| Noise bars | 4 runs of closed loop + recovery; 3 seeds of the scorecard | which differences are real |

Caveat on the reference: expert games are Fox vs human opponents of many
characters; the sim is a Fox ditto against the same policy. Own-input and
own-movement distributions transfer; damage/kill rates depend on the opponent.

### Noise bars on the old evals (mean ± sd, n = 4)

| | base | event buttons | prev-action d0 |
| --- | --- | --- | --- |
| recovery drill | 0.272 ± 0.024 | 0.290 ± 0.038 | 0.157 ± 0.014 |
| vs idle: damage / min | 61.6 ± 11.7 | 20.9 ± 2.7 | 9.5 ± 2.7 |
| vs idle: SD / min | 1.14 ± 0.27 | 3.27 ± 0.32 | 1.08 ± 0.23 |

So: base vs event-buttons recovery is NOT a real difference (earlier "best
yet 0.309" was noise); prev-action's recovery deficit and all three damage
levels are real; event-buttons' extra SDs are real.

### Fidelity scorecard (self-play, 3 seeds; expert split-half floor in brackets)

| | base | event buttons | prev-action d0 | expert |
| --- | --- | --- | --- | --- |
| **fidelity distance** (mean of 7) | 0.360 ± 0.011 | **0.225 ± 0.002** | 0.243 ± 0.006 | [≈0.03] |
| button hold lengths | 0.840 | 0.246 | 0.311 | [0.045] |
| stick dwell | 0.279 | 0.282 | **0.087** | [0.014] |
| action-state mix | 0.351 | 0.285 | 0.305 | [0.014] |
| position | 0.225 | 0.125 | 0.136 | [0.012] |
| landing lag | 0.237 | 0.333 | 0.567 | [0.071] |
| jump peak | 0.435 | 0.171 | 0.143 | [0.040] |
| input repeats previous | 0.24 | 0.33 | 0.65 | 0.76 |
| SD / min | 0.83 ± 0.15 | 2.34 ± 0.33 | 2.54 ± 0.04 | 0.44 |
| offstage return rate | 0.83 | 0.75 | 0.71 | 0.89 |
| damage dealt / min | 108 ± 7 | 63 ± 5 | 29 ± 5 | 133 (vs humans) |
| L-cancel rate | 0.76 | 0.57 | 0.28 | 0.83 |
| short-hop share | 0.44 | 0.42 | 0.45 | 0.41 |
| tech rate | 0.03 | 0.11 | 0.13 | 0.45 |
| wavedashes / min | 2.8 | 2.2 | 3.2 | 4.6 |
| dashes / min | 22.6 | 27.4 | 39.2 | 40.5 |
| A presses / min | 201 | 22.5 | 12.8 | 22.0 |

### Calibration and loss by head (teacher-forced, 18,795 held-out samples)

| | base | event buttons | prev-action d0 |
| --- | --- | --- | --- |
| total | 3.02 | 2.50 | 1.23 |
| buttons (8) | 0.82 | 0.28 | 0.27 |
| main stick x + y | 1.87 | 1.89 | 0.77 |
| c-stick + shoulder | 0.33 | 0.33 | 0.19 |
| worst button ECE | 0.023 | 0.005 | 0.006 |
| main_x accuracy | 0.62 | 0.62 | 0.88 |

P(down | previous state), event model vs expert: up→down 0.006 vs 0.005 (A),
down→down 0.88 vs 0.81 (A), 0.91 vs 0.91 (R) — the hazards are right.

### Change events (offline, model's own feedback), recall / precision

| | base | event buttons | prev-action d0 |
| --- | --- | --- | --- |
| press ±0 | 0.05 / 0.01 | 0.02 / 0.02 | 0.01 / 0.01 |
| press ±5 | 0.72 / 0.13 | 0.16 / 0.14 | 0.12 / 0.09 |
| stick ±5 | 0.79 / 0.20 | 0.78 / 0.19 | 0.23 / 0.17 |

### What the suite says

1. **Everything is calibrated; the rates are right.** Worst button ECE 0.005
   for the event head. The remaining error is not "wrong probabilities".
2. **The event button head captures the WHOLE button benefit of the
   previous-action channel** (button loss 0.28 vs 0.27) with nothing in the
   trunk. What the channel still buys is the sticks: main-stick loss 0.77
   vs 1.89. That is the case for the hold-or-change stick head (queue 4).
3. **Timing precision is ~0.13 at ±5 frames for every variant.** base's high
   press recall is just pressing constantly. No variant knows WHEN better
   than another; an unknown share of this is the human floor.
4. **Fidelity ranks event-buttons best overall (0.225), base worst (0.360)**,
   noise ≈0.01 — yet base deals the most damage and SDs least. "Looks like
   the expert" and "does well in the sim" are different axes; both columns
   are needed.
5. **Technique**: short-hop share is expert-like everywhere (0.42–0.45 vs
   0.41). L-cancel falls as inputs get stickier (0.76 → 0.57 → 0.28 vs 0.83).
   Tech rate is far below expert in all (0.03–0.13 vs 0.45).
6. **Why event-buttons SDs more (hypothesis with one piece of evidence):** it
   spends 12.5 % of frames in dodge/roll states vs the expert's 2.1 %, and
   presses R 28/min vs 19; an air dodge offstage is a death. Not yet tested.

## Queue 4 result (10-02 12:50) — hold-or-change stick heads (`evt2`)

`--button-events --stick-events`: buttons AND the four stick axes get the
output-level "previous state" treatment (K change logits + 1 hold logit per
axis, collapsed by the previous bucket); the trunk sees nothing of the
previous input.

| | base | evt (buttons) | evt2 (buttons + sticks) | prev_d00 (trunk channel) | expert |
| --- | --- | --- | --- | --- | --- |
| teacher-forced loss | 3.02 | 2.50 | 1.53 | 1.23 | |
| main-stick loss (x + y) | 1.87 | 1.89 | 0.99 | 0.77 | |
| vs idle damage/min | 61.6 ± 11.7 | 20.9 ± 2.7 | 21.3 ± 5.2 (n = 4) | 9.5 ± 2.7 | |
| vs idle SD/min | 1.14 ± 0.27 | 3.27 ± 0.32 | 2.56 ± 0.23 (n = 4) | 1.08 ± 0.23 | |
| recovery drill | 0.272 ± 0.024 | 0.290 ± 0.038 | 0.240 ± 0.027 (n = 4) | 0.157 ± 0.014 | |
| fidelity distance | 0.360 ± 0.011 | 0.225 ± 0.002 | 0.253 ± 0.008 | 0.243 ± 0.006 | floor ≈ 0.03 |
| self-play SD/min | 0.83 | 2.34 | 1.25 | 2.54 | 0.44 |
| self-play damage/min | 108 | 63 | 39 | 29 | 133 |
| L-cancel | 0.76 | 0.57 | 0.62 | 0.28 | 0.83 |
| input repeat share (self-play) | 0.27 | 0.33 | 0.62 | 0.65 | 0.76 |
| A presses/min (self-play) | 201 | 22.5 | 16.4 | 12.8 | 22.0 |

Reading: the output-level heads recover most of the channel's
teacher-forced benefit (1.53 vs 1.23) with nothing in the trunk — and play
like the channel model, not like base (damage 39 vs 108 in self-play). So
the DELIVERY ROUTE of the previous input (trunk input vs output-side
selection) is not what makes the policy passive; knowing its own previous
input at all is. Any model that conditions on its own last input inherits
"mostly keep doing it", and in closed loop that compounds.

## Queue 5 result (10-02 14:18) — scheduled sampling on the AR head

`--prev-action --scheduled-sampling P --ss-steps 4 --ss-ramp-start 2000
--ss-ramp-steps 8000` (the model's own SAMPLED actions replace the channel
on the last 4 window positions for fraction P of samples, ramped in).

| | base | prev_d00 (P = 0) | ss25_k4 | ss50_k4 | ss100_k4 | expert |
| --- | --- | --- | --- | --- | --- | --- |
| teacher-forced loss (calibration total) | 3.02 | 1.23 | 1.38 | 1.52 | 2.15 | |
| P(A down \| A down), teacher-forced | — | 0.84 | 0.65 | 0.75 | 0.42 | 0.81 |
| input repeat share (self-play) | 0.27 | 0.65 | 0.275 | 0.278 | 0.261 | 0.758 |
| A presses/min (self-play) | 201 | 12.8 | 39.7 | 98.8 | 111.3 | 22.0 |
| vs idle damage/min | 61.6 ± 11.7 | 9.5 ± 2.7 | 20.6 | 17.9 | 54.2 | |
| vs idle SD/min | 1.14 ± 0.27 | 1.08 ± 0.23 | 1.38 | 0.75 | 1.06 | |
| recovery drill | 0.272 ± 0.024 | 0.157 ± 0.014 | 0.240 | 0.201 | 0.250 | |
| fidelity distance | 0.360 | 0.243 | 0.354 | 0.337 | 0.357 | |
| self-play SD/min | 0.83 | 2.54 | 2.00 | 2.95 | 0.39 | 0.44 |
| self-play damage/min | 108 | 29 | 32 | 46 | 76 | 133 |
| L-cancel | 0.76 | 0.28 | 0.70 | 0.76 | 0.84 | 0.83 |

**It fails the pass criteria at every rate**, and not as a dial: the
closed-loop input repeat share is at base's flicker level (0.26–0.28) for
P = 0.25, 0.5 and 1.0 alike, while the teacher-forced numbers move
gradually (loss 1.38 → 1.52 → 2.15). Under teacher forcing these models
still use the channel; in their own closed loop they behave as if it were
not there. P = 1.0 is base again (and the cleanest player of the three);
P = 0.25 and 0.5 are worse than both parents (flicker AND low damage).

Two explanations, which queue 6 separates:

1. **The objective (Huszár 2015).** When the channel holds the model's own
   sample, the target is still the expert's action, which was not chosen
   given that sample. The best prediction given "this input is my own
   noise" is the marginal — i.e. ignore the channel. A model that can tell
   self-generated history from the teacher's learns two modes, and in play
   it is always in the "ignore" mode.
2. **A format tell that makes (1) trivial (found 10-02).** The training
   channel holds the replay's RAW analog values (stick 0.9875, shoulder =
   the real L analog); the live channel and the scheduled-sampling splice
   hold the bucket-DECODED values (stick on a 1/8 grid, shoulder =
   max(l, r) bucket). A scheduled-sampling model can tell its own inputs
   from the teacher's by format alone, with no need to read the content.
   `test/exphil/training/streaming_prev_action_test.exs` pins the gap (raw
   0.9875 / 0.0 vs live 0.875 / 0.5 for the same frame).

`--prev-action-quantize` (new) passes the training channel through the same
bucket round trip, so both look identical in format.

## Stick targets never use the top bucket (found 10-02, GOTCHA #138)

`Data.controller_to_action` buckets sticks with `floor(v * 16)` capped at
15; the decode is `bucket / 16`. Measured on 64 k validation frames: bucket
16 has zero targets, bucket 15 holds 16.8 % (full right), bucket 0 18.3 %
(full left). So every policy trained with these targets tilts full
right / up at 0.875 game units and full left / down at −1.0, and every
intermediate tilt is shifted left/down by up to 0.125. In Melee, run speed
and air drift scale with stick x. All seven testbed models recover better
from the right ledge side (drifting left) than the left (0.38 vs 0.26 for
base; 4 cases vs 32, not difficulty-matched, so suggestive only).
`EXPHIL_STICK_ROUNDING=nearest` (experimental, runtime config) rounds to the
nearest bucket, the inverse of the decode. Existing checkpoints and baked
corpora are floor-built; the default is unchanged.

## Queue 6 (launched 10-02 14:41, ~1 h 45) — `scripts/coherence_queue6.sh`

1. Re-calibrate prev_d00 / ss25 / ss50 / ss100 teacher-forced with the
   channel QUANTIZED (`calibration_quant.json`). If the scheduled-sampling
   models' P(down | down) collapses when the teacher's values are put on
   the live grid, they were keying on format (explanation 2).
2. `base_rn`: base with nearest stick rounding. Pass = recovery and
   damage at or above base, left/right recovery gap closed.
3. `prev_q`: channel model trained on the quantized channel (parity fix
   alone).
4. `ss50_k4_q`: scheduled sampling 0.5 on the quantized channel — the fair
   test of scheduled sampling. If it still flickers, explanation 1 stands
   and the next candidate is change-frame (keyframe) loss weighting.

### Queue 6 item 1 (14:44): the format tell is real for L/R, absent for the face buttons

P(down | down), teacher-forced, raw channel → channel quantized to the live
grid (expert: A 0.81, L 0.93, R 0.91):

| | A | L | R | total loss |
| --- | --- | --- | --- | --- |
| prev_d00 | 0.84 → 0.85 | 0.95 → 0.91 | 0.86 → 0.86 | 1.23 → 1.27 |
| ss25_k4 | 0.65 → 0.64 | 0.91 → 0.69 | 0.87 → 0.45 | 1.38 → 1.49 |
| ss50_k4 | 0.75 → 0.75 | 0.92 → 0.75 | 0.92 → 0.56 | 1.52 → 1.61 |
| ss100_k4 | 0.42 → 0.41 | 0.66 → 0.58 | 0.68 → 0.46 | 2.15 → 2.19 |

The scheduled-sampling models stop trusting a held L/R the moment the
shoulder slot looks like the live one (the plain channel model does not
care), so explanation 2 is confirmed for the shoulder buttons. For A/B/X/Y
the distrust is already there on the teacher's raw values (0.65 vs 0.84)
and does not change with format: that part is explanation 1, the objective
itself. Prediction for `ss50_k4_q`: L/R holds improve, face buttons still
flicker.

### Queue 6 items 2–4 (15:55)

| | base | base_rn | prev_d00 | prev_q | ss50_k4 | ss50_k4_q | expert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| teacher-forced loss | 3.02 | 3.10 | 1.23 | 1.22 | 1.52 | 1.55 | |
| input repeat share (self-play) | 0.27 | 0.24 | 0.65 | 0.71 | 0.28 | 0.28 | 0.76 |
| A presses/min (self-play) | 201 | 147 | 12.8 | 27.8 | 98.8 | 95.8 | 22.0 |
| vs idle damage/min | 61.6 ± 11.7 | 16.6 | 9.5 ± 2.7 | 33.5 | 17.9 | 68.0 | |
| vs idle SD/min | 1.14 ± 0.27 | 3.13 | 1.08 ± 0.23 | 1.13 | 0.75 | 1.00 | |
| recovery drill | 0.272 ± 0.024 | 0.302 | 0.157 ± 0.014 | 0.153 | 0.201 | 0.253 | |
| recovery left / right side | 0.26 / 0.38 | 0.30 / 0.34 | | | | | |
| fidelity distance | 0.360 | 0.344 | 0.243 | 0.264 | 0.337 | 0.341 | |
| self-play damage/min | 108 | 85 | 29 | 29 | 46 | 55 | 133 |
| self-play SD/min | 0.83 | 0.99 | 2.54 | 2.14 | 2.95 | 0.77 | 0.44 |
| L-cancel | 0.76 | 0.77 | 0.28 | 0.24 | 0.76 | 0.74 | 0.83 |

- **Scheduled sampling is out.** On the live-format channel (the fair test)
  it flickers exactly as before: repeat share 0.28, A 96/min, and even
  teacher-forced it under-holds every button (A 0.69, L 0.79, R 0.78 vs
  expert 0.81 / 0.93 / 0.91). My prediction that L/R holds would recover
  was wrong. Removing the format tell did not change the outcome, so the
  cause is the objective (explanation 1): the target is never conditioned
  on the model's own sampled input, so the model learns to discount it.
- **`prev_q` = `prev_d00`.** Training on the live format is the correct
  parity but does not change how the channel model plays (passive, SDs,
  recovery 0.15, L-cancel 0.24). The freeze is not a format artifact.
- **`base_rn` is inconclusive.** Left/right recovery gap narrowed (0.12 →
  0.05, weak: 4 right-side cases), but vs idle it is far worse (damage 16.6,
  SD 3.1). The ± figures in every table are repeat EVALUATIONS of one
  trained model; training-seed variance has never been measured, so a
  single-run difference of this size cannot be read. Queue 7 measures it.

## Queue 7 (launched 15:58, after the Mamba fidelity runs; ~2 h) — `scripts/coherence_queue7.sh`

1. `base_s906`, `base_s907`: the base recipe with two other training seeds
   (`SEED=`). Gives the spread every single-run comparison must clear.
2. `prev_q_tw4`, `prev_q_tw16`: change-frame loss weighting (`--transition-weight`,
   already wired on this path: frames whose target differs from the previous
   frame get weight max(1, X)) on the live-format channel model.
   Why it might work (copycat problem, de Haan 2019; Wen 2021): with the
   previous input visible, 76 % of frames are solved by copying, so the
   loss barely rewards learning WHEN to change from the game state. Unlike
   scheduled sampling this keeps every target conditioned on a real
   history. Pass: repeat share stays ≥ 0.6 and A presses within 2× of
   expert, AND recovery ≥ 0.25, self-play damage ≥ 85, L-cancel ≥ 0.7,
   change-event precision at ±5 above 0.13. Known risk: over-weighting
   change frames makes the model change too often (calibration will show
   P(down | down) falling below expert).

## Replay seeding (10-02) — three causes on our side, fixed (49970218)

1. Controller port and starting facing were never sent to the sim (AUTO:
   port = slot). Entry animation now matches on every port pair.
2. `ucf_cardinals: 1` was hardcoded. The 1.0-cardinals rule postdates the
   whole ranked corpus (Slippi 1.7.1 – 3.15.0; 73 % is 2.0.1); it caused the
   ~0.01 x offset at the first walk/dash. `:auto` picks by replay version.
3. Replays before 3.7.0 record a hit's damage/hitstun one frame after the
   sim shows it (positions identical, converges next frame); the detector
   accepts either frame's action/percent for those versions.

40-game survey, tolerance 0.01: first divergence was before frame 30 in
every game; now median ≈ 450 frames on the 30 non-Dream-Land games, 1 clean
to 3000. Still open: Dream Land first-frame y (37.0 vs 37.2), port-based
spawns on some 1.7.1 – 3.0.0 games, real drift after a few hundred frames
(x off by 1–2 units, percent off by exactly 1.0 on 3.9.0 games — consistent
with the netplay code set, e.g. offscreen damage, which the batch API
hardcodes on). The sim's own validator cannot read any corpus file (missing
scene / hitlag / animation_index / playedOn), so none of the remainder can
be attributed to the sim with it. Upstreamable: exposing the match profile
flags in the batch API (`MslMatchConfig`), nothing else yet.

## Queue 7 result (10-02 17:30) — seed spread measured; change-frame weighting fails

Training-seed spread of the base recipe (seeds 905 / 906 / 907, one eval
each): recovery 0.278 / 0.299 / 0.226; vs-idle damage 57 / 67 / 56, SD/min
1.25 / 1.56 / 0.69; self-play damage 108 / 87 / 89, SD/min 0.83 / 0.80 /
0.92; fidelity 0.360 / 0.355 / 0.355; repeat share 0.24 / 0.32 / 0.24;
L-cancel 0.76 / 0.88 / 0.86; A presses 100 / 85 / 98. Fidelity, SDs and
repeat share are seed-stable; recovery, L-cancel and damage move 10–20 %
relative. Every single-run comparison in this doc has to clear that.

- `base_rn` (nearest stick rounding): fidelity 0.344 is barely outside the
  band, recovery 0.30 / self-play damage 85 inside it, and its vs-idle
  result (16.6 damage, 3.1 SD/min) is far outside (56–67, 0.7–1.6). Not a
  win; dropped. `EXPHIL_STICK_ROUNDING` stays experimental and off.
- The channel model's passivity is real: recovery 0.15 vs 0.23–0.30,
  L-cancel 0.24 vs 0.76–0.88, self-play damage 29 vs 87–108.
- `prev_q_tw4` / `prev_q_tw16` (change-frame weight 4× / 16×): coherence
  holds (repeat share 0.70, A 17 / 14 per min) but recovery 0.215 / 0.181,
  self-play damage 25 / 26, L-cancel 0.17 / 0.13 — FAIL on every play
  criterion. Not over-changing either (P(A down | down) 0.87 vs expert
  0.81). Change recall rose 0.244 → 0.304 → 0.33 and nothing downstream
  followed. Loss re-weighting is out alongside scheduled sampling.
- Mamba fidelity (`scripts/mamba_fidelity.sh`): v1 ep2 0.302 ± 0.006
  (self-play SD 1.7/min, damage 83, repeat share 0.39); v2 prev-action
  0.179 ± 0.004 — the closest to the expert of anything measured — with
  SD 4.1/min, damage 46, repeat share 0.72. The big model has the same
  profile as the testbed channel model: expert-like inputs, passive play.

Per-case recovery (36 fixed starts × 8 trials, scratchpad
`recovery_cases.js`): the channel models lose on the FAR starts (|x| > 100,
usually no jump — a Firefox / Illusion is required): 0.00–0.08 vs base
0.18–0.44 by geometry band; on near starts they are at par with base.

## Interp readout (10-02 evening) — `scripts/interp_coherence_probe.exs`, `recovery_drill.exs --trace / --warm-override`

Results in `eval_runs/1002_interp/` (unit `exphil-interp-coh`,
`scripts/interp_coherence_queue.sh`). Offline part: 36,779 teacher-forced
windows over the 16 held-out games (9,210 change frames).

| | base | prev_q (channel, live format) | prev_q_tw4 |
|---|---|---|---|
| KL when the CURRENT frame's game state is swapped for another window's (change / hold frames) | 1.80 / 1.51 | 0.125 / 0.072 | 0.139 / 0.086 |
| KL when the prev-action slot is zeroed on every frame | 0 (no channel) | 6.92 / 4.83 | 6.55 / 4.65 |
| \|grad × input\| share on change frames: last-frame prev / last-frame state / history prev / history state | 0 / 0.33 / 0 / 0.67 | 0.38 / 0.25 / 0.14 / 0.23 | 0.38 / 0.24 / 0.14 / 0.24 |
| head's teacher-forced decode: change recall / false-change on hold | 0.84 / 0.51 | 0.50 / 0.047 | 0.51 / 0.053 |
| linear probe on trunk, "button change this frame" (balanced acc; shuffled control ≈ 0.5) | 0.63 | 0.70 | 0.70 |
| probe "any change within 6 frames" | 0.64 | 0.68 | 0.68 |
| probe on the raw last-frame embedding (input floor), button change | 0.59 | 0.69 | 0.69 |

**Q1 (does the channel model still read the game state?)** At the
output, barely: the current frame has ~15× less leverage than in base
(0.13 vs 1.8 nats), while the channel carries ~7 nats. Gradient shares are
more even (the state still gets ~45 % of saliency on change frames), so the
state is used inside the network but rarely decides the output.

**Q2 (is "change now" represented?)** Yes, and better than in base — the
channel model's trunk linearly encodes an upcoming button change at 0.70
while its head emits one on 0.50 of change frames (teacher-forced). This
is the represented-but-not-emitted case: a target/emission problem, not a
representation problem. `tw4`'s probe numbers are identical to `prev_q`'s
— the weighting changed nothing upstream of the head.

**Q3 (where exactly does recovery fail?)** Frame traces on the far
no-jump starts, died trials: base presses B in 80 % of trials (first B at
frame ~20, up-B in 70 %, plus ~5 jump edges and ~5 side-B edges per trial
with no jump available — it recovers by mashing). The channel model holds
the stick toward the stage (59 % of frames; only 13 % no-input) and does
NOT press B: any B in 42 % of trials, up-B in 16 %, first B at frame 37.
Counterfactual on the channel only (warm history's last 3 inputs
rewritten, game states untouched):

| warm history ends in | base | prev_q |
|---|---|---|
| (real) | 0.257 | 0.149 |
| stick up, no buttons | 0.274 | 0.184 |
| stick up + B (up-B in progress) | 0.288 | **0.392** |
| stick up + X | 0.309 | 0.184 |

With the channel saying "an up-B is already happening" the channel model
carries it out from frame 1–3 (up-B in 83–98 % of trials) and recovers at
0.39, above every base seed. Base is unmoved by the override (no channel).
So the model knows the continuation and the route; the single missing
piece is INITIATING the B press from a hold state. Stick-up alone is not
enough; the press is the thing.

**Q5 (R presses).** Not a format tell. At an expert L-press frame the
model gives P(L) = 0.27, P(R) = 0.27 (base 0.11 / 0.12): the corpus uses
the two triggers about equally (200 L edges vs 214 R in these games) and
nothing in the state says which one this player uses, so each press is a
coin flip; once R lands it is held like the expert holds L (P(L | L held)
= 0.97). The R logit keys on the shoulder channel value and main-stick x,
not on the button bits. Functionally harmless (L and R are equivalent);
the low L-cancel rate is a timing failure, not a trigger-choice one. `tw4`
tilted the flip to R (0.28 vs 0.11), which is its 23/min R.

### What it points to

The defect is press initiation from a hold state, with the information
already in the trunk. Loss re-weighting (tw4/tw16) and self-sampled
conditioning (scheduled sampling) do not touch it. Two candidates do:

1. **Carried-state BPTT over long sequences** (slippi-ai trains this way;
   we never have with the channel). Long holds and the presses that end
   them inside one gradient instead of 80-frame windows that mostly
   contain the hold. Infrastructure: `--bptt` (GRU only,
   `imitation.ex:265`) reads the same precomputed layout, so the
   quantized channel rides along; the testbed driver
   (`train_fox_mamba.exs`) refuses GRU/bptt, and the coherence evals read
   `split.json`, which the bptt path's own 16-game holdout does not write.
   Plumbing needed: run the pair through `train.exs --bptt --backbone gru`
   (windowed GRU control with the same holdout) and emit `split.json`.
2. **Chunk / multi-step targets** — predict the next K inputs. For frames
   t+1…t+K the copy-the-channel shortcut is unavailable, so the press has
   to come from the state. Only the ACT/diffusion policies have
   `action_horizon` today; the AR standard head needs a design (shared
   head over per-offset features so the t-step emission is reshaped too).

Also open from Q3: "committed recovery" as a situation class — extend
`Checkmate` from yes/no to the set of surviving route classes (0 =
checkmate, 1 = forced/coverable, 2+ = mixup), and split `sd_per_min` by it
(already-checkmate kill / blunder from a recoverable position / pure SD).

## 10-02 night: route counting, a drill bug, and the three builds

### Recovery drill was contaminated — re-baselined

`recovery_drill.exs` restored each case with `frames: false` and then read
`Env.frames`, which still held the PREVIOUS case's final states (a dead
Fox elsewhere on the stage). That stale frame went into the windowed
agent's history and sat inside the policy's window for the next 79 frames
of every trial. Found 2026-10-02 when the route verdict (computed from the
same "restored" state) differed between two runs on identical cases.
Fixed (one `Env.observe` after the restores); every number below is on
the fixed drill. Rankings held, levels moved, and the counterfactual
effect grew.

| policy | old | **fixed** | 1 route (1) | 2–3 routes (11) | 4+ routes (24) |
|---|---|---|---|---|---|
| base s905 / s906 / s907 | 0.28 / — / — | **0.462 / 0.486 / 0.434** | 1.0 / 1.0 / 0.75 | 0.25 / 0.35 / 0.16 | 0.54 / 0.53 / 0.55 |
| base_rn | — | 0.476 | 0.88 | 0.16 | 0.60 |
| prev_q | 0.149 | **0.226** | 0.75 | 0.11 | 0.26 |
| prev_d00 | — | 0.167 | 0.88 | 0.07 | 0.18 |
| prev_q_tw4 / tw16 | 0.215 / 0.181 | 0.181 / 0.167 | 0.38 / 0.75 | 0.02 / 0.01 | 0.25 / 0.21 |
| evt2 | — | 0.281 | 0.25 | 0.09 | 0.37 |
| ss50_k4_q | — | 0.278 | 0.25 | 0.08 | 0.37 |
| mamba_v2_prevact | — | 0.212 | 0.63 | 0.02 | 0.28 |
| prev_q + warm "up-B in progress" (upb:3) | 0.392 | **0.594** | 1.0 | 0.41 | 0.66 |
| prev_q + warm stick-up only (up:3) | 0.184 | 0.236 | 0.75 | 0.15 | 0.26 |

Base seed spread on the fixed drill: ±0.03. The channel deficit is 0.23
(prev_q) — eight times the spread. The thin-mixup bucket (2–3 routes, the
far starts) is where every channel model collapses (0.01–0.11 vs base
0.16–0.35); with the channel told "up-B in progress" prev_q beats every
base seed everywhere. Nearest stick rounding does not change recovery
(base_rn inside the band).

### Surviving routes (`ExPhil.Melee.Checkmate.routes/2`)

Every plan in the search is simulated; a route = means (resources spent,
in order: `[]` drift, `[:jump]`, `[:up_b]`, `[:jump, :side_b]`, …) +
destination (`:ledge | :stage | :platform`). Timing variants of one
means to one place are one route (one ledgehog covers them all); the
timing freedom is reported per route (`plans`, `delays`, `fastest`).
Verdict: 0 routes `:checkmate`, 1 `:forced` (the opponent has one thing
to cover — Bradley's "checkmate in one"), 2+ `:mixup`. ~90 ms per state.
Wired into the drill (per-case `routes`/`verdict`, `by_routes` buckets)
and the closed loop (`deaths_by_verdict` at the decision frame = the
latest offstage out-of-hitstun frame since the last hit: `kill` none
existed, `checkmate`, `forced`, `mixup` = threw the stock). Vs idle every
death of base (15) and prev_q (33) is `mixup`, as it must be.

### Builds landed for queue 8

- `--bptt-holdout-split PATH` (bptt holds out another run's `split.json`
  validation games; bptt runs now always write `split.json`) and
  `--recurrent-state zeros|legacy_random` (stamped windowed GRU);
  `train_fox_mamba.exs` accepts `--backbone gru`;
  `coherence_experiment.sh` takes `BACKBONE=` / `TRAINER=` (the parser
  takes a flag's FIRST occurrence, so trailing overrides do not work).
- `--chunk-horizon K` / `--chunk-weight W`: K independent six-component
  heads on the trunk's features predict the controller at t+1..t+K
  (`Heads.build_future_heads/4`); targets ride in the actions map
  (`future_*`, `future_mask` 0 past the game's end); loss = main + W ·
  mean_j; val_loss scores the main head only; export drops the `future*`
  params so the live model is unchanged. Mechanism: t+K is not in the
  prev-action channel, so the trunk must read the game state to score it
  — the copy shortcut loses its monopoly on the gradient. Prediction if
  right: the Q1 state-swap KL of the MAIN head rises toward base's and
  far-start up-B initiation returns; if the trunk already reads the state
  and the head just doesn't emit, nothing moves.
- `scripts/coherence_queue8.sh`: gru_q (windowed GRU control, zero
  state) → bptt_q (`train.exs --bptt --unroll 80`, same holdout) →
  prev_q_ck8 (MinGRU + 8-frame chunk targets). `SMOKE=1` = 200 files.

**Timing is not smeared (closes the "probe within k" question).** The Q2
probes already include "change within 3/6": prev_q 0.65 (now) → 0.65 (≤3)
→ 0.68 (≤6); base 0.59 → 0.61 → 0.64. Widening the window barely helps,
so the trunk does not hold a sharp "soon" with a fuzzy "when" — it holds
a weak signal at every horizon. The chunk-target prediction is therefore
narrowed: if ck8 helps it is because the future heads push MORE state
information into the trunk (Q1 state-swap KL and these probe numbers
should both rise), not because they sharpen timing.

### Queue 8 result 1: the carried-state BPTT pair (10-02 21:40)

Same 3000-file slice, seed 905, prev_q recipe, same 16 held-out games.
Windowed GRU control = zero initial state, window 80, stride 5 (16,996
updates); bptt = `train.exs --bptt --unroll 80`, carry across chunks,
per-timestep loss (3,340 updates at 241 ms; train loss 0.96 = the
windowed model's regime).

| | gru_q (windowed) | **bptt_q (carried)** | prev_q (MinGRU) |
|---|---|---|---|
| offline repeat / change recall / hold agreement | 0.74 / 0.30 / 0.57 | **0.96 / 0.17 / 0.26** | 0.72 / 0.29 / — |
| closed loop vs idle: dmg/min, SD/min, repeat, max frozen run | 13.5, 1.44, 0.69, 157 f | **0.0, 0.0, 0.97, 1731 f** | 20, 2.06, 0.68, 86 f |
| drill (fixed): recovery, never/always | 0.226, 15/1 | **0.149, 30/3** | 0.226, 9/0 |

**Carried-state BPTT is OUT** for the channel recipe: it is the freeze
taken to its limit — a non-neutral input held for the whole rollout, zero
damage, zero deaths. The control is a faithful twin of the MinGRU defect
(0.226 = 0.226), so this is the carry, not the backbone. Reading: with a
carried state the hidden state can keep "previous input" indefinitely on
top of the channel — the copy shortcut gains a second route — and a
per-timestep teacher-forced objective over whole games asks for
initiation no more than windows do. slippi-ai trains this way AND needs
RL afterwards; this is what "pure IL makes a lot of mistakes" looks like.
Caveats: one epoch, 6× fewer updates than the control (equal data
exposure; equal train loss); the bptt `val_loss` 11.6 is a measurement
bug (GOTCHA #140, val set embedded without the channel), not the model.

### Queue 8 result 2: chunk targets (prev_q_ck8, 10-02 22:10)

MinGRU prev_q recipe + `--chunk-horizon 8` (8 future heads, weight 1;
22 ms/update vs 8). Main-head val 1.02 (prev_q 0.97).

| | prev_q | **prev_q_ck8** | base |
|---|---|---|---|
| self-play dmg/min (3 seeds) | 29 ± 7 | **56 ± 5** | 108 |
| self-play SD/min | 2.14 | **1.03** | 0.83 |
| L-cancel rate | 0.24 | 0.28 | 0.76 |
| fidelity distance | 0.264 | 0.272 | 0.360 |
| vs idle: dmg/min, SD/min, repeat | 20, 2.06, 0.68 | 38, 1.63, 0.65 | 25, 0.94, 0.28 |
| offline repeat / change recall | 0.72 / 0.29 | 0.68 / 0.27 | 0.23 / 0.30 |
| drill (fixed) recovery, thin-mixup bucket | 0.226, 0.11 | 0.25, 0.08 | 0.46, 0.25 |
| Q1 state-swap KL change / hold | 0.125 / 0.072 | **0.239 / 0.138** | 1.80 / 1.51 |
| Q1 prev-slot-zeroed KL change / hold | 6.92 / 4.83 | 6.45 / 4.55 | — |
| Q2 head TF change recall / probe "button change now" | 0.51 / 0.70 | 0.51 / 0.71 | 0.84 / 0.63 |

**First lever that moves a channel model the right way without giving
back coherence**: damage doubled, SDs halved, fidelity and repeat share
unchanged. The mechanism check agrees in direction — the main head's
sensitivity to the current game state doubled while the trunk's "change
now" content and the head's emission did not move — and in size: still
7× below base, which is what a half-way live result looks like. Not
moved: L-cancel timing, far-start up-B initiation (drill 0.25, thin
bucket 0.08). Queue 9 = more pressure on the same mechanism: K=4, K=16,
K=8 with chunk weight 3 (`scripts/coherence_queue9.sh`).

## Queue 9 result (10-03 00:40) — chunk-target sweep: K=4 inert, K=16 expert-like but passive, weight 3 = most offence, timing defects untouched

`scripts/coherence_queue9.sh`: prev_q recipe + chunk horizon 4 / 16, and
8 with chunk weight 3. All on the fixed drill; ± = 3-seed spread of the
scorecard, not training seed (queue 10 replicates ck8 / ck8w3 at seed 906).

| | prev_q | ck8 | ck4 | ck16 | ck8w3 | base |
|---|---|---|---|---|---|---|
| self-play dmg/min | 29 ± 7 | 56 ± 5 | 27 ± 1 | 42 ± 1 | **62 ± 4** | 108 |
| self-play SD/min | 2.14 | **1.03** | 4.50 | 1.68 | 2.48 | 0.83 |
| offstage return rate (expert 0.89) | — | — | 0.59 | **0.75** | 0.63 | — |
| L-cancel (expert 0.83) | 0.24 | 0.28 | 0.28 | 0.30 | 0.26 | 0.76 |
| fidelity distance | 0.264 | 0.272 | 0.252 | **0.244** | 0.281 | 0.360 |
| stick-zone distance | — | — | 0.126 | 0.092 | **0.313** | — |
| wavedashes/min (expert 4.6) | — | — | 0.9 | 1.7 | 0.3 | — |
| A presses/min (expert 22) | 27.8 | 13.8 | 17.3 | 7.0 | 19.7 | 201 |
| B presses/min (expert 20) | 21.4 | 11.1 | **39.9** | 17.1 | **35.4** | 67 |
| input repeat share (expert 0.76) | 0.69 | 0.69 | 0.75 | 0.68 | 0.69 | 0.27 |
| vs idle: dmg/min, SD/min, repeat | 20, 2.06, 0.68 | 38, 1.63, 0.65 | 21, 2.44, 0.71 | 31, 2.38, 0.68 | **67**, 3.19, 0.69 | 25, 0.94, 0.28 |
| vs idle deaths by verdict | all mixup | all mixup | 39 mixup / 1 kill | 38 mixup | 48 mixup / 4 kill | 15 mixup |
| drill recovery; thin-mixup bucket (11 cases) | 0.226; 0.11 | 0.25; 0.08 | 0.253; 0.08 | 0.253; 0.10 | **0.309**; 0.10 | 0.46; 0.25 |
| drill 4+-route bucket (24 cases) | — | — | 0.32 | 0.30 | **0.38** | — |
| main-head val (teacher-forced) | 0.97 | 1.02 | 1.00 | 1.02 | 1.03 | 2.43 |
| calibration total / buttons | 1.22 / 0.268 | 1.21 / 0.260 | 1.18 / 0.258 | 1.20 / 0.255 | (q9b) | 3.02 / 0.823 |

Reading:

- **K=4 does nothing good.** Within four frames the expert's input is
  almost always unchanged, so the future heads can be served by the copy
  shortcut too — no new pressure on the trunk — and the extra loss just
  perturbs: worst SDs in the sweep (4.5/min), B presses doubled.
- **K=16 is the most expert-like and the most passive.** Best fidelity
  (0.244), best offstage return (0.75), SDs halved vs prev_q, but damage
  42 and A presses 7/min. Sixteen frames out the targets are uncertain
  enough that the heads learn trajectory statistics (where am I going)
  rather than what-to-press — good for the histograms, not for
  decisiveness.
- **Weight 3 at K=8 pushes offence hardest** (62 self-play, 67 vs idle =
  5× prev_q; best drill 0.31, from the 4+-route cases) **and starts to
  cost fidelity**: stick-zone distance 0.09 → 0.31, wavedashes gone, B
  presses 35/min. With 3 × mean-of-8 the future heads are the dominant
  loss term and the next-frame stick distribution drifts. The dial has a
  far end.
- **Nothing in the sweep moves the timing-precision defects**: L-cancels
  0.24–0.30 everywhere (expert 0.83), thin-mixup recoveries 0.08–0.10
  (base 0.25). Chunk targets make the trunk read state harder; they do
  not make the head emit a single press on the right frame. That is the
  "represented, not emitted" signature from the Q2 probe, and the
  motivation for combining with the event heads (a separate press-now
  output) in queue 10.
- Teacher-forced numbers are flat across the sweep (val 1.00–1.03,
  buttons calibration loss 0.255–0.268): the auxiliary target costs the
  main head nothing offline. The whole effect is in closed loop — the
  scoreboard remains blind to it, as it has been to every live defect.

Queue 10 (`scripts/coherence_queue10.sh`, overnight, unit `exphil-q10`):
evt2_ck8, evt2_ck8w3 (events + chunk), base_ck8 (does the aux target
change a model that already reads state?), prev_q_ck8_s906 and
prev_q_ck8w3_s906 (training-seed replicates), prev_q_ck12, and
mamba_prev_q_ck8 (small Mamba on the same slice: does the lever port?),
then the Q1/Q2 probe on each (`eval_runs/1002_interp/probe_<name>.json`).

### Queue 9 probes (01:15) — state sensitivity is not the whole story

| | prev_q | ck4 | ck8 | ck16 | ck8w3 | base |
|---|---|---|---|---|---|---|
| Q1 state-swap KL change / hold | 0.125 / 0.072 | 0.197 / 0.128 | **0.239 / 0.138** | 0.202 / 0.127 | **0.238 / 0.137** | 1.80 / 1.51 |
| Q1 prev-slot-zeroed KL change / hold | 6.92 / 4.83 | 6.59 / 4.79 | 6.45 / 4.55 | 6.62 / 4.82 | 6.56 / 4.68 | — |
| Q2 head TF change recall / false-change | 0.505 / 0.047 | 0.524 / 0.048 | 0.514 / 0.050 | 0.517 / 0.052 | 0.523 / 0.059 | 0.84 / — |
| Q2 probe "button change now" / "within 6" | 0.70 / 0.68 | 0.71 / 0.69 | 0.71 / 0.69 | 0.71 / 0.68 | 0.71 / 0.70 | 0.63 / — |

Every chunk variant raises the head's state sensitivity (0.125 → 0.20–0.24)
and none moves the head's change emission (0.51–0.52) or the trunk's
"change now" content (0.70–0.71). Two things follow. (1) K=4 raised
sensitivity as much as K=16 and plays worst, and K=8 at weight 1 and 3
have the same sensitivity with different play — so the KL is a necessary
sign, not a predictor: WHAT the trunk reads the state for matters, not
just how much. (2) The emission numbers are frozen across the whole
sweep, exactly where L-cancels and thin-mixup recoveries are frozen —
consistent with those being a head-emission defect that trunk-side
losses cannot reach (queue 10's events + chunk combination).

## Queue 10 result (10-03 05:10, overnight) — events + chunk targets compose; self-play damage is seed-noisy; the lever ports to Mamba

`scripts/coherence_queue10.sh`, unit `exphil-q10` (01:18–05:07). Drill is
the fixed drill; base's own drill seed spread (905/906/907 + rn) is
0.43–0.49 overall, **thin-mixup bucket 0.16–0.35**, 4+ bucket 0.53–0.60.

| | prev_q | ck8 | ck8 s906 | ck8w3 | ck8w3 s906 | evt2 | **evt2_ck8** | **evt2_ck8w3** | base | base_ck8 | mamba ck8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| self-play dmg/min | 29 | 56 | **20** | 62 | 55 | 39 | 65 | 67 | 108 | 87 | 58 |
| self-play SD/min | 2.14 | 1.03 | 1.34 | 2.48 | 1.18 | 1.25 | 1.23 | 1.43 | 0.83 | 1.38 | **0.97** |
| vs idle dmg/min | 20 | 38 | 37 | 67 | 44 | 28 | 46 | 32 | 57 | 48 | 37 |
| vs idle SD/min | 2.06 | 1.63 | 2.31 | 3.19 | 1.25 | 2.5 | 1.81 | 2.19 | 1.25 | 2.13 | 1.25 |
| L-cancel (expert 0.83) | 0.24 | 0.28 | 0.30 | 0.26 | 0.25 | 0.62 | 0.51 | **0.71** | 0.76 | **0.83** | 0.25 |
| fidelity distance | 0.264 | 0.272 | 0.298 | 0.281 | 0.273 | 0.253 | 0.250 | **0.225** | 0.360 | 0.339 | 0.308 |
| input repeat share (expert 0.76) | 0.69 | 0.67 | 0.66 | 0.69 | 0.72 | 0.62 | 0.64 | 0.66 | 0.24 | 0.29 | 0.56 |
| A presses/min (expert 22) | 28 | 14 | 11 | 20 | 18 | 16 | 28 | 20 | 201 | 131 | 50 |
| drill recovery | 0.226 | 0.25 | 0.27 | 0.31 | 0.29 | 0.28 | **0.43** | 0.35 | 0.46 | 0.45 | 0.26 |
| drill thin-mixup bucket | 0.11 | 0.08 | 0.08 | 0.10 | 0.13 | — | **0.30** | 0.19 | 0.16–0.35 | 0.17 | 0.21 |
| drill 4+-route bucket | — | 0.31 | 0.36 | 0.38 | 0.34 | — | 0.48 | 0.41 | 0.53–0.60 | 0.57 | 0.26 |
| drill never-recovered cases | 9 | 11 | 12 | 10 | 8 | 7 | **2** | 4 | 2–5 | 4 | 9 |
| teacher-forced val | 0.97 | 1.02 | 1.01 | 1.03 | 1.03 | 1.37 | 1.33 | 1.34 | 2.43 | 2.41 | 1.05 |

1. **The two levers compose.** `evt2_ck8` (event heads + chunk 8) is the
   first channel-free coherent model that recovers like base: drill 0.43
   (base 0.43–0.49), thin-mixup 0.30 (inside base's 0.16–0.35; every
   trunk-channel model is 0.05–0.13), only 2/36 cases never recovered,
   with repeat share 0.64 and 28 A presses/min (base: 201). The probe
   prediction held: chunk targets alone never touched the thin bucket,
   the press-now output did. `evt2_ck8w3` trades some of that for the
   best fidelity of the program (0.225) and L-cancel 0.71 — the first
   coherent model near the 0.7 pass bar — with A/B/R press rates within
   ~25 % of the expert's.
2. **Self-play damage is a training-seed lottery on this testbed**:
   ck8 56 → 20 at seed 906 (prev_q 29). The "damage doubled" headline
   from queue 8 does not survive a seed. What replicates: SD (1.0/1.3 vs
   prev_q 2.1), vs-idle damage (38/37 vs 13.5–20), drill (0.25/0.27).
   ck8w3 replicates better (62/55, 67/44, 0.31/0.29) and is the more
   robust chunk setting. Rank by vs-idle, drill, fidelity, L-cancel; treat
   self-play damage as ±20.
3. **base_ck8 = base** on recovery (0.45; buckets inside base's spread)
   and still flickers (repeat 0.29; the aux target is no substitute for a
   coherence mechanism). L-cancel 0.83 (= expert) vs base 0.76 and
   damage 87 vs 108, SD 1.4 vs 0.8 — a sideways move; chunk targets are
   not a free general improvement on a model that already reads state.
4. **The lever ports to Mamba**: `mamba_prev_q_ck8` (16-min train, val
   1.05) has the ck8 profile — self-play SD 0.97 (lowest of any channel
   model), damage 58, vs-idle 37, drill 0.26 — but with Mamba's known
   weaker coherence on this recipe (repeat 0.56, A presses 50/min).
5. ck12 lies on the K curve between 8 and 16 (damage 46, SD 2.2,
   fidelity 0.261); nothing new.

Still open after queue 10: L-cancel (0.51/0.71 vs 0.83) and self-play
damage (≈65 vs 133) for the event+chunk models; the evt2 models' main-stick
histograms (stick_zone 0.21–0.30) are the worst part of their fidelity.
Infrastructure: calibration unwraps chunk outputs (`Loss.main_head/1`);
the interp probe now builds event-head inputs (collapse layers +
`prev_buttons`/`prev_sticks`, trunk slot zeroed) and the Mamba probe needs
`--batch 64` (saliency gradient OOMs at 256).

### Queue 10 probes (05:45)

| | prev_q | ck8 | ck8 s906 | ck8w3 | ck8w3 s906 | ck12 | evt2_ck8 | evt2_ck8w3 | mamba ck8 | base_ck8 | base |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 state-swap KL change / hold | 0.125 / 0.072 | 0.239 / 0.138 | 0.193 / 0.123 | 0.238 / 0.137 | 0.212 / 0.124 | 0.240 / 0.144 | **0.409 / 0.229** | **0.402 / 0.220** | 0.361 / 0.230 | 1.65 / 1.38 | 1.80 / 1.51 |
| Q1 prev-slot-zeroed KL change / hold | 6.92 / 4.83 | 6.45 / 4.55 | 6.00 / 4.30 | 6.56 / 4.68 | 6.82 / 4.94 | 6.24 / 4.51 | 4.71 / 3.20 ¹ | 4.76 / 3.19 ¹ | 5.87 / 4.12 | 0 / 0 | 0 / 0 |
| Q2 head TF change recall / false-change | 0.505 / 0.047 | 0.514 / 0.050 | 0.513 / 0.048 | 0.523 / 0.059 | 0.527 / 0.059 | 0.520 / 0.055 | 0.233 / 0.018 ² | 0.228 / 0.018 ² | 0.525 / 0.057 | 0.835 / 0.512 | 0.84 / — |
| Q2 probe "button change now" (act) | 0.70 | 0.71 | 0.71 | 0.71 | 0.72 | 0.72 | 0.63 | 0.62 | 0.72 | 0.63 | 0.63 |

¹ For event-head models the probe zeroes the trunk slot (already zero) AND
the heads' `prev_buttons`/`prev_sticks` inputs — this row measures the
HEAD's dependence on the previous input, not the trunk's.
² Hazard heads under argmax: press probabilities sit below 0.5 and are
sampled, so deterministic "change recall" is not comparable to state heads.

Reading: the event + chunk models have the highest state sensitivity of
any coherent model (0.41, vs 0.24 for chunk alone and 1.8 for base) — the
trunk, with no previous-input shortcut at all, reads state harder than the
trunk-channel models do even with chunk targets. Their "change now" probe
drops to base's 0.63: without the previous input in the trunk, "is this a
change frame" is not a trunk-representable quantity, which is consistent
with the whole premise — the decision moves into the heads. The seed
replicates confirm the chunk-alone sensitivity sits at 0.19–0.24 at both
seeds while closed-loop damage varied 20–56, i.e. the KL tracks the
training recipe, not the per-seed outcome.

## 10-04: live look at evt2_ck8 / evt2_ck8w3, queue 11, the SD review

### Live (Bradley, 10 + 8 games) vs the sim's prediction

`scripts/live_scorecard.exs` (same PlayStats as the sim card, over .slp):

| | evt2_ck8 sim → live | evt2_ck8w3 sim → live | expert |
|---|---|---|---|
| fidelity distance | 0.250 → 0.264 | 0.225 → 0.226 | — |
| SD/min | 1.23 → 1.43 | 1.43 → 1.65 | 0.44 |
| L-cancel | 0.51 → 0.52 | 0.71 → 0.65 | 0.83 |
| input repeat share | 0.64 → 0.68 | 0.66 → 0.72 | 0.76 |
| A / B presses per min | 23 / 24 | 18 / 15 | 22 / 20 |
| damage dealt / taken per min | 54 / 189 | 60 / 196 | 133 / 141 |

Bradley: "pretty similar … both SD'd a fair amount; the first side-B'd
more, the other drifted off / didn't drift back and up-B." No freeze, no
flicker. The closed-loop suite is now validated on event-head models
(every metric within 0.02–0.2 of live). Coherence is closed as a
problem; **recovery (SD ≈ 1.5/min, 3.5× expert) is the gap.**

### Queue 11 (`scripts/coherence_queue11.sh`, 15:23–18:20)

| | evt2_ck8 | evt2_ck8 s906 | evt2_ck8w3 | evt2_ck8w3 s906 | evt2_ck8w2 | evt2_ck8 ×3 epochs |
|---|---|---|---|---|---|---|
| self-play dmg/min | 65 | 74 | 67 | **96** | 73 | 80 |
| self-play SD/min | 1.23 | 1.28 | 1.43 | **1.02** | 1.91 | 1.81 |
| vs idle dmg/min | 46 | 79 | 32 | **108** | 42 | 55 |
| L-cancel | 0.51 | 0.58 | 0.712 | **0.715** | 0.67 | 0.64 |
| fidelity distance | 0.250 | 0.238 | 0.225 | 0.235 | 0.221 | **0.213** |
| repeat share | 0.64 | 0.64 | 0.66 | 0.64 | 0.63 | 0.65 |
| drill; thin bucket; never | **0.43**; 0.30; 2 | 0.38; 0.18; 4 | 0.35; 0.19; 4 | 0.34; 0.26; 5 | 0.32; 0.18; 9 | 0.38; 0.28; 3 |
| val (main head) | 1.33 | — | 1.34 | — | — | 1.295 |

- **The events + chunk result replicates across training seed** (unlike
  plain ck8): drill 0.43/0.38 and 0.35/0.34, L-cancel 0.715/0.712
  exactly, fidelity within 0.012, repeat 0.64 everywhere. Damage still
  swings by seed (w3: 67 → 96 self-play, 32 → 108 vs idle) but upward.
- **Weight orders fidelity and L-cancel, not recovery** (w1 0.43/0.38,
  w2 0.32, w3 0.35/0.34).
- **Three epochs**: fidelity 0.213 (best), L-cancel 0.64, damage 80 —
  and SD 1.81, drill 0.38. More of the same training improves
  everything except SDs. Fourth knob this week with that signature.

### SD review of the live games (`scripts/sd_review.exs`, `eval_runs/1004_live/sd_review_*.json`)

Every bot death, read back to the decision frame (same state machine as
`sim_closed_loop.exs`), judged by `Checkmate.routes/1`, with what the
bot did on the way down. 54 deaths: **38 mixup** (2+ routes open — thrown
stocks), 13 died in stun, 2 checkmate, 1 forced.

| attempt on the 38 thrown stocks | n | trace |
|---|---|---|
| side-B, never up-B | 13 | `TUMBLING > FOX_ILLUSION > SHORTENED > DEAD_FALL` from BELOW ledge height (y −10 … −53): illusion falls short, helpless. Only up-B works there. |
| jump only → airdodge | 11 | at the ledge (x ≈ ±86, y ≈ 0, 10–15 routes): double-jump, then AIRDODGE into special fall |
| laser / shine offstage | 8 | at ledge height with a jump left |
| nothing | 4 | |
| up-B, still died | 3 | `FIREFOX_AIR > DEAD_FALL`: aimed short |

The channel model's defect was "never presses B". This model presses
plenty — it **chooses the wrong means for its height**: side-B when
low, airdodge/laser when at the ledge. Both of Bradley's impressions
(ck8 side-B'd; w3 drifted and didn't up-B) are in the table. That is a
conditioning failure on a few state dims (y, jumps left), on a trunk the
probe says reads state harder than any coherent model before it — the
next interp question is whether the recovery choice is keyed on height
at all (state-swap KL restricted to y / jumps on offstage frames).

### Queue 12 (`scripts/coherence_queue12.sh`, launched 18:25, ~3 h)

Carried-state BPTT retested with the recipe that works windowed, at a
matched update count: per-timestep event heads and chunk targets now
run under `train.exs --bptt` (collapse layers broadcast over time; future
head j scores the chunk's own targets shifted by j; commit 2dfdcdd9).
`bptt_evt2_ck8_e5` (5 epochs ≈ 16.7k updates = the windowed count) and
`bptt_q_e5` (the queue 8 recipe at 5 epochs: under-training control).
Verdicts: beats windowed evt2_ck8 on SD/drill → the carry was a
casualty of the trunk channel; freezes anyway → the carry is out for
good; bptt_q_e5 un-freezes → queue 8 was an under-training artifact.
