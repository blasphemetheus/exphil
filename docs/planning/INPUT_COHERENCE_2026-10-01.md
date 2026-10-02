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
