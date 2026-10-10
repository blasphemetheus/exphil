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

### Queue 12 result (22:10) — carried state is out: not under-training, not a missing knob

Bradley's question: could `bptt_q` have needed one of the other knobs, or
more training? Both tested at once (`scripts/coherence_queue12.sh`; the
chunk run hit a latent NaN — all-zero future weights on a batch of
one-frame segments divided by zero in `imitation_loss`; found by replaying
the captured batch, fixed at the denominator, 90d32b9f).

| | windowed evt2_ck8 (1 ep) | carried evt2_ck8, 5 ep | carried bptt_q, 1 ep (q8) | carried bptt_q, 5 ep |
|---|---|---|---|---|
| updates | 17k | 16.7k | 3.3k | 16.7k |
| val (teacher-forced) | 1.33 | 1.44 | (11.6, #140) | 1.65 |
| vs idle: dmg/min, repeat, max frozen run | 46, 0.65, 50 f | **0.9, 0.91, 1707 f** | 0, 0.97, 1731 f | **0, 0.997, 1800 f** |
| self-play SD/min, dmg/min, L-cancel | 1.23, 65, 0.51 | **4.33**, 16, 0.74 | — | — |
| fidelity distance | 0.250 | 0.273 | 0.567 | 0.681 |
| drill (fixed); never | 0.43; 2 | 0.32; 9 | 0.15; 30 | 0.15; 24 |
| smoke (200 files, ~220 updates) max frozen run | — | 48 f | — | — |

- **More training makes the carried models freeze harder**, not softer:
  bptt_q at 5 epochs is neutral on 99.7 % of frames. Under-training is
  ruled out.
- **The working recipe freezes under the carry too.** Event heads + chunk
  targets, no previous input anywhere in the trunk, matched updates: one
  input held for 1707 of 1800 frames vs idle, SD 4.3/min in self-play.
  Softer than the channel version (it wakes up against a moving opponent:
  L-cancel 0.74, fidelity 0.27), but out by every pass criterion.
- The smoke had NOT frozen at ~220 updates: the freeze develops with
  training. Reading: whatever the carry learns to hold over a whole game
  under teacher forcing — with no shortcut to copy, it must be game-state
  history — becomes a reason to wait rather than act once the model runs
  on its own outputs. The windowed model cannot hold anything past 80
  frames and so cannot learn that. This is consistent with slippi-ai
  needing RL on top of exactly this training.
- **Decision: carried state is out for the imitation program. The port
  recipe is windowed MinGRU/Mamba + event heads + chunk targets.**

## 10-05 — the recovery defect pinned: a head that cannot see the stick

Bradley's asks (01:00): an eval that pinpoints the recovery defect; then
the interp step; whether carried state might work with Mamba where it
failed with MinGRU; whether we are "doing carried state wrong".

### Carried state: not an implementation bug (answered first)

- Trainer (`train_loop.ex:293-322`): carry zeroed on `is_resetting` rows,
  passed as a plain argument (detached at the chunk edge) — textbook.
- The offline coherence eval runs the carried model through the SAME
  `Agent` stateful step the live bot uses, over whole expert games:
  bptt_evt2_ck8_e5 repeat 0.707 (expert 0.738), neutral 0.159 — not
  frozen; val 1.44 vs windowed 1.33. Closed loop: repeat 0.91 vs idle,
  0.77 in self-play. The freeze scales with distance from the corpus:
  exposure bias through the hidden state, not code.
- Mamba carried state: untestable with the current methodology (BPTT
  path is GRU-only; porting the initial-state threading to the Mamba
  scan is ~a day). Deferred unless the windowed Mamba port underperforms.

### Recovery-means scorecard (`lib/exphil/eval/recovery_means.ex`)

Every offstage trip (the death list in `sd_review` is survivorship-
biased): situation at the decision frame = height band (high > 0 /
ledge > -20 / low > -60 / deep) × distance beyond the edge (near < 20 /
mid < 60 / far) × jumps (0 / 1+); first means (jump / side_b / up_b /
airdodge / attack / drift / none); outcome. Scored against an expert
table (`scripts/expert_recovery_means.exs`, 150 FD Fox games, 2264
episodes; split-half floor mismatch 0.08 / JS 0.168) by
`scripts/recovery_means.exs` (sim rollouts 3 seeds × 32 envs, or live
`.slp`). Gotcha fixed on the first pass: dead/rebirth action states
(0..13) sit at y ≈ -141 for 59 frames and then teleport — they looked
like "deep, drift, returned" trips (expert deep band 274 → 47 episodes).

| | expert | evt2_ck8 | evt2_ck8w3 | base | bptt_evt2_ck8_e5 |
|---|---|---|---|---|---|
| mismatch (floor 0.08) | — | 0.128 | 0.103 | 0.166 | 0.207 |
| return rate | 0.929 | 0.651 | 0.670 | 0.702 | 0.399 |
| side_b_low | 0.107 | 0.333 | 0.15 | 0.25 | 0.0 |
| airdodge_with_jump | 0.034 | 0.335 | 0.184 | 0.25 | 0.231 |
| **high band return** | **0.952** | 0.283 | 0.337 | 0.379 | 0.21 |
| died with a jump in hand (high) | 0.73 | 0.46 | 0.70 | 0.64 | 0.96 |
| up-B anywhere in the death sequence (high) | 0.17 | 0.06 | 0.05 | 0.00 | 0.01 |
| median frames to death (high) | 102 | 72 | 75 | 75 | 57 |

The stock goes from the HIGH band (at the edge, above ledge height):
one move then nothing — `side_b` alone (evt2_ck8 40 %), `attack` alone
(laser/shine/aerial 17-40 %), `airdodge>jump` (11-24 %) — then ~70
frames of free fall. Not "wrong means for its height": **no second
means**, never the up-B.

### Offstage B presses (`scripts/b_press_stick.exs`, live replays vs expert)

| | expert (2163) | evt2_ck8 (76) | w3 (50) |
|---|---|---|---|
| stick NEUTRAL on the press frame | 2.5 % | 27.6 % | 20 % |
| late-up (not up on press, up within 3 f) | 1.9 % | 14.5 % | 16 % |
| result = none (B while helpless) | 5.1 % | 36.8 % | 20 % |
| side-B share of presses at low | 5.2 % | 16.7 % | 38 % |

A special's identity is the stick on the frame B goes down. Neutral = laser,
side = illusion (aerial illusion ends HELPLESS — the "none" presses are B
mashed after it). And in 96 % of the expert's up-Bs the stick was already
up ≥ 1 frame before the press (75 % ≥ 3 frames): the decision is "stick
up while falling", made before the button.

### Interp (`scripts/interp_recovery_probe.exs`, evt2_ck8, 24 holdout games, teacher-forced)

- Q1 at press frames the model's zone shares match the expert's by band
  (±0.02) — **confounded**: with the head's previous-stick input centred
  (Q1b) P(up) on the expert's up-B presses falls 0.93 → 0.24 (deep 0.99 →
  0.21). At the press frame the stick is copied from the previous stick —
  which is what the data does too (96 % above).
- Q4 the stick-UP event offstage: model P(up) on the expert's event frame
  0.12 high / 0.25 ledge / 0.29 low / 0.45 deep vs 0.01-0.10 on stay-down
  frames; **y ablation 0.30 → 0.13 (control 0.30 → 0.30)**. Height IS
  represented and used, at the right decision. Jumps-left: no effect.
- Q2b B hazard once the stick is up: model 0.017-0.026 vs expert
  0.033-0.044 (half).
- **Q2c B-press hazard by previous stick zone — model 0.011 neutral /
  0.018 side / 0.025 down / 0.021 up; expert 0.001 / 0.011 / 0.040 /
  0.037.** The model's hazard is flat where the expert's spans 40×. The
  button head presses B blind to the stick.
- Structural cause (`heads.ex` `build_autoregressive_head`): the button
  head is `component.(r0)` with `r0 = dense(trunk)`; the `prev` nodes only
  SELECT press-vs-release / hold-vs-change, they are never a feature; and
  the event recipe zeroes the prev slot in the trunk. No head can
  condition on "the stick is up now". The AR order (buttons → sticks)
  makes the previous stick the only stick information a press could use.

### Lever: `--event-context` (queue 13, running)

Previous buttons / stick buckets as zero-initialised embeddings added to
the head residual `r0` (`ar_prev_buttons_embed`, `ar_prev_stick_{j}_embed`);
trunk still blind, so the copy shortcut stays closed; absent params =
plain event head (sampler mirror in `sampling.ex` `ar_prev_context/2`).
Tests: `test/exphil/networks/policy/event_context_test.exs`.
`scripts/coherence_queue13.sh`: smoke → evt2ctx_ck8 → evt2ctx_ck8w3 →
recovery_probe on the evt2_ck8w3 control. Pass criteria in the script
header: Q2c neutral ≤ 0.004 / up ≥ 0.03; high-band return ≥ 0.6 (0.28);
mismatch ≤ 0.15; airdodge_with_jump ≤ 0.10; coherence criteria kept;
neutral-stick press share ≤ 5 % on replays. `coherence_experiment.sh` now
runs `recovery_means` and `recovery_probe` by default.

### Queue 13 result (03:10) — the lever works where it was aimed; the stock goes somewhere else

| | evt2_ck8 | evt2_ck8w3 | **evt2ctx_ck8** | **evt2ctx_ck8w3** | expert |
|---|---|---|---|---|---|
| Q2c B hazard neutral / side / down / up | .011/.018/.025/.021 | .007/.015/.019/.017 | .007/.009/**.042/.039** | **.004**/.008/.021/.024 | .001/.011/.040/.037 |
| fidelity distance | 0.250 | 0.225 | **0.185** | **0.181** | — |
| offline repeat (expert 0.76) / frozen run | 0.65 / 50 f | — | 0.75 / 119 f | 0.75 / 184 f | — |
| self-play SD/min | 1.23 | — | 1.30 | — | 0.44 |
| drill (fixed) | 0.43 | 0.34 | 0.40 | — | — |
| recovery-means mismatch (floor 0.08) | 0.117 | 0.103 | **0.069** | 0.082 | — |
| return rate | 0.635 | 0.67 | 0.667 | 0.501 | 0.929 |

- **Mechanism confirmed**: with the previous input as a head feature the
  B-press hazard spans the stick zones (6–10× range; expert 40×) and the
  up/down hazards land on the expert's. Without it, both controls are flat
  (≤ 2.5×). Fidelity improves by 0.04–0.065 — the largest single move of
  the program — with no coherence regression (repeat 0.75, no freeze).
- **But the return rate does not move** (0.64 → 0.67; w3 0.50).

**Why (timing fields, 03:11):** expert first means fires a median **9 f**
after becoming actionable (side-B 12 f; 17.6 % of side-Bs from low/deep).
Model: latency **0**, side-B from low **0.0** — the first means is ALREADY
IN PROGRESS on the first offstage frame. evt2_ck8: side-B 53/54 episodes
at frame 0 (49 died), airdodge 87/96, attack 71/98; only jumps start
offstage. The bot's offstage trips are mostly **moves started on stage
that carry it off the edge** — illusion off the lip (helpless → dead),
wavedash/airdodge off, laser/aerial off. The expert is launched off and
then decides; the bot walks off. The SD review's "side-B from below the
ledge" and Bradley's "drifted off stage" are the same thing.

So: the offstage machinery was real and is now fixed at its bottleneck
(stick-up reads height; B conditions on the stick), but it is not where
the stocks go. Next eval = **edge self-destructs**: offstage episodes with
the first means active at frame 0, by move, vs the expert's rate; next
probe = P(side-B / dash-off | onstage within 20 u of the edge, facing out)
teacher-forced vs the expert's labels, and whether that reads x / facing.
Queue 14 (running: ctx seed replicate, 3 epochs, Mamba testbed) answers the
lever's robustness and the port question in the meantime.

### Queue 14 result (05:39) — the lever is seed-robust, keeps paying with epochs, and carries to Mamba

| | evt2_ck8 | ctx_ck8 | ctx_ck8 s906 | ctx_ck8 **3 ep** | ctx_ck8w3 | **mamba** ctx_ck8 |
|---|---|---|---|---|---|---|
| Q2c neutral / up (expert .001 / .037) | .011 / .021 | .007 / .039 | .006 / .040 | **.003 / .036** | .004 / .024 | .010 / .038 |
| Q4 stick-up y ablation (base → y:=high) | .30 → .13 | .35 → .21 | .43 → .25 | .45 → .29 | .35 → .23 | .26 → .22 |
| fidelity | 0.250 | 0.185 | 0.198 | **0.149** | 0.181 | 0.197 |
| L-cancel | 0.51 | 0.63 | 0.53 | **0.80** | 0.69 | 0.55 |
| self-play dmg/min | 65 | 52 | 53 | 61 | 73 | 45 |
| self-play SD/min (expert 0.44) | 1.23 | 1.30 | 1.27 | **2.87** | 2.17 | 2.44 |
| vs idle repeat / frozen run | .63 / 50 f | .74 / 119 f | .74 / 122 f | .76 / 414 f | .77 / 184 f | .72 / 100 f |
| drill (fixed); never | .43; 2 | .40; 5 | .25; 10 | .45; 1 | .26; 10 | .22; 13 |
| recovery-means mismatch / return | .117 / .64 | .105 / .66 | .083 / .71 | .105 / .51 | .082 / .50 | .105 / .53 |
| first-means latency (expert 9 f) | 0 | 0 | 0 | 4 | 0 | 0 |

- **Seed-robust**: s906 reproduces Q2c, fidelity (0.198 vs pre-context
  s906 0.238) and the coherence card.
- **3 epochs**: every teacher-forced number lands on the expert (neutral
  .003), fidelity 0.149, L-cancel 0.80 — and **SD/min doubles** (2.87),
  offstage trips 10/min, return 0.51. More training = more expert-like
  inputs AND more walking off the stage. The walk-off is learned, not
  under-trained. Idle-opponent frozen run grows (414 f) but no freeze.
- **Mamba carries the recipe** at 256×2: Q2c, fidelity 0.197, coherence
  (repeat .72, 100 f). Weaker on the drill (0.22, never 13/36) and the
  y-ablation (.26 → .22) — recovery reading is softer on Mamba at this
  size; worth re-checking at the port width before reading much into it.
- The drill is noisy across seeds (0.25–0.45 for the same recipe); the
  recovery-means return rate is steadier (0.50–0.71) and now reads mostly
  **edge self-destructs** (latency 0 everywhere but e3's 4 f).

**Port recipe (settled by this queue): windowed + prev_q + event heads +
`--event-context` + chunk 8.** Epochs: ≥ 3 for fidelity/L-cancel, with the
edge-SD defect to be fixed first or it will scale with training.

### 10-05 07:30 — edge self-destructs measured; the honest recovery picture

Two eval fixes first: (1) the offstage threshold y < -5 caught wavedashes
and onstage illusions (they dip to y ≈ -5.5 on the surface) as "trips"
and inflated every return rate above — `RecoveryMeans` now uses y < -12
(PlayStats keeps -5; shared noise); (2) the scorecard splits CARRIED-OFF
trips (first means already active on the first offstage frame) from
DECIDED trips, and records the 30-frame approach.

| | return (all trips) | carried-off share | carried-off died | side-B off (died) | decided-trip return |
|---|---|---|---|---|---|
| expert (1788 trips) | **0.911** | 0.055 | 0.16 | 5 (3) | **0.915** |
| evt2_ck8 | 0.292 | 0.431 | 0.93 | 52 (52) | 0.459 |
| evt2ctx_ck8 | 0.299 | 0.389 | 0.94 | 62 (61) | 0.450 |
| evt2ctx_ck8 3 ep | 0.312 | **0.140** | 0.88 | 12 (12) | 0.344 |
| mamba ctx | 0.269 | 0.361 | 0.89 | 41 (38) | 0.359 |

- **The bot returns from ~30 % of offstage trips; the expert from 91 %.**
  The fixed drill (0.25–0.45) only ever measured launched recoveries.
- **Illusion off the lip** (`KNEE_BEND > JUMPING_FORWARD > FOX_ILLUSION
  > FOX_ILLUSION_SHORTENED` at x ≈ 86, y 2–31, facing the edge, run
  speed): the SHAPE of the expert's ledge-cancelled illusion, from the
  wrong height/speed; ~100 % fatal; 52–62 per rollout vs the expert's 5.
  3 epochs cuts it to 12 → it is learnable from more training/data.
- **Edge probe** (`scripts/interp_edge_probe.exs`, teacher-forced on
  expert onstage frames): every walk-off hazard is calibrated — near /
  facing-edge P(B press) .006 | .006, P(stick→edge) .135 | .132, and
  P(B press | stick already toward the edge) .001 | .003 — and the x /
  facing ablations barely move it. **The excess is not in the policy on
  expert states; it is in the states the bot drives itself into** —
  closed-loop compounding on a precise technique. Same shape as the
  carried-state freeze, localized.
- **Decided trips** return 0.34–0.46 vs 0.915 — the second half of the
  gap, not yet dissected (next: the same approach/sequence dump on
  decided deaths; the offstage probe says stick-up reads height and B
  reads the stick, so the loss is downstream of both).

### 10-05 12:55 — decided-trip deaths dissected: the silent fall

Tooling: every offstage trip in `recovery_means.json` now carries a
`trace` (every 3rd frame: action, x, y, vy, stick, B, jump, jumps left);
`scripts/expert_recovery_means.exs` writes the expert's episodes with
traces to `expert_recovery_means_fd_episodes.json`;
`scripts/recovery_trace_read.js` reads either. One 6-min re-roll
(evt2ctx_ck8_e3: return 0.323, decided 0.347 — reproduces) and the rest
was node one-liners.

**Where the decided deaths are.** ~90 % of decided trips start in the
HIGH band (y ≥ 0, |x| ≈ 86 = the FD lip): the bot runs/dashes off the
edge (pre_actions DASHING/RUNNING, facing the edge) with its double jump
in hand. The expert returns from that band 95 % (n = 1218); the bot
29–40 %. 92–98 % are self-destructs.

**What the died trips look like** (e3, 190 high-band decided deaths):

- **The silent fall.** Stick dead-centre `(0.5, 0.5)`, no B, no jump,
  for 30–50 frames of free fall. Died trips hold toward-stage 0.12 of
  frames (returned trips 0.46; expert returned 0.49). 134 of 190 deaths
  never reached helpless (up-B still available); 65 never used the
  double jump. Expert: 9 of 41 high-band deaths with resources in hand.
- **Run-off laser / shine, then nothing.** Top action string
  `FALLING > LASER(344-346) > FALLING` — the expert's run-off-laser →
  double-jump-back, second half missing. When the bot does press jump it
  is at y −60…−100 with no jumps left.
- **Inputs during a committed aerial** — dair from y −68 down with stick
  up + B pressed mid-animation: right intent, 30 frames late, swallowed.
- 29 % of deaths end helpless (`JUMP > AIRDODGE > DEAD_FALL`: airdodge
  offstage with the jump in hand).

**Hazard of resuming input vs length of silence** (below stage level,
ledge frames 252..263 EXCLUDED — ledge hangs are input-free and
offstage, and they inflated the expert's "silence" on the first pass):

| silence so far | 3 f | 6 f | 9 f | 12 f | 15 f | 18 f | 24 f |
|---|---|---|---|---|---|---|---|
| e3 closed-loop, P(resume)/3 f (at risk) | .16 (414) | .13 (326) | .14 (277) | .12 (226) | .09 (191) | .09 (167) | .06 (113) |
| expert, P(resume)/3 f (at risk) | .18 (424) | .16 (185) | .18 (132) | .13 (94) | .18 (74) | .24 (55) | .27 (15) |

The expert's silent falls are rare and resolve within ~15 frames with a
hazard that holds or rises; the bot's hazard DECAYS and its silences
persist. Silence ≈ death for both (expert deaths show the same 30-f
silence signature) — the bot simply enters the silent fall 190/268 times
vs the expert's 41/886.

**Teacher-forced (Q5, new in `interp_recovery_probe.exs`, e3, 24 holdout
games, ledge frames excluded): P(any input now | expert silent k frames)
model | expert, below stage:** k1-3 .097|.081 (259) · k4-6 .043|.039
(207) · k7-12 .057|.052 (347) · k13-24 .070|.096 (292) ·
**k25-48 .163|.247 (73) · k49+ .092|.207 (29)**; P(jump) in the k25-48
bin .022|.082. Above stage: calibrated at every k.

**Reading.** On expert states the model is calibrated through 24 frames
of silence and under-fires by ~35–55 % in the long-silence tail — a tail
the expert visits ~4 frames per game (73 + 29 frames in 24 games). The
rising-hazard regime that ends a silent fall is DATA-STARVED in the
corpus, and closed-loop the bot lives there (190 trips × ~30 frames per
rollout). So: closed-loop compounding into a thin tail, with the
under-fit of that tail already visible teacher-forced. Not the
hold/change head (the k ≤ 24 bins are calibrated), not the carried state
(windowed model), and not the trunk's input history — with event heads
the trunk's controller slot is zeroed over the whole window, so "k frames
of silence" reaches this model only through the game state (FALLING with
a growing action_frame, constant vy, drifting x) and the one-frame prev.

**Lever candidates (imitation-side, in order):**

1. **Upweight the tail** — loss weight on expert offstage frames with
   silence ≥ 13 frames (they exist: ~4/game × 3000 files ≈ 12k frames).
   No sim, no new data; tests "thin tail" directly. Pass: Q5 k25+
   model ≥ expert; closed-loop hazard no longer decaying (≥ .15 at 18–24
   f); high-band decided return ≥ 0.6; coherence/fidelity unchanged.
2. **Manufacture the states** — bit-exact replay seeding (09-21 Seed
   module): seed the sim at an expert offstage actionable frame, inject
   k ∈ 6..30 frames of neutral on the own port (opponent's recorded
   inputs), embed the resulting window, label with the expert's recorded
   decision at the seed frame, keep only while the decision stays valid
   (jump still in hand, y above up-B range, label = stick toward/up or
   jump). Mixed in like the drill sets. This is the DAgger-style
   correction set §7y said had no reference — the reference is the
   expert's own decision, carried over a short silence.
3. More epochs/data: does NOT move this (e3 decided return 0.344 vs
   0.450 at 1 epoch) — it fixed the carried-off illusions, not this.

### 10-05 15:40 — lever 1 (loss weights) fails; lever 2 (sim DAgger) running

**Lever 1 = queue 15** (`--silent-fall-weight 20`, k ≥ 13, ledge/dead/
helpless/hitstun excluded — `ExPhil.Training.SilentFallWeighting`, a
per-chunk `loss_weights_fn` seam in `ChunkPipeline`; and `--offstage-weight
3`, the V2 knob now wired on the windowed path). Coverage on 20 holdout
games: offstage = 8.5 % of frames (ledge hangs, edgeguards), gated
silent-fall frames = 0.12 % (~9/game, expert acts on 8 % of them — the
probe's number once ledges were excluded). Single seed each, 1 epoch,
against evt2ctx_ck8:

| arm | val | fidelity | mismatch | return | carried-off | decided | high-band decided | Q5 k25-48 (expert .247) |
|---|---|---|---|---|---|---|---|---|
| evt2ctx_ck8 (baseline) | 1.058 | 0.185 | 0.161 | 0.299 | 0.389 | 0.450 | 0.40 | **0.154** |
| sf20 | 1.054 | 0.197 | 0.209 | **0.191** | 0.311 | **0.228** | **0.14** | **0.124** |
| off3 | 1.058 | 0.209 | **0.088** (floor 0.084) | **0.373** | **0.281** | 0.463 | 0.40 | 0.151 |
| e3 (3 ep, for reference) | — | 0.149 | 0.129 | 0.312 | 0.140 | 0.344 | 0.29 | 0.163 |

- **sf20 is worse on the quantity it targeted**: teacher-forced P(input |
  silent ≥ 25 f) fell 0.154 → 0.124, closed-loop resume hazard 0.12 →
  0.08 at 9–24 f, high-band decided return 0.40 → 0.14. Coherence intact
  (repeat 0.746). Twenty-fold weight on ~3 frames/game does not sharpen
  the conditional; it moved it the wrong way. (`loss_weights` multiply
  the base weights, so the 4:1 active:neutral ratio inside the regime
  was preserved — not a ratio artefact.) Verdict: more gradient on the
  same thin tail is not the lever.
- **off3 is a real, separate win**: mismatch 0.088 — at the expert's
  split-half floor for the first time — carried-off share 0.389 → 0.281,
  return 0.299 → 0.373, resume hazard higher at 6–12 f (0.23/0.22/0.18
  vs 0.13–0.14). It ends silences sooner and picks the expert's tool; the
  follow-up still fails (high-band decided return unchanged at 0.40).
  Candidate for the port recipe pending a seed replicate; the tail
  (Q5 k25+) did not move.
- Neither arm touches the data-starved states themselves — which is
  what both results say the problem is.

**Lever 2 (replay seeding) is dead on this corpus**: `Seed.from_replay`
is bit-exact until the first divergence and then a different game, and
38 of 40 ranked FD Fox games diverge inside the first few hundred
frames (ports 3/4, early percent mismatches) — 1 usable seed from 141
decision frames (`scripts/silent_fall_set.exs`, kept for local Dolphin
games where seeding is exact).

**Lever 2 as sim DAgger (= queue 16, running)**: the bot manufactures
the silent-fall states itself (190/rollout). `scripts/sim_recovery_dagger.exs`
rolls the 1-epoch baseline out in the sim (3 seeds × 32 envs × 3600 f,
self-play) and relabels every offstage/below airborne frame with
`ExPhil.Agents.FoxRecoveryExpert` — the July E2 post-mortem relabeler
("mashed jump 55–161 frames offstage, pressed B zero times": the same
disease): jump if one is left, else Firefox aimed at the ledge, steer
mid-special, DI in hitstun. Onstage play and ledge hangs are NOT
relabeled. The policy's actual press rides in `:prev_controller`; each
trip carries 90 input-only context frames from its own env. Set r1: 494
trips, 16.4k relabeled frames (+44.5k context), the policy silent on
35 % of them, labels B 37 % / jump 18 %. New plumbing:
`MixFrames.load_lists` + `Streaming.create_dataset(frame_lists:)` keep
trip boundaries on the windowed mix path (the flat mix path let lazy
windows straddle trips and rejected input-only prefixes). Arms: mix
oversample 4 and 16 (the mix is appended after the main corpus each
epoch). Pass: high-band decided return ≥ 0.6, resume hazard ≥ .15 at
18–24 f, mismatch ≤ 0.16, coherence/fidelity unchanged.

**Two ways the mix wrecked the model before it taught anything (16:30):**

1. **Appended block.** The mix convention appended all mix batches after
   the main corpus: a ~470-batch recovery-only block at the end of the
   epoch sent held-out val loss 1.06 → **4.92** (catastrophic forgetting
   from ordering; training loss looked normal all epoch). Fixed: mix
   batches are now INTERLEAVED by credit, 1 per ~151 main batches at
   oversample 4 (`pipeline.ex`). Val back to 1.043.
2. **DAgger prev + event heads = "change everywhere".** The drill
   protocol puts the policy's ACTUAL press in `:prev_controller` and the
   expert's correction in `:controller`. With event heads the previous
   input is only the hold/change SELECTOR (the trunk's copy is zeroed),
   and by construction 98 % of the set were change events (label ≠ prev;
   the expert corpus is ~76 % holds). 108 interleaved batches of that
   taught the bot to change its input every frame everywhere: closed-loop
   repeat share **0.75 → 0.36**, dashes 40 → 3/min, wavedashes 4.6 →
   0.15/min, SDs 1.3 → **6.5/min**, offstage trips 6.7 → 15/min, fidelity
   0.185 → 0.472 — while val loss (1.043), coherence (repeat 0.756) and
   every teacher-forced number were untouched. The states are not the
   problem (per-dim sim-vs-replay embedding gap: 6 of 296 dims shift, the
   largest = opponent character, Fox in self-play). Fixed: for event-head
   models `prev_controller` = the PREVIOUS LABEL inside the trip (82 %
   holds), the first trip frame keeps the real press at t−1; the states
   stay the bot's own. `sim_recovery_dagger.exs` now prints the hold share.

Lesson for every mixed-in set from here: check (a) ordering — interleave;
(b) hold share of label-vs-prev ≈ the corpus's; (c) closed-loop rates
(repeat share, dashes/min) — teacher-forced metrics are blind to both.

3. **Dose and style (attempt 2, 17:00).** Interleaved + label-prev, full
   set × 4 (108 batches, val 1.07): still fidelity 0.33, SDs 2.3/min,
   repeat share 0.57, **neutral share 0.05** — and this time it shows
   teacher-forced too: coherence neutral 0.26 → 0.10, repeat 0.75 → 0.65,
   B presses 14 → 40/min on expert frames. Not ordering, not the prev
   channel: the labels. Within offstage states the set is +50 % of the
   corpus's offstage data (expert ≈ 120k offstage targets/epoch; mix
   15.5k × 4 = 62k) in a robotic style — full stick deflection on every
   frame, B on 35 % of frames, X taps — and a 256×2 trunk carries that
   style onstage. Evals kept in `eval_runs/1001_queue/evt2ctx_ck8_dag4_fullset_attempt2`.
   Next (queue 16 as now): `--only-silent` — relabel ONLY the frames where
   the policy's actual press was neutral (the diagnosed defect and nothing
   else; 6.0k frames, hold share 0.77, B 45 % / jump 20 %) at oversample 2
   (~27 batches), plus the full set at oversample 1 as a dose check.

4. **Silent-only set, oversample 2 (18:30 / 19:00).** Style damage gone:
   val 1.044–1.046, coherence repeat 0.75–0.76 / neutral 0.25–0.29,
   **fidelity 0.172–0.177 (best 1-epoch yet)**, closed-loop rates
   expert-like (repeat 0.73, neutral 0.30, dashes 51/min). Recovery NOT
   better: high-band decided return **0.32 / 0.27** (baseline 0.40), return
   0.32 / 0.24. The first run (`evt2ctx_ck8_dags2_illusionlabel`) raised
   `jump>side_b` deaths 7 → 24 — which exposed a **bug in
   `FoxRecoveryExpert` since July**: `tap_upb` pressed B on the ledge-aim
   vector, and a sideways-dominant aim (player near ledge height) fires
   ILLUSION, not Firefox — the "side-B low" self-destruct, taught as the
   label. Fixed (straight-up press, aim during the charge; test added).
   The corrected set's run instead grew a new carried-off mode: **grounded
   Firefox at the lip** (x ≈ 86–87, y = 0, FIREFOX startup in
   `pre_actions`; 20 trips, 18 fatal) — "B + stick up near the edge"
   generalised from the airborne silent-fall states onto the lip. Silent
   falls got a little shorter (never-helpless deaths 57 % vs 70 %), the
   hazard no longer decays as cleanly, but the follow-up still fails.

**Where this leaves lever 2.** Sim DAgger with the rules expert is a net
negative for recovery at every dose and label variant tried today, while
being a small positive for fidelity when kept to silent frames. The
labeler is the weak part: a rules expert has a *style* (full deflection
every frame, robotic taps) and *mistakes* (the Illusion press), and a
256×2 trunk carries both into neighbouring states. If this route is
pursued, the labeler must be the expert distribution itself (e.g. sample
the means from the recovery-means table per situation and the stick from
expert offstage frames), not rules.

**Extrapolation hypothesis (Q6, 19:00).** The silence decays because the
state leaves the data: FALLING with a growing `action_frame` (embedded as
frames/60, clipped at 2.0 — in range, but the expert never falls 30 frames
straight offstage). If P(input) falls as that counter grows on expert
frames, the lever is a saturation cap on the feature — no new labels, no
reweighting, just stopping the extrapolation. Probe: on expert silent
below-stage frames, overwrite the own action_frame dim (last 12 window
frames) with the value for 1 / 30 / 60 frames.
**Result (Q6):** base 0.081 → af:=30 0.074 → af:=60 0.070. Real, minor
(−14 %). Not the driver.

### 10-05 20:00 — the silence is entered, not failed to escape

**Probe on the bot's OWN states** (`scripts/interp_silent_fall_probe.exs`:
the silent-only DAgger export is 5.7k frames where the policy's actual
press was neutral, each with 90 frames of its own context; embedded like
training, teacher-forced with a neutral previous input and the neutral
action — i.e. the proper joint P(any input) = 1 − P(all heads neutral)):

- Model P(any input) on its own silent frames: 0.047 (k 1–6) → 0.027
  (k 49+); by depth 0.052 (ledge) → 0.026 (< −90). Matches the live hazard.
- Ablations: action_frame := 1 → 0.049; jumps := 0 → 0.046; the whole
  80-frame history replaced by the last frame → 0.043. **Replacing every
  dim except the previous-input slot with an EXPERT silent offstage
  window → 0.047.** The state does not silence the model.
- The expert's own silent offstage frames through the same protocol:
  model 0.058 | expert acts 0.045 (n=177 in 12 games — the expert is
  almost never silent offstage). Calibrated. (Q5's "tail under-fire" was
  partly a protocol artefact: it teacher-forces the expert's actual
  action, which conditions the stick heads on the pressed buttons.)

So: both resume from silence at ~0.05/frame; the policy's resume hazard
is state-insensitive and roughly the expert's. **The difference is
upstream — the bot lets go in danger.** From the traces, P(enter silence |
active) per 3 f, decided trips below stage level:

| state | bot (e3) | expert | ratio |
|---|---|---|---|
| −20…−60, jumpless | **0.072** | 0.017 | 4× |
| −20…−60, jump in hand | **0.126** | 0.044 | 3× |
| < −60, jumpless | **0.098** | 0.019 | 5× |
| ledge band, jump in hand | 0.156 | 0.094 | 1.7× |

and among active frames the bot holds toward the stage half as often
(0.17–0.45 vs 0.39–0.76), holding OUT 0.50 vs 0.38 on the first offstage
frame. The approach differs too: dash/run in the last 30 frames 0.68 vs
0.35, facing the edge 0.79 vs 0.39, speed med 1.4 vs 0 — the bot arrives
at the lip running outward; the expert arrives standing or already turned.

**The defect, finally named:** not "frozen", not "can't resume" — a
3–5× excess hazard of RELEASING a deflected stick in dangerous offstage
states, plus too little drift toward the stage. Q7 (recovery probe) tests
whether the release excess is already there teacher-forced on expert
frames (structural: the hold/change head lets go too easily offstage) or
only in the bot's own states (closed-loop).

**Q7 — the release floor (teacher-forced, expert frames, P(release to
neutral | stick deflected, offstage below stage) per frame):**

| band | baseline | e3 (3 ep) | off3 | **rel** (release head) | expert |
|---|---|---|---|---|---|
| −20…−60 jumpless | 0.024 | 0.022 | **0.015** | 0.026 | 0.007 |
| −20…−60 jump in hand | 0.042 | 0.053 | 0.042 | 0.039 | 0.028 |
| < −60 jumpless | 0.026 | 0.012 | 0.012 | **0.038** | 0.000 |
| < −60 jump in hand | 0.027 | 0.033 | 0.019 | 0.039 | 0.000 |
| ledge, jumpless | 0.030 | 0.027 | 0.024 | 0.025 | 0.024 |
| ledge, jump in hand | 0.071 | 0.084 | 0.066 | 0.066 | 0.046 |

Structural, yes — the floor is there on expert frames — but NOT a
parametrisation artefact: **`--stick-release`** (release logit beside
hold, `Heads.collapse_hold_release_change`, queue 17 `evt2ctx_ck8_rel`)
learned the same floor and a worse deep band; closed-loop it entered
silence MORE (< −60 jumpless 0.143 vs 0.098 per 3 f; high-band decided
return 0.23 vs 0.40). Val 1.049, coherence repeat 0.753 / neutral 0.315
(the closest neutral share to the expert's 0.303 yet), fidelity 0.195 —
the head is sound, it just doesn't change what is learned. Read: the
hold/release/change decision is a weakly state-conditioned prior
(hold_agreement 0.60, change_recall 0.30 say the same thing globally);
~42k training frames where the expert never releases deep offstage do not
move it. The silent fall is one symptom of that under-conditioning.
Epochs don't help (e3 worse with the jump in hand). The one lever that
moved the floor is the offstage LOSS WEIGHT (×3 halved it in the jumpless
bands) — queue 18 runs ×8 as the dose point.

**`--offstage-weight 3` replicates (22:20, seed 906):** mismatch **0.084**
(= the expert's split-half floor; s905 0.088), means_js 0.265, return
**0.394** (s905 0.373; baseline 0.299), decided-trip return 0.487,
carried-off share 0.222, fidelity 0.198 (s905 0.209), coherence repeat
0.78 / neutral 0.309. Every recovery number the best of any 1-epoch arm,
across two seeds. **Into the port recipe**: windowed + prev_q + events +
context + chunk 8 + `--offstage-weight 3`, ≥ 3 epochs. `rel_off3` (release
head + weight): Q7 jump-in-hand 0.031 vs expert 0.028 — the closest yet
there — but the ledge band over-holds (0.016 vs 0.024), neutral share
drifts to 0.169 and recovery is no better than off3 alone (return 0.315);
the head redistributes, the weight does the work.

**But the silent fall itself did not move — in any arm.** High-band
decided-trip return (the silent fall's own metric; expert 0.95):

| baseline | off3 s905 | off3 s906 | off8 | e3 | mamba | sf20 | dags2 | rel | rel_off3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.40 | 0.41 | 0.34 | 0.29 | 0.29 | 0.14 | 0.27–0.32 | 0.23 | 0.37 |

Both off3 seeds enter silence in danger at the baseline rate (0.05–0.12
per 3 f vs the expert's 0.017–0.044) and their Q7 floors disagree with
each other (s905 0.015 / s906 0.027 at −20…−60 jumpless) while their
closed-loop outcomes agree — the teacher-forced floor is seed-noisy and
does not predict the loop. off3's gain is means selection (mismatch at
the floor) and fewer carried-off trips (0.39 → 0.22–0.28), i.e. the lip,
not the fall.

**Where this leaves the silent fall (22:45):** resistant to loss weights
(tail or band), to DAgger with a rules labeler, to an explicit release
decision, and to epochs. The hold/change decision is under-conditioned on
state on this 256×2 MinGRU testbed; whether that is capacity (the Mamba
port is larger) or a learning-signal problem is the open question for
the morning. Candidate next levers, all imitation-side: (1) a dedicated
danger readout for the hold/release/change logits (y, jumps, distance to
the edge as head features — the `--event-context` pattern applied to
state); (2) a targeted hold-weight on offstage frames where the expert
HOLDS a deflected stick (off3's win, aimed); (3) test the capacity
hypothesis directly on the Mamba testbed (Q7 + entering-silence hazard
on `mamba_evt2ctx_ck8`). Decided with Bradley.

### 10-05 22:45 — queue 18: `--offstage-weight 8` closes the loss-weight lever

The dose point. ×8 pushed the teacher-forced release floor to the expert's
and nothing downstream followed:

| | off3 (s905/s906) | **off8** | expert |
|---|---|---|---|
| Q7 −20…−60 jumpless | 0.015 / 0.027 | **0.017** | 0.007 |
| Q7 < −60 jumpless | 0.012 / — | **0.004** | 0.000 |
| Q7 < −60 jump in hand | 0.019 / — | 0.020 | 0.000 |
| closed-loop P(enter silence) −20…−60 j0 / < −60 j0 (per 3 f) | 0.055 / 0.103 · 0.054 / 0.092 | **0.067 / 0.081** | 0.019 / 0.034 |
| high-band decided return | 0.40 / 0.41 | **0.34** | 0.95 |
| means mismatch (expert floor 0.084) | 0.088 / 0.084 | **0.151** | — |
| return | 0.373 / 0.394 | 0.318 | 0.911 |
| carried-off share | 0.22–0.28 | 0.363 | 0.055 |
| fidelity | 0.209 / 0.198 | 0.205 | 0 |
| val | 1.07 | 1.073 | — |

Coherence repeat 0.738 / neutral 0.245 (expert 0.764 / 0.303). The Q7
floor is now within +0.010 / +0.004 of the expert, but the bot still
enters silence in danger at 3–4× the expert's rate and carries itself off
MORE than baseline (0.36 vs 0.39 → off3 had cut it to 0.22). The weight
is a dose with an optimum at ~3: at 8 it buys the teacher-forced release
statistic while un-learning the lip (mismatch back to 0.151 = baseline
territory) — classic reweighting over-fit to the up-weighted frames.

Two conclusions. **(a) The teacher-forced release floor is not the causal
variable** — three arms now have floors from 0.004 to 0.027 in the deep
bands with the same closed-loop hazard (0.05–0.10). Q7 is a readout of
what the head does on EXPERT states; the loop enters silence from the
bot's OWN states (running off the lip facing out, stick held out), which
donor swaps already showed is not the input the model reads. **(b) The
loss-weight lever is closed**: ×3 is in the recipe for the lip, ×8 is
worse everywhere that matters. Remaining imitation-side levers are the
ones that change what the hold/change decision is conditioned on (danger
readout features) or where the loop's states come from — and the capacity
question, being tested now on the Mamba checkpoint (Q7 + traced roll).

### 10-05 23:30 — capacity check on `mamba_evt2ctx_ck8`: same floor, same hazard

Existing checkpoint (Mamba backbone on the testbed recipe, 1.80 M params vs
the MinGRU's 1.21 M, 1 epoch, val 1.090), no training — Q7 and a traced
recovery roll only:

| | MinGRU baseline | **Mamba** | expert |
|---|---|---|---|
| Q7 −20…−60 jumpless / jump in hand | 0.024 / 0.042 | **0.029 / 0.033** | 0.007 / 0.028 |
| Q7 < −60 jumpless / jump in hand | 0.026 / 0.027 | **0.036 / 0.023** | 0.000 / 0.000 |
| Q7 ledge jumpless / jump in hand | 0.030 / 0.071 | 0.024 / 0.071 | 0.024 / 0.046 |
| P(enter silence) −20…−60 j0 / < −60 j0 | 0.05–0.07 / 0.08–0.10 | **0.059 / 0.103** | 0.019 / 0.034 |
| P(enter silence) −20…−60 j1+ | 0.085–0.13 | 0.158 | 0.032 |
| high-band decided return | 0.40 | 0.34 | 0.95 |
| mismatch / return / carried-off | 0.16 / 0.30 / 0.39 | 0.183 / 0.244 / 0.349 | 0.084 / 0.911 / 0.055 |

Q5/Q6 on the Mamba match the MinGRU story too (resume from silence
~0.07–0.15 vs expert, state-insensitive to action_frame). A 1.5× larger
backbone of a different family learns the identical release floor and the
identical entering-silence hazard. Caveat: this is 1.8 M params at 1
epoch, not the production Mamba; but if capacity were the limit the
bigger model should at least bend the floor, and it doesn't move at all.
**Read: a learning-signal problem, not a capacity problem.** The hold/
change decision is under-conditioned on danger because the imitation loss
never asks it to be — ~42 k deep-offstage frames where the expert holds
are 0.2 % of the epoch and are already fit to within the per-frame noise
(off8 showed the teacher-forced statistic can be driven to the expert's
without the loop following).

**Verdict for the morning.** The silent fall is (1) entering silence from
the bot's OWN states — running off the lip facing out with the stick held
out, states the expert rarely produces; (2) invisible teacher-forced; (3)
untouched by loss weights, epochs, backbone size, an explicit release
head, and a rules-labeled DAgger set. What is left imitation-side, in
order of my recommendation:

1. **Danger readout for the hold/change logits** — feed y, jumps_left,
   signed distance to the nearest edge, and speed_y directly into the
   stick heads' hold/change dense (the `--event-context` pattern applied
   to state). Cheap (one flag, one arm), tests whether the decision can
   use danger when handed it instead of having to find it in the trunk.
   Pass = P(enter silence) −20…−60 j0 ≤ 0.04 AND high-band decided return
   ≥ 0.6 at fidelity ≤ 0.21.
2. **Expert-labeled DAgger** — the DAgger plumbing works (fidelity
   improved with the silent-only set); the labeler was the problem. A
   labeler = the policy's own expert-trained stick head teacher-forced on
   the HOLD branch (i.e. relabel the bot's silent frames with "keep
   holding what you held") is not a rule in the decode sense; it changes
   training data only. Needs Bradley's ok on the data-pollution concern.
3. **Approach, not fall** — the bot arrives at the lip running outward
   (dash in last 30 f 0.68 vs 0.35, facing edge 0.79 vs 0.39). off3's
   win was the lip. An edge-approach weight (frames within 15 units of the
   edge, grounded, moving outward where the expert stops/turns) attacks
   carried-off share directly rather than the fall after it.

Mamba port with the recipe (windowed + prev_q + events + context + chunk
8 + `--offstage-weight 3`, ≥ 3 ep) and the live look: Bradley's call.

## 10-06 — the silent fall as a map; the semi-Markov main stick

Bradley's direction (10-06 morning): (1) an eval that tracks entering
silence everywhere, by situation — "turn the silent fall into a map we
can read for every arm"; (2) build the most principled lever, **change
the timescale of the decision**; if that is not it, attack the copy
shortcut generally; (3) a sim branch for replay parity so real DAgger
is available whenever we want it (standing, independent of the lever).
The danger-readout idea is a probe, not a fix (it would not generalise
across characters); the edge-approach weight is tuning — dropped.

### The map — `ExPhil.Eval.SilenceMap`

`lib/exphil/eval/silence_map.ex`. For every frame with a successor it
classifies the input transition (active → enter_silence / change / hold,
silent → resume) and counts it in three bucket families: universal
physical **state** bins (grounded centre / edge, airborne onstage,
offstage high / low / deep × jumps, ledge hang, hitstun), every
`ExPhil.Situations` **label** the frame carries, and **age** (onstage /
offstage × how long the current input has been held). Same game shape
as `PlayStats`, so `fidelity_scorecard.exs` now writes `silence_map.json`
beside `fidelity.json` for every arm (RESULT lines: worst buckets with
z ≥ 3, by state, by age) and `expert_reference.exs --silence-map-out`
builds the expert map once (`eval_runs/1002_fidelity/expert_silence_map_fd.json`,
split halves agree to ±0.002; the existing `expert_fd.json` reproduced
byte-identical). Compare = model / expert hazard with a binomial z,
min n 200. Test: `test/exphil/eval/silence_map_test.exs`.

**First read, `off3` (3 seeds × 32 × 3600 f), P(enter silence | active) per frame:**

| state | bot | expert | ratio |
|---|---|---|---|
| grounded centre | 0.048 | 0.052 | 0.92 |
| grounded edge | 0.041 | 0.048 | 0.86 |
| airborne onstage | 0.054 | 0.042 | 1.3 |
| hitstun | 0.040 | 0.025 | 1.6 |
| offstage high j0 / j1+ | 0.033 / 0.057 | 0.007 / 0.028 | **4.7** / 2.0 |
| offstage low j0 / j1+ | 0.030 / 0.050 | 0.010 / 0.019 | **3.1** / 2.7 |
| offstage deep j0 / j1+ | 0.051 / 0.060 | 0.007 / 0.010 | **7.0** / **5.9** |

**The bot's hazard is flat — 0.03–0.06 in every state — while the
expert's spans 0.007 offstage to 0.052 onstage.** Onstage the bot is at
or below the human; the silence budget is right and mislocated, not too
big (closed-loop neutral share 0.325 vs 0.28). Worst situation labels:
`below_ledge` ×3.8, `resource_exhausted` ×3.5, `near_blastzone` ×3.3,
`being_edgeguarded` ×2.8, `recovery_high` ×2.7, **`shield_pressure_theirs`
×2.4** (letting go of shield under pressure — the same defect outside
recovery, as "general" predicts), `edge_danger` ×2.2. Resume is also
higher offstage (deep ×4). Change hazard is close to the expert's
(×1.1–1.8) — it is the release that is wrong, not changes in general.

**By age (frames the current input has been held), offstage:**

| age | bot | expert | ratio |
|---|---|---|---|
| 1–3 | **0.059** | 0.013 | 4.5 |
| 4–7 | 0.037 | 0.014 | 2.6 |
| 8–15 | 0.025 | 0.012 | 2.0 |
| 16–31 | 0.029 | 0.007 | 4.1 |
| 32+ | (n < 200) | 0.004 | — |

Onstage: bot 0.066 / 0.044 / 0.038 / 0.031 / 0.027 vs expert 0.041 /
0.048 / 0.044 / 0.023 / 0.019. Two facts for the design: the expert's
hazard **falls with age** (3× from short to 32+ frames; long holds get
safer to continue), and the bot's excess is **largest right after a
press** — a fresh offstage input is fidgeted away at 6 %/frame, the same
rate as onstage (0.066), where the expert's differs 3×. The bot almost
never sustains a 32-frame hold offstage.

### The lever — `--stick-duration C` (semi-Markov main stick)

The event-head recipe zeroes the previous input in the trunk (that
closed the copy shortcut) and the head sees only last frame's bucket as
a selector, so the policy has **no notion of how long it has been
holding**, and the per-frame hold/change decision is re-drawn 50 times
over a fall. The semi-Markov head decides the main-stick pair only at
**decision frames** — the frame the pair changes, and every C-th frame
of a continuing hold — and a **duration head** says, given the pair just
chosen, how long it is held (classes 1..C−1, C+ = re-decide at age C).
Between decisions the sampler holds. A duration chosen once, in the
state the press was made in, cannot be fidgeted away frame by frame —
which is where the excess is. Honest caveat: a state-blind hazard
rescaled to a coarser timescale gives the same survival curve; the gain
has to come from the decision being made in the informative state and
from the age feature, so the first C frames after a press are where
this should show first (age 1–3 and 4–7 rows of the map).

Implementation (windowed AR path only; BPTT refuses):
- `Heads.build_autoregressive_head(stick_duration: C)`: `"prev_age"`
  input → `Heads.age_bucket/1` (11 bins) → zero-init `ar_prev_age_embed`
  added to r0; `ar_duration_{hidden,logits}` on r3 (after the pair);
  output a 7-tuple (`Loss.split_duration_head/1`).
- `Imitation.Loss`: `prev_age_from_window/3` reads the age off the
  window's prev-action slots (trailing run of equal main-stick pairs);
  `stick_decision_targets/3` builds the decision mask (event ∨ age ≡ 0
  mod C) and the duration class from the chunk futures (needs
  `--chunk-horizon ≥ C−1`); `Policy.Loss.imitation_loss(main_stick_mask:)`
  scores main_x/main_y at decision frames only, renormalised; duration CE
  × `--stick-duration-weight`. Val uses the same likelihood (NOT
  comparable with per-frame arms' val).
- `Sampling`: `:event_prev_age` / `:stick_commit` ride in the head map;
  committed rows get a one-hot spike on prev for main_x/main_y; the
  duration head is sampled (temperature of main_x) when commit = 0;
  result carries `stick_commit` (frames left). Mode-of-N and the
  deterministic path handled.
- `Agent`: `{age, commit}` per batch row (`batch.stick_runs`) and for
  the single path (`stick_run`), advanced on every emitted/observed
  controller, reset with the rows; `stick_duration` read from the export
  config. Checkpoints without the head behave exactly as before.
- Tests: `test/exphil/networks/policy/stick_duration_test.exs` (6).

Pass criteria for the first arm (`evt2ctx_ck8_off3_dur8`, recipe +
`--stick-duration 8`): SilenceMap enter_silence offstage ratios ≤ 2 in
every jumpless band (from 3–7) and age 1–3 offstage ≤ 0.03 (from 0.059);
high-band decided return ≥ 0.6 (0.40); fidelity ≤ 0.21; mismatch ≤ 0.10;
coherence repeat ≥ 0.70 / neutral 0.22–0.33; no new SD mode (SDs/min,
dashes/min within the fidelity reference). Dose point: `dur16` with
`--chunk-horizon 16`.

**`dur8` read (queue 19, 1 ep, val 3.35 — its own likelihood):** the map
moved for the first time in any arm, and a new failure appeared.

| enter_silence | off3 | **dur8** | expert |
|---|---|---|---|
| offstage deep j0 / j1+ | 0.051 / 0.060 | **0.026 / 0.043** | 0.007 / 0.010 |
| offstage high j0 | 0.033 | **0.024** | 0.007 |
| offstage low j0 / j1+ | 0.030 / 0.050 | **0.020 / 0.046** | 0.010 / 0.019 |
| age 1–3 / 4–7 offstage | 0.059 / 0.037 | **0.039 / 0.022** | 0.013 / 0.014 |
| grounded centre / airborne onstage | 0.048 / 0.054 | 0.035 / 0.035 | 0.052 / 0.042 |

Offstage release hazards roughly halved (ratios ×7 → ×3.5, ×4.7 → ×3.4,
×3.1 → ×2.0); onstage it now under-releases (×0.67–0.83). But recovery got
WORSE: return 0.206 (off3 0.37–0.39), mismatch 0.234, decided-trip return
0.28, fidelity 0.225 — and I read the death table as the reason:
**`side_b 35 (35†)`**, buttons sampled BEFORE a committed sideways stick =
Illusion instead of Firefox. **That reading was WRONG (corrected 10-06
22:00, after dur8e):** the "by move" table is `carried_off_by_move` — the
move the bot was performing when it got HIT off, not the first recovery
move — and off3's own table already said `side_b 33 (32†)`. The
side-B-then-die mode is the baseline's (the bot Illusions near the edge,
gets hit, and the carried-off death rate is 0.86–0.95 in EVERY arm vs the
expert's 0.16). The traces say dur8 actually pressed B with the stick UP
more than any other arm (45 up-B onsets in died trips vs off3's 11). So
the edge fix below addresses a real but different thing (a press should
be free to re-aim the stick — it is what the data does), not dur8's
recovery drop, whose cause is still open. The semi-Markov decision as the
JOINT controller change: a button edge is a decision frame too (training
mask `event = stick pair changed ∨ any button edge`; sampler: committed
only if `commit > 0 ∧ no button edge this frame`, an edge re-samples the
duration; Agent unchanged). Implemented as `dur8e`.

**`dur16` read (queue 19, 1 ep, val 3.99, `--chunk-horizon 16`):** the
dose point adds nothing on the map and confirms the Illusion diagnosis.

| | off3 | dur8 | **dur16** | expert |
|---|---|---|---|---|
| enter_silence deep j0 / j1+ | 0.051 / 0.060 | 0.026 / 0.043 | **0.027 / 0.054** | 0.007 / 0.010 |
| enter_silence high j0 / low j0 | 0.033 / 0.030 | 0.024 / 0.020 | **0.027 / 0.018** | 0.007 / 0.010 |
| enter_silence age 1–3 / 4–7 offstage | 0.059 / 0.037 | 0.039 / 0.022 | **0.045 / 0.025** | 0.013 / 0.014 |
| onstage grounded centre / airborne | 0.048 / 0.054 | 0.035 / 0.035 | **0.039 / 0.039** | 0.052 / 0.042 |
| return / mismatch | 0.37–0.39 / ~0.10 | 0.206 / 0.234 | **0.313 / 0.144** | 0.911 / 0.084 |
| decided-trip return | 0.40 | 0.28 | **0.456** | 0.915 |
| fidelity / repeat / neutral | 0.21 / 0.73 / 0.25 | 0.225 / 0.733 / 0.209 | **0.222 / 0.754 / 0.263** | — / 0.764 / 0.303 |
| side-B trips (deaths) | — | 35 (35†) | **18 (18†)** | 5 (3†) |

Read: the offstage hazard gains are a property of the semi-Markov
decision itself, not of C (dur16 ≈ dur8 on every offstage band); the
onstage under-release is also the same. Recovery sits between dur8 and
off3. (The "side-B row" is carried-off-by-move — see the correction
above; its 18/18 is the baseline's carried-off death rate, not an
Illusion count.) Coherence is the best of any arm so far (repeat 0.754,
neutral 0.263 — both inside the expert band for the first time). Kept
C = 8 (lower val) and ran the joint-decision fix → `dur8e` (queue 20,
20:43–21:42).

**`dur8e` read (queue 20, 1 ep, val 3.25 — lowest of the three):**

| | off3 | dur8 | dur16 | **dur8e** | expert |
|---|---|---|---|---|---|
| enter_silence deep j0 / high j0 / low j0 | 0.047 / 0.032 / 0.029 | 0.026 / 0.024 / 0.020 | 0.027 / 0.027 / 0.018 | **0.025 / 0.023 / 0.028** | 0.007 / 0.007 / 0.010 |
| enter_silence age 1–3 offstage | 0.059 | 0.039 | 0.045 | **0.059** | 0.013 |
| onstage grounded centre / airborne (ratio) | 0.94 / 1.33 | 0.67 / 0.83 | 0.76 / 0.94 | **0.86 / 1.10** | 1 |
| return / mismatch / decided return | 0.37–0.39 / ~0.10 / 0.46 | 0.206 / 0.234 / 0.28 | 0.313 / 0.144 / 0.456 | **0.23 / 0.168 / 0.264** | 0.911 / 0.084 / 0.915 |
| carried-off share / died | 0.281 / 0.859 | 0.326 / 0.944 | 0.356 / 0.947 | **0.172 / 0.936** | 0.055 / 0.163 |
| fidelity / repeat / neutral | 0.21 / 0.73 / 0.25 | 0.225 / 0.733 / 0.209 | 0.222 / 0.754 / 0.263 | **0.191 / 0.777 / 0.296** | — / 0.764 / 0.303 |

Against the pass criteria: fidelity ≤ 0.21 ✓ (first arm ever), onstage
ratio ≥ 0.8 ✓, deep/high j0 kept ✓, coherence inside the expert band ✓
(the best repeat/neutral of any arm, and val lowest); return ≥ 0.37 ✗
(0.23), mismatch ≤ 0.10 ✗ (0.168), age 1–3 offstage back to the baseline
✗. The closed-loop decided-trip hazard (`recovery_enter_silence.js`,
−20..−60 j0) is 0.070 vs off3 0.055 / dur8 0.047: the button-edge
decision frames gave the press back its freedom and the offstage
early-release came back with it.

**What the traces say the deaths ARE (`scripts/recovery_death_shape.js`,
self-destruct trips, expert = traced FD reference):**

| | expert (104 SDs) | off3 | dur8 | dur16 | dur8e |
|---|---|---|---|---|---|
| input changes per died trip (median) | 17 | 8 | 11 | 11 | 9 |
| died with a jump left | 7 % | 26 % | 15 % | 13 % | **30 %** |
| B onsets below y −40: stick UP share | **81 %** (48/59) | 24 % | 35 % | 18 % | 17 % |
| B onsets in died trips, neutral stick (laser) | 3 | 43 | 20 | 22 | 35 |
| B onsets in died trips, down stick (shine) | 5 | 9 | 31 | 56 | 40 |

The expert's own deaths are long fights (17 changes) that end in Firefox
attempts; the bot's are short (8–11 changes), one in four ends with a
jump unused, and when it presses B deep the stick is up a fifth of the
time — the rest are lasers and shines offstage. This is the 10-05
`b_press_stick.exs` finding (neutral on the press frame 27.6 % vs 2.5 %)
seen from the death side, and **no duration arm moved it** (dur8 moved
the up share most, to 35 %, and died anyway). The duration arms shifted
the wrong-stick presses from neutral toward down — the shine — which is
a change of label, not of behaviour.

**Verdict on the semi-Markov stick (lever 2 of Bradley's 10-06 plan):**
it is a real coherence/fidelity win — `dur8e` is the first arm inside the
fidelity target and the expert's repeat/neutral band at the same time,
with the lowest val — and it does not fix recovery. The silence map it
halved (dur8) did not convert into returns, and the edge-decision variant
gave the hazard back. The recovery defect has two faces the duration
head cannot reach: (a) entering silence from the bot's own offstage
states (the map), and (b) pressing B deep with the wrong stick and dying
with a jump in hand — both are what the policy decides at rare offstage
states, i.e. the learning-signal problem of the 10-05 verdict. Keep
`--stick-duration 8` + edge decisions as a candidate for the port recipe
on coherence grounds (needs the ≥ 3-epoch replication the recipe got);
the recovery lever is now lever 3 — attack the shortcut generally.

**Replication (queue 21).** `dur8e_s906` (seed 906, 1 ep): val 3.22,
fidelity **0.202**, repeat **0.776** / neutral **0.235** — the
coherence/fidelity pass holds on a second seed (off3's seeds gave
0.21–0.23 / 0.73 / 0.25). Recovery again unmoved: return 0.271, mismatch
0.171, decided-trip return 0.366, deep-B stick-up 6/37, died with a jump
left 26/142. Offstage map: deep j0 0.028, high j0 0.021, low j0 0.024
(the dur8 gains persist on this seed too; age 1–3 0.044). `dur8e_e3`
(3 ep) pending.

Note on the Q7 teacher-forced floor for duration arms (dur8 0.078, dur16
0.097 in the −20..−60 j0 band vs off3's 0.05): Q7 scores the per-frame
head on every frame, but a duration checkpoint is only trained at
decision frames, so its mid-hold outputs are unconstrained. Q7 is not a
valid readout for these arms; the closed-loop map is.

## 10-07 — `dur8e` at 3 epochs: the port-recipe candidate passes its bar

Queue 21 (seed 905, 3 ep; the s906 1-ep replication is in "10-06").
Overnight was lost to a mid-edit nx tree (HANDOFF §8d); scored 12:17.

| arm | ep | val | fidelity | repeat | neutral | return | decided | mismatch | off age 1–3 enter-sil. | deep-B stick-up | dies w/ jump |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expert | — | — | 0 | 0.764 | 0.303 | 0.911 | 0.915 | 0.084 | 0.013 | 81 % | 7 % |
| off3 (recipe) | 1 | 3.6x | 0.198 | 0.73 | 0.19 | 0.37–0.39 | 0.49 | 0.084–0.088 | 0.047 | 17–35 % | 26–30 % |
| dur8e | 1 | 3.25 | 0.191 | 0.777 | 0.296 | 0.23 | — | 0.168 | 0.059 | 17 % | 30 % |
| dur8e s906 | 1 | 3.22 | 0.202 | 0.776 | 0.235 | 0.27 | — | — | — | — | — |
| **dur8e e3** | **3** | **3.064** | **0.175** | 0.759 | 0.221 | 0.338 | 0.394 | 0.112 | 0.034 | 31 % | 27 % |

(val for duration arms is the decision-frame likelihood, comparable only
among duration arms.) Read against the queue-21 header: fidelity ≤ 0.21
✓, repeat 0.70–0.80 ✓, neutral 0.22–0.33 ✓ (at the edge), on both seeds →
**`--stick-duration 8` + ≥ 3 ep is the port-recipe candidate.** Recovery
moved the right way with epochs (return 0.23 → 0.34, age 1–3 offstage
hazard 0.059 → 0.034, mismatch 0.168 → 0.112) without reaching off3, and
the two faces of the death shape (`scripts/recovery_death_shape.js`)
barely moved — the recovery *decision* is still not being learned by
coherence or epochs; lever 3 continues in queues 22/23 (full-scale v1/v2
scorecard, pd15, window 240, 9k files).

### 10-07 12:50 — the full-scale scorecard names the shortcut

Queue 22 scored the two full-scale Mamba checkpoints (512×2, ALL Fox files,
1 ep, windowed 80) on the recovery scorecard + silence map + fidelity
(`EVALS="coherence fidelity recovery_means recovery_probe"`, PROBE_BATCH 64):

| | `fox_mamba_v1` (no prev-action) | `fox_mamba_v2` (prev-action, dropout 0.15) | expert | testbed off3 |
|---|---|---|---|---|
| return / decided | **0.518 / 0.558** | 0.202 / 0.241 | 0.911 / 0.915 | 0.37–0.39 / 0.49 |
| died with a jump left | **11/129 = 9 %** | 94/388 = 24 % | 7 % | 26–30 % |
| deep-B presses stick-up | **100/212 = 47 %** | 34/198 = 17 % | 81 % | 17–35 % |
| input changes per died trip (q50) | 23 | 13 | 17 | 8–11 |
| carried-off died | 0.756 | 0.989 | 0.163 | — |
| offstage age 1–3 enter-silence | 0.109 (×8.4) | 0.058 (×4.5) | 0.013 | 0.047 |
| input repeat / fidelity | 0.384 / 0.298 | 0.725 / 0.176 | 0.758 / 0 | 0.73 / 0.198 |
| sd/min, deaths/min | 1.40, 1.67 | 3.88, 4.34 | 0.44, 0.96 | — |

Same data, same width, one flag apart. The model **without** the
prev-action channel learns the recovery *decision* — it spends its jump
like the expert (9 % vs 7 %) and aims B up half the time — and recovers
0.52; the model **with** the channel is coherent (repeat 0.725) and
recovers 0.20, dying with a jump in hand a quarter of the time, exactly
like every testbed arm (all of which carry `--prev-action`). Against the
queue-22 header (≥ 0.6 ⇒ data/scale, ≤ 0.45 ⇒ not) v1 sits between; the
v1/v2 split is the finding: **scale and data do not help while the copy
path exists**. v1's own defect is the one the whole 10-01 program fixed —
no coherence (enter-silence onstage ×4.9 means it never holds anything,
which is also *why* it never silently falls).

So the recovery lever and the coherence lever were the same channel
pulling opposite ways, and the duration head is the first coherence
mechanism that is not the channel. **Queue 24** (`coherence_queue24.sh`,
after 23): dur8e WITHOUT `--prev-action` (`dur8e_nopq`) and with dropout
0.5 (`pd50`); pd15 (queue 22) is the first point. Pass: return ≥ 0.45 or
dies-with-jump ≤ 15 % with the coherence band kept. If nopq holds the
band, the prev-action channel leaves the port recipe.

### 10-07 13:52 — `dur8e_pd15` read (queue 22): breaks the silent fall, not a pass

`--prev-action-dropout 0.15` on the dur8e candidate (1 ep, seed 905):
fidelity 0.213, repeat 0.782, **neutral 0.141** (band 0.22–0.33), return
0.313, decided 0.388, mismatch 0.144; offstage age 1–3 enter-silence
**0.031** (dur8e 0.059, expert 0.013); death shape: died holding neutral
**10/100** (dur8e 84/196, expert 88/104 — the expert dies holding
neutral because it has already committed), dies with a jump left 25 %,
deep-B stick-up 31 %. Bradley: "how exactly does the bot die actively wrong? this could be a
separate issue." It is — `recovery_death_shape.js` now decomposes every
death (all died episodes, bot and expert):

| | expert | dur8e | pd15 | e3 |
|---|---|---|---|---|
| offstage frames, stick toward / away the stage | 55 / 6 % | 23 / 17 | **40** / 19 | 22 / 12 |
| died without ever using a special | 30 % | 59 % | 55 % | 46 % |
| double jump used before death | 91 % | 72 % | 72 % | 71 % |
| ended in an aerial attack / passive Fall | 5 / 6 % | 14 / 42 | **21 / 24** | 16 / 38 |
| Firefox aimed toward / away the stage | 80 / 1 % | 31 / 36 | **63 / 26** | 32 / 39 |
| Firefox started deep (y<−40) / high (y>−10) | 44 / 26 % | 17 / 69 | **37 / 41** | 11 / 68 |

Three separable defects. **(1) The copy-shortcut family** — silent holds,
stick held away, Firefox aimed at the blastzone and fired early — moves
with dropout (pd15) and NOT with epochs (e3): that is what the
prev-action channel was holding wrong. **(2) "Never commits to a recovery
special"** (55 % vs 30 %): untouched; pd15 trades passive falling for
attacking offstage (aerials 14 → 21 %, expert 5 %) — a decision defect
the data should teach (teacher-forced, an aerial and an up-B score the
same), plausibly credit rather than copying. **(3) The double jump**
(72 / 72 / 71 % vs 91 %): no arm of any kind has moved it — its own
mechanism, unexplained. pd15 is the first point on the dropout curve;
pd50 and nopq (queue 24) complete it; (2) and (3) need their own levers.

### 10-07 15:30 — the double jump: a height-conditioned decision the bot does not condition

Bradley: "look into why the double jump never gets used." From the sim
traces (all died episodes), per-3-frame hazard of a jump (X/Y edge or
jumps_left drop) while offstage, falling, jump in hand, by height:

| height | expert | off3 | dur8e | pd15 | e3 | v2 (full, prev-act) | v1 (full, no prev-act) |
|---|---|---|---|---|---|---|---|
| 0 … −20 | 15 % | 9 | 6 | 7 | 10 | 10 | 13 |
| −20 … −40 | **44 %** | 23 | 11 | 18 | 26 | 24 | 18 |
| −40 … −60 | **60 %** | 26 | 9 | 25 | 26 | 10 | 16 |
| < −60 | 32 % | 14 | 16 | 29 | 12 | 7 | 16 |
| frames spent below −60 | 69 | 150 | 358 | 70 | 214 | 553 | 31 |

The expert's jump is a **height decision**: hazard rises 15 → 44 → 60 %
as it gets low, so it rarely gets deep (69 frames). Every bot's hazard is
**flat in height** (6–26 % at every band), so it drifts through the band
where the expert spends the jump and dies with it in hand (28 % of deaths
vs 10 %). Not an order defect (special-before-jump is a minority), not
helplessness (those deaths end in plain Fall or mid-aerial, 43/60 with
≥ 30 offstage frames above −60). Same signature as the enter-silence map:
a decision under-conditioned on state. The full-scale no-prev-action
model is the exception on the outcome (jump in hand at death 11 % vs
v2's 24 %) even though its hazard curve is also flattish — it jumps early
and never gets deep — so the jump is most likely in the copy-shortcut
family as well and dropout 0.15 did not free it. **Prediction for queue
24: `nopq` moves jump-in-hand toward 10 %; `pd50` partway.**

Instrument: probe **Q8** (`interp_recovery_probe.exs`, teacher-forced
P(X∨Y press) on expert offstage airborne frames with a jump in hand, by
height, split by expert press / no press) tells whether the head learned
the height conditioning at all (rising column) or learned it and loses
it closed-loop (flat column). Lands with w240 / mf9k / nopq / pd50.

### 10-07 16:00 — new eval: `ExPhil.Eval.DecisionMap` (Bradley: "should this mean a new eval?")

Yes, as the SilenceMap's sibling rather than a fourth script. The map
asks "what does the player decide offstage, by state" for every recovery
decision at once: onset hazard per eligible frame (free-falling offstage,
actionable, not in a special) of **jump**, **special_up / side / down /
neutral** (special onset by the stick zone it was started with —
character-agnostic), **airdodge**, **aerial**, bucketed by height band
(`y>0`, `0..-20`, `-20..-40`, `-40..-60`, `<-60`) × jumps left; model vs
expert with binomial z; and the **height slope** of a decision (hazard at
−40..−60 over 0..−20 with a jump in hand) as the single conditioning
readout — expert jump ≈ 4, a flat head ≈ 1. `fidelity_scorecard.exs`
writes `decision_map.json` beside `silence_map.json` and prints the
slope line + per-decision hazard-by-height lines; the expert map comes
from `expert_reference.exs --decision-map-out`. Pass criteria to carry
into queue headers: **jump slope ≥ 2**; up-B start deep share ≥ 30 % and
aimed toward ≥ 60 % (from the death decomposition until the map has an
aim lane); offstage aerial hazard ≤ 2× expert. The death-shape
decomposition becomes a derived view of this table. Written and
parse-checked; compile + tests + expert map at the first `mix` gap (the
scorecard guards on `Code.ensure_loaded?` so the running queue is safe).

Calibration note (17:10, from the 3-frame trace version of the table,
`scripts/recovery_jump_hazard.js`, all offstage episodes): the expert's
slope is **8.7** (6 → 39 → 52 %), the arms' 2.2–5.3 — so "slope ≥ 2" is
too loose a bar; the ledge-band denominator is tiny for everyone. Carry
the **absolute mid-depth hazard** instead: jump hazard at −20..−40 and
−40..−60 ≥ 30 % (expert 39/52, every arm 9–24). Recalibrate once the
compiled per-frame map has the expert reference.

**Compiled + tested 18:30** (12 tests, DecisionMap + SilenceMap) in the
beam gap after queue 23; expert map written
(`eval_runs/1002_fidelity/expert_decision_map_fd.json`, 150 FD Fox games,
`logs/expert_decision_map_1007.log`). The expert's recovery *program*,
per-frame onset hazard by band (`y>0 / 0..-20 / -20..-40 / -40..-60 / <-60`):

| decision | jump in hand | jump spent |
|---|---|---|
| jump | 0.065 / 0.082 / **0.202 / 0.297** / 0.299 (slope 3.63) | — |
| special_up (Firefox) | 0.000 / 0.002 / 0.001 / 0.001 / 0.003 | 0.009 / 0.020 / 0.021 / **0.056 / 0.056** |
| special_down (shine) | 0.006 / 0.021 / 0.039 / 0.031 / 0.005 | ~0 |
| special_side (Illusion) | 0.002 / 0.005 / 0.003 / 0.002 / 0 | 0.007 / 0.016 / 0.003 / 0.003 / 0.003 |
| aerial / airdodge | ≤ 0.004 everywhere | ≤ 0.011 / ≤ 0.006 |

Two things the bots never show: the expert **does not Firefox with a
jump in hand** (≤ 0.3 % per frame at every height) — the jump comes
first, conditioned on height, and *then* Firefox, conditioned on depth
(2 % → 5.6 % per frame once the jump is gone); and shine offstage with a
jump in hand is its stall/positioning tool (2–4 % per frame in the
−20..−60 bands), not a mistake. So the "never commits to a special"
defect reads as the second stage of the same program: the bot does not
reach the "jump spent, now up-B" state in the right place. Bars for the
queue headers, per-frame map: jump hazard at −20..−40 ≥ 0.15 and
−40..−60 ≥ 0.20 (jump in hand); up-B hazard at −40..−60 ≥ 0.04 (jump
spent); up-B with a jump in hand ≤ 0.01.

### 10-07 17:10 — `w240` read (queue 23): context is not the lever; a trade

Window 240 on the dur8e recipe, 1 ep, seed 905, vs its header (return
≥ 0.37 with repeat 0.70–0.80 / neutral 0.22–0.33 / fidelity ≤ 0.21;
deep-B stick-up > 35 %, dies-with-jump < 20 % = the decision moved):

| | dur8e (w80) | dur8e_e3 | **w240** | header |
|---|---|---|---|---|
| val | 3.09 | 3.064 | 3.252 | — |
| fidelity | 0.191 | 0.175 | **0.219** | ≤ 0.21 ✗ |
| repeat / neutral | 0.777 / 0.296 | 0.759 / 0.221 | 0.755 / **0.196** | band ✗ (neutral) |
| return | 0.23 | 0.338 | 0.34 | ≥ 0.37 ✗ |
| offstage age 1–3 enter-silence | ~0.03 | — | 0.031 (×2.4) | — |
| deep B onsets stick-up | 17–35 % | — | **4/60 = 7 %** | > 35 % ✗ |
| died with a jump left | 26–30 % | — | **21/129 = 16 %** | < 20 % ✓ |
| never special / passive Fall | — | — | 48 % / 27 % | expert 30 / 6 |
| Firefox aimed toward / started high | — | — | 23 % / 86 % | expert 80 / 26 |
| jump hazard −20..−40 / −40..−60 | 11 / 9 % | 18 / 21 % | 14 / 17 % | expert 39 / 52 |

Return 0.34 is above dur8e's 1-ep 0.23 and level with e3, but it comes
with the coherence band broken (neutral 0.196 — the longer window
makes the bot *busier*, not more expert-like; fidelity 0.219) and the
recovery decisions unchanged: deep B is now **stick-down 54 / side 35 /
up 10** of 118 onsets (shine and Illusion offstage, the worst
stick-up share of any arm), Firefox starts high and aims away half the
time. The one header number that moved — dies with a jump left 16 % —
moved for the wrong reason: the bot spends the jump early (hazard flat
at 14–20 % down to < −60, slope 2.4) and still dies. **Q8** (teacher
forced): model P(press) on expert-press frames 0.016 → 0.026 → 0.042 →
0.053 by height vs expert hazard 0.009 → 0.010 → 0.084 → 0.087 — a
rising column, about 1/2 of the expert's rise, so the head *learns a
weak version* of the height conditioning and the loop flattens it
further (same shape as the enter-silence hazard). Verdict: **w240 is a
trade, not a win**; the recovery is not window-limited (an 80-frame
window already covers the decision band). Not a port-recipe change.

### 10-07 17:30 — `mf9k` read (queue 23): 3× data makes the silent fall *worse*

MAX_FILES=9000 on the dur8e recipe, 1 ep, seed 905 — the first point of
the testbed data-scaling curve. Caveat first: the held-out slice and the
expert reference band move with the file set (expert neutral here 0.371
vs 0.303 on 3000 files; val 3.030 is on a different held-out), so the
coherence numbers are read against *their own* expert column, not the
w80 table.

| | dur8e (3000) | **mf9k (9000)** | expert |
|---|---|---|---|
| fidelity | 0.191 | 0.218 | 0 |
| repeat / neutral | 0.777 / 0.296 | 0.78 / 0.365 | 0.711 / 0.371 |
| return | 0.23 | **0.258** | 0.911 |
| offstage age 1–3 enter-silence | ~0.03 (×2.4) | **0.078 (×6.0)** | 0.013 |
| died holding neutral+no buttons | ~50 % | **72/95 = 76 %** | 85 % (but *after* a full recovery attempt) |
| died with a jump left | 26–30 % | **40/95 = 42 %** | 7 % |
| never special / passive Fall | — | **53 % / 51 %** | 30 / 6 |
| jump hazard −20..−40 / −40..−60 | 11 / 9 % | **5 / 13 %** | 39 / 52 |
| input changes per died trip (median) | 9 | **6** | 17 |

Everything in the copy-shortcut family moved the **wrong** way with 3×
the data at the same recipe: the bot enters silence offstage six times
as often as the expert on the first frames of a hold, half of its
deaths are a passive Fall with nothing pressed, and the double jump is
left in hand on 42 % of deaths. The decided-trip return is the lowest
of any dur8e arm. Teacher-forced the head is *better* (Q8 press prob on
expert-press frames 0.035 → 0.046 → 0.075 at low; val lower), which is
the by-now familiar split: more imitation data sharpens the
prev-action copy and the loop pays for it. Read next to full-scale v2
(all files, prev-action, return 0.202): **the data-scaling curve on
this recipe slopes down for recovery**. That is the strongest evidence
yet that the port should not carry `--prev-action` as-is (lever 3), and
it makes queue 24's `nopq` the deciding arm. Not a port-recipe change.

### 10-07 18:45 — queue 24's `nopq` was not a valid arm; queue 25 `pd100`

`nopq` (dur8e recipe without `--prev-action`) died at config time:
`button_events / stick_events require ... use_prev_action: true` — the
event heads are defined on the label sequence through the prev-action
plumbing, so the channel cannot be omitted inside the event recipe
(queue 24 went straight on to `pd50`). The equivalent arm keeps the
channel and never lets the model see it: `--prev-action-dropout 1.0`
(training zeroes it on every frame) + `EVAL_ABLATE=1` (new hook in
`coherence_experiment.sh`: every closed-loop eval passes
`--ablate-prev-action`, which zeroes the same channel live;
`recovery_means.exs` gained the flag). Queue 25 =
`evt2ctx_ck8_off3_dur8e_pd100`, chained after queue 24, read against
queue 24's header + the DecisionMap bars above. Caveat: the
teacher-forced probes feed the expert's prev input, which this model
never saw — off-distribution for pd100; its closed-loop numbers are the
read.

### 10-07 19:40 — `pd50` read (queue 24): over-corrects the silence, first DecisionMap read

`--prev-action-dropout 0.5` on the dur8e recipe, 1 ep, seed 905, vs the
queue-24 header (return ≥ 0.45 or dies-with-jump ≤ 15 % with repeat
≥ 0.70 / neutral 0.22–0.33 / fidelity ≤ 0.23):

| | dur8e | pd15 | **pd50** | header |
|---|---|---|---|---|
| val | 3.09 | — | **3.73** | — |
| fidelity | 0.191 | 0.213 | **0.244** | ≤ 0.23 ✗ |
| repeat / neutral | 0.777 / 0.296 | 0.782 / 0.141 | 0.729 / **0.097** | neutral band ✗ |
| return / decided | 0.23 / — | 0.313 / — | 0.311 / 0.366 | ≥ 0.45 ✗ |
| offstage age 1–3 enter-silence | ×2.4 | 0.031 | 0.044 (×3.4) | — |
| died holding neutral, nothing pressed | ~50 % | 10 % | **13 %** | expert 85 % |
| died with a jump left | 26–30 % | — | 22 % | ≤ 15 % ✗ |
| never special / passive Fall | — | 55 % | **65 % / 30 %** | expert 30 / 6 |
| trailing identical frames before death (median) | 9 | — | **3** | expert 27 |

The dropout dose-response is monotone and goes past the target: at 0.5
the bot *never stops inputting* (neutral 0.097, 3 identical frames
before death vs the expert's 27, airdodges offstage 36–46× the expert's
rate near the ledge) and the teacher-forced loss pays for the half-blind
channel (val 3.73). The copy-shortcut family is gone; the two decision
defects are untouched — and the **first DecisionMap read** puts numbers
on both (per-frame hazard, model | expert):

| | 0..−20 | −20..−40 | −40..−60 | < −60 |
|---|---|---|---|---|
| jump, jump in hand | 0.043 \| 0.082 | **0.035 \| 0.202** | **0.113 \| 0.297** | 0.100 \| 0.299 |
| Firefox, jump spent | 0 \| 0.020 | 0.006 \| 0.021 | **0.002 \| 0.056** | **0.003 \| 0.056** (n = 1,394) |
| shine, jump in hand | 0.002 \| 0.021 | 0.005 \| 0.039 | 0.003 \| 0.031 | 0.001 \| 0.005 |
| airdodge, jump in hand | **0.007 \| 0.0002** | 0.004 \| 0 | 0.003 \| 0 | 0.002 \| 0 |

Stage one (the jump) runs at 1/3–1/6 of the expert's hazard in the two
bands that matter; stage two (Firefox once the jump is spent) at
**1/20** — 1,394 frames below −60 with no jump and a 0.3 % per-frame
chance of pressing up-B. Both stages are under-conditioned in the same
direction (too rare where the expert acts), and neither moved with
dropout 0.15 → 0.5, so they are **not** in the copy-shortcut family.
The slope (2.6) is misleading here, as predicted — read the absolute
rows. `pd100` (queue 25, channel fully absent) is the clean test of
whether the prev-action channel is what suppresses them; if its rows
look like these, the lever is elsewhere (danger-readout features /
offstage-conditioned heads, doc "10-05 23:30" first pick).

### 10-07 20:30 — `pd100` read (queue 25): the channel out — prediction holds, coherence collapses

`--prev-action-dropout 1.0` + ablated evals (the channel is zero at every
training frame and live), dur8e recipe, 1 ep, seed 905:

| | dur8e | pd15 | pd50 | **pd100** | expert |
|---|---|---|---|---|---|
| val | 3.09 | — | 3.73 | 3.80 | — |
| fidelity | 0.191 | 0.213 | 0.244 | **0.349** | 0 |
| repeat (offline / closed loop) | 0.777 / — | 0.782 | 0.729 | **0.19 / 0.22** | 0.764 / 0.758 |
| neutral | 0.296 | 0.141 | 0.097 | 0.234 | 0.303 |
| return / decided | 0.23 / — | 0.313 | 0.311 / 0.366 | **0.418 / 0.515** | 0.911 / 0.915 |
| died with a jump left | 26–30 % | — | 22 % | **8/90 = 9 %** | 7 % |
| deep B onsets stick-up | 17–35 % | — | 22 % | **36/80 = 45 %** | 81 % |
| Firefox aimed toward / started high | — | — | 29 / 76 % | **59 / 77 %** | 80 / 26 % |
| passive-Fall deaths / never special | — | — | 30 / 65 % | 9 / 52 % | 6 / 30 % |
| input changes per died trip (median) | 9 | — | 12 | **19** | 17 |

**The prediction held**: jump-in-hand deaths 9 % (written: "toward
10 %"), and it is the highest testbed return (0.418; v1 full-scale
0.518). Firefox is the most expert-like of any arm (stick-up 45 %, aimed
toward 59 %). And the arm is unusable as a recipe: without the channel
the inputs flicker (repeat 0.19 — the expert holds 76 % of frames; the
duration head does not carry the buttons and did not replace the
channel for the sticks either), fidelity 0.349, enter-silence hazard
×16 because every neutral blip counts as an entry. DecisionMap rows
(per frame, model | expert):

| | y>0 | 0..−20 | −20..−40 | −40..−60 | < −60 |
|---|---|---|---|---|---|
| jump, jump in hand | **0.183 \| 0.065** | **0.133 \| 0.082** | 0.060 \| 0.202 | 0.149 \| 0.297 | 0.154 \| 0.299 |
| Firefox, jump spent | 0.001 \| 0.009 | 0.001 \| 0.020 | 0.010 \| 0.021 | **0.013 \| 0.056** | **0.011 \| 0.056** |
| shine, jump in hand | 0.005 \| 0.006 | 0.006 \| 0.021 | 0.004 \| 0.039 | 0.006 \| 0.031 | 0 \| 0.005 |

So the channel *was* suppressing both stages partly: with it gone the
jump fires (but **early and flat** — 0.18 above the stage, 0.13 at the
ledge, slope 1.1; the bot spends it at once rather than at −20..−60, the
v1 pattern exactly) and Firefox-once-spent rises **5×** (0.003 → 0.013
deep) — still 1/4 of the expert. Neither decision became
height-conditioned. Reading the dose series pd15 → pd50 → pd100 as a
whole:

1. **The copy-shortcut family** (silent holds, direction, Firefox aim)
   is cured by dropout 0.15–0.5 and over-cured at 0.5.
2. **Input coherence is carried only by the channel** on this recipe —
   remove it and repeat falls 0.78 → 0.19. The port recipe keeps
   `--prev-action` (dropout ~0.15, quantized); "the channel leaves the
   recipe" is ruled out.
3. **The two decision defects are conditioning defects**, not channel
   defects: the channel accounts for part of their suppression (5× on
   Firefox) but at no dose do the jump or the up-B track height. The
   lever is a head that sees danger: height / jumps-left / distance into
   the decision logits (danger-readout features, doc "10-05 23:30" first
   pick), or an offstage-conditioned mixture for the hold/change
   decision. Pass bar for that arm = the DecisionMap rows (jump ≥ 0.15 /
   0.20 at −20..−60 with a jump in hand; up-B ≥ 0.04 deep when spent)
   **with repeat ≥ 0.70 kept**.

Queue 24/25 headers: no arm passed. `dur8e` + ≥ 3 ep stays the
port-recipe candidate; the recovery stays lever 3 with a named target.

### 10-08 01:30 — overnight: dose bracket, the candidate at 3 ep, and the danger-readout head

Bradley: "let's continue overnight." Three queues chained on the testbed:

- **Queue 26** (no code): `pd30` (dropout 0.3, 1 ep — is there a dose that
  keeps neutral in band with died-holding-neutral ≤ 20 %?) and
  **`pd15_e3`** (the port-recipe candidate `dur8e` + the silent-fall cure at
  3 ep; pass = port bar + offstage age 1–3 enter-silence ≤ 0.035 + died
  holding neutral ≤ 20 % → pd15 joins the recipe). DecisionMap rows are
  the control here (expected flat).
- **Queue 27** (code, compiled + tested at the gap): **`--danger-context`**
  — lever 1 of "10-05 23:30", built now that "10-07 20:30" has shown the
  two recovery decisions are conditioning defects. The AR heads get the
  current frame's own **y, jumps left, on_ground, speed_y, ledge distance**
  (`Player.danger_columns/1`, sliced in-graph from the same
  `state_sequence` input the trunk reads — Axon coalesces same-named
  inputs, so no new input, no loss/probe/export plumbing) through a
  32-wide ReLU readout whose output layer is zero-initialised (starts as
  the plain head). Live: `Sampling` mirrors it (`ar_danger_context`) and
  the Agent hands the newest embedded frame's columns as `:danger`; the
  checkpoint carries `danger_columns`. Windowed AR path only. Arm
  `pd15_dng` (1 ep), then `pd15_dng_e3` if a decision row moves ≥ 1.5×.
  **Pass** (per-frame DecisionMap): jump hazard with a jump in hand ≥ 0.15
  at −20..−40 and ≥ 0.20 at −40..−60 (pd50 0.035 / 0.113); Firefox once
  spent ≥ 0.04 below −40 (pd50 0.002); enter-silence offstage age 1–3
  ≤ 0.035; with the coherence band kept. Rows move without the band = the
  lever is real and needs the coherence fix alongside; no move = the heads
  were not the bottleneck either → the training signal (expert-labeled
  DAgger relabel, lever 2, needs Bradley's ok).

### 10-08 03:30 — queue 26 read: no dropout dose keeps neutral; `pd15` does NOT join the recipe

Both arms against the header (`logs/exphil-queue26.log`;
`eval_runs/1001_queue/evt2ctx_ck8_off3_dur8e_{pd30,pd15_e3}`):

| arm | val | repeat | neutral | fidelity | offstage a1–3 enter-silence | died holding neutral | died w/ jump left | return | decided return |
|---|---|---|---|---|---|---|---|---|---|
| expert | — | 0.764 | 0.303 | 0 | 0.013 | 88/104 (85 %) | 7 % | 0.911 | 0.915 |
| `dur8e_e3` (no dropout, the bar-passer) | — | 0.759 | **0.221** | **0.175** | **0.0338** | 49 % | 27 % | 0.338 | — |
| `pd15` (1 ep) | — | 0.782 | 0.141 | 0.213 | 0.0313 | **10 %** | 25 % | 0.313 | — |
| **`pd30`** (1 ep) | 3.78 | 0.770 | 0.118 | 0.229 | 0.0359 | 20 % | 23 % | 0.319 | 0.359 |
| **`pd15_e3`** | 3.34 | 0.734 | 0.156 | 0.195 | 0.0363 | 37 % | 17 % | **0.395** | **0.496** |

- **`pd30` fails** (neutral 0.118, fidelity 0.229). The dose series is now
  monotone on neutral: 0.221 (none) → 0.141 (0.15) → 0.118 (0.3) → 0.097
  (0.5) → there is **no dropout dose that keeps neutral in band**; the
  channel's "stand still" prior and its copy shortcut are the same
  prior, and dropout dilutes both together.
- **`pd15_e3` fails the port bar** on neutral (0.156 vs ≥ 0.22) and fails
  the silent-fall criterion it was supposed to keep: died holding neutral
  **37 %** (pd15 at 1 ep: 10 %; dur8e_e3: 49 %; bar ≤ 20 %) and offstage
  enter-silence 0.0363 (bar ≤ 0.035; dur8e_e3 without dropout already
  0.0338). **The 1-ep silence cure does not survive two more epochs**: with
  more training the policy re-learns the hold prior through the 85 % of
  the channel it still sees. Fidelity 0.195 is inside the bar but behind
  dur8e_e3's 0.175.
- What `pd15_e3` does show, and the reason it is worth keeping as a
  reference arm: **best in-band testbed return so far** (0.395; decided
  trips 0.496 — pd100's 0.418 came with repeat 0.19), died-with-a-jump-left
  down to 17 % (nearest the expert's 7 % of any coherent arm), and the
  **first non-floor Firefox-once-spent row**: special_up at `-40..-60:j0`
  0.0205, `<-60:j0` 0.0159 (pd50 0.002–0.003, pd30 0.004, expert 0.056) —
  still 1/3 of the expert but a 5–10× move, and it came from EPOCHS, not
  dose. Jump-in-hand rows: 0.124 / 0.156 in the mid bands (pd50 0.035 /
  0.113; bar 0.15 / 0.20), slope 2.54 (expert 3.63) — the jump decision
  moved toward height-tracking at 3 ep too. New failure mode on the death
  table: side-B deaths 27 of 114 (pd15: ~0; the bot now commits to a
  special and picks the wrong one from high, "started high 70 %").
- **Verdict:** the port recipe stays **`dur8e` ≥ 3 ep without prev-action
  dropout**; dropout is a diagnostic (it names the copy-shortcut family)
  not a recipe term. The two decision rows move with epochs and not with
  dose, which is consistent with "10-07 20:30": they are conditioning /
  signal-limited. Queue 27's `pd15_dng` was launched with pd15 in it (set
  before this read); it is read against `pd15` 1-ep as its control (same
  dose, same epochs), and if the danger head moves a row the 3-ep
  follow-up carries dropout too — a `dur8e_dng_e3` without dropout is the
  arm to add after that, since the recipe no longer contains pd15.

### 10-08 04:40 — `pd15_dng` read (queue 27): the danger head is LIVE and the decision rows do not move

Queue 27 compiled and passed the four targeted test files (17 tests, 0
failures — `danger_context_test` included) before the arm. `pd15_dng` vs its
same-dose, same-epoch control `pd15` (`logs/exphil-queue27.log`):

| | `pd15` | `pd15_dng` | expert |
|---|---|---|---|
| val | — | 3.54 | — |
| repeat / neutral / fidelity | 0.782 / 0.141 / 0.213 | 0.775 / 0.146 / 0.227 | 0.764 / 0.303 / 0 |
| offstage a1–3 enter-silence | 0.0313 | **0.0267** | 0.013 |
| DecisionMap jump, jump in hand, −20..−40 / −40..−60 / <−60 | (trace 13 % / 14 % / 15 %) | 0.090 / 0.084 / 0.048 (trace 13 / 14 / 7 %) | 0.202 / 0.297 / 0.299 |
| height slope (jump) | 2.8 (trace) | 1.51 | 3.63 |
| Firefox once spent, −40..−60 / <−60 | — | 0.0022 / 0.0025 | 0.056 / 0.056 |
| died w/ jump left / holding neutral / passive Fall | 25 % / 10 % / 24 % | 31 % / 24 % / 42 % | 7 % / 85 % / 6 % |
| return / decided | 0.313 / — | 0.335 / 0.39 | 0.911 / 0.915 |

- **No decision row moved** (the queue's own check said "no"; the 3-ep
  follow-up with dropout did not run — queue 28 is the no-dropout 3-ep arm
  instead). The jump rows are *flatter* than pd15's (slope 1.5, deep band
  halved), Firefox-once-spent is at the pd50 floor, jump-in-hand deaths went
  up. The one gain is a smaller offstage enter-silence at age 1–3 (0.027, the
  best of any arm with the channel) — the readout does feed the hold/change
  decision a little.
- **The head is not dead.** Reading the checkpoint directly
  (`scratchpad/danger_norms.exs`, `Edifice.Checkpoint.load` on the ebins, no
  mix): `ar_danger_embed.kernel` mean |w| 0.044 after one epoch from zero
  (the prev-stick embeds sit at 0.08–0.12, c_y at 0.039), and the hidden
  layer's weight mass per input column is **jumps 14.6**, speed_y 6.7,
  on_ground 6.3, ledge 5.8, y 2.5 — the readout leans on jumps-left and
  barely on height. So the AR heads were handed height and jumps directly,
  used them (mostly jumps), and still did not learn "jump deeper / Firefox
  once spent". **Conditioning was not the bottleneck; the teacher-forced
  signal for those decisions is** (the "no move" branch of the 01:30 plan).
- Caveat before closing the lever: 1 ep at dropout 0.15. Queue 28
  (`dur8e_dng_e3`, the real recipe, 3 ep, control `dur8e_e3`) is the last
  word — pd15_e3 showed the Firefox-once-spent row moves with epochs
  (0.021), so the e3 arm can still show the head helping where the 1-ep arm
  could not. If `dur8e_dng_e3` leaves the rows where `dur8e_e3` has them,
  `--danger-context` stays in the tree as an off-by-default probe (Bradley's
  10-06 read: "a probe not a fix") and the lever is the training signal:
  expert-labeled relabel / DAgger on the sim (needs Bradley's ok) or
  search-distilled labels.

### 10-08 06:10 — `dur8e_dng_e3` read (queue 28): the head on the real recipe makes the silent fall WORSE — lever CLOSED

The last word on `--danger-context`: the real recipe (no dropout), 3 ep,
against the bar-passer `dur8e_e3` (`logs/exphil-queue28.log`):

| | `dur8e_e3` | `dur8e_dng_e3` | expert |
|---|---|---|---|
| val | 3.064 | 3.054 | — |
| repeat / neutral / fidelity | 0.759 / 0.221 / 0.175 | 0.728 / 0.217 / 0.193 | 0.764 / 0.303 / 0 |
| offstage a1–3 enter-silence | 0.0338 | **0.0608** (×4.7) | 0.013 |
| jump hazard, jump in hand (trace) −20..−40 / −40..−60 / <−60 | 18 / 21 / 11 % (slope 5.3) | **8 / 7 / 2 %** (slope 1.8) | 39 / 52 / 35 % |
| DecisionMap jump −20..−40 / −40..−60 / <−60 | — | 0.033 / 0.033 / 0.022 (slope 0.59) | 0.202 / 0.297 / 0.299 |
| Firefox once spent (any band) | — | ≤ 0.0015 | 0.02–0.056 |
| died holding neutral / w/ jump left / passive Fall | 49 % / 27 % / 38 % | **91 % / 40 % / 46 %** | 85 % / 7 % / 6 % |
| return / decided | 0.338 / — | **0.278** / 0.351 | 0.911 / 0.915 |

- Teacher-forced numbers are unchanged (val within 0.01, fidelity inside
  the bar, neutral in band) and **every closed-loop recovery row is worse**:
  the jump decision is the flattest of any coherent arm (0.033 at every
  depth; `pd50` had 0.113 at −40..−60), Firefox-once-spent is at the floor,
  and 91 % of deaths end holding neutral with no buttons — the silent fall,
  stronger than before the whole program started.
- The head is live and larger than at 1 ep (`ar_danger_embed` mean |w|
  0.070; hidden mass jumps 11.1, speed_y 6.7, on_ground 5.8, ledge 4.6,
  y 1.5). It learned something: **given the danger features, the
  likelihood-optimal lesson in this corpus is "offstage and falling → hold
  what you are holding"**, because the expert's recovery is 97 % holds and
  the onsets are a few hundred frames per game. The head sharpened the hold
  prior in exactly the states where we wanted a decision — val loss went
  down by the same margin the closed loop went up.
- **Verdict: lever 1 ("a head that sees danger") CLOSED, and closed in the
  informative direction.** Information was never the bottleneck (pd15_dng);
  with the information made easy the BC objective uses it *against* the
  decision (dur8e_dng_e3). This is the mechanism behind every earlier
  negative too (offstage weight ×8, silent-fall loss weight, 3× data,
  epochs): the teacher-forced likelihood of the expert's offstage frames is
  dominated by holds, and every lever that optimises that likelihood harder
  learns the hold harder. `--danger-context` stays in the tree
  off-by-default as a probe (as Bradley read it on 10-06).
- **What is left is the training signal itself**, and the three candidates
  all change what the label is, not how it is read: (a) **expert-labeled
  relabel / DAgger on the parity sim** — the bot's own offstage states with
  the expert's (or a search's) decision as the label, so the onset frames
  are over-represented where the bot actually is (needs Bradley's ok; the
  parity branch is 88.5 % bit-exact for exactly this); (b) **search-
  distilled labels** — a Firefox/jump that makes it back under the sim is a
  label even where the expert held; (c) **onset-weighted loss on the
  decision heads only** (weight the frames where jumps_left or the
  special_up button changes, not the whole offstage slice — offstage weight
  ×3/×8 weighted the holds too, which is why it moved the Q7 floor and
  nothing else). (c) is the only one that needs no sim and no ok; it is a
  one-flag build (`--onset-weight W` on the frames where the expert's
  jumps-left drops or B+up starts, within the offstage slice) and the
  DecisionMap rows are the pass bar. (a) and (b) wait for Bradley; (c) is
  built and running — next section.

### 10-08 06:50 — `--onset-weight` built; queue 29 running (`on10`, `on30`, better W at 3 ep)

`SilentFallWeighting.onset?/2` + `onset_weight` (commit 012aabfd; parser /
config / flag docs / TRAINING.md; 7 weighting tests + config tests pass at
the gap): an offstage, falling (not ledge / helpless / hitstun) frame whose
label presses a jump button (X/Y) or B that the previous frame did not
gets `max(w, W)`; the holds keep the recipe's offstage ×3. Measured on 20
training replays (`scratchpad/onset_check.exs`, mix-free): **9.5 onsets per
game, 1.8 % of offstage frames** — so at W=10 the decision frames carry ~6 %
of the offstage loss mass (from 1.8 %), at W=30 ~15 %. That is the dose
range; if `on30` moves the rows and keeps the band, 100 is the next rung,
if it leaks (onstage jump hazard, fidelity) 10 is the ceiling.

Queue 29 (`scripts/coherence_queue29.sh`, started 06:05): `on10`, `on30` at
1 ep on the recipe, then the better W (by the −40..−60 jump-in-hand row,
fidelity ≤ 0.25 required) at 3 ep vs `dur8e_e3`. Pass as in the 06:10
section: jump ≥ 0.15 / 0.20 in the mid bands, Firefox once spent ≥ 0.04
deep, band kept, onstage jump hazard (y>0, jump in hand) ≤ 0.15.

### 10-08 08:00 — `on10` / `on30` read (queue 29, 1 ep): **the jump rows MOVE for the first time**; Firefox does not

| | `dur8e` (1 ep) | `on10` | `on30` | expert |
|---|---|---|---|---|
| val | — | 3.253 | 3.273 | — |
| repeat / neutral / fidelity | — | 0.782 / 0.257 / 0.210 | 0.781 / **0.222** / 0.223 | 0.764 / 0.303 / 0 |
| DecisionMap jump, jump in hand: y>0 / 0..−20 / −20..−40 / −40..−60 / <−60 | — | 0.053 / 0.066 / 0.065 / 0.113 / 0.147 | **0.102 / 0.129 / 0.163 / 0.207 / 0.351** | 0.065 / 0.082 / 0.202 / 0.297 / 0.299 |
| jump trace (−20..−40 / −40..−60 / <−60) | 11 / 9 / 13 % | 12 / 14 / 18 % | **21 / 21 / 23 %** | 39 / 52 / 35 % |
| Firefox once spent (−40..−60 / <−60) | — | 0.006 / 0.003 | 0.000 / 0.0005 | 0.056 / 0.056 |
| offstage a1–3 enter-silence | — | 0.049 | 0.042 | 0.013 |
| died holding neutral / w/ jump left | — | 41 % / 20 % | 26 % / **12 %** | 85 % / 7 % |
| recovery mismatch (expert split-half 0.084) | — | 0.162 | **0.053** | — |
| return / decided | — | 0.333 / 0.407 | **0.358 / 0.48** | 0.911 / 0.915 |
| deaths by move | — | side_b 24, attack 17 | **side_b 38**, attack 29 | attack 78, up_b 11, side_b 5 |

- **`on30` passes the jump half of the bar at 1 ep** (0.163 ≥ 0.15 at
  −20..−40, 0.207 ≥ 0.20 at −40..−60; below −60 it is above the expert) with
  the band kept (repeat 0.781, neutral 0.222 — in band for the first time
  on a 1-ep arm — fidelity 0.223, onstage y>0 jump 0.102 ≤ 0.15). Deaths
  with a jump in hand fall to 12 % (expert 7 %) and the recovery-means
  mismatch is 0.053, *below* the expert's own split-half floor. Height
  slope is still shallow (1.6 vs 3.63: it jumps early too — 0..−20 at ×1.57)
  — a dose/epochs matter; the queue picked W=30 and `on30_e3` is running
  from 07:57.
- **Firefox once the jump is spent did not move at all** (0.000 / 0.0005),
  and side-B is now the dominant death (38 of 156): the B-edge weight
  teaches "press B offstage", and with the semi-Markov stick already held
  toward the stage that press is an Illusion into the wall. The expert's
  Firefox is two onsets — the stick goes UP a few frames before B — and
  `--stick-duration` supervises the stick only at its change events, so
  weighting the B frame never weights the stick-up change that aims it.
  Fix (not yet applied — lib files are frozen while queue 29's eval
  stages still invoke `mix run`): extend `onset?` to (i) the main stick
  entering UP (y ≥ 0.75) from not-up while offstage and the jump is spent,
  and (ii) count a B edge only with the stick up (a Firefox onset); a
  side-B press stays at the offstage weight. Arm `on30u` (W=30 with the
  extended onset) after queue 29 — pass = the Firefox-once-spent row
  ≥ 0.04 deep with the jump rows kept where `on30` has them and side-B
  deaths back under 15.

### 10-08 09:30 — `on30_e3` read (queue 29): **passes the port bar AND the jump rows — new recipe candidate**

| | `dur8e_e3` (old candidate) | **`on30_e3`** | expert |
|---|---|---|---|
| val | 3.064 | 3.110 | — |
| repeat / neutral / **fidelity** | 0.759 / 0.221 / 0.175 | 0.780 / **0.304** / **0.162** | 0.764 / 0.303 / 0 |
| DecisionMap jump, jump in hand: y>0 / 0..−20 / −20..−40 / −40..−60 / <−60 | — | 0.097 / 0.070 / **0.176 / 0.282 / 0.258** | 0.065 / 0.082 / 0.202 / 0.297 / 0.299 |
| height slope (jump) | — | **4.01** | 3.63 |
| jump trace (−20..−40 / −40..−60 / <−60) | 18 / 21 / 11 % | 23 / 27 / 10 % | 39 / 52 / 35 % |
| Firefox once spent (−20..−40 / −40..−60 / <−60) | — | 0.005 / 0.005 / 0.007 | 0.021 / 0.056 / 0.056 |
| offstage a1–3 enter-silence | 0.034 | 0.044 | 0.013 |
| died holding neutral / w/ jump left / passive Fall | 49 % / 27 % / 38 % | 50 % / **19 %** / 38 % | 85 % / 7 % / 6 % |
| return / decided | 0.338 / — | **0.389 / 0.481** | 0.911 / 0.915 |
| deaths by move | — | side_b 27, attack 17, airdodge 7, up_b 2 | attack 78, up_b 11, side_b 5 |

- **Port bar passed with margin**: fidelity 0.162 is the lowest of any arm
  (bar ≤ 0.21), neutral 0.304 sits on the expert's 0.303, repeat 0.78.
  Val is 0.05 *higher* than dur8e_e3 — the weight trades a little
  likelihood for behaviour, the opposite of the danger head.
- **Jump rows at 0.86–0.95× the expert in every offstage band, slope 4.0
  (expert 3.6)** — the double jump is now a height decision. Onstage leak
  stays inside the check (y>0 0.097 ≤ 0.15; 0..−20 *below* the expert, so
  the early jumping of the 1-ep arm went away with epochs).
- Still open: **Firefox once spent 0.005** (1/10 expert), side-B the top
  death (27), offstage enter-silence 0.044 (×3.4; dur8e_e3 0.034) — the
  bot now jumps like the expert and then still falls silently or
  Illusions into the wall. That is the stick-up half of the onset
  (08:00 section): applied as `onset?/3` (stick entering UP once the
  jump is spent; B edge only with the stick up; 8.1 onsets per game),
  8 weighting tests pass. **Queue 30**: `on30u` (1 ep), `on30u_e3` if the
  jump row is kept, and **`on30_e3_s906`** (seed replication of this
  candidate; the recipe claim needs it before the port).
- **Recipe candidate (pending s906 + on30u):** windowed + prev_q (no
  dropout) + events + context + chunk 8 + `--offstage-weight 3` +
  `--stick-duration 8` + **`--onset-weight 30`**, ≥ 3 ep.

### 10-08 12:00 — `on30u` / `on30u_e3` read (queue 30): the jump half is SOLVED; the Firefox half is not a label problem

| | `on30_e3` | `on30u` (1 ep) | **`on30u_e3`** | expert |
|---|---|---|---|---|
| val | 3.110 | 3.279 | 3.133 | — |
| repeat / neutral / **fidelity** | 0.780 / 0.304 / 0.162 | 0.764 / 0.240 / 0.226 | 0.755 / 0.240 / **0.156** | 0.764 / 0.303 / 0 |
| DecisionMap jump, jump in hand: y>0 / −20..−40 / −40..−60 / <−60 | 0.097 / 0.176 / 0.282 / 0.258 | 0.179 / 0.147 / 0.267 / 0.269 | **0.074 / 0.181 / 0.286 / 0.385** | 0.065 / 0.202 / 0.297 / 0.299 |
| height slope (jump) | 4.0 | 1.5 | **4.1** | 3.6 |
| jump trace (−20..−40 / −40..−60 / <−60) | 23 / 27 / 10 % | 12 / 30 / 25 % | **27 / 35 / 18 %** (slope 5.0) | 39 / 52 / 35 % |
| Firefox once spent (−20..−40 / −40..−60 / <−60) | 0.005 / 0.005 / 0.007 | 0.002 / 0.007 / 0.001 | 0.002 / 0.004 / 0.010 | 0.021 / 0.056 / 0.056 |
| died holding neutral / w/ jump left | 50 % / 19 % | 29 % / **6 %** | 41 % / 18 % | 85 % / 7 % |
| return / decided | 0.389 / 0.481 | 0.386 / 0.463 | **0.439 / 0.528** | 0.911 / 0.915 |
| deaths by move | side_b 27, attack 17, airdodge 7, up_b 2 | side_b 24, attack 23, airdodge 14, up_b 10 | attack 18, **side_b 17**, airdodge 12, up_b 4 | attack 78, up_b 11, side_b 5 |

- **`on30u_e3` is the best arm of the program on every coherence and
  recovery number but one**: fidelity 0.156 (new floor), band kept
  (repeat 0.755, neutral 0.240), jump rows 0.90–0.96× the expert in the
  mid bands and above it below −60, slope 4.1, onstage leak 0.074 (the
  1-ep arm's 0.179 was epoch noise, as with on30), **return 0.439 /
  decided 0.528** (dur8e_e3: 0.338). Side-B deaths halved from on30_e3
  (27 → 17, bar < 15 — close), up-B deaths back to 4.
- **The Firefox row did not move** (0.004 at −40..−60 — 1/15 of the
  expert; 0.010 below −60 — 1/6) even with the stick-up change weighted
  as its own onset and B counted only with the stick up. Two full
  attempts at the label side (B frame; aim frame + B-with-aim) have
  moved the jump to the expert and left the up-B where it was, so
  **the Firefox defect is not a label-weight problem**. The mechanism the
  08:00 section named is structural: inside a `--stick-duration` hold the
  button head samples B *blind* to the stick it cannot re-aim (AR order
  buttons → main stick, the same as slippi-ai's, but slippi-ai re-predicts
  the stick every frame, so it has no hold to be trapped in). The
  weighted aim frames are learned teacher-forced (the row is non-zero now
  where it was 0.000 on `on30`) but in the loop the aim and the press
  have to be chosen in the same hold, and nothing in the head couples
  them.
- **Proposal for Bradley (not built): put the stick event BEFORE the
  buttons in the AR order** on the windowed semi-Markov head — sample the
  stick change (direction + duration) first, then buttons conditioned on
  it — so "aim up now" and "press B" are one decision. It is a head
  ordering change (training + sampler + tests), not a decode rule; pass
  = the Firefox-once-spent row ≥ 0.04 deep with on30u_e3's rows and band
  kept. Alternative with the same aim: a dedicated "special" event head
  (direction × B) alongside the stick-duration head.
- **13:00 — the AR-order proposal is withdrawn; the missing half is the
  AIM, not the press.** `scripts/recovery_aim_vs_press.js` over the loop
  traces (jump spent, falling, y < −20, every 3 f):

  | | stick-up share | up-onset / 3 f | P(B \| up) | P(B \| not up) |
  |---|---|---|---|---|
  | expert | 42.5 % | 12.3 % | 14.7 % | 2.6 % |
  | `dur8e_e3` | 14.9 % | 1.9 % | 8.1 % | 3.6 % |
  | `on30_e3` | 13.4 % | 2.1 % | 4.7 % | 2.7 % |
  | `on30u_e3` | 13.5 % | 3.4 % | **26.7 %** | 2.6 % |

  `on30u` fixed the press (P(B | stick up) 4.7 → 26.7 %, above the
  expert); the bot moves the stick UP once spent at 1/4 the expert's onset
  rate and holds it up 1/3 as often (hold lengths match, ≈ 10–12 f). The
  aim is a stick-head decision and sits in the same place in either AR
  order, and with `--event-context` the button head already sees the held
  stick — so stick-before-buttons would trade the common cases (stick
  conditioned on jump/shine/aerial buttons) for a blind spot that is not
  the one we have. Bradley asked the trade-off; answered in chat. Next
  read: **Q9 `scripts/recovery_aim_probe.exs`** (queue 31, after 30) —
  teacher-forced P(stick → up) on the expert's own jump-spent, deep,
  not-yet-up frames, model vs expert: same ≈ the loop states differ
  (DAgger on the parity sim, needs Bradley's ok); lower = never learned
  (aim-onset weight up, or the aim as its own event).
- **13:20 — `on30_e3_s906` (seed replication) HOLDS:** fidelity 0.174,
  repeat 0.75, neutral 0.217 (at the band's edge), DecisionMap jump
  **0.194 / 0.302 / 0.397** (expert 0.202 / 0.297 / 0.299; onstage y>0
  0.108 ≤ 0.15, though 0..−20 at ×1.5 — slope 2.5), return 0.429 /
  decided 0.479, jump-in-hand deaths 22 %, side-B 19. Two seeds now put
  the jump rows at the expert with the bar met → **`--onset-weight 30`
  is a replicated recipe term.**
- **13:45 — Q9 aim probe (queue 31): the aim is LEARNED, exactly; the
  loop deficit is distribution shift.** `scripts/recovery_aim_probe.exs`
  on held-out expert frames, jump spent, below the ledge, previous stick
  not up. Per frame the model over-aims (P(up) 0.07–0.09 vs expert share
  0.024; 0.53–0.61 on the frames where the expert aimed vs 0.06–0.08
  elsewhere) — but under `--stick-duration` the stick head is trained and
  sampled only at **decision frames** (pair change, any button edge, every
  C-th frame of a hold; `Loss.stick_decision_targets`), and on those:

  | Q9b decision frames (n=318) | expert share up | model P(up) |
  |---|---|---|
  | `dur8e_e3` | 0.088 | 0.093 |
  | `on30u_e3` | 0.088 | **0.087** |

  Calibrated to the third decimal on the expert's states; 1/4 of the
  expert's onset rate on the bot's own (aim_vs_press: 3.4 % vs 12.3 % per
  3 f). The imitation signal for the aim is not the problem; **the states
  are** — after its own jump the bot is somewhere the expert never was
  (and/or reaches fewer decision frames there: a silent hold consults the
  stick head once per 8 frames, the expert's recovery has an edge every
  ~3). This is the textbook DAgger case: the expert's labels on the
  **bot's** offstage states. On the parity sim (88.5 % bit-exact) that is
  exactly runnable — expert-labeled relabel needs Bradley's ok (standing).
  The one in-imitation lever left for the aim is decision frequency:
  **`--stick-duration 4`** (a hold re-decides twice as often; dur4 was not
  in the 10-06 series — dur8/dur16 were) with `--onset-weight 30`, arm
  `on30u_dur4_e3`; pass = loop up-onset ≥ 7 % / 3 f with on30u_e3's rows
  and band kept. Queued as 32 with `on30u_e3_s906`.
- **14:30 — Bradley's ok: expert-labeled relabel on the bot's own offstage
  states on the parity sim** ("if we need to fix the parity sim further
  for this to work, let me know"). Built as queue 33 (chained after 32):
  - **Labeler = the expert distribution itself**, not rules
    (`scripts/lib/expert_recovery_labeler.exs`): an index of every
    offstage airborne frame of 176 training-split FD Fox games (214k
    rows, hold share 0.785, B 18.9 %, jump 11.1 %, stick up 28.4 %) as a
    mirrored, scaled feature vector — y, distance past the edge, speeds
    toward the stage, jumps left, facing, action class (special / hitstun
    / airdodge / jump), action frame, previous stick and buttons, opponent
    offset — with that frame's input and whether it was a hold. For a bot
    state + its previous label, one of the k=8 nearest rows is **sampled**:
    a hold keeps the previous label, a change becomes that expert input on
    the bot's side. Sampling (not voting) keeps the expert's hazards
    hazards; conditioning on the previous label keeps the hold share.
  - **Rollout** (`scripts/sim_recovery_dagger_expert.exs`): `on30u_e3` in
    the sim, 3 seeds × 32 envs × 3600 f self-play, every offstage airborne
    frame relabelled in order inside a trip (prev = previous label), 90
    input-only context frames per trip — the 10-05 plumbing unchanged.
  - **Gate** before any arm trains: the labeler on 24 held-out expert
    games must reproduce the expert's own hazards on the same frames
    (jump edge with a jump in hand, stick-up onset and B edge once spent)
    within ×0.5–×2 (`scripts/expert_labeler_selfcheck.exs`), and the
    set's hold share must be ~0.76–0.80.
  - **Arms**: `on30u_dagx2_e3`, `on30u_dagx4_e3` (mix oversample 2 / 4,
    interleaved, 3 ep). **Pass**: loop up-onset once spent ≥ 7 %/3 f
    (on30u_e3 3.4 %, expert 12.3 %) AND Firefox-once-spent ≥ 0.04 deep
    (on30u_e3 0.004) with the jump rows, the band and the closed-loop
    rates (SDs/min, dashes/min) kept.
- **15:20 — `dur4e_on30u_e3` (queue 32): decision frequency is NOT the
  lever.** Loop up-onset once spent **2.8 %** / 3 f (on30u_e3 3.4 %, bar
  ≥ 7 %), stick-up share 11.5 %, side-B deaths 40 (worst), jump rows
  0.141 / 0.287 / 0.184 (the <−60 row fell), fidelity 0.176, repeat 0.757,
  neutral 0.239, return 0.424 / decided 0.571 (the decided number is the
  best yet — it commits more and dies by side-B). Re-deciding the stick
  twice as often did not raise the aim: the policy is not being *asked*
  too rarely, it answers "not up" on its own states. That closes the
  in-imitation list for the aim and leaves queue 33 (expert relabel on
  the bot's states) as the test of the coverage reading. `--stick-duration
  8` stays.
- **Recipe claim now = `on30u` (extended onset) rather than `on30`**: same
  flag (`--onset-weight 30`; `onset?/3` is the shipped definition), ≥ 3
  ep. `on30_e3_s906` (running from 11:52) is the seed replication of the
  W=30 recipe at the earlier onset definition; it still certifies the
  flag's effect across seeds, and an `on30u_e3_s906` follows if it holds.
- **16:45 — `on30u_e3_s906` (queue 32, seed 906): the jump half
  replicates; the return gain does not.** Band kept (repeat 0.760,
  neutral 0.260, fidelity 0.194 ≤ 0.21; val 2.273); jump rows **0.254 /
  0.334 / 0.267** (bar ≥ 0.15 / 0.20; ×1.26 / ×1.12 / ×0.89 expert,
  slope 2.9), jump-in-hand deaths 32/141 = 23 % (seed 905: 19 %, expert
  7 %). Firefox-once-spent still 0.0 at ≤ −40, up-onset 2.4 %/3 f, P(B|up)
  9.5 %, up-B first-means 2, side-B deaths 29 — the aim half is as absent
  at seed 906 as at 905. Recovery-means return **0.323** / decided 0.411
  (seed 905: 0.439 / 0.528; dur8e_e3 0.23) → the on30u return number is
  partly seed noise; the jump rows and band are the replicated claim.
  Rates: SD/min 1.39 (expert 0.44), dashes/min 24 (expert 40), damage
  48/min (expert 133) — the baseline the DAgger arms must keep.
- **16:45 → 16:49 — queue 33 gate: `LABELER_FAILED` on the FIRST run was
  an underpowered gate, not the labeler.** The self-check read the
  testbed's validation split, which holds 16 files — ONE of them FD Fox:
  n=1,500 frames, 91 spent / 75 in-hand decision frames, 0 up-onsets and
  0 B edges sampled against 3 and 1 actual (jump 2 vs 6). Rerun on 24 FD
  Fox games OUTSIDE the testbed (`data/silent_fall/heldout_fd_fox_split.json`,
  1,068 such games; n=28,547 frames): hold share 79.4 % actual | 80.6 %
  label, 16-bucket agreement 68.0 %, **jump edge 8.76 % | 9.02 % (×1.03),
  stick-up onset 3.1 % | 2.43 % (×0.78)**, B edge 0.24 % | 0.52 % (5 vs 11
  events, ×2.2 — underpowered). The labeler reproduces the expert's
  hazards on the expert's states. Gate rewritten: held-out split, n ≥
  10,000, jump + up within ×0.5–×2, a hazard with < 20 actual events is
  reported not gated (`labeler_selfcheck.json` now carries `n_spent` /
  `n_in_hand`). Queue 33 relaunched 16:49 (first log kept as
  `logs/exphil-queue33_gate1.log`, one-game JSON as
  `labeler_selfcheck_1game.json`).
- **16:56 — second queue 33 run died of ENOSPC (root disk 100 %, 11 MB
  free; GOTCHA #116 again).** The rollout itself worked (3 seeds × ~188
  offstage runs, 71,012 relabelled frames) and the set's `File.write!`
  stopped at exactly 4 MiB with no report; `dagx2_e3` / `dagx4_e3` then
  `TRAIN_FAILED` at once. Not a labeler or sim defect. Freed 6.3 GB
  reversibly: the 82 testbed arm directories `checkpoints/coh_*` were
  MOVED to `/data/exphil/checkpoints_coh/` with symlinks left in
  `checkpoints/` (every queue path still resolves; `/` now 6.4 GB free,
  99 %). The remaining 554 GB is older work (`checkpoints/` 49 GB,
  `eval_runs/` 43 GB: `0921_evals` 8.2 GB, `0921_step8` 7.4 GB,
  `0924_mewtwo_ppo` 5.6 GB …) — **Bradley's to triage**; nothing
  deleted. Truncated set removed, log kept as
  `logs/exphil-queue33_enospc.log`, queue 33 relaunched 16:58 (gate
  passed again, rollout running).
- **18:35 — `on30u_dagx2_e3` (queue 33, set r1 ×2): the expert labels on
  the bot's states move the closed loop the right way, NOT to the aim
  bar.** Set r1 (`data/silent_fall/sim_dagger_expert_r1.frames`): 490
  offstage runs, 18,619 relabelled frames + 44,100 context, policy silent
  on 22 %, labels B 13.3 % / jump 10.5 %, hold share **0.793** (in band).
  On the bot's own states the labels carry the expert's hazards
  (`scripts/dagger_set_label_hazards.exs`): stick-up onset once spent
  **3.08 %** (expert held-out 3.1 %), jump edge in hand 6.3 % (8.8 %), B
  edge 0.56 % — but that is **71 up-onsets and 152 jump edges** in the
  whole set; at ×2 oversample the arm saw ~10 % more onset examples than
  the corpus already holds. Arm: val 2.396 (on30u_e3 2.273 — the mix
  costs likelihood), fidelity 0.170, repeat 0.739, **neutral 0.21** (band
  floor 0.22), jump rows 0.141 / 0.270 / **0.294** (−20..−40 a hair under
  the 0.15 bar; <−60 the best yet, ×0.99 expert). **Aim: up-onset once
  spent 3.9 %/3 f (3.4 % → bar 7 %), Firefox-once-spent 0.006 / 0.007 at
  ≤ −40 (0.0035 / 0.0095 → bar 0.04)** — FAIL, direction right at
  −20..−60, flat deep. P(B|up) 17.4 % (26.7 % → expert 14.7 %). Closed
  loop improved across the board: offstage return rate 0.814 (s906
  0.708), deaths/min 1.29 (1.61), SD/min 1.14 (1.39), wavedashes/min 4.18
  (1.56; expert 4.63), damage 60/min (48), jump-in-hand deaths 13/105 =
  **12 %** (on30u_e3 19 %, expert 7 %), died holding neutral 35 %, up-B
  first-means 5 (2), recovery-means return 0.439 / decided 0.569. New
  defect: `airdodge_with_jump` 0.125 (expert 0.006; airdodge deaths 13)
  — the labels teach "act" and the bot picks the wrong act sometimes.
  `dagx4_e3` (×4 dose) running from 18:33; the dose being 71 onsets is
  the reading → queue 34 = 12 rollout seeds (4× set) as DAgger round 2
  from the newest policy, oversample 4.
- **20:00 — `on30u_dagx4_e3` (queue 33, r1 ×4): the dose is MONOTONE and
  the closed loop is the best of the program; the aim bar is still far.**
  Band fully kept: repeat 0.767, neutral 0.283, fidelity 0.179. Aim:
  up-onset once spent **4.6 %**/3 f (on30u_e3 3.4 → dagx2 3.9 → dagx4
  4.6; bar 7, expert 12.3), stick-up share 21.5 % (13.5 → 14.4 → 21.5;
  expert 42.5), Firefox-once-spent **0.0085** at −40..−60 / **0.0131**
  deep (0.0035/0.0095 → 0.006/0.007 → 0.0085/0.0131; bar 0.04, expert
  0.056) — every aim number rises with dose, none near the bar. Jump
  rows 0.226 / 0.183 / 0.249 (−40..−60 under the 0.20 bar, ×0.62 —
  the mix shifts the jump height distribution; slope 2.4). Closed loop:
  deaths/min **1.01** (expert 0.96; s906 1.61), SD/min **0.92** (1.39),
  offstage return rate 0.821, **dashes/min 44** (24 → 44; expert 40),
  wavedashes 2.48, damage 55/min, recovery-means return **0.491** /
  decided **0.623** (n=122), mismatch **0.097** (expert split-half
  0.084), means_js 0.321 (floor 0.174) — all program bests. Deaths: 81
  decided, side-B 24 (top; `side_b_low` 0.2), attack 13, airdodge 6
  (`airdodge_with_jump` 0.062, halved from dagx2), jump-in-hand 17/81 =
  21 % (dagx2 12 %, on30u_e3 19 %), died holding neutral 40 %. Reading:
  the expert labels on the bot's own states are the first lever that
  moves the Firefox aim at all, and they do it in proportion to how many
  onset examples they carry; the closed-loop gains (dashes, SDs, deaths)
  come free. **Queue 34 launched 20:02**: DAgger round 2 — `dagx4_e3`
  rolled out on 12 seeds (`--label-seed 8`, set r2 ≈ 4× r1), r1 + r2
  concatenated (`scripts/dagger_set_concat.exs`), one arm
  `on30u_dag2x4_e3` (oversample 4 ≈ 5× dagx4's onset examples). Same
  pass bar. If it rises again without reaching the bar, the next dose is
  more rounds/seeds, not a new mechanism.
- **22:12 — `on30u_dag2x4_e3` (queue 34; round 2, r1 + r2 = 66,884
  relabelled frames ×4 ≈ 5× dagx4's dose): the AIM HALF PASSES and the
  dose overshoots.** (Queue 34's first run died at seed 8 of a 12-seed
  rollout: GPU `RESOURCE_EXHAUSTED` — see 22:16 below; rerun as four
  3-seed beams.) Set r2: 1,808 runs / 48,265 frames, labels stick-up onset
  2.74 % (expert 3.1 %), jump edge 6.6 %, B edge 0.42 %, 190 up-onsets +
  316 jump edges. Arm: **loop up-onset once spent 8.3 %/3 f (bar 7;
  expert 12.3), stick-up share 42.0 % (expert 42.5)** — the bot now aims
  up at the expert's rate on its own states. But **P(B|up) 4.9 %**
  (expert 14.7; dagx4 24.7) — it holds up and does not press —
  Firefox-once-spent 0.005 / 0.0068 (bar 0.04, unchanged); and the mix
  damaged the rest: fidelity **0.278** (bar 0.21), neutral 0.18, repeat
  0.727, damage/min **16** (dagx4 55), dashes/min **16** (44), jump rows
  collapsed 0.038 / 0.12 / 0.20, `airdodge_with_jump` **0.316** with
  airdodge the top first-means (45) and 18 airdodge deaths; deaths/min
  1.07, return rate 0.83, recovery-means return 0.365 / decided 0.479.
  Dose curve so far (effective mix frames → up-onset): 37k 3.9 %, 74k
  4.6 %, 268k 8.3 % + damage. Two readings to settle: the dose middle
  (×1, ×2 of r1+r2 = 67k / 134k) and whether the airdodge/side-B labels
  are k-NN extrapolation on states the expert never visits.
- **22:16–22:27 — coverage readout, a labeler bug, and labeler v2.**
  `scripts/expert_labeler_distance.exs` (nearest expert row's squared
  distance, held-out expert states vs the bot's states in a set, with a
  per-feature decomposition). First run: 77 % of r2's states beyond the
  expert's own q95, d² carried by **vy 1.35 + vx 0.58** — and
  `scripts/dagger_set_speed_check.exs` showed why: **the index's vy/vx
  columns are ALL ZERO** (`speed_y_self` / `speed_air_x_self` exist in
  Slippi ≥ 3.5 only; this corpus has none) while the sim fills them. A
  constant per query, so the neighbour RANKING — and every label so far —
  was unaffected; the distances were inflated. **v2**: velocity = position
  delta (`prev_p` in `features/5`; `{p, prev, opp, prev_p}` query states;
  index rebuilt → `expert_recovery_index_v2.bin`, same 214,443 rows;
  self-check on 24 held-out games jump ×1.16 / up ×1.14 / B ×1.4 — passes;
  sets now carry `prev_player`). Also found and fixed the **GPU leak**:
  each 256-query chunk made a 256×214k f32 distance matrix (219 MB) as an
  eager EXLA buffer freed only at the beam's next GC → `RESOURCE_EXHAUSTED`
  after ~100 chunks (the 12-seed death, and the diag itself at 22:16);
  now one jitted `defn` kernel (distance + top-k inside the executable) +
  `:erlang.garbage_collect()` per chunk. With real velocity on both sides
  (one full v2 seed): bot states q50 0.41 / q95 2.88 vs expert 0.09 /
  0.78; **36 % beyond the expert's q95, carried by the OPPONENT's
  relative position (opp_dx 0.48 + opp_dy 0.22 of ~1.9) then own height
  / edge distance (0.20 / 0.19)** — self-play puts the other bot where no
  human edgeguarder stands, and the bot goes deeper than the expert ever
  does. `--max-d2` gate added to the rollout (a gated frame stays in the
  trip as input-only context; report carries `gated_share`).
- **22:27 — queue 35 launched** (`scripts/coherence_queue35.sh`): (A)
  `on30u_dag2x2_e3` (r1+r2 ×2 = 134k, the dose middle); (B) gated v2
  round: `dagx4_e3` rolled out on 12 seeds in 4 beams with the v2 index
  and `--max-d2 0.78` → r3g (+ its label hazards and coverage readout) →
  `on30u_dag3g_x4_e3`; (A′) `on30u_dag2x1_e3` (×1 = 67k). Same pass bar
  plus damage/min ≥ 45. Read-out ~04:30.
- **10-09 00:30 — `on30u_dag2x2_e3` (dose middle, r1+r2 ×2 = 134k): FAILS,
  and the dose curve is not the whole story.** Aim: loop up-onset once
  spent **5.1 %/3 f** (bar 7; dagx4 4.6, dag2x4 8.3), stick-up share
  21.0 % (dagx4 21.5, dag2x4 42.0), P(B|up) 7.9 %, Firefox-once-spent
  0.0085 / 0.0048 / 0.0043 (bar 0.04). Jump rows 0.11 / 0.186 / 0.179
  (rows −40..−60 and <−60 ×0.6 of the expert). Band: repeat 0.748,
  neutral 0.21 (bar 0.22), fidelity 0.181 — just inside. **Closed loop
  WORSE than dagx4: SD/min 1.68 (bar 1.4; dagx4 0.92), deaths/min 1.79
  (dagx4 1.01), offstage eps/min 7.9 (5.7), return 0.77 (0.82),
  side-B deaths 53/53 (dagx4 24/24), carried-off share 0.32 (0.26),
  decided return 0.469 (0.623), mismatch 0.153 (0.097).** Damage/min 47
  and dashes/min 25 pass. So at 134k effective frames the aim moved
  +0.5 pt over dagx4 while the closed loop regressed to dagx2's level —
  the r2 set (labels on dagx4's states, 2.7 % up-onset) buys aim only at
  the 5× dose and costs recoveries at every dose. Dose curve: 37k 3.9 %,
  74k 4.6 %, 134k 5.1 %, 268k 8.3 %. Reading: the r2 labels are not the
  r1 labels — r2's states are deeper / further and 36 % of them are
  beyond the expert's own q95 (22:16), i.e. the extra dose is partly
  extrapolated labels (side-B, airdodge). The gated round is the test.
- **00:27 — gated v2 rollout (round r3g) runs clean:** three of four
  3-seed beams done (no `RESOURCE_EXHAUSTED` — the jitted kernel + GC
  hold), each ≈ 450–470 runs, **7.9–8.4k relabelled + 40–42k input-only
  + 3.3–3.7k gated (28.5–30.7 % of labelable; forecast was 36 %)**, hold
  share 0.79–0.81, label B 7.7–9.3 %, jump 8.7–10.3 %, policy silent on
  17–23 %. Expected r3g ≈ 33k relabelled frames ×4 ≈ 132k effective — the
  same dose as dag2x2 with the extrapolated 30 % removed, so
  `dag3g_x4_e3` vs `dag2x2_e3` isolates coverage from dose.
- **00:38–00:50 — r3g complete; `dag3g_x4_e3` died at the loader; queue
  36.** r3g: 1,821 runs, **32,533 relabelled frames** (fourth beam 35.6 %
  gated), label hazards jump edge 8.05 % (expert 8.76), up-onset 2.91 %
  (3.1; 4.4 % below −40), B edge 0.52 % (0.24) — the gated labels keep
  the expert's edge rates. Coverage readout on the gated set: bot states
  q50 0.20 / q95 0.67 / q99 0.75 vs expert 0.09 / 0.78 / 1.86, 1 of 32,533
  beyond the gate (the gate works by construction; the residual far state
  is a prev-stick mismatch). The arm then failed in 4 s:
  `Data.from_frame_lists/2` requires each list to be `[input-only prefix,
  targets]` and raises on an input-only frame AFTER a target — the gate
  marks frames input-only in the MIDDLE of a trip. Fix without touching
  lib (queue live): `scripts/dagger_set_split_gated.exs` re-cuts each trip
  into one list per maximal target run with the trip's earlier frames as
  input-only context (1,821 → 2,421 lists, targets unchanged, context
  180k → 260k; written `[:compressed]` — the first uncompressed write was
  686 MB on a root disk with 4 GB free). `scripts/coherence_queue36.sh`
  waits for queue 35 to finish (dag2x1_e3 is training) and runs
  `on30u_dag3g_x4_e3` on `sim_dagger_expert_r3g_split.frames` ×4, readout
  vs dagx4 / dag2x2. Unit `exphil-queue36`, log `logs/exphil-queue36.log`.
- **02:06 — `on30u_dag2x1_e3` (r1+r2 ×1 = 67k): not at the bar, but the
  best Firefox-once-spent of the series and the dose reading is now
  clear.** Aim: up-onset **5.5 %/3 f** (bar 7), stick-up share 18.1 %,
  **P(B|up) 18.8 %** (expert 14.7; dag2x2 7.9, dag2x4 4.9),
  **Firefox-once-spent 0.0116 / 0.0128 / 0.009 / 0.0145** by band (bar
  0.04; dagx4 0.0085, dag2x2 0.0085/0.0048). Jump rows 0.078 / **0.259** /
  0.179 (the −20..−40 row ×0.39 fails 0.15; −40..−60 ×0.87 passes). Band:
  repeat 0.763, neutral 0.242, fidelity 0.168 — inside. Closed loop:
  **SD/min 1.22 (passes), deaths/min 1.30, damage/min 71 (best of the
  series), dashes/min 23.6 (passes), return 0.785**, recovery-means
  return 0.527 / decided 0.613 (dagx4 0.491 / 0.623), mismatch 0.141,
  side-B deaths 22/22, died trips 81 with trailing-identical q50 3
  (expert 27 — the bot dies while still changing inputs, not silent).
  Dose series on r1+r2 (×1 / ×2 / ×4 = 67k / 134k / 268k): up-onset 5.5 /
  5.1 / 8.3, P(B|up) 18.8 / 7.9 / 4.9, SD/min 1.22 / 1.68 / 1.04,
  damage/min 71 / 47 / 16, fidelity 0.168 / 0.181 / 0.278. **Reading:
  oversampling the relabelled set past ×1 buys stick-up share at the
  price of the press and of style; at ×1 the set carries aim +0.9 pt over
  dagx4 with the press intact.** What is missing at every dose is the
  Firefox itself (≤ 0.015 vs 0.04), and the −20..−40 jump row. The
  coverage question (queue 36, same ×4 oversample on the gated r3g) is
  the remaining lever in flight; next doses should ADD SEEDS at ×1, not
  oversample.
- **03:34 — `on30u_dag3g_x4_e3` (queue 36; gated v2 round r3g ×4 ≈ 130k,
  the same dose as dag2x2): THE COVERAGE GATE IS THE LEVER — the aim half
  of the bar passes with the press intact and the closed loop kept; the
  Firefox itself still does not fire.** Aim: **loop up-onset once spent
  9.7 %/3 f (bar 7; dag2x2 at the same dose 5.1)**, stick-up share
  33.6 % (expert 42.5), **P(B|up) 14.3 % (expert 14.7)**, P(B|not up)
  0.6 %. Band: repeat 0.760, neutral 0.254, fidelity 0.174 — inside.
  Closed loop: **SD/min 1.23, dashes/min 30.1, damage/min 47.8 — all
  pass**; deaths/min 1.29, return 0.785, decided return 0.641 (best),
  mismatch 0.109, side-B deaths 45 (43†), airdodge_with_jump 0.073.
  Jump rows 0.122 / **0.265** / 0.240 (−20..−40 ×0.6 fails 0.15; the
  deeper rows pass). Side by side with dag2x2 (ungated, same 134k):
  up-onset 5.1 → 9.7, P(B|up) 7.9 → 14.3, SD/min 1.68 → 1.23, dashes
  25 → 30, side-B deaths 53 → 45. **Removing the 30 % of states beyond
  the expert's own range turned the dose into aim without the press and
  style damage — the damage at ×2/×4 was extrapolated labels, not dose.**
  **What still fails: Firefox-once-spent 0.0027 / 0 / 0.0042 / 0.0061
  (bar 0.04; expert 0.02 / 0.02 / 0.056 / 0.056).** Per-frame special
  onsets of any kind in `<−60:j0` sum to 0.0084 vs the expert's 0.06.
  Trace check (jump-spent, y < −20, every 3 f; helpless = FallSpecial
  35–37, aerials 65–69): **the bot is HELPLESS on 54–62 % of its
  spent-low samples (expert 31 %)** — the side-B / airdodge / early
  Firefox that preceded; on the actionable remainder its fresh B-press
  rate per sample is 2.2 % (dag3g; dagx4 4.7 %, expert 4.8 %), 10 of 12
  with the stick up (expert 94 / 105), and its HELD B-down samples were
  pressed inside an aerial (prev action 65–69: 14 of 16; expert 9 of 65)
  — a press that lands during an attack does nothing and is then held
  into the fall, where a held B never fires. The labels agree with that
  defect: in r3g the labelled B edges once spent (129; 116 with the
  stick up, so the direction is right) start **28 % inside an aerial**
  (36 / 129 at action 65–69; the expert's own fresh presses 11 %) —
  the labeler has `jumping` / `airdodge` / `hitstun` / `special` dims
  but no ATTACKING dim, so a bot mid-fair matches an expert in free fall
  at the same height and inherits the press. Labelled B edges by band
  0.95 / 1.5 / 2.6 % (−20..−40 / −40..−60 / <−60), so the labels carry
  height conditioning (×1.6; the expert's onsets ×2.8). Two things
  follow, both label-side: **labeler v3 = `attacking` dim** (+ helpless
  36/37 excluded from `labelable?`, 35 already was) and a gated v3 round
  from dag3g_x4_e3's own states; the model-side question (the AR head
  decodes buttons before the stick — the HELD reorder) stays with
  Bradley. The helpless share is the other half: what the bot does
  BEFORE the press (side-B 45 deaths, airdodge) — read next from the
  r3g labels on those states.
- **03:42–05:43 — labeler v3 + queue 37.** v3 = `attacking` dim (65..69,
  weight 1.5) + helpless 36/37 not labelable; index v3 (same 176 games,
  214,443 rows). Held-out self-check passes: jump ×1.17, up-onset ×1.20,
  B ×1.58, hold 0.808 vs 0.794, agreement 68 %. v3 gate (held-out q95)
  0.814. Lost two hours: the queue ran the self-check mix-free, which
  leaves Nx on BinaryBackend (no config loaded) — the 214k-row nearest
  kernel crawled on one core with no output (`logs/exphil-queue37_cpuhang.log`);
  GPU-side labeler reads go through `mix run --no-compile` (the queue
  only runs with no other beam alive; `mf` stays for the CPU-only set
  tools). Queue 37: gated v3 round r4g3 from dag3g_x4_e3's own states
  (12 seeds) → `dag4g3_x4_e3`, then `dag3g_x2_e3` (dose middle on the
  gated r3g), `dag34g_x2_e3` (r3g + r4g3 ×2).
- **08:03 — r4g3 + `on30u_dag4g3_x4_e3`: the Firefox moves for the first
  time (×3), height-conditioned, still under the bar; the aim gives
  back; style slips.** r4g3: 1,918 runs, **42,014 relabelled** (29–36 %
  gated), hazards jump edge 7.5 % (expert 8.8), up-onset 4.34 % (3.1;
  5.9 % below −40), **B edge 1.35 % (expert held-out 0.24; r3g 0.52)** —
  the v3 labels press B on the bot's deep states at 5× the expert's
  held-out rate, which the self-check (×1.58 on the expert's own states)
  did not show: the bot's spent-low states are where the expert presses.
  Coverage: 8 of 42,014 beyond the gate. Arm: **Firefox-once-spent
  0.0017 / 0.005 / 0.0151 / 0.0186 by band (expert 0.02 / 0.02 / 0.056 /
  0.056; bar 0.04 at ≤ −40; dag3g 0.0027 / 0 / 0.0042 / 0.0061)** — the
  first arm with a height-conditioned Firefox (×3 at −40..−60, ×3 below
  −60). Jump rows **0.202 / 0.384 / 0.507 — pass** (above the expert's
  0.20 / 0.30 / 0.30). But aim: up-onset 6.7 % (bar 7; dag3g 9.7),
  stick-up share 25.3 % (33.6), P(B|up) 7.5 % (14.3). Band: repeat 0.768,
  fidelity 0.181, neutral **0.216** (bar 0.22, marginal). Closed loop:
  SD/min 1.28 (pass), damage/min 63 (pass), **dashes/min 19.9 (bar 22,
  fail)**, deaths/min 1.43, return 0.80, decided return 0.585, mismatch
  0.122, side-B deaths 28 (dag3g 45), up-B deaths 6 (6†), died trips
  112 with 66 holding neutral (dag3g 35/110; "passive Fall" 31 % vs
  20 %), stick toward 21 % / away 19 % (dag3g 42 / 12). Two things
  changed at once here (labeler v3, and round 4 = labels on dag3g's
  states), so which moved the Firefox is not separated; `dag34g_x2_e3`
  (r3g v2 + r4g3 v3 at ×2) is the mix of both. Reading so far: the
  gated rounds move the Firefox and the jump rows monotonically
  (0.006 → 0.019 at <−60 across r3g → r4g3) while the stick aim
  oscillates with the set; the press itself (B edge 1.35 % in the
  labels) is now over-supplied relative to the expert's own rate and
  the bot's passive-fall deaths rose with it.
- **09:31 — `on30u_dag3g_x2_e3` (gated r3g at ×2 ≈ 65k): the aim does not
  survive the half dose.** Up-onset **4.5 %** (×4: 9.7), stick-up share
  13.2 % (33.6), P(B|up) 9.9 %, Firefox-once-spent 0 / 0.0045 / 0.0074 /
  0.0093 (×4: 0.003–0.006; dag4g3 0.015–0.019). Jump rows 0.137 / 0.178 /
  0.306. Band and closed loop all pass (repeat 0.760, neutral 0.278,
  fidelity 0.172; SD/min 1.26, dashes/min 27.4, damage/min 61,
  deaths/min 1.38, return 0.79); airdodge deaths 13 (12†),
  airdodge_with_jump 0.102. So on the gated set the aim is a ×4 effect
  (as on r1: dagx2 3.9 → dagx4 4.6 was the same direction, smaller) —
  with the caveat that every arm here is one training seed (905) and
  the s906 replicate of on30u_e3 moved the return rate by 0.12, so a
  4.5-vs-9.7 split on one seed each is suggestive, not settled. The
  ×4 arms are the ones to carry forward; `dag34g_x2_e3` (r3g + r4g3 ×2
  = 75k relabelled ×2 ≈ 150k) reads next.
- **11:01 — `on30u_dag34g_x2_e3` (r3g v2 + r4g3 v3, 74,547 relabelled ×2
  ≈ 150k): THE AIM HALF IS AT THE EXPERT, THE FIREFOX AT HALF THE BAR,
  THE BAND AT ITS EDGE.** Aim: **up-onset once spent 12.1 %/3 f (expert
  12.3; bar 7), stick-up share 47.7 % (42.5), P(B|up) 20.5 % (14.7)**,
  P(B|not up) 4.4 %. **Firefox-once-spent 0.0027 / 0.0117 / 0.0199 /
  0.0137** (expert 0.02 / 0.02 / 0.056 / 0.056; bar 0.04 at ≤ −40) —
  the −40..−60 row is now the expert's −20..−40 value; the series at
  −40..−60: dagx4 0.0085 → dag3g 0.004 → dag4g3 0.015 → dag34g 0.020.
  Death shape: B onsets in died trips **up 75 / side 39 / down 16 /
  neutral 4 (expert 79 / 47 / 5 / 3), 96 below −40 with 65 stick-up
  (expert 59 / 48)** — the bot now dies pressing up-B like the expert,
  and no longer dies holding nothing (10 / 106 vs dag3g 35, dag4g3 66;
  the expert's 88 / 104 are trips already lost). "never special" 44 %
  (expert 30) is the residual. Jump rows 0.093 / 0.220 / 0.281 (the
  −20..−40 row fails 0.15 again; deeper rows pass). Closed loop:
  **SD/min 1.03, deaths/min 1.09, return 0.83** (the best since dagx4),
  damage/min 57 (pass), **dashes/min 21.1 (bar 22, marginal fail)**,
  decided return 0.636, mismatch 0.192, side-B deaths 32, airdodge 14
  (11†), airdodge_with_jump 0.148, side_b_low 0.2. **Band at its edge:
  repeat 0.708 (bar 0.70), neutral 0.192 (bar 0.22, fail), fidelity
  0.221 (bar 0.21, marginal fail)**, short-hop share 0.284 (expert
  0.405). Reading: two gated rounds compound — the aim is done, the
  Firefox hazard doubles per round, and the dose (150k at ×2) is where
  the style starts to pay (neutral/fidelity), exactly as the r1+r2
  series showed at ×2. Next dose = a third gated round (v3, from
  dag34g_x2's own states) at **×1** of r3g + r4g3 + r5g (≈ 115k
  relabelled), plus the seed replicate this program has been owing
  (dag34g_x2 at s906) — queue 38.
- **11:55 — gated v3 round 5 (`sim_dagger_expert_r5g3`, from
  dag34g_x2_e3's own states, 12 seeds, gate 0.814):** 2206 runs, 39,367
  relabelled, gated 31–40 % of labelable (r4g3: similar), policy silent
  on 10–11 %, label hold share 0.80–0.82. Label hazards: jump edge 7.1 %
  (expert held-out 8.8), stick-up onset 2.6 % (3.1), **B edge 0.61 %
  (expert 0.24; r4g3 was 1.35 — v3 labels converge toward the expert's
  rate as the states get nearer)**, near|far edges 1.4|0 / 1.2|0 / 1.1|0.
  Split 2206 → 2944 lists; r345g_split = 8411 runs, 113,914 relabelled.
- **13:31 — `on30u_dag345g_x1_e3` (r3g + r4g3 + r5g3 at ×1 ≈ 114k):
  BEST CLOSED LOOP OF THE SERIES, FIREFOX AT −40..−60 STUCK AT 0.019,
  THE REPEAT SHARE FALLS OUT OF THE BAND.** Closed loop: **SD/min 0.83
  (best; expert 0.44), deaths/min 0.98 (expert 0.96), return 0.852,
  drill recovery 0.559 (best; never 1, always 2), dashes/min 24.7
  (pass), damage/min 54 (pass), fidelity 0.208 (pass), neutral 0.243
  (pass)**, side_b_low 0.125 (expert 0.107; was 0.2), decided return
  0.583, side-B deaths 21, airdodge 9 (8†), airdodge_with_jump 0.127.
  **Fails: input repeat share 0.643 closed loop (bar 0.70; teacher-forced
  0.703)** — trailing identical frames before death q25/50/75 = 0/0/6
  (expert 12/27/36), i.e. the bot now keeps changing inputs to the last
  frame; jump rows 0.099 / 0.140 / 0.198 (bar 0.15 / 0.20 — all three
  deep rows now miss, dag34g_x2 passed the deeper two). Aim: **up-onset
  19.0 %/3 f (expert 12.3 — overshoot), stick-up share 47.3 %, P(B|up)
  8.4 % (14.7; dag34g_x2 20.5)** — the stick goes up more and the B
  follows it less. **Firefox-once-spent 0.0057 / 0.0018 / 0.0188 /
  0.0350** — the −40..−60 row is the third arm in a row at 0.015–0.020
  (bar 0.04), the < −60 row 0.035 is the best of the series (expert
  0.056). Death shape: B onsets up 50 / side 37 / neutral 12 / down 10,
  65 below −40 with 35 stick-up; died holding nothing 22 / 89. Reading:
  the ×1 three-round set buys the closed loop (SD, drill, return, band
  on neutral/fidelity) and the deep Firefox row, but the B press at
  −40..−60 is a level the gated rounds have not moved since r4g3 — the
  aim keeps rising past the expert while P(B|up) halves, which is the
  "stick without the press" shape from the dose series, now from more
  rounds instead of more copies. The repeat-share fall (0.64) is the new
  style cost. dag345g_x2 and the s906 replicate still to read.
- **12:56–13:05 — LIVE LOOK (Bradley, `dag34g_x2_e3`, 5 games BF/FoD/FD/DL/FD,
  `eval_runs/local_recording_temp_1_20261009_125549`):** "interesting, it
  seemed to make progress on the problem … it would airdodge sometimes
  offstage, side-B from stage to offstage, up-B slightly out of range, do
  the wrong up-B angle, etc. but it was certainly better at recovering than
  predecessors." Death trips pulled from the .slp (scratch
  `live_death_trips.exs`: action changes from the last on-stage frame to
  death, with x/y/stick/buttons), 18 stocks lost: **Firefox fired at the
  wrong angle 8** (away ×3, straight down ×1, straight up from 45 units
  out ×1, so shallow it flew under the stage ×3), ground side-B straight
  off the edge with no jumps 2, side-B too low / into the underside /
  short 3, airdodge offstage away from the stage 2 (+1 overlap), Firefox
  too late / out of range 2 (one with B *held* since a laser so no new
  press until y −138; one started at −129 with the double-jump unused),
  legit edgeguard 1. In 5 of the 8 angle deaths the stick was fine at
  charge start and **wandered during the ~42-frame charge** (FoD f1063
  up-toward → down; FD f831 up-toward → up-away; DL f4972 up → sideways).
  17 of 18 deaths had an input attempt — the silent fall is mostly gone;
  the attempt is wrong. **The pass metric (Firefox-once-spent) scores the
  press only and is blind to the angle.**
- **13:50 — Firefox angle instrument `scripts/recovery_firefox_angle.js`
  (per 354 charge run in the traces: stick on the last charge row vs the
  unit vector to the near ledge, bucket toward-up / vertical / horizontal /
  away / down / neutral, charge stability = share of charge rows on the
  fire row's 16-bucket stick):**

  | | Firefox / eps | angle err q50 / q75 | > 60° | away | down | neutral | died after Firefox | first == fire | jump in hand at charge |
  |---|---|---|---|---|---|---|---|---|---|
  | expert (FD) | 365 / 1788 (20 %) | 19° / 34° | 7 % | 0 % | 12 % | 2 % | 16 % | 15 % | 2 % |
  | dagx4 | 17 / 165 | 46° / 63° | 29 % | 12 % | 12 % | 18 % | 47 % | 24 % | 18 % |
  | dag3g_x4 | 19 / 229 | 24° / 51° | 11 % | 5 % | 0 | 21 % | 42 % | 11 % | 11 % |
  | dag4g3_x4 | 24 / 238 | **15° / 31°** | 4 % | 0 | 0 | 8 % | 38 % | 29 % | 8 % |
  | dag34g_x2 | 26 / 213 | 35° / 41° | 15 % | **12 %** | 4 % | 0 | 46 % | 31 % | 19 % |
  | dag345g_x1 | 35 / 180 (19 %) | 35° / 55° | 20 % | 3 % | 14 % | 14 % | 57 % | 9 % | 20 % |

  Reading: the sim agrees with the live look — the bot's Firefox dies
  3× as often as the expert's (46–57 % vs 16 %) and the angle is why
  (q50 35° vs 19°; > 60° 15–20 % vs 7 %; "away" 12 % in the arm Bradley
  played, expert 0). **The expert also moves the stick during the charge
  (first == fire 15 %, ≥ 3 distinct sticks 82 %)** — wander per se is
  normal; the difference is where it ends. Usage is at the expert's rate
  in dag345g_x1 (19 % of episodes). dag4g3_x4 had the best angle (15°) and
  the worst aim — the two halves have traded across the series. Jump in
  hand at charge 8–20 % (expert 2 %) is the standing jump-row miss seen
  from the Firefox side. n = 17–35 per arm: direction, not decimals.
  **Label side (scratch `label_firefox_angle.exs`, charges inside the
  gated sets r3g / r4g3 / r5g3):** 122 / 142 / 158 labelled charges, fire
  bucket toward-up 42–46 %, vertical 20–35 %, horizontal 8–20 %, neutral
  ~7 %, down 5–8 %, **away ≤ 1 %**, same-as-fire share 0.67–0.73 (held
  better than the expert's own 0.43). So the labels do NOT carry the
  wrong angle — the away/down fires are the policy's, i.e. a
  generalisation miss on the fire-frame stick, not a label artefact. No
  left/right asymmetry (away+down L 2/10 R 2/16 in dag34g_x2). Next: put
  the instrument in every readout (queue 39+), and weigh a fire-frame
  weight (the last charge frame's stick is the decision; `--onset-weight`
  weights the press, nothing weights the fire) against a label-side
  "hold the fire stick through the charge" — the labels already hold it
  at 0.7, so the model-side weight is the cheaper first test.
- **14:30 — the knob, not the fire frame (angle instrument over every
  onset arm, no DAgger data, same recipe otherwise):** `dur8e_e3` (no
  onset weight) **13° / away 0 %**; `on10` 34° / 6 %; `on30_e3` 31° / 0 %,
  its s906 30° / 9 %; `on30u_e3` 30° / 14 %, its s906 **51° / 15 %**. Any
  onset weight costs angle; the stick-enters-up term (`y ≥ 0.75`, blind to
  x — `silent_fall_weighting.ex:107-114`) costs the most. A fire-frame
  weight would be a Fox-specific patch on a general knob's side effect
  (Bradley's "is it overtuning?" — yes); the arms are to remove or narrow
  the knob. Also Bradley: the state BEFORE the offstage one is probably
  the better target — and the sim has had the number all week: **carried-
  off share 0.27–0.33 vs expert 0.055, carried-off died 0.8–0.9 vs 0.16**;
  plus the ground side-B off the edge and the airdodge-off-the-stage
  deaths start in states the labeler never sees (`labelable?` = airborne
  past the ledge). → queue 40 = widen the label window to grounded /
  above-stage near-edge states.
- **15:05 — `on30u_dag345g_x2_e3` (three rounds at ×2 ≈ 228k): the dose
  law a third time.** dashes/min **11.7**, offstage eps/min 4.8, Firefox-
  once-spent 0 / 0.005 / **0.003** / 0.016, P(B|up) 7.0 %, drill 0.438,
  return 0.78, repeat 0.70, neutral 0.201, fidelity 0.226, carried-off
  died 0.91; angle 27° / away 10 %. ×1 is the dose.
- **16:37 — `on30u_dag34g_x2_e3_s906` (the seed replicate of the arm
  Bradley played): SEED VARIANCE IS LARGE.** Same recipe, same data, seed
  906 vs 905: **SD/min 1.50 vs 1.03, deaths/min 1.66 vs 1.09, drill 0.424
  vs 0.49, Firefox-once-spent −40..−60 0.0082 vs 0.0199**, stick-up share
  30 % vs 48 %, P(B|up) 13.7 vs 20.5, angle err 44° vs 35°, neutral-fire
  22 % vs 0 — but **up-onset 11.7 % vs 12.1 %** (the aim replicates),
  jump rows 0.125 / 0.274 / 0.355 (deeper rows pass, better than s905),
  fidelity **0.159**, neutral 0.32, repeat 0.761, dashes 23.5, damage 47
  (the band passes on this seed), side_b_low 0.0, airdodge 16 (15†).
  Reading: what survives two seeds — the aim at ~12 %, the dose law
  (three pairs), gated ≫ ungated (large effects), the angle being bad
  (35–44°). What does NOT — "Firefox at half the bar" (0.020 is one seed;
  0.008 on the other), SD/min 1.03, the band failing/passing. The
  program's single-seed reads this week are within seed noise on the
  press and style axes; queue 39 carries its own replicate (on0 s906).
- **16:38 — queue 39 started** (`scripts/coherence_queue39.sh`; patch
  `--onset-buttons-only` applied in its beam-free window, compiled): arms
  `dur8e_dag345g_x1_e3` (no onset weight), `on30b_dag345g_x1_e3` (press
  edges only), `on5u_dag345g_x1_e3`, `dur8e_dag345g_x1_e3_s906`; readout
  + `recovery_firefox_angle.js`; bar = queue 38's + angle err q50 ≤ 25°,
  away ≤ 3 %, died after Firefox ≤ 30 %.
- **18:13 — `dag345g_x1_e3` (queue 39 arm 1: the ×1 three-round set with
  NO onset weight): THE GATED LABELS CARRY THE PRESS WITHOUT THE KNOB,
  THE ANGLE MEDIAN HALVES, THE TAIL DOES NOT, FIREFOX AT −40..−60 IS THE
  LOWEST YET.** Aim: **up-onset 9.8 %/3 f (expert 12.3; bar 7 — no
  overshoot, on30u was 19.0), stick-up share 35.7 % (42.5), P(B|up)
  23.1 % (14.7; on30u 8.4)** — with the knob off, the stick goes up less
  and the B follows it more, i.e. the "stick without the press" shape
  was the knob's. Angle: **err q50 23° (bar 25, pass; on30u 35, dur8e_e3
  13) but q75 55° (same as on30u), > 60° 18 %, away 5 % (bar 3, fail),
  died after Firefox 18/39 = 46 % (bar 30, fail)**; fire bucket
  toward-up only 18 % (expert 48), vertical 33, neutral 18; the deaths
  fire at q50 55° and wander (first ≠ fire 83 %); charge stability
  same-as-fire 0.36 (expert 0.43), **jump in hand at charge 26 % (expert
  2 %), fire distance q50 10 units (expert 35)** — it fires late and close,
  a quarter of the time with a jump still unused. Press: **Firefox-once-
  spent 0.0034 / 0.0152 / 0.0026 / 0.0156 — the −40..−60 row is the
  lowest of the series (bar 0.04; the on30 arms sat at 0.015–0.020)**;
  jump rows 0.05 / 0.17 / 0.16 / 0.14 (bar 0.15 / 0.20 — one of two);
  jump hazard with a jump in hand x0.55 / x0.41 / x0.61 of the expert at
  −20 and below. Closed loop: **SD/min 1.54 (bar 1.4, fail; on30u 0.83
  — but the 10-09 16:37 seed read says 0.5 of SD/min is one seed)**,
  deaths/min 1.77, return 0.765, drill decided return 0.512, dashes
  24.8 (pass), damage 51.9 (pass), **repeat 0.736 (pass; teacher-forced
  0.747), neutral 0.354 closed loop (bar 0.33, fail; teacher-forced
  0.24), fidelity 0.192 (pass)**; airdodge_with_jump 0.124, side_b_low
  0.0; carried-off share 0.347 / died 0.874 (unchanged — the queue-40
  number). val 3.079. Reading: removing the knob fixes the two things
  it broke (aim overshoot, angle median) and nothing else — the B at
  −40..−60 is now at 0.003, so the knob was holding that row at 0.02,
  and the angle tail (q75 55°, 46 % die after firing) is NOT the knob's:
  it is the late/close fire with the stick wandering through the charge,
  which no onset arm has moved. on30b / on5u / on0 s906 to read.
- **18:55 — labeler v4 (`:wide` window) + queue 40 queued behind 39
  (`scripts/coherence_queue40.sh`, unit `exphil-queue40`).** The window
  is now a property of the index (`index.window`; v1–v3 files load as
  `:offstage`, nothing changes for them): `labelable?(p, edge, :wide)` =
  the offstage rule OR any live state (grounded too) within
  `near_edge/0` = 15 units of the edge, ledge/helpless/dead excluded;
  `:wide` dims = the 18 + `grounded` (weight 3 — a grounded query cannot
  borrow an airborne label through the gate), `shield` (178..182, 1.5),
  `dash` (20..23, 1.0). `build_expert_recovery_index.exs --window wide`;
  selfcheck / distance / sim_recovery_dagger_expert read the window off
  the index. CPU check: grounded at |x| 75 and airborne above the stage
  at 80 labelable under `:wide` only, ledge/helpless/dead never, vectors
  18/21, save/load round-trip. Queue 40 (after `QUEUE 39 DONE`, ~23:00):
  index v4 (same 176-game split) → self-check (x0.5–x2) → held-out q95
  gate → gated round r6w from the queue-39 winner
  (`eval_runs/1001_queue/queue40_winner.sh`, default on0; set by hand
  after the 39 reads) → arms `dag3456w_x1_e3` (r345g + r6w) and
  `dag6w_x1_e3` (the wide round alone). Pass = queue 39's bar AND the
  upstream number moves: carried-off share ≤ 0.20 (0.35; expert 0.055),
  carried-off died ≤ 0.60 (0.87; 0.16), airdodge_with_jump ≤ 0.06 (0.12;
  0.006), side-B deaths by move down from 40. Dose caveat: a wide round
  labels more frames (near-edge grounded time); r345g + r6w may cross the
  ×1 cliff (~150k) — the `dag6w` arm is the control for that.
- **19:47 — `on30b_dag345g_x1_e3` (queue 39 arm 2: `--onset-weight 30
  --onset-buttons-only`): THE PRESS-EDGE TERMS ALONE BUY THE JUMP ROWS
  AND LOSE THE AIM; THE ANGLE IS THE WORST OF THE FOUR.** Aim: up-onset
  **5.4 %** (bar 7, fail; on0 9.8, expert 12.3), stick-up share 22 %,
  P(B|up) 8.3 %. Firefox-once-spent 0.011 / 0.007 / 0.007 / 0.011 (fail).
  **Jump rows 0.08 / 0.20 / 0.30 / 0.28 — the first arm of the series to
  pass both deep rows (bar 0.15 / 0.20)**, jump hazard x0.6 / x0.72 /
  x0.77 of the expert. Angle: **err q50 39° / q75 51°, > 60° 22 %, away
  9 %, died after Firefox 15/23 = 65 %** (all fail); fire distance q50
  13, jump in hand at charge 9 %. Closed loop: SD/min 1.34 (pass),
  return 0.793, drill decided 0.424, repeat 0.699 (bar 0.70 — a hair
  under), neutral 0.229 (pass), fidelity 0.185 (pass), dashes 21.6 (bar
  22, fail), damage 50 (pass); airdodge_with_jump 0.17 (worst), carried-
  off share 0.261 / died 0.878; side-B deaths 15 (on0 42), airdodge 13.
  val 3.110. Reading: the ×30 on the jump edge is what the jump rows
  were always waiting for (they pass here and nowhere else), but the B
  term is `b_edge && stick_up?` — it is ALSO blind to x, and this arm's
  39° says the fire-angle bend was never only the stick-enters-up term:
  weighting "B with the stick up" ×30 teaches the up-stick at the press
  and the charge holds it. The buttons-only flag is not the knob's fix.
