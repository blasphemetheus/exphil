# Latency architecture — findings, direction, and migration plan

Started 2026-07-28. The question: should the system accommodate input lag
(padding, staleness stats, robustness hacks) or be designed so lag is a
known constant? Answer: **lag cannot be zero (platform physics) but it can
and must be a KNOWN CONSTANT, budgeted explicitly.** Variable, unbudgeted
lag is the thing generating bespoke accommodation everywhere.

## The three lag layers (treat differently)

1. **Engine-inherent (~1 frame, fixed).** libmelee applies inputs on the
   next frame, like a physical controller. Cannot remove; trivial to model
   (`--action-delay 1`).
2. **Async-runner jitter (variable 1-2+ frames).** The async loop infers
   on the freshest frame and sends when done; crossing a frame boundary
   slips the input. INCIDENTAL complexity — the source of the staleness
   health-stat, discard thresholds, reboot rules, and fuzzy offline
   probes. This layer gets engineered away.
3. **Online delay (18+ frames, future netplay).** A deliberate, trained
   capability (`--online-robust` / delay-matched training), not something
   backed into via a jittery harness.

## Evidence that layer 2 is real and unbudgeted (2026-07-28)

- Offset calibration in `probe_cycle_margins.exs` (transition-frame parity
  at offsets 0/-1/-2 against live replays): **no single offset exceeds
  ~0.67 parity; -2 beats -1 beats 0 for nearly every seed.** The effective
  delay is not a constant — it is jitter. Offline decision reconstruction
  is fuzzy BECAUSE the live system has no well-defined delay.
- The sustain mechanism (fat X-holds vs razor X-spikes at the jump-cancel,
  INIT_FORENSICS_OPTIONS.md) is plausibly jitter-robustness in disguise:
  razor-spike seeds fail when effective delay wobbles. Pre-registered
  test in flight (see Experiments).

## What slippi-ai does (surveyed 2026-07-28 — the architecture guide)

vladfi1's system independently converged on the full stack we sketched:

- **Synchronous frame-locked outer loop** with `blocking_input=True` —
  Dolphin WAITS for the bot's controller write each frame; the game
  cannot outrun inference (slippi_ai/dolphin.py, scripts/eval_two.py).
- **Delay is the load-bearing abstraction.** The policy is TRAINED to
  emit the action for frame t+D given the observation at t (D=18-21 for
  netplay; the imitation loss slices states/actions apart by D). At
  runtime D splits between Dolphin's own input buffer (`console_delay`:
  2 local, 15 netplay, max 24) and a local action queue.
- **"Async" = one worker thread hiding inside the delay budget.** The
  local budget remainder (policy.delay - console_delay) is called `headroom`; at
  zero it logs "No headroom, agent will effectively run synchronously."
  Async is a CONSUMER of declared budget, never unbudgeted decoupling.
- **Stateful streaming inference (O(1)/frame).** LSTM hidden state kept
  on device, one step per frame, never re-encode a window; training uses
  the same step function under scan, so train/deploy provably match.
- **Netplay:** donate nearly the whole budget to console delay
  (`D-1`, capped 24) so the human opponent feels local ping; rollback
  frames never reach the agent (libmelee skips resimulated frames).
- **Delay is a data-pipeline property, not architectural** — they retrofit
  a checkpoint's delay with a script (update_delay.py).
- Frame-budget watchdog: warn >12ms/frame, error >16ms.

Verdict for ExPhil: "timestamped vs synchronous" is a false dichotomy —
the target is a sync loop CONTAINING an explicit delay queue (C inside A),
delay-matched training (B), stateful streaming (D). E (palliatives:
reboots, CPU pinning, staleness discard thresholds) exists only because
layer 2 is unbudgeted; in the target architecture it disappears.

## Target architecture

```
train:  policy(observation_t) -> action_{t+D}   (D explicit, default 1 local)
deploy: sync frame-locked loop; delay queue holds in-flight actions;
        D split console/local exactly as trained; stateful O(1) step
netplay: same policy retrained/retrofitted to D=18+; budget donated to
        console_delay; rollback invisible
```

## Implemented so far (2026-07-28)

- [x] Sync runner revived: `scripts/play_dolphin.exs` — dummy-opponent
  passthrough, `--seconds` + SD-to-finalize, frame-skip stat (the sync
  analog of staleness), fixed a fatal top-level `@flag_groups` bug (the
  script had been broken since a refactor; nothing ran it).
- [x] `--runner sync` in `scripts/eval_live_protocol.sh` (same scoring
  pipeline for both loops; protocol.txt records the runner).
- [x] Offset calibration in `probe_cycle_margins.exs` (the evidence
  instrument for layer-2 jitter).
- [ ] In flight: sync smoke test -> jitter experiment (below).

## Experiments (pre-registered)

**Jitter experiment** (RAN 2026-07-28): eval razor-spike breakers
(b, d, k) and controls (a, n) under `--runner sync`, 3x60s vs idle.
Predictions and outcomes:
1. Frame-skip ~0 — **CONFIRMED** (3600/3600 frames, 0.0% skipped, every
   run; the loop is genuinely frame-locked).
2. Transition parity ~1.0 at a fixed offset — **REFUTED**: sync replays
   calibrate WORSE than async (max 0.371 flat across offsets 0..-4, vs
   async's clear structure peaking 0.666 at -2 and collapsing at -4).
3. Breakers improve — **REFUTED, inverted for everyone**:

   | seed | async self/min · chains | sync self/min · chains |
   |---|---|---|
   | a | 99-129 · 19-22 | 42-49 · 2 |
   | n | 114-141 · 14-18 | 44-48 · 2-3 |
   | k | 69-81 · 3-4 | 39-50 · 1-2 |
   | b | 68 · 2 | 37-39 · 1 |
   | d | 86-90 · 2 | 2-7 · 1 |

**Reading (2026-07-28): a uniform ~45/min chains-2 ceiling across wildly
different policies is a harness signature.** Two candidate causes, not
yet separated:
(i) constant sync delay differs from async's typical effective delay by
+1 (controller flushes on the NEXT step) and delay-0-trained policies
are brittle to the shift — the af 3-4 JC window is consistently missed;
(ii) the sync script's inference path (`Agent.get_controller`, the Agent
GenServer's internal window/prev bookkeeping) diverges from the async
runner's inference process in embedding/history construction — which
would also explain the WORSE offline parity. embed_path_parity_test is
prior art for exactly this class of bug.
**Either way the deep conclusion stands: every historical number is
calibrated to the async harness's particular delay distribution, and
policies are delay-brittle because delay was never a trained property.**
Next diagnostics, in order: (a) side-by-side embed/window dump of
Agent-path vs AsyncRunner-path on identical states; (b) if paths match,
train delay-1/delay-2 seeds and re-eval under sync (the original gate).
Defaults stay UNFLIPPED — the gates did their job.

**Delay-matched training** (RAN 2026-07-28, farm 7): champion recipe +
`--action-delay 1`, 3 seeds x {async+0, async+1, sync}. **GATE FAILED,
structurally**: all three seeds are METRONOMES (steady single shines,
chains 1, every condition — the zz phenotype 3/3, no seed spread):

| seed | async+0 | async+1 | sync |
|---|---|---|---|
| d1a | 39-49/min c1 | 46-47/min c1 | 6-33/min c1-2 |
| d1b | 49-50/min c1 | 47-49/min c1 | 3-43/min c1 |
| d1c | 26-41/min c1-2 | 39-41/min c1 | 2-4/min c1 |

Two informative fragments in the failure: (1) RATE became flat across
+0/+1 (delay-0 seeds lose rate under +1; delay-1 seeds don't) — the
single-shine behavior genuinely became delay-robust; only CHAINING died.
(2) 3/3 uniformity = the label shift makes chain-critical sequencing
unlearnable, not an init lottery.

**DIAGNOSED (same day, task 'diagnose-metronomes')**: (1) d1 seeds'
offset calibration peaks at -3 where every delay-0 seed peaks at -2 —
the training shifted temporal alignment by EXACTLY the trained delay;
the mechanism worked. (2) Their replays contain ZERO jc_events and ZERO
aerial_shine_events — the JC branch didn't degrade, it ceased to exist.
Mechanism: the .slp records APPLIED inputs, and applied[t] was decided
from state[t-1], so the fixture is ALREADY a one-frame-shifted dataset
(delay-0 training = matched to the live pipeline; z chains-27
consistent). Stacking --action-delay 1 demands two-frame anticipation:
the JC must be committed while seeing reflector af 1-2 — the same
states that carry B-hold labels — so chain-critical states collide,
the model averages, and only widely-spaced single shines survive.
Basin escape was unaffected (per-epoch probes: esc@2 by epoch 10) —
collisions only occur at two-frame-resolution decisions.

**Prime suspect — DOUBLE ANTICIPATION**: the closed-loop teacher already
compensates the bridge delay ("pressing at af 3 lands at af 4",
record_multishine.exs), so the fixture's (state, controller) pairs bake
in one frame of anticipation; `--action-delay 1` stacks a second.
Corollary: delay-0 training IS the match for the current async harness
(z chains-27 consistent), and adapting to a slower harness likely means
RE-RECORDING the fixture through a D-delay bridge, not shifting labels.
Second gap vs slippi-ai: their delay-trained policies SEE the in-flight
action queue as an input (delayed_actions); ours is blind to its own
committed inputs. Both threads tasked (diagnose-metronomes,
double-anticipation, in-flight-channel). Defaults gate remains CLOSED —
second bad flip caught in two days.

**Re-record experiment (RAN 2026-07-28, task 'double-anticipation')**:
shifted teacher (closed_loop_d1, triggers one frame earlier) through an
online_delay=1 bridge held ONE UNBROKEN CHAIN OF 787 (original: 791) —
teacher-level adaptation works perfectly. But ms_rerec_a (delay-0
training on the d1 fixture, full recipe):

| condition | self/min | chains |
|---|---|---|
| async+0 | 104-113 | 4-5 |
| async+1 (the "matched" one) | 80-83 | 2 |
| sync | 80-96 | 2 |

- **P3 CONFIRMED: sync ≈ async+1** (first direct estimate of sync's
  effective delay: one frame more than async).
- **P1 FAILED — and the margins explain why**: rerec_a has the fattest
  critical margins ever measured (jc +5.8, aerial +4.2, flip 0.0, first
  positive crit_p10_min) yet zero aerial_shine_events at +1 (empty_hop
  breaks). Same-frame (state_t, applied_t) training CANNOT express
  anticipation: it reproduces a stream lagging by the deployment delay
  regardless of the recording's delay — a different recording only
  changes which lagged copy you learn (crisply).

**THE SYNTHESIS (what actually confers delay-matching)**: slippi-ai's
scheme is label shift AND the in-flight action channel TOGETHER —
state_i paired with action_{i+D+1}, with action_{i+D} (the queued
action) as an INPUT. Farm 7 had the shift without the channel: the
state collisions it died of (JC-commit vs B-hold on the same af 1-2
states) are exactly what the queued-action input disambiguates. The
re-record had neither. Neither half works alone; the in-flight channel
(task 3) is not an enhancement — it is the missing half of delay
adaptation. The shifted teacher + d1 fixture remain valuable as the
matched-EVAL ground truth for that work.

## Defaults migration plan (NOT yet flipped — gates first)

| Default | Current | Target | Gate |
|---|---|---|---|
| eval protocol `--runner` | async | sync | jitter experiment passes predictions 1-2; one full seed-eval block runs clean |
| training `--action-delay` | 0 | 1 | a delay-1 seed matches/beats delay-0 seeds under sync eval (3 seeds, protocol rules) |
| inference path | window re-encode | stateful O(1) step | Edifice.Stateful parity test vs window path on fixture replay (bit-identical logits) |
| staleness discard rule | >10% discard | replaced by skip-stat rule | sync default lands |

Comparability warning: the day defaults flip, chain/rate tables start a
NEW baseline era — do not compare across the flip without noting it
(0a/0b rules). Old-recipe numbers (seeds a-s) are async/delay-0 forever.

## Direction / open items

1. Validate + flip defaults per gates above.
2. Stateful streaming step path for the GRU policy (exists in Edifice;
   needs the parity test + agent wiring). 16x less per-frame compute.
3. Explicit delay queue in the bridge (the C component) — needed the day
   models outgrow the frame budget or D > 1: apply action at its target
   frame, split budget console/local like slippi-ai.
4. blocking-input equivalent for our bridge (Dolphin waits for the bot)
   — investigate libmelee/Slippi ini support; this is what makes sync
   robust to inference spikes rather than frame-skipping through them.
5. Netplay track: delay retrofit tooling (slippi-ai's update_delay.py
   pattern), --online-robust delay-distribution training, displayName
   port discovery.
