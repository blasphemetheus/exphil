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

**Jitter experiment** (running): eval razor-spike breakers (b, d, k) and
controls (a, n) under `--runner sync`, 3x60s vs idle. Predictions:
1. Frame-skip stat ~0 (sync loop keeps up; GRU inference is ~ms).
2. Margin-probe transition parity on sync replays jumps to ~1.0 at a
   single fixed offset (the instrument becomes exact).
3. THE decisive one: if b/d/k chains rise materially under deterministic
   delay, part of "sustain seed variance" was harness jitter, and every
   historical chain number gets an asterisk.

**Delay-matched training** (after sync validates): train 3 seeds with
`--action-delay 1`, eval under sync. Gate for the defaults flip below.

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
