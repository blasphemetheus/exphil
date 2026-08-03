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

**Channel analysis revision (2026-07-28 evening, task 'in-flight
channel', from reading the agent code)**: at D=1 the in-flight channel
ALREADY EXISTS with aligned semantics — training prev-action threads
the SHIFTED stream (shift_actions runs before precompute), making
channel[t] = label of t-1; the live agent feeds its own previous decode
= exactly the same thing. The prev channel IS the full one-item queue
at D=1. New plumbing is only required at D >= 2 (netplay). This
retracts the "farm 7 died for lack of the channel" story and reopens
the metronome cause. NEW PRIME SUSPECT: --x-hold-extend widens X at
UNSHIFTED positions and shift_actions then moves labels underneath
(590 widened frames -> X=true on hold-phase states = the collision,
caused by transform ORDERING, not architecture). Farm 8 (running,
pre-registered): arm A = shift WITHOUT x-hold (prediction: chains under
async+1 — the matched-training gate finally passing); arm B = shift+SS
0.5 (channel exposure robustness). If arm A passes, the transform-order
fix is: apply x-hold-extend AFTER shift_actions, or key it on shifted
positions.

**Farm 8 (2026-07-28 evening): BOTH remaining hypotheses refuted.**
Arm A (shift, no x-hold): metronomes, chains 1, both harnesses — x-hold
ordering was NOT the cause. Arm B (shift + SS 0.5): metronomes — channel
exposure bias was NOT the cause. Running tally: SIX shift-trained seeds,
six metronomes, all memorizing to loss ~0.002, none with an optimum at
ANY harness delay. The label shift kills closed-loop chaining through a
mechanism that has now survived five falsified explanations (missing
channel, x-hold interaction, exposure bias, harness arithmetic,
collision-by-collision). ms_shift_a's calibration peaks at -1 — the
shift moves alignment FRESHER, not more anticipatory. OPEN PROBLEM;
next instrument: extend BasinRollout to a full-cycle offline simulator
and watch a shift seed drop the loop frame by frame.
**Simulator BUILT 2026-08-02 (ExPhil.Interp.CycleSim +
scripts/cycle_sim.exs), gate NOT yet passed.** Dynamics = transition
graph {action, af, b_edge, x_edge} -> {action', af'} extracted from
fixture + --graph-replays, with af-tolerant lookup (±4), edge-drop
fallback (both counted as :soft), per-state statistical reconstruction
of y + speeds, phase-labeled hard breaks, ShineChain chain scoring.
Pre-registered gate (champion z chains, metronome zz doesn't) FAILS:
both sim as endless singles — z never aerial-shines in sim (35-frame
airborne wait), through four fixes (edge keying, graph enrichment,
y integration, absolute y+speed reconstruction, af tolerance). Next
suspect list: dump z's buttons-head logits over the sim's airborne
windows vs matched live-replay windows and diff the embedding fields
(embed_path_parity_test is prior art); stick-decode fidelity;
window-boundary semantics. Usable TODAY for break forensics (hard
breaks are exact); NOT yet for chain ranking.

**Strategic position after the day**: async+0 + delay-0 training is a
validated, excellent production combination (z: 147/min chains 27);
sync's delay is pinned at +1; no policy-adaptation path to +1 exists
yet (shift and re-record both fail differently). Most promising
untried direction for +1/netplay: DAGGER THROUGH THE DELAYED BRIDGE —
the shifted teacher demonstrably masters the +1 harness (787-chain),
so dagger_drill against it UNDER async+1 yields exactly-matched
(state, expert-correction) pairs with genuine closed-loop coverage,
bypassing label surgery entirely. The D>=2 queue channel remains
future work gated on netplay ambitions.

## Delay-id patch probe (2026-08-02): jq collapse SOLVED, d2 inversion is dynamics-side

New interp flag `--delay-id-override N` (cli.ex + both play scripts):
forces the delay-id one-hot regardless of --frame-delay. Six 1-run
screening evals (sync headless, stand, grind protocol;
eval_runs/0802_delayid_probe.sh + .log):

| run | self/min | maxchain | baseline |
|---|---|---|---|
| jq d3 ctrl | 0 | 0 | 0 x4 (reproduced) |
| jq d3, id=2 | 72.9 | 1 | — |
| jq d3, id=4 | **431.4** | **430** | jq @ real d4: 101 c3 |
| mdq_ss d2 ctrl | 139.8 | 6 | 139.8 c6 (EXACT repro) |
| mdq_ss d2, id=3 | 70.9 | 1 | — |
| mdq_ss d3, id=2 | 214.7 | 174 | mdq_ss @ d3 id=3: 380.5 c367 |

**Verdicts:**
- **jq d3 collapse: CHANNEL-TRIGGERED, confirmed.** True delay held at 3,
  patching only the id rescues shining completely — id=4 produces a
  431/min whole-run chain, BETTER than jq at any real rung. The crouch
  absorber entry at d3 was conditioned by the id one-hot, not the delay
  dynamics. jq's behavior is dominated by the id channel (0 → 431 on a
  one-hot flip).
- **mdq_ss d2 inversion: NOT channel-driven — prereg refuted.** Forcing
  the good rung's id (3) at d2 makes it WORSE (71 c1 vs 140 c6): id and
  dynamics must agree; mismatch always costs. The d2 weakness is
  dynamics/data-side — the handoff's suspect (jitterless d2 sources
  interacting with SS) stands. Note the SS asymmetry: mdq_ss keeps
  c174 under a wrong id at d3 (robust), jq flips 0↔431 (brittle) —
  SS-on-queue also buys id-mismatch robustness.
- Eval determinism reconfirmed: control reproduced 139.8 c6 exactly.

**Grind-3 (2026-08-02 evening): jitter × SS REFUTED — jq_ss flattens
every rung.** Arm jq_ss = mdq_ss recipe + --shift-jitter 1
(eval_runs/0802_grind3_jqss.{sh,log}; checkpoint ms_g3_jq_ss.bin,
loss 0.039 vs champion 0.008): d2 84.9 c1 / d3 75.9 c3 / d4 79.9 c1.
All three preregs fail — jitter doesn't fix d2, kills the d3/d4
records, and hurts d2 too. Mechanism: per-source jitter makes identical
states carry differently-shifted labels; SS-on-queue then self-samples
against inconsistent targets (the 5x loss floor is that
irreducibility). R3's accidental smear predates the queue channel — its
benefit does NOT transfer to queue-SS recipes. Rule: NO shift-jitter in
SS-on-queue recipes. The d2 inversion's remaining suspect is the SOURCE
DISTRIBUTION (every rollout pool was collected at d3); next probe =
collect a d2 rollout pool (champion through --frame-delay 2, qtrace
protocol) and retrain mdq_ss with mixed-rung sources.

**Grind-4 (2026-08-02 night): source distribution ALSO refuted — the d2
inversion goes to the interp bench.** 12 temperature-0.4 d2-native
rollouts (eval_runs/0802_d2pool) mixed into the mdq_ss recipe
(ms_g4_d2mix.bin, --resume through a couch-game interruption):
d2 **138.8 c3** (vs champion 139.8 c6 — unchanged), d3 **424.4 c423 —
NEW d3 RECORD** (beats 380.5 c367), d4 105.8 c33 (degraded from 435).
Verdicts: (a) d2 is PINNED at ~139-140/min chains<=6 across four
different recipes — a metronome cadence (1 shine/~26f) too stable to be
a data artifact; prime structural suspect = the farm-7/8 shift-collision
mechanism (JC-commit vs B-hold labels colliding on shared states) biting
at shift 4 (=d2+2) but not 5/6. Next instruments: probe_cycle_boundary
per rung on the same checkpoint; CycleSim once its gate passes.
(b) Pool composition is a PER-RUNG trade: the d2/temp pool boosted d3 to
the record while halving d4 — g4_d2mix is a d3 SPECIALIST; mdq_ss stays
the all-round champion; per-target-rung pool tuning is now a real axis.

**Grind-5 (2026-08-03, ms_g5_ladder8, multi-delay {2,4,6,8}):** d2
**205.8 c73** / d4 90.8 c2 / d6 80.8 c1 / d8 87.8 c1
(eval_runs/0803_g5_rerun.log; the 00:37 sweep died to GOTCHA #83 first).
Two verdicts: (a) **SS-on-queue does NOT ladder past d4** — d6/d8 are
metronomes; the d<=8 netplay plan leans on d4 (enough for good
connections: 2-frame Slippi buffer + intrinsic 2) until a new idea.
(b) **THE d2 PIN SHATTERED**: four recipes with shift-set {4,5,6} pinned
d2 at ~140 c<=6; this arm's only structural change is RUNG SPACING
({4,6,8,10} — 2 apart) and d2 jumped +50%/12x-chains. Refined #18
hypothesis: ADJACENT-shift rungs interfere (1-frame-apart label sets
collide on shared states; 2-frame spacing dodges it). Decisive next
arm: multi-delay "2,4" only (spacing 2, minimal pool) vs "2,3" (spacing
1) — if 2,4 keeps d2 >= 200 and 2,3 re-pins it at 140, the mechanism is
confirmed causally and the champion recipe becomes spacing-2.



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
2. Stateful streaming step path for the GRU policy — LANDED 2026-07-29:
   fixture-level bit-parity gate
   (test/exphil/networks/stateful_fixture_parity_test.exs) pins the step
   path against the windowed forward on real embedded multishine windows
   at deployment dims, GRU+LSTM, rolling offsets, tol 1e-5 (fp noise
   measures ~5e-7; structural bugs ~1.0). Informational drift probe:
   carried-state vs sliding-window logits differ by ~0.04 after 40 frames
   (random init) — the truncation semantics are REAL, which is why the
   agent's cold-start pad replication and per-game reset matter.
   LIVE BLOCK 2026-07-28 (eval_runs/stateful_arm vs windowed_arm, 3+3
   runs, async+0, cpu-1, 60s — REGIME-CAVEATED: uptime 7h19, load ~3.5,
   uptime check skipped before the block; the windowed arm itself sat
   ~2x below champion (68.2 mean vs 121-147), so absolute numbers are
   not champion-comparable):
   - windowed 63.4/72.6/68.5 self/min chains 4/5/5 stale 1.4-1.6%
   - stateful 57.4/53.4/37.3 self/min chains 3/4/3 stale 0.5-1.0%
   - Arm delta 1.38x = UNRESOLVED by the <2x rule, but suggestive:
     arms don't overlap (stateful best 57.4 < windowed worst 63.4),
     chains consistently one lower. Consistent with carried-state
     divergence: the GRU is TRAINED as sliding h0+16-frame windows;
     stateful deploy carries state all game — a function it was never
     trained as (the fixture drift probe's 0.04 gap, compounded).
   - CONFIRMED win: O(1) compute is real — staleness 0.5-1.0% vs
     1.4-1.6%, ~1.4x inference rate, on a loaded laptop CPU.
   - Decision: --stateful-step stays OPT-IN. If the gap ever needs
     resolving: (a) clean-regime rerun, (b) hybrid deploy (re-encode
     the window every k frames to re-sync state, step between — keeps
     most of the compute win with bounded drift), (c) slippi-ai's real
     fix: train the unroll = deploy step.
3. Explicit delay queue in the bridge (the C component) — needed the day
   models outgrow the frame budget or D > 1: apply action at its target
   frame, split budget console/local like slippi-ai.
4. blocking-input + polling — LANDED 2026-07-29. blocking_input was
   already plumbed (default: headless only); `--blocking-input` now
   forces it for windowed sync evals. Polling mode
   (Console(polling_mode/polling_timeout), config console_timeout,
   default 0.1s) makes console.step return no-frame instead of blocking
   forever; MeleePort absorbs no_frame transparently for legacy callers
   (poll: false) and surfaces :no_frame to LRAS-aware runners
   (poll: true). LRAS is UNGATED: @lras_frames 120 in async_runner +
   lras_frames 120 in play_dolphin.exs, L+R+A held with Start PULSED
   (toggle per tick — each toggle rides its own console.step flush, so a
   fresh Start edge reaches the pause screen every other cycle; a
   continuous hold never re-edges). Past the LRAS window a paused game
   gets a Start-only pulse to unpause into the hold-left walk-off.
   VALIDATED 2026-07-28 (eval_runs/lras_smoke1, blocking_sync1,
   blocking_sync_stand):
   - LRAS: instant quit on the FIRST chord frame in all 5 games (async
     smoke + 3 sync + 1 stand) — `[SD] f1 phase=LRAS` then
     CHARACTER_SELECT, replay finalized, no pause ever occurred (the
     no-frame pulse machinery is an unexercised safety net so far).
     Contrast 2026-07-28 pre-fix: pause, one [SD] frame, stall.
   - Skip gate PASSED 3/3: sync + --blocking-input = 3600/3600 game
     frames, skipped 0 (0.0%), every run.
   - Offset calibration: the jitter smear is GONE — single dominant peak,
     reproducible ±0.02 across runs (cal 0 => 0.74-0.77, -1 => ~0.51,
     rest <=0.26). But the pre-registered ">0.9" FAILED: peak is
     0.71-0.77, and a stand-dummy run (no CPU interference) came in at
     0.714 — interference falsified as the explanation. The ~0.75
     ceiling is intrinsic to the parity measure at this offset (policy's
     own transition-frame consistency), not harness jitter. Note also
     the peak offset MOVED: async delay-0 fingerprint was -2; blocking
     sync is 0. Offsets are harness-specific — do not compare across.
5. Netplay track: delay retrofit tooling (slippi-ai's update_delay.py
   pattern), --online-robust delay-distribution training, displayName
   port discovery.

## The dummy artifact + headless trust campaign (2026-07-28 late)

**The ~70/min "collapse" was a comparability artifact.** Champion
numbers (121-147, chains 16-27) were STAND-dummy (0728_open_z_idle);
every post-polling block defaulted to cpu-1, whose jabs cap chains.
Post-reboot stand-dummy windowed: **165-186 self/min, chains 15-23**
(prereg CONFIRMED) — new all-time rate records THROUGH the polling
harness. Windowed + --blocking-input: 170-182, **chain 30 all-time
record** — blocking input costs nothing windowed. Polling A/B (cpu):
~70 vs 53 (n=1) — polling exonerated. Failed preregs #7-#9 along the
way: fresh-regime recovery (wrong — dummy), headless single-clock
recovery, send-late recovery (both wrong — see below).

**Headless trust ladder** (stand dummy, all 3-run blocks, mainline
beta AppImage `netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage`):

| config                        | self/min  | chains | cal peak |
|-------------------------------|-----------|--------|----------|
| windowed async (either input) | 165-186   | 15-30  | 0.78 @-3 |
| headless speed1.0 pace-hz 0   | 122-163   | 10-16  |          |
| headless speed1.0 pace-hz 60  | 118-138   | 7-12   | 0.67 @-2 |
| headless speed0  pace-hz 60   | 74-119    | 2-10   | 0.51 flat|

**Finding: chain capability tracks offset-calibration SHARPNESS**
(probe_cycle_margins cal spread) — the harness-quality metric. Speed-0
doesn't shift the effective delay, it SMEARS it per-frame (which game
frame an input lands in depends on userspace write-arrival timing when
the emulator is unthrottled); smear = variable jitter = the one thing
the one-frame-brittle policy cannot survive. Neither a drift-free
sleep+spin pacer nor pace-before-send reordering recovers it (both
landed and kept — they are correct engineering and inert in the
recommended configs; the collapse is speed-0-specific).

**Operational verdicts:**
- Timing-critical eval: windowed async, stand dummy for capability
  numbers, cpu-1 only for its own comparison lineage. NEVER compare
  across dummies (the eval opponent is part of the distribution).
- Best headless recipe today: `--headless --pace-hz 0` (speed 1.0,
  emulator throttle as the single clock): 122-163, chains 10-16 —
  usable where windowed isn't, NOT yet windowed-equivalent. Trust gate
  for full equivalence: cal peak >=0.75 concentrated at one offset.
- `--emulation-speed 0` = menus at max fps (game-start overhead
  ~2min -> ~12s/run, block wall time halved) but gameplay offset
  smear — do NOT use for timing-critical runs until a runtime
  speed-switch (menus 0, gameplay 1.0) exists. That switch is the
  next headless lever.
  **SUPERSEDED 2026-08-02 (tasks #9/#10, eval_runs/0802_wvh_gap.\*)**:
  with `--blocking-input`, `--emulation-speed 0` now scores the FULL
  record (380.5 c367, 3/3 identical, ms_g2_mdq_ss @ d3) at HALF the
  block wall time (1:55 vs 3:52 per 3-run block) — the July smear
  verdict predated the blocking-input fixes. New fast headless recipe:
  `--headless --emulation-speed 0 --blocking-input`. No runtime
  speed-switch needed.
  **And the windowed-vs-headless "gap" DID NOT REPRODUCE**: windowed =
  headless = 380.5 c367 (3/3 each, deterministic) on an idle unlocked
  desktop. The 08-01 "windowed 105 c19" (locktest) was a REGIME
  artifact — screen lock + DPMS-off and/or ambient load, not
  windowed-ness (tonight's only dip was a HEADLESS run, 113.8 c17, in
  the block whose 5-min loadavg still carried grind-3's tail).
  Follow-up: lock-state A/B (same block, locked vs unlocked screen) to
  pin which ingredient degrades — that's the regime live netplay
  sessions must avoid.
  **A/B RAN same night (task #17, eval_runs/0802_{lockarm,loadarm}.\*):
  ALL REGIME SUSPECTS EXONERATED** — windowed under active lock +
  DPMS-off: 380.5 c367 x3; windowed under 12 busy-loop CPU saturation:
  380.5 c367 x3. Remaining explanation, by timeline: the 08-01 locktest
  ran BEFORE `2bd9577` (one decision per game frame, landed later that
  day) — a non-blocking windowed sync loop re-infers when inference
  outpaces frames, corrupting the queue slots of a queue-depth-4 policy;
  headless blocking dispatch never re-infers, which is why only windowed
  suffered. Tonight's clean windowed runs ARE the regression test for
  that fix. Consequence: the "live behaves like windowed-degraded" fear
  is retired on current code — windowed/local-live = record pace. Eval
  determinism note: 10 consecutive windowed/lock/load runs tonight
  produced literally identical scoreboards (380.5 c367).
  **Ops cost of the lock arm (Bradley, same night): hyprlock ended in a
  FAILED state** after `loginctl lock-session` + windowed Dolphin under
  lock + `hyprctl dpms off/on` — recovery = `lockfix` from another tty.
  So the lock+window-map hazard is REAL but aims at the session/lock
  stack, not bot performance. Rule: don't run windowed evals under an
  active lock unless someone can reach a tty; headless (record-equal,
  0802) is the unattended default.
  Determinism corollary for COLLECTION runs: deterministic decode makes
  N rollouts N identical replays — pool collection must pass
  --temperature (0.4 used for the 0802 d2 pool; caught after training
  briefly started on 12 copies of one game).

## DAgger through the delayed bridge (2026-07-29, campaign live)

The +1-adaptation path the delay campaign queued. Protocol per round:
student collects through `--frame-delay 1` (windowed, stand dummy);
`dagger_drill.exs --expert multishine --fixture fox_multishine_closed_d1.slp`
builds the SHIFTED teacher table from the d1 fixture automatically
(MultishineExpert.from_frames keys {action, af, grounded} — the
fixture's pairs ARE the one-frame-early triggers) and relabels every
visited frame; retrain; eval at delay-1. Farm-5 label rule satisfied
by construction. Caveat: expert RECOVERY RULES remain delay-0-flavored
(table covers the happy path; revisit if break forensics point there).

**Round 0** (ms_open_z through delay-1): 113.6-115.4 self/min, chains
3-5 — the one-frame collapse reproduced under the polling harness.
Bridge fingerprint: stable bimodal -2/-3 (0.65-0.70, ±0.02 across
runs) — NOT a smear; the reduced peak vs delay-0 (0.78) is
policy-harness mismatch, i.e. the trainable part. Gate passed.

**Round 1** (17.5k frames: d1 fixture + 3 relabeled rollouts, GRU w16
prev-action, converged 0.0006 @ epoch 77, 46 min laptop CPU):
- delay-1: 35-75/min, chains 2-3. delay-0: 11-44/min, chains 1-2.
- **DELAY-PREFERENCE INVERSION: the policy scores BETTER through the
  delayed bridge than without it** — first learned policy in the
  campaign to do so. Re-recording never transferred this (rerec_a:
  fattest margins, zero aerial shines at +1). The DAgger ingredients
  (student-visited states + real presses in the prev channel) carry
  the anticipation signal BC-through-rerec could not.
- Capability thin: cal peak 0.52 flat-ish, aerial-shine flip 27-31%
  (the chain-critical B-press is marginal) — data thinness, 3 rollouts.
- Prereg "chains beat 3-5" FAILED on absolute numbers; the inversion
  was not pre-registered and is the real result.

**Round 2**: 12 aggregated rollouts. Rate RECOVERED (120-123/min) but
chains 2-4 — prereg >=8 FAILED. Forensics: every break unforced;
post-break dwell in reflector states (366/368, 20s spells) — the
switch decisions (JC out of shine, airborne B) are the weak joints.

**Round 3**: anti-copycat package (--transition-weight 2.0
--prev-action-dropout 0.6, the R17A pairing). Converged in the SAME
12 epochs (soft-falsifies the copy-shortcut diagnostic — a drill
state space with ~hundreds of unique {action,af} keys memorizes fast
regardless of weighting; convergence speed is not diagnostic here).
Eval pending mains.

**Mechanistic probe (interp_d1_timing.exs)**: encode-horizon curves —
balanced accuracy of "X-edge within k frames" probes, champion vs
dagger3 trunk vs raw-embedding floor, identical 6-replay mix. Prereg
(dagger holds intent at longer leads) FAILED: champion ~0.916 flat,
dagger ~0.905 flat, floor 0.82->0.88. No representational timing
shift; both trunks add ~0.09 of press-timing info over the floor.
NARROWS the mechanism: the d1 teacher's rules are the same features
with SHIFTED THRESHOLDS, and a threshold shift needs only the HEADS.

**Boundary maps (probe_cycle_boundary.exs, offline)**: mean B/X logits
by {cycle phase, af} over both teacher fixtures settle it:
- champion: textbook d0 boundary — X-trigger at ground-reflector af3
  (the d0 JC window), margins ±2-5 everywhere.
- dagger3: boundary SHIFTED EARLIER (X positive from af0-1, peak af1;
  jumpsquat B moved to af1-2 where champion releases — pressing B in
  jumpsquat so it lands on the first airborne frame AFTER the +1
  delay) but margins THIN (0.3-0.9) — right place, barely held. The
  anticipation is real and printed in the logits; thin margins are
  the chains-2-4 / 30%-flip story.
- head/trunk swap hybrids: DEGENERATE (champT+d1H mashes B at +4
  logits; d1T+champH inert). Lesson recorded: independently-trained
  trunks have unaligned feature bases — head-swap surgery is only
  valid within a shared-trunk fine-tune lineage. Mains hybrid evals
  cancelled. The correct head-level test: freeze the champion trunk,
  fine-tune HEADS ONLY on d1 labels; if the delay preference
  reproduces, adaptation is head-implementable (fast, composable).
- Round-4 lever implied by thin margins: the drill's loss target says
  "memorized" while margins say "barely" — train past convergence /
  lower --target-loss / margin-aware objective to fatten the held
  boundary before more data.
  **SHIPPED 2026-08-02 (task #4)**: train_multishine_policy.exs
  `--select-by margin --post-converge N` — keeps training N epochs past
  the loss bar and exports the probe epoch with max crit_p10_min
  (CycleMargins on fixture edges). Smoke confirmed the premise: loss
  converged by ~ep12 while crit_p10_min kept fattening 1.25 -> 2.48
  through ep20. A margin-aware LOSS term remains unbuilt (escalate only
  if selection alone doesn't move live chains). Drill margin wiring
  still pending (embeddings sharded).
