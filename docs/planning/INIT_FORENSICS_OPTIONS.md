# Contrastive init forensics — options ledger

Created 2026-07-27 evening. The question: on IDENTICAL data (crouch recipe,
data seed 42), why do 6/12 inits learn absorber escape and 6/12 stay
crouch-absorbable — and what does ms_crouch_a's init learn that nothing
else found? Assets: 12 checkpoints cleanly split 6/6 by live outcome
(EXPOSURE_BIAS item 8 table), one extreme failure (g: absorbs at f104,
never shines), one oscillator (l), one champion (a: chains 19-22).

Check items off / append findings as they land. Status legend:
[ ] not started · [~] in progress · [x] done (see Findings).

## The options

- [x] **1. Decision-boundary cartography** (behavioral probing on
  constructed states). Policy = function; measure it directly on the
  basin manifold: SquatWait af 1..40 windows, prev-action threaded both
  ways (train-style alternating vs live-style constant-no-B), read
  P(B) off the buttons head per seed. Resolves underfit-vs-deployment-
  dynamics, measures membrane thickness, gives the scoring metric every
  other option needs. Script: `scripts/probe_crouch_boundary.exs`.

- [x] **1b. T-threshold calibration** (rides on 1). Live decode is
  `sigmoid(logit_B / T)` Bernoulli per frame (agent.ex sample_buttons —
  NOTE: Policy.Sampling.sample_buttons is a temperature-LESS older path;
  the agent path scales). From measured basin logits, predict rescue
  rate vs T and test against item 9's flat curve (0.3/0.5/1.0 all
  ~60-81/min). If the numbers land, the absorbing-state theory becomes
  quantitative.

- [ ] **2. Input attribution** (gradient x input, `ExPhil.Interp.
  Attribution`). For the B-decision in basin states, which input groups
  carry it: prev-action (self-reinforcing copy loop = failure story) vs
  own action state/af (expert rule = escape story). Needs a custom
  objective (B logit, not argmax logit). Hypothesis-DRIVEN saliency only.

- [ ] **3. Linear probes on trunk activations** (representation vs use).
  Can a linear readout decode "basin state" / "af parity" / "B should
  fire" from each seed's trunk? Failure with intact representation =>
  decision-layer problem; collapsed representation => upstream problem.
  Must beat the random-init-trunk control (INTERP_ROADMAP rule).
  Edifice `linear_probe` gets its first validated harness here.

- [ ] **4. Causal interventions** (steering/erasure; `steering.ex`,
  `erase.ex`/LEACE). Erase the putative crouch direction from a failed
  seed's trunk, or steer along the escapee-vs-failure activation
  difference, and watch the B logit. Within-model only — cross-seed
  activation patching needs basis alignment (independently trained nets
  don't share coordinates).

- [x] **5. Loss-landscape geometry / mode connectivity.** Interpolate
  weights a<->g (and escapee<->escapee as control), score each point
  with probe 1's escape metric. Are escape and absorbed solutions
  separate basins with a barrier, or one connected valley? Caveat known
  in advance: naive linear paths between independent inits usually hit
  a barrier from permutation symmetry alone — the interesting contrast
  is a<->g vs the a<->c/i/k controls, not any single curve.
  Script: `scripts/interp_mode_connectivity.exs`.

- [ ] **6. Training-dynamics forensics** (watch the lottery get
  decided). Per-epoch probe-1 measurements during fresh training runs:
  when does escape competence emerge (or fail to)? Long-term payoff:
  an early-reject test at ~epoch 5-10 that converts the 50% seed yield
  into ~100% effective yield and replaces live evals on the escape
  axis. Hook: `--probe-crouch` flag in train_multishine_policy.exs
  writing JSONL per epoch.

- [ ] **7. Data attribution / subset-loss forensics.** Per-seed training
  loss restricted to the crouch-synth slice. High for failures =>
  optimization never fit the escape labels; low => the labels are fit
  and the failure is closed-loop shift. The afternoon-sized version of
  influence functions.

## Named anti-patterns (what NOT to do)

- **Weight-statistics numerology**: comparing raw norms/spectra between
  seeds with no causal link to behavior. Produces confident-sounding
  numbers that constrain nothing.
- **Hypothesis-free saliency heatmaps**: rendering |grad x input| over
  inputs and narrating the pretty picture post-hoc. Saliency enters only
  as a two-class contrast against a stated hypothesis (option 2).

## Priorities (user, 2026-07-27)

Start: #5 (interesting) using #1 as the metric. Pipeline #1 -> #6 (most
useful long-term). #1b if achievable. #2 next. Edifice payoff target:
the capture/probe/patch harness pieces generalize upstream.

## Findings

### 2026-07-27 late session (options 1, 1b, 5 executed; taxonomy landed)

**THE MECHANISM: the crouch absorber is a HELD-B fixed point.** Probe v2
(`probe_replay_basin.exs`, 93.9% offline/live parity on seed g's replay)
shows g pressing B on 100% of its 3425 absorbed basin frames (mean logit
+0.35..+0.45). Melee registers button EDGES; held B is a no-op. The
failure is not "never press" — it is "never RELEASE." The expert labels
alternate B every frame precisely for the edge; a seed that fails to
learn the alternation's conditioning collapses to the label MEAN
("press, weakly") = behaviorally identical to never pressing.

**Fixed-point taxonomy of the 6 failures** (release-conditioning metric,
`probe_crouch_boundary.exs` live_held variant):
- hold-B absorber (output B given prev=held-B): g, h, j — the three
  deadest seeds live.
- silent absorber (output no-B given prev=no-B): e, f.
- NO fixed point (press on no-B, release on held): l — forced 2-cycle,
  matching its live oscillation (repeated ~300f spells with escapes).

**1b CLOSED — flat T curve explained quantitatively.** Rescue = random
RELEASE: p(release) = 1 - sigmoid(logit/T) with basin logits ~+0.35
gives 0.24 / 0.33 / 0.41 at T = 0.3 / 0.5 / 1.0 — all fast-escape,
hence the flat dose-response. Had the basin logit been -3, the curve
would be strongly T-dependent. The near-boundary logits ARE the
saturation.

**On-manifold probing is uninformative about live outcome** (chance:
6/12 via `analyze_boundary_map.exs` closed-loop button sim). Coverage
worked where it covered — every seed answers the training-style block
acceptably. Seeds diverge on SELF-GENERATED 16-frame histories.

**Mental rollout (v3, `probe_basin_rollout.exs`) — the real instrument.**
Simulates the basin closed loop in embedding space (policy's own output
fed back as prev, window evolving; escape = B edge). Three tiers:
- universal escapers: a, c, i — escape even from OTHER seeds' real
  absorbed entries (g@104, e@196, h@454);
- self-consistent escapers: b, d, k — live-escape via their own routes
  but cannot exit foreign holes;
- absorbable: e, f, g, h, j, l.
Real-entry prediction 9/12; the 3 misses are exactly tier-2. Full
prediction needs rollouts from each seed's OWN entry routes (start from
cycle states + perturbation — next step; also the option-6 early-reject
candidate: run N mental rollouts per epoch, escape-fraction is the
metric).

**Option 5 DONE — escape and absorbed are separate loss basins.**
`interp_mode_connectivity.exs`, rollout-scored:
- a<->g: barrier by alpha=0.1-0.2; ENTIRE interior absorbed on both
  entries (interior worse than both endpoints — permutation-mismatch
  barrier + genuinely different solutions).
- a<->c control: synthetic-entry escape survives at EVERY interior
  point; only foreign-hole rescue flickers. Escapers are mutually
  near-connected; escaper<->failure is not.

**ACTIONABLE TRAINING BUG FOUND (--dump-prev): synthetic crouch tails
train with the prev-action channel ABSENT.** `build_crouch` tails reuse
the source frame NUMBER; `precompute_frame_embeddings` threads prev only
when frame numbers are consecutive -> the tails embed prev as absent.
The release-conditioning signal (release-when-prev-B) was therefore
INVISIBLE in exactly the states it matters; seeds that learned escape
inferred it from af parity by init luck. Fix: renumber tail frames
consecutively in RecoverySynth (one line) + thread lead-in boundary.
Prediction to test (3+ seeds): escape rate rises well above 6/12, and
the release-conditioning gap becomes uniformly large. THIS IS THE NEXT
INTERVENTION.

Status updates: 1 [x] (v1 chance-level as classifier but source of the
conditioning metric; v2/v3 are the instruments), 1b [x], 5 [x],
2 [ ] (saliency contrast now well-posed: does the B logit's gradient
weight prev-action dims for escapers vs af dims?), 6 [ ] (design ready:
per-epoch mental-rollout escape fraction).
