# Mechanistic Interpretability Roadmap

**Status doc — revisit and check off milestones as they land.**
Created 2026-07-13. Owner: Bradley + Claude sessions.

## Strategic context

Decision (2026-07-13): pause time-expensive training loops and invest in interp
fundamentals first. A week of DAgger drill iteration produced a ~20-28% conversion
plateau where checkpoint quality is a NaN lottery, ranking policies takes 3+
Dolphin games (~25 min, noisy), and failure modes are diagnosed by eyeballing
gameplay. Interp attacks all three directly, and Melee is an unusually strong
interp venue: **complete ground-truth game state, cheap causal interventions
(replay counterfactuals, state edits), objectively scored behavior**. Every probe
or circuit claim can be validated against reality — which LLM interp can never do.
That makes this both bot infrastructure AND a platform for demonstrating things
about interpretability itself.

Division of labor:
- **edifice** (sibling project): architecture-agnostic interp *methods*. Already
  has: `linear_probe`, `das_probe`, `leace`, SAE family (`sparse_autoencoder`,
  `gated_sae`, `jump_relu_sae`, `batch_top_k_sae`, `matryoshka_sae`),
  `transcoder`, `crosscoder`, `cross_layer_transcoder`, `concept_bottleneck`
  (all in `lib/edifice/interpretability/`). These are model-builders; they need
  harnesses. Generic additions (activation capture API, patching utilities)
  belong here — the pitch: one interp API across 30+ backbone families.
- **exphil**: everything melee-specific — ground-truth feature dictionary,
  activation datasets from replays, probe↔performance validation, Dolphin-in-
  the-loop interventions, training observability.

## Assets on hand

**Checkpoint zoo** — ~10 GRU drill policies with measured conversion scores
spanning 0-28%, all same architecture (GRU, window 60, 2 layers, hidden 256, 288-dim
embeddings). This is a ready-made validation set: any internal metric claiming
to measure "policy quality" must rank these correctly.

| Checkpoint (`checkpoints/`) | Pooled conv % | Knockdowns | Export loss |
|---|---|---|---|
| mewtwo_combo_poolgrow_r1 | 27.9% | 43 | 0.225 |
| mewtwo_combo_daggerloop_20260711_215741_i1 | ~26% | 39 | 0.039 |
| mewtwo_combo_replicate215741 | 25.0% | 12 | 0.190 |
| mewtwo_combo_lr15 | 21.1% | 38 | 0.106 |
| mewtwo_combo_stopab_tl08 | 12.1% | 33 | 0.075 |
| mewtwo_combo_poolgrow_r3 | 11.1% | 9 | 0.397 |
| mewtwo_combo_daggerloop_20260712_035205_i2 | ~6% | 16 | 0.002 |
| mewtwo_combo_stopab_tl15 | 0% | 17 | 0.142 |
| mewtwo_combo_daggerloop_20260712_035205_i1 | 0% | 8 (1 probe) | 0.122 |
| mewtwo_combo_poolgrow_r2 | ~0% | 1 | 0.305 |
| mewtwo_combo_nanrobust | (pending) | | |

**Ground truth**: replay parsing already extracts action states, positions,
knockdowns, conversions (`trace_tech_chase.exs`); libmelee gives live state.
**Data**: 42-replay drill pool (811k frames) + 9 fresh probe replays.
**Note**: export loss visibly does NOT rank these checkpoints — that's the
motivating observation.

## Concepts, in Melee form

Reference glossary: each core interp concept, its concrete instantiation in
our bot (window-60 2-layer GRU, 256-d hidden state, six controller heads), and the
phase that operationalizes it.

**Features** — properties encoded as *directions* in activation space, not
individual neurons. "Opponent is in knockdown" = some direction **v** in the
256-d GRU state. Unlike language interp, we can verify against exact ground
truth. (P1)

**Superposition** — more features than dimensions, stored non-orthogonally
at the cost of interference. 256 dims must hold percents, distances, action
states for two characters, momentum, timing... Predicts crisp handling of
common drill situations and noisy encoding of rare compounds — the DAgger
distribution-shift story, told geometrically.

**Polysemanticity** — the neuron-level symptom: one unit fires for unrelated
things (GRU unit fires for both shield pressure and respawn invincibility).
Why "look at neurons" failed and dictionary learning exists.

**SAEs / dictionary learning** — re-express activations in a wide sparse
basis of (hopefully) monosemantic features. Edifice has 5 variants. Melee's
edge: score every learned feature against the ground-truth dictionary —
"feature 412 fires on 94% of tech-in-place frames" is exact and falsifiable,
which language-domain SAE evaluation never is. (P6)

**The epistemic ladder** — four methods, same feature, increasingly causal:
1. *Linear probe* (correlational): is it **known**? Present in activations ≠
   used by the heads — a policy can know the tech direction and not act on
   it, which would explain good-representation/0%-conversion checkpoints. (P1)
2. *Ablation* (necessity): is it **needed**? Zero the direction (or LEACE-erase
   the concept; ablate the prev-action input channel) and watch behavior. (P3)
3. *Activation patching* (sufficiency): is it **enough**? Transplant the
   tech-direction components from a "teched left" frame into a "teched right"
   run — does the bot chase left? Replays give thousands of matched
   counterfactual pairs, which language interp struggles to construct. (P4/P6)
4. *Steering* (control): can I **use** it? Add
   mean(state|aggressive) − mean(state|passive) at inference, measure
   knockdown-rate shift in Dolphin. (P6)

**Circuits** — a named algorithm: features connected through weights. Target:
(opp action-state dims) → knockdown detector → tech-direction feature
(appearing only post-visibility) → gates on the main-stick head. Each arrow
gets named by a patching experiment; the 6-head output keeps "what feeds
this head" tractable.

**Logit lens** — read intermediate state through the output heads early: run
the heads on the hidden state at each of the 60 window timesteps and watch
the decision form. Reacting policy: stick logits swing when the tech
animation becomes visible mid-window. Guessing policy: no swing. (P4)

**Attribution** (gradient×input / integrated gradients) — for one decision,
which input dims drove it? "70% of this jump logit comes from the
prev-action dims" is the mechanistic form of the jump-spam diagnosis. (P3)

**World models** — does the policy simulate or pattern-match? Probe the
hidden state for quantities in NO single frame's embedding but computable
over the window: remaining hitstun frames, time-since-knockdown, drift
toward ledge. If they probe out, the GRU integrates an opponent model. (P4)

**Feature formation** — features appear at particular training points,
sometimes abruptly. Probe checkpoints across epochs: does tech-direction
formation *precede* conversion-% gains? If so, probe curves become the
export-time selector — ship the checkpoint when features are ripe, not when
loss is low. (P2/P5)

One-line summary: probes ask *known?*, ablation asks *needed?*, patching
asks *enough?*, steering asks *usable?* — superposition/SAEs explain how 256
numbers hold a game; circuits explain what algorithm runs on them; and Melee
grades every rung against an exact answer key.

## Phase 0 — Foundations (everything depends on this)

Goal: get (activations, ground-truth features) pairs out of any exphil policy
on any replay. Effort: ~1-2 days. GPU: minutes.

- [x] **Activation capture** (2026-07-13): `Policy.build_temporal_trunk`
      (name-compatible with exports) + `ExPhil.Interp.Activations` — trunk
      output per decision frame via the training-identical embedding path.
      Remaining "better" tier: per-timestep states across the window (needed
      for P4's logit lens) + per-head logits (needed for P3 attribution).
- [x] **Ground-truth feature dictionary** (2026-07-13):
      `ExPhil.Interp.GroundTruth` — 15 features incl. tech_choice (entry
      state = choice) and next_kd_choice (60-frame lookahead). Gap: offstage
      hardcodes the FD edge; per-stage lookup before mixed-stage replays.
- [x] **Paired dataset builder** (2026-07-13): `Activations.capture/3` +
      `scripts/interp_capture.exs` → capture.bin with replay_index for
      by-replay splits. Gap: no (policy, replay)-keyed disk cache yet.
- [x] Verified NO-MIX compatible: capture is an inference-only beam; smoke
      test ran clean in the beam-free gap.

Success criterion MET: smoke test on poolgrow_r1 × one probe replay produced
{18042, 256} f32 + aligned labels in **6.9 s** (< 5 min bar). Base rates
sane and instantly diagnostic: own_airborne 66.8% (jump-spam quantified),
own_shielding 6.4% (shield-lock), opp_shielding 0% (tech dummy never
shields), opp_knockdown 2.9%.

## Phase 1 — Probe-based checkpoint selection (payoff #1: kill the probe-game lottery)

Goal: test the core premise — do internal representations predict behavioral
quality? Effort: ~1 day after Phase 0. GPU: minutes per checkpoint.

- [ ] Probing harness around `Edifice.Interpretability.LinearProbe`: train/eval
      probes on frozen activations with train/val split by replay (not by
      frame — frames within a game are correlated).
- [ ] Probe targets (start): "opponent in knockdown," "tech window open,"
      "opponent will tech left/right/in-place/miss" (the reaction feature).
- [ ] **THE validation experiment**: run identical probes across all ~10 zoo
      checkpoints. Correlate probe accuracy with known conversion %.
- [ ] Baseline controls: probe on raw embeddings (input-only skill floor) and
      on a random-init network (representation ≠ luck).

Success / decision gate: Spearman ρ ≥ ~0.7 between probe accuracy and
conversion rank → adopt probe score as the drill loop's checkpoint selector
(seconds of GPU replacing 25-min Dolphin probes; Dolphin stays as periodic
verification). Weak correlation is still a result — it means policies fail on
acting, not knowing, which redirects Phase 3.

**v1 RESULT (2026-07-13): suite SATURATED — targets leak from the input.**
All 10 checkpoints AND both controls probe at 0.95-0.98 mean balanced
accuracy; tech_choice = 1.000 everywhere including the raw-input floor. The
controls did their job: v1 targets (action-state-derived features) are
(near-)deterministic functions of embedding dims — the lifecycle IDs 183-201
are a contiguous range, linearly separable from the raw action-ID scalar.
Spearman −0.248 is meaningless against ceiling-saturated probes. Classic
probing pitfall (cf. Hewitt & Liang control tasks); ours caught it in one run.

**Open mystery worth chasing**: next_kd_choice (the dummy's FUTURE tech
choice) probes at 1.000 from a single raw frame, with only 2.5% of labeled
rows in currently-visible states, and the dummy's RNG confirmed unseeded
(--deterministic only affects policy sampling). Some channel in the current
frame predicts the "random" choice well before the animation. Candidates:
dummy plan leaking through physics/DI, facing→roll-direction coupling,
label-construction subtlety. Needs the v2 offset-bucketed analysis.

**v2 redesign (next)**: memory/world-model targets that CANNOT be copied
from the current frame — time_since_knockdown, frames_until_kd (bucketed,
for the decodability-vs-lead-time curve), opponent percent K frames ago —
plus Hewitt-Liang-style control tasks (shuffled labels) to calibrate probe
capacity. Captures are cached; v2 needs only relabeling.

## Phase 2 — Training observability (payoff #3: solve the NaN, watch representations form)

Goal: instrument training so numeric failures and representation quality are
visible live. Effort: ~0.5 day. Can run parallel to Phase 1 (edits to drill
script only; respect NO-MIX around live runs).

- [ ] Per-epoch logging in `dagger_drill.exs`: per-head losses, max |logit|,
      per-layer activation/param norms.
- [ ] **NaN forensics**: confirm/refute the top suspect — the per-head loss
      normalization `loss / max(stop_grad(loss), 0.1)`
      (`lib/exphil/networks/policy/loss.ex:208`) amplifies gradients 10x once
      a head's loss < 0.1. Prediction: head losses cross ~0.1 shortly before
      every NaN. If confirmed: raise the clamp floor / remove normalization —
      likely worth more than every LR experiment run this week.
- [ ] (After Phase 1) cheap probe eval every N epochs → "when do features
      form" curves; export-epoch selection by probe score instead of loss.

Success: next training run produces a numeric post-mortem for any NaN, and the
NaN root cause gets a verdict in GOTCHAS.md.

## Phase 3 — Failure-mode attribution (payoff #2: diagnose instead of eyeball)

Goal: convert observed pathologies (double-jump spam, shield-lock, dash-dance
loops — all imitation-under-distribution-shift artifacts) into measured causes.
Effort: ~1-2 days. GPU: minutes + a few Dolphin games for behavioral A/Bs.

- [ ] **Ablation harness**: zero/mean-substitute input embedding groups at
      inference. First target: prev-action conditioning — does jump spam
      vanish when prev-action is ablated? (Tests the self-reinforcing-loop
      hypothesis directly.)
- [ ] Input attribution (gradient x input, or integrated gradients) from each
      head's logit back to embedding dims: what drives the shield logit —
      opponent approach or a bias? Aggregate over frames where the pathology
      fires.
- [ ] Hysteresis A/B: same checkpoint probed with/without 0.45/0.3 hysteresis
      to separate model pathology from controller-shim pathology (shield-lock
      is suspected shim, jump momentum suspected model).
- [ ] Write each confirmed mechanism into GOTCHAS.md / the handoff.

Success: for at least two of the three observed pathologies, a one-line causal
statement backed by an ablation number, plus the implied fix.

## Phase 4 — Representation tracking across DAgger rounds (payoff #4: reaction vs guessing)

Goal: answer the tech-chase question mechanistically. Effort: ~1 day.

- [ ] Temporal probing: probe "opponent's tech choice" from the GRU state at
      each frame offset relative to tech animation start. When does the info
      appear vs when it's humanly visible? (Reacting policies show it after
      visibility; "guessing" policies never show it; distribution-exploiting
      policies show it before.)
- [ ] Track the same probes across a DAgger loop's rounds: does pool growth
      grow the representation, and does representation growth precede
      conversion-% growth?

Success: a plot (probe accuracy vs frame offset, per checkpoint) that says
which of our policies actually read techs. Feeds directly into whether the
tech_random dummy is even the right drill.

## Phase 5 — Probe-driven data curation (payoff #5: targeted data beats blind scale)

Goal: close the loop — let representation deficits choose what to farm.
Effort: harness ~0.5 day, then it's an operating mode. This is the honest
counter to slippi-ai's scale advantage on a single 5090.

- [ ] **Deficit report**: per-checkpoint scorecard of probe accuracies across
      the feature dictionary; lowest features = missing knowledge.
- [ ] Map deficits → rollout recipes (e.g. no "opponent offstage"
      representation → farm edgeguard scenarios; weak tech-direction → more
      tech_random games with the probe as the metric, not conversion %).
- [ ] One full cycle: deficit report → targeted farming → retrain → deficit
      report delta + conversion delta.

Success: one documented cycle where targeted data measurably closed a named
representation gap and moved (or knowably didn't move) live performance.

## Phase 6 — Activation steering + SAE features (payoff #6, and interp-for-its-own-sake)

Goal: behavior control without retraining; ground-truth-validated feature
discovery. Effort: ~2-3 days. Stretch tier — unblock after Phases 1-3.

- [ ] Steering vectors: difference-of-means directions in GRU state space
      (aggressive vs passive frames from ground truth); add at inference,
      measure conversion/knockdown shift in Dolphin.
- [ ] Train an SAE (edifice has five variants) on GRU hidden states over the
      drill pool; name features by their top-activating frames **scored
      against the ground-truth dictionary** — quantified feature
      interpretability, not vibes.
- [ ] Cross-architecture comparison (the edifice pitch): same drill data,
      GRU vs Mamba backbone — do the same ground-truth features emerge?
      This is a publishable/demoable novelty; almost nobody can validate
      SAE features against complete ground truth.
- [ ] rwing integration idea: render top-activating frames/clips for a
      feature directly in the replay viewer.

## Constraints & ground rules

- NO-MIX rule applies to every phase: capture/probe runs are beams; one at a
  time, never `mix` beside a live one.
- Interp compute is cheap (probes are linear; SAEs are small) — the expensive
  resource stays Dolphin game time and training runs. Design experiments to
  spend GPU-minutes, not GPU-hours.
- Every claim gets validated against ground truth or a behavioral A/B before
  it enters the handoff as fact.

## Experiment log

| Date | Phase | Experiment | Result |
|---|---|---|---|
| 2026-07-13 | P2 | NaN forensics: code-read eliminated head-normalize amplification + unguarded log paths (all off/stable in drill path); found drill trains PURE bf16 (params+optimizer, no f32 master) — prime suspect. Instrumented run (`--nan-forensics`, vitals every 100 steps) in flight. | pending |
| 2026-07-13 | P0 | Wrote capture stack: `Policy.build_temporal_trunk` (lib refactor, name-compatible with exports), `ExPhil.Interp.Activations` (trunk capture over replays, training-identical embedding path), `ExPhil.Interp.GroundTruth` (15 features incl. tech_choice + next_kd_choice lookahead), `scripts/interp_capture.exs`. | written; smoke test pending (beam busy with forensics) |
| 2026-07-13 | P0 | Smoke test: poolgrow_r1 × Game_20260713T015257.slp → {18042, 256} f32 + aligned labels in 6.9 s. Base rates: own_airborne 66.8%, own_shielding 6.4%, opp_shielding 0%, opp_knockdown 2.9% (529 lifecycle frames). 84/84 policy tests pass post-refactor. | **P0 COMPLETE** (gaps listed in checklist) |
| 2026-07-13 | P2 | Forensics verdict: NaN at step 456,055 (ep 36). Trail: param_max flat 5.13→5.15 over final 3k steps, nu pinned 0.048, loss finite (0.058–0.58) to the last sample, then total poisoning within ≤55 steps. **Single-step detonation confirmed; slow forward-overflow refuted.** Per-epoch shuffle did NOT prevent it (died ~456k, same zone as fixed-seed r2's 457k) — seed-resonance co-suspect exonerated. Bonus: `backbone_defaults(:gru)` already specifies `precision: :f32` — the drill silently drops it (cherry-picks shape fields only) and trained pure bf16 against the stack's own recommendation. Discovered en route: drill GRU is actually window 60 / 2 layers (the `\|\| 16` / `\|\| 1` fallbacks never fire). | precision control queued |
| 2026-07-15 | P3 | **Case #3 ROOT CAUSE (15 lines of reading, directed by attribution): the TEACHER is direction-blind.** MewtwoFairExpert's fine_key uses abs(dx) — no sign, no facing. The expert table cannot distinguish opponent-in-front from opponent-behind, so labels never depend on side, so the heads never learned to consult it — while the trunk still learned the feature (0.72 probe) from prediction pressure alone. Fix queued (#21): facing-relative {distance, side} key, unlocking turnaround exemplars already in the recordings. Full causal chain for pathology #3: expert key design → direction-independent labels → heads ignore opponent side → fair-in-place-behind loop. | expert fix queued |
| 2026-07-15 | P3 | **Case #3 (opponent-position) VERDICT: knowing-without-acting, at the layer level.** Attribution (grad×input, main-stick head, empirically-discovered dim groups): opponent-position dims get only ~5-8% of saliency in BOTH eras, front or behind — not facing-gated, just low everywhere; own_action reliance RISES when opponent is behind (27→37%) = "keeps doing its routine". Compression probe SURPRISE: trunk decodes opp_behind at 0.71-0.72, BETTER than the raw input floor (0.64) — the trunk enriches opponent info (P1 compression theory wrong for this feature); the heads just don't consult it. #21 implication: aux-head representation shaping is pointless; suspect ONE LEVEL UP first — audit whether MewtwoComboExpert itself issues turnarounds when the opponent is behind (a blind TEACHER can't be fixed by loss weighting). | #22 closed; #21 redirected |
| 2026-07-14 | — | **NEW ERA, FIRST SCOREBOARD**: 2-round DAgger loop vs the REAL tech dummy (cpu0, walking, random techs) with anti-copycat objectives (prev-action dropout 0.4, transition weight 2.0) and the clamped loss: r1 (3-game pool) = 1/16 = 6.3%; r2 (6-game pool) = **3/14 = 21.4%**. Zero NaN events at LR 2e-4 (clamp in production), all games full-length, dummy teching throughout. NOT comparable to old-era scores (harder, honest task) — this is baseline zero for everything that follows. Pool now 9 games. | new benchmark era begins |
| 2026-07-14 | P3 | **#30 L-cancel origin CONFIRMED (user hypothesis)**: 46-50% of shield onsets fall within 12f after an aerial (L-cancel timing), and the training fixtures contain ONLY ~7f trigger presses (p50=7, max=13, zero sustained shields) — the policy's 200f+ holds were copy-loop amplification of the user's L-cancel habit; no shield concept exists in the data. **#17 tap-inhibition on follow-ups: directional support** — ablation raises P(A) on combo frames (prevA ∧ opp hitstun) +31% vs +11% baseline (n=45, thin); residual follow-up failure likely opponent-tracking (#21/#22). | both closed |
| 2026-07-14 | P2 | **NaN SAGA CLOSED — clamp validated.** Run with the min∘max logit clamp survived to step 556k (all six unclamped deaths: 284-456k), loss 0.098, param_max 5.40 — PAST the old "critical scale" pseudo-invariant, confirming it was only a proxy for logit growth. En route, the clip-based first attempt exposed an UPSTREAM Nx BUG: `Nx.clip` gradients are g² instead of g (double-multiply in grad.ex's clip rule; invisible when g=1, so sum-based tests never caught it) — pinned with a 6-test suite (fails on nx main, wild-caught via a training collapse), FIXED in the local nx checkout (pin 6/6, existing grad tests 233/233). exphil ships the min∘max form regardless. | CLOSED |
| 2026-07-14 | P2 | **NaN SAGA SOLVED — crime-scene autopsy (scene_step284154).** Exact-repro replay of the fatal step came back CLEAN (same batch+params, freshly compiled program, no NaN) while the fatal batch carried button logits at **−89.91 — precisely f32's exp-overflow cliff (88.7)**. Verdict: the stable BCE form is safe on paper, but XLA's algebraic simplifier may rewrite it (or its adjoint) into exp(+x) variants during fusion — stability depends on the compiled program, and the training pipeline's fused graph made the unstable choice. Explains everything: precision-independence (f32/bf16 share exponent range), buttons-only (BCE's form; softmax heads get stable rewrites), steps-to-NaN ∝ 1/LR (logit growth rate on never-pressed rare buttons), single-step detonation (first frame crossing the cliff), regime pseudo-invariants (proxies for logit magnitude). **Fix: clamp logits to ±60 in imitation_loss — mathematically inert (sigmoid saturated to 1e-26), rewrite-proof.** Validation run in flight (expect survival past 500k). Six blind runs couldn't find this; one captured scene + three 2-minute replays did. | fix shipped; validating |
| 2026-07-14 | P6 | **LEACE copy-signal erasure: offline surgical success, closed-loop failure — the week's lesson in a new costume.** Closed-form Cholesky-whitened fit (rank 4, guarantee exact: cross-cov 0.2648 → 2.3e-5) on 3 replays; offline: shieldHoldPersistence 0.639→0.155 with unconditional rates preserved (surgical!), and A|prevA kept legitimate animation-state elevation while losing the echo — textbook. LIVE (eraser injected trunk→heads): shield eliminated (0.0%, 2f max), airborne 72.5%, knockdowns 5 (worse than baseline 12, ablation 22). Teacher-forced eval underestimated closed-loop compounding: the erased policy immediately visits states outside the eraser's 3-replay fit coverage, where the affine edit is uncontrolled. **Follow-up: iterative eraser fitting (fit→play→refit = DAgger-for-erasers) or much broader fit data; also try edit scaling (α<1).** Live observation (user): erased policy jumped and GRABBED (tomahawk-adjacent — emergent when the copy-echo is cut), minimal shielding. User hypothesis worth its own test: the original "shield-lock" may be metastasized L-CANCEL taps (trigger presses at aerial landings, abundant in drill data) stretched into holds by copy-loop + hysteresis — deliberate shielding is nearly absent from training data, so there may be no learned shield concept at all (task #30). En route: eigh-based ZCA whitening REPLACED with Cholesky (BinaryBackend eigh silently under-converges at 1e6 eigenvalue spread → guarantee-violating eraser; also 6.5min→1s), edifice LEACE rebuilt with real fit + 12/12 functional tests. | offline ✓ / live needs iteration |
| 2026-07-14 | P3 | **LIVE ABLATION CLOSES BOTH CASE STUDIES.** One prev-action-ablated game vs baseline (same policy/dummy/stage): shield 10.0%→**0.1%** of frames, max hold 215f→**16f** — **shield-lock IS the prev-action feedback loop** (shield→"r held" reported→prob stays elevated→hysteresis holds; open-loop probs couldn't sustain holds, closed-loop does). Jump: airborne 43.9%→51.4%, confirming the offline sign (channel SUPPRESSES jumps). Same channel, opposite signs per action: prev-action acts as "continue" prior for holds, inhibitor for taps. Bonus: the ablated policy plays BETTER (knockdowns 12→22, tech-situation frames 170→338) — prev-action conditioning looks net-harmful for this policy class; new-era training should trial use_prev_action=false. Live behavioral notes (user): SH/FH from ground with fair/bair on landing; a fair-in-place loop when Fox stands BEHIND (opponent-position blindness — matches P1's info-compression finding; P3 case #3 candidate: does main-stick head respond to opponent x?); tomahawk-like empty hops. | shield mechanism SOLVED; jump story rewritten |
| 2026-07-14 | P2 | **f32-loss fix REFUTED** — died at step 342,407, same buttons-head signature, cast verified in the executed path. Post-mortem exposed a detector flaw: sum-of-squares norms overflow at \|g\|~1e19 in ANY 32-bit float, so ":NONFINITE" couldn't distinguish inf grads from large finite bursts — and `clip_by_global_norm` has the SAME overflow internally, making the clipper itself a candidate NaN amplifier for large-but-finite bursts. Detector rebuilt max-based (overflow-free, trips at 1e15). Open question: are the buttons bursts finite (→ stable-clip fix) or infinite (→ deeper op hunt)? One more instrumented run would discriminate. | detector v2 ready; run optional |
| 2026-07-14 | P3 | **Jump-spam ablation stage 1: hypothesis REFUTED, direction REVERSED.** Ablating the prev-action channel INCREASES jump probability (mean 0.240→0.299; frames ≥ press 21%→33%; grounded frames 0.34→0.48). The channel SUPPRESSES jumping — the loop story is dead; the spam lives in the state→action mapping's jump-heavy marginal. An ablation killed the pet theory before any Dolphin time was spent on it. | case study #1 complete (offline) |
| 2026-07-14 | P3 | **Shield-lock offline: shim-only insufficient.** Raw shield probs (max(l,r), teacher-forced on replays): 9.7% frames ≥ press, dead-zone occupancy 2.5%, holdable-run p99 = 203 frames, max 237, ZERO runs ≥ the 450-frame break horizon. On-distribution, the shim can't hold shield to break. Conjecture: breaks need CLOSED-LOOP dynamics — once shielding, prev-action reports r-held and states drift off-distribution — i.e. possibly the same self-reinforcement mechanism jump-spam DIDN'T have. Live A/B (hysteresis-off vs prev-action-ablated) discriminates. | needs live A/B |
| 2026-07-14 | P2 | **GRAD-LOCALIZER NAMED THE KILLER.** Per-step fused grad checks (armed from step 250k) caught the birth frame at step 344,769: loss FINITE (0.556), non-finite grads in exactly buttons_hidden/buttons_logits + the trunk they backprop into; every other head finite. **NaN is born in the buttons-BCE backward under bf16 compute.** Unifies all refutations: mixed precision failed because it also runs backward in bf16; window/seed/pool are irrelevant to the buttons head; LR only sets arrival speed at the extreme-rare-button-logit regime. Fix shipped: cast logits to f32 at the loss boundary (loss.ex build_loss_fn) — network stays bf16, fragile BCE adjoint gets full f32. Confirmation run in flight (prior deaths 323-345k; survival past 500k = fixed). | confirmation running |
| 2026-07-13 | P1 | **v2 suite ran; then ground-truth audit found THE drill-invalidating bug.** v2 memory targets de-saturated the probes (shuffled control ≈ chance = methodology sound), but lead-time decodability was 1.000 everywhere → dug in → **all 91 knockdown episodes across 12 replays are missed techs; the dummy never teched once.** Replay forensics: dummy's digital R press (button_r=true) recorded in-window, even held THROUGH touchdown — ignored by the game. Root cause: probe/drill recipes pass `--dummy-cpu-level 3`, and the bridge's menu helper sets port 2 as a **CPU level 3** — the game AI owns the port; the Elixir TechRandom state machine has been pressing into a pipe the game ignores since the drill began. **Every rollout and probe this week was vs a vanilla lvl-3 CPU Fox.** Relative scoreboard comparisons survive (same opponent everywhere); the "reaction drill" premise was never active. Fix: `--dummy-cpu-level 0` (verification game running). Probe harness hardened: balanced accuracy now nil on single-class evals (the degeneracy that masked this). | root cause found; fix in verification |
| 2026-07-13 | P1 | v2 numbers (pre-fix data, still informative): trained trunks memBA 0.65-0.73 < random-init trunk 0.77 < raw input floor 0.84 — **trained policies COMPRESS AWAY temporal/tech information relative to their own input**; training prunes what behavior doesn't use. Also: Melee frame state is near-Markovian (action_frame + action + percent ≈ rolling history summary), so "memory" targets need careful construction. Spearman memBA↔conversion −0.47 → representations do NOT rank behavior → **Phase 3 (acting, not knowing) confirmed as the bottleneck**, exactly the pre-registered weak-correlation branch. | P1 verdict: redirect to P3 |
| 2026-07-13 | P2 | **Window-16 test REFUTED the exploding-BPTT theory too**: died at step 323,040 (ep 26), param_max 4.48 — same zone as w60 runs despite 16 vs 60 Jacobians. Three runs, three same-neighborhood detonations; invariants are the REGIME (LR ~2e-4 near peak, loss 0.2-0.4, weight scale 4.5-5.2), not precision/window/seed/steps. Since tanh/sigmoid gating bounds forward activations, no proposed slow mechanism produces an inf — the fatal step itself is unobserved. Next: per-layer grad-finiteness instrumentation armed after step 250k to catch the detonation in the act and name the layer. | grad-localizer queued |
| 2026-07-13 | P2 | **Mixed-precision control REFUTED the precision hypothesis**: f32 master weights + f32 optimizer state died the same way at step 336,695 (ep 27). Key clue: both forensics runs detonated at nearly identical weight scale — **param_max 5.15 (bf16) vs 5.21 (f32-master)** — at different step counts. New leading theory: **exploding BPTT gradients through the 60-step GRU recurrence** — LR sets the weight GROWTH RATE, but the trigger is a critical spectral radius (~param_max 5.2) where the product of 60 backward Jacobians overflows (any float precision; bf16/f32 share exponent range). Explains LR-dependence of steps-to-NaN (2e-4 ~340-460k, 1.5e-4 ~673k, 1e-4 ~762k), single-step death, clip impotence (inf is born in backward, clip(inf)=NaN). En route: fixed real `MixedPrecision` bug (blanket-cast clobbered u32 RNG keys → GRU compile crash). Discriminating test queued: window 16 (16 Jacobians ≫ headroom). | window-16 run queued |

## Milestone summary (check off here)

- [ ] P0: activation+label pairs from any checkpoint/replay
- [ ] P1: probe↔conversion validation verdict (decision gate)
- [ ] P2: NaN root cause confirmed + training telemetry live
- [ ] P3: two pathologies causally explained
- [ ] P4: reaction-vs-guessing verdict per checkpoint
- [ ] P5: one closed curation cycle
- [ ] P6: steering demo + ground-truth-scored SAE features
