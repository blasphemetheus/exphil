# Mechanistic Interpretability Roadmap — v2 (the application arc)

**Status doc — created 2026-08-04.** Successor to
[INTERP_ROADMAP.md](INTERP_ROADMAP.md) (v1, P0–P6: ALL CLOSED with
verdicts — keep it as the instrument log and the record of what each
tool proved). Research inputs:
[INTERP_NEXT_RESEARCH_2026-07-20.md](INTERP_NEXT_RESEARCH_2026-07-20.md)
(the literature survey; its adoption queue is mostly unbuilt).

## Strategic context

v1's arc was "build the instruments, validate each rung causally." It
worked: probes → ablation → patching-adjacent → steering all reached
verdicts, three pathologies got end-to-end causal stories with shipped
fixes, and the NaN saga died to a crime-scene autopsy. The program has
transitioned from *build interp* to *interp is how this lab debugs*.

v2's organizing principle is therefore different: **every workstream has
a named current customer** — a live problem from the 08-03/04 sessions
that the instrument serves. Survey items without a customer (attribution
graphs, auditing game, crosscoders, manifold geometry) stay explicitly
parked until the customers are served; they are pulled in *by* a
workstream when needed, not scheduled on their own.

The four customers, from HANDOFF_2026-08-04:

| Customer | Problem | v1 instrument that applies |
|---|---|---|
| **Absorber** (#34) | YS offline collapse, 52% squat, stochastic contrastive pair | BasinRollout, margin probes, CycleSim |
| **Dummy-vs-human inversion** | ms_g6_sp1: dummy record, human ZERO; promote_check thresholds uncalibrated | ablation harness (P3), g6-vs-g4 labeled contrast |
| **Stage blindness** (#33) | stage is a network input that has never seen variety | probe + ablation = knowing-vs-acting split (P3) |
| **Fight-state gap** | opp_knockdown ≈ 0.07 in BOTH archs (P4 + SAE agree) | deficit report → curation loop (P5); cycle 3 is the live test |

## W1 — Absorber-entry detection (customer: #34; HIGHEST LEVERAGE)

The YS repro gives what the crouch zoo had: same checkpoint, same stage,
good run vs absorbed run, everything else held constant. Diagnose the
*entry*, not the steady state — by the time squat occupancy is 52% the
interesting event is long past.

- [ ] Run `BasinRollout.entry_from_absorbed_replay` + cycle-margin
      probes over the YS pair (instrument list already in task #34).
      Deliverable: the state signature at absorber ENTRY — which dims /
      margins separate the runs in the frames *before* divergence.
- [ ] Turn the signature into a detector: a scalar (margin distance,
      probe projection) computable per-frame offline. Floor-test it per
      the GOTCHA #79 template before trusting it (v1 rule: every new
      metric gets a floor test).
- [ ] **Feeds cycle 4**: if cycle 3 lands P2 (core holds, YS unchanged
      — the pre-registered branch), the next curation iteration anchors
      snippets on ABSORBER ENTRY instead of on getting hit. That anchor
      IS this detector. W1 is the prerequisite for the pre-registered
      next move of the curation program.

Success gate: detector separates the absorbed from the good YS runs at
entry time (not steady state) and survives its floor test. Stretch: one
CycleSim intervention — perturb the entry state and show the collapse
doesn't happen (necessity, the P3 rung).

## W2 — Static-overfit pre-screen for promote_check (customer: the g6 disaster)

Mechanistic version of the 08-04 promotion rule. A policy whose cycle
does not consult opponent state is dummy-overfit *by construction* — no
human game needed to find out.

- [ ] Opponent-dependence score: ablate (zero / mean-substitute) the
      opponent embedding dims during offline cycling; measure behavior
      delta (chain length, cycle margins via CycleSim). The P3 ablation
      harness already does the hard part.
- [ ] Validate on the labeled pair: ms_g6_sp1 (human-zero) must score
      LOW dependence, ms_g4_d2mix (human-best) HIGH. ms_g2_mdq_ss as
      the third point. If the ranking is wrong, the hypothesis "static
      overfit = opponent-blindness" is refuted — also worth knowing,
      and cheap.
- [ ] If validated: wire as an advisory rung in
      `scripts/promote_check.sh` (seconds of GPU before any human game
      is spent). Advisory like the rest — one labeled pair is not a
      calibration set.

Success gate: correct ranking on g6/g4/g2. Refutation is a publishable
negative (the policies may all be equally opponent-blind and differ
elsewhere — which would redirect the fight-state story).

## W3 — Data attribution for the snippet miner (customer: curation loop)

Persona-vector application from the survey (Anthropic Aug 2025 recipe),
composed with the loop cycle 3 is running.

- [ ] Bad-habit projection scoring: project candidate snippets'
      activation deltas onto known bad-habit directions (shield-lock
      vector exists; absorber-entry direction arrives from W1). Score
      BEFORE training, filter or downweight.
- [ ] Relapse dashboard: per-checkpoint projections onto the same
      directions across generations (g2→g4→g6→g8→g10) — does the
      absorber direction grow generation over generation? Retroactive
      curve is free once the directions exist.
- [ ] Apply to cycle 4's snippet pool as the first live use.

Success gate: one cycle where projection-filtered snippets measurably
change the trained outcome vs unfiltered (or a knowable null — the P5
"knowably didn't move" standard).

## W4 — Stage representation audit (customer: #33, BEFORE fixture farming)

**CLOSED 2026-08-04 by the W1 patching kill test, which answered the
question this workstream existed to ask:** the platform failure is
carried by the OWN-Y channel (patching y alone to 23.45 silences the JC
head; stage flags identical in both conditions), so #33's fixture
farming targets the right variable — grounded-at-height exemplars — and
the stage-FLAG consultation question is moot for the absorber. Residual
rider (one probe, run whenever multi-stage fixtures land): verify the
trained-on-variety policy consults the stage input rather than re-deriving
everything from y. Original design kept below for that rider.

One probe and one ablation, the knowing-vs-acting split from P3, decides
whether #33 is a data problem or worse.

- [ ] Probe trunk activations for stage identity across existing
      mixed-stage captures (need: per-stage offstage lookup — the P0
      gap — for clean labels).
- [ ] Ablate/swap the stage input dims offline; measure behavior delta
      on FD-vs-YS cycling. Knows-and-consults → fixture variety will
      transfer; knows-but-ignores → expect fixture farming to
      disappoint exactly like case #3 did until the teacher changed.
- [ ] Cheap rider: does the trunk represent platform-relevant position
      differently on YS at all, or is YS just "FD with noise" inside?
      (Connects W4 to W1 — the absorber lives on YS.)

Success gate: a one-line verdict ("stage is known/consulted:
{yes,yes}|{yes,no}|{no,–}") BEFORE multi-stage fixture time is spent.

## W5 — Mamba instrument upgrades (customer: bake-off #27 re-aim + any new line)

The survey's days-scale Mamba items, prerequisite for taking the
re-aimed architecture bake-off's interp readouts seriously.

- [ ] Bake the Mamba probing gotchas into `ExPhil.Interp.Activations`:
      conv off-by-one (lag labels t+1..t+3), probe pre-gate SSM output
      / gate branch / post-gate separately, exclude Δ episode-boundary
      sinks.
- [ ] Hidden-attention maps (arXiv:2403.01590): materialize the
      selective scan's data-dependent attention per decision — "which
      past frames did this press depend on." Trivial at our scale;
      second instrument for any future reaction-training claim (the P4
      acceptance-test curve stays primary).
- [ ] dt-distribution dump + f32 check for A_log/dt/D on any newly
      trained Mamba line (survey checklist items; near-zero cost).

Success gate: cross-arch runs in the #27 bake-off report per-site
probes (not just trunk output) without the known Mamba artifacts.

## Parked (pull in only when a workstream demands it)

- **Attribution graphs at 256-dim** — THE flagship, and the right next
  rung after steering; earns its 2-4 weeks only once W1–W4's customers
  are served. Natural trigger: W1 finds an entry signature and we want
  the full "why does squat win here" circuit.
- **Auditing game** (plant a behavior, blind-audit) — the transferable
  safety result; needs the stack idle-stable first.
- **Crosscoders** (DAgger-round diffing, GRU-vs-Mamba shared
  dictionary), **window-averaged SAEs**, **manifold geometry**,
  **timescale map**, **WriteSAE rank-1 atoms** — all still good, none
  has a current customer.
- **RL-era note**: when RL staging (#29) unparks, W3's dashboard is the
  KL-drift monitor (watch bad-habit projections during RL, not just
  reward) and W2's pre-screen is the exploit detector (RL vs lvl-9 CPU
  will overfit the CPU; the opponent-dependence score says how much).

## Ground rules (carried from v1, still binding)

- NO-MIX beside a live training beam; capture/probe runs are beams too.
- Every new metric gets a floor test (GOTCHA #79 template) before use.
- Every probe claim gets a causal check before it enters a handoff as
  fact (survey: only ~5% of representational signatures were causally
  load-bearing in Mamba-2 — knowing-without-acting is the DEFAULT).
- Score chains from replays, never qtrace press counts.
- Spend GPU-minutes, not GPU-hours; Dolphin time and training runs stay
  the scarce resources.

## Experiment log

| Date | Workstream | Experiment | Result |
|---|---|---|---|
| 2026-08-06 | Flagship Stage 3 pass 2 | **The multishine circuit is now LEGIBLE: features named by their own top activations** (scripts/interp_graph2.exs): f1665/f1186 "grounded shine (hold)" → f126 "aerial shine in progress" (the cycle-drive node) → X/JC; f958 "JC in progress"; f760 "aerial shine, X-suppress" (don't JC mid-air — a sensible inhibitor); **f1889 = "ground-reflect ON PLATFORM, B held" — the absorber-state feature, self-named** (plat=1.0 in its top-32 activations). Two diagnosed limitations: (1) full-path suppression checks weak/sign-mixed because the site selection sits on the JC ANIMATION frame, not the DECISION — base X-logit −4.68 there; the X decision happens ~decode-lag (≈5f) earlier (the qtrace lesson applies to attribution sites; pass-3 fix = shift sites by the pipeline offset). (2) delay-id-4 contrast INVALID: the dictionary corpus embedded everything at id 3, so the id-4 one-hot dim has ~0 std and standardization explodes (gains of 4,500 = 1/1e-6 artifacts); needs a multi-id dictionary refit. Remaining for #9: decision-shifted sites, multi-id refit + the id-4 margin-carrier question, replay counterfactuals. |
| 2026-08-06 | Flagship Stage 3 (first pass) | **Mini attribution graph around f126 built (scripts/interp_graph.exs) — and it adds a mediation layer to the absorber story.** Gate decomposition of f126's pre-act gap (ground +0.58 vs platform −0.76, straddling the ReLU): **97.5% trunk-carried** (dims 111/84/188/152/35...), NOT direct input — own-y silences f126 through the ACCUMULATED representation, exactly matching the patch-probe asymmetry (whole-window y-patches restored, single-frame didn't). Feature→feature edges into f126: f1186 (0.71), f1665 (0.48), f958 (0.45), f238, f1215... — suppression checks match to 4 decimals, WITH the honesty caveat that pre-act linearity is exact by construction (these validate plumbing; the approximation-bearing checks go through top-k/ReLU/heads to the X logit). Remaining Stage-3: nonlinear-path + replay-counterfactual edge validation, name f1186/f1665 by top-activating states, extend to a second decision (aerial-B). |
| 2026-08-06 | Flagship Stage 2 COMPLETE | **First attribution-graph-grade result: the platform X-silence decomposes into MISSING DRIVE + ACTIVE SUPPRESSION, named at feature level with causally validated arrows.** scripts/interp_attribution.exs: frozen-gate (identity-carry) attribution over the Stage-1 update dictionary — a_{j} = sum_t h_t[j]·<W_dec[:,j]⊙y_std, dLogit/dtrunk> — with a built-in causal check (ablate the feature's decoded contribution from the trunk state, compare actual vs predicted logit delta). Findings, g10b X head at JC sites: (1) drive half — **f126** (+2.51 mean attribution at every ground site; also f1545/f2047/f234) carries the X decision and fires ZERO on platform sites — the cycle-drive circuit is absent at altitude; (2) suppression half — **f760/f1889** actively push X down in platform hold-states, ablation recovers +0.33/+0.21 logit vs +0.37/+0.24 predicted, sign agreement 1.0. En route: heads-grad via batched central differences (nested Axon/defn jit fights back), GOTCHA-#1 relearned for dictionary saves, and a policy fact worth keeping — **g10b's own cycle JCs from reflector af=1 (one frame), far tighter than the teacher's af3-4 window** (the af>=3 site filter matched 1 frame on FD). Stage-3 target picked by the data: trace f126's upstream causes; f126 fire-rate is also a candidate live cycle-health meter. |
| 2026-08-06 | Flagship Stage 1 COMPLETE (+ NIF fix en route) | **Trunk-update transcoder passes its gate: R^2 = 0.56** (scripts/interp_transcoder.exs, g10b, 88k pairs over FD+YS+human corpus; dictionary saved for Stage 2). The road there was three diagnosed failures, each a real lesson: v1 input_t->trunk_t R^2 −0.01 (a single frame cannot predict 60 frames of recurrence — P1's compression finding as a design error); v2 raw next-state target −0.03 (k active features cannot represent the rank-256 state-carry — why CLTs sit on residual streams); v3 residual target STILL −0.05 → the actual culprit was **underfitting** (SAETrainer's default lr 1e-3 barely moved the loss; the P6 SAE runs likely also underfit — unmeasured, they had no R^2 gate). At lr 0.1/3000 steps/k=32: R^2 0.557. Feature F1s vs STATE labels are modest (0.12-0.40) but the yardstick is wrong for UPDATE features — v5 refinement queued: score against label TRANSITIONS (onset events). Stage 2 (frozen-gate attribution) is GO. En route: Roy character-id NIF bug fixed (post-frames carry INTERNAL ids; identity convention pinned by test, zero checkpoint drift; ethnum 1.5.3 unblocked the rebuild). |
| 2026-08-05 | #20 margin lever | **P1-at-id3: training weight BUYS aerial-shine margin.** `ms_g12_margin` (g10b recipe + --margin-weight 3 on the one B-press-edge frame per cycle, 9,177 frames = 2.71% of pool): delay-id margin sweep vs g10b's −0.523/+0.091/+0.248 → **−0.227 / +0.278 / +0.327** — id3 3x wider (clears the +0.2 prereg bar), id2 deficit halved, id4 short of its +0.4 stretch bar. Stand d3 423.4 c423 (gate holds at record level). YS 3-of-3 collapsed-tier (3/4/3) — the recipe's YS axis stays seed-unstable and margin weighting doesn't touch it; g10b REMAINS production. Standing result: **2.7% of frames at x3 moved the critical margin more than a whole delay rung** — the margin is trainable directly. Next: g12-vs-g10b human netplay A/B (FD; the chain prediction: g12's id3 mode should out-chain g10b's), and if confirmed, fold --margin-weight into the standard recipe. |
| 2026-08-05 | #20 delay-break study (NEW ARC, first pass complete) | **The delay gap is a PER-LINK BERNOULLI at the aerial-shine decision, and declared delay-id sets the margin — three independent legs agree.** (1) Forensics (scripts/probe_delay_breaks.exs, three-latency corpus): real chains die by :air_shine overrun in every regime; :empty_hop nearly vanishes under netplay (49 local → 5/3). (2) Air-stretch histogram is **BIMODAL** — links are tight (<=5f) or ballooned full-hops (>12f), the 6-8 bucket has ONE entry per netplay regime: not gradual stretching but per-link failure. Tight fraction: 58% local → ~30% netplay; per-link balloon probability ~1/22 local vs ~1/3-4 netplay (~6x). (3) Margin sweep per delay-id (probe_cycle_margins --delay-id, g10b, own stand replay): aerial-shine p10 margin **-0.523 (id2, flip 1.0) → +0.091 (id3) → +0.248 (id4)** — monotone widening with declared id; explains d4>d3 vs a human AND the historical d2-rung shatter. Levers this opens (pick next): {3,4,5} retrain (does the margin keep widening?); #36 probe-as-regularizer aimed at the aerial-B margin; deploy-id-above-rung experiments (declare id5 while at d4?— UNTRAINED-ID warning applies, would need training). Caveats: one replay, teacher-forced margins, n=2 human games per rung. |
| 2026-08-05 | HUMAN SESSION (the gate) | **g10b vs Bradley: HUMAN-NETPLAY RECORD chain 3 — and the local/netplay contrast reframes the human gap as a DELAY problem.** Direct netplay (lag peak 5 @ 99.8% agreement — sharpest ever; bot port verified via code both games): 33.0/min c3 and 27.8/min c2 across 2 games — beats g4's c2 record, pre-registered question (c>=3) answered YES. LOCAL play same day, same opponent (~2f intrinsic): 24-45/min with **c22** (bot-confirmed by eye test), c3, c3. **Within-policy measurement: the netplay delay regime costs c22 -> c3.** Implication: the residual human gap is substantially cycle-x-delay interaction, NOT only fight-state — task #20 (rung-composition theory) promotes from open theory to THE bottleneck; the fight-state work (curation arc) got the policy to where the delay is what's left. g10b: promotion case now complete on every rung (stand 421 c421 / YS 0-of-11 / rung-0 1.44 / human c3 record). One human game replay truncated mid-write (4th local game), rest in eval_runs/0805_{human,direct}_g10b. |
| 2026-08-04 | FINAL n=8 replications — the day's verdict INVERTS at proper n | **g10b is the real best card; g11b's YS excellence was luck.** g10b YS at n=11 (3 gate + 8 replication): **ZERO collapses**, chains 43-208 (median ~110) — robustly good, the cycle-3b P1 YS half REINSTATED for this artifact (the bakeoff_gru seed-collapse still shows the recipe isn't seed-guaranteed, but g10b itself is solid). g11b YS at n=10: **2 good (c226-263) / 8 collapsed-tier (c2-26)** — y-augment at the 12k dose does NOT fix YS; its two spectacular gate runs were lucky draws (both at loadavg 3.5+, load-dependence noted as an open question, not a conclusion). g11b keeps exactly one real claim: the stand record 428.4 c427 (deterministic). **EVAL_PROTOCOL vindicated hard: every n<=3 YS comparison today was noise — the run-level collapse bucket at n>=8 is the only YS instrument that means anything.** Standing hierarchy: g10b (stand 421 c421 + YS 11/11 robust) > g4 > g11b (stand record, YS broken) > everything else. g10b's remaining promotion rung: a human game. |
| 2026-08-04 | #27 bake-off + seed-variance CORRECTION | **Bake-off verdict: P3 — GRU stays.** mamba_2 on the champion recipe (fresh retrain, same data): stand **121.8 c1** vs GRU arm's 434.4 c435, YS 56-71 c1, rung-0 2.91 (danger band, the g6 signature) — loss converged identically (0.0016 vs 0.0013) so it FIT the data and failed closed-loop; against the survey's MaIL small-data prediction. One seed, GRU-tuned recipe — a mamba line would need its own recipe work to be fair, which P3 says isn't worth buying now. **The bake-off's real payoff was the CONTROL ARM: same-recipe GRU seeds vary wildly** — bakeoff_gru (cycle-3b recipe, new seed): YS **3-of-3 collapse** where g10b got 0-of-3; platX_mean spans −5.8/−3.6/−2.9 across same-recipe seeds. CORRECTIONS this forces: (1) cycle-3b's "human snippets closed YS" is partly SEED LUCK — the snippet effect on YS is not seed-stable; (2) 4a run-1's platX_mean −1.35 "dose-response" is within seed noise — platX_mean needs a variance baseline before any further dose claims (GOTCHA #79 spirit: the metric lacks a floor). What SURVIVES the noise: `ms_g11b_yaug` stand 428.4 c427 (record) + YS c263/c226 in 2-of-2 surviving runs — the best card ever, but n=2 (my probe killed r1) and one training seed. **Next: replication before promotion** — 8-10 YS runs on g11b (cheap) + one same-recipe reseed of 4a2; platX_fire stayed ~0 everywhere (0.0026-0.044), so altitude COMPETENCE remains unproven — the behavioral gains may be avoidance+core-strength. |
| 2026-08-04 | cycle 4a2 | **Y-augmentation at the safe dose (12,232 frames, frame-budgeted): BEST BEHAVIORAL CARD IN LAB HISTORY.** `ms_g11b_yaug.bin`: stand d3 **428.4/min c427** (new record; g4 423, g10b 421) AND YS **288.6 c263 / 271.6 c226** — chains ~double g10b's (103-177), zero collapses. The P1 dose-response branch fired behaviorally: synthetic grounded-at-height data at snippet scale improves BOTH the core and the off-distribution stage. platX dashboard readout pending (must wait for the bake-off chain to finish — see ops note). YS r1 lost: **my own standalone dashboard probe ran beside the live eval beam and the dual-EXLA-client SIGBUS killed eval run 1** — NO-MIX applies to EVAL stages, not just training; probes wait for a fully quiet machine. Full promote_check + human game still owed before any production claim. First OOM'd attempt also recorded: 0.7 VRAM fraction + 71 extra small lists OOM'd mid-epoch; 0.75 works. |
| 2026-08-04 | cycle 4a | **Y-augmentation run 1: P3 on dose, but DOSE-RESPONSE CONFIRMED on the mechanism.** `--y-augment 0.25` (list-fraction sampling) landed 44,450 grounded-at-height frames — 3.5x the validated snippet scale — and collapsed stand 421 -> **81.9 c1** (ms_g11_yaug; YS ~70 c1-2). BUT the primary readout moved exactly as the causal story predicts: **plat X_mean −5.8 (g10b) -> −1.35**, a 4.5-logit shift toward JC-at-altitude, with rung 0 still robust (1.15) and shield still dead. The hole responds to synthetic altitude data; only the dose was wrong (list-count sampling also skews composition — the cycle-1 trade in a new costume). Fix shipped: frame-budgeted augmentation (default 12k, smallest-lists-first); cycle 4a2 queued with prereg P1 dose-response / P2 windows-don't-overlap (-> F3 anchor becomes the enabler) / P3 altitude-interferes (-> record real platform play). Ops note: 4a's runner greps ate the dashboard gate's error output — gates now run unfiltered. |
| 2026-08-04 | W3 (dashboard half) | **Relapse dashboard v1 shipped (scripts/relapse_dashboard.exs, decode-level — persona-vector directions don't transfer across checkpoint bases per #36, so v1 reads the HEADS on common states).** Lineage g2/g4/g6/g7/g8/g10b × {shield-lock on common FD; X-competence on plat-JC states from the absorbed-YS fixture}: (1) **shield-lock is dead across the entire lineage** (0.0% everywhere) — the 07-17 fix never relapsed. (2) **platX_fire = 0.0 for ALL checkpoints INCLUDING g10b** (X_mean −4.6 to −7.7): the platform JC hole exists in every generation. Combined with g10b's 0 platform landings: **g10b AVOIDS the absorber (reroutes around platforms) but did not REPAIR it (still X-silent if placed there)** — human snippets taught avoidance, not the missing competence. Cycle-4 implication: multi-stage fixtures are still required to fill the y-OOD hole; avoidance is one stray shine-jump away from relapse in a real match. Remaining W3 half: snippet pre-filtering for cycle 5. |
| 2026-08-04 | W2 follow-up | **Percent-freeze falsification attempt: story SHARPENED, behavioral test deferred, CycleSim d3-gap found.** (a) Stand-eval forensics: hit-ind=0 — the shines never touch the dummy, so opponent percent is CONSTANT 0 in every stand eval. The "percent-as-climbing-clock" reading is wrong; the correct claim is **g6 overfit to opponent-percent==0 as frozen context** — any nonzero percent (i.e., any real opponent) is the destabilizing perturbation, which the fine probe already tests directly (0→80 moves g6 more than a 120-unit teleport). Behavioral confirm needs a pre-damaged-dummy eval config (doesn't exist; queued as a small harness item). (b) Instrument gap: CycleSim does NOT reproduce chains for the d3/queue policy class (g4, live chain 423, gets isolated singles at decode_lag 2/4/5/6, soft-lookup dominated) — its validation gate was delay-0-era (ms_open_z). Calibrating CycleSim for delay-trained policies is real work, recorded as a known limitation; do not read d3-class CycleSim numbers until it re-passes a gate with a d3 champion. En route: cycle_sim.exs comma-glob silent-match bug fixed (+ --decode-lag flag). |
| 2026-08-04 | curation (P5 arc) | **CYCLE-3B VERDICT: P1 — the curation loop is validated end to end, and cycle-3a's failure was 100% the miner misalignment.** Aligned human snippets (12,339 frames, ad2 re-mine) trained into the g4 recipe → `ms_g10b_human.bin`: stand d3 **421.4/min chain 421** (gate >=300; g4's own 423) — vs invalid 3a's 104.9 on the same replay content, so the whole 3a regression was the action-delay-5 channel corruption. YS: **0-of-3 collapses** (228.7 c103 / 322.6 c139 / 190.7 c177) vs g4's 2-of-3 at chain 1-2; AbsorberEntry: **0 platform landings in all 3 runs** (failure avoided upstream, not merely survived; squat 3-11.5%). Rung-0 opponent-sensitivity: **1.44** — g4's robust band (1.34), not g6's 3.84; human-state training did not induce static-opponent coupling. Convergence caveat recorded: loss 1.0e-5 @ 17 epochs (aligned prev-controller makes the task much easier; behavior evals clean, but watch for it). Real human states did what synthetic pressure could not — at 1/40th the frame count of cycle 1. Next: human-game verification (g10b vs a person — the ledger row that matters), then cycle 4 with AbsorberEntry anchoring if YS residual (~10% squat, chains 100-177 vs FD 421) warrants. |
| 2026-08-04 | W1 close + F1 | **W1 CLOSED: absorber-entry detector shipped + floor-tested.** `ExPhil.Interp.AbsorberEntry` (platform landing = airborne→grounded at y>15; entry = landing with ≥50% Squat/SquatWait occupancy over the next 120f — occupancy not spell-length, because r3's absorbed texture is many short spells). Floor test per GOTCHA #79 template (test/exphil/interp/absorber_entry_floor_test.exs, 3/3): fires on the absorbed fixture, zero entries on the good run (its 42-frame healthy platform touch correctly rejected — an entry-on-contact detector would anchor cycle-4 snippets on healthy escapes). YS contrastive pair promoted to durable fixtures (test/fixtures/replays/ys_multishine_{good,absorbed}_2026-08-04.slp). Ready as the cycle-4 snippet anchor. **F1 DONE:** docs/guides/EVAL_PROTOCOL.md — CRN pairing (deterministic FD/BF = 1 run, not 3; YS outcomes are bimodal run-level data, never means), sequential stopping incl. pre-registered human-session stopping questions, successive-halving budget (10 candidates ≈ 25 games vs 90+), and the calibration ledger seeded with the n=3 (opp-sens, FD chain, human outcome) rows — FD chain does not rank human outcome, opp-sens inversely does. |
| 2026-08-04 | F2 | **Trunk-Mahalanobis OOD scalar (scripts/probe_ood_score.exs): WITHIN-policy validation PASSES, cross-policy comparison is compression-confounded — do not use it as a promote rung alone.** Fit: ridge f64 Cholesky precision on own stand-FD activations (Nx.LinAlg.invert NaNs even in f64 — no pivoting; cholesky+triangular_solve is the stable route, LEACE lesson re-learned). Floor test: self p50=35/p95=88 (g4), 20/189 (g6), deterministic runs identical — PASS. (a) Absorber angle PASSES for g4: good YS r1 p50=470 with platform-touch spike to 12,599 vs ground 3,248; absorbed r2/r3 p50~7,450 (~200x floor) — the score detects platform occupancy and absorbed spells cleanly within-policy. Human games score 7.6k-10.9k for g4: **human play is absorber-level OOD** — the human gap and the absorber are the same off-manifold phenomenon from the trunk's perspective. (b) Cross-policy angle INVERTED: g6 maps the SAME human games ~3x closer to its manifold (p50 2.6-3.0k, ~140x its floor) than g4 does (~250x) — but low distance here likely means REPRESENTATION COMPRESSION (P1 v2: trained trunks prune what behavior doesn't use; an overfit trunk maps everything near its cycle manifold), not robustness. Cross-policy discrimination stays with the W2 sensitivity probe; the OOD scalar's niche is within-policy drift/absorber monitoring (and a live RL drift monitor later). Caveats: human games not played by g6 (common-game design), one truncated replay auto-skipped. |
| 2026-08-04 | W1/W2 | **Patching-on-contrastive-pairs (the fast-tracked attribution tier) — both pairs yield causal channel-level verdicts in one pass.** (a) Platform X-silence (scripts/probe_platform_patch.exs, g4, matched JC-phase reflector af>=3 windows): KILL decisive — patching ONLY own-y to 23.45 on good ground windows silences X (mean −0.99 → −6.0 whole-window, max −2.9, fire 0); RESTORE partial — y→0 on platform windows recovers ~4 logits (−4.6 → −0.7) but fire stays ~0-2%: the GRU's accumulated squat history carries the rest. Verdict: **grounded-at-height is the OOD channel; the absorber is own-y + history, NOT stage flags** → multi-stage fixtures (#33) should genuinely fix it; DL likely safe offline only because its platforms are too high to land on mid-cycle. (b) g6 destabilization (probe_opponent_dependence.exs --fine): **g6's static overfit = opponent-PERCENT as cycle clock** (percent-only dB 3.78/dX 3.33/flip 0.23, exceeding the 120-unit teleport; x_small +20 near-zero so the far-x delta is likely offstage-flag flip) — the dummy's percent climbs with every shine, a metronome a human never provides. g4 instead reads opponent ACTION (dX 2.62) with weak percent coupling — the healthy profile. Falsifiable prediction: freeze the dummy's percent and g6's stand chains should degrade; g4's shouldn't. |
| 2026-08-04 | W2 | **Opponent-dependence probe: blindness hypothesis REFUTED, replaced by something better — the score INVERSELY ranks human performance.** scripts/probe_opponent_dependence.exs (common state distribution = g4's deterministic FD stand run; perturbations: opponent +120x teleport, state-neutralize; metric: mean B/X logit delta): g6_sp1 3.84 / g2_mdq_ss 1.83 / g4_d2mix 1.34 — vs human shines 0 / ~25 / 40. Perfect inverse monotonic ordering at n=3. Static overfit is NOT ignoring the opponent; it is overfitting TO the static opponent (dummy state becomes cycle context; any perturbation — i.e. a human — destabilizes). Wired into promote_check.sh as advisory rung 0 (offline, seconds, runs first; LOW=robust HIGH=red-flag, reference numbers printed). Caveats: one probe replay, delay-id 3 for all three, n=3 calibration. |
| 2026-08-04 | W1+infra | **336-dim instrument extension SHIPPED + X-head silence CONFIRMED at logit level.** Extension: `Activations.embed_frames/3` + `embed_config_for/1` (policy-config-aware embedding: queue_depth, with_delay_id, use_prev_action; delay_id REQUIRED for with_delay_id policies — defaulted ids are behavioral modes); threaded through `Activations.capture`, `BasinRollout` (entry builders + closed loop with decoded-controller ring), `CycleSim.rollout` (per-policy entries in cycle_sim.exs), `probe_replay_basin.exs`, `probe_cycle_margins.exs` (`--delay-id`). 10/10 targeted tests green; verified end-to-end on g4×YS (77.8% raw B parity). Confirm result (g4, delay-id 3, YS trio, plat=y>15 grounded): **platform X_fire = 0.0000 (r2 n=2668, r3 n=2727), X_max NEGATIVE (−0.2)** vs ground X_fire 0.14–0.38, X_max 3.4–3.9; B_mean stays ~+0.2 on plat. The JC head is hard-off in platform context while the B motif persists — the absorber at the head level. Remaining W1: floor-test the platform-landing detector; wire as cycle-4 anchor. |
| 2026-08-04 | W1 | **Absorber entry NAMED (model-free forensics, scripts/probe_absorber_entry.exs on the YS trio): the absorber is the PLATFORM, and the broken link is the JC press.** Basin frames are 97%+ at platform height in both absorbed runs (r2 1763/1806, r3 1259/1303); absorbed runs spend ~2750/3482 frames on platforms while good r1 touched one for 42 frames and left. Entry trajectory (r2 t1841-1878): mid-cycle shine-jump rises 29f and LANDS on the left platform (y=23.5, YS platforms are the game's lowest) → Squat/SquatWait spell starts on landing. On-platform behavior: stick FULL DOWN every frame, B held 62-90%, **X pressed ZERO times in ~3000 platform frames** — the policy keeps running the down+B multishine motif but the jump-cancel never fires in platform context (the exact single-frame-critical phase margin cartography flagged), so it degenerates to shine-hold/squat. Matches the live "holds shine on DL" observation. r3 texture differs (many <120f spells, no single long one) — the detector must key on platform-context entry, not spell length. Entry event for cycle-4 snippet anchoring: **grounded landing at y>15**. Remaining: logit-level confirm (X-head margin on plat vs ground) blocked on 336-dim queue/delay-id embed support in the interp instruments (crouch-era scripts + BasinRollout are default-layout only, cf. dagger_drill.exs:1123); then floor-test the detector per GOTCHA #79. |
