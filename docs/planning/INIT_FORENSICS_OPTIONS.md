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

### 2026-08-02: option 6 productionized — probes DEFAULT-ON + early-reject

Task #2 of the backlog ("kill the seed farm"). Shipped:
- `ExPhil.Interp.CycleMargins` (lib): the jc/aerial/ground margin
  instrument extracted from probe_cycle_margins.exs, per-epoch-able on
  the TEACHER FIXTURE's own edges (one batched forward over <=512 event
  windows). Alignment fact learned: on training frames the applied
  controller lags state by one — event family must be read from the
  PRE-edge frame (current-frame classification finds 0 events; prev
  finds all 373).
- train_multishine_policy.exs: basin rollout + fixture margins default
  ON (--no-probe-basin to disable), both in one JSONL row per probe
  epoch. `--reject-at N --reject-on basin|margin|either` halts WITHOUT
  export, writes <out>.rejected.json, exit 3. Margin rule: jc_flip>0.5
  with n>=5 (the 10/10 sign separator).
- dagger_drill.exs: basin probe + basin-only reject (exit 6), guarded to
  compatible embed configs (prev-action, no queue/delay-id — extending
  BasinRollout to those configs is task #5 territory).
- Farms unchanged: rejected seeds export no .bin, existing [ -f ] guards
  skip their Dolphin evals.
- Verified: epoch-1 reject on an untrained net (both entries absorbed,
  jc_flip 1.0 — textbook breaker signature), exit 3, marker written.

Remaining option-6 gap = task #3: probe entries still can't catch
OPENING deaths (needs each seed's own opening trajectory as an entry).

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

### 2026-07-28 midnight session (options 2 + 6 executed; fix tested live)

**Option 2 DONE (`probe_b_attribution.exs`).** Gradient x input on the
deep-basin B logit, with the prev-B dim located separately
(discover_dims' probe controller never toggles B — its :prev_action
group misses the load-bearing dim; fixed in-script). Result: attribution
MAGNITUDE does not separate outcomes (g attends prev-B 0.109 ~= a's
0.093); combined with the boundary gap's SIGN it completes the
taxonomy — three circuit solutions to the same labels:
- prev-B NEGATIVE coupling (release-on-held = the edge rule): a, c, d
- prev-B POSITIVE coupling (perseveration — copy what I just did):
  g, h, j = the hold-B absorbers
- af-parity/state-driven alternation (near-zero prev coupling): b, i, k
Note the structural artifact: prev_b share is necessarily 0 in the
no-B variant (gradient x input on a zero input) — only the held variant
is informative for that dim.

**Option 6 DONE + RecoverySynth fix tested (commit 4afb006).** Tail
frames renumbered -> prev threads -> the release rule became learnable.
Seeds m, n, o trained with per-epoch mental rollouts
(`--probe-basin`, JSONL curves):
- Convergence 2-4x faster (20/34/40 epochs vs ~80+): the alternating
  labels went from half-noise to predictable.
- The lottery, watched: ALL THREE seeds acquire covered-basin escape
  between epoch 1 and 10 (absorbed -> esc@2) — on the fixed synthesis,
  covered-region escape is no longer init luck.
- Live: n = 104-124/min chains 20/10/8 (SECOND SEED EVER past chain 10;
  caveat: 12-17% staleness, clean re-eval queued), o = 77-84/min clean,
  m = DEAD (0 shines).

**Farm 4 (overnight): the fix's live verdict.** Fixed-synth n=7: escape
3/7 (n, o, q) — the ">> 6/12" prediction is FALSIFIED as stated. But the
sustain distribution transformed: 2 of 3 escapees are chains-18+
sustainers (n re-evaled CLEAN: 114-141/min chains 14/18/17; q: 116/min
chains 21/10), where the old recipe produced 2 sustainers in 12 seeds.
Covered-basin escape is now universal (probe curves: ALL fixed seeds
esc@2 offline by epoch 10-30, including the live-dead p/r/s) — the
lottery consolidated into the OPENING route, which now kills outright
(~4/7) instead of leaving seeds mediocre. Early-reject caveat: the
current probe entries CANNOT catch opening deaths (p/r/s pass them);
the discriminating state is each seed's OWN opening trajectory, which
argues for opening-synth coverage rather than a better probe entry.

**SUSTAIN MECHANISM FOUND (margin cartography + break species,
`probe_cycle_margins.exs` + `analyze_break_phases.exs`).** Event-margin
probing over each seed's own replay (measure at the bot's successful
button EDGES — v1 labeled whole phases "should press" and produced
flip=1.0 for the champion; presses are events, holds are not presses):

- **jc_event X-margin separates sustainers from breakers 10/10 by
  SIGN**: a +2.86, c +1.28, n +1.23, q +0.26 (flip 0.0) vs b -7.75,
  d -4.83, e -4.16, i -4.33, j -4.74, k -5.84 (flip ~1.0). Sustainers
  hold X positive through the JC window (fat plateau — harmless, the
  edge already registered); breakers emit single-frame X spikes that
  evaporate under one frame of drift. Known artifact: logits align one
  frame late vs the controller stream (bridge delay), so aerial-event
  numbers read the NEXT decision — the JC separation survives because
  it is asymmetric across tiers under the same misalignment.
- **Break species differ by tier** (chains>=2 only): sustainers a/n/q
  break in the AIR (air_shine/empty_hop — the one-frame aerial B is
  their only remaining weakness); old-recipe chains-2-4 escapees
  b/i/k break on the GROUND (other_action 60-95% — they voluntarily
  leave the cycle; alternation decays); o is pure empty_hop (JCs fine,
  misses aerial B). "Why did the good run go well": fat X-hold at the
  JC + staying in the loop on the ground; only the aerial shine still
  breaks it.
- Next: fix the 1-frame alignment for citable aerial margins; the
  jc-margin sign is an early-reject-friendly SUSTAIN predictor
  candidate (add to per-epoch probes); training lever suggested by the
  mechanism: reward/label X-holds over X-spikes through the JC window
  (the expert's own labels already hold X 2 frames — check which seeds
  copied that vs compressed it).

**FARM 5 (combined recipe, 2026-07-28 afternoon): REGRESSED — and the
root cause is a general lesson about synthesis from policy replays.**
Seeds t/u/v/w (crouch + opening + X-hold): 0-1/4 escape, u/v absorbed at
frame 104 via the exact route opening-synth targets. Root cause
(commit ccae060): build_opening's extra_sources lead-ins kept the DEAD
seeds' recorded controllers as labels — the farm was trained to IMITATE
the absorbed policies' opening behavior. **Rule: any synthesis that
harvests states from a policy replay must relabel with the expert;
recorded controllers are only valid labels when the recorder was the
teacher.** (DAgger knows this — it is the entire point of relabeling —
and build_crouch never hit it because its lead-ins come from the teacher
fixture.) Fixed + regression-tested; farm 6 reruns deconfounded (arm A
opening-fix only, arm B + X-hold), since farm 5 also confounded the
X-hold intervention with the poisoning. Interesting residue: seed t
(marginal, 6-16/min) reached chains 4-10 — noted, not interpreted.

**FARM 6 (two-arm, post-relabel-fix, 2026-07-28 13:11): the combined
recipe WORKS — new all-time champion.** All clean staleness (1.1-1.8%),
3x60s vs idle:

| arm | seed | self/min | chains | notes |
|---|---|---|---|---|
| A: +opening(fixed) | x | 87-93 | 3-6 | escape |
| A | y | 108-115 | 7-15 | near-sustainer, SPIKE jc-margins |
| B: A + X-hold(3) | z | **121-147** | **16-27** | ALL-TIME RECORD both axes |
| B | zz | 49-56 | 1 | metronome: steady singles, never JCs |

- **Escape 4/4, opening deaths 0/4** (baselines: 3/7 fixed recipe, 6/12
  original). Opening-synth with expert relabeling closes the opening
  absorber at n=4.
- **z beats every seed ever measured** (prev bests: 129/min, chains 22
  by ms_crouch_a) — trained in 17 epochs on the full recipe:
  `--synth-recovery --synth-crouch --synth-opening --opening-replays
  <dead-seed openings> --x-hold-extend 3 --prev-action`.
- jc-margin check: z has the positive-plateau signature (+0.617,
  flip 0.0); y sustains at chains 15 WITH spike margins — the plateau
  is sufficient-not-necessary at n=4. zz emits no jc events at all
  (single-shine metronome — a NEW low-sustain phenotype worth forensics).
- Sustain variance persists within arms (z 27 vs zz 1) — X-hold is not
  a sustain guarantee; per-arm n=2 cannot rank the arms yet. Next: seed
  farm the full recipe (the operational cost question is now "how often
  does the recipe produce a z"), and forensics on zz's metronome.

**The remaining lottery is the GAME OPENING.** m absorbs at frame 104
via the same entry route as g (324x20 > 29x10 > 42x30 = spawn-platform
fall -> crouch, never shines). Rollout cross-test: m reproduces its
death offline from its own m@104 entry; n and o ALSO fail from m's
hole (they live-escape only because their own openings never dig it);
**ms_crouch_a escapes m's hole in 1 frame** — the champion is a
genuinely different tier. The uncovered state region = basin windows
whose HISTORY is entry-animation frames (fixture contains none). Next
coverage target: OPENING SYNTH — graft crouch tails onto entry-route
lead-ins. Prediction: kills the m/g failure mode; escape rate -> ~1.
Overnight farm 4 (seeds p-s + clean n re-eval) pins the fixed-synth
rate meanwhile.
