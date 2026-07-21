# Interp roadmap v2 inputs: research survey (2026-07-20)

Three-agent literature sweep (Anthropic interp 2024-26; SSM/Mamba interp;
Mamba game-agent practice), synthesized against ExPhil's venue: 2-layer
256-dim GRU/Mamba policies, complete per-frame ground truth, pure-Elixir
constraint, checkpoint zoo + DAgger round history. Full agent reports are
condensed here; arXiv/source links inline.

## TL;DR decisions this survey forces

1. **Netplay: stay WINDOWED for v1.** Windowed inference is rollback-correct
   for free (a rollback = recompute the window with corrected frames = what
   we do every frame). The stateful O(1) path is a latency/memory
   optimization, NOT a correctness need — and windowed-trained Mamba does
   not safely carry state past training length without a cheap
   state-passing finetune (arXiv:2507.02782: ~500 steps fixes it). Fallback
   design with precedent: slippi-ai's fixed 18f delay, no rollback at all.
2. **Mamba line caveats are cheap to check:** (a) verify A_log/dt/D stay
   f32 under bf16 training (official practice); (b) dump the learned dt
   distribution post-training — piled at clamp boundaries = memory-horizon
   fight; (c) prefer Mamba-2 for any new line (DRAMA: better + 2-8x faster
   at exactly our scale: 2 layers, d_state 16); (d) literature predicts
   GRU≈Mamba at 60-frame horizons — our own bake-off card (mamba 8.8/9 vs
   GRU 6.0/9) is currently BEATING that prediction, which makes the
   cross-arch feature comparison more interesting, not less.
3. **Probe/steer Mamba at the trunk OUTPUT first** (diff-of-means transfers
   as-is: arXiv:2404.05971 did exactly this on Mamba), THEN the recurrent
   state h_t — a one-shot state edit persists-and-decays through frames
   ("play scared for ~1s"), a capability the GRU pipeline never had.
4. **Every probe claim gets a causal check** before we believe it:
   arXiv:2606.00930 showed only ~5% of components with a feature's
   representational signature were causally load-bearing in Mamba-2. We
   already learned this lesson behaviorally (knowing-without-acting); the
   literature now says it's the default expectation.

## Adoption queue (ranked by effort/payoff for us)

**Days-scale, immediate:**
- **Persona-vector pipeline applications** (Anthropic, Aug 2025): we have
  the vectors; the unmined value is (a) *monitoring* — project every
  checkpoint's activations onto known bad-habit directions (shield-lock
  relapse dashboard across rounds); (b) *data attribution* — score each
  candidate pool replay by how far it moves projections onto bad-habit
  vectors, filter BEFORE training (composes with #35 conversion
  weighting); (c) *preventative steering during training* — independent
  validation of our #36 probe-as-regularizer design from the LLM world.
- **SAE ground-truth eval harness**: grade features by causal/downstream
  effect, not activation examples (Anthropic May 2026: look-alike features
  have different causal effects); benchmark SAE features as classifiers vs
  our linear probes (Oct 2024 protocol). This is also where ExPhil can
  CONTRIBUTE: exact feature-vs-ground-truth scoring is something LLM labs
  structurally cannot do.
- **Hidden-attention maps for Mamba** (arXiv:2403.01590): the selective
  scan is exactly a data-dependent attention matrix — materialize it
  per-decision ("which past frames did this button press depend on"),
  validate against known reaction windows. Trivial at our scale.
- **Mamba probing gotchas** to bake into Activations before the cross-arch
  run: conv off-by-one (current-frame info lands in the state at t+1..t+3
  — lag probe labels); probe pre-gate SSM output, gate branch z, and
  post-gate separately (the SiLU gate is where linear attribution breaks:
  MambaLRP); check Δ for episode-boundary "sinks" and exclude those frames.

**1-2 weeks:**
- **Stage-wise diffing then crosscoders across DAgger rounds**: start by
  briefly fine-tuning existing SAEs per round and inspecting moved
  features (Dec 2024); escalate to a proper crosscoder with the Feb 2025
  designated-shared-feature sparsity fix (without it, diff features come
  out polysemantic). DAgger rounds are narrow fine-tunes — expect vanilla
  crosscoders to struggle (Delta-Crosscoder, arXiv:2603.04426).
- **Cross-architecture GRU-vs-Mamba crosscoder** (validated by
  arXiv:2602.11729): upgrades our planned "same probes on both trunks"
  comparison into a shared-dictionary feature-level answer to "do the two
  backbones learn the same game?" — publishable; nobody else can validate
  the answer against ground truth.
- **Window-averaged SAEs** (turn-averaging analog, Anthropic Jun 2026):
  train SAEs on activations averaged over interaction chunks (a combo, an
  edgeguard, a neutral exchange) instead of per-frame — surfaces
  tactical-level features instead of 60Hz input echoes.
- **Manifold geometry for continuous variables** (Manifolds paper, Oct
  2025): percent/distance/timers are almost certainly manifolds, not
  sparse features. Fit them with exact labels, steer ALONG the manifold
  (translate represented percent), verify behaviorally. Also a check on
  our SAEs: do they shatter known-continuous quantities into range
  features?
- **Timescale map of the Mamba state**: per-channel effective decay
  spectrum mean(exp(Δ·A)) over real gameplay = which channels are
  reaction-speed vs episode-memory. Paired with time-lagged probes
  (probe h_t for facts at t−k) this gives per-feature memory-horizon
  curves; Stuffed-Mamba scaling (arXiv:2410.07145) predicts ~80-frame
  forget threshold and ~600-frame recall ceiling at d_state=16 — check
  our training window (60) sits below the forget threshold (it does,
  barely; a length curriculum with state passing is the fix if we ever
  need match-scale memory).

**2-4 weeks (the flagship):**
- **Attribution graphs / circuit tracing at 256-dim** (Anthropic Mar
  2025): cross-layer transcoder with ~2-16k features (vs their 30M),
  linearize by freezing GRU gates / Mamba selective params for a given
  window, build the feature→feature graph for one decision ("why did it
  shield-grab here"), validate every edge with replay counterfactuals.
  Their worst limitations (unvalidatable graphs, untraced QK) largely
  dissolve at our scale + ground truth. Error nodes become a measurable
  honesty metric. This is THE next rung after steering.
- **Auditing game** (Marks et al. Mar 2025): plant a secret behavior in a
  checkpoint (e.g. trigger-conditioned DI habit), blind-audit with our
  own stack, score which tools find it. Turns the whole interp stack into
  a benchmarked instrument; closest thing to a transferable
  safety-relevant result from this lab.
- **WriteSAE-style rank-1 write atoms** (arXiv:2605.12770) as SAE variant
  #6 for the Mamba state: the state is a MATRIX; dictionary atoms shaped
  like the model's own writes (u⊗v), giving a principled state-steering
  basis.

**Analysis-layer follow-ons:** J-lens/output-poised subspace decomposition
(Jacobian hidden-state→heads, "acting now" vs "remembering for later"
split); playstyle-space PCA over condition-averaged states (assistant-axis
recipe); scalar-gain steering (multiply a causal subspace 2-5x instead of
adding a vector: arXiv:2602.22719).

**Explicitly not applicable:** natural-language autoencoders (needs an LLM
verbalizer), introspection protocols (no verbal channel), SSM state
quantization (our whole state is ~80KB).

## Mamba-for-game-bots practice notes (beyond interp)

- **Rollback fit**: full recurrent state ≈ 80KB f32; a ring buffer of the
  last ~10 frames of state is <1MB and snapshot/restore is a memcpy —
  categorically nicer than transformer KV caches. But per-step dispatch
  overhead in Nx/XLA at batch 1 is the real cost of resimulation — if we
  build stateful, benchmark 7 sequential tiny steps, not one.
- **Windowed vs stateful**: DeMa (arXiv:2405.12094) explicitly chose the
  windowed operating mode over stateful at policy scale (influence decays
  ~exponentially; long sequences added compute, not performance). Our
  window-60 line is on the literature's good side.
- **Small-data BC advantage is real** (MaIL, CoRL 2024: Mamba beats
  transformer policies by up to ~30% with limited demos, advantage fades
  with scale) — matches our replay-limited DAgger regime and may explain
  the bake-off card gap.
- **Dropout**: never inside the selective scan; residual/output dropout or
  drop-path between blocks only. Small-data Mamba overfits without it.
- **The honest case against**: at 60-frame horizons, minGRU/minLSTM-class
  recurrence matches Decision-Mamba-class models in published RL results
  (arXiv:2410.01201); our differentiator is data + the DAgger loop, not
  the mixer. The bake-off's job is to keep beating that null hypothesis
  with cards, not vibes — and if the mamba line wins, prefer Mamba-2.

## Roadmap wiring

- P4 (reaction-vs-guessing) gains the hidden-attention maps as a second,
  cheaper instrument alongside per-timestep logit lens.
- P5 (curation) absorbs persona-vector data attribution.
- P6's SAE half gets the eval harness + window-averaging + (mamba)
  rank-1 atoms; steering gains scalar-gain + state-injection variants.
- New P7 candidate: attribution graphs. New P8 candidate: auditing game.
- #36 probe-as-regularizer now has external precedent (preventative
  steering) — cite it in the r15 writeup.
