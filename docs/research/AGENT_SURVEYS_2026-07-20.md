# Research-agent surveys, 2026-07-20 (verbatim)

Three parallel literature sweeps run the evening mamba_full landed. The
DECISIONS distilled from these live in
`docs/planning/INTERP_NEXT_RESEARCH_2026-07-20.md` (adoption queue +
roadmap wiring); this file preserves the full agent reports for future
roadmapping — sources, numbers, and per-paper applicability notes that
the synthesis compressed away.

Survey briefs: (1) Anthropic mech-interp 2024-26 vs our
ground-truth-validated 256-dim venue; (2) SSM/Mamba-specific interp:
where to probe/steer, state capacity, conv/gate gotchas; (3) Mamba as a
real-time game-agent backbone: stability, stateful inference, rollback,
and the case against.

---

# Report 1: Anthropic Mechanistic Interpretability, 2024 – July 2026

Verified against transformer-circuits.pub directly (index fetched
2026-07-20). Third-party "J-Space consciousness" blog coverage is hype
around one real paper (Gurnee et al., July 2026) — the primary sources
below are what actually exists.

## Timeline of the actual publications (primary-source verified)

| Date | Work |
|---|---|
| May 2024 | Scaling Monosemanticity (Templeton et al.) |
| Oct 2024 | Sparse Crosscoders; Dictionary Features as Classifiers |
| Dec 2024 | Stage-Wise Model Diffing |
| Feb 2025 | Insights on Crosscoder Model Diffing |
| Mar 2025 | Circuit Tracing (Ameisen et al.) + On the Biology of a Large Language Model (Lindsey et al.); Auditing Hidden Objectives (Marks et al.) |
| May–Jul 2025 | Open-source circuit-tracer + Neuronpedia; Tracing Attention Computation Through Feature Interactions; Automated Auditing agents |
| Aug 2025 | Persona Vectors (Chen et al.) |
| Oct 2025 | Emergent Introspective Awareness (Lindsey); When Models Manipulate Manifolds (Gurnee et al.) |
| Dec 2025 | Activation Oracles |
| Jan 2026 | The Assistant Axis |
| Feb 2026 | The Persona Selection Model |
| Apr 2026 | Emotion Concepts and their Function in a Large Language Model (Sofroniew et al.) |
| May 2026 | Natural Language Autoencoders; Circuits Update: features via downstream connections; HeadVis |
| Jun 2026 | Circuits Update: turn-averaged SAEs |
| Jul 2026 | Verbalizable Representations Form a Global Workspace (Gurnee et al., 2026-07-06) — the "J-lens" paper |

## Tier 1 — Directly answers "what's the next rung after steering"

### 1. Circuit Tracing / Attribution Graphs (cross-layer transcoders)

**Source:** "Circuit Tracing: Revealing Computational Graphs in Language
Models" + "On the Biology of a Large Language Model",
transformer-circuits.pub, March 2025; attention-QK extension July 2025;
open-source circuit-tracer library mid-2025.

**Method:** Train a cross-layer transcoder (CLT): sparse features that
read the residual stream at layer L and reconstruct MLP outputs at all
layers >= L. Then, for a specific prompt, freeze attention patterns and
normalization denominators — this makes all feature-to-feature
interactions *linear*, so you can build an attribution graph where each
feature's preactivation is exactly the sum of incoming edges. "Error
nodes" absorb reconstruction gaps; pruning keeps the ~10% of nodes
carrying ~80% of the explanation. Hypotheses from graphs are then
validated by perturbation (feature ablation/patching), with ~0.72
Spearman correlation between predicted edge influence and actual
ablation effects.

**Requirements at Anthropic scale:** 300K–30M CLT features, billions of
tokens of activations, significant training compute, plus an interactive
graph UI. The stated limitations: QK-circuits (why attention attends
where it does) are unexplained, and error compounds through layers.

**ExPhil applicability: HIGHEST. This is the canonical next rung after
steering, and it is *more* tractable at 256-dim than at frontier scale.**
- A CLT for a 2-layer, 256-dim backbone needs maybe 2K–16K features —
  trainable in minutes-to-hours in Nx, and we already have 5 SAE
  variants plus activation capture, so the transcoder variant
  (reconstruct block *outputs* from block *inputs*, decoders per
  downstream layer — with 2 layers this is nearly trivial: layer-0
  features get two decoders) is a small delta.
- The linearization story adapts cleanly: for the GRU, freeze the
  update/reset gate values for the specific input sequence (the analog
  of freezing attention patterns) — the remaining computation is linear
  in the candidate-state path. For Mamba, freeze the input-dependent
  selective-SSM parameters (delta, B, C gating); the frozen system is a
  linear time-varying recurrence, which is exactly what attribution
  needs. Notably, the biggest hole in Anthropic's method — untraced QK
  circuits — has a much smaller analog here (gate formation), and at
  256-dim you can afford to trace gate attribution too.
- The validation loop is where ExPhil is uniquely strong: Anthropic
  validates graphs with expensive, indirect perturbations; we validate
  against ground-truth game state and cheap replay counterfactuals /
  live Dolphin A/Bs, and score the outcome objectively. E.g., trace "why
  did it shield-grab here" end-to-end from input embedding features to
  controller-head logits, then confirm every claimed edge by ablation in
  replay.
- Error nodes become a *measurable honesty metric*: with complete ground
  truth you can report exactly how much of each decision the graph fails
  to explain.

**Effort:** Medium. ~2–4 weeks: CLT training objective on captured
activations (small), per-prompt linearized attribution (a backward pass
through frozen gates — a structured generalization of existing
gradient×input machinery), pruning, and a graph visualization (even
Graphviz/JSON dump suffices initially). No Python dependency issues —
it's all linear algebra Nx already does.

### 2. Crosscoders + Model Diffing (the r1..r15 / GRU-vs-Mamba tool)

**Source:** "Sparse Crosscoders for Cross-Layer Features and Model
Diffing" (Oct 2024); "Stage-Wise Model Diffing" (Dec 2024); "Insights on
Crosscoder Model Diffing" (Feb 2025). Community follow-ups:
cross-architecture diffing with crosscoders (arXiv 2602.11729, 2026) and
Delta-Crosscoder for narrow fine-tuning regimes (arXiv 2603.04426, 2026).

**Method:** A crosscoder is an SAE with one shared feature dictionary
but separate encoder/decoder weights per model (or per layer). Trained
jointly on activations from two models, each feature's per-model decoder
norms tell you whether the feature is shared, model-A-exclusive, or
model-B-exclusive — the exclusive set *is* the diff. Key pitfall (Feb
2025): exclusive features come out polysemantic and dense because shared
features soak up the easy variance; fix by designating shared features
with a reduced sparsity penalty so exclusive features stay monosemantic.
Stage-wise diffing (Dec 2024) is the cheap alternative: take an existing
SAE, briefly fine-tune it on the new model, and inspect which features
moved.

**Requirements:** Paired activation datasets from both models on the
same inputs; SAE-scale compute times number of models. No gradients
through the target models needed.

**ExPhil applicability: YES — exactly the right tool for both diffing
questions, and the checkpoint zoo makes it unusually clean.**
- *DAgger rounds r1..r15:* run the same replay corpus through adjacent
  checkpoints, train a crosscoder per pair (or one across many
  checkpoints — the shared-dictionary formulation extends to N models).
  Exclusive/changed features answer "what did round 12 actually learn?"
  — validated against ground truth (did the r12-exclusive feature fire
  on the situations where r12's win rate improved?) and causally
  (ablate it, replay, measure). Since DAgger rounds are narrow
  fine-tunes, the 2026 Delta-Crosscoder result (narrow fine-tuning
  regimes break vanilla crosscoders; diff-focused variants help) is
  directly relevant; stage-wise diffing may be the better first move
  because it reuses existing SAEs — hours, not days.
- *GRU vs Mamba:* both see identical 288-dim inputs and produce 256-dim
  hidden states, so a cross-architecture crosscoder on hidden states
  over the same frames is well-posed (2026 community work confirms
  cross-architecture crosscoders work). This answers "do the two
  backbones learn the same features?" — genuinely publishable at a scale
  where the answer is checkable.
- Apply the Feb 2025 shared-feature sparsity trick from day one; it is
  the single most important operational detail.

**Effort:** Low–medium. Stage-wise diffing on existing SAEs: days. Full
crosscoder: 1–2 weeks (an SAE with a model-indexed decoder — small
extension to the sibling SAE library).

### 3. SAE evaluation practices when ground truth exists

**Source:** "Using Dictionary Learning Features as Classifiers" (Oct
2024); Circuits Update May 2026 (features via downstream connections);
Circuits Update June 2026 (turn-averaged SAEs); Aug 2024 update
(interpretability evaluations).

**Findings that matter:**
- *Downstream connections (May 2026):* features that look identical by
  top-activating-examples and unembedding projections can have
  *different causal effects*; characterizing a feature by its downstream
  connectivity predicts steering behavior better than activation
  patterns. Lesson: activation-based interpretation is not enough —
  grade features by their causal role.
- *Features as classifiers (Oct 2024):* the honest test of an SAE
  feature is whether it beats/matches a probe on a downstream task, not
  whether its max-activating examples look clean.
- *Turn-averaged SAEs (June 2026):* training SAEs on activations
  averaged over a whole conversational turn (instead of per token) kills
  the flood of "boring" low-level features and yields features
  describing transcript-level properties; generalizes to turns 150x
  longer than training.

**ExPhil applicability: HIGH, and this is where ExPhil can *contribute*,
not just consume.** Anthropic's central unsolved problem is that they
can never fully check a feature label. We can: for every SAE feature,
compute exact mutual information / F1 against every ground-truth state
variable, then compute its causal score via ablation+replay. Concretely:
(a) adopt causal characterization as the primary feature metric; (b)
benchmark each of the 5 SAE variants as feature-classifiers against
linear probes; (c) build the turn-averaged analog — *window-averaged
SAEs* over interaction chunks (a combo attempt, an edgeguard sequence, a
neutral exchange) instead of per-frame, which should surface
tactical-level features rather than 60-per-second input echoes.
**Effort:** Low — mostly evaluation harness code; the window-averaged
SAE is a data-pipeline change, not a new model.

## Tier 2 — Strong methodological transfer

### 4. Persona Vectors — the *pipeline*, not the personas

**Source:** "Persona Vectors: Monitoring and Controlling Character
Traits in Language Models", anthropic.com/research, Aug 2025 (Chen et
al.), code released.

**Method:** Fully automated: given a trait name + description, generate
contrastive prompt pairs eliciting/suppressing the trait, take mean
activation difference = trait vector. Three applications beyond
steering: (1) *monitoring* — project activations during
deployment/training onto the vector; (2) *preventative steering* — steer
*toward* the undesired direction during fine-tuning so the weights don't
need to learn it, preserving capability while blocking trait drift; (3)
*data attribution* — score training samples by projection shift to
predict which data will induce the trait before training on it.

**ExPhil applicability: HIGH — we already have steering vectors; the
un-mined value is applications 2 and 3.** (1) Monitor bad-habit
directions (shield-lock relapse, panic-roll) across r1..r15 projections
— a one-day dashboard complementing crosscoder diffing. (2)
*Preventative steering during DAgger training* is the standout — this
is, independently, the design of our #36 probe-as-regularizer. (3) Data
attribution: score each DAgger-collected trajectory by how much it moves
projections onto bad-habit vectors; filter the corpus before the next
round. **Effort:** Low for monitoring/attribution (days); medium for
preventative steering (touches the training loop; done — #36).

### 5. When Models Manipulate Manifolds (feature geometry for continuous variables)

**Source:** transformer-circuits.pub, Oct 2025 (Gurnee et al.).

**Method:** Deep-dive into how Claude 3.5 Haiku tracks character counts
for linebreaking: counts are represented not as discrete features but as
points on curved low-dimensional manifolds (helix/circle-like),
manipulated by geometric operations; attention computes differences
between manifold positions. Methodologically: targeted probing for
continuous quantities, visualizing the manifold, then intervening along
it.

**ExPhil applicability: HIGH conceptual fit.** Melee state is dominated
by continuous quantities — percent, positions, velocities, timers, stage
boundaries — and a 256-dim recurrent state almost certainly encodes them
on low-dim manifolds rather than as sparse features. This paper is the
blueprint for "how does the bot represent opponent percent /
distance-to-ledge?": fit the manifold with ground-truth labels (we have
exact values every frame — Anthropic had to construct synthetic tasks),
then steer *along the manifold* (e.g., translate the represented
percent) and verify behavior changes accordingly in replay. Also a
caution for SAE work: SAEs shatter manifolds into discretized range
features; check the 5 variants for this failure mode against
known-continuous labels. **Effort:** Low–medium (1–2 weeks; builds
directly on the probe/LEACE stack).

### 6. Auditing games as a validation protocol

**Source:** "Auditing Language Models for Hidden Objectives", Marks et
al., Mar 2025; "Building and Evaluating Alignment Auditing Agents",
alignment.anthropic.com, Jul 2025.

**Method:** Red team trains a model with a deliberately planted
objective (RM-sycophancy via synthetic docs + RL); blind auditing teams
try to uncover it with different tool access. Teams with
interpretability tools (SAE features surfaced the hidden objective) and
data access succeeded; the game format measures which interp tools
actually *work*. The 2025 follow-up automated the auditors as agents
(42% success uncovering hidden goals).

**ExPhil applicability: HIGH as *methodology* — this is how you find out
if your interp stack is real.** Train a checkpoint with a planted secret
behavior (e.g., "always DI toward the opponent when above 80% on
Battlefield", or a trigger-conditioned bad habit), then blind-audit it
using only probes/SAEs/attribution graphs — no peeking at training data.
Because the plant is exact, this yields a quantitative benchmark for
every method (which tools find the plant, how fast). Also the closest
thing to a transferable safety-relevant result from the lab: evidence
about which interp methods detect planted behaviors at small scale.
**Effort:** Medium (~1–2 weeks: one planted-behavior training run +
audit protocol; reuses everything else).

## Tier 3 — Adaptable ideas, lower priority

### 7. Global Workspace / J-lens (Jul 2026)

**Source:** "Verbalizable Representations Form a Global Workspace in
Language Models", transformer-circuits.pub/2026/workspace, 2026-07-06
(Gurnee et al.). (The paper behind the "J-Space" blog hype.)

**Method:** The "J-lens" (Jacobian lens) identifies residual-stream
directions corresponding to tokens the model is *poised to produce* — a
small privileged subspace ("J-space", <10% of activation variance, ~25
concepts at a time) that supports verbal report, directed modulation,
and reasoning; causal effects concentrate almost entirely in it, atop a
large mass of automatic processing the model can't report on.

**ExPhil applicability: MODERATE, cheap to try.** The direct analog:
compute Jacobians from the 256-dim hidden state to each of the 6
controller heads and ask whether causal influence concentrates in a
small "output-poised" subspace vs. a larger automatic mass (likely
tracking state that only matters frames later, given recurrence). At
256-dim the Jacobian is trivially cheap. Gives a principled
decomposition of the hidden state into "acting now" vs. "remembering for
later" — useful for deciding where steering interventions should live,
and a nice frame for GRU-vs-Mamba comparison. **Effort:** Low (days) for
the core analysis; the cognitive-workspace framing doesn't transfer, the
subspace decomposition does.

### 8. Activation Oracles (Dec 2025)

**Source:** alignment.anthropic.com/2025/activation-oracles.

**Method:** Train an LLM to answer arbitrary natural-language questions
about another model's activations, injected as embeddings into layer 1.
Generalizes to auditing tasks on activations from models it never
trained on. Non-mechanistic, can confabulate.

**Applicability: LOW–MODERATE.** ExPhil's ground truth already gives the
honest version (probes decoding full game state). The one transferable
idea is *generalization of the reader*: train a single decoder network
mapping hidden states to structured game-state summaries across all
checkpoints, then use its failures/surprises on a new checkpoint as an
audit signal (what does r15 encode that the shared reader can't
decode?). Only worth it after crosscoder diffing exists.

### 9. Natural Language Autoencoders (May 2026)

**Source:** "Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations", transformer-circuits.pub, May 2026.

**Method:** Two copies of the target LLM trained jointly: a Verbalizer
converts an activation vector into English text (GRPO with
reconstruction as reward), a Reconstructor maps text back to the
activation; 0.6–0.8 fraction-of-variance-explained. Found "unverbalized
evaluation awareness" (model internally suspects it's being tested
without saying so). Confabulation and steganography risks acknowledged.

**Applicability: NOT DIRECTLY — requires an LLM-scale verbalizer, no
Elixir path.** The structural idea has a cheap ExPhil analog: an
autoencoder whose bottleneck is a *structured game-state schema* instead
of English — activation -> predicted symbolic situation summary ->
reconstructed activation, with reconstruction FVE measuring how much of
the hidden state is "about" nameable game state vs. dark matter. That
residual fraction is itself a finding. Medium effort; interesting later.

### 10. Emergent Introspective Awareness (Oct 2025)

**Method:** Inject concept steering vectors mid-forward-pass; ask the
model whether it notices an injected thought. **Applicability: NOT
APPLICABLE** — requires verbal self-report; a controller-head GRU has no
channel to report internal states. The injection protocol is subsumed by
the existing steering + probing loop.

### 11. Assistant Axis (Jan 2026) / Persona Selection Model (Feb 2026) / Emotion Concepts (Apr 2026)

**Methods:** Persona space built from activations averaged over
role-play conditions; PC1 is "assistant-likeness," steering along it
stabilizes/destabilizes the default persona. Persona-selection:
post-training selects among pre-existing personas rather than building
new ones. Emotions: 171 concept vectors extracted by prompted stories,
organized by valence/arousal, causally potent (a "desperate" vector
increases reward hacking; "calm" reduces it).

**Applicability: LOW–MODERATE, as a template.** The shared recipe —
build a *space* of condition-averaged activations across many behavioral
conditions, PCA it, interpret and steer along principal axes — maps to a
"playstyle space": average hidden states over matches vs. different
opponents/characters/conditions across the checkpoint zoo; PC1 is
plausibly an aggression-passivity or quality axis; steering along it is
directly scoreable. The DAgger analog of persona-selection (does
fine-tuning select among pre-existing behavioral modes rather than
creating new ones?) is testable with crosscoder diffs. Effort: low.

### 12. Scaling Monosemanticity (May 2024) — superseded for our purposes

The foundational result is already internalized in the 5-variant SAE
library; its successors are Tiers 1–3 above. Circuits-Updates
refinements worth skimming: topk/gated SAE comparison (Jun 2024),
oversampling rare data (Sep 2024), dictionary-learning optimization
tricks (Jan 2025), dictionary initialization (Oct 2025).

## Direct answers to the four posed questions

1. **Next rung after steering:** Attribution graphs — from "this
   direction causes shield-lock" to "this is the feature-to-feature
   circuit from input embedding to controller head that produces
   shield-lock," validated by cheap interventions. In parallel, the
   persona-vectors playbook (preventative steering + data attribution)
   turns existing steering wins into training-time tools.
2. **Circuit tracing tractable at 256-dim?** Yes — comfortably. Feature
   counts drop from 30M to ~10^4, the linearization trick maps onto
   frozen GRU/Mamba gates, error nodes are directly measurable, and the
   method's worst limitation at scale (unvalidatable graphs, untraced
   QK) largely dissolves with ground truth and cheap counterfactuals.
3. **Crosscoders right for DAgger-round and GRU-vs-Mamba diffing?** Yes,
   with two caveats: use the Feb 2025 designated-shared-feature sparsity
   fix, and since DAgger rounds are narrow fine-tunes expect vanilla
   crosscoders to struggle (Delta-Crosscoder) — start with cheap
   stage-wise diffing and escalate. Cross-architecture crosscoders are
   validated by 2026 community work.
4. **SAE evaluation with ground truth:** (a) grade features by
   causal/downstream characterization, not activation examples; (b)
   benchmark features as classifiers against linear probes; (c)
   window-averaged SAEs over interaction chunks. Exact
   feature-vs-ground-truth alignment scoring plus manifold-shattering
   checks is territory where ExPhil can produce evaluation results
   Anthropic structurally cannot.

**Suggested sequencing by effort/payoff:** persona-vector pipeline
applications (days) -> SAE ground-truth eval harness + stage-wise
diffing (days–1wk) -> crosscoders across r1..r15 (1–2wk) -> CLT +
attribution graphs (2–4wk) -> auditing game to benchmark the whole stack
(1–2wk) -> manifold geometry and J-lens analyses as follow-ons.

Sources: transformer-circuits.pub index; attribution-graphs
methods/biology (2025); crosscoder-diffing-update (2025);
anthropic.com/research: persona-vectors, assistant-axis,
persona-selection-model, emotion-concepts-function;
transformer-circuits.pub/2026: workspace, may-update, june-update;
alignment.anthropic.com: activation-oracles, automated-auditing; arXiv
2602.11729 (cross-architecture crosscoders); arXiv 2603.04426
(Delta-Crosscoder).

---

# Report 2: Interpretability for State-Space Models (Mamba/S4 family), 2023–July 2026

Context assumed: 2-layer Mamba-1-style blocks, d_model 256, expand 2
(d_inner 512), d_state 16, per-frame ground-truth game state, existing
diff-of-means steering pipeline validated on a GRU trunk output, linear
probes + 5 SAE variants in Elixir.

## Ranked findings

### 1. Transformer interp methods mostly transfer to Mamba — including difference-of-means steering, on both activations AND the recurrent state

**Source:** Paulo, Marshall, Belrose (EleutherAI), "Does Transformer
Interpretability Transfer to RNNs?" — arXiv:2404.05971 (Apr 2024, AAAI
2025). Tested Mamba-2.8b and RWKV-v5.

- Contrastive Activation Addition (CAA — exactly difference-of-means
  over paired prompts, h <- h + av) works on Mamba's residual stream,
  comparably to transformers of similar size.
- **They also steered by editing the compressed recurrent state
  directly** — works for both Mamba and RWKV-v5. Combining state +
  activation steering slightly increases effect but is **not additive**
  (activation steering already flows into the state). Some behaviors
  (sycophancy, refusal) barely steered at 2.8b scale.
- Tuned lens and linear probing confirm largely linear representations;
  prompt-vector arithmetic works.

**For the lab:** The GRU-trunk diff-of-means recipe is expected to
transfer to the Mamba **trunk/block output with no changes** — best
evidenced target, step one. State steering is *also* known to work and
is the interesting second target because it's mechanistically different:
a one-shot edit to h_t **persists and decays** through subsequent frames
(via the recurrence), whereas trunk steering must be re-applied every
frame. For a 60fps realtime bot, a persistent state nudge ("play scared
for the next ~1s") is a capability the GRU pipeline never had cleanly.

**Experiments:** (1) replicate GRU steering on Mamba block outputs; (2)
inject the same diff-of-means vector (recomputed in state space: mean
h_t over frames of behavior A minus behavior B, flattened 512x16) into
h_t *once*, and measure behavioral-effect half-life in frames — ground
truth scores it. Compare vs per-frame reinjection; check non-additivity.

### 2. SAEs on the recurrent state itself work, but the state is a *matrix* — dictionary atoms should be rank-1 "writes," not flat vectors

**Sources:** "WriteSAE: Sparse Autoencoders for Recurrent State" —
arXiv:2605.12770 (May 2026; Gated DeltaNet, Mamba-2-370M, RWKV-7).
Earlier: joelburget/mamba-sae (2024, SAEs on Mamba layer activations);
"Superposition and Polysemanticity in Mamba's Selective State Space" —
SSRN 6465858 (2026, 114 TopK SAEs on Mamba-130M/370M vs GPT-2 — Mamba
shows superposition/polysemanticity comparable to transformers, so SAEs
are as applicable).

WriteSAE's key move: train the SAE on the recurrent cache with **rank-1
matrix atoms shaped like the model's own state writes** (outer products,
B_t (x) x_t shaped). Atom-for-write substitution matched the model's
output distribution in ~90% of positions, transferred to Mamba-2 at 88%,
and enabled the first reported **cache-level steering**: writing a
chosen atom direction into a few consecutive state positions at 3x write
norm reliably surfaced target behavior.

**For the lab:** The 5 SAE variants eat flat vectors. For the SSM state,
don't just flatten 512x16: the natural units are (i) per-channel 16-dim
state rows, and (ii) rank-1 write atoms u (x) v, u in R^512, v in R^16.
A sixth SAE variant with rank-1 matrix atoms is a small change in Nx and
matches the 2026 state of the art. Also gives a principled
state-steering basis (inject atoms scaled to typical write norm, not
arbitrary vectors).

### 3. State capacity laws: with d_state = 16, memory horizon is short and quantifiable — reframes what probing can find

**Sources:** "Stuffed Mamba: Oversized States Lead to the Inability to
Forget" — arXiv:2410.07145 (Oct 2024, rev. 2025); "Understanding Input
Selectivity in Mamba" — arXiv:2506.11891 (ICML 2025); "Characterizing
Mamba's Selective Memory using Auto-Encoders" — arXiv:2512.15653 (Dec
2025, AACL oral).

- Stuffed Mamba, with N_S = per-channel state dimension (our d_state):
  robust *forgetting* is only learned when training length exceeds
  T_forget ~= 5.17*N_S (~= **80 frames** at N_S=16), and exact
  passkey-style *recall* horizon scales as T_recall ~= 4.76*(1.365^N_S -
  1) (~= **600–700 frames** at N_S=16). Fits are on language models —
  treat as order-of-magnitude; the qualitative law (linear forget
  threshold, exponential-in-d_state recall ceiling) is robust (R^2 >
  0.999 in their sweeps, including small from-scratch models).
- Under-trained-length models show pathological recency-bias failure:
  they *fail to forget*, causing interference beyond training length.
- ICML 2025 theory: selectivity (input-dependent delta) is precisely
  what lets S6 **counteract memory decay** on demand — memory horizon is
  not fixed per channel; it's gated.
- The AACL paper (autoencoders reconstructing the input sequence from
  the hidden state) shows forgetting is **frequency-biased**: rare
  tokens/entities are lost first (r ~= -0.82 with pretraining
  frequency).

**For the lab:** d_state=16 doesn't prevent interp — it makes it
*easier* (small, enumerable state) — but it bounds claims: don't expect
probes to recover events thousands of frames back; expect a working
memory of roughly one interaction/combo (~1–10s of gameplay), with
anything longer held only via slow channels. Two direct experiments:
(1) **time-lagged probes** — probe h_t for ground-truth facts at t-k for
sweeping k; the AUC-vs-k decay curve is the empirical memory horizon per
feature, checkable against the Stuffed-Mamba scale. (2) **Frequency-bias
check** — bin game situations by training-set frequency and test whether
rare situations decay faster from the state; we have exact ground truth,
which the LM papers didn't. Also confirm training sequence length >= ~80
frames, else expect the "can't forget" pathology (window 60 sits BELOW
this — relevant to any stateful plans).

### 4. Circuit-style analysis works on Mamba; tooling exists; the key architectural motif is an off-by-one from the conv

**Sources:** Ensign & Garriga-Alonso, "Investigating the IOI circuit in
Mamba" — arXiv:2407.14008 (Jul 2024); Wang et al., "Towards
Universality: Studying Mechanistic Similarity Across LM Architectures" —
arXiv:2410.06672 (Oct 2024); tooling: Phylliida/mamba_interp and its
mamba_lens (HookedMamba, TransformerLens-style hooks — PyTorch).

- IOI-in-Mamba: positional Edge Attribution Patching found a real
  circuit; one layer was the bottleneck; **the depthwise conv shifts
  entity info one position forward**; entities were stored **linearly in
  the SSM state**.
- Universality paper (SAE-based, Mamba vs Pythia): most features match
  across architectures; induction circuits are structurally parallel;
  independently confirms the **"Off-by-One motif"** — token info is
  written into the SSM state at the *next* position, because of the
  conv.
- Sen Sharma, Atkinson, Bau, "Locating and Editing Factual Associations
  in Mamba" — arXiv:2404.03646 (COLM 2024): causal tracing and
  ROME-style rank-one editing port to Mamba with transformer-like
  localization.

**For the lab:** Nobody has published circuits on a 2-layer game-playing
Mamba — open territory, and at 2 layers exhaustive patching is cheap.
MambaLens won't help directly (PyTorch), but its hook list is the spec
for Elixir instrumentation: expose per-frame x (pre-conv), post-conv,
delta, B, C, h_t, SSM output pre-gate, gate z, post-gate block output.
**Gotcha:** expect current-frame features to appear at frame t+1..t+3 in
the state (d_conv-window shift) — align probe labels accordingly or
probe accuracy is under-measured by a frame.

### 5. "Attention-like" reformulations give exact per-frame attribution maps — very cheap at this scale

**Sources:** Ali, Zimerman, Wolf, "The Hidden Attention of Mamba Models"
— arXiv:2403.01590 (Mar 2024, ACL 2025): the selective scan is exactly a
data-dependent lower-triangular "attention" matrix alpha_{t,s} = C_t
(prod_{k=s+1}^{t} A-bar_k) B-bar_s, enabling transformer-style
attribution/rollout. Dao & Gu's SSD/state-space duality ("Transformers
are SSMs," ICML 2024, arXiv:2405.21060) is the general form (basis of
Mamba-2). MambaLRP — arXiv:2406.07592 (NeurIPS 2024) — is the principled
attribution alternative; it identified exactly **where naive attribution
breaks in a Mamba block: the SiLU, the selective SSM, and the output
gate multiplication**.

**For the lab:** At seq length ~a few hundred frames, d_inner 512,
materializing the full implicit attention matrix per layer is trivial.
This gives "which past frames did this button press depend on" heat maps
— directly checkable against known Melee causal structure (reaction to a
hitbox N frames ago, DI read, etc.). The cheapest high-value new
capability on the list. MambaLRP's finding is the gotcha list: the gate
(x) SiLU(z) and SiLU nonlinearities are where linear-attribution
assumptions fail — probe pre-gate and post-gate as separate sites and
don't attribute *through* the gate linearly.

### 6. Steering/ablation via causal subspaces; and a warning that probe hits != causal sites

**Sources:** "Interpreting and Steering State-Space Models via
Activation Subspace Bottlenecks" — arXiv:2602.22719 (Feb 2026):
identified causal subspaces in Mamba mixers via progressive ablation;
merely **multiplying bottleneck-subspace activations by a scalar**
(2–5x) improved performance ~8% across 7 SSMs — steering need not be
additive vectors. "Detection vs. Execution: Single-Bucket Probes Miss
Half the Mamba-2 State Sink" — arXiv:2606.00930 (May 2026): Mamba has an
attention-sink analogue (delta-gate concentration on boundary/BOS
tokens); crucially, probes found many components with the sink's
*representational* signature, but only ~5% were *causally* load-bearing.
Related: PerfMamba — arXiv:2511.22849 (Nov 2025) on pruning/ablation
sensitivity of selective SSMs.

**For the lab:** (1) Add **scalar gain steering** (multiply a
probe-identified subspace, not just add a vector) — one-line change,
published wins. (2) Extend ground-truth validation to *causal*
validation: every probe direction gets an ablation/patching test before
being believed, since representational similarity provably doesn't imply
function in Mamba-2. Also check whether the model develops a "sink" on
episode-start frames (inspect delta on frame 0/respawn frames) — if so,
exclude those frames from probe training sets.

## Direct answers to the practical questions

**(a) Trunk output or internal state h_t?** Both, in this order.
Trunk/block output first: strongest published evidence, identical to the
GRU recipe, and the gate has already merged conv/SSM info there. Then
h_t: published precedent exists (Paulo et al. state steering; WriteSAE
cache steering), and it buys persistence-with-decay semantics
unavailable at the trunk. Difference-of-means on recurrent states
specifically: **yes, known to work** (Paulo et al. 2024, on Mamba 2.8b),
with the caveats that effects were behavior-dependent and non-additive
with activation steering.

**(b) Does d_state=16 change what's possible?** It changes *scope*, not
feasibility. Memory horizon is bounded (~10^2-frame recall ceiling by
Stuffed-Mamba scaling; forgetting learned only if trained on >~80-frame
sequences); features needing long persistence must live in slow
(near-unit A-bar) channels. Practical upside: 512x16 per layer is fully
enumerable — compute the per-channel effective decay spectrum
mean(exp(delta*A)) over real gameplay and get a *timescale map* of the
state (fast channels = reactions, slow channels = stock/percent-like
episode facts), infeasible at LM scale. Note the state (8192-d) is a
*bigger* probe target than the 256-d trunk, and it's a matrix — see #2.

**(c) Conv/gate gotchas?** (i) Off-by-one: conv shifts current-frame
info ~1 frame forward into the state (two independent confirmations) —
lag probe labels. (ii) The output gate y (x) SiLU(z) breaks linear
attribution and can make trunk-level probes conflate "SSM knew it" with
"gate let it through" — probe pre-gate SSM output, gate branch z, and
post-gate separately. (iii) Writes are delta-modulated: the same game
event writes strongly or weakly depending on selectivity — condition
analyses on delta, and inspect delta for boundary-frame sinks. (iv)
Probe hits need causal (patching) confirmation.

## Suggested experiment queue (cheap -> ambitious)

1. Instrument all block-internal sites per the MambaLens hook list;
   rerun existing linear probes at every site incl. flattened h_t; check
   the off-by-one.
2. Time-lagged probes -> per-feature memory-horizon curves; compare to
   decay-spectrum map from mean(exp(delta*A)).
3. Port diff-of-means steering to trunk output (expected to just work),
   then one-shot h_t injection with half-life measurement.
4. Implicit-attention maps for selected decisions; validate
   frame-attribution against known reaction windows.
5. Scalar-gain steering on probe subspaces; causal patching (swap h_t
   between matched game states) as the standard validation gate.
6. WriteSAE-style rank-1 write-atom SAE as SAE variant #6; use its atoms
   as a steering basis.
7. Full 2-layer circuit analysis of one behavior (e.g., tech-chase
   decision) via exhaustive activation patching — a publishable first
   for game-playing SSMs.

Sources: arXiv 2404.05971, 2605.12770, 2410.07145, 2506.11891,
2512.15653, 2407.14008, 2410.06672, 2404.03646, 2403.01590, 2406.07592,
2602.22719, 2606.00930, 2511.22849; SSRN 6465858; Phylliida/mamba_interp;
joelburget/mamba-sae; FarnoushRJ/MambaLRP.

---

# Report 3: Mamba for real-time game agents (2024 – Jul 2026)

Scope: policy backbones in RL/imitation, training gotchas, O(1)
inference and rollback compatibility, windowed vs stateful behavior,
small-model concerns. Ranked by importance to the ExPhil decision. The
case against is at the end and is substantive.

## Tier 1 — Directly decision-relevant

### 1. Windowed-trained Mamba does NOT safely transfer to stateful (carry-state) inference — the single biggest gotcha for the planned O(1) step path

**Finding:** Mamba trained on fixed windows learns a limited effective
receptive field and a limited *state distribution*. Run statefully past
the training length, hidden states drift into regions never seen in
training and quality degrades — the "unexplored states hypothesis."

- **DeciMamba / LongMamba** (ICLR 2025): global channels' hidden state
  contributions decay exponentially; models trained on length L fail to
  use context beyond ~L; usable context "is dictated by training
  sequence lengths." (arXiv 2406.14528, 2504.16053)
- **"Understanding and Improving Length Generalization in Recurrent
  Models"** (arXiv 2507.02782, Jul 2025): the fix is cheap — ~500
  post-training steps (~0.1% of budget) of either (a) Gaussian-noise
  state initialization or (b) cross-sequence state passing (initialize
  each training chunk with the final state of another sequence). Took
  models from 2k -> 128k usable length.

**Implication:** our window-60 checkpoint is calibrated to *exactly 60
frames of history, starting from zero state*. Building the stateful step
path and carrying state across a whole game (~30k frames) risks
behavioral drift. The fix is nearly free: short finetune with
state-passing/noise-init, or TBPTT-style chunked training where chunk k
inherits chunk k-1's final state (standard practice — Mamba-2 chunked
algorithm; state-spaces/mamba issue #536). Alternative cheap mitigation:
reset state every N frames or at natural boundaries (stock loss,
respawn). **Do this finetune before trusting any stateful deployment.**

### 2. Rollback netcode: fixed-size SSM state is genuinely well-suited — and the existing windowed path is *already* rollback-correct for free

**Finding:** No published game-agent work addresses rollback +
recurrent-state snapshotting directly (nothing in fighting games/esports
uses SSMs publicly — ExPhil is the visible frontier here). Mechanics
from rollback literature (SnapNet netcode series) + SSM properties:

- A 2-layer, d_model=256, expand=2, d_state=16 Mamba has per-layer
  state of (512x16 SSM + 512x4 conv) ~= 10K floats -> **~80 KB total in
  fp32**. A ring buffer of the last ~7-10 frames of state (Slippi's max
  rollback window) is under 1 MB and snapshot/restore is a memcpy.
  Categorically nicer than a transformer KV cache, which must be rewound
  by truncation and whose per-frame append/evict logic interacts badly
  with resimulation.
- Subtlety: on a rollback, the *game observations* fed for the
  mispredicted frames were wrong, so a stateful agent must restore the
  state at the rollback point and **re-run its recurrent update over the
  corrected frames** (up to ~7 steps). At this scale each step is
  sub-millisecond of FLOPs, but per-step dispatch overhead in Nx/XLA is
  the real cost — benchmark 7 sequential tiny steps, not one.
- **Key observation:** the current windowed-recompute path (8.93 ms) is
  *stateless* — a rollback just means "recompute the window with
  corrected frames," which is what it does every frame anyway.
  **Windowed inference gives rollback correctness with zero extra
  machinery.** The stateful path trades that simplicity for latency
  headroom and longer memory.
- Prior art: slippi-ai (vladfi1) sidesteps rollback entirely by playing
  at a fixed **18+ frame delay** with buffer donation ("feels like local
  at up to 300ms ping"). A proven alternative: train with delay, never
  resimulate.

**Implication:** three viable netplay designs, in increasing effort:
(a) keep windowed inference, rollback is free; (b) slippi-ai-style fixed
delay, no rollback handling at all; (c) stateful + state ring buffer +
resim. Don't build (c) until (a) is shown insufficient.

### 3. Evidence Mamba is a *good* policy backbone in imitation with limited data — but the advantage shows on long history and small data, not short windows

- **MaIL** (CoRL 2024, arXiv 2406.08234): Mamba beats Transformer
  policies by up to ~30% success on LIBERO **when data is limited** (20%
  of demos); advantage disappears as data scales. Critically: 5-step
  history often matched or underperformed 1-step history — history
  itself wasn't the win. Encoder-decoder variant needed learnable
  alignment tokens because Mamba requires matching input/output lengths.
- **MTIL** (May 2025, arXiv 2505.12410): full-history Mamba encoding
  solves temporally-dependent manipulation tasks where Markovian/
  short-window policies fail — relevant only if Melee decisions
  genuinely need >60-frame memory (opponent habit modeling does;
  reactions don't).
- **Decision Mamba family**: Decision Mamba (NeurIPS 2024, arXiv
  2406.05427) and successors established Mamba as a competitive
  offline-RL trajectory backbone; **DeMa** (NeurIPS 2024, arXiv
  2405.12094) beat Decision Transformer with 30% fewer params — **but
  found long sequences add compute without performance because Mamba's
  influence decays ~exponentially, and chose the Transformer-like
  (windowed) DeMa over the RNN-like (stateful) DeMa**. A direct
  published data point: at policy scale, windowed processing was the
  *better* operating mode.
- **DRAMA** (ICLR 2025, arXiv 2410.08893): Mamba-2 world model,
  DreamerV3-style, 105% human-normalized Atari100k at **7M params vs
  DreamerV3's 200M**; Mamba-2 beat Mamba-1 in 3/4 ablated games and
  trains 2-8x faster; imagination context of only 8 steps; no manual
  hidden-state resets at episode boundaries. Config strikingly close to
  ours: **2 Mamba layers, hidden 512, d_state 16**. Proof that tiny
  2-layer Mamba works as a sequence backbone in an RL loop.
- **PPO+Mamba:** MAMBA meta-RL (ICLR 2024) uses PPO over recurrent
  latents; no prominent published on-policy Mamba-policy PPO result yet
  — the PPO stage will be mildly novel territory.

## Tier 2 — Training gotchas (practical recipes)

### 4. Precision and stability

- Keep **A_log, dt (delta), and D in fp32** and run the recurrence
  accumulation in fp32 even under bf16 training — official
  state-spaces/mamba practice. Mamba is comparatively well-behaved:
  "Mamba SSMs Are Lyapunov-Stable Learners" (arXiv 2406.00209, 2024)
  shows low divergence under fp16/bf16 mixed precision and no
  post-finetune deviation spikes, unlike Pythia/OpenELM.
- **Delta/dt pitfalls:** dt goes through softplus with init biased so dt
  in [~0.001, 0.1]; discretization gives decay factor exp(A*dt).
  Too-large dt -> state forgets everything; too-small -> no selectivity.
  At 60 fps, 1/dt is roughly "frames of memory" — check the learned dt
  distribution post-training; piled at clamp boundaries = memory
  horizons being fought over. Large-scale hybrid reports (Jamba,
  Nemotron 2026) add internal RMSNorm inside/after Mamba mixers to kill
  activation-spike instabilities — cheap insurance for spikes.
- **Chunked vs full-sequence training:** chunked with state passing
  (TBPTT) is standard and inoculates against the Tier-1 #1 mismatch.
  LongSSM (arXiv 2406.02080) analyzes why length-extension tricks work.
  A sequence-length curriculum (60 -> 300 -> 900 with state passing) is
  a reasonable path to match-scale memory.
- **Dropout:** stock Mamba blocks contain **no internal dropout**.
  Published practice: residual/output dropout after the block or
  drop-path between blocks, never inside the selective scan. Community
  evidence that small-data Mamba overfits without it
  (state-spaces/mamba issue #454).

### 5. Inference latency and quantization

- Real-time numbers at robot scale: Mamba backbone 13.14 ms/step vs
  Transformer 23.07 ms with 10-frame history under a 20 ms budget
  (FlowRAM-adjacent robotics, 2025); Mamba-OTR (2025) does online
  streaming detection trained on fixed chunks, deployed recurrently.
- **Quantization of SSM states:** active subfield (Quamba ICLR 2025;
  Quamba-SE Jan 2026; Ternary Mamba Jun 2026) — all fighting activation
  outliers in SSM states. **Verdict for us: irrelevant.** These exist
  for 2.8B models on edge devices; our entire state is ~80 KB and naive
  int8 states would only add risk.
- Our 8.93 ms windowed already fits 16.6 ms. The O(1) step will be
  FLOP-trivial but **dispatch-bound in Nx/XLA at batch 1** — the win may
  be smaller than the 60x FLOP reduction suggests. Benchmark before
  building.

## Tier 3 — The case AGAINST Mamba at this scale

1. **Selectivity is probably not the bottleneck at 2 layers / 256 hidden
   / 60-frame windows.** "Were RNNs All We Needed?" (arXiv 2410.01201,
   Oct 2024) shows trivially-simplified minGRU/minLSTM match Decision
   Mamba on RL benchmarks; DeMa found influence decays exponentially
   anyway; MaIL found H5 ~= H1. At this scale the differentiator is
   data, features, and the DAgger loop — not the mixer. A cuDNN-class
   LSTM/GRU gives the same O(1) stateful step, *simpler*
   snapshot/restore (h,c vectors), decades of stability folklore, and no
   custom-kernel maintenance in Nx. slippi-ai reached
   top-human-competitive Melee play on exactly that recipe.
2. **We own the implementation.** The precision traps (fp32 recurrence
   params, dt clamps), the windowed->stateful mismatch, and the absence
   of fused kernels in Elixir/Nx are all costs a reference-implementation
   user doesn't pay.
3. **Reaction-relevant horizons are 5-20 frames; window-60 is already
   1 s.** Nothing reviewed suggests selective SSMs beat simple
   recurrence *below* ~1k-step horizons. Published Mamba wins are
   long-context. The honest pro-Mamba argument for Melee is *future*
   opponent-adaptation memory across a whole match — which can't be
   cashed in without the length-generalization finetune of #1.
4. **Windowed mode negates Mamba's headline advantage.** Staying
   windowed (rollback-free, fits budget) means paying Mamba's complexity
   for a mode where a small transformer or TCN over 60 frames is equally
   fast, trains more stably, and DeMa suggests performs as well.
5. Counterpoint keeping it alive: DRAMA's 7M-param, 2-layer Mamba-2
   result and MaIL's small-data advantage are real, at exactly this
   scale, and the fixed-size state is architecturally the cleanest
   rollback story of any sequence model. If Mamba stays the main line,
   prefer **Mamba-2 over Mamba-1** (DRAMA ablation: better and 2-8x
   faster to train; Mamba-3, Mar 2026, further improves state tracking
   and decode efficiency but no small-scale RL results yet).

**Bottom-line recommendation:** keep the windowed Mamba line running (it
works, fits latency, and is rollback-correct for free), but (a) run an
LSTM/GRU baseline at matched params before declaring Mamba the main line
— the literature predicts a tie at this scale; (b) if/when building the
stateful path, budget the state-passing/noise-init finetune (~500 steps)
and a rollback ring buffer of per-frame states (~1 MB); (c) skip
SSM-state quantization entirely; (d) consider slippi-ai's fixed-delay
design as the lowest-effort netplay path.

Sources: arXiv 2406.14528, 2504.16053, 2507.02782, 2406.08234,
2505.12410, 2406.05427, 2405.12094, 2410.08893, 2403.09859, 2406.00209,
2406.02080, 2410.13229, 2606.18114, 2410.01201; Mamba-2 chunked
algorithm (tridao.me); state-spaces/mamba issues #454/#536;
vladfi1/slippi-ai; SnapNet rollback series; Mamba-3 blog (together.ai);
Nemotron hybrid (arXiv 2606.15007); Mamba-OTR (arXiv 2507.16342);
FlowRAM (arXiv 2506.16201).
