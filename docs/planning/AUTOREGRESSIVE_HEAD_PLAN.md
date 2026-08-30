# AUTOREGRESSIVE HEAD — plan (fox_gen_v1.1 recipe change)

**Written 2026-08-30 01:45.** Status: PLANNED, not started. Owner: next
training session. Pre-registered decision rule at the bottom — read it
before looking at any result.

## 1. The finding that motivates this

`fox_gen_v1`'s six controller heads (buttons, main_x, main_y, c_x, c_y,
shoulder) are computed **in parallel** from the trunk. Training is six
parallel cross-entropies; inference is one fused sampler drawing each head
independently. The joint per-frame action is therefore modelled as

    p(a | s) = p(buttons|s) · p(main_x|s) · p(main_y|s) · p(c_x|s) · p(c_y|s) · p(shoulder|s)

The name "autoregressive" in the codebase (`Policy.build_autoregressive`,
`build_autoregressive_loss_and_grad_fn`) means "six heads"; the one path that
actually conditions heads on earlier components (`Heads.build_autoregressive`,
with `prev_*` inputs) is **dead code** — no builder, loss, agent, or export
calls it (verified 08-30).

What that costs, measured in expert play (`scripts/joint_head_audit.exs`,
`eval_runs/0830_joint_head_audit/RESULTS.md`): see §7 for the numbers. The
qualitative consequence was already visible in A2/C4: an up-B is "B AND
stick-up on the same frame"; sampled independently, the two coincide at the
product of their marginals, so the bot drifts to routes that need only one
head (side-B: B + whatever the stick is doing; airdodge: L/R + whatever the
stick is doing) and dies on them at 50–85% (expert 15%).

This is a **training** change (Bradley's rule, 08-29: no decode fixes), and a
prerequisite for anything downstream — a critic or PPO on top of an
independent-heads policy inherits the incoherence.

## 2. Theory, briefly

- **Chain rule.** Any joint over the six components factors exactly as
  p(a1|s)·p(a2|s,a1)·…·p(a6|s,a1..a5). The independent model drops the
  conditioning; the error is the total correlation
  TC(a|s) = Σ H(ai|s) − H(a|s) ≥ 0, in bits per frame, and it is exactly the
  extra cross-entropy the independent model pays at the optimum.
- **Why it matters for sparse joint events.** If P(B|s)=0.3 and P(up|s)=0.3
  but the expert only ever presses B *with* up in state s, the true
  P(B∧up|s)=0.3 while the independent model gives 0.09 — and 0.21 of
  "B with the stick elsewhere" that never happens in the data.
- **Academic options** for a multi-component discrete action:
  1. *Autoregressive discretization* (Metz et al. 2017 "Discrete sequential
     prediction of continuous actions"; slippi-ai's `AutoRegressive` head;
     the standard for controller-like action spaces). Exact factorization,
     one small step per component at inference, sampling stays cheap and
     temperature-controllable per head. **This is the plan.**
  2. *Latent-variable* (CVAE — our `:act` policy type; VQ-VAE action tokens).
     Captures joint structure through a latent; sampling needs a latent draw;
     temperature semantics are murkier.
  3. *Diffusion / flow matching* over the continuous controller (Chi et al.
     2023 Diffusion Policy; our `:diffusion` and `:flow_matching` types).
     Strong for multi-modal continuous actions; costs multiple network
     evaluations per frame and gives up the discrete per-head decode we
     have instrumented (L9, B2).
  4. *Joint softmax* over the product space (7 buttons × 17 × 17 × 17 × 17 × 5
     ≈ 5·10⁷ classes) — not viable.
  Option 1 is the right size for a 60 Hz controller and keeps every
  instrument we built this week applicable.

## 3. Design

**Order** (slippi-ai's order, coarse-to-fine, the control-flow-bearing
component first): `buttons → main_x → main_y → c_x → c_y → shoulder`.

**Head.** A residual stream `r0 = W·trunk` (128-d). For component k:
`logits_k = MLP_k(r_{k-1})`; after sampling (or teacher-forcing) `a_k`,
`r_k = r_{k-1} + E_k(a_k)` where `E_k` embeds the discrete component
(buttons: 7-d multi-hot → linear; sticks: bucket one-hot → linear). Each
`MLP_k` is 1 hidden layer (64) — the same size as today's heads.
This is slippi-ai's `AutoRegressiveComponent` (residual_size 128,
component_depth 0), which we can reuse almost line for line.

**Training: teacher forcing.** The target components of the same frame are
fed as the `a_k` inputs (all six logits computed in one forward from
ground truth — no sequential loop in training). Loss stays six
cross-entropies with the existing label smoothing / focal / pos-weight
options; `Loss.build_autoregressive_loss_and_grad_fn` keeps its shape, the
predict_fn just takes `(state, prev_components)`.

**Inference: sequential sampling.** Six small kernels instead of one fused:
sample buttons → embed → main_x logits → sample → … Per-head temperature,
`deterministic_buttons`, hysteresis, and `mode_of_n` all still apply
(mode-of-N remains disqualified for play; the option stays as an
instrument). Budget: the heads are tiny; expect <1 ms added at 60 Hz.
The stateful-step path (`heads_predict_fn` on the trunk state) is the
natural home: the trunk step is unchanged, only the head changes.

**Export / interp.** ONNX export becomes six sub-graphs or one graph with
the loop unrolled; the interp tools that read `action.logits.*` keep
working (logits are now conditional on the sampled prefix — document
it; B3 entropies become conditional entropies, which is the right thing).

**Config.** `--head autoregressive | independent` (default stays
`independent` until the decision rule passes), saved into the checkpoint
config; `Activations.load_heads_only` and the Agent read it. Guard #6 /
GOTCHA #105 apply: the flag must reach the checkpoint config, the Agent,
the interp loaders, and the canary.

## 4. Work items

| # | item | where | size |
|---|---|---|---|
| 1 | Run and record the joint-head audit (this doc §7) | `scripts/joint_head_audit.exs` | done 08-30 |
| 2 | `Heads.build_autoregressive_head(trunk, opts)` — residual stream + 6 conditional heads; teacher-forced inputs `tf_*` | `lib/exphil/networks/policy/heads.ex` (replaced the dead `build_autoregressive`) | **done 08-30** |
| 3 | Predict fn plumbing: training forward takes targets as prefix; `Policy.build_temporal(head: :autoregressive)` | `policy.ex`, `imitation/loss.ex` (+ `imitation.ex` init template/warmup) | **done 08-30** |
| 4 | `Sampling.sample_autoregressive/4` — sequential draw with per-head T; keep `sample/4` for `:independent` | `policy/sampling.ex` (stage1/stage2 fused kernels; hysteresis applied BEFORE conditioning) | **done 08-30** |
| 5 | Agent: dispatch on checkpoint `head`; stateful-step path (AR = trunk-only model, `heads_predict_fn` nil) | `agents/agent.ex` | **done 08-30** |
| 6 | Checkpoint config + `--head` flag + `Activations` plumbing (canary untouched — head doesn't change the embedding) | `training/config.ex` + `config/parser.ex`, `imitation/checkpoint.ex`, `interp/activations.ex` (`load_heads`/`load_heads_only` branch on head) | **done 08-30** |
| 7 | Tests: head builds; zero-init = independent; teacher-forced logits equal sequential logits given the same prefix; synthetic P(up\|B)=0.997 vs P(up\|!B)=0.003 recovered (head-only fit, frozen constant features — 8a's mechanism verified in miniature) | `test/exphil/networks/policy/autoregressive_head_test.exs` | **done 08-30** |
| 8a | **Head-only fit (cheap first — Bradley, 08-30)**: **v1.1-ARhead** = FROZEN ep10 trunk + AR head trained on cached trunk activations. Control **v1.1-INDhead** = frozen trunk + re-initialised independent head, same data. If ARhead already moves A2 recovery routes per §6 → done cheap, skip 8/9. If PARTIAL (coincidence moves, routes don't) → unfreeze and continue per 8/9. **RAN 08-30 (run 2; run 1 died at save — 2 GB term_to_binary limit, capture is streamed now):** 1,000 files → 12.14 M rows (1,144 port-streams; dittos both ports, per-file fox detection — NOT v1's blind port-1, see `eval_runs/0830_corpus_mix`), 4 epochs each, split by game. **ARhead val 2.231 vs INDhead val 2.768 — gap 0.537 nats ≈ 0.77 bits/frame, matching §7's 0.86 bits/frame TC almost exactly.** Checkpoints `fox_gen_v1.1_{ARhead,INDhead}_policy.bin`. §6 scoring: `scripts/arhead_score.sh` → `eval_runs/0830_arhead_score/`. | `scripts/train_ar_head.exs` | **fit done 08-30** |
| 8 | Train **v1.1-AR**: B1's recipe (3 epochs from ep10's trunk, seed 828) with the new head; the trunk initialises from ep10, the head from scratch. Only if 8a is PARTIAL. | `scripts/train.exs --resume … --head autoregressive` | 3 h GPU (B1 took 3h20 incl. cache) |
| 9 | Control **v1.1-IND**: identical run with the independent head re-initialised from scratch (so "new head params" is controlled) | same | 3 h GPU |
| 10 | Score per §6; write RESULTS; human look only if the rule passes | `scripts/awbc_score.sh` pattern | 1.5 h |

Why some training is unavoidable (recorded for the "just do AR inference"
question): the shipped checkpoint's heads have no input wires for each
other's samples — no logit depends on another head's draw, so sequential
sampling from the existing params changes nothing. The dependency
P(up|B, offstage) lives in weights that must be trained. 8a is the
minimal training that creates those weights.

No lib edits while any training unit is active (the loop law).

## 5. Risks

- **Within-frame exposure bias**: training feeds true prefixes, inference
  feeds sampled ones. Mild at 6 steps; slippi-ai lives with it. If it
  shows, scheduled sampling on the prefix is the standard fix.
- **Order effects**: buttons-first means the stick is conditioned on the
  press; the reverse order is defensible. Pre-register buttons-first (it
  is the control-flow component) and do not re-order on a null.
- **Speed**: six sequential kernels; measure inferences/frame and
  staleness in the bracket (the mode-of-N bracket's readout).
- **The trunk was trained for independent heads**; three epochs may not
  re-shape it. If v1.1-AR beats v1.1-IND on the joint metrics but not on
  play, a from-scratch AR run is the follow-up, not a decode tweak.

## 6. Pre-registered evaluation and decision rule

All on the instruments from EVAL_DIRECTIONS, same day, same protocol
(8 × 120 s CPU, T=0.5/buttons 0.5, delay 0), plus the human look.

Primary (the thing the head is supposed to fix):
- **A2 recovery**: first-route share for up-B and double-jump UP, airdodge
  DOWN; died-given-route for side-B/airdodge DOWN. Expert reference:
  up-B 16%, double-jump 24%, airdodge 5%; died|side-B 15%, died|airdodge 14%.
- **Joint coincidence live**: from the AR run's replays, P(stick up | B)
  and P(stick side | L/R) vs the expert's (the audit's pair table).

Secondary: A1/B2 TV distance (floor 0.05 mean / 0.1 per situation);
C4 % at death (expert 109); C1 exchange win rate (floor pending);
d_up/min and held-action (floors 1.1× / 1.2×); durations to cap and
frozen-input (no collapse).

Decision:
- **SIGNAL** — v1.1-AR beats v1.1-IND on A2 first-route (up-B + double-jump
  share ≥ 2× IND's, airdodge ≤ ½) with died-given-route lower, no collapse
  signature, and TV not worse by more than the floor → AR becomes the
  default head; go to Bradley's live look (g6 rule: the look gates the
  recipe, not the metric).
- **PARTIAL** — joint coincidence moves toward the expert but A2 routes do
  not → the head works, the trunk needs the from-scratch AR run before
  judging.
- **NULL** — no A2 movement and coincidence unchanged → the independence
  is not what loses the up-B; record and stop. Do not tune the decode.

## 7. Audit numbers

Ran 2026-08-30 13:04 on 1,500 expert games (16.2 M frames):
`eval_runs/0830_joint_head_audit/RESULTS.md`.

- **Total correlation of (buttons, main_x, main_y): 0.856 bits/frame**
  unconditional, **0.841** given the action-state id, **0.858** given
  action-state + situation labels. Conditioning on coarse state removes
  none of it — the dependency is between the same frame's inputs, not
  something a better trunk reads off the state. Joint entropy is 5.65
  bits; the independent factorization mis-spends ~15% of it.
- **Up-B pair:** P(stick up | B) = 23.7% vs P(stick up) = 7.4% (**3.2×**);
  offstage P(up | B) = 41.7% vs 14.6% (**2.9×**). An independent head that
  has learned both marginals produces "B with stick up" on 15% of its
  offstage B presses; the expert does it on 42%. TV between
  P(stick_y | B, offstage) and P(stick_y | offstage) = 0.36 — the B-press
  frames put 42% of stick-y mass in the top four buckets vs 15% otherwise.
- L/R ↔ stick-side (airdodge / wavedash direction): 1.18× — weak.
  A ↔ stick-down: 0.78× (anti-correlated).

The up-B is the case the independent head cannot represent, and it is the
one A2 found missing. This is the audit's green light for §4.
