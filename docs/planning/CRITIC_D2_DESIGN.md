# D2 critic + Best-of-N — design and pre-registration

**Status:** scripts written 2026-08-29 (overnight, while the AWBC arm chain
held the GPU), parse-checked, NOT yet run. First run = its own shakedown.
**Motivation:** Leg S (`eval_runs/0828_legS/RESULTS.md`): on decision
frames fox_gen_v1's one-sample match rate is 14.5% and its 16-sample
containment rate is 43.7%. A perfect chooser would triple pass@1 without
touching a weight. This is the chooser.

## What is being built (three scripts, one shared module)

| file | does |
|---|---|
| `scripts/lib/critic_features.exs` | `CriticFeatures`: per replay, trunk activations ⊕ raw scalars, master action, K policy samples + match flags, return-to-go, decision flag |
| `scripts/critic_extract.exs` | replays → one Nx-serialized dataset |
| `scripts/critic_train.exs` | fits V(s) (ridge) and the selector S(s,a) (bilinear softmax), reports by RANKING, exports `checkpoints/critic_*.bin` |
| `scripts/interp_bestofn.exs` | Best-of-N on FRESH replays: sampling vs selector vs oracle vs a critic-free baseline |

Kept out of `lib/` deliberately: the arm chain recompiles `lib/` at every
arm start. Promote to `lib/exphil/critic/` after the first successful run.

## Features — the G4 correction, applied

`phi(s) = trunk(s) ⊕ raw(s)`.
- `trunk(s)`: the policy's own GRU output for the 60-frame window ending
  at s (256 dims). Fertile for hitstun/offstage (G4 R² 0.83–0.93).
- `raw(s)` (62 dims): percents, stocks, positions, relative position,
  grounded, hitstun, shield, jumps, invulnerable, facing (both players);
  stage compact one-hot (7); **opponent character one-hot (33)**. These are
  the variables G4 found the trunk discards (percent R² ~0.5 vs ~0.85 from
  input; character "dead", 0.33). Percent is the most important variable
  in a Melee value function; it must not be routed through the trunk.

Actions `a` use the 13-dim continuous controller encoding
(`Embeddings.Controller.embed_continuous/1`): 8 buttons + main xy + c xy
+ shoulder — identical to the previous-action channel the policy already
consumes, so nothing new has to be learned about the action space.

## Two heads, both LINEAR first (G4: a linear probe reads this trunk)

**V(s) = wᵀ φ_std(s) + b.** Target: discounted standard-reward
return-to-go (stock ±1, damage 0.01, win 5; γ=0.99, horizon 600 frames),
computed within-replay only. Closed-form ridge. This is the
`Networks.Value`-shaped object online RL will want later.

**S(s, a) = aᵀ W φ_std(s) + vᵀ a.** Bilinear (W is 13×318 — 4k params).
Trained as a softmax over {master's action} ∪ {K=8 policy samples at the
deploy decode, T=0.5}, target = the master. At decode time, N samples are
drawn and the argmax-scored one is played. **This is the head that cashes
in Leg S**; V is recorded alongside for the critic's future.

Why a selector rather than V alone: a state-value cannot rank actions
without a simulator to roll each candidate forward; CycleSim exists but
its fidelity under arbitrary counterfactual inputs is unmeasured. The
selector needs no simulator and its validation metric IS the deployment
metric.

## Validation is by RANKING, never MSE alone

Held-out split is BY REPLAY. On held-out decision frames:

| metric | chance | reads |
|---|---|---|
| V pair-rank accuracy: P(V orders two states like their returns) | 0.50 | does V know good from bad states |
| V shuffled-target control | 0.50 | must stay at chance |
| S top-1 among K+1 (master included) | 1/(K+1) | can it find the master |
| **sampling pass@1** (random sample == master) | – | today's decode |
| **selector pass@1** (argmax-scored sample == master) | – | Best-of-N |
| oracle pass@K (any sample == master) | – | the Leg S ceiling |
| S shuffled-label control | ≈ sampling | must NOT beat sampling |
| mode-of-N baseline (bestofn only) | – | critic-free re-ranker S must beat |

`gap recovered = (selector − sampling) / (oracle − sampling)`.

## Decision rule (declared before any run)

- **STRONG** — gap recovered ≥ 50% on held-out replays AND on the fresh
  corpus (`fox_il_v1`, auto-port) AND selector beats mode-of-N by ≥ 5 pts
  → wire Best-of-N into the live decode (Agent: trunk tap + N-sample
  scoring; `lib/` edit) and go to Bradley's live look with loop_report +
  coach_report as the instruments.
- **PARTIAL** — 20–50% recovered, controls clean → replace the bilinear
  head with a 1-hidden-layer MLP (still on φ ⊕ a), re-run; do not wire
  live yet.
- **NULL** — < 20%, or the shuffled-label control also beats sampling, or
  mode-of-N ties the selector → the linear selector cannot read "which
  sample is the master's" from φ; Leg S headroom is real but not
  linearly harvestable. Next: check whether V alone ranks (if V works
  but S doesn't, the problem is the action pairing → try CycleSim
  rollouts scored by V).

## Known limits, declared

- Trained on the master's *exact* action as the only positive. Other
  samples can be equally good Melee; the metric under-counts them. The
  gap-recovered number is therefore conservative.
- Corpus = `erickfm_ranked/FOX/extracted` (v1's training set; val files
  unrecoverable). Held-out-by-replay limits leakage of frames, not of
  player style. The fresh-corpus run is the real test.
- Off-distribution states (the bot's own loops) are not in the training
  set; the selector may be worst exactly there. After a live wiring,
  score BOT replays with `interp_bestofn` as well.
- Best-of-N costs N head evaluations per frame; the heads are tiny, the
  trunk runs once. Live budget should be fine at N=8–16; measure.

## Runbook (when the GPU is free)

```bash
mix run scripts/critic_extract.exs --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --limit-files 40 \
  --out cache/critic/fox_gen_v1_ep10_erickfm40.nx
mix run scripts/critic_train.exs --data cache/critic/fox_gen_v1_ep10_erickfm40.nx \
  --out checkpoints/critic_fox_gen_v1_ep10.bin --report eval_runs/0829_critic/train.md
mix run scripts/interp_bestofn.exs --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
  --critic checkpoints/critic_fox_gen_v1_ep10.bin --replays 'replays/fox_il_v1/*.slp' \
  --n 16 --out eval_runs/0829_critic/bestofn_fox_il_v1.md
```
