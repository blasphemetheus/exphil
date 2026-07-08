# slippi-ai (Phillip II): Reference-Project Deep Dive

*2026-07-07 — mined from a local clone at `~/git/slippi-ai` (head b066a66,
2026-05-22). All file references are into that clone. Context: their
RL-refined agent beat Moky 10–3 in a Fox-ditto FT10 and took Zain to 5–3,
playing at a deliberate 18–21 frames of delay.*

## Skeleton

Two-stage: behavior cloning on ~100k+ replays (per-player counts in
`nametags.py`: Hax 85k, iBDW 52k, aMSa 26k, …) → self-play PPO anchored to
the frozen BC policy. Active code is `slippi_ai/jax/` (Flax nnx); the
TensorFlow tree is legacy. Flagship config (`jax/launch_scripts/imitation/
tx_like.py`): 3×512 `tx_like` policy + separate 1×512 value net, delay 21,
batch 512, unroll 80, AdamW 1e-4.

## Architecture notes

- **Backbone `tx_like`** (`jax/networks.py:520`): alternating
  residual-wrapped recurrent layer (LSTM/GRU) + FFW ResBlock (LayerNorm →
  Linear → GELU → Linear, zero-init output). Not a transformer, not Mamba.
- **Controller heads** (`jax/controller_heads.py`): default is
  `Independent`; the sophisticated option is `AutoRegressive` (:185) — a
  128-d residual stream visits components in fixed order, each samples then
  adds a zero-init decode of its sample back into the stream; teacher-forced
  in `.distance()`. ExPhil's autoregressive 6-head design matches this
  pattern.
- **Stock action space** (`jax/embed.py:525`): 8 independent button bools +
  4 stick axes × 17 buckets + shoulder × 5 — *identical shape to ExPhil's*.
- **`custom_v1` action space** (`action_space/custom_v1.py`) — their
  redesign and the most stealable idea: main stick bucketed in POLAR
  coordinates with log-spaced radii and radius-dependent ("ragged") angle
  counts (1+4+16+64 = 85 buckets); c-stick 13; correlated buttons merged
  (L/R, X/Y, Z+A+shoulder→7-way) and jointly classified with the c-stick as
  one 416-way categorical. Two categoricals replace 12 heads; encodes real
  tech constraints (comments cite frame-perfect shield drop).

## Embedding notes

- Handcrafted base features closely mirror ExPhil's (percent/facing/xy/
  action/character/jumps/shield/on_ground/speeds; Nana recursive w/ exists
  bit; stage one-hot; optional Randall + FoD platform heights + items).
- **Enhanced path** (`jax/networks.py:986`): learned char + action-state
  embed tables PLUS a zero-init **factored character×action joint table**
  (`MultiEmbed`) — a learned residual on the marginals. Items are embedded
  per-slot through a shared MLP then **sum-pooled** (permutation
  invariant).
- **Name conditioning**: player identity one-hot (top-16 players by count,
  rest → unknown) in the state-action tuple; some players banned on request.
- **Delay is a data-shift property** (`jax/policies.py:114-153`): the net
  sees states [0,U-1], predicts actions [D+1,U+D]. Architecture-independent,
  changeable per training run.
- **Observation filters** (`observations.py`): `AnimationFilter` masks the
  first ~7 frames of tech animations (humanly indistinguishable — prevents
  superhuman info leakage; per-character windows sourced from fightcore).
  `FrameSkipFilter` freezes perceived state every N frames while KEEPING
  the controller channel live (simulated reaction delay). Recent commits
  randomize the frame-skip phase per trajectory.

## Training notes

- **Loss: plain NLL.** Softmax-CE on discretized axes, sigmoid-BCE on
  buttons, summed. **No focal loss. No pos_weight. No label smoothing.**
  ExPhil's focal+pos_weight stack has produced two bugs (GOTCHAS #54, the
  frame-weighted-path focal drop) — A/B against plain NLL is warranted.
- Teacher forcing at both levels: across controller components within a
  frame, and ground-truth previous actions along the unroll. `burnin=5`
  frames shed stale hidden state; hidden state carries across chunks.
- **RL stage** (`jax/rl/learner.py`): PPO clipped objective + forward
  KL-to-frozen-teacher (weight 0.1) + optional reverse KL/entropy; hard
  trust region (revert step if mean actor-KL > 1e-4); value-function burn-in
  before policy updates. **Zero-sum reward** (stocks + 0.01×damage, ledge-
  stall/approach shaping) explicitly to prevent self-play collusion.

## 2025–2026 directions

1. **TF→JAX migration** (dominant 2026 effort; done for policy/value).
2. **`sim_env/`** (May 2026): native fast batched Melee physics sim
   (melee-sim-light) replacing Dolphin for RL rollouts. Fox/Falco, 6
   stages. The RL-throughput bet.
3. **`jax/nash/`** (May 2026): zero-sum Nash-equilibrium policy learning
   (interior-point/qpax/LP solvers; nash_weight mixed with imitation
   anchor) — toward exploitability-minimizing play beyond naive self-play.

## Ranked takeaways for ExPhil

1. **When adding RL: copy the KL-anchored PPO recipe + zero-sum reward.**
2. **Polar/joint `custom_v1` action space** — strictly better stick
   discretization than the uniform Cartesian grid.
3. **Observation filters** (tech masking, frame-skip-with-live-controller,
   delay-as-data-shift) — pure data transforms, no model changes.
4. **Factored char×action learned embedding + item sum-pooling.**
5. **Player-identity conditioning** once data scales past a handful of
   players.
6. **A/B plain NLL vs the focal/pos_weight stack** (they never needed it at
   scale; ours may be compensating for 200-replay smallness — or hurting).

## Scale calibration

Them: ~100k replays, ~3×512 recurrent+FFW, batch 512 × unroll 80, cloud
training (skypilot). Us (2026-07): 200 replays, 3.7M params, single 5090.
Weak main-stick accuracy (~35%) at our scale is expected; scale data before
blaming architecture.
