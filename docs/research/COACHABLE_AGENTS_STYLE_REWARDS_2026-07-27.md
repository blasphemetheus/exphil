# Coachable Agents / Style-Conditioned Rewards — relevance to the multishine bot

Date: 2026-07-27. Prompted by a Discord suggestion (jab-san): "I wonder if you
could coach the agent into multishining better by endowing the reward function
with a multishine behavior vector interactively instead of coding it and
retraining it."

Paper: **"Coachable agents for interactive gameplay"** — Capobianco, van
Seijen, Bard, Burch, et al. (Sony AI + Guerrilla Games / Polyphony Digital
collaborators), arXiv:2607.00642, July 2026.
https://arxiv.org/abs/2607.00642
Domains: Gran Turismo 7 (GT Sophy), Horizon Forbidden West, DMC humanoid.

## What the paper actually does

- **Style-conditioned UVFA.** Reward is split `r(s,a;θ) = r_task(s,a) +
  r_style(s,a,θ)` where θ is a multi-dimensional style vector: linear mixing
  weights, thresholds (gating/ramping), and setpoints penalizing deviation
  from target ranges. The policy takes θ as an *input* and training samples
  across θ values and curated scenarios.
- **The payoff is a runtime dial.** After training, the user picks θ at
  execution time — no retraining — and moves along a pre-learned Pareto
  surface (tire conservation costs 2-3 s/lap in GT7; banning the dominant
  HFW weapon costs win rate; Fig. 4b shows explicit style-weight vs win-rate
  Pareto curves).
- **Machinery needed to make it learn:** scenario training, data
  augmentation, and *style-specific replay buffers*. For HFW they built a new
  SAC-family algorithm, **Cat-RAC** (Categorical Regularized Actor-Critic),
  because the hybrid controller action space broke standard ones. Scale:
  4.5M gradient steps × 5 seeds, AAA infrastructure; open-source code
  released with the DMC humanoid experiments.
- Styles compose (weapon × elemental styles executed correctly even when
  seen together <1/1M training scenarios; humanoid arm poses × continuous
  gait length).

## Correction to the Discord framing

This is **not interactive reward specification**. The style reward terms are
still hand-coded by engineers; what's interactive is choosing the mixing
vector θ live. "Reward endowed interactively by a human" is a different
literature — preference-based RL (Christiano et al. RLHF) or TAMER, where
live human feedback *becomes* the reward. The Sony answer to "no retraining"
is: pre-train a *family* of behaviors indexed by θ, then steer within it.
That recipe is far more tractable and is the one worth stealing.

## What transfers to ExPhil

1. **Runtime dial > data-mix commitment — we already measured why.** The
   ledge experiment (`ms_ledge_a`, HANDOFF_2026-07-27c) showed coverage
   mixing is a training-time commitment with a measurable dilution cost
   (~9% ledge data → 42-56/min and 38% crouch occupancy). The paper resolves
   the same tradeoff by baking variation into a conditioning input and
   choosing the operating point *after* training. Multishine translation:
   condition on a multishine-intensity scalar, sweep the dial at eval,
   find where escape/sustain peaks per run instead of per retrain.

2. **BC-compatible version, doable without GPU (near-term candidate).**
   The honest r_style needs the PPO loop (queue #7, GPU-gated). But the
   conditioning trick has an imitation cousin: label each training frame
   with a behavior descriptor computed from its own trajectory (shines/min
   in surrounding window, in-chain vs out-of-chain), feed it as extra
   conditioning input during BC, clamp to "high multishine" at inference.
   This is return-conditioned BC / the Decision Transformer idea (DT already
   exists in Edifice). Caveat: hindsight conditioning can teach the policy
   to *expect* outcomes rather than *cause* them — judge by the standard
   protocol (≥3 seeds × ≥3 runs, EXPOSURE_BIAS 0a/0b).

3. **Their coverage machinery validates ours.** Style-specific replay
   buffers + scenario training are structurally what
   `RecoverySynth.build_crouch` does: manufacture the states the target
   behavior needs covered. The absorbing-state theory predicts why they
   need this: off-support, a conditioned policy falls back to its marginal
   mode regardless of what θ requests.

4. **Sleeper artifact: Cat-RAC.** HFW's action space is 14 boolean buttons +
   two 2D sticks + two continuous triggers — nearly a GameCube controller.
   Cat-RAC is a documented answer to "which actor-critic variant survives a
   gamepad-shaped hybrid action space at 60 Hz." When PPO work resumes,
   read their appendix before committing to an algorithm variant.

5. **Scale realism.** Even at 4.5M steps their five seeds diverged on *how*
   styles get satisfied (five seeds, five different elemental-weapon
   preferences) — the same seed-variance phenomenon as our 2-vs-22 chain
   sustain, and they too resolve it by acknowledgment (evaluate across
   seeds), not elimination. Encouraging: our domain (one character, one
   technique) plausibly supports a 1-2 dim θ at our scale.

## Concrete next steps if pursued (in leverage order)

1. Descriptor-conditioned BC experiment on the multishine recipe: add a
   per-frame shines/min (or chain-membership) conditioning channel to
   train_multishine_policy, clamp high at eval. Cheap; laptop-feasible;
   fits the existing seed-farm protocol.
2. When PPO resumes on GPU: implement `r_style` as shine-event/chain-
   continuation terms with a runtime weight θ appended to the observation
   embedding; expose θ as a CLI/runtime flag rather than a baked constant.
3. Read the Cat-RAC appendix + released code before choosing the RL
   algorithm variant for the controller action space.
