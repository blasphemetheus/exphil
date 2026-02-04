# World Models, JEPA, and The Bitter Lesson: Research Foundations for ExPhil

> **Purpose:** This document synthesizes Yann LeCun's work on Joint Embedding Predictive Architectures (JEPA), world models, and energy-based learning with Richard Sutton's foundational contributions to reinforcement learning and the Bitter Lesson. We analyze how these research directions inform ExPhil's architecture and roadmap.

**Last Updated:** 2026-02-03

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Yann LeCun's Vision for Autonomous Intelligence](#yann-lecuns-vision-for-autonomous-intelligence)
   - [Energy-Based Models](#energy-based-models)
   - [Joint Embedding Predictive Architecture (JEPA)](#joint-embedding-predictive-architecture-jepa)
   - [V-JEPA and Video Understanding](#v-jepa-and-video-understanding)
   - [Hierarchical JEPA (H-JEPA)](#hierarchical-jepa-h-jepa)
   - [The World Model Framework](#the-world-model-framework)
3. [Richard Sutton's Contributions](#richard-suttons-contributions)
   - [Temporal Difference Learning](#temporal-difference-learning)
   - [The Reward Hypothesis](#the-reward-hypothesis)
   - [The Bitter Lesson](#the-bitter-lesson)
4. [Synthesis: Where LeCun and Sutton Converge and Diverge](#synthesis-where-lecun-and-sutton-converge-and-diverge)
5. [Applications to ExPhil](#applications-to-exphil)
   - [Current Architecture Analysis](#current-architecture-analysis)
   - [JEPA-Inspired Modifications](#jepa-inspired-modifications)
   - [Bitter Lesson Alignment](#bitter-lesson-alignment)
   - [Proposed Experiments](#proposed-experiments)
6. [Implementation Roadmap](#implementation-roadmap)
7. [References](#references)

---

## Executive Summary

This document explores two of the most influential research directions in modern AI and their implications for ExPhil:

**Yann LeCun's JEPA/World Model paradigm** proposes that intelligent systems should:
- Learn in abstract representation space, not pixel/token space
- Build predictive world models that understand physics and causality
- Use hierarchical planning at multiple time scales
- Avoid autoregressive generation in favor of joint embedding prediction

**Richard Sutton's RL fundamentals and Bitter Lesson** argue that:
- General methods that leverage computation outperform domain-specific engineering
- Reward maximization is sufficient for intelligent behavior (reward hypothesis)
- Temporal difference learning enables efficient credit assignment
- Scale (compute + data) beats clever engineering "by a large margin"

**Key insight for ExPhil:** These perspectives are complementary, not competing. LeCun's world models provide the *architecture* for understanding game dynamics, while Sutton's scaling principles guide *how* to train them. The synthesis suggests:

1. **Replace autoregressive BC** with JEPA-style representation learning
2. **Build a learned Melee world model** that predicts in representation space
3. **Use hierarchical planning** for multi-timescale decision making
4. **Scale aggressively** while minimizing hand-crafted features

---

## Yann LeCun's Vision for Autonomous Intelligence

### Energy-Based Models

LeCun's foundational insight is that intelligent systems can be understood through **energy functions**. An energy-based model (EBM) assigns a scalar "energy" E(W, X, Y) measuring compatibility between inputs X and outputs Y given parameters W.

```
E(W, X, Y) → ℝ   (lower energy = higher compatibility)
```

**Key properties:**
- No need for normalized probabilities (avoids intractable partition functions)
- Can model many-to-many relationships (multiple valid outputs for one input)
- Works in continuous, high-dimensional spaces where generative models struggle

**Training approaches:**

| Method | Description | Challenge |
|--------|-------------|-----------|
| **Contrastive** | Push down energy on data, push up everywhere else | Requires sampling negatives |
| **Regularized** | Limit the volume of low-energy regions | Architectural constraints |
| **Latent variable** | Introduce latent z that captures uncertainty | Training stability |

LeCun argues that contrastive methods (like SimCLR) hit walls in high dimensions because you need exponentially more negative samples. **Regularized methods** via architectural constraints are more promising.

**Application to Melee:** Rather than modeling P(action | state) directly, model an energy E(state, action) where low-energy state-action pairs are compatible. This naturally handles multi-modal action distributions (multiple "correct" actions in a given state).

### Joint Embedding Predictive Architecture (JEPA)

JEPA is LeCun's proposed architecture for learning world models without generative decoding. The key insight is **predicting in representation space**, not observation space.

```
┌─────────────┐     ┌─────────────┐
│   Input x   │     │   Target y  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
  ┌─────────┐         ┌─────────┐
  │Encoder_x│         │Encoder_y│
  └────┬────┘         └────┬────┘
       │                   │
       ▼                   ▼
    s_x (repr)          s_y (repr)
       │                   ▲
       │   ┌─────────┐     │
       └──▶│Predictor│─────┘
           └─────────┘
```

**Why predict representations instead of observations?**

1. **Abstraction:** Encoder removes irrelevant details (lighting, rendering artifacts)
2. **Efficiency:** No need to model every pixel/dimension
3. **Semantics:** Learned representations capture task-relevant information
4. **Uncertainty:** Predictor can represent uncertainty about future states

**JEPA vs Autoregressive Models:**

| Aspect | Autoregressive (GPT-style) | JEPA |
|--------|---------------------------|------|
| Prediction target | Next token/pixel | Representation of target |
| Handles uncertainty | Via sampling | Via representation space |
| Required detail | Must model everything | Can ignore irrelevant info |
| Training signal | Reconstruction | Representation matching |

### V-JEPA and Video Understanding

**V-JEPA** (Video JEPA) extends the architecture to temporal sequences, directly relevant to Melee's frame-by-frame nature.

**Architecture (V-JEPA 2):**
- **Encoder:** Vision Transformer (ViT-g, ~1B params) with 3D Rotary Position Embeddings
- **Input:** Video divided into "tubelets" (2 frames × 16×16 pixels)
- **Predictor:** Transformer that predicts masked tubelet representations
- **Training:** Self-supervised on 1M+ hours of internet video

**Key results:**
- 77.3% top-1 accuracy on Something-Something v2 (motion understanding)
- 39.7 recall@5 on Epic-Kitchens-100 (action anticipation)
- **Zero-shot robot control** via model-predictive control

**V-JEPA 2-AC (Action Conditioned):**
Extends V-JEPA for robotic planning:

```
State_t + Action_t → Predictor → State_{t+1} representation
```

The robot imagines consequences of actions in representation space, then executes the best action via MPC (Model Predictive Control). Achieved 65-80% success on manipulation tasks **zero-shot** in unseen labs.

**Application to Melee:**
V-JEPA's approach to video understanding maps directly to Melee:
- Melee states are "frames" that evolve according to game physics
- A V-JEPA-style encoder could learn representations of game states
- A predictor conditioned on actions could simulate game dynamics
- MPC-style planning could select optimal actions

### Hierarchical JEPA (H-JEPA)

LeCun proposes **H-JEPA** for multi-timescale planning:

```
         Long-term goals (high level, abstract)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Mid-level    Mid-level   Mid-level
    sub-goals    sub-goals   sub-goals
        │           │           │
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
    ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
   Low-level actions (frame-by-frame)
```

**Key insight:** Different levels of abstraction enable different prediction horizons:
- **Low-level:** Detailed predictions, short term (1-10 frames)
- **Mid-level:** Abstract predictions, medium term (10-60 frames)
- **High-level:** Goal-directed planning, long term (60+ frames)

**Melee application:**
| Level | Melee Context | Time Horizon | Example |
|-------|---------------|--------------|---------|
| High | Win the game | Minutes | "Edge guard opponent" |
| Mid | Win the exchange | Seconds | "Execute combo sequence" |
| Low | Execute technique | Frames | "L-cancel the aerial" |

This maps naturally to Melee's structure where micro-level execution (L-cancels, tech) serves macro-level strategy (neutral, advantage, disadvantage).

### The World Model Framework

LeCun's 2022 position paper "A Path Towards Autonomous Machine Intelligence" proposes a six-module architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Configurator                               │
│   (sets goals, modulates other modules based on context)         │
└───────────────────────────────┬──────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   Perception  │───▶│    World Model       │◀───│ Short-Term Mem  │
│    Module     │    │                      │    │                 │
│               │    │  Predicts next state │    │  Stores recent  │
│ (sensory →    │    │  given action        │    │  states/actions │
│  repr)        │    │                      │    │                 │
└───────────────┘    └──────────┬───────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │    Cost Module       │
                    │                      │
                    │  Intrinsic cost:     │
                    │  - Avoid bad states  │
                    │  Trainable critic:   │
                    │  - Predict future    │
                    │    cost              │
                    └──────────┬───────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │    Actor Module      │
                    │                      │
                    │  Proposes action     │
                    │  sequences that      │
                    │  minimize cost       │
                    └─────────────────────┘
```

**Critical insight:** The world model predicts in representation space, not observation space. This makes planning tractable in high-dimensional domains like video/games.

**Intrinsic motivation:** LeCun argues that beyond external rewards, agents need intrinsic drives:
- Curiosity (reduce uncertainty about the world)
- Competence (improve predictions/skills)
- Homeostasis (maintain internal state)

This aligns with Sutton's work on intrinsic motivation in RL.

---

## Richard Sutton's Contributions

### Temporal Difference Learning

Sutton's foundational 1988 paper introduced **TD learning**, which enables credit assignment across time without waiting for episode completion.

**Key insight:** Learn from the difference between successive predictions:

```
V(s_t) ← V(s_t) + α[r_{t+1} + γV(s_{t+1}) - V(s_t)]
                   └──────────── TD error ────────────┘
```

**TD(λ)** generalizes via eligibility traces:
- λ=0: One-step TD (bootstrap from next state)
- λ=1: Monte Carlo (wait for episode end)
- λ∈(0,1): Weighted combination of n-step returns

**Melee application:** TD learning is critical because:
- Episodes are long (minutes, thousands of frames)
- Rewards are sparse (stocks, game outcome)
- Credit assignment is hard (which frame caused the KO?)

TD(λ) with appropriate λ can propagate reward signal back through combo sequences and neutral exchanges.

### The Reward Hypothesis

Sutton's **reward hypothesis** (2004) states:

> "All of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)."

The 2021 paper "Reward is Enough" (Silver, Singh, Precup, Sutton) extends this:

> "Intelligence, and its associated abilities, can be understood as subserving the maximization of reward."

**Implications:**
- No need for separate modules for perception, language, planning
- All capabilities emerge from reward maximization given sufficient compute
- Simple reward signals (win/loss) can drive complex behavior

**Melee application:** This supports using sparse rewards (game outcome) over shaped rewards (damage, positioning). Given enough training, the model should discover:
- Which positions are advantageous
- When to approach vs. retreat
- Character-specific optimal strategies

### The Bitter Lesson

Sutton's 2019 essay "The Bitter Lesson" is perhaps the most influential meta-observation in AI:

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

**Historical evidence:**
| Domain | Human Knowledge Approach | Scale Approach | Winner |
|--------|-------------------------|----------------|--------|
| Chess | Hand-crafted evaluation | MCTS + learning | AlphaZero |
| Go | Expert patterns | Self-play RL | AlphaGo Zero |
| Speech | Phoneme models | End-to-end DNNs | Scale |
| Vision | Feature engineering | ConvNets + data | Scale |
| NLP | Linguistic rules | Transformers + data | Scale |

**2024 empirical validation:** "Learning the Bitter Lesson: Empirical Evidence from 20 Years of CVPR Proceedings" found:
- "Scalability with Computation" dimension shows consistent upward trend
- "Learning Over Engineering" dimension shows consistent upward trend
- Both align with the shift toward foundation models

**Sutton's position on LLMs (2024):**

> "I expect there to be systems that can learn from experience. Which could perform much better and be much more scalable. In which case, it will be another instance of the bitter lesson."

> "The scalable method is you learn from experience. You try things, you see what works. No one has to tell you. First of all, you have a goal. Without a goal, there's no sense of right or wrong or better or worse. Large language models are trying to get by without having a goal."

**Critical clarification (Twitter, 2023):**
> "The point of the bitter lesson is that the right learning algorithms (those that scale efficiently with massive computation) are exactly what we need. Massive computation does not alleviate the need for data efficiency."

This matters for ExPhil: we need algorithms that scale, not just raw compute.

---

## Synthesis: Where LeCun and Sutton Converge and Diverge

### Points of Agreement

| Aspect | LeCun | Sutton | Synthesis |
|--------|-------|--------|-----------|
| **Scale matters** | World models need massive pretraining | General methods + compute wins | Both predict scale dominates engineering |
| **Learning > engineering** | Let encoder learn what matters | Bitter Lesson | Minimize hand-crafted features |
| **Temporal credit** | Hierarchical prediction | TD(λ) eligibility traces | Multi-scale credit assignment |
| **Intrinsic motivation** | Curiosity in world model | Exploration bonuses | Both see value in novelty-seeking |

### Points of Tension

| Aspect | LeCun | Sutton | Resolution for ExPhil |
|--------|-------|--------|----------------------|
| **Role of reward** | Intrinsic + cost modules | Reward is enough | Start with sparse reward, add intrinsic if needed |
| **World models** | Explicit predictive model | Often implicit in value function | Explicit world model enables planning |
| **Representation** | Learned via SSL | Emerges from RL | Hybrid: JEPA pretraining → RL finetuning |
| **Architecture** | JEPA modules | General RL algorithms | Use JEPA architecture with RL training |

### The Key Insight for ExPhil

**LeCun provides the "what":** Build a world model that predicts in representation space, enable hierarchical planning.

**Sutton provides the "how":** Train with scalable RL algorithms, trust reward maximization, avoid over-engineering.

**Combined approach:**
1. Use JEPA-style architecture to learn Melee representations
2. Train a world model that predicts game state transitions
3. Use RL (PPO/TD) to optimize policy within this learned model
4. Scale model size and training compute aggressively

---

## Applications to ExPhil

### Current Architecture Analysis

ExPhil's current architecture through the lens of JEPA/Bitter Lesson:

**What we're doing right:**

| Aspect | Current Implementation | JEPA/Bitter Lesson Alignment |
|--------|----------------------|------------------------------|
| **Learned embeddings** | 287 dims (action/char IDs → 64-dim embeddings) | ✓ Learning over engineering |
| **Temporal backbones** | Mamba, Attention, LSTM options | ✓ Multiple approaches to test |
| **Self-play infrastructure** | GenServer-based with Elo | ✓ Scalable RL training |
| **Autoregressive heads** | 6 heads for controller output | ✓ Coordination structure |

**Areas for improvement:**

| Aspect | Current | JEPA/Bitter Lesson Suggestion |
|--------|---------|------------------------------|
| **Prediction target** | Classify controller actions | Predict state representations |
| **World model** | Implicit in policy | Explicit learned dynamics |
| **Representation learning** | Supervised on actions | Self-supervised from states |
| **Planning** | Reactive (no lookahead) | MPC-style with world model |
| **Hierarchy** | Flat policy | Multi-timescale H-JEPA |

### JEPA-Inspired Modifications

#### Modification 1: Representation Learning via State Prediction

Instead of only training on (state → action) pairs, add a **state prediction objective**:

```
Current:    state_t → Policy → action_t (supervised)

JEPA-style: state_t → Encoder → repr_t
            repr_t + action_t → Predictor → pred_{t+1}
            state_{t+1} → Encoder → repr_{t+1}
            Loss: ||pred_{t+1} - repr_{t+1}||²
```

**Benefits:**
- Encoder learns task-relevant representations
- Predictor becomes an explicit world model
- Representations transfer across characters/stages

**Implementation in ExPhil:**

```elixir
# lib/exphil/networks/world_model.ex
defmodule ExPhil.Networks.WorldModel do
  @moduledoc """
  JEPA-style world model for Melee state prediction.
  """

  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)

    # State encoder (shared for current and target states)
    encoder = Axon.input("state", shape: {nil, @state_dim})
    |> Axon.dense(hidden_size, activation: :gelu)
    |> Axon.dense(hidden_size, name: "state_repr")

    # Action-conditioned predictor
    predictor_input = Axon.concatenate([
      Axon.nx(encoder, & &1),
      Axon.input("action", shape: {nil, @action_dim})
    ])

    predicted_repr = predictor_input
    |> Axon.dense(hidden_size, activation: :gelu)
    |> Axon.dense(hidden_size, name: "predicted_repr")

    %{encoder: encoder, predictor: predicted_repr}
  end
end
```

#### Modification 2: Hierarchical Policy Architecture

Implement H-JEPA-style multi-timescale planning:

```
High-level (every 60 frames):
  "What strategic phase am I in? (neutral/advantage/disadvantage)"
  → Outputs goal representation

Mid-level (every 10 frames):
  "What tactical objective pursues this goal? (approach/retreat/combo/edgeguard)"
  → Outputs sub-goal representation

Low-level (every frame):
  "What input achieves this sub-goal?"
  → Outputs controller action
```

**Implementation sketch:**

```elixir
# lib/exphil/networks/hierarchical_policy.ex
defmodule ExPhil.Networks.HierarchicalPolicy do
  @high_level_frames 60   # Strategic decisions
  @mid_level_frames 10    # Tactical decisions

  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)

    # High-level: strategic goal (runs every 60 frames)
    high_level = build_high_level_policy(hidden_size)

    # Mid-level: tactical sub-goal (runs every 10 frames)
    mid_level = build_mid_level_policy(hidden_size)

    # Low-level: frame-by-frame actions
    low_level = build_low_level_policy(hidden_size)

    %{high: high_level, mid: mid_level, low: low_level}
  end

  def forward(model, state, high_goal, mid_goal, frame) do
    # Update high-level goal every 60 frames
    high_goal = if rem(frame, @high_level_frames) == 0 do
      Axon.predict(model.high, %{state: state})
    else
      high_goal
    end

    # Update mid-level goal every 10 frames
    mid_goal = if rem(frame, @mid_level_frames) == 0 do
      Axon.predict(model.mid, %{state: state, goal: high_goal})
    else
      mid_goal
    end

    # Always compute low-level action
    action = Axon.predict(model.low, %{
      state: state,
      goal: high_goal,
      sub_goal: mid_goal
    })

    {action, high_goal, mid_goal}
  end
end
```

#### Modification 3: Model-Predictive Control for Planning

Instead of reactive policy execution, use the world model for lookahead:

```
At each decision point:
1. Generate K candidate action sequences (random or learned proposal)
2. For each sequence, simulate N steps using world model
3. Score each trajectory using value function
4. Execute first action of best trajectory
5. Repeat
```

This is exactly what V-JEPA 2-AC does for robot control.

**Melee-specific considerations:**
- K=64-256 candidate sequences (parallel on GPU)
- N=10-30 frame lookahead (reaction time horizon)
- Score: estimated damage dealt - damage taken + position value

### Bitter Lesson Alignment

Following Sutton's principles, here's what we should minimize vs maximize:

**Minimize (hand-engineered knowledge):**

| Current | Action | Rationale |
|---------|--------|-----------|
| Character-specific context windows | Remove, let model learn | Don't encode "Mewtwo needs 90 frames" |
| Shaped rewards (approach, combo, etc.) | Start with sparse (win/loss) | Let model discover what matters |
| Fixed embedding features | Experiment with minimal embeddings | Test if model learns what it needs |
| Per-character models | Single model + char ID | Scale over specialization |

**Maximize (scalable components):**

| Current | Action | Rationale |
|---------|--------|-----------|
| Model size: 256-512 hidden | Scale to 1024+ | Bigger models learn more |
| Training data: limited replays | Use ALL available Slippi data | More data > better features |
| Training compute: single GPU | Distributed training on cluster | Scale enables generalization |
| Self-play iterations | Run longer | More experience = better policy |

### Proposed Experiments

#### Experiment 1: JEPA-Style Representation Learning

**Hypothesis:** Self-supervised state prediction pretraining improves downstream policy learning.

**Protocol:**
1. Pretrain encoder + predictor on state transitions (no action labels)
2. Freeze encoder, train policy head on (repr → action)
3. Compare to direct (state → action) training

**Metrics:**
- BC loss (lower = better imitation)
- Representation quality (linear probe on game events)
- Sample efficiency (games to X% accuracy)

#### Experiment 2: Sparse vs Shaped Rewards

**Hypothesis:** With sufficient scale, sparse rewards (win/loss) match or beat shaped rewards.

**Protocol:**
1. Train with shaped rewards (current): damage + position + combo + edgeguard
2. Train with sparse rewards: +1 win, -1 loss, 0 otherwise
3. Match total training compute (samples × model size)

**Metrics:**
- Win rate vs CPU levels 1-9
- Win rate in self-play tournament
- Strategy diversity (action entropy per situation)

#### Experiment 3: Hierarchical vs Flat Policy

**Hypothesis:** H-JEPA-style hierarchy improves long-horizon planning.

**Protocol:**
1. Train flat policy (current architecture)
2. Train 3-level hierarchical policy (60/10/1 frame decisions)
3. Compare on tasks requiring planning (edgeguarding, tech chasing)

**Metrics:**
- Edgeguard success rate
- Tech chase conversion rate
- Combo length distribution

#### Experiment 4: Model-Based Planning (MPC)

**Hypothesis:** Learned world model + MPC outperforms reactive policy.

**Protocol:**
1. Train world model on state transitions
2. Use world model for N-step lookahead planning
3. Compare to reactive policy on same backbone

**Metrics:**
- Reaction time distribution (model-based can "pre-react")
- Win rate in neutral situations
- Computational overhead (must stay under 16.7ms)

#### Experiment 5: Minimal Embedding Ablation

**Hypothesis:** Model can learn effective representations from minimal input.

**Protocol:**
1. Full embedding: 287 dims (current default)
2. Minimal embedding: ~50 dims (just positions, percent, stocks, action)
3. Flat embedding: ~200 dims (all libmelee fields, no structure)

**Metrics:**
- BC loss (is minimal sufficient?)
- Training time (smaller = faster)
- Win rate after same compute budget

---

## Implementation Roadmap

### Phase 1: Representation Learning (Month 1)

**Goal:** Add JEPA-style state prediction as auxiliary objective.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `WorldModel` module | Medium | None |
| Add state prediction loss to training | Low | WorldModel |
| Pretrain encoder on Slippi replays (SSL) | Medium | WorldModel |
| Evaluate representation quality | Low | Pretrained encoder |

**Success criteria:** Pretrained encoder matches or beats random init on BC loss.

### Phase 2: Scaling Experiments (Month 2)

**Goal:** Test Bitter Lesson predictions about scale.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Scale model to 1024+ hidden | Low | None |
| Download 100k+ Slippi replays | Medium | Storage |
| Distributed training setup | High | Infrastructure |
| Run sparse vs shaped reward experiment | Medium | Self-play infra |

**Success criteria:** Larger models + more data = better performance (validate scaling laws).

### Phase 3: Hierarchical Architecture (Month 3)

**Goal:** Implement H-JEPA-style multi-timescale planning.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `HierarchicalPolicy` module | High | None |
| Design goal representation space | Medium | Research |
| Train hierarchical policy | High | Infrastructure |
| Compare to flat baseline | Medium | Both implementations |

**Success criteria:** Hierarchical policy shows improved performance on planning tasks.

### Phase 4: Model-Based Planning (Month 4+)

**Goal:** Use learned world model for lookahead.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Train accurate world model | High | Data |
| Implement MPC planning loop | Medium | WorldModel |
| Optimize for real-time (< 16.7ms) | High | Engineering |
| Evaluate vs reactive policy | Medium | Both implementations |

**Success criteria:** MPC achieves better win rate within latency budget.

---

## References

### Yann LeCun's Work

1. **A Path Towards Autonomous Machine Intelligence** (2022)
   - [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
   - Introduces JEPA, H-JEPA, world model architecture

2. **V-JEPA: The next step toward advanced machine intelligence** (2024)
   - [Meta AI Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
   - Video understanding without generation

3. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** (2025)
   - [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
   - Zero-shot robot control, MPC planning

4. **VL-JEPA: Joint Embedding Predictive Architecture for Vision-language** (2025)
   - [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)
   - Language integration, 50% fewer parameters than autoregressive VLMs

5. **Introduction to Latent Variable Energy-Based Models** (2023)
   - [arXiv:2306.02572](https://arxiv.org/abs/2306.02572)
   - Tutorial on EBMs and connection to JEPA

6. **A Tutorial on Energy-Based Learning** (2006)
   - [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
   - Foundational EBM concepts

### Richard Sutton's Work

1. **The Bitter Lesson** (2019)
   - [incompleteideas.net](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
   - Scale beats engineering

2. **Learning to Predict by the Methods of Temporal Differences** (1988)
   - [PDF](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
   - Foundational TD learning paper

3. **Reward is Enough** (2021)
   - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370221000862)
   - Intelligence emerges from reward maximization

4. **Intrinsically Motivated Reinforcement Learning** (2004)
   - [NeurIPS](https://papers.nips.cc/paper/2004/hash/4be5a36cbaca8ab9d2066debfe4e65c1-Abstract.html)
   - Curiosity and exploration

5. **Reinforcement Learning: An Introduction** (2018, 2nd ed.)
   - [Book website](http://incompleteideas.net/book/the-book-2nd.html)
   - Comprehensive RL textbook

### Empirical Validation

1. **Learning the Bitter Lesson: Empirical Evidence from 20 Years of CVPR Proceedings** (2024)
   - [arXiv:2410.09649](https://arxiv.org/html/2410.09649v1)
   - Quantitative support for Bitter Lesson

2. **Demystifying MuZero Planning: Interpreting the Learned Model** (2024)
   - [arXiv:2411.04580](https://arxiv.org/html/2411.04580v2)
   - World model planning analysis

### Related Work

1. **MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model** (2019)
   - [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
   - Learned world model + MCTS

2. **Decision Transformer: RL via Sequence Modeling** (2021)
   - [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
   - RL as sequence prediction

3. **Dwarkesh Patel Interview with Richard Sutton** (2024)
   - [Podcast](https://www.dwarkesh.com/p/richard-sutton)
   - Sutton's views on LLMs and RL
