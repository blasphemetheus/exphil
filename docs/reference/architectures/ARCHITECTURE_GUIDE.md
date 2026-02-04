# Neural Network Architectures - A Beginner's Guide

This guide explains all 15 backbone architectures supported by ExPhil in beginner-friendly terms. For benchmark data and CLI options, see [ARCHITECTURES.md](ARCHITECTURES.md).

## The Big Picture

Neural networks for game AI need to do two things:
1. **Understand the current game state** (where is everyone, what's happening)
2. **Remember what happened before** (to predict patterns, react to combos)

Different architectures handle #2 (memory/context) in different ways. That's the main thing that distinguishes them.

---

## Table of Contents

- [Single-Frame (No Memory)](#single-frame-no-memory)
  - [MLP](#mlp---multi-layer-perceptron)
- [Recurrent (Sequential Memory)](#recurrent-sequential-memory)
  - [LSTM](#lstm---long-short-term-memory)
  - [GRU](#gru---gated-recurrent-unit)
- [Attention-Based (Parallel Context)](#attention-based-parallel-context)
  - [Attention](#attention-sliding-window)
  - [Decision Transformer](#decision-transformer)
- [State Space Models](#state-space-models-ssm)
  - [Mamba](#mamba---selective-state-space-model)
  - [Mamba-2 SSD](#mamba-2-ssd---structured-state-space-duality)
  - [S5](#s5---simplified-state-space)
- [Hybrid Architectures](#hybrid-architectures)
  - [Jamba](#jamba---mamba--attention-hybrid)
  - [Zamba](#zamba---shared-attention-hybrid)
- [Linear Attention Variants](#linear-attention-variants)
  - [GLA](#gla---gated-linear-attention)
  - [RWKV](#rwkv---receptance-weighted-key-value)
  - [HGRN](#hgrn---hierarchical-gated-recurrent-network)
- [Continuous-Time (Neural ODE)](#continuous-time-neural-ode)
  - [Liquid](#liquid---liquid-neural-networks)
- [Summary](#summary-which-to-choose)

---

## Single-Frame (No Memory)

### MLP - Multi-Layer Perceptron

**Why the name:** It's a "perceptron" (single neuron) concept extended to "multiple layers" stacked together.

**What it is:** The simplest neural network. Just layers of neurons where each layer connects to the next. Like a flowchart: input → process → process → output.

**The catch:** No memory. It only sees the current frame. Can't recognize "opponent is doing a 3-hit combo" because it doesn't remember the previous 2 hits.

**When to use:** Fast iteration, baselines, or when temporal context doesn't matter.

```
Frame → [Layer 1] → [Layer 2] → [Layer 3] → Action
         (no memory between frames)
```

**Melee example:** An MLP sees Fox at position (50, 0) but doesn't know if Fox is running right (will be at 55 next frame) or running left (will be at 45). It lacks the context.

---

## Recurrent (Sequential Memory)

These process frames one-by-one and maintain a "hidden state" that acts as memory.

### LSTM - Long Short-Term Memory

**Why the name:** Traditional RNNs had "short-term memory" (forgot things quickly due to vanishing gradients). This architecture added gates to create "long-term memory" that can persist across many timesteps.

**What it is:** Has three "gates" that control information flow:
- **Forget gate:** What to erase from memory ("opponent switched characters, forget old patterns")
- **Input gate:** What new info to store ("opponent just grabbed, remember this")
- **Output gate:** What to use for the current decision ("I'm in hitstun, output defensive option")

**The catch:** Processes frames sequentially (can't parallelize), so it's slow. 229ms inference = impossible for 60 FPS.

**When to use:** Best accuracy in benchmarks, but only for offline analysis or research.

```
Frame 1 → [LSTM] → memory₁
Frame 2 → [LSTM + memory₁] → memory₂
Frame 3 → [LSTM + memory₂] → memory₃ → Action
```

**The gates visualized:**
```
                    ┌─────────────┐
Previous memory ───►│ Forget Gate │──► What to keep
                    └─────────────┘
                    ┌─────────────┐
Current input ─────►│ Input Gate  │──► What to add
                    └─────────────┘
                    ┌─────────────┐
Combined memory ───►│ Output Gate │──► What to output
                    └─────────────┘
```

### GRU - Gated Recurrent Unit

**Why the name:** It's a "recurrent unit" (processes sequences) with "gates" (like LSTM), but simplified.

**What it is:** LSTM's simpler cousin. Combines forget+input gates into one "update gate", and removes the output gate. Fewer parameters, faster training, similar performance.

**The tradeoff:** Slightly less expressive than LSTM, but often "good enough" and trains faster.

**When to use:** When you want LSTM-like behavior but faster, or when LSTM is overfitting.

---

## Attention-Based (Parallel Context)

### Attention (Sliding Window)

**Why the name:** The network learns to "pay attention" to relevant parts of the input sequence, like how you focus on the opponent's character rather than the background.

**What it is:** Every frame can directly "look at" every other frame in the window. Computes relevance scores between all pairs of frames.

**The math:** For 60 frames, it computes 60×60 = 3,600 attention scores. This is why it's O(L²) - quadratic in sequence length.

**How attention works:**
```
All 60 frames loaded at once
    ↓
[Which frames matter for predicting frame 30's action?]
    ↓
Frame 12: 0.4 relevance (opponent started attack)
Frame 28: 0.8 relevance (attack connecting)
Frame 29: 0.9 relevance (hitstun started)
    ↓
Weighted combination of relevant frames → Action
```

**The attention mechanism:**
```
Q (Query):  "What am I looking for?"
K (Key):    "What do I contain?"
V (Value):  "What information do I provide?"

Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Each frame asks: "Which other frames are relevant to me?"
```

**When to use:** Good accuracy, parallelizable (fast training), but memory-hungry for long sequences.

### Decision Transformer

**Why the name:** It's a "Transformer" (attention-based architecture, like GPT) that makes "decisions" based on a target return/goal.

**What it is:** Instead of just predicting "what would a human do?", it predicts "what would a human do IF they wanted to achieve X reward?". You condition on the desired outcome.

**Key innovation:** Input is (Return-to-go, State, Action) triplets.
- **Return-to-go** = "how much reward do I want from here to the end of the game?"

**The paradigm shift:**
```
Traditional:  State → Model → Action (imitate human)

Decision Transformer:
  "I want +2 stock lead" + State → Model → Aggressive Action
  "I want to survive"    + State → Model → Defensive Action
```

**When to use:** Goal-directed behavior. Want aggressive play? Set high return target. Want safe play? Set conservative target. Great for controllable AI behavior.

**Melee application:**
- Return = stock differential at end of game
- High return-to-go → Model outputs aggressive, kill-seeking actions
- Low return-to-go → Model plays safe, avoids risk

---

## State Space Models (SSM)

These model sequences as continuous dynamical systems. Think of it like physics equations that evolve over time, rather than discrete memory cells.

### Mamba - Selective State Space Model

**Why the name:** Named by the authors (probably after the snake - fast, efficient, and selective). The technical name is "S6" (Selective Structured State Space Sequence model), but "Mamba" is catchier.

**What it is:** Maintains a hidden "state" that evolves according to learned dynamics. The key innovation: the dynamics are **input-dependent** (selective).

**Traditional SSM (S4):**
```
h(t) = A · h(t-1) + B · x(t)    # A, B are fixed
y(t) = C · h(t)

Same transition rules regardless of input
```

**Selective SSM (Mamba):**
```
h(t) = A(x) · h(t-1) + B(x) · x(t)    # A, B computed FROM input
y(t) = C(x) · h(t)

"Should I remember this? Let me look at what it is first."
```

**Why "selective" matters:**
- The model can learn to **remember** important events (opponent grabbed → high retention)
- The model can learn to **forget** irrelevant noise (random movement → low retention)
- This happens dynamically based on the content, not fixed rules

**Why it's fast:**
- O(L) complexity (linear in sequence length)
- Can be parallelized during training using scan operations
- Inference can cache the state (process one frame at a time)

**When to use:** Best balance of speed and accuracy. Recommended default for production.

### Mamba-2 SSD - Structured State Space Duality

**Why the name:** "SSD" = Structured State Space Duality. The Mamba-2 paper proved that SSMs and attention are mathematically dual - two different views of the same underlying computation.

**What it is:** Mamba version 2. The insight: SSM computation can be written two equivalent ways:
- **Recurrent form:** Process one step at a time → fast for inference
- **Matrix form:** One big matrix multiply → fast for training (uses tensor cores)

**The duality:**
```
Recurrent (inference):          Matrix (training):
for t in 1..L:                  Y = (L ⊙ (CB)) × X
  h = A·h + B·x
  y = C·h                       One matmul, tensor cores go brrrr
```

**When to use:** When you want Mamba but faster training on modern GPUs with tensor cores.

### S5 - Simplified State Space

**Why the name:** "S5" = Simplified Structured State Space Sequence model. The "5" distinguishes it from S4 (the predecessor, which was already called Structured State Space).

**What it is:** Uses a single big state matrix (MIMO = Multiple-Input-Multiple-Output) instead of many small independent ones (SISO = Single-Input-Single-Output like Mamba uses).

**SISO (Mamba) vs MIMO (S5):**
```
SISO (Mamba):                    MIMO (S5):
[SSM₁] [SSM₂] [SSM₃] ...        [    Big SSM    ]
Each operates independently      All dimensions interact
Simpler, but limited coupling    Richer, but more compute
```

**When to use:** Ablation studies - helps understand whether Mamba's complexity is necessary, or if simpler approaches work just as well.

---

## Hybrid Architectures

These combine multiple approaches to get the best of both worlds.

### Jamba - Mamba + Attention Hybrid

**Why the name:** "J" + "amba" = Jamba. Created by AI21 Labs. The J might reference "joint" (combining approaches), or it's just a catchy name.

**What it is:** Alternates between Mamba blocks and Attention blocks in a fixed pattern. Gets benefits of both:
- **Mamba's efficiency:** O(L) complexity for most layers
- **Attention's global context:** Periodic attention layers capture long-range dependencies

**The structure:**
```
[Mamba] → [Mamba] → [Attention] → [Mamba] → [Mamba] → [Attention] → ...
   └── Local patterns ──┘    └── Global ──┘    └── Local ──┘    └── Global
```

**Why hybrid helps:** Mamba is great at local patterns but can miss distant relationships. Attention is great at global patterns but expensive. Combining them: cheap local processing with occasional global synchronization.

**When to use:** When pure Mamba misses long-range patterns, or when you have compute budget for some attention layers.

### Zamba - Shared Attention Hybrid

**Why the name:** "Z" + "amba" - created by Zyphra as their version of the hybrid approach.

**What it is:** Like Jamba, but with a twist: uses ONE attention layer with **shared weights** that gets called multiple times at different positions in the network.

**Jamba vs Zamba:**
```
Jamba:  [Mamba] [Mamba] [Attn₁] [Mamba] [Mamba] [Attn₂] [Mamba] [Mamba] [Attn₃]
                         ↑                       ↑                       ↑
                    Different weights for each attention layer

Zamba:  [Mamba] [Mamba] [Attn*] [Mamba] [Mamba] [Attn*] [Mamba] [Mamba] [Attn*]
                         ↑                       ↑                       ↑
                    Same weights reused! (the * means shared)
```

**Why share weights?**
- Fewer parameters (more efficient)
- The insight: you don't need different attention patterns at each layer - one good attention pattern applied repeatedly works well

**When to use:** Want hybrid benefits with fewer parameters and less memory.

---

## Linear Attention Variants

Standard attention is O(L²) because it computes all pairwise relationships. These architectures approximate attention with O(L) complexity.

### GLA - Gated Linear Attention

**Why the name:** "Gated" (has learnable gates like LSTM) + "Linear Attention" (linear complexity approximation of attention).

**What it is:** Reformulates attention to avoid materializing the L×L attention matrix. Uses data-dependent gating to maintain expressiveness despite the simplification.

**How it achieves linear complexity:**
```
Standard Attention:
  Attention = softmax(QK^T) × V     # QK^T is L×L matrix

Linear Attention (simplified):
  Attention = Q × (K^T × V)         # K^T×V is d×d matrix (much smaller!)

The trick: associativity lets us compute (K^T × V) first
```

**The gating:** Adds learnable gates that control information flow, compensating for the lost expressiveness from removing softmax.

**When to use:** Short sequences where you want attention-like behavior but faster. Particularly good when L (sequence length) is large relative to d (dimension).

### RWKV - Receptance Weighted Key Value

**Why the name:** The four main components spell out the name:
- **R**eceptance - how much to accept/receive
- **W**eighted - time-decay weighting
- **K**ey - what to match against
- **V**alue - what to retrieve

**What it is:** A clever reformulation that gives attention-like behavior with RNN-like efficiency. The key insight: you can reformulate attention as a recurrent computation.

**The WKV mechanism:**
```
At each timestep t:
  wkv_t = Σᵢ (e^(-(t-i)w + kᵢ) × vᵢ) / Σᵢ (e^(-(t-i)w + kᵢ))

This looks like attention, but can be computed incrementally:
  - Keep running sums
  - Update with each new token
  - O(1) memory per step!
```

**Why it's special:** O(1) memory during inference. You don't need to store the whole sequence - just running statistics.

**When to use:** When memory is critical (inference on limited hardware, very long sequences).

### HGRN - Hierarchical Gated Recurrent Network

**Why the name:** "Hierarchical" (operates at multiple time scales) + "Gated Recurrent Network" (RNN with gates).

**What it is:** Processes sequences at multiple time scales simultaneously. Some units track fast patterns (frame-by-frame), others track slow patterns (over many frames).

**The hierarchy:**
```
Fast units:   [●●●●●●●●●●●●]  Track individual moves, reactions
              Frame-by-frame changes

Medium units: [●───●───●───●]  Track sequences, combos
              Updates every few frames

Slow units:   [●───────────●]  Track match flow, adaptation
              Updates over longer spans
```

**Why hierarchy helps in Melee:**
- Fast: "Opponent pressed A" (frame-level)
- Medium: "Opponent is doing a jab combo" (move sequence)
- Slow: "Opponent tends to approach with dash attack" (playstyle)

**When to use:** When your data has multi-scale temporal structure (like Melee with frame-level tech + match-level strategy).

---

## Continuous-Time (Neural ODE)

### Liquid - Liquid Neural Networks

**Why the name:** The hidden state "flows" continuously like liquid, rather than jumping discretely between frames. Also called "LTC" (Liquid Time-Constant) networks.

**What it is:** Instead of discrete update rules (h_new = f(h_old, x)), models the hidden state as a continuous differential equation that evolves over time:

```
dx/dt = (activation - x) / τ

Where:
  x = current hidden state
  activation = where the state "wants" to be (computed from input)
  τ (tau) = time constant (how fast to adapt)
```

**The time constant τ:** This is learned per-neuron and controls adaptation speed:
- **Small τ:** Fast adaptation, quickly follows new inputs
- **Large τ:** Slow adaptation, maintains momentum, smooths noise

**ODE Solvers:** Since it's a differential equation, we need numerical solvers:
- `:euler` - Simplest, fastest, least accurate
- `:midpoint` - Better accuracy, moderate speed
- `:rk4` - Good accuracy, the default
- `:dopri5` - Best accuracy, adaptive step size (Dormand-Prince 4/5)

**Why continuous-time?**
- More natural for continuous control (Melee analog sticks!)
- Can adapt integration precision based on how "interesting" the dynamics are
- Robust to irregular time intervals

**When to use:** When you want smooth, adaptive dynamics. Good for continuous control problems. Interesting research direction for game AI.

```
Traditional RNN:          Liquid Network:
h₁ → h₂ → h₃ → h₄        h(t) flows continuously
Discrete jumps            ~~~~~~~~~~~~~~~~~~~→
                          Smooth evolution
```

---

## Summary: Which to Choose?

### Quick Decision Guide

| Your Need | Best Choice | Why |
|-----------|-------------|-----|
| **Just starting out** | MLP | Simple, fast, good baseline |
| **Best accuracy (offline)** | LSTM | Highest benchmark scores |
| **Production / 60 FPS** | Mamba | Best speed/accuracy tradeoff |
| **Goal-directed behavior** | Decision Transformer | Controllable via return target |
| **Very limited memory** | RWKV | O(1) inference memory |
| **Multi-scale patterns** | HGRN | Hierarchical temporal modeling |
| **Smooth dynamics** | Liquid | Continuous-time adaptation |
| **Hybrid approach** | Jamba or Zamba | Mamba efficiency + Attention context |
| **Research / ablations** | S5 | Simplified baseline |

### The Speed/Accuracy Tradeoff

```
More Context/Memory ─────────────────────────────► Better Decisions
     │
     │  MLP (none)
     │       ↓
     │  RWKV, GLA, HGRN (linear approximations)
     │       ↓
     │  Mamba, S5 (state space)
     │       ↓
     │  Jamba, Zamba (hybrid)
     │       ↓
     │  Attention, Decision Transformer (full pairwise)
     │       ↓
     │  LSTM, GRU (sequential, thorough)
     ▼
Slower Inference
```

### For 60 FPS Melee

Target: <16.7ms per frame (subtract ~2ms for game state I/O = **<15ms for model**)

| Architecture | Inference | Verdict |
|--------------|-----------|---------|
| MLP | ~9ms | ✅ Safe |
| Mamba | ~9ms | ✅ Safe |
| GLA, RWKV, HGRN | ~10-12ms | ✅ Safe |
| Attention | ~17ms | ⚠️ Borderline |
| Jamba, Zamba | ~15-20ms | ⚠️ Borderline |
| LSTM, GRU | ~150-230ms | ❌ Too slow |

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE FAMILY TREE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  No Memory          Sequential Memory       Parallel Context     │
│      │                     │                      │              │
│     MLP              ┌─────┴─────┐         ┌─────┴─────┐        │
│                      │           │         │           │        │
│                    LSTM        GRU    Attention    Decision     │
│                                            │      Transformer   │
│                                            │                    │
│  ┌─────────────────────────────────────────┼────────────────┐   │
│  │              LINEAR APPROXIMATIONS      │                │   │
│  │    ┌──────────┬──────────┬─────────┐   │                │   │
│  │   GLA       RWKV      HGRN        │   │                │   │
│  │    │          │          │         │   │                │   │
│  │    └──────────┴──────────┴─────────┘   │                │   │
│  └────────────────────────────────────────┘                │   │
│                                                             │   │
│  ┌──────────────────────────────────────────────────────────┘   │
│  │         STATE SPACE MODELS                                   │
│  │    ┌──────────┬──────────┬─────────┐                        │
│  │  Mamba    Mamba-2     S5        │                        │
│  │    │       SSD         │         │                        │
│  │    └────────┬──────────┘         │                        │
│  │             │                     │                        │
│  │      ┌──────┴──────┐             │                        │
│  │    Jamba        Zamba            │     CONTINUOUS TIME    │
│  │   (hybrid)    (shared)           │           │            │
│  │                                   │        Liquid         │
│  └───────────────────────────────────┴───────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Further Reading

- [ARCHITECTURES.md](ARCHITECTURES.md) - Benchmark data, CLI options, technical specs
- Individual architecture docs: [MLP](MLP.md), [LSTM](LSTM.md), [Mamba](MAMBA.md), etc.
- Original papers linked in each architecture's module documentation
