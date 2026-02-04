# Future Architectures and Research Directions for ExPhil

> **Purpose:** Comprehensive survey of architectures and research directions beyond ExPhil's current implementation. Organized by relevance to real-time game AI.

**Last Updated:** 2026-02-03

---

## Table of Contents

1. [Current Architecture Baseline](#current-architecture-baseline)
2. [Tier 1: High Priority Architectures](#tier-1-high-priority-architectures)
   - [xLSTM](#xlstm-extended-lstm)
   - [RWKV](#rwkv)
   - [Griffin/Hawk](#griffinhawk)
   - [RetNet](#retnet)
3. [Tier 2: Promising Research Directions](#tier-2-promising-research-directions)
   - [Diffusion Policy](#diffusion-policy)
   - [Flow Matching](#flow-matching)
   - [Action Chunking (ACT)](#action-chunking-act)
   - [Test-Time Compute Scaling](#test-time-compute-scaling)
4. [Tier 3: Architectural Variants](#tier-3-architectural-variants)
   - [Liquid Neural Networks](#liquid-neural-networks)
   - [KAN (Kolmogorov-Arnold Networks)](#kan-kolmogorov-arnold-networks)
   - [State Space Variants](#state-space-variants)
5. [Tier 4: Speculative/Long-Term](#tier-4-speculativelong-term)
   - [Neuro-Symbolic Approaches](#neuro-symbolic-approaches)
   - [Memory-Augmented Networks](#memory-augmented-networks)
6. [Research Areas by Problem](#research-areas-by-problem)
7. [Implementation Priority Matrix](#implementation-priority-matrix)
8. [References](#references)

---

## Current Architecture Baseline

ExPhil currently supports 6 backbone architectures:

| Backbone | Type | Complexity | Inference | Status |
|----------|------|------------|-----------|--------|
| MLP | Feedforward | O(1) | 2-5ms | ✅ Production |
| LSTM | Recurrent | O(L) seq | 220ms | ✅ Working |
| GRU | Recurrent | O(L) seq | ~150ms | ✅ Working |
| Mamba | SSM | O(L) parallel | **8.9ms** | ✅ Primary |
| Attention | Transformer | O(L²) | 17ms | ✅ Working |
| Jamba | Hybrid | O(L) + O(L²/n) | ~20ms | ✅ Working |

**Constraint:** Real-time play requires <16.7ms inference (60 FPS).

---

## Tier 1: High Priority Architectures

These architectures directly address ExPhil's constraints (real-time inference, temporal modeling) and have proven implementations.

### xLSTM (Extended LSTM)

**Source:** [Hochreiter et al., NeurIPS 2024](https://arxiv.org/abs/2405.04517)

**What it is:** The inventor of LSTM (Sepp Hochreiter) created xLSTM to address LSTM's fundamental limitations:
1. Inability to revise storage decisions
2. Limited storage capacity
3. Lack of parallelizability

**Key innovations:**
- **Exponential gating:** More expressive than sigmoid gating
- **sLSTM:** Scalar memory with new memory mixing
- **mLSTM:** Matrix memory, fully parallelizable with covariance update rule

**Why it matters for ExPhil:**
- O(L) time, O(1) memory complexity (same as Mamba)
- Outperforms Mamba on language modeling benchmarks
- Native support for RL and time series (Hochreiter's stated applications)
- Potentially better credit assignment than Mamba

**Performance claims:**
- Outperforms Transformers and SSMs on perplexity
- 7B parameter model trained on 2.3T tokens available
- Competitive with Llama-2 scale models

**Implementation complexity:** Medium
- Official PyTorch implementation available
- Would need Axon/Nx port

```elixir
# Proposed ExPhil integration
# lib/exphil/networks/xlstm.ex
defmodule ExPhil.Networks.XLSTM do
  @moduledoc """
  Extended LSTM with exponential gating and matrix memory.

  Two variants:
  - sLSTM: scalar memory, recurrent (for sequential processing)
  - mLSTM: matrix memory, parallelizable (for training)
  """

  def build_slstm_block(hidden_size, opts \\ []) do
    # Scalar LSTM with exponential gating
    # ...
  end

  def build_mlstm_block(hidden_size, opts \\ []) do
    # Matrix LSTM, parallelizable
    # ...
  end
end
```

---

### RWKV

**Source:** [Peng et al., EMNLP 2023](https://arxiv.org/abs/2305.13048), [RWKV-7 "Goose" 2025](https://github.com/BlinkDL/RWKV-LM)

**What it is:** RNN that trains like a Transformer. Combines efficient parallelizable training with O(1) inference memory.

**Key innovations:**
- **R, W, K, V mechanism:** Receptance (gating), Weight-decay, Key, Value
- **Linear attention with decay:** Smooth, learnable decay on past contributions
- **Dynamic State Evolution (RWKV-7):** Goes beyond attention/linear attention expressivity

**Why it matters for ExPhil:**
- Linear time O(L), constant space O(1)
- No KV-cache needed (unlike Transformers)
- Infinite context length theoretically possible
- RWKV-7 "Goose" specifically addresses expressivity limitations

**Performance claims:**
- Scaled to 14B parameters (largest dense RNN ever)
- Matches similarly-sized Transformers on benchmarks
- Vision-RWKV accepted at ICLR 2025 (Spotlight)

**Implementation complexity:** Medium-High
- Official PyTorch implementation available
- WKV kernel needs custom CUDA (or pure Nx fallback)

**Unique advantage:** The "free sentence embedding" property could enable better state representation for Melee game states.

---

### Griffin/Hawk

**Source:** [De et al., Google DeepMind 2024](https://arxiv.org/abs/2402.19427)

**What it is:** Google DeepMind's answer to Mamba. Hawk is pure gated linear recurrence; Griffin adds local attention.

**Key innovations:**
- **RG-LRU (Real-Gated Linear Recurrent Unit):** Novel recurrent layer
- **Local attention mixing:** Griffin interleaves RG-LRU with sliding-window attention
- **Hardware efficiency:** Matches Transformer efficiency during training

**Why it matters for ExPhil:**
- Exceeds Mamba on downstream tasks
- Matches Llama-2 with 6x fewer training tokens
- Extrapolates to 4x longer sequences than training
- Lower latency, higher throughput than Transformers

**Performance claims:**
- 14B parameter models trained
- Better than Mamba on most benchmarks
- Excellent length extrapolation

**Implementation complexity:** Medium
- Architecture is well-documented
- RG-LRU is simpler than Mamba's selective scan

```elixir
# Proposed ExPhil integration
# lib/exphil/networks/griffin.ex
defmodule ExPhil.Networks.Griffin do
  @moduledoc """
  Google DeepMind's Griffin: RG-LRU + Local Attention hybrid.

  Similar philosophy to Jamba but with simpler recurrent unit.
  """

  @doc """
  Real-Gated Linear Recurrent Unit.
  Simpler than Mamba's selective SSM.
  """
  def build_rg_lru(hidden_size, opts \\ []) do
    # Implementation based on paper equations
  end

  def build_griffin_block(hidden_size, opts \\ []) do
    attention_every = Keyword.get(opts, :attention_every, 3)
    # Interleave RG-LRU with local sliding-window attention
  end
end
```

---

### RetNet

**Source:** [Sun et al., Microsoft 2023](https://arxiv.org/abs/2307.08621)

**What it is:** Microsoft's "successor to Transformer" with three computation paradigms in one architecture.

**Key innovations:**
- **Retention mechanism:** Replaces attention with decay-based retention
- **Triple paradigm:** Same weights work for parallel (training), recurrent (inference), and chunkwise (long sequences)
- **Multi-scale retention:** Different decay rates for different heads

**Why it matters for ExPhil:**
- O(1) inference complexity per token
- Training parallelism preserved
- Better memory/throughput than FlashAttention Transformers
- Outperforms RWKV, H3, Hyena

**Implementation complexity:** Medium
- Well-documented equations
- Multiple community implementations available

**Unique advantage:** The ability to switch between parallel (training) and recurrent (inference) mode without architecture changes is perfect for ExPhil's BC→inference pipeline.

---

## Tier 2: Promising Research Directions

These are research paradigms that could transform how ExPhil approaches the problem, not just architectural swaps.

### Diffusion Policy

**Source:** [Chi et al., IJRR 2024](https://arxiv.org/abs/2303.04137)

**What it is:** Represent robot policy as a conditional denoising diffusion process. Generate actions by iteratively denoising random noise.

**Key insights:**
- **Multimodal action distributions:** Naturally handles multiple valid actions (unlike classification which averages)
- **Action sequences:** Predicts whole chunks of future actions, not just next step
- **Stable training:** Regression objective more stable than adversarial

**Why it matters for ExPhil:**
- Melee has multimodal optimal actions (multiple "correct" responses)
- Could handle the autoregressive action head more naturally
- Proven 46.9% average improvement over baselines on robot tasks

**Challenge:** Diffusion requires multiple denoising steps at inference. Standard is 10-100 steps.

**Potential solution:**
- One-Step Diffusion Policy (OneDP) distills to single step
- Consistency models reduce steps to 1-4

```elixir
# Potential implementation
# lib/exphil/networks/diffusion_policy.ex
defmodule ExPhil.Networks.DiffusionPolicy do
  @moduledoc """
  Diffusion-based policy for multimodal action generation.

  Instead of classifying actions, iteratively denoise from
  random noise to valid action sequences.
  """

  @denoising_steps 4  # Use consistency distillation for speed

  def build(opts \\ []) do
    # U-Net style denoiser conditioned on state
  end

  def sample(model, params, state, opts \\ []) do
    # DDIM or consistency sampling for fast inference
  end
end
```

---

### Flow Matching

**Source:** [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)

**What it is:** Train continuous normalizing flows (CNFs) by regressing vector fields instead of maximum likelihood. Simpler and more stable than diffusion.

**Key insights:**
- **Simulation-free training:** No ODE solving during training
- **Optimal transport paths:** Straighter trajectories than diffusion
- **Faster sampling:** Fewer steps needed than diffusion

**Why it matters for ExPhil:**
- Could be even faster than diffusion for action generation
- Naturally handles continuous action spaces
- OT-CFM variant is particularly efficient

**Implementation complexity:** Medium
- TorchCFM library available
- Would need Nx port

---

### Action Chunking (ACT)

**Source:** [Zhao et al., RSS 2023](https://arxiv.org/abs/2304.13705)

**What it is:** Instead of predicting single actions, predict chunks of k future actions. Reduces compounding errors by k-fold.

**Key insights:**
- **Temporal consistency:** Actions within chunk are coherent
- **Error reduction:** Horizon effectively reduced by chunk size
- **CVAE training:** Handles multimodality via latent variable

**Why it matters for ExPhil:**
- Melee techniques span multiple frames (wavedash: ~10 frames)
- Current autoregressive head predicts frame-by-frame
- Chunking could capture technique-level actions

**Implementation:**
```elixir
# Proposed integration
defmodule ExPhil.Networks.ActionChunking do
  @chunk_size 10  # ~6 frames per Melee technique

  def build(opts \\ []) do
    # Predict next @chunk_size actions at once
    # Use CVAE for multimodality
  end

  def execute_chunk(actions, frame) do
    # Return action at frame index within chunk
    # Re-predict when chunk exhausted
  end
end
```

**Synergy:** Combines well with hierarchical policy - high/mid levels set goals, low level predicts action chunks.

---

### Test-Time Compute Scaling

**Source:** [Snell et al., 2024](https://openreview.net/forum?id=4FWAwZtd2n), [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

**What it is:** Allocate more compute at inference time for harder decisions. "Think longer on hard problems."

**Key mechanisms:**
- **Parallel scaling:** Generate multiple candidates, select best
- **Sequential scaling:** Iteratively refine answer
- **Self-consistency:** Sample multiple times, majority vote

**Why it matters for ExPhil:**
- Some Melee situations are harder than others (edgeguard vs neutral)
- Could allocate more compute for critical moments
- Natural fit with MPC planning (more candidates = better plan)

**Challenge:** Real-time constraint limits how much we can scale.

**Potential approach:**
- Detect "hard" situations (low confidence, critical moments)
- Budget extra compute for those frames
- Use action repeat on "easy" frames to bank compute

```elixir
defmodule ExPhil.Inference.AdaptiveCompute do
  @easy_budget_ms 5
  @hard_budget_ms 25
  @frame_budget_ms 16.7

  def infer_with_budget(state, model, banked_ms) do
    difficulty = estimate_difficulty(state)

    budget = if difficulty > @hard_threshold do
      min(@hard_budget_ms, @frame_budget_ms + banked_ms)
    else
      @easy_budget_ms
    end

    {action, compute_used} = infer_with_timeout(state, model, budget)
    new_banked = banked_ms + (@frame_budget_ms - compute_used)

    {action, max(0, new_banked)}
  end
end
```

---

## Tier 3: Architectural Variants

More speculative but potentially interesting.

### Liquid Neural Networks

**Source:** [Hasani et al., MIT 2021](https://arxiv.org/abs/2006.04439)

**What it is:** Continuous-time neural networks based on liquid time-constant (LTC) neurons. Dynamics evolve continuously, not discretely.

**Key properties:**
- Very compact (orders of magnitude fewer neurons)
- Naturally handles irregular time intervals
- Interpretable dynamics

**Why it might matter:**
- Could be extremely efficient for real-time
- Native handling of variable frame timing
- Potentially more robust to delay variations

**Note:** ExPhil already has `lib/exphil/networks/liquid.ex` in modified state - this could be expanded.

---

### KAN (Kolmogorov-Arnold Networks)

**Source:** [Liu et al., 2024](https://arxiv.org/abs/2404.19756)

**What it is:** Replace fixed activation functions with learnable univariate functions on edges (splines). Based on Kolmogorov-Arnold representation theorem.

**Key properties:**
- Learnable activation functions
- Often more parameter efficient
- Better interpretability
- Can outperform MLPs on some tasks

**Why it might matter:**
- Could reduce model size while maintaining performance
- Learnable activations might capture Melee-specific nonlinearities
- Active research area with rapid improvements

---

### State Space Variants

Beyond Mamba, several SSM variants exist:

| Variant | Key Innovation | Potential Benefit |
|---------|----------------|-------------------|
| **S5** | Simplified S4, diagonal state | Faster, simpler |
| **H3** | Hungry Hungry Hippos, gating | Better on language |
| **Hyena** | Long convolutions, implicit | Very long context |
| **Based** | Linear attention + sliding window | Simple, effective |

**Mamba-2** (already researched) unifies SSMs with attention through State Space Duality.

---

## Tier 4: Speculative/Long-Term

### Neuro-Symbolic Approaches

**Idea:** Combine neural networks with symbolic reasoning for Melee.

**Potential applications:**
- Explicit frame data knowledge (startup, active, endlag)
- Symbolic planning for combo trees
- Rule-based constraints (can't double jump if no jumps left)

**Challenge:** Integration complexity, may not scale (Bitter Lesson warning).

---

### Memory-Augmented Networks

**Examples:** Neural Turing Machines, Differentiable Neural Computers, Memory Networks

**Potential applications:**
- Explicit opponent modeling (remember their habits)
- Long-term strategy memory (they always tech in place)
- Experience retrieval (similar situations from training)

**Challenge:** Complex, may be superseded by scale.

---

## Research Areas by Problem

| Problem | Current Approach | Alternative Research |
|---------|------------------|---------------------|
| **Credit assignment** | TD(λ), Mamba memory | xLSTM matrix memory, Transformers |
| **Multimodal actions** | Softmax classification | Diffusion policy, Flow matching |
| **Long-term planning** | None (reactive) | MPC, Hierarchical policy, MCTS |
| **Real-time constraint** | Fast backbone (Mamba) | One-step diffusion, Griffin, RetNet |
| **Generalization** | Data augmentation | Meta-learning (MAML), Domain randomization |
| **Sample efficiency** | BC pretraining | Offline RL, Representation learning |
| **Opponent modeling** | Implicit in policy | Explicit memory, Theory of mind |

---

## Implementation Priority Matrix

Based on effort vs expected impact:

```
                    HIGH IMPACT
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │  Griffin/Hawk      │  Diffusion Policy  │
    │  xLSTM             │  Test-time Scaling │
    │  RetNet            │  Action Chunking   │
    │                    │                    │
LOW ├────────────────────┼────────────────────┤ HIGH
EFFORT                   │                    EFFORT
    │                    │                    │
    │  RWKV-7            │  Flow Matching     │
    │  Mamba-2 SSD       │  Liquid Networks   │
    │                    │  KAN               │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT
```

### Recommended Implementation Order

**Phase 1 (Next Month):**
1. **Griffin/Hawk** - Similar to existing Mamba, lower effort
2. **Action Chunking** - Orthogonal to backbone, can test with any architecture

**Phase 2 (Month 2-3):**
3. **xLSTM** - Promising, from LSTM inventor
4. **RetNet** - Multi-paradigm is elegant for training→inference

**Phase 3 (Month 3-4):**
5. **One-Step Diffusion** - If action multimodality is a bottleneck
6. **Test-Time Scaling** - If compute budget allows adaptive allocation

**Phase 4 (Long-term):**
7. **RWKV-7** - If infinite context proves valuable
8. **Flow Matching** - Cleaner than diffusion if diffusion works

---

## Quick Reference: Architecture Comparison

| Architecture | Training | Inference | Memory | Context | Expressivity |
|--------------|----------|-----------|--------|---------|--------------|
| Transformer | O(L²) | O(L²) | O(L²) | Limited | High |
| LSTM | O(L) | O(L) | O(1) | ~100s | Medium |
| Mamba | O(L) | O(L) | O(1) | Long | High |
| xLSTM | O(L) | O(L) | O(1) | Long | Higher |
| RWKV | O(L) | O(1) | O(1) | Infinite | High |
| Griffin | O(L) | O(L) | O(1) | 4x+ extrap | High |
| RetNet | O(L) | O(1) | O(1) | Long | High |

---

## References

### Architectures
- [xLSTM](https://arxiv.org/abs/2405.04517) - Hochreiter et al., NeurIPS 2024
- [RWKV](https://arxiv.org/abs/2305.13048) - Peng et al., EMNLP 2023
- [RWKV-7](https://github.com/BlinkDL/RWKV-LM) - BlinkDL, 2025
- [Griffin/Hawk](https://arxiv.org/abs/2402.19427) - De et al., Google DeepMind 2024
- [RetNet](https://arxiv.org/abs/2307.08621) - Sun et al., Microsoft 2023
- [Mamba-2](https://arxiv.org/abs/2405.21060) - Dao & Gu, ICML 2024

### Research Directions
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al., IJRR 2024
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Lipman et al., ICLR 2023
- [Action Chunking (ACT)](https://arxiv.org/abs/2304.13705) - Zhao et al., RSS 2023
- [Test-Time Scaling](https://openreview.net/forum?id=4FWAwZtd2n) - Snell et al., 2024
- [One-Step Diffusion](https://arxiv.org/abs/2403.03206) - OneDP, 2024

### Surveys
- [Post-Transformer Architectures](https://www.latent.space/p/2024-post-transformers) - Latent Space, NeurIPS 2024
- [Diffusion for Robotics Survey](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247/full) - Frontiers 2025
- [Test-Time Compute Survey](https://arxiv.org/abs/2501.02497) - 2025
