# New Research Deep Dive On Architectures

**Date:** 2026-01-31
**Purpose:** Exhaustive analysis of neural network architectures for real-time game AI, comparing ExPhil's implementations against cutting-edge research (2024-2026).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current ExPhil Architecture Inventory](#current-exphil-architecture-inventory)
3. [Research Landscape (2024-2026)](#research-landscape-2024-2026)
4. [Gap Analysis](#gap-analysis)
5. [Missing Architectures - Implementation Plans](#missing-architectures---implementation-plans)
6. [Optimization Opportunities for Existing Architectures](#optimization-opportunities-for-existing-architectures)
7. [Priority Recommendations](#priority-recommendations)
8. [Citations and References](#citations-and-references)

---

## Executive Summary

ExPhil implements **7 backbone architectures** (MLP, LSTM, GRU, Attention, Mamba, Jamba, GatedSSM) with strong coverage of 2023-era approaches. However, **significant gaps exist** in the 2024-2026 research landscape:

### What ExPhil Has
| Architecture | Status | Inference | Production Ready |
|--------------|--------|-----------|------------------|
| MLP | Complete | 9.4ms | Yes |
| LSTM | Complete | 228ms | No (too slow) |
| GRU | Complete | ~150ms | No (too slow) |
| Attention (Sliding Window) | Complete | 17ms | Borderline |
| Mamba (S6) | Complete | 8.9ms | Yes |
| Jamba (Mamba + Attention) | Complete | ~12ms | Yes |
| GatedSSM | Complete | - | Research only |

### What's Missing
| Architecture | Priority | Estimated Effort | Impact |
|--------------|----------|------------------|--------|
| **RWKV-7** | High | Medium | O(1) space, shipped to 1.5B devices |
| **Zamba (Single Shared Attention)** | High | Low | 10x KV cache reduction |
| **Mamba-2 SSD** | High | Medium | 2-8x training speedup |
| **GLA/HGRN-2** | Medium | Medium | Faster than FlashAttention on short seqs |
| **Liquid Neural Networks** | Medium | High | Continuous adaptation |
| **Decision Transformer** | Medium | Medium | Return-conditioned policy |
| **S5 (Simplified State Space)** | Low | Medium | Simpler than Mamba |
| **DreamerV3** | Low | Very High | World model (different paradigm) |

### Key Insight
ExPhil's architecture diversity is **good for 2023-era approaches** but lacks:
1. **Linear attention variants** (RWKV, GLA, RetNet)
2. **Minimal attention hybrids** (Zamba's single-shared-attention pattern)
3. **Continuous-time/adaptive models** (Liquid Neural Networks)
4. **Return-conditioned RL** (Decision Transformer)

---

## Current ExPhil Architecture Inventory

### 1. MLP (Multi-Layer Perceptron)

**File:** `lib/exphil/networks/policy.ex:552-582`

**Characteristics:**
- Single-frame feedforward (no temporal context)
- Fastest inference: 9.4ms
- Configurable hidden layers, activation, dropout
- Residual connections optional (`--residual`)

**Strengths:**
- Extremely fast inference
- Simple to train and debug
- Good baseline

**Limitations:**
- No temporal context (can't learn combos, reactions)
- Performance ceiling lower than temporal models

### 2. LSTM (Long Short-Term Memory)

**File:** `lib/exphil/networks/recurrent.ex:183-211`

**Characteristics:**
- Best validation loss (2.95) among all architectures
- Sequential processing with 4 gates per cell
- cuDNN-optimized kernels

**Strengths:**
- Best accuracy in benchmarks
- Well-understood, stable training
- Strong long-range dependency modeling

**Limitations:**
- **Fatal for real-time:** 228ms inference (10x over budget)
- O(L) sequential computation (not parallelizable)
- High memory usage (~500MB VRAM, 13GB RAM for sequences)

### 3. GRU (Gated Recurrent Unit)

**File:** `lib/exphil/networks/recurrent.ex:502-519`

**Characteristics:**
- Simplified LSTM with 3 gates (vs 4)
- Faster training than LSTM (~8.3 b/s vs 7.7)
- Similar accuracy to LSTM

**Strengths:**
- Fewer parameters than LSTM
- Slightly faster training

**Limitations:**
- Same fundamental problem as LSTM: sequential, slow inference
- Not real-time capable

### 4. Attention (Sliding Window Transformer)

**File:** `lib/exphil/networks/attention.ex:333-409`

**Characteristics:**
- Sliding window self-attention (default window=60 frames)
- Multi-head attention (default 4 heads, 64 dim per head)
- Pre-LayerNorm for stability
- Optional QK LayerNorm

**Attention Variants Implemented:**
1. **Standard scaled dot-product** - O(L²)
2. **Chunked attention** - O(L × chunk) memory
3. **Memory-efficient (online softmax)** - True O(L) memory

**Strengths:**
- Parallelizable training (all positions at once)
- Good accuracy (3.065 val loss)
- Explicit attention weights (interpretable)

**Limitations:**
- O(L²) time complexity
- 17ms inference (borderline for 60 FPS)
- Memory grows quadratically with sequence length

### 5. Mamba (Selective State Space Model - S6)

**File:** `lib/exphil/networks/mamba.ex` (~500 lines)

**Characteristics:**
- Selective state space with input-dependent A, B, C matrices
- Parallel associative scan for O(L) time
- Causal 1D convolution + SiLU gating
- Pure Nx/XLA implementation

**Strengths:**
- **Production ready:** 8.9ms inference (best temporal model)
- Linear complexity O(L)
- Near-best accuracy (3.0 val loss)
- State caching for inference

**Limitations:**
- Slower convergence than attention (needs 10-20+ epochs)
- Hyperparameter sensitive
- No hardware-specific optimizations (vs cuDNN LSTM)

### 6. Jamba (Hybrid Mamba + Attention)

**File:** `lib/exphil/networks/hybrid.ex` (~300 lines)

**Characteristics:**
- Interleaved Mamba blocks with attention layers
- Configurable ratio via `attention_every` (default: 3 = 2:1 Mamba:Attn)
- Pre-LayerNorm + QK LayerNorm for stability

**Strengths:**
- Combines O(L) Mamba efficiency with O(K²) attention expressiveness
- Good balance of speed and accuracy
- ~12ms inference (60 FPS ready)

**Limitations:**
- More complex than pure approaches
- Tuning `attention_every` ratio is trial-and-error
- Higher parameter count than pure Mamba

### 7. GatedSSM (Simplified Mamba)

**File:** `lib/exphil/networks/gated_ssm.ex` (~400 lines)

**Characteristics:**
- Simplified SSM approximation using gating instead of parallel scan
- Mean pooling + projection instead of learned convolution
- Numerically stable (no NaN issues)

**Strengths:**
- Simpler than true Mamba
- Achieved 2.99 val loss (competitive)
- Good stability

**Limitations:**
- Not true Mamba architecture
- Approximation may lose expressiveness
- Research-only (not benchmarked for production)

### Experimental Mamba Variants

| Variant | File | Description | Status |
|---------|------|-------------|--------|
| Mamba NIF | `mamba_nif.ex` | Rust FFI for CUDA-accelerated scan | Experimental |
| Mamba Cumsum | `mamba_cumsum.ex` | Cumsum-based scan for training | Experimental |
| Mamba Hillis-Steele | `mamba_hillis_steele.ex` | Alternative parallel scan | Experimental |
| Mamba SSD | `mamba_ssd.ex` | Chunked selective scan decomposition | Experimental |

---

## Research Landscape (2024-2026)

### State-Space Models (SSMs)

#### Mamba-2 with SSD Algorithm (2024)

**Paper:** [State Space Duality](https://tridao.me/blog/2024/mamba2-part3-algorithm/)

**Key Innovation:** The State Space Duality (SSD) algorithm fundamentally reimagines Mamba's computation by connecting SSMs to structured matrix multiplication.

**Benefits:**
- Steps 1, 2, 4 are pure matmuls that run on tensor cores
- Only step 3 requires sequential processing on chunks
- 100x reduction in sequential operations
- Exploits H100's 15x gap between matmul and arithmetic TFLOPS

**Implementation Complexity:** 25 lines of code vs hundreds for Mamba-1 selective scan

**Relevance to ExPhil:**
- **Training speedup:** 2-8x faster training via better hardware utilization
- **Inference caveat:** For short sequences (<2K tokens), Mamba-1 may be faster due to mature optimization

#### Mamba-3 (2025)

**Paper:** [OpenReview](https://openreview.net/forum?id=HwCvaJOiCj)

**Key Innovation:** "Inference-first perspective" with more expressive recurrence formulations.

**Benefits:**
- Multi-input, multi-output for better hardware parallelism during decoding
- Constant memory regardless of sequence length
- Better for production deployment

**Relevance to ExPhil:** Future upgrade path for Mamba backbone.

#### RWKV-7 "Goose" (March 2025)

**Paper:** [RWKV Architecture Wiki](https://wiki.rwkv.com/basic/architecture.html)

**Key Innovation:** Generalized Delta Rule that surpasses TC0 constraint, comprehensively outperforming Transformers.

**Version History:**
- RWKV-6 "Finch" (2024): Dynamic recurrence based on LoRA
- RWKV-7 "Goose" (2025): Surpasses TC0, beats Transformers
- RWKV-8 "Heron" (2025): DeepEmbed for sparse large models on edge

**Real-World Deployment:** RWKV v5 shipped to 1.5 billion Windows machines for on-device Copilot.

**Benefits:**
- O(T) time complexity, O(1) space complexity
- Proven at massive scale
- Efficient edge deployment

**Relevance to ExPhil:**
- **High priority:** Proven production architecture with edge deployment
- O(1) space is ideal for real-time game AI
- Could replace or augment Mamba

#### S5 (Simplified State Space)

**Paper:** [arXiv:2208.04933](https://arxiv.org/abs/2208.04933) (ICLR 2023 Oral, top 5%)

**Key Innovation:** Uses one multi-input, multi-output SSM instead of many independent SISOs.

**Performance:** 87.4% on Long Range Arena, 98.5% on Path-X.

**Relevance to ExPhil:**
- Simpler than Mamba
- Good for research/ablation studies
- May not outperform Mamba on game AI tasks

### Linear Attention Variants

#### GLA (Gated Linear Attention)

**Paper:** [ACM DL](https://dl.acm.org/doi/10.5555/3692070.3694403)

**Key Innovation:** Data-dependent gates with linear complexity.

**Performance:** Faster than FlashAttention-2 even on short sequences (1K tokens).

**Relevance to ExPhil:**
- Could provide attention-like expressiveness with O(L) complexity
- Good for short game sequences (60-120 frames)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) provides efficient implementations

#### HGRN-2 (Hierarchically Gated Linear RNN)

**Paper:** [arXiv:2404.07904](https://arxiv.org/html/2404.07904v1)

**Key Innovation:** State expansion with hierarchical gating.

**Relevance to ExPhil:**
- Another O(L) alternative to attention
- Worth benchmarking against Mamba

#### RetNet

**Characteristics:** Linear attention with exponential decay.

**Limitation:** Fixed decay limits long-range recall.

**Relevance to ExPhil:** Lower priority due to fixed decay limitation.

### Hybrid Architectures

#### Zamba (May 2024)

**Paper:** [arXiv:2405.16712](https://arxiv.org/abs/2405.16712)

**Key Innovation:** "One attention layer is all you need." Uses Mamba backbone with single shared attention module.

**Architecture:**
```
6 Mamba layers → residual → 1 shared full attention + MLP
```

**Benefits:**
- Significantly faster inference than Jamba
- 10x KV cache reduction
- Substantially less memory for long sequences

**Versions:**
- Zamba (7B): Mamba-1 based
- Zamba2 (2.7B, 1.2B): Mamba-2 based with optimizations

**Relevance to ExPhil:**
- **High priority:** Simpler than Jamba, potentially faster
- Easy to implement (just add one shared attention layer)
- Good experiment: compare Jamba (interleaved) vs Zamba (single shared)

#### Jamba (AI21 Labs, March 2024)

**Paper:** [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)

**Already Implemented in ExPhil:** Yes, in `lib/exphil/networks/hybrid.ex`.

**Research Finding:** Mamba-1 + Attention works better than Mamba-2 + Attention in hybrid architectures.

**ExPhil Status:** Current implementation uses Mamba-1 style, which is correct per research.

### Novel Approaches

#### Liquid Neural Networks

**Research:** [Liquid AI](https://www.liquid.ai/research/liquid-neural-networks-research)

**Key Innovation:** Use continuous-time dynamics and differential equations for real-time adaptability.

**Characteristics:**
- Parameters change over time based on nested differential equations
- Robust to noisy environments and distributional drift
- Inspired by C. elegans (302 neurons generating complex dynamics)

**Recent Developments:**
- LFM2 (July 2025): 2x decode/prefill performance on CPUs
- $250M funding from AMD (December 2024)

**Relevance to ExPhil:**
- **High potential:** Adaptive during inference (not just training)
- Could handle opponent adaptation without retraining
- Complex to implement in Elixir/Nx

#### Decision Transformer

**Paper:** [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)

**Key Innovation:** Frame RL as sequence modeling by conditioning on desired returns.

**Input Format:**
```
(R₁, s₁, a₁, R₂, s₂, a₂, ..., Rₜ, sₜ) → aₜ
```

**Benefits:**
- No TD learning required (avoids deadly triad)
- No reward discounting (avoids short-sighted behaviors)
- Trajectory stitching from different replays

**Multi-Game DT:** Single model plays 46 Atari games at near-human performance.

**When to Use DT (2024 study):**
1. DT requires more data than CQL but is more robust
2. DT is substantially better in sparse-reward and low-quality data settings
3. DT and BC are preferable as task horizon increases

**Relevance to ExPhil:**
- **Medium priority:** Could replace BC+RL with pure sequence modeling
- Natural fit for replay data (already have trajectories with outcomes)
- Experiment: condition on stock lead or damage output

#### Test-Time Training (TTRL)

**Paper:** [arXiv:2504.16084](https://arxiv.org/abs/2504.16084) (NeurIPS 2025)

**Key Innovation:** Self-evolution of models using RL on unlabeled test data.

**Performance:** Boosts Qwen-2.5-Math-7B pass@1 by 211% on AIME 2024.

**Relevance to ExPhil:**
- Interesting for opponent adaptation
- Could fine-tune during play session
- Complex to implement in real-time

### World Models

#### DreamerV3 (Nature 2025)

**Paper:** [Nature](https://www.nature.com/articles/s41586-025-08744-2)

**Key Achievement:** First algorithm to collect diamonds in Minecraft from scratch without human data.

**Characteristics:**
- General algorithm that outperforms specialized methods across 150+ tasks
- Single configuration (no per-task tuning)
- Block GRU with RMSNorm

**Relevance to ExPhil:**
- **Different paradigm:** World model approach vs direct policy learning
- High implementation effort
- Could be used for synthetic data generation

#### GameNGen (Google, 2024)

**Paper:** [OpenReview](https://openreview.net/forum?id=P8pqeEkn1H)

**Key Innovation:** First game engine powered entirely by a neural model, running DOOM at 20 FPS.

**Relevance to ExPhil:**
- Interesting for generating training scenarios
- Not directly applicable to policy learning

### Transformer Optimizations

#### FlashAttention-3 (July 2024)

**Paper:** [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) (NeurIPS 2024 Spotlight)

**Performance:**
- 75% theoretical max FLOP utilization on H100 (vs 35% for FlashAttention-2)
- 1.5-2x speedup with FP16, reaching 740 TFLOPs/s
- ~1.2 PFLOPs/s with FP8

**Relevance to ExPhil:**
- Could optimize Attention backbone
- H100-specific optimizations
- Lower priority (current attention is already borderline-viable)

---

## Gap Analysis

### Architecture Gaps

| Gap | Current in ExPhil | Research Alternative | Priority |
|-----|------------------|---------------------|----------|
| **Linear Attention** | None | RWKV-7, GLA, HGRN-2 | **High** |
| **Minimal Hybrid** | Jamba (interleaved) | Zamba (single shared) | **High** |
| **Training-Optimized SSM** | Mamba-1 | Mamba-2 SSD | **High** |
| **Return-Conditioned** | None | Decision Transformer | **Medium** |
| **Continuous Adaptation** | None | Liquid Neural Networks | **Medium** |
| **Simplified SSM** | GatedSSM (approximate) | S5 (principled) | **Low** |
| **World Model** | None | DreamerV3 | **Low** |

### Optimization Gaps

| Gap | Current in ExPhil | Research Technique | Priority |
|-----|------------------|-------------------|----------|
| **Mamba Training Speed** | Sequential scan | SSD algorithm | **High** |
| **Attention Memory** | Chunked/memory-efficient | FlashAttention-3 | **Medium** |
| **INT4 Quantization** | ONNX INT8 | AWQ/GPTQ INT4 | **Medium** |
| **FP8 Training** | FP32 (BF16 issues) | Native FP8 | **Low** (H100 only) |
| **KV Cache Compression** | N/A | MiniKV, ALISA | **Low** |

### Why These Gaps Exist

1. **Recency:** RWKV-7, Mamba-2, Zamba are 2024-2025 papers
2. **Ecosystem:** Nx/Axon lacks flash-linear-attention, FlashAttention-3
3. **Paradigm:** Decision Transformer is different training paradigm (not BC+RL)
4. **Complexity:** Liquid Neural Networks require ODE solvers

---

## Missing Architectures - Implementation Plans

### 1. RWKV-7 "Goose"

**Priority:** High
**Effort:** Medium (3-5 days)
**Expected Impact:** O(1) space complexity, proven at scale

**Implementation Plan:**

1. **Study Architecture:**
   - Read [RWKV-7 architecture docs](https://wiki.rwkv.com/basic/architecture.html)
   - Understand Generalized Delta Rule
   - Compare to Mamba's selective scan

2. **Core Implementation:**
   ```elixir
   # lib/exphil/networks/rwkv.ex
   defmodule ExPhil.Networks.RWKV do
     @moduledoc "RWKV-7 architecture for game AI"

     def build(opts) do
       # Time-mixing: wkv + time_first attention
       # Channel-mixing: R-gate * K-gate value
       # Generalized Delta Rule for state updates
     end
   end
   ```

3. **Integration Points:**
   - Add to `Policy.build_temporal/2`
   - Add `--backbone rwkv` CLI flag
   - Config: `num_layers`, `hidden_size`, `state_size`

4. **Testing:**
   - Unit tests for forward pass shapes
   - Benchmark against Mamba on same dataset
   - Verify O(1) memory during inference

**Advantages:**
- O(1) space complexity (vs Mamba's O(L) for states)
- Proven at massive scale (1.5B Windows devices)
- Active development community

**Pain Points:**
- Less documented than Mamba
- May need to port from PyTorch implementation
- Generalized Delta Rule is mathematically complex

### 2. Zamba (Single Shared Attention)

**Priority:** High
**Effort:** Low (1-2 days)
**Expected Impact:** Simpler than Jamba, potentially faster

**Implementation Plan:**

1. **Architecture Modification:**
   ```elixir
   # Modify lib/exphil/networks/hybrid.ex
   # Instead of interleaved attention, use single shared layer

   def build_zamba(opts) do
     # 6 Mamba layers (sequential)
     # 1 shared attention layer (called after every N Mamba layers)
     # Residual connection around attention
   end
   ```

2. **Key Differences from Jamba:**
   - Single attention layer vs multiple interleaved
   - Attention layer is "shared" (same weights reused)
   - Simpler architecture, fewer parameters

3. **Testing:**
   - Compare Zamba vs Jamba on same benchmark
   - Measure KV cache memory reduction
   - Profile inference latency

**Advantages:**
- 10x KV cache reduction
- Simpler than Jamba
- Easy to implement (modify existing hybrid.ex)

**Pain Points:**
- May lose some expressiveness vs interleaved
- Optimal layer count for shared attention unclear

### 3. Mamba-2 SSD Algorithm

**Priority:** High
**Effort:** Medium (2-3 days)
**Expected Impact:** 2-8x training speedup

**Implementation Plan:**

1. **Replace Selective Scan:**
   ```elixir
   # lib/exphil/networks/mamba_ssd.ex already exists (partial)
   # Complete the implementation

   def ssd_forward(x, A, B, C, delta) do
     # Step 1: Compute structured matrices (matmul on tensor cores)
     # Step 2: Chunk-wise processing
     # Step 3: Sequential step (minimal)
     # Step 4: Output combination (matmul)
   end
   ```

2. **Preserve Mamba-1 Inference:**
   - Keep current selective scan for inference (fast)
   - Use SSD for training (better hardware utilization)

3. **Benchmark:**
   - Compare training throughput: batches/sec
   - Verify identical outputs (SSD is mathematically equivalent)

**Advantages:**
- 2-8x training speedup
- Same accuracy (mathematically equivalent)
- Simpler code (25 lines vs hundreds)

**Pain Points:**
- Requires careful tensor layout for tensor core efficiency
- May need CUDA-specific tuning for H100

### 4. GLA (Gated Linear Attention)

**Priority:** Medium
**Effort:** Medium (3-4 days)
**Expected Impact:** O(L) with attention-like expressiveness

**Implementation Plan:**

1. **Core Implementation:**
   ```elixir
   # lib/exphil/networks/gla.ex
   defmodule ExPhil.Networks.GLA do
     @moduledoc "Gated Linear Attention for game AI"

     def gla_layer(input, opts) do
       # Q, K, V projections
       # Data-dependent gating
       # Linear attention with gates
     end
   end
   ```

2. **Reference:** [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

**Advantages:**
- Faster than FlashAttention-2 on short sequences
- O(L) complexity
- Good for ExPhil's 60-120 frame windows

**Pain Points:**
- Need to port from Python/CUDA
- Less battle-tested than Mamba

### 5. Decision Transformer

**Priority:** Medium
**Effort:** Medium (3-4 days)
**Expected Impact:** Alternative training paradigm

**Implementation Plan:**

1. **Modify Training Pipeline:**
   ```elixir
   # Add return-to-go embedding
   # Modify sequence format: (R, s, a, R, s, a, ...)
   # Train with next-action prediction (not BC loss)
   ```

2. **Return Conditioning Options:**
   - Stock lead (±1, ±2, ±3)
   - Damage dealt
   - Game outcome (win/loss)

3. **Evaluation:**
   - Can model learn to output actions for different desired outcomes?
   - Compare to BC on same data

**Advantages:**
- No TD learning (avoids deadly triad)
- Can condition on desired outcomes
- Works well with sparse rewards

**Pain Points:**
- Need to restructure training data
- Different training loop than BC
- May require more data than BC

### 6. Liquid Neural Networks

**Priority:** Medium
**Effort:** High (1-2 weeks)
**Expected Impact:** Continuous adaptation during play

**Implementation Plan:**

1. **ODE Integration:**
   ```elixir
   # lib/exphil/networks/liquid.ex
   # Requires numerical ODE solver in Elixir

   def liquid_cell(input, state, params) do
       # Solve: dx/dt = f(x, input; params)
       # Where params themselves evolve
     end
   end
   ```

2. **Challenges:**
   - ODE solver needed (no native Nx support)
   - Continuous-time dynamics in discrete-frame game
   - Training stability

**Advantages:**
- Adapts during inference
- Could handle opponent changes
- Robust to distributional drift

**Pain Points:**
- Highest implementation effort
- ODE solvers are expensive
- May not be necessary for Melee AI

### 7. S5 (Simplified State Space)

**Priority:** Low
**Effort:** Medium (2-3 days)
**Expected Impact:** Research comparison

**Implementation Plan:**

1. **MIMO SSM:**
   ```elixir
   # Single multi-input, multi-output SSM
   # Simpler than Mamba's parallel SISOs
   ```

2. **Use Case:** Ablation study to understand what Mamba's complexity buys us.

**Advantages:**
- Simpler than Mamba
- Good for understanding SSM fundamentals

**Pain Points:**
- May not outperform Mamba
- Lower priority

---

## Optimization Opportunities for Existing Architectures

### Mamba Optimizations

#### 1. SSD Algorithm for Training
**Status:** Partial implementation exists
**Effort:** Medium
**Impact:** 2-8x training speedup

Current Mamba uses sequential-ish parallel scan. SSD reformulates as:
```
Output = matmul × chunked_sequential × matmul
```

This moves 99% of computation to tensor cores.

#### 2. State Caching for Inference
**Status:** Not implemented
**Effort:** Low
**Impact:** Zero-latency "context loading"

Pre-compute Mamba states from training data, load at inference:
```elixir
# Pre-compute states for each character/matchup
cached_states = MambaStateCache.compute(policy, typical_game_states)

# At inference, start from cached state instead of zero
initial_state = MambaStateCache.get(character: :mewtwo, matchup: :fox)
```

#### 3. CUDA NIF Optimizations
**Status:** Experimental (`mamba_nif.ex`)
**Effort:** High
**Impact:** 5x inference speedup

Use Rust FFI to call optimized CUDA kernels.

### Attention Optimizations

#### 1. Flash Attention Integration
**Status:** Not implemented
**Effort:** High (external dependency)
**Impact:** 30-50% memory reduction, longer sequences

Would require:
- Porting FlashAttention to Elixir/Nx or
- CUDA NIF wrapper

#### 2. Window Size Reduction
**Status:** Configurable
**Effort:** None (already implemented)
**Impact:** Faster inference at cost of context

Current default: 60 frames. For 60 FPS:
- Try window_size=45 or 30
- Trade temporal context for speed

#### 3. Speculative Decoding
**Status:** Not applicable
**Effort:** N/A
**Impact:** N/A

Speculative decoding helps autoregressive text generation. ExPhil's policy outputs are fixed-size (not variable-length tokens), so this doesn't apply.

### Quantization Optimizations

#### 1. INT4 Quantization (AWQ/GPTQ)
**Status:** Not implemented
**Effort:** Medium
**Impact:** 2.7x faster inference, 50% memory reduction

Current: ONNX INT8 (0.55ms)
Target: INT4 with <2% accuracy loss

#### 2. FP8 Training
**Status:** Not available (XLA issues)
**Effort:** External dependency
**Impact:** 2x training speedup on H100

Blocked on EXLA/XLA FP8 support. Low priority until infra improves.

---

## Priority Recommendations

### Immediate (This Week)

1. **Implement Zamba architecture**
   - Easy: Modify existing Jamba to use single shared attention
   - High impact: Simpler, potentially faster
   - Low risk: Just a configuration of existing code

2. **Complete Mamba-2 SSD training path**
   - Build on existing `mamba_ssd.ex`
   - Keep Mamba-1 inference (already fast)
   - 2-8x training speedup

### Short-Term (This Month)

3. **Implement RWKV-7**
   - O(1) space complexity
   - Proven at massive scale
   - Could replace Mamba as default

4. **Add Decision Transformer option**
   - Different training paradigm worth exploring
   - Natural fit for replay data
   - Could outperform BC on sparse reward tasks

### Medium-Term (This Quarter)

5. **Implement GLA/HGRN-2**
   - O(L) with attention expressiveness
   - Good for short game sequences
   - Benchmark against Mamba

6. **Explore Liquid Neural Networks**
   - Continuous adaptation is unique
   - Could handle opponent changes
   - High effort but high potential

### Low Priority (Backlog)

7. **S5 implementation** - Research comparison only
8. **DreamerV3** - Different paradigm, very high effort
9. **FlashAttention-3** - Current attention is viable, lower priority

---

## Citations and References

### State-Space Models
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" - [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- [Mamba-2 State Space Duality](https://tridao.me/blog/2024/mamba2-part1-model/)
- Smith et al. (2023). "Simplified State Space Layers (S5)" - [arXiv:2208.04933](https://arxiv.org/abs/2208.04933)
- Peng et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era" - [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
- [RWKV-7 Architecture Wiki](https://wiki.rwkv.com/basic/architecture.html)

### Linear Attention
- [Gated Linear Attention](https://dl.acm.org/doi/10.5555/3692070.3694403)
- [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/html/2404.07904v1)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

### Hybrid Architectures
- [Jamba: AI21 Labs](https://arxiv.org/abs/2403.19887)
- [Zamba: Single Shared Attention](https://arxiv.org/abs/2405.16712)
- [Mamba-Transformer Hybrids Overview](https://n1o.github.io/posts/ssm-transformer-hybrids-guide/)

### Novel Approaches
- [Liquid Neural Networks](https://www.liquid.ai/research/liquid-neural-networks-research)
- Chen et al. (2021). "Decision Transformer" - [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- [TTRL: Test-Time RL](https://arxiv.org/abs/2504.16084)

### Transformer Optimizations
- Dao et al. (2024). "FlashAttention-3" - [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- [vLLM PagedAttention](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention)

### World Models
- Hafner et al. (2025). "DreamerV3" - [Nature](https://www.nature.com/articles/s41586-025-08744-2)
- Micheli et al. (2023). "IRIS" - [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)
- [GameNGen](https://openreview.net/forum?id=P8pqeEkn1H)

### Quantization
- [AWQ: MLSys 2024 Best Paper](https://github.com/mit-han-lab/llm-awq)
- [NVIDIA Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)

### Fighting Game AI
- [Pro-Level Fighting Game AI (IEEE 2021)](https://ieeexplore.ieee.org/document/9314886/)
- [Deep RL with Self-Play and MCTS](https://ieee-cog.org/2020/papers/paper_207.pdf)

---

**Last Updated:** 2026-01-31
**Author:** Claude (Architecture Research Agent)
