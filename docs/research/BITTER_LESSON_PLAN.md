# The Bitter Lesson: Applied to ExPhil

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."
> — Rich Sutton, 2019

This document outlines concrete changes to ExPhil based on the Bitter Lesson's implications.

---

## What We're Currently Doing Wrong (By Bitter Lesson Standards)

### 1. Hand-Crafted Embeddings
**Previous:** 1204-dimensional vector with specific fields (one-hot actions, characters)

**Updated (2026-02):** Now using ~287-dimensional learned embeddings by default:
- Action IDs → 64-dim trainable embedding (saves ~670 dims)
- Character IDs → 64-dim trainable embedding (saves ~64 dims)
- Compact stage mode (7 dims vs 64)

**Remaining concern:** Still hand-picking which features to include (positions, velocities, etc.). The model could potentially learn relevant features from raw game memory.

### 2. Fixed Stick Discretization
**Current:** 9 buckets per axis = 81 positions (uniform grid)

**Problem:** We picked 9 arbitrarily. Eric Gu found K-means with 21 clusters worked better than finer discretization. The optimal binning is data-dependent.

### 3. Character-Specific Reward Shaping
**Current plan:** Mewtwo gets recovery bonus, Ganon gets spacing bonus, etc.

**Problem:** We're encoding character knowledge. Pure RL with win/loss would discover what matters for each character given enough compute.

### 4. Behavioral Cloning as Shortcut
**Current:** BC from human replays → RL refinement

**Problem:** Human knowledge as prior may be a ceiling, not a floor (AlphaGo Zero beat AlphaGo 100-0).

---

## What to Change (Prioritized)

### Phase 1: Low-Hanging Fruit — COMPLETE

#### 1A. Learned Stick Discretization — DONE
**Previous:** Uniform 9-bucket grid
**Implemented:** K-means clustering on replay action data

```bash
# Train K-means centers
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx

# Use in training
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx
```

**Result (2026-01-30):** K-means didn't significantly improve accuracy because stick errors are timing-based (when to move), not resolution-based (where to move). See GOALS.md for full analysis.

#### 1B. Larger Models (Scale Up)
**Current:** 512×512 MLP, 256-dim Mamba
**Change to:** Experiment with 1024, 2048 hidden sizes

```bash
# Already supported:
mix run scripts/train_from_replays.exs --hidden 1024,1024 --temporal --backbone mamba
```

**Why this helps:** Bigger models can learn more complex representations. Our fast Mamba inference gives headroom.

#### 1C. More Data
**Current:** Training on limited replay sets
**Change to:** Use ALL available Slippi data (millions of games available)

```bash
# Download from Slippi Discord:
# - Ranked collections (anonymized)
# - Tournament sets
# Focus on volume over curation
```

**Why this helps:** Scale laws suggest more data > better features.

### Phase 2: Reduce Hand-Engineering (Next Month)

#### 2A. Simplified Embeddings Experiment — PARTIALLY COMPLETE

**Previous:** Carefully crafted 1204-dim vector (one-hot everything)
**Now default:** ~287-dim learned embeddings

| Variant | Description | Dims | Status |
|---------|-------------|------|--------|
| Full (one-hot) | Original embedding | 1204 | Available (`--action-mode one_hot`) |
| **Learned** | Embed action/char IDs in network | ~287 | **Default since Jan 2026** |
| Minimal | Just (x, y, action, percent, stocks) × 2 players | ~50 | Not implemented |
| Flat | All libmelee fields concatenated (no design) | ~200 | Not implemented |

**Result so far:** Learned embeddings allow 6x larger networks at same training speed.

**Remaining experiment:** Compare Minimal/Flat against Learned to test if we're still over-engineering.

#### 2B. Single Sparse Reward
**Current plan:** Character-specific shaped rewards
**Experiment:** Train with ONLY win/loss reward

```elixir
# lib/exphil/rewards/sparse.ex
defmodule ExPhil.Rewards.Sparse do
  def compute(%{winner: :p1}, :p1), do: 1.0
  def compute(%{winner: :p2}, :p1), do: -1.0
  def compute(_, _), do: 0.0
end
```

**Hypothesis:** With enough self-play iterations, sparse reward will discover optimal play without us specifying what "good Mewtwo play" looks like.

#### 2C. Remove Character-Specific Code Paths
**Current:** Considering separate models/rewards per character
**Change to:** Single model with character ID as input

**Why:** Let the model learn character-specific strategies from data.

### Phase 3: Toward Pure RL (Long Term)

#### 3A. Pure RL Baseline Experiment
**Goal:** Measure how much BC actually helps vs pure RL

```bash
# Train pure RL from scratch (mock environment first)
mix run scripts/train_self_play.exs \
  --game-type mock \
  --no-bc-init \
  --timesteps 10_000_000

# Compare sample efficiency:
# - BC+RL: How many samples to X win rate?
# - Pure RL: How many samples to X win rate?
```

**Expected result:** BC+RL is 10-100× more sample efficient on our compute. But this quantifies the tradeoff.

#### 3B. Curriculum Learning (No Human Data)
If pure RL is too slow, try curriculum without human data:

1. **Stage 1:** Learn to not fall off stage (trivial)
2. **Stage 2:** Learn to hit opponent (basic)
3. **Stage 3:** Learn to not get hit (defense)
4. **Stage 4:** Full self-play (emergent strategy)

**Why:** AlphaStar used curriculum. This is self-supervised complexity scaling.

#### 3C. Scale Compute
The honest answer: if Bitter Lesson fully applies, we need more compute.

Options:
- **RunPod/Lambda:** Rent GPUs for extended training
- **Distributed training:** BEAM's distribution for multi-node
- **Efficient architectures:** Mamba already helps here

---

## Experiments & Metrics

### Experiment 1: Stick Discretization
| Metric | Uniform 9×9 | K-means 21 | K-means 32 |
|--------|-------------|------------|------------|
| BC loss | ? | ? | ? |
| Action accuracy | ? | ? | ? |
| In-game feel | ? | ? | ? |

### Experiment 2: Model Scaling
| Metric | 256×256 | 512×512 | 1024×1024 |
|--------|---------|---------|-----------|
| BC loss | ? | ? | ? |
| Inference ms | ~2ms | ~4ms | ~8ms |
| Memory MB | ? | ? | ? |

### Experiment 3: Embedding Ablation
| Metric | One-hot (1204) | Learned (287) | Minimal (50) | Flat (200) |
|--------|----------------|---------------|--------------|------------|
| BC loss | baseline | ✓ similar | ? | ? |
| Training time | baseline | ✓ 6x faster networks | faster | faster |
| Win rate | ? | ? | ? | ? |

**Note:** Learned embeddings (287 dims) are now the default. One-hot available via `--action-mode one_hot --character-mode one_hot`.

### Experiment 4: BC vs Pure RL
| Metric | BC+RL | Pure RL (10M samples) |
|--------|-------|----------------------|
| Win rate vs CPU L9 | ? | ? |
| Training time | ? | ? |
| Strategy diversity | ? | ? |

---

## What NOT to Change

Some things are practical necessities, not "knowledge encoding":

1. **Autoregressive action heads** — This is coordination structure, not domain knowledge
2. **Temporal modeling** — Melee is sequential; models need history
3. **60 FPS inference constraint** — Physical requirement, not design choice
4. **Population-based training** — General RL best practice, not Melee-specific

---

## Implementation Order

### Week 1-2: Immediate
- [ ] Run K-means stick discretization experiment
- [ ] Train with 2× larger models (1024 hidden)
- [ ] Download more replay data (target: 100k+ games)

### Week 3-4: Embeddings
- [ ] Implement Minimal embedding variant
- [ ] A/B test Full vs Minimal on same data
- [ ] Document findings

### Month 2: Rewards
- [ ] Implement sparse (win/loss only) reward
- [ ] Compare shaped vs sparse in self-play
- [ ] Measure strategy diversity

### Month 3: Pure RL Exploration
- [ ] Train pure RL baseline (mock env)
- [ ] Measure sample efficiency gap
- [ ] Document compute requirements for parity

---

## Key Insight

The Bitter Lesson doesn't mean "throw away everything." It means:

1. **Bet on scale** — Bigger models, more data, more compute
2. **Minimize brittle assumptions** — Don't hardcode what can be learned
3. **Measure the tradeoffs** — Know what your shortcuts cost
4. **Plan for obsolescence** — Your clever engineering will be beaten by scale eventually

For ExPhil specifically: BC+RL is the right choice **given our compute budget**. But we should minimize additional hand-engineering and run experiments to understand where our knowledge helps vs hurts.

---

## References

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Sutton, 2019
- [AlphaGo Zero](https://www.nature.com/articles/nature24270) — Beat human-data-trained version 100-0
- [OpenAI Five](https://arxiv.org/abs/1912.06680) — Pure RL at 770 PFlops·days
- [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020
