# Architecture Recommendations for ExPhil

Analysis of Eric Gu's Transformer approach vs ExPhil's current architectures, with recommendations for training improvements.

## Current State

### ExPhil Backbones (7 options)

| Backbone | Val Loss | Inference | 60 FPS | Notes |
|----------|----------|-----------|--------|-------|
| **Attention** | **3.68** | 15ms | Borderline | Best accuracy |
| Jamba | 3.87 | 12ms | Yes | Hybrid Mamba+Attention |
| GRU | 4.48 | 150ms | No | Reliable baseline |
| LSTM | 4.75 | 220ms | No | Legacy |
| Mamba | 8.22 | **8.9ms** | **Yes** | Best for production |
| MLP | - | 1-2ms | Yes | Single-frame baseline |

### Eric Gu's Approach

| Aspect | Value |
|--------|-------|
| Architecture | Decoder-only Transformer (GPT-style) |
| Parameters | ~20M |
| Context | ~300 frames (5 seconds) |
| Input format | Interleaved `[s₀, a₀, s₁, a₁, ...]` |
| Training | Next-token prediction (teacher forcing) |
| Data | All characters, 3B frames |
| Cost | $5, 5 hours on 2x RTX 3090 |
| Result | 95% vs L9 CPU |

## Key Insight

**Eric Gu's main contribution isn't the architecture - it's the training strategy.**

His experiments showed:
> "A single Transformer trained on all character replays would out-perform and be more efficient to train. General concepts like spacing and positioning are fundamentally the same across all matchups."

ExPhil's attention backbone already achieves 3.68 val loss (best). The gap is in:
1. Multi-character training
2. Longer context windows
3. Interleaved sequence format

## Recommendations

### Priority 1: Multi-Character Unified Training

**Why**: Eric Gu's key finding - all-character > single-character

**Current Support**: ExPhil already supports `--character mewtwo,ganondorf,link`

**Missing**: Character-balanced sampling to handle data imbalance

```bash
# Train unified low-tier model
mix run scripts/train_from_replays.exs \
  --preset production \
  --character mewtwo,ganondorf,link,gameandwatch,zelda \
  --temporal --backbone attention
```

**Implementation**:
1. Add `--balance-characters` flag to weight sampling
2. Track per-character loss in metrics
3. Add `--preset low_tier_unified` combining all 5 characters

---

### Priority 2: Longer Context Window

**Why**: Eric Gu uses ~300 frames (5 sec), ExPhil defaults to 60 (1 sec)

**Hypothesis**: Longer context improves recovery decisions, combo extensions, neutral spacing

**Experiment**:
```bash
# Test 180-frame context (3 seconds)
mix run scripts/train_from_replays.exs \
  --preset production \
  --window-size 180 \
  --temporal --backbone attention
```

**Trade-offs**:
- Longer context = O(K²) attention cost
- But 180 frames is still only ~10K attention weights
- Mamba would be O(n), but currently has higher val loss

**Recommendation**: Run ablation study at 60, 120, 180, 300 frames.

---

### Priority 3: Learned Character Embedding

**Why**: Better transfer learning across characters

**Current**: 33-dim one-hot character embedding

**Proposed**: 64-dim trainable embedding (like `action_mode: :learned`)

**Implementation**:
```elixir
# In player.ex
@type character_mode :: :one_hot | :learned

# In policy.ex - add character embedding layer
character_embed = Axon.embedding(input_char_ids, 33, 64, name: "character_embed")
```

**Benefits**:
- Model learns character similarity (e.g., floaties vs fastfallers)
- Enables character-conditioned outputs
- Smaller input dimension (64 vs 33)

---

### Priority 4: Interleaved State-Action Sequences

**Why**: Better gradient signal through teacher forcing

**Current**: `[s₀, s₁, s₂, ...]` → predict `a_final`

**Eric Gu**: `[s₀, a₀, s₁, a₁, ...]` → predict at every position

**Trade-offs**:
| Aspect | Current | Interleaved |
|--------|---------|-------------|
| Sequence length | N | 2N |
| Predictions per sequence | 1 | N |
| Gradient signal | Sparse | Dense |
| Memory | Lower | Higher |
| Inference | Direct | Need last position |

**Implementation complexity**: Medium - requires data pipeline changes

**Recommendation**: Lower priority since current approach works well.

---

## Architecture Decision Tree

```
Q: Need real-time (60 FPS)?
├─ YES:
│   ├─ Best accuracy? → Jamba (3.87 val loss, 12ms)
│   └─ Fastest? → Mamba (8.22 val loss, 8.9ms)
└─ NO (research/training):
    ├─ Best accuracy? → Attention (3.68 val loss)
    └─ Baseline? → GRU

Q: Implement GPT-style decoder?
└─ NO - Attention backbone already achieves best val loss
   Focus on training strategy instead:
   1. Multi-character training
   2. Longer context
   3. Character-balanced sampling
```

## Experiments to Run

### Experiment 1: Multi-Character vs Single-Character

**Hypothesis**: Unified model outperforms character-specific models

```bash
# Single character (baseline)
mix run scripts/train_from_replays.exs --preset production --character mewtwo

# Multi-character
mix run scripts/train_from_replays.exs --preset production \
  --character mewtwo,ganondorf,link,gameandwatch,zelda
```

**Metrics**:
- Per-character validation loss
- Cross-evaluation (train multi, eval single)

---

### Experiment 2: Context Length Ablation

**Hypothesis**: Longer context helps recovery and neutral

```bash
for window in 60 120 180 300; do
  mix run scripts/train_from_replays.exs \
    --preset production \
    --window-size $window \
    --checkpoint checkpoints/window_${window}.axon
done
```

**Metrics**:
- Validation loss at each window size
- Inference time vs accuracy trade-off

---

### Experiment 3: Attention vs Mamba Trade-off

**Hypothesis**: Mamba needs more epochs to match attention

```bash
# Attention (current best)
mix run scripts/train_from_replays.exs --backbone attention --epochs 50

# Mamba (same compute budget)
mix run scripts/train_from_replays.exs --backbone mamba --epochs 100
```

**Metrics**:
- Final validation loss
- Training time to convergence

---

## Implementation Roadmap

### Phase 1: Training Strategy (No Code Changes)

1. Run multi-character experiment with existing flags
2. Run context length ablation
3. Document results

### Phase 2: Character Balancing

1. Add `--balance-characters` flag
2. Implement weighted sampling based on character frequency
3. Track per-character metrics

### Phase 3: Learned Embeddings

1. Add `character_mode: :learned` option
2. Add character embedding layer to policy
3. Run comparison experiments

### Phase 4: Interleaved Sequences (Optional)

1. Modify data pipeline for `[s, a, s, a, ...]` format
2. Add teacher forcing training mode
3. Compare dense vs sparse gradient signal

---

## References

- [Eric Gu's Transformer Blog](https://ericyuegu.com/melee-pt1)
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - TransformerLike architecture
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Scale > inductive bias
- [ExPhil Architecture Docs](../architectures/ARCHITECTURES.md)
