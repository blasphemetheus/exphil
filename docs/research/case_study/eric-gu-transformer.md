# Eric Gu's Transformer Melee AI Case Study

**Author**: Eric Gu
**Blog**: [ericyuegu.com/melee-pt1](https://ericyuegu.com/melee-pt1)
**Status**: Research project (2024)
**Approach**: Decoder-only Transformer with next-token prediction

## Overview

Eric Gu trained a 20 million parameter Transformer on 3 billion frames of professional Fox player replays. The model achieves a 95% win rate against the in-game level 9 CPU and cost approximately $5 to train on two RTX 3090s in 5 hours.

**Key Achievement**: Demonstrated that GPT-style next-token prediction works for fighting game AI with minimal compute.

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────┐
│       Decoder-Only Transformer          │
├─────────────────────────────────────────┤
│ Input: [s_0, a_0, s_1, a_1, ..., s_t]  │
│                                         │
│ Causal Attention (no future peeking)    │
│                                         │
│ Output: P(a_t | s_0:t, a_0:t-1)        │
├─────────────────────────────────────────┤
│ Parameters: ~20 million                 │
│ Context length: ~300 frames (~5 sec)    │
└─────────────────────────────────────────┘
```

### Training Objective

Pure next-token prediction, identical to GPT:

```python
# For each frame position, predict next action
loss = cross_entropy(
    predicted_action_logits,
    actual_next_action
)
```

**Why this works**: Melee actions are sequential decisions conditioned on game state history. The Transformer learns implicit state representations through attention over the sequence.

### Key Design Decisions

1. **All-character training**: Single model for all characters
2. **Short context**: ~5 seconds of history (300 frames)
3. **State + action interleaving**: `[s, a, s, a, ...]` format
4. **No RL refinement**: Pure imitation learning

## Data Pipeline

### Data Source

Professional Fox player replays from Slippi ranked collections:
- ~3 billion frames total
- Focused on high-skill play
- Anonymized player data

### Preprocessing

```
Raw .slp files
    ↓ (peppi parser)
Frame-by-frame state-action pairs
    ↓ (tokenization)
Sequence windows for training
```

### Sequence Construction

```python
# Window of ~300 frames (5 seconds)
# Interleaved state-action format
sequence = [
    state_embed(frame_0), action_embed(action_0),
    state_embed(frame_1), action_embed(action_1),
    ...,
    state_embed(frame_t)  # Predict action_t
]
```

## Key Insights

### 1. All-Character > Single-Character

Eric's experiments showed that a model trained on all character replays outperformed character-specific models:

> "I strongly believe a single Transformer trained on all character replays would out-perform and be more efficient to train. General concepts like spacing and positioning are fundamentally the same across all matchups."

**Implication for ExPhil**: Consider multi-character training even when targeting specific characters.

### 2. The Bitter Lesson Applies

Eric explicitly references Sutton's Bitter Lesson:

> "The bitter lesson certainly applies to Melee."

More data + bigger models + general architecture > specialized inductive biases.

### 3. Short Context Sufficient

> "In Melee, game states beyond the past 5 seconds rarely matter for the next action in a given frame. This is good because it means sequences can be kept short and memory usage low instead of training on entire replays."

This finding is critical for attention-based models where context is O(n²).

### 4. Infrastructure Dominates

> "So far, I spent over 300 hours setting up infrastructure, munging data, dealing with emulator bugs, thinking about the right feature representations, and running experiments."

Most time is infrastructure, not architecture tweaking.

## Results

### Quantitative

| Metric | Value |
|--------|-------|
| Parameters | 20M |
| Training frames | 3B |
| Training time | 5 hours |
| Training cost | ~$5 |
| GPUs | 2× RTX 3090 |
| Win rate vs L9 CPU | 95% |

### Qualitative

- Model exhibits recognizable human tech skill
- Plays like the training distribution (Fox mains)
- Not yet set up for human opponents

### Limitations

- No real-time inference optimization yet
- Not tested against humans
- Pure imitation (no RL refinement)
- Single blog post, project ongoing

## Future Directions (From Blog)

Eric planned to explore in subsequent posts:
1. **10× more data**: Scale training set
2. **Bigger models**: Test scaling laws
3. **Inference optimization**: Real-time play
4. **Offline RL**: Refine without environment
5. **Multi-token prediction**: Handle network latency
6. **Human opponents**: The real test

## Comparison to Other Approaches

| Aspect | Eric Gu | slippi-ai | Phillip |
|--------|---------|-----------|---------|
| Architecture | Transformer | GRU | MLP/GRU |
| Training | IL only | BC + RL | Pure RL |
| Characters | All | Single | Single |
| Compute | $5, 5h | Days | Months |
| Human play | Not yet | Yes | Yes |
| Win rate | 95% CPU | ~60% pros | Beat pros |

## Relevance to ExPhil

### Applicable Ideas

1. **Multi-character training**: Train on all low-tier characters together
2. **Simple objective**: Next-token prediction is effective
3. **Scaling matters**: More data > architecture tricks
4. **Short context**: Mamba's O(n) not strictly necessary for short windows

### Differences from ExPhil Approach

| Aspect | Eric Gu | ExPhil |
|--------|---------|--------|
| Architecture | Transformer | Mamba |
| Complexity | O(n²) | O(n) |
| Context | ~300 frames | 90+ frames |
| Training | IL only | BC + RL |
| Target | All chars | Low-tier |

### Key Takeaways

1. **Transformers work**: Don't need specialized architecture
2. **Data efficiency**: 3B frames sufficient for strong results
3. **Cheap training**: Consumer GPUs viable
4. **Multi-char helps**: Generalization aids learning

## Code Availability

As of the blog post, no public repository was available. The blog documents methodology but implementation remains private.

## References

- [Blog Post](https://ericyuegu.com/melee-pt1) - Primary source
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Referenced philosophy
- [Slippi](https://slippi.gg/) - Data source
- [Peppi](https://github.com/hohav/peppi) - Likely replay parser used
