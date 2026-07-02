# Next Training Experiments

## Prerequisites
- Current 200-file Mamba run completes with entropy_weight=0.01
- Verify mode collapse is resolved (action diversity > 10 by epoch 5)
- All tests passing

## Experiment 1: Backbone Comparison (200 files)

**Goal:** Find the best temporal backbone for Melee BC at scale.

**Setup:**
```bash
mix run scripts/benchmark_backbones.exs \
  --replays ./replays/huggingface --max-files 200 --epochs 10 \
  --backbones mamba,griffin,gated_delta_net,min_gru,rwkv
```

**Success criteria:**
- val_loss < 4.0 after 10 epochs
- Action diversity > 20
- Inference < 16.67ms (60fps)

**Expected outcome:** Mamba is fastest inference but Griffin/GatedDeltaNet may have better accuracy. MinGRU is the dark horse — simple and fast.

## Experiment 2: Window Size Tuning

**Goal:** Find optimal temporal context window.

**Setup:**
```bash
for ws in 30 60 120 180; do
  mix run scripts/train.exs --backbone mamba --max-files 200 --epochs 10 \
    --window-size $ws --name window_${ws} --seed 42
done
```

**Hypothesis:**
- 30 frames (0.5s): enough for L-cancels, wavedashes
- 60 frames (1s): current default, covers most neutral game
- 120 frames (2s): edgeguarding, recovery
- 180 frames (3s): full interaction sequences

**Trade-offs:** Larger windows use more VRAM and train slower.

## Experiment 3: Character-Specific Training

**Goal:** Specialist model for one character.

**Setup:**
```bash
mix run scripts/train.exs --backbone mamba --max-files 200 --epochs 30 \
  --train-character mewtwo --name mewtwo_specialist --seed 42
```

**Why Mewtwo:** Low-tier with unique mechanics (teleport, shadow ball charge). Less data available but more consistent play patterns. Window_size=90 recommended (teleport recovery).

## Experiment 4: Scale to 500+ Files

**Goal:** Test if more data improves quality or if we hit diminishing returns.

**Setup:**
```bash
# First precompute embeddings in separate process
mix run scripts/precompute_embeddings.exs --replays ./replays/huggingface --max-files 500

# Then train
mix run scripts/train.exs --backbone mamba --max-files 500 --epochs 20 \
  --entropy-weight 0.01 --name mamba_500 --seed 42
```

**Concern:** BFC pool fragmentation at 500 files. May need `precompute_embeddings.exs` to avoid embedding-in-training GPU contamination.

## Experiment 5: Curriculum Learning

**Goal:** Test if graduated scaling improves convergence.

**Setup:** Use the Curriculum callback:
```elixir
{Curriculum, [stages: [{100, 5}, {200, 30}]]}
```
Train on 100 files for 5 epochs, then expand to 200 for remaining 25.

**Hypothesis:** The model establishes basic action representations on the easier subset before being challenged by the full distribution.

## Metrics for All Experiments

| Metric | Target | How |
|--------|--------|-----|
| Val loss | < 3.0 | Lower is better |
| Action diversity | > 30 | More unique predicted combos |
| Button accuracy | > 90% | Per-button correct rate |
| L/R/Z recall | > 30% | Rare action detection |
| Stick top-1 acc | > 40% main, > 80% c-stick | Correct bucket prediction |
| Inference speed | < 16.67ms | 60fps requirement |
| Temporal consistency | 70-90% | Frame coherence (not 100% which = collapse) |
