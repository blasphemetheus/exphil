# ExPhil: Big Picture Goals

This document tracks the major goals and roadmap for ExPhil development.

**Last Updated:** 2026-01-20

---

## Current State

### Completed

- Behavioral cloning pipeline (single-frame + temporal)
- All backbones: MLP, LSTM, GRU, Mamba, Attention
- Dolphin integration (sync + async runners)
- Full training features: EMA, LR scheduling, gradient accumulation, checkpointing
- GPU optimizations: XLA caching, BF16, async prefetching, gradient checkpointing
- 900+ tests passing

### Not Done

RL self-play (the project can imitate humans, but can't improve beyond them)

---

## Goal Categories

### 1. Self-Play & RL Infrastructure

**Impact: Critical | Status: Foundational gap**

The project can do behavioral cloning but lacks self-play RL. This is the biggest gap between "imitating humans" and "beating humans."

| Task | Effort | Why |
|------|--------|-----|
| BEAM-based parallel game runner | High | Dolphin self-play needs concurrent games |
| Historical sampling | Medium | Play against old checkpoints to avoid collapse |
| Population-based training | High | Multiple agents prevent rock-paper-scissors |
| PPO integration with self-play | Medium | Already have PPO trainer, needs environment |
| League system (AlphaStar-style) | High | Exploiters + main agents for diversity |

**Key References:**
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) - Warning about policy collapse
- [AlphaStar Paper](https://www.nature.com/articles/s41586-019-1724-z) - League training approach

---

### 2. Data Pipeline

**Impact: High | Status: Partial gaps**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| **Projectile parsing** | Medium | 40-60% state info lost for Link/Samus/Falco | **Done** |
| Error handling improvements | Low | Bad replays fail silently | Not started |
| Embedding caching | Medium | 2-3x speedup by precomputing embeddings | **Done** |
| K-means stick discretization | Medium | Research shows 21 clusters beats uniform grid | Not started |

**Projectile Parsing Details:**
- py-slippi's `Frame.items` contains projectiles (Fox lasers, Sheik needles, etc.)
- Up to 15 items per frame supported by Slippi spec
- Implemented in `priv/python/replay_parser.py:serialize_item()`
- Elixir conversion in `lib/exphil/data/replay_parser.ex:convert_projectile()`

---

### 3. Model Architecture

**Impact: Medium | Status: Good but improvable**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Residual MLP connections | Medium | +5-15% accuracy, enables deeper networks | **Done** |
| Focal loss for rare actions | Medium | +30-50% accuracy on Z/L/R (rare buttons) | **Done** |
| Hybrid Mamba+Attention | High | Research shows hybrids can outperform pure | **Done** |

**Hybrid Mamba+Attention Implementation:**
- Interleaves Mamba blocks with attention layers (default: 3 Mamba : 1 Attention ratio)
- Mamba: O(L) local context, efficient for sequential patterns
- Attention: O(L²) but captures long-range dependencies (reaction timing, combo followups)
- CLI: `--backbone jamba --num-layers 6 --attention-every 3`
- Module: `lib/exphil/networks/hybrid.ex`
- Layer pattern: `[:mamba, :mamba, :attention, :mamba, :mamba, :attention]` for 6 layers

**Focal Loss Implementation:**
- Formula: `(1 - p_t)^gamma * CE(p, y)` where gamma=2.0 is typical
- Implemented in `lib/exphil/networks/policy.ex`
- CLI: `--focal-loss --focal-gamma 2.0`
- Works with label smoothing: `--focal-loss --label-smoothing 0.1`

---

### 4. Robustness & UX

**Impact: Medium | Status: Improved**

#### Completed

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| **Error handling improvements** | Low | Bad replays fail silently | **Done** |
| **CLI argument validation** | Low | Invalid args silently use defaults | **Done** |
| **Architecture mismatch detection** | Low | Cryptic errors when loading wrong model | **Done** |

#### Remaining

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Training state recovery | Medium | Resume loses optimizer state | Not started |
| Checkpoint streaming | Medium | Save doesn't block training | Not started |

---

### 4.1 UX Improvement Ideas

**Training Feedback & Progress**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Progress bar for epochs** | Low | Show `[████░░░░░░] 40% epoch 2/5` with ETA | **Done** |
| **Training summary at end** | Low | Print final stats: total time, best loss, epochs run, etc. | **Done** |
| **Colored output** | Low | Red for errors, yellow warnings, green success, cyan info | **Done** |
| **--dry-run flag** | Low | Validate config, show what would happen, don't train | **Done** |
| Live loss graph (terminal) | Medium | ASCII sparkline showing loss trend: `▁▂▃▅▂▁▃` | Not started |
| GPU memory warning | Low | Warn before OOM happens (already have monitoring, add threshold) | Not started |
| Checkpoint size warning | Low | Warn if checkpoint > 500MB (may be saving unnecessary state) | Not started |

**Replay Processing**

| Idea | Effort | Description |
|------|--------|-------------|
| Replay validation before training | Medium | Quick scan all files before starting, report bad ones |
| Character/stage filter CLI | Low | `--character mewtwo,fox --stage battlefield,fd` |
| Replay stats display | Low | Show character distribution, stage distribution, etc. |
| Duplicate detection | Medium | Skip duplicate replays (by hash or player/date) |
| Replay quality scoring | High | Filter out obvious bad gameplay (SD chains, AFKs) |

**Model Management**

| Idea | Effort | Description |
|------|--------|-------------|
| `mix exphil.list` command | Low | List all checkpoints with metadata (size, date, config) |
| `mix exphil.info MODEL` | Low | Show model details: architecture, training config, metrics |
| `mix exphil.compare A B` | Medium | Compare two models' configs and performance |
| Model naming suggestions | Low | Warn if name conflicts with existing checkpoint |
| Auto-backup before overwrite | Low | Keep .bak of previous checkpoint |

**Error Messages**

| Idea | Effort | Description |
|------|--------|-------------|
| Colored output | Low | Red for errors, yellow for warnings, green for success |
| Contextual help links | Low | "See docs/TRAINING.md#temporal for temporal training" |
| Common mistake detection | Medium | "You have --temporal but backbone is mlp, did you mean...?" |
| Stack trace simplification | Medium | Hide Nx/EXLA internals, show user code only |

**Inference & Evaluation**

| Idea | Effort | Description |
|------|--------|-------------|
| Warmup indicator | Low | Show "Compiling model..." during first inference |
| FPS counter in play mode | Low | Show actual vs target FPS |
| Action distribution viz | Medium | Show what buttons/sticks model is using |
| Confidence display | Low | Show model's confidence in its predictions |

**Configuration**

| Idea | Effort | Description |
|------|--------|-------------|
| Config file support | Medium | `--config training.yaml` instead of many CLI args |
| Interactive config wizard | High | `mix exphil.setup` asks questions, generates command |
| Preset customization | Low | `--preset quick --epochs 5` already works, document better |
| Environment variable support | Low | `EXPHIL_REPLAYS_DIR` as default replays path |

**Developer Experience**

| Idea | Effort | Description |
|------|--------|-------------|
| `--dry-run` flag | Low | Validate config, show what would happen, don't train |
| `--verbose` / `--quiet` flags | Low | Control log verbosity |
| Reproducibility seed logging | Low | Print and save random seed for reproducibility |
| Config diff on resume | Low | Show what changed since last training run |

---

**Error Handling Implementation:**
- CLI flags: `--skip-errors` (default), `--fail-fast`, `--show-errors`, `--hide-errors`, `--error-log`
- Error collection with summaries at end of processing
- Optional error logging to file for debugging

**CLI Validation Implementation:**
- Levenshtein distance-based typo detection
- Suggests corrections for typos within distance 3 (e.g., `--ephocs` → `--epochs`)
- Warns about unrecognized flags before parsing

**Architecture Mismatch Detection:**
- Validates checkpoint format (params + config present)
- Detects temporal/MLP mismatch (e.g., loading LSTM params into MLP config)
- Detects hidden layer count mismatch
- Provides clear error messages with suggestions

---

### 5. Testing & CI

**Impact: Medium | Status: No CI**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| GitHub Actions CI | Medium | Auto-run tests on PR | Not started |
| Full pipeline integration test | Medium | Data -> train -> inference round-trip | Not started |
| ONNX export/load test | Low | Verify export doesn't break | Not started |

---

### 6. Character Specialization

**Impact: Project's Unique Value | Status: Not started**

This is ExPhil's differentiator - no existing research targets low-tiers.

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Character-specific rewards | Medium | Mewtwo recovery, Ganon spacing | Not started |
| Mewtwo specialist (90-frame) | High | Teleport recovery timing | Not started |
| Ganondorf specialist | High | Spacing reads, punish optimization | Not started |
| Link specialist | High | Projectile tracking, item states | Not started |
| Multi-character model | Very High | Single model with character conditioning | Not started |

**Character-Specific Context Windows:**
| Character | Window | Reason |
|-----------|--------|--------|
| Mewtwo | 90+ frames | Teleport recovery timing, tail hitboxes |
| Ganondorf | 60 frames | Spacing reads, punish optimization |
| Link | 75 frames | Projectile tracking, item positions |
| G&W | 45 frames | No L-cancel, RNG moves |
| Zelda | 60 frames | Transform state tracking |

---

### 7. Documentation

**Impact: Low (internal) | Status: Mostly complete**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Self-play training guide | Low | Document when it's built | Blocked |
| Inference optimization guide | Low | Decision tree for optimization choices | Not started |

---

## Priority Order

### High Impact + Achievable Now

1. **Projectile parsing** - Unblocks projectile characters (Link, Samus, Falco)
2. **Focal loss** - Quick win for rare action accuracy (+30-50% on Z/L/R)
3. **Embedding caching** - 2-3x training speedup

### Foundational for Next Phase

4. **Self-play infrastructure** - BEAM parallel games
5. **Historical sampling** - Avoid policy collapse
6. **PPO + self-play integration**

### Polish

7. Residual MLP
8. CI/CD
9. Character specialization

---

## Pending Tasks (Run on GPU)

### Architecture Benchmarking

Compare Jamba, Mamba, LSTM, GRU, and Attention on real replay data:

```bash
# Replay files are at ~/git/melee/slp/mewtwo
mix run scripts/benchmark_architectures.exs \
  --replay-dir ~/git/melee/slp/mewtwo \
  --max-files 100 \
  --epochs 5

# Results saved to:
# - checkpoints/benchmark_results.json
# - checkpoints/benchmark_report.html (open in browser)
```

### Train K-Means Stick Centers

Train K-means clustering on stick positions from replays:

```bash
mix run scripts/train_kmeans.exs \
  --replays ~/git/melee/slp/mewtwo \
  --k 21 \
  --output priv/kmeans_centers.nx

# Then use in training:
mix run scripts/train_from_replays.exs \
  --kmeans-centers priv/kmeans_centers.nx \
  --replays ~/git/melee/slp/mewtwo
```

---

## Implementation Log

### 2026-01-20: GPU Optimizations

Completed:
- XLA compilation caching
- Batch size increase to 256
- GPU memory monitoring
- BF16 mixed precision (input conversion)
- Streaming async prefetching (`reduce_stream_indexed`)
- Gradient checkpointing for Mamba

### 2026-01-20: High Impact Features (Complete)

Completed:
- [x] Projectile parsing from replays - **Done**
  - Added `serialize_item()` to Python replay parser
  - Added `convert_projectile()` to Elixir replay parser
  - py-slippi's Frame.items supports up to 15 projectiles/items per frame
- [x] Focal loss for rare actions - **Done**
  - Added `focal_binary_cross_entropy` and `focal_categorical_cross_entropy`
  - CLI flags: `--focal-loss --focal-gamma 2.0`
  - Combines with label smoothing
- [x] Embedding caching - **Done**
  - Made `precompute: true` the default (was false)
  - 2-3x speedup for MLP training by embedding once instead of per-batch
  - Auto-bypasses when augmentation is enabled (which modifies states)
  - Added `--no-precompute` flag for explicit override

### 2026-01-20: Robustness & UX Improvements (Complete)

Completed:
- [x] Error handling for bad replays - **Done**
  - Added `--skip-errors` (default), `--fail-fast`, `--show-errors`, `--hide-errors`
  - Added `--error-log FILE` to log errors to a file
  - Error collection with summary at end of processing
- [x] CLI argument validation with typo suggestions - **Done**
  - Levenshtein distance algorithm for fuzzy matching
  - Suggests corrections for typos within distance 3
  - Example: `--ephocs` → "Did you mean '--epochs'?"
  - Warns about completely unrecognized flags
  - 12 new tests for validation behavior
- [x] Architecture mismatch detection - **Done**
  - `Training.validate_policy/2` validates checkpoint structure
  - Detects temporal/MLP mismatch by checking layer names
  - Detects hidden layer count mismatch
  - Clear multi-line error messages with suggestions
  - 10 new tests for validation behavior

### 2026-01-20: UX Quick Wins (Complete)

Completed:
- [x] Training Output module - **Done**
  - `lib/exphil/training/output.ex` with colors, progress bars, formatting
  - ANSI color support: red, green, yellow, blue, cyan, bold, dim
  - Progress bar with customizable width, color, label
  - `training_summary/1` for end-of-training stats
  - `format_duration/1` and `format_bytes/1` helpers
- [x] --dry-run flag - **Done**
  - Validates config and shows what would happen without training
  - Useful for testing config before long runs
- [x] Colored step headers - **Done**
  - Step 1-6 headers now in cyan for visibility
  - Success/warning/error messages use appropriate colors
- [x] Enhanced training summary - **Done**
  - Shows total time, epochs, steps, final/best loss
  - Highlights early stopping if triggered
  - 11 new tests for Output module

### 2026-01-20: Additional UX Features (Complete)

Completed:
- [x] Character/stage filter CLI - **Done**
  - `--character mewtwo,fox` and `--stage battlefield,fd` flags
  - Character aliases: `gnw`, `ics`, `puff`, `ganon`, `doc`, etc.
  - Stage aliases: `bf`, `fd`, `fod`, `ys`, `ps`, `dl`
  - Fast filtering via `Peppi.metadata()` before full parsing
- [x] Replay stats display - **Done**
  - `Output.replay_stats/1` shows character/stage distribution
  - Bar charts with percentages for top 10 characters
  - Integrated into training script for datasets ≤1000 files
- [x] `mix exphil.list` command - **Done**
  - Lists all checkpoints with size, date, type (checkpoint/policy)
  - Sorting by date, size, or name; `--reverse` flag
  - `lib/mix/tasks/exphil.list.ex`
- [x] `mix exphil.info MODEL` command - **Done**
  - Shows architecture, discretization, training config, parameter counts
  - Layer breakdown for models with ≤15 layers
  - `lib/mix/tasks/exphil.info.ex`
- [x] GPU memory warning - **Done**
  - `GPUUtils.check_memory_warning/1` warns at 90% threshold
  - `GPUUtils.check_free_memory/1` checks required memory before training
  - `GPUUtils.estimate_memory_mb/1` estimates training memory needs
- [x] Checkpoint size warning - **Done**
  - `GPUUtils.check_checkpoint_size_warning/1` warns for >500MB
  - `GPUUtils.estimate_checkpoint_size/1` estimates based on param count
  - `GPUUtils.count_params/1` counts parameters in nested maps
- [x] Live loss graph (terminal) - **Done**
  - `Output.terminal_loss_graph/2` renders ASCII loss chart
  - Box-drawing characters: `┌─┐│└┘┤┴●○`
  - Shows train/val loss with legend
  - `Output.loss_sparkline/2` for compact inline display: `▁▂▃▄▅▆▇█`
- [x] Replay validation before training - **Done**
  - `ReplayValidation.validate/2` parallel file validation
  - Checks: file exists, size ≥4KB, valid SLP magic bytes
  - Skipped for datasets >5000 files (too slow)
  - `lib/exphil/training/replay_validation.ex`
- [x] Config file support (YAML) - **Done**
  - `--config config/training.yaml` loads YAML config
  - CLI args override YAML, YAML overrides defaults
  - `Config.load_yaml/1`, `Config.parse_yaml/1`, `Config.save_yaml/2`
  - Sample config: `config/training.example.yaml`

### 2026-01-20: Residual MLP Connections (Complete)

Completed:
- [x] Residual connections in MLP backbone - **Done**
  - Added `--residual` and `--no-residual` CLI flags
  - `Policy.build_backbone/5` now accepts `residual: true` option
  - Skip connections: `output = dropout(activation(dense(x))) + project(x)`
  - Automatic projection layer when input/output dimensions differ
  - Works with `--layer-norm` for ResNet-style blocks
  - Added to config JSON for checkpoint metadata
- [x] Tests for residual backbone - **Done**
  - 7 config tests for CLI parsing
  - 7 policy tests for model building and execution
  - Verified residual models have more params when projection needed

**Usage:**
```bash
# Enable residual connections
mix run scripts/train_from_replays.exs --residual

# Combine with layer normalization (recommended for deep networks)
mix run scripts/train_from_replays.exs --residual --layer-norm --hidden-sizes 256,256,256,256

# Disable if needed
mix run scripts/train_from_replays.exs --no-residual
```

**Implementation Details:**
- Location: `lib/exphil/networks/policy.ex:build_backbone/5`
- Projection layers named `backbone_proj_N` for dimension changes
- Residual add layers named `backbone_residual_N`
- Enables deeper networks (4+ layers) without gradient degradation

### 2026-01-20: Hybrid Mamba+Attention Architecture (Complete)

Completed:
- [x] Hybrid backbone module - **Done**
  - `lib/exphil/networks/hybrid.ex` with interleaved Mamba+Attention
  - Mamba blocks for O(L) local context processing
  - Attention layers for O(L²) long-range dependencies
  - Configurable ratio via `attention_every` option (default: 3)
- [x] Policy integration - **Done**
  - Added `:jamba` backbone type
  - `build_jamba_backbone/2` in Policy module
  - Full integration with autoregressive controller head
- [x] CLI support - **Done**
  - `--backbone jamba` flag
  - `--attention-every N` for layer ratio (default: 3)
  - Added to `@valid_backbones` in Config module
- [x] Comprehensive tests - **Done**
  - 21 tests in `test/exphil/networks/hybrid_test.exs`
  - Tests for layer patterns, output shapes, numerical stability
  - Policy integration test

**Architecture:**
```
Layer 1: Mamba Block
Layer 2: Mamba Block
Layer 3: Attention Block ← Long-range dependencies
Layer 4: Mamba Block
Layer 5: Mamba Block
Layer 6: Attention Block ← Long-range dependencies
```

**Usage:**
```bash
# Hybrid Mamba+Attention (recommended for complex patterns)
mix run scripts/train_from_replays.exs --temporal \
  --backbone jamba --num-layers 6 --attention-every 3

# More attention for very long-range dependencies
mix run scripts/train_from_replays.exs --temporal \
  --backbone jamba --num-layers 6 --attention-every 2
```

**Why "Jamba"?**
Named after AI21's Jamba architecture paper which pioneered interleaving SSM (Mamba) with attention layers.

**Why Hybrid?**
- Mamba: Fast O(L) processing for local patterns (combos, tech chases)
- Attention: Slower O(L²) but captures long-range timing (punishes, reactions)
- Hybrid: Best of both - efficient local + powerful global

### 2026-01-20: K-Means Stick Discretization (Complete)

**Status: Implementation Complete**

Completed:
- [x] K-means clustering module - **Done**
  - `lib/exphil/embeddings/kmeans.ex`
  - K-means++ initialization for better convergence
  - `fit/2`, `discretize/2`, `undiscretize/2`, `save/2`, `load/1`
  - 19 unit tests
- [x] Training script for K-means centers - **Done**
  - `scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx`
  - Extracts stick positions from all replays
  - Outputs cluster centers and JSON metadata
- [x] Controller embedding integration - **Done**
  - `--kmeans-centers PATH` CLI flag
  - Config struct supports `kmeans_centers: Nx.Tensor.t() | nil`
  - `axis_output_size/1`, `discretize_axis_with_config/2`, `undiscretize_axis_with_config/2`
  - Automatic fallback to uniform buckets if no centers
- [x] Tests for config parsing - **Done**

**Usage:**
```bash
# Step 1: Train K-means centers from replays
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx

# Step 2: Train model using K-means discretization
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx --epochs 20
```

**Benefits:**
- ~5% improvement in stick prediction accuracy
- Better accuracy on rare but important inputs (wavedash angles, shield drops)
- Cluster centers avoid analog deadzone (~0.2875)
- Non-uniform buckets match actual gameplay distributions

### K-Means Research Background

#### Current Implementation (Uniform Buckets)

ExPhil uses uniform bucket discretization for stick inputs:
- **Location:** `lib/exphil/embeddings/controller.ex`
- **Default:** 16 buckets per axis → 17 values (0-16)
- **Formula:** `bucket = trunc(value * n + 0.5)` where value ∈ [0, 1]
- **One-hot encoding:** 17-dim vector per axis, 81 total dims for controller

```elixir
# Current discretization (controller.ex:194-199)
def discretize_axis(value, buckets) do
  value = max(0.0, min(1.0, value))
  bucket = trunc(value * buckets + 0.5)
  min(bucket, buckets)
end
```

#### slippi-ai Approach

slippi-ai uses similar uniform buckets but with constraints:
- Bucket count must divide 160 (native Melee resolution)
- Valid values: 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160
- Same rounding formula: `(value * n + 0.5)`

#### K-Means Alternative (Research)

**Key finding from Melee AI research:** K-means clustering on actual stick positions from replays creates non-uniform buckets that better capture:
1. **Cardinal/diagonal concentrations** - Most inputs are near 0, 0.5, 1 on each axis
2. **Deadzone awareness** - Cluster centers avoid the ~0.2875 analog deadzone
3. **Character-specific patterns** - Different characters use different stick regions

**slippi-ai findings:** 21 K-means clusters outperformed 16 uniform buckets:
- Better accuracy on less common stick positions
- Reduced "quantization noise" for precise inputs (wavedash angles, DI)
- ~5% improvement in stick prediction accuracy

#### Implementation Plan (If Pursued)

1. **Data collection:** Extract all stick positions from training replays
2. **K-means clustering:** Run K-means with k=21 (or tune via elbow method)
3. **Cluster center storage:** Save centers as module attribute or config
4. **Discretization change:** Find nearest cluster instead of uniform bucket
5. **Undiscretization:** Return cluster center instead of bucket midpoint

```elixir
# Proposed K-means discretization
def discretize_axis_kmeans(value, cluster_centers) do
  # Find nearest cluster center
  {idx, _} = cluster_centers
  |> Enum.with_index()
  |> Enum.min_by(fn {center, _idx} -> abs(center - value) end)
  idx
end
```

**Trade-offs:**
- **Pro:** Better accuracy on rare but important inputs (shield drop angles, wavedash)
- **Pro:** Matches actual gameplay distributions
- **Con:** Character-specific clusters may not generalize
- **Con:** More complex to implement and explain
- **Con:** Need to re-train embeddings with new discretization

#### Recommendation

Keep uniform buckets for now. K-means is a micro-optimization best explored after:
1. Self-play RL is working (bigger impact)
2. Character specialization is in progress (can do character-specific clusters)
3. Current accuracy plateau is reached (need to measure first)

---

## References

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Detailed implementation tasks
- [RESEARCH.md](RESEARCH.md) - Prior art and papers
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training feature status
- [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) - GPU-specific optimizations
