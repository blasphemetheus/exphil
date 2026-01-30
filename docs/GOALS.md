# ExPhil: Big Picture Goals

This document tracks the major goals and roadmap for ExPhil development.

**Last Updated:** 2026-01-23

---

## Current State

### Completed

- Behavioral cloning pipeline (single-frame + temporal)
- All backbones: MLP, LSTM, GRU, Mamba, Attention, Jamba (Mamba+Attention hybrid)
- Dolphin integration (sync + async runners)
- Full training features: EMA, LR scheduling, gradient accumulation, checkpointing
- GPU optimizations: XLA caching, BF16, async prefetching, gradient checkpointing
- UX improvements: colored output, progress bars, dry-run, validation, YAML config
- Frame delay augmentation for online play robustness
- GitHub Actions CI (test, format, dialyzer)
- **Self-play RL infrastructure** (GenServer architecture, PPO integration, Elo matchmaking)
- Mock environment with physics (for fast self-play testing)
- 1933 tests passing

### Next Step

Large-scale self-play training (train on GPU cluster, tune hyperparameters)

---

## Goal Categories

### 1. Self-Play & RL Infrastructure

**Impact: Critical | Status: ✅ Complete**

GenServer-based self-play infrastructure with BEAM concurrency, historical sampling, and Elo matchmaking.

| Task | Effort | Status |
|------|--------|--------|
| BEAM-based parallel game runner | High | **Done** - GamePoolSupervisor + GameRunner GenServers |
| Historical sampling | Medium | **Done** - PopulationManager with configurable history |
| Population-based training | High | **Done** - Matchmaker with skill-based pairing |
| PPO integration with self-play | Medium | **Done** - train_self_play.exs script |
| Mock environment | Medium | **Done** - Physics-based MockEnv.Game |
| Elo rating system | Low | **Done** - Matchmaker with leaderboard |
| League system (AlphaStar-style) | High | Not started (future enhancement) |

**Implementation:**
- `lib/exphil/self_play/` - GenServer architecture
  - `supervisor.ex` - Top-level supervisor
  - `game_runner.ex` - Per-game GenServer with physics mock
  - `game_pool_supervisor.ex` - DynamicSupervisor for parallel games
  - `population_manager.ex` - Policy versioning and historical sampling
  - `experience_collector.ex` - Batched experience collection
  - `matchmaker.ex` - Elo ratings and skill-based matchmaking
  - `elo.ex` - Elo calculation utilities
- `lib/exphil/mock_env/` - Physics-based mock Melee environment
  - `game.ex` - Game loop, stage collision, blast zones
  - `player.ex` - Player physics (gravity, jumping, movement)

**Usage:**
```bash
# Quick test with mock environment (short episodes)
mix run scripts/train_self_play.exs --game-type mock --timesteps 1000 --max-episode-frames 600

# Full training with pretrained policy
mix run scripts/train_self_play.exs \
  --pretrained checkpoints/imitation_policy.bin \
  --num-games 8 --timesteps 100000 --track-elo
```

**Verified (2026-01-24):**
- Mock environment runs successfully with parallel games
- Game results are reported to matchmaker for Elo tracking
- Elo system handles wins, losses, and draws correctly (26 tests passing)
- Note: Random policies typically result in draws (equal stocks at timeout), so Elo changes are 0. Use trained policies or longer episodes to see meaningful Elo movement.

**Key References:**
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) - Warning about policy collapse
- [AlphaStar Paper](https://www.nature.com/articles/s41586-019-1724-z) - League training approach

---

### 2. Data Pipeline

**Impact: High | Status: Partial gaps**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| **Projectile parsing** | Medium | 40-60% state info lost for Link/Samus/Falco | **Done** |
| Error handling improvements | Low | Bad replays fail silently | **Done** |
| Embedding caching | Medium | 2-3x speedup by precomputing embeddings | **Done** |
| K-means stick discretization | Medium | Research shows 21 clusters beats uniform grid | **Done** |
| **Embedding disk caching** | Medium | Save precomputed embeddings to disk for reuse across runs | **Done** |
| **Augmented embedding caching** | Medium | Pre-cache multiple augmented versions per state | **Done** |

**Embedding Disk Caching (Done):**
- Use `--cache-embeddings` to save precomputed embeddings to disk
- Use `--cache-dir PATH` to specify cache location (default: `cache/embeddings`)
- Cache key includes embedding config hash (action_mode, character_mode, stage_mode, etc.)
- Cache is invalidated when replay files change

**Augmented Embedding Caching (Done):**

Pre-computes multiple augmented versions of each state for ~100x faster training with augmentation.

**Usage:**
```bash
# Use --cache-augmented to pre-compute original + mirrored + noisy variants
mix run scripts/train_from_replays.exs \
  --cache-augmented \
  --augment \
  --num-noisy-variants 2 \
  --noise-scale 0.01
```

**How it works:**
- Pre-computes: 1 original + 1 mirrored + N noisy variants per frame
- Stores as 3D tensor: `{num_frames, num_variants, embed_size}`
- During training, randomly selects variant per sample (stochastic like online augmentation)
- Cache key includes augmented flag, num_noisy_variants, and noise_scale

**Benefits:**
- ~100x faster training with augmentation (GPU utilization stays high)
- Same generalization benefits as online augmentation
- Compatible with disk caching (`--cache-dir`)

**Trade-offs:**
- 2-4x more memory/disk for embeddings
- Noise uses deterministic seeds (reproducible but slightly less random)

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

**Embedding Dimension Optimization:**

Current embedding size is **1204 dimensions** with all optimizations enabled.

*Current breakdown (1204 total):*
| Component | Dimensions | Notes |
|-----------|------------|-------|
| Player 1 | 488 | Base 440 + speeds 5 + frame_info 2 + stock 1 + ledge_dist 1 + compact Nana 39 |
| Player 2 | 488 | Same as Player 1 |
| Stage | 64 | One-hot encoded (64 stages) |
| Player names | 112 | One-hot for human player identification (unused) |
| Spatial features | 4 | Distance 1 + relative_pos 2 + frame_count 1 |
| Projectiles | 35 | 5 slots × 7 dims |
| Controller | 13 | 8 buttons + 4 sticks + 1 shoulder |

*Completed optimizations:*
- ✅ **Compact Nana** (455 → 39 dims, preserves IC tech)
- ✅ **Normalized jumps** (7 → 1 dim, preserves ordinal info)
- ✅ **Projectiles enabled** (essential for Link/Samus/Falco)
- ✅ **Frame info enabled** (hitstun/action frame for punish timing)
- ✅ **Spatial features** (distance, relative position, game frame)
- ✅ **Learned action embedding** (399×2=798 → 64×2=128 dims, -670 dims, better generalization)
- ✅ **Enhanced Nana mode** (39 → 14 dims + Nana action ID, precise IC tech via learned embedding)
- ✅ **4 action IDs** (2 players + 2 Nanas when `nana_mode: :enhanced, with_nana: true`)

*Future optimizations (TODO):*

| Option | Current | Potential | Benefit | Status |
|--------|---------|-----------|---------|--------|
| **Stage embedding modes** | 64 dims | 7 (compact) or 1 ID (learned) | `stage_mode: :one_hot_compact` (7 dims) or `:learned` (1 ID + embedding) | **Done** |
| **Player names embedding** | 112 dims | 0-32 dims | Style-conditional imitation | **Done** |
| **IC Tech Feature Block** | N/A | +32 dims | Dedicated grab/regrab/desync features | Not started |

**Learned Action Embedding Implementation:**
- Use `action_mode: :learned` in PlayerEmbed config
- Action IDs appended at end of game embedding (2 or 4 values depending on Nana mode)
- Policy network uses `action_embed_size: 64` and `num_action_ids: 2 or 4`
- Network learns action similarities (e.g., "all aerials" vs "all tilts")
- Supports both single-frame and temporal (mamba, sliding_window, mlp) backbones

**Enhanced Nana Mode for IC Tech:**
- Use `nana_mode: :enhanced` with `action_mode: :learned` for optimal IC training
- Nana action ID allows precise action learning (dair vs fair vs nair)
- Reduces Nana embedding from 39 → 14 dims continuous + 1 action ID
- Total network input: 254 continuous + 4×64 action embedding = 510 dims (under 512 target!)

**Stage Embedding Modes (Implemented):**
- `:one_hot_full` (default): 64-dim one-hot
- `:one_hot_compact`: 7-dim (6 competitive + "other"), saves 57 dims
- `:learned`: 1 stage ID + trainable 64-dim embedding, saves 63 dims
- Competitive stages: FoD=2, PS=3, YS=8, DL=28, BF=31, FD=32
- CLI: `--stage-mode compact` or `--stage-mode learned`

**Player Names Feature (Implemented):**
- 112 dims one-hot encoding for player tag IDs
- PlayerRegistry maps tags ("Plup", "Jmook") → numeric IDs
- Tags extracted from replay metadata during parsing
- CLI: `--learn-player-styles --player-registry players.json`
- Registry persists to JSON for reproducible training runs
- Unknown players use hash bucketing to spread across ID space

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
| Training state recovery | Medium | Resume loses optimizer state | **Done** |
| Checkpoint streaming | Medium | Save doesn't block training | **Done** |

**Async Checkpoint Implementation:**
- `lib/exphil/training/async_checkpoint.ex` - GenServer for background saves
- Queue-based save with atomic writes (temp file + rename)
- `save_async/2`, `await_pending/0`, `pending_count/0` API
- Deep binary copy to avoid EXLA tensor cross-process issues
- `test/exphil/training/checkpoint_roundtrip_test.exs` - Verifies optimizer state preservation

---

### 4.1 UX Improvement Ideas

**Script Standardization (CLI Module)**

Created `ExPhil.CLI` module for standardized argument parsing across scripts.

| Task | Effort | Status |
|------|--------|--------|
| Create CLI module with flag groups | Medium | **Done** |
| Migrate eval_model.exs | Low | **Done** |
| Migrate benchmark_architectures.exs | Low | Todo |
| Migrate play_dolphin.exs | Low | Todo |
| Migrate train_ppo.exs | Low | Todo |
| Migrate analyze_replays.exs | Low | **Done** |
| Add `:benchmark` flag group | Low | Todo |
| Add `:dolphin` flag group | Low | Todo |
| Add button press analysis to analyze_replays.exs | Low | **Done** |
| Add `:analysis` flag group | Low | **Done** |

**Benefits:**
- Consistent verbosity handling (`--quiet`/`--verbose`) across all scripts
- Reusable flag groups (`:verbosity`, `:replay`, `:checkpoint`, `:training`)
- Auto-generated help text from flag definitions
- Testable parsing logic (31 tests in `cli_test.exs`)

**Training Feedback & Progress**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Progress bar for epochs** | Low | Show `[████░░░░░░] 40% epoch 2/5` with ETA | **Done** |
| **Training summary at end** | Low | Print final stats: total time, best loss, epochs run, etc. | **Done** |
| **Colored output** | Low | Red for errors, yellow warnings, green success, cyan info | **Done** |
| **--dry-run flag** | Low | Validate config, show what would happen, don't train | **Done** |
| **Live loss graph (terminal)** | Medium | ASCII sparkline showing loss trend: `▁▂▃▅▂▁▃` | **Done** |
| **GPU memory warning** | Low | Warn before OOM happens (already have monitoring, add threshold) | **Done** |
| **Checkpoint size warning** | Low | Warn if checkpoint > 500MB (may be saving unnecessary state) | **Done** |

**Replay Processing**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Replay validation before training** | Medium | Quick scan all files before starting, report bad ones | **Done** |
| **Character/stage filter CLI** | Low | `--character mewtwo,fox --stage battlefield,fd` | **Done** |
| **Replay stats display** | Low | Show character distribution, stage distribution, etc. | **Done** |
| **Duplicate detection** | Medium | Skip duplicate replays (by hash or player/date) | **Done** |
| **Replay quality scoring** | High | Filter out obvious bad gameplay (SD chains, AFKs) | **Done** |

**Model Management**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **`mix exphil.list` command** | Low | List all checkpoints with metadata (size, date, config) | **Done** |
| **`mix exphil.info MODEL`** | Low | Show model details: architecture, training config, metrics | **Done** |
| **`mix exphil.compare A B`** | Medium | Compare two models' configs and performance | **Done** |
| Model naming suggestions | Low | Warn if name conflicts with existing checkpoint | **Done** |
| Auto-backup before overwrite | Low | Keep .bak of previous checkpoint | **Done** |

**Error Messages**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Colored output** | Low | Red for errors, yellow for warnings, green for success | **Done** |
| **Contextual help links** | Low | "See docs/TRAINING.md#temporal for temporal training" | **Done** |
| **Common mistake detection** | Medium | "You have --temporal but backbone is mlp, did you mean...?" | **Done** |
| **Stack trace simplification** | Medium | Hide Nx/EXLA internals, show user code only | **Done** |

**Inference & Evaluation**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Warmup indicator** | Low | Show "Compiling model..." during first inference | **Done** |
| FPS counter in play mode | Low | Show actual vs target FPS | **Done** |
| **Action distribution viz** | Medium | Show what buttons/sticks model is using | **Done** |
| Confidence display | Low | Show model's confidence in its predictions | **Done** |
| Per-button accuracy | Low | Break down button accuracy per button (A, B, X, Y, Z, L, R) | **Done** |
| Inference timing | Low | Show ms/frame to assess real-time playability (need <16.7ms) | **Done** |
| Button prediction rates | Low | Compare model's predicted vs actual button press rates | **Done** |
| Loss component breakdown | Low | Show individual loss terms (buttons, stick_x, stick_y, etc.) | **Done** |
| Stick confusion analysis | Medium | Show common stick prediction errors (e.g., neutral→up) | **Done** |
| Export predictions | Low | Save predicted vs actual for Livebook analysis | Todo |
| Frame delay testing | Medium | Test accuracy at different frame delays for online robustness | Todo |

**Configuration**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **Config file support** | Medium | `--config training.yaml` instead of many CLI args | **Done** |
| **Interactive config wizard** | High | `mix exphil.setup` asks questions, generates command | **Done** |
| **Preset customization** | Low | `--preset quick --epochs 5` already works, document better | **Done** |
| **Environment variable support** | Low | `EXPHIL_REPLAYS_DIR` as default replays path | **Done** |

**Developer Experience**

| Idea | Effort | Description | Status |
|------|--------|-------------|--------|
| **`--dry-run` flag** | Low | Validate config, show what would happen, don't train | **Done** |
| **`--verbose` / `--quiet` flags** | Low | Control log verbosity | **Done** |
| **Reproducibility seed logging** | Low | Print and save random seed for reproducibility | **Done** |
| **Config diff on resume** | Low | Show what changed since last training run | **Done** |
| **Model naming collision warnings** | Low | Warn if checkpoint name conflicts with existing file | **Done** |
| **Auto-backup before overwrite** | Low | Copy existing checkpoint to `.bak` before overwriting | **Done** |

---

### 4.2 Planned UX Features (Detailed Specs)

#### Environment Variable Support

**Purpose:** Allow default configuration via environment variables, reducing repetitive CLI args.

**Supported Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `EXPHIL_REPLAYS_DIR` | `~/melee/replays` | Default replay directory |
| `EXPHIL_CHECKPOINTS_DIR` | `./checkpoints` | Default checkpoint output directory |
| `EXPHIL_WANDB_PROJECT` | `exphil` | Default W&B project name |
| `EXPHIL_DEFAULT_PRESET` | `nil` | Default preset to use |

**Priority:** CLI args > Environment variables > Hardcoded defaults

**Usage:**
```bash
export EXPHIL_REPLAYS_DIR=/data/melee/replays
export EXPHIL_CHECKPOINTS_DIR=/models/exphil
mix run scripts/train_from_replays.exs --epochs 10  # Uses env defaults
```

---

#### Verbosity Control (`--verbose` / `--quiet`)

**Purpose:** Control output verbosity for different use cases (debugging vs CI/scripted runs).

**Levels:**
| Flag | Level | Output |
|------|-------|--------|
| `--quiet` | 0 | Errors only, no progress bars |
| (default) | 1 | Normal output with progress bars |
| `--verbose` | 2 | Debug info: batch timing, memory, gradients |

**Verbose output includes:**
- Per-batch timing breakdown
- GPU memory after each epoch
- Gradient norm statistics
- Data loading time vs training time
- Cache hit rates

**Quiet output:**
- Suppresses progress bars (for CI logs)
- Only prints errors and final summary
- Suitable for `nohup` or background jobs

---

#### Reproducibility Seed Logging

**Purpose:** Enable exact reproduction of training runs for debugging and research.

**Implementation:**
1. If `--seed N` provided, use that seed
2. Otherwise, generate random seed from system entropy
3. Print seed at startup: `[12:34:56] Random seed: 42`
4. Save seed in checkpoint metadata
5. Log seed to W&B if enabled

**Affected randomness:**
- Nx/EXLA random operations (parameter init, dropout)
- Data shuffling order
- Augmentation random choices
- Train/val split

**Usage:**
```bash
# First run (generates seed)
mix run scripts/train_from_replays.exs --epochs 5
# Output: [12:34:56] Random seed: 1234567890

# Reproduce exactly
mix run scripts/train_from_replays.exs --epochs 5 --seed 1234567890
```

---

#### Model Naming Collision Warnings

**Purpose:** Prevent accidental overwrites of valuable checkpoints.

**Behavior:**
1. Before saving, check if target path exists
2. If exists and not `--resume`, warn with file details:
   ```
   ⚠️  Checkpoint 'checkpoints/mewtwo_v1.axon' already exists
       Size: 45.2 MB, Modified: 2026-01-23 14:30
       Use --overwrite to replace, or choose a different --name
   ```
3. With `--overwrite` flag, proceed (after backup if enabled)
4. With `--no-overwrite`, fail with error

**Related flags:**
- `--name MODEL_NAME` - Explicit model name
- `--overwrite` - Allow overwriting existing checkpoints
- `--no-overwrite` - Fail if checkpoint exists (for CI)

---

#### Auto-Backup Before Overwrite

**Purpose:** Protect against data loss from failed saves or accidental overwrites.

**Behavior:**
1. When `--overwrite` is used (or implicit in `--resume`)
2. Before writing new checkpoint, copy existing to `{name}.bak`
3. If `.bak` already exists, rotate: `.bak` → `.bak.1`, `.bak.1` → `.bak.2`
4. Keep at most 3 backups (configurable with `--backup-count N`)

**Example:**
```
checkpoints/
  mewtwo_v1.axon      # Current (being overwritten)
  mewtwo_v1.axon.bak  # Previous version
  mewtwo_v1.axon.bak.1  # Two versions ago
```

**Disable:** `--no-backup` to skip backup (faster, for ephemeral training)

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

**Impact: Medium | Status: CI Complete**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| **GitHub Actions CI** | Medium | Auto-run tests on PR | **Done** |
| Full pipeline integration test | Medium | Data -> train -> inference round-trip | **Done** |
| ONNX export/load test | Low | Verify export doesn't break | **Done** |

---

### 6. Character Specialization

**Impact: Uncertain | Status: Reconsidering**

> **Note:** See [BITTER_LESSON_PLAN.md](BITTER_LESSON_PLAN.md) for why we're reconsidering this approach.

Originally, this was ExPhil's differentiator. However, the Bitter Lesson suggests that character-specific engineering may be outperformed by simply scaling a general model. We should run experiments before committing to character-specific code paths.

**Original Plan (Now Experimental):**

| Task | Effort | Bitter Lesson Concern | Status |
|------|--------|----------------------|--------|
| Character-specific rewards | Medium | Model should discover what matters | Experiment first |
| Per-character models | High | Single model + scale may beat specialists | Experiment first |
| Character-specific context windows | Medium | Let model learn optimal window | Experiment first |

**Bitter Lesson Alternative:**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Single multi-character model | Medium | Scale > specialization | **Preferred approach** |
| Character ID as input feature | Low | Let model learn differences | **Preferred approach** |
| Sparse win/loss reward | Low | Don't encode "good play" | Experiment needed |
| Larger model + more data | High | Scale beats engineering | **Preferred approach** |

**Experiments to Run:**

1. **Character-specific vs general rewards:** Train with shaped rewards vs sparse (win/loss only). Measure strategy diversity.
2. **Specialist vs generalist:** Train Mewtwo-only model vs all-character model. Compare win rates.
3. **Context window ablation:** Try 30, 60, 90 frame windows. Let data determine optimal.

**Character-Specific Context Windows (For Reference Only):**
| Character | Original Hypothesis | Experiment Status |
|-----------|--------------------|--------------------|
| Mewtwo | 90+ frames for teleport timing | Unvalidated |
| Ganondorf | 60 frames for spacing | Unvalidated |
| Link | 75 frames for projectiles | Unvalidated |
| G&W | 45 frames (no L-cancel) | Unvalidated |
| Zelda | 60 frames for transform | Unvalidated |
| Ice Climbers | 60+ frames for Nana coordination | Unvalidated |

**Ice Climbers Note:** Nana support exists in embeddings (`with_nana: true` by default). The model receives Nana's position, percent, action, and facing. Nana's jumps/shield/invuln are zeroed (libmelee limitation). Agent controls Popo only; Nana follows via game AI. Desync techniques would need to emerge from learning.

---

### 7. Documentation

**Impact: Low (internal) | Status: Mostly complete**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Self-play training guide | Low | Document when it's built | Blocked |
| Inference optimization guide | Low | Decision tree for optimization choices | **Done** |

---

## Priority Order

### ✅ Completed (High Impact)

1. ~~**Projectile parsing**~~ - ✓ Unblocks projectile characters
2. ~~**Focal loss**~~ - ✓ Rare action accuracy
3. ~~**Embedding caching**~~ - ✓ 2-3x training speedup
4. ~~**Residual MLP**~~ - ✓ Deeper networks
5. ~~**CI/CD**~~ - ✓ GitHub Actions

### Current Focus

6. **Self-play infrastructure** - BEAM parallel games
7. **Historical sampling** - Avoid policy collapse
8. **PPO + self-play integration**

### Future (Bitter Lesson-Aligned)

> See [BITTER_LESSON_PLAN.md](BITTER_LESSON_PLAN.md) for rationale.

9. **Scale up:** Larger models (1024+ hidden), more data (100k+ replays)
10. **Simplify:** Minimal embeddings experiment, sparse rewards experiment
11. **Validate assumptions:** Character-specific vs general model comparison

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

### 2026-01-22: Architecture Benchmark Results & Fixes

**Status:** COMPLETE

Ran comprehensive benchmarks on RunPod GPU pod (RTX 4090, 64GB RAM) comparing backbone architectures.

#### Benchmark Results

| Rank | Architecture | Val Loss | Batch/s | Training Time | Notes |
|------|-------------|----------|---------|---------------|-------|
| 1 | **Attention** | **3.68** | 1.0 | 31min | Best val loss, 5x faster than GRU |
| 2 | Jamba | 3.87 | 1.7 | 72min | Hybrid Mamba+Attention, fast convergence |
| 3 | GRU | 4.48 | 0.2 | 2.6h | Slow but reliable |
| 4 | LSTM | 4.75 | 0.2 | 3h | Similar to GRU |
| 5 | Mamba | 8.22 | 1.3 | 48min | Fast but struggles with temporal patterns |

**Key findings:**
- **Attention wins:** Pure attention achieved the best val loss (3.68) while being 5x faster than GRU. Global attention captures Melee's temporal patterns better than recurrent processing.
- **Jamba is the sweet spot:** Hybrid Mamba+Attention (3.87) is nearly as good as pure attention but 70% faster per batch. Good balance of speed and accuracy.
- **Mamba needs work:** The 8.22 val loss suggests Mamba alone struggles with Melee's temporal patterns. May need more epochs or architectural tuning.
- **Recurrent is reliable but slow:** GRU/LSTM achieve decent loss but are 5x slower than attention-based approaches.

#### Fixes Made

1. **GPU OOM during embedding precomputation** (Gotcha #25)
   - 175K sequences × 30 frames × 1204 dims × 4 bytes = 25GB > 24GB VRAM
   - Fixed by adding `Nx.backend_copy(Nx.BinaryBackend)` after each batch
   - Embeddings now stored on CPU RAM, transferred per-batch during training

2. **Jamba "bad argument in arithmetic expression"**
   - `num_heads: nil` from benchmark overrode default value of 4
   - `nil * head_dim` crashed in attention layer sizing
   - Fixed by filtering nil values: `|> Enum.reject(fn {_k, v} -> is_nil(v) end)`

3. **Attention "no case clause matching: :attention"**
   - `:attention` backbone type not handled in Policy.build_temporal
   - Added `:attention` as alias for `:sliding_window` in three case statements

4. **JSON encoding error for tuples**
   - `Keyword.take` returns tuples, Jason can't encode them
   - Fixed with `Map.new(Keyword.take(...))`

5. **Embedding precomputation optimization**
   - Previously embeddings were recomputed for EVERY architecture (~45 min each)
   - Fixed by precomputing embeddings ONCE before the architecture loop
   - Added per-architecture batch sizes (Mamba: 64, Jamba: 32)

#### Memory Requirements Documented

Added comprehensive memory requirements to `docker-workflow.md`:
- 64GB RAM: Full dataset (~175K sequences) with `--max-files 30`
- 32GB RAM: Use `--max-files 15` (~87K sequences)
- 16GB RAM: Use `--max-files 8` (~46K sequences)

Formula: `sequences × 30 frames × 1204 dims × 4 bytes = ~150 bytes/frame × 30 = 4.5KB/sequence`

#### Files Changed

- `scripts/benchmark_architectures.exs` - Precompute embeddings once, filter nil opts, per-arch batch sizes
- `lib/exphil/training/data.ex` - GPU→CPU copy after embedding batches
- `lib/exphil/networks/policy.ex` - Added `:attention` backbone case
- `docs/GOTCHAS.md` - Added Gotcha #25
- `docs/docker-workflow.md` - Added memory requirements section

### 2026-01-25: Benchmark Report Enhancements

**Status:** COMPLETE

Enhanced `scripts/benchmark_architectures.exs` with comprehensive reporting and visualizations.

#### New Features

1. **Machine Detection**
   - GPU model and memory detected via `nvidia-smi`
   - Displayed in config summary and HTML report
   - Stored in JSON results for reproducibility

2. **Inference Benchmarking**
   - Measures actual inference latency per architecture
   - Runs 10 inference passes after warmup (JIT compile)
   - Reports ms/batch and μs/sample

3. **Theoretical Complexity Annotations**
   - Each architecture tagged with Big-O complexity
   - MLP: O(1), LSTM/GRU: O(L) sequential, Mamba: O(L) parallel
   - Attention: O(L²), Jamba: O(L) + O(L²/3) hybrid

4. **New Charts in HTML Report**
   | Chart | Description |
   |-------|-------------|
   | Validation Loss Comparison | Bar chart of final val loss per architecture |
   | Loss Curves | Line chart showing convergence over epochs |
   | Measured Training Speed | Batches/sec bar chart (higher is better) |
   | Measured Inference Latency | ms/batch bar chart (lower is better) |
   | Theoretical Complexity | Log-scale comparison of Big-O complexity |

5. **Enhanced Results Table**
   - Added Inference column (ms/batch)
   - Added Complexity column (O() notation)
   - Console and HTML tables match

6. **Notes Section**
   - Template for recording observations
   - Prompts for common analysis questions

#### Usage

```bash
mix run scripts/benchmark_architectures.exs \
  --replays /workspace/replays/mewtwo \
  --max-files 10 \
  --epochs 3

# Results saved to:
# - checkpoints/benchmark_results.json (with machine info)
# - checkpoints/benchmark_report.html (with all charts)
```

#### JSON Schema

```json
{
  "timestamp": "2026-01-25T12:00:00Z",
  "machine": {
    "gpu": "NVIDIA GeForce RTX 4090",
    "gpu_memory_gb": 24.0
  },
  "config": {
    "replay_dir": "/workspace/replays/mewtwo",
    "max_files": 10,
    "epochs": 3,
    "batch_size": 128,
    "train_frames": 74067,
    "val_frames": 8230
  },
  "results": [
    {
      "id": "attention",
      "name": "Attention",
      "final_val_loss": 2.921,
      "avg_batches_per_sec": 7.9,
      "inference_us_per_batch": 1234.5,
      "theoretical_complexity": "O(L²)",
      ...
    }
  ],
  "best": "attention"
}
```

### 2026-01-30: Model Evaluation Insights & Training Analysis

**Status:** COMPLETE

Comprehensive evaluation metrics added to `scripts/eval_model.exs` revealed actionable training insights.

#### Evaluation Metrics Added

| Metric | Purpose |
|--------|---------|
| **Per-button accuracy** | Break down overall accuracy by button (A, B, X, Y, Z, L, R, D-Up) |
| **Inference timing** | Measure ms/frame to verify real-time playability (<16.7ms for 60 FPS) |
| **Button prediction rates** | Compare model's predicted vs actual button press rates |
| **Loss component breakdown** | Show individual loss terms (buttons BCE, stick_x CE, stick_y CE) |
| **Stick confusion analysis** | Top N most common stick prediction errors |
| **Action distribution viz** | Show what buttons/sticks the model is using (via ActionViz) |

#### Training Insights from MLP Model (81.7% accuracy)

**Strengths:**
- Button accuracy high (86-99% per button)
- Inference fast (0.038 ms/frame = 26,315 FPS theoretical)
- Button head dominates prediction (very confident)

**Weaknesses:**
- **Stick accuracy is the main weakness** (56-68% per axis)
- **Neutral↔far confusion** is the primary stick error pattern
  - Model predicts center (8) but truth was edge (0 or 16)
  - Model predicts edge but truth was center
- **Button under-prediction** (~2% gap between predicted and actual rates)
  - Model more conservative than human players
  - Especially noticeable on rare buttons (Z, L, R)

#### Recommended Training Improvements

Based on evaluation analysis:

| Improvement | Rationale | CLI |
|-------------|-----------|-----|
| **Higher button_weight** | Address under-prediction of rare buttons | `--button-weight 2.0` |
| **Data augmentation** | Expose model to more stick positions | `--augment --mirror` |
| **K-means stick discretization** | Better bucket placement for common positions | `--kmeans-centers priv/kmeans.nx` |
| **Focal loss** | Weight rare button predictions higher | `--focal-loss --focal-gamma 2.0` |
| **Larger network** | More capacity for stick patterns | `--hidden-sizes 512,512,512` |
| **More epochs** | Allow convergence on harder patterns | `--epochs 20` |
| **Label smoothing** | Reduce overconfidence on sticks | `--label-smoothing 0.1` |

#### Stick Confusion Analysis Example

From evaluation output:
```
Top 5 Main Stick X Confusions:
  Predicted 8 → Actual 0    (12.3%)  # Neutral when should go left
  Predicted 8 → Actual 16   (10.7%)  # Neutral when should go right
  Predicted 0 → Actual 8    (8.2%)   # Left when should be neutral
  Predicted 16 → Actual 8   (7.1%)   # Right when should be neutral
  Predicted 8 → Actual 4    (5.4%)   # Neutral when should go slight left
```

This pattern suggests the model defaults to neutral (center) too often. Solutions:
1. **Focal loss on sticks** - Higher weight for non-center predictions
2. **Temporal context** - Use LSTM/Mamba to understand when movement is needed
3. **Augmentation** - Mirror states to balance left/right training data

#### Additional Training Ideas (Future Exploration)

**Stick-Focused Improvements:**

| Idea | Rationale | Effort |
|------|-----------|--------|
| **Per-bucket loss weighting** | Weight edge buckets (0, 16) higher than center (8) | Low |
| **Stick histogram equalization** | Augment dataset to flatten stick distribution | Medium |
| **Curriculum on stick complexity** | Start with cardinal directions, add diagonals | Medium |
| **Separate stick head** | Dedicated network branch for stick prediction | High |

**Calibration & Confidence:**

| Idea | Rationale | Effort |
|------|-----------|--------|
| **Temperature scaling** | Post-training calibration of confidence | Low |
| **Prediction entropy tracking** | Monitor when model is uncertain | Low |
| **Monte Carlo dropout** | Uncertainty estimation during inference | Medium |

**Architecture:**

| Idea | Rationale | Effort |
|------|-----------|--------|
| **Multi-scale temporal** | Combine MLP + LSTM heads (immediate + long-term) | High |
| **Auxiliary action prediction** | Predict next Melee action state as regularizer | Medium |
| **Cross-attention to opponent** | Explicit opponent modeling in attention | High |

**Data & Sampling:**

| Idea | Rationale | Effort |
|------|-----------|--------|
| **Class-balanced sampling** | Oversample rare stick positions in batch creation | Medium |
| **Online hard example mining** | Focus training on samples with high stick loss | Medium |
| **Action-conditional batching** | Batch by action state (aerial, grounded, etc.) | Medium |

#### Files Changed

- `scripts/eval_model.exs` - Added 6 new evaluation metrics
- `lib/exphil/training/action_viz.ex` - Fixed button list parsing bug
- `docs/GOALS.md` - Added this analysis section

---

### 2026-01-25: Batch Embedding Optimization for --no-precompute

**Status:** COMPLETE

Optimized the `--no-precompute` path to use vectorized batch embedding instead of per-frame loops, achieving ~30x speedup.

#### Problem

The `--no-precompute` flag was very slow (~16s/batch for 512 frames) because it used `Task.async_stream` with individual `Embeddings.embed()` calls per frame:
- 512 frames × ~25 Nx operations each = 12,800 tensor operations per batch
- Each `Nx.concatenate` creates intermediate tensors
- Process overhead from async tasks

#### Solution

Use `Embeddings.Game.embed_states_fast/3` which was already implemented but not used in this path:
1. **Extract all values first** (pure Elixir - very fast)
2. **Batch Nx operations** (~25 operations total for entire batch)
3. **Single GPU transfer** at the end

```elixir
# Old approach (slow): 512 separate embed() calls
Task.async_stream(game_states, fn gs -> Embeddings.embed(gs, ...) end)

# New approach (fast): single batch operation
Embeddings.Game.embed_states_fast(game_states, 1, config: embed_config)
```

#### Performance

| Approach | Time/Batch (512 frames) | Tensor Ops | Peak Memory |
|----------|------------------------|------------|-------------|
| Old (per-frame) | ~16s | ~12,800 | ~5-10MB |
| New (batch) | ~0.5s | ~25 | ~1-2MB |
| **Speedup** | **~30x** | **~500x** | **~5x better** |

#### Memory Safety

Added chunking for very large batches (>1024 frames) to limit peak memory:
```elixir
@embed_chunk_size 1024
# Chunks large batches to avoid OOM
```

For typical batch sizes (32-512), the batch approach is actually more memory-efficient because it avoids intermediate tensor allocations.

#### Files Changed

- `lib/exphil/training/data.ex` - Replaced `embed_states_parallel/3` with batch embedding

#### Research References

Based on [Nx.Serving](https://hexdocs.pm/nx/Nx.Serving.html) and [Nx.Defn](https://hexdocs.pm/nx/Nx.Defn.html) best practices:
- JIT caches based on tensor shapes - consistent batch sizes avoid recompilation
- Vectorized operations outperform per-element loops
- Extract values in Elixir first, then batch process in Nx

---

### 5. Upstream Contributions

**Impact: Medium | Benefits entire Nx ecosystem**

| Task | Target | Effort | Status |
|------|--------|--------|--------|
| **Fix orthogonal initializer for RNN shapes** | elixir-nx/axon | Medium | Planned |
| IC Tech Feature Block | ExPhil | Medium | Planned |

#### Axon Orthogonal Initializer Fix

**Problem:** Axon's `orthogonal` initializer fails with LSTM/GRU weight shapes (`[hidden, 4*hidden]`) due to QR decomposition constraints.

**Why it matters:** Orthogonal initialization is the recommended approach for RNNs (Saxe et al., 2013) - preserves gradient norms through time, preventing explosion/vanishing.

**Proposed solutions (in order of preference):**

1. **Pad-and-Slice (recommended):** Generate larger square orthogonal matrix, slice to needed shape
   - Simple, covers common failure case
   - Minimal API change

2. **Block-Orthogonal:** Initialize each LSTM gate separately as orthogonal
   - Most theoretically correct for RNNs
   - Could be `orthogonal_block(num_blocks: 4)`

3. **SVD-Based:** Use SVD instead of QR for arbitrary shapes
   - Most general but more expensive
   - Overkill for most cases

4. **Newton Iteration:** Iteratively orthogonalize random matrix
   - Approximate, non-standard
   - Doesn't require QR/SVD

**Implementation plan:**
1. Open issue on elixir-nx/axon with reproduction case
2. Submit PR for Option 1 (pad-and-slice)
3. Follow up with Option 2 if maintainers interested

**See:** [docs/AXON_ORTHOGONAL_INIT_FIX.md](AXON_ORTHOGONAL_INIT_FIX.md) for full design doc.

---

## References

- [BITTER_LESSON_PLAN.md](BITTER_LESSON_PLAN.md) - **Scaling vs hand-engineering tradeoffs**
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Detailed implementation tasks
- [RESEARCH.md](RESEARCH.md) - Prior art and papers
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training feature status
- [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) - GPU-specific optimizations
