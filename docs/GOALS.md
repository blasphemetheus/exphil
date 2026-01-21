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
| Residual MLP connections | Medium | +5-15% accuracy, enables deeper networks | Not started |
| Focal loss for rare actions | Medium | +30-50% accuracy on Z/L/R (rare buttons) | **Done** |
| Hybrid Mamba+Attention | High | Research shows hybrids can outperform pure | Not started |

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

---

## References

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Detailed implementation tasks
- [RESEARCH.md](RESEARCH.md) - Prior art and papers
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training feature status
- [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) - GPU-specific optimizations
