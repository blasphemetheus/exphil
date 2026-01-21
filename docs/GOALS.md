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

**Impact: Medium | Status: Gaps in error handling**

| Task | Effort | Why | Status |
|------|--------|-----|--------|
| Architecture mismatch detection | Low | Cryptic errors when loading wrong model | Not started |
| CLI argument validation | Low | Invalid args silently use defaults | Not started |
| Training state recovery | Medium | Resume loses optimizer state | Not started |
| Checkpoint streaming | Medium | Save doesn't block training | Not started |

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

---

## References

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Detailed implementation tasks
- [RESEARCH.md](RESEARCH.md) - Prior art and papers
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training feature status
- [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) - GPU-specific optimizations
