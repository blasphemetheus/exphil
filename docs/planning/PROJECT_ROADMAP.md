# ExPhil Project Roadmap

Comprehensive roadmap of improvements across all project areas. See also:
- [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) - Training-specific optimizations
- [TESTING.md](TESTING.md) - Test infrastructure improvements

## Critical Issues

### 1. Self-Play Dolphin Integration
**Status:** ✅ COMPLETE (GenServer infrastructure done, needs real-game testing)
**Files:** `lib/exphil/self_play/` (GameRunner, PopulationManager, ExperienceCollector, Matchmaker)
**Completed:**
- GenServer-based game runner (`game_runner.ex`)
- DynamicSupervisor for game pool (`game_pool_supervisor.ex`)
- Population management with historical policies (`population_manager.ex`)
- Experience batching and collection (`experience_collector.ex`)
- Elo ratings and matchmaking (`matchmaker.ex`, `elo.ex`)
- Training supervisor (`supervisor.ex`)

**Next steps:**
- Test with actual Dolphin (MeleePort integration)
- Run large-scale self-play training on GPU cluster
- Tune Elo K-factor and matchmaking strategies

### 2. Projectile Parsing
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/embeddings/projectile.ex`, `lib/exphil/embeddings/game.ex`
**Completed:**
- Projectile embedding (type, position, velocity, owner, damage)
- Up to 15 projectiles per frame
- Integrated into game state embedding
- 1204-dim total embedding includes projectiles

### 3. Data Pipeline Error Handling
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/training/data.ex`
**Completed:**
- `--skip-errors` (default), `--fail-fast`, `--show-errors`, `--hide-errors` flags
- `--error-log FILE` for separate error logging
- Error collection with summary at end of processing

---

## High Priority Features

### Residual MLP Architecture
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/networks/policy.ex`
**Completed:**
- `--residual` and `--no-residual` CLI flags
- Automatic projection layers when input/output dimensions differ
- Works with `--layer-norm` for ResNet-style blocks
- Enables deeper networks (4+ layers) without gradient degradation

### Focal Loss for Rare Actions
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/networks/policy.ex`
**Completed:**
- `focal_binary_cross_entropy` and `focal_categorical_cross_entropy`
- CLI flags: `--focal-loss --focal-gamma 2.0`
- Combines with label smoothing

### Embedding Caching
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/training/data.ex`, `lib/exphil/embeddings.ex`
**Completed:**
- `precompute: true` is the default
- 2-3x speedup for MLP training by embedding once instead of per-batch
- Auto-bypasses when augmentation is enabled
- `--no-precompute` flag for explicit override

---

## Robustness Improvements

### Model Architecture Mismatch Detection
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/training/training.ex`
**Completed:**
- `Training.validate_policy/2` validates checkpoint structure
- Detects temporal/MLP mismatch by checking layer names
- Detects hidden layer count mismatch
- Clear multi-line error messages with suggestions

### Training State Recovery
**Effort:** Medium | **Impact:** Crash resilience
**Files:** `lib/exphil/training/recovery.ex`
**Issue:** Resume loses optimizer state, schedule position
**Fix:**
- Save complete training state (optimizer, schedule, metrics history)
- Restore optimizer warmup position
- Document expected +5% loss spike on resume

### CLI Argument Validation
**Status:** ✅ COMPLETE
**Files:** `lib/exphil/training/config.ex`
**Completed:**
- Levenshtein distance algorithm for fuzzy matching
- Suggests corrections for typos within distance 3
- Warns about completely unrecognized flags
- Displays parsed config before training

---

## Documentation Gaps

### Self-Play Training Guide
**Status:** TODO
**Proposed:** `docs/SELF_PLAY.md`
**Contents:**
- GenServer architecture overview
- Elo rating system explanation
- Matchmaking strategies (self-play, historical, skill-based, exploiter)
- Hyperparameter tuning for self-play PPO
- Troubleshooting policy collapse
- Example configs per character

### Inference Optimization Decision Tree
**Status:** Partial in INFERENCE.md
**Add:**
- Decision tree: "Which optimization for my use case?"
- Benchmark table for CPU/GPU/mobile
- Complete Python/ONNX integration example

---

## Testing & CI

### Integration Tests
**Status:** Gap  
**Add:**
- Full training pipeline test (data → train → inference)
- ONNX export/load round-trip
- Policy loading in agent tests
- Frame buffering with temporal policies

### CI/CD Pipeline
**Status:** Missing  
**Add:** `.github/workflows/`
- Unit tests on PR (fast, BinaryBackend)
- Integration tests on release
- ONNX export validation
- Credo linting

---

## Quick Wins

| Item | Effort | Status |
|------|--------|--------|
| Increase GPU batch size presets to 256-512 | Low | Todo |
| Add early stopping warmup (don't count first N epochs) | Low | Todo |
| Add action distribution analysis script | Low | Todo |
| Hyperparameter sweep automation | Medium | Todo |
| Checkpoint streaming (don't block training) | Medium | Todo |

---

## Implementation Priority

1. ~~**Projectile parsing**~~ ✅ COMPLETE
2. ~~**Error handling**~~ ✅ COMPLETE
3. ~~**Focal loss**~~ ✅ COMPLETE
4. ~~**Residual MLP**~~ ✅ COMPLETE
5. ~~**Embedding caching**~~ ✅ COMPLETE
6. ~~**Self-play infrastructure**~~ ✅ COMPLETE (GenServers done)

**Current priorities:**
1. **Self-play Dolphin testing** - Run with real MeleePort games
2. **Large-scale self-play training** - GPU cluster deployment
3. **Training state recovery** - Save/restore optimizer state on resume
4. **Self-play documentation** - Write SELF_PLAY.md guide

**Recently completed (2026-01-23):**
- [x] `mix exphil.compare` - Side-by-side model comparison
- [x] Contextual help links - "See docs/TRAINING.md#..." in errors/warnings
- [x] Warmup indicator - Reusable JIT compilation status functions
- [x] Action distribution viz - Track/display button and stick usage
- [x] Replay quality scoring - Filter bad replays (SD chains, AFKs)
- [x] Duplicate detection - Skip duplicate files by hash (`--skip-duplicates`)
- [x] Stack trace simplification - Hide Nx/EXLA internals in errors

---

## Completed (Recent)

### Infrastructure (2026-01)
- [x] Self-play GenServer architecture (GameRunner, PopulationManager, ExperienceCollector, Matchmaker)
- [x] Elo rating system with matchmaking strategies
- [x] Projectile embedding (15 projectiles/frame, integrated into 1204-dim state)
- [x] Architecture benchmark script with per-backbone batch sizes

### Training Features (2026-01)
- [x] Focal loss for rare actions (`--focal-loss --focal-gamma 2.0`)
- [x] Residual MLP connections (`--residual`)
- [x] Embedding precomputation (2-3x speedup, enabled by default)
- [x] Data pipeline error handling (`--skip-errors`, `--error-log`)
- [x] CLI argument validation with typo suggestions

### Robustness (2026-01)
- [x] Model architecture mismatch detection
- [x] Hybrid Jamba backbone (Mamba + Attention)
- [x] K-Means stick discretization

### Testing (Previous)
- [x] Benchmark tests for performance regression
- [x] Snapshot testing for embeddings
- [x] Replay fixtures for testing
- [x] Mutation testing with Muzak
- [x] Doctest coverage for EarlyStopping
- [x] Flaky test retry mechanism
- [x] Property-based testing with StreamData

---

## Future Directions

Remaining tasks grouped by category. See also [GOALS.md](GOALS.md) for the full list with status.

### High Impact (Core Training)
| Task | Effort | Description |
|------|--------|-------------|
| Training state recovery | Medium | Resume loses optimizer state (restarts from scratch) |
| K-means stick discretization | Medium | Research shows 21 clusters beats uniform grid |
| Full pipeline integration test | Medium | Data → train → inference round-trip test |

### Medium Impact (UX/Tooling)
| Task | Effort | Description |
|------|--------|-------------|
| Checkpoint streaming | Medium | Save doesn't block training |
| Error handling improvements | Low | Bad replays fail silently |
| Model naming suggestions | Low | Warn if name conflicts with existing checkpoint |
| Auto-backup before overwrite | Low | Keep .bak of previous checkpoint |

### Low Effort Quick Wins
| Task | Effort | Description |
|------|--------|-------------|
| FPS counter in play mode | Low | Show actual vs target FPS |
| Confidence display | Low | Show model's prediction confidence |
| ONNX export/load test | Low | Verify export doesn't break |

### Research/Experimental
| Task | Effort | Description |
|------|--------|-------------|
| League system (AlphaStar-style) | High | Population-based league training |
| IC Tech Feature Block | Medium | Dedicated grab/regrab/desync features |
| Player names embedding | Low | Style-conditional imitation (currently unused) |
| Character-specific rewards | Medium | Model should discover what matters |
| Per-character models vs single model | High | Scale vs specialization experiment |
