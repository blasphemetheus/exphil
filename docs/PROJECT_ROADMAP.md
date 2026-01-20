# ExPhil Project Roadmap

Comprehensive roadmap of improvements across all project areas. See also:
- [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) - Training-specific optimizations
- [TESTING.md](TESTING.md) - Test infrastructure improvements

## Critical Issues

### 1. Self-Play Dolphin Integration
**Status:** Mock games only  
**Files:** `lib/exphil/training/self_play/league_trainer.ex`, `self_play_env.ex`  
**Issue:** Self-play trainer uses `game_type: :mock`, not actual Dolphin gameplay  
**Impact:** Cannot collect realistic RL experience  
**Fix:**
- Implement MeleePort-based game runner for self-play
- Add game state synchronization between parallel agents
- Document port/frame timing for concurrent games

### 2. Projectile Parsing Missing
**Status:** TODO stub  
**Files:** `lib/exphil/data/replay_parser.ex:326`  
**Issue:** Projectiles not parsed from replays  
**Impact:** 40-60% state information lost for projectile characters (Link, Falco, Samus)  
**Fix:**
- Extract projectile data from replay via peppi NIF
- Add projectile embedding module (type, position, velocity, owner)
- Include in state embedding

### 3. Data Pipeline Error Handling
**Status:** Generic error catch  
**Files:** `lib/exphil/training/data.ex`  
**Issue:** Malformed replays fail silently without feedback  
**Fix:**
- Detailed error messages per file (path, reason)
- `--skip-errors` flag to continue past bad files
- Separate error log for debugging

---

## High Priority Features

### Residual MLP Architecture
**Effort:** Medium | **Impact:** +5-15% accuracy  
**Files:** `lib/exphil/networks/policy.ex`  
**Current:** Plain feedforward MLP  
**Proposed:**
```elixir
# Add skip_connections: true option
Policy.build(embed_size: 1024, hidden_sizes: [256, 256], skip_connections: true)
```
**Why:** Enables deeper networks without gradient degradation

### Focal Loss for Rare Actions
**Effort:** Medium | **Impact:** +30-50% accuracy on Z/L/R  
**Files:** `lib/exphil/networks/policy.ex`  
**Issue:** Cross-entropy treats all actions equally; rare actions (<2% frequency) ignored  
**Implementation:**
```elixir
# Focal loss: (1 - p)^gamma * CE(p, y)
def focal_loss(logits, targets, gamma \\ 2.0) do
  ce = Axon.Losses.categorical_cross_entropy(logits, targets)
  pt = Nx.exp(Nx.negate(ce))
  Nx.multiply(Nx.pow(Nx.subtract(1.0, pt), gamma), ce)
end
```

### Embedding Caching
**Effort:** Medium | **Impact:** 2-3x speedup  
**Files:** `lib/exphil/training/data.ex`, `lib/exphil/embeddings.ex`  
**Current:** Embeddings computed every batch, every epoch  
**Proposed:** Precompute all embeddings before training loop  
**Status:** Partial implementation exists

---

## Robustness Improvements

### Model Architecture Mismatch Detection
**Effort:** Low | **Impact:** Better UX  
**Files:** `lib/exphil/agents/agent.ex`, `lib/exphil/training/imitation.ex`  
**Issue:** Loading policy with wrong architecture gives cryptic shape errors  
**Fix:**
- Validate architecture before loading
- Store architecture fingerprint in policy files
- Clear error: "Policy expects hidden_sizes [512,512], got [64,64]"

### Training State Recovery
**Effort:** Medium | **Impact:** Crash resilience  
**Files:** `lib/exphil/training/recovery.ex`  
**Issue:** Resume loses optimizer state, schedule position  
**Fix:**
- Save complete training state (optimizer, schedule, metrics history)
- Restore optimizer warmup position
- Document expected +5% loss spike on resume

### CLI Argument Validation
**Effort:** Low | **Impact:** Fewer misconfigurations  
**Files:** `lib/exphil/training/config.ex`  
**Issue:** Invalid args silently use defaults  
**Fix:**
- Validate option types and combinations
- Suggest did-you-mean for typos
- Display parsed config before training

---

## Documentation Gaps

### Self-Play Training Guide
**Status:** Missing  
**Proposed:** `docs/SELF_PLAY.md`  
**Contents:**
- Simple vs league mode comparison
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

1. **Projectile parsing** - Critical for character coverage
2. **Error handling** - Quick win, big debugging improvement
3. **Focal loss** - Addresses known rare-action problem
4. **Residual MLP** - Try when hitting accuracy ceiling
5. **Embedding caching** - Biggest speedup opportunity
6. **Self-play Dolphin** - Required for RL beyond imitation

---

## Completed (Recent)

- [x] Benchmark tests for performance regression
- [x] Snapshot testing for embeddings
- [x] Replay fixtures for testing
- [x] Mutation testing with Muzak
- [x] Doctest coverage for EarlyStopping
- [x] Flaky test retry mechanism
- [x] Property-based testing with StreamData
