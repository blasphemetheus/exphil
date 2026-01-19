# Training Features Design

This document outlines planned and implemented features for the ExPhil training system.

## Implemented Features ✅

### Model Naming with Memorable Names
```
checkpoints/{backbone}_{memorable_name}_{YYYYMMDD_HHMMSS}.axon
checkpoints/{backbone}_{memorable_name}_{YYYYMMDD_HHMMSS}_policy.bin
checkpoints/{backbone}_{memorable_name}_{YYYYMMDD_HHMMSS}_config.json
```

Example: `mamba_wavedashing_falcon_20260119_123456.axon`

Memorable names combine adjectives with nouns from three categories:
- General (50%): brave, cosmic, swift + falcon, phoenix, dragon
- Melee tech (30%): wavedashing, multishining + tipper, sweetspot, shine
- Hardware/mods (20%): notched, phob, rollbacked + goomwave, rectangle, slippi

### Training Presets ✅

**CPU Presets (No GPU Required):**
```bash
--preset quick      # 1 epoch, 5 files - code testing
--preset standard   # 10 epochs, 50 files - balanced w/ augmentation
--preset full_cpu   # 30 epochs, 200 files - max CPU quality w/ EMA
```

**GPU Presets (Requires CUDA/ROCm):**
```bash
--preset gpu_quick     # 3 epochs, 20 files - GPU validation
--preset gpu_standard  # 20 epochs, Mamba - standard w/ all features
--preset full          # 50 epochs, Mamba - high quality
--preset production    # 100 epochs, Mamba - max quality w/ cosine_restarts
```

**Character Presets (Built on production):**
```bash
--preset mewtwo       # 90-frame window for teleport recovery
--preset ganondorf    # 60-frame window for spacing
--preset link         # 75-frame window for projectiles
--preset gameandwatch # 45-frame window (no L-cancel)
--preset zelda        # 60-frame window for transform
```

**Preset Feature Matrix:**

| Feature | quick | standard | full | production |
|---------|-------|----------|------|------------|
| Augmentation | - | ✓ | ✓ | ✓ |
| Label Smoothing | - | 0.05 | 0.1 | 0.1 |
| EMA | - | - | ✓ | ✓ |
| LR Schedule | constant | cosine | cosine | cosine_restarts |
| Val Split | - | 0.1 | 0.1 | 0.15 |
| Early Stopping | - | ✓ | ✓ | ✓ |
| Warmup Steps | - | - | 1000 | 2000 |
| Grad Accum | - | - | 2 | 4 |

### Config Validation ✅
Full validation with helpful error messages and warnings.

### Early Stopping ✅
```bash
--early-stopping --patience 5 --min-delta 0.01
```

### Training Resumption ✅
```bash
--resume checkpoints/model.axon
```
Saves and restores: model weights, optimizer state, step counter, config.

### Best Model Checkpointing ✅
```bash
--save-best  # Saves when val_loss improves
```

### Learning Rate Scheduling ✅
```bash
--lr 1e-4 --lr-schedule cosine --warmup-steps 1000 --decay-steps 10000
```
Schedules: constant, cosine, cosine_restarts, exponential, linear

### Cosine Annealing with Warm Restarts ✅
```bash
--lr-schedule cosine_restarts --restart-period 1000 --restart-mult 2
```
SGDR (Stochastic Gradient Descent with Warm Restarts):
- LR decays following cosine curve, then "restarts" to max
- Periods grow geometrically: T_0, T_0*T_mult, T_0*T_mult^2, ...
- Helps escape local minima by periodically increasing learning rate

### Gradient Clipping ✅
```bash
--max-grad-norm 1.0  # Clip by global norm (default: 1.0, 0 = disabled)
```
Prevents gradient explosion during training:
- Clips gradients by global L2 norm before optimizer update
- Essential for stable training with large learning rates
- Default 1.0 is good for most cases, reduce to 0.5 for instability

### Gradient Accumulation ✅
```bash
--batch-size 32 --accumulation-steps 4  # Effective batch = 128
```

### Validation Split ✅
```bash
--val-split 0.1  # 10% validation, 90% training
--val-split 0.0  # No validation (default)
```

### Model EMA ✅
```bash
--ema --ema-decay 0.999
```
Exponential Moving Average (EMA) of model weights:
- Maintains shadow weights: `ema = decay * ema + (1 - decay) * current`
- EMA weights often generalize better than raw training weights
- Typical decay values: 0.999 (standard), 0.9999 (slower), 0.99 (faster)
- Saves EMA weights alongside model checkpoint for inference

### Learning Rate Finder ✅
```bash
mix run scripts/find_lr.exs --replays /path/to/replays
mix run scripts/find_lr.exs --min-lr 1e-8 --max-lr 10 --num-steps 200
```
Automatically discover optimal learning rate:
- Exponentially increases LR over ~100 steps while tracking loss
- Suggests LR where loss is decreasing fastest
- Saves hours of manual hyperparameter tuning

### Replay Scanner ✅
```bash
mix run scripts/scan_replays.exs --replays /path/to/replays
mix run scripts/scan_replays.exs --replays /path/to/replays --max-files 1000
```
Analyze your replay collection:
- Shows games per character
- Identifies low-tier character availability
- Recommends best character preset based on your data
- Reports data sufficiency for training

---

## Proposed Features (Optional)

### 1. Directory Structure by Character

Organize checkpoints by character for cleaner management:

```
checkpoints/
├── mewtwo/
│   ├── mamba_20260119_123456.axon
│   ├── mamba_20260119_123456_config.json
│   └── mamba_20260119_123456_policy.bin
├── ganondorf/
│   └── mlp_20260118_234567.axon
└── registry.json
```

---

### 2. Frame Delay Augmentation ✅

Train with variable frame delays for models that work well both locally and online.

**Quick usage (recommended):**
```bash
# Add --online-robust to any preset for Slippi online compatibility
mix run scripts/train_from_replays.exs --preset production --online-robust
```

**Manual configuration:**
```bash
mix run scripts/train_from_replays.exs \
  --preset production \
  --frame-delay-augment \
  --frame-delay-min 0 \
  --frame-delay-max 18
```

**When to use:**
| Scenario | Recommended Setting |
|----------|---------------------|
| Local play only | No delay (default) |
| Online play only | `--frame-delay 18` (fixed) |
| Both local & online | `--online-robust` (variable 0-18) |

**How it works:**
- Each training sample randomly uses a delay between min and max
- State from frame `t-delay` is paired with action from frame `t`
- Model learns to both react (low delay) and predict (high delay)

See [RESEARCH.md](RESEARCH.md) for the research background.

---

## Priority Order (Remaining)

Based on research from [slippi-ai](https://github.com/vladfi1/slippi-ai), [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html), and related papers. See [RESEARCH.md](RESEARCH.md) for full details.

### High Priority (Enables RL Stage)
1. **Self-Play Infrastructure** - BEAM concurrency for parallel games
2. **Population-Based Training** - Avoid policy collapse (critical lesson from Nabla)
3. **Historical Sampling** - Play against old checkpoints for diversity

### Medium Priority (Quality Improvements)
4. ~~**Frame Delay Augmentation**~~ ✅ - `--online-robust` or `--frame-delay-augment`
5. **K-means Stick Discretization** - Research shows 21 clusters outperforms uniform grid
6. **PPO Integration** - Clipped objective, proven in fighting games

### Lower Priority (Specialization)
7. **Character-Specific Rewards** - Mewtwo recovery, Ganon spacing, etc.
8. **Curriculum Learning** - Easy opponents → hard opponents

---

## Implementation Roadmap

### Phase 1: Core Improvements ✅ DONE
- [x] Training presets (`--preset quick|standard|full`)
- [x] Validation in Config module
- [x] Character flag (`--character mewtwo`)
- [x] Early stopping (`--early-stopping`)
- [x] Training resumption (`--resume`)
- [x] Best model checkpointing (`--save-best`)
- [x] LR scheduling (`--lr-schedule`)
- [x] Memorable model naming
- [x] Gradient accumulation (`--accumulation-steps`)
- [x] Validation split (`--val-split`)

### Phase 2: Organization ✅ DONE
- [x] Model registry JSON
- [x] Registry CLI commands (`scripts/registry.exs`)
- [x] Checkpoint pruning (`--keep-best N`)
- [ ] Directory structure by character (optional)

### Phase 3: Advanced Training ✅ DONE
- [x] Data augmentation (`--augment`, mirror states + noise)
- [x] Label smoothing (`--label-smoothing`)
- [x] Model EMA (`--ema`, `--ema-decay`)
- [x] Lineage tracking (via registry parent_id)
- [x] Cosine annealing with warm restarts (`--lr-schedule cosine_restarts`)
- [x] Learning rate finder (`scripts/find_lr.exs`)
- [x] Frame delay augmentation (`--online-robust`, `--frame-delay-augment`)

### Phase 4: Self-Play & RL
See [RESEARCH.md](RESEARCH.md) for detailed background on these features.

**Infrastructure:**
- [ ] Self-play infrastructure (BEAM concurrency)
- [ ] Historical sampling (play against old checkpoints)
- [ ] Population-based training (multiple agents)
- [ ] League system (main agents + exploiters)

**RL Training:**
- [ ] PPO integration with self-play
- [ ] Character-specific reward shaping
- [ ] Curriculum learning (easy → hard opponents)

**Research-Backed Experiments:**
- [x] Frame delay augmentation (0-18 frames variable delay) ✅
- [ ] K-means stick discretization (vs uniform grid)
- [ ] Mamba vs Transformer comparison
- [ ] DAgger for distributional shift

### Phase 5: Character Specialization
- [ ] Mewtwo specialist (90+ frame context, recovery focus)
- [ ] Ganondorf specialist (spacing, punish optimization)
- [ ] Link specialist (projectile tracking)
- [ ] Multi-character model (single model, character conditioning)
