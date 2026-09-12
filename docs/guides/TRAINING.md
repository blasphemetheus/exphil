# Training Guide

Complete reference for training ExPhil models.

## Quick Start

**New to ExPhil?** Use the interactive wizard:
```bash
mix exphil.setup
```

For new runs, use the callback-based trainer:

```bash
mix run scripts/train.exs --preset quick --replays ./replays
mix run scripts/train.exs --backbone gru --replays ./replays --max-files 100 --epochs 2
mix run scripts/train.exs --help
```

Follow [contributor setup](../../CONTRIBUTING.md#development-setup) first. Run
these commands away from any active training environment. See
[current status](../planning/CURRENT_STATUS.md) for the causal-label convention,
supported BPTT evaluation modes, and checkpoint-specific deployment guidance.

## Historical trainer examples

The `train_from_replays.exs` examples below document the legacy trainer and
earlier experiments. They are retained for reference, not as the supported
quick start. The generated training flag reference later in this document
describes the current parser. Historical timing and architecture rankings do
not establish current performance.

Legacy preset examples:
```bash
# Using presets (recommended)
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration (~5 min)
mix run scripts/train_from_replays.exs --preset standard  # Balanced (~30 min)
mix run scripts/train_from_replays.exs --preset full      # Maximum quality (~2 hrs)
mix run scripts/train_from_replays.exs --preset mewtwo    # Character-specific

# Manual configuration
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 5
```

## Training Modes

### Single-Frame (Default)

Predicts actions from individual game states. Fast to train, good baseline.

```bash
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100
```

### Temporal Training

Uses sequences of frames to learn temporal patterns (combos, reactions).

```bash
# === Recommended Backbones ===

# Mamba (recommended default - best speed/accuracy balance)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# Mamba NIF (CUDA-accelerated, fastest inference)
mix run scripts/train_from_replays.exs --temporal --backbone mamba_nif

# LSTM (best accuracy, but slow - offline only)
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# === New Architectures (2026-02) ===

# Liquid Neural Networks (ODE-based, adaptive dynamics)
mix run scripts/train_from_replays.exs --temporal --backbone liquid

# RWKV-7 (O(1) memory inference)
mix run scripts/train_from_replays.exs --temporal --backbone rwkv

# Decision Transformer (goal-conditioned)
mix run scripts/train_from_replays.exs --temporal --backbone decision_transformer

# Zamba (shared attention + Mamba hybrid)
mix run scripts/train_from_replays.exs --temporal --backbone zamba

# GLA (Gated Linear Attention - fast on short sequences)
mix run scripts/train_from_replays.exs --temporal --backbone gla

# === Other Options ===

# Sliding window attention
mix run scripts/train_from_replays.exs --temporal --backbone sliding_window

# Jamba (Mamba + interleaved attention)
mix run scripts/train_from_replays.exs --temporal --backbone jamba
```

See [Architecture Guide](../reference/architectures/ARCHITECTURE_GUIDE.md) for detailed explanations of each backbone.

**Note:** `mamba_nif` requires the Rust NIF to be compiled (see `native/selective_scan_nif/`).
It's 5x faster than pure Mamba (~11ms vs ~55ms inference).

**Recommended workflow:** Train with `mamba`, infer with `mamba_nif`:
```bash
# Training (mamba has correct gradients)
mix run scripts/train_from_replays.exs --temporal --backbone mamba --checkpoint model.axon

# Inference/playing (mamba_nif is 5x faster, uses same checkpoint!)
mix run scripts/play_dolphin_async.exs --policy model.axon --backbone mamba_nif
```

The NIF breaks the computation graph (can't backprop through `Nx.to_binary`), so use pure Mamba for training. Both use identical layer names, so checkpoints are interchangeable.

**Tradeoffs:**
- Slower per-epoch (sequences larger than single frames)
- Better at learning temporal patterns
- Recommended after establishing single-frame baseline

### Character & Port Selection

When training from replay archives, the target character may be on different ports across games.

**Option 1: Fixed port (default)**
```bash
# Always learn from player 1
mix run scripts/train_from_replays.exs --player-port 1
```

**Option 2: Auto-select by character (recommended for single-character training)**
```bash
# Learn from whichever port has Mewtwo
mix run scripts/train_from_replays.exs --train-character mewtwo
```

**Option 3: Dual-port (2x data)**
```bash
# Learn from BOTH players in every game
mix run scripts/train_from_replays.exs --dual-port
```

| Mode | Use Case | Data Volume |
|------|----------|-------------|
| `--player-port N` | Know which port to train on | 1x |
| `--train-character X` | Train specific character, unknown ports | 1x (filtered) |
| `--dual-port` | Maximum data, mixed characters | 2x |

**Note:** `--dual-port` trains on all characters in the replays, not just one. Use `--train-character` for pure single-character training.

## Command-Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--replays PATH` | `./replays` | Directory containing .slp files |
| `--corpus DIR` | nil | Pre-built MmapCorpus dir (`scripts/build_corpus.exs`) — skips parse/embed, trains straight off disk; overrides `--replays`. Temporal only. RAM is O(batch), so corpus size is unbounded (built for the 4,461-game fox_il_v2 run) |
| `--epochs N` | 10 | Number of training epochs |
| `--batch-size N` | 64 | Batch size |
| `--max-files N` | nil | Limit number of replay files |
| `--player-port N` | 1 | Which player to learn from (1 or 2) |
| `--train-character CHAR` | nil | Filter to replays containing this character (files only — does NOT pick the port by itself; see `--select-character-port`) |
| `--select-character-port` | false | With `--train-character`: imitate that character's actual port per file (singles → their port, dittos → port 1). Without it the streaming loader imitates port 1 regardless (the fox_gen_v1 43%-non-fox corpus bug, E1). Streaming pipeline only |
| `--dual-port` | false | Train on BOTH players (2x data) |
| `--hidden N,N,N` | 512,512,256 | Hidden layer sizes |
| `--lr X` | 1e-4 | Learning rate (alias: `--learning-rate`) |
| `--dropout X` | 0.1 | Dropout rate |
| `--name NAME` | nil | Custom checkpoint name |
| `--preset NAME` | nil | Training preset (quick, standard, full, mewtwo) |
| `--config PATH` | nil | YAML config file path |
| `--dry-run` | false | Validate config without training |

### Error Handling

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-errors` | true | Continue past bad replay files |
| `--fail-fast` | false | Stop on first error |
| `--show-errors` | true | Show individual file errors |
| `--hide-errors` | false | Hide individual file errors |
| `--error-log PATH` | nil | Log errors to file |

### Data Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `--characters CHAR,...` | [] | Filter replays by character |
| `--stages STAGE,...` | [] | Filter replays by stage |
| `--balance-characters` | false | Weight sampling by inverse char frequency |
| `--skip-duplicates` | true | Skip duplicate replay files by hash |
| `--no-skip-duplicates` | false | Include all files even if duplicates |
| `--min-quality N` | nil | Minimum quality score (0-100) for replays |
| `--show-quality-stats` | false | Show quality distribution after filtering |

### Temporal Options

| Option | Default | Description |
|--------|---------|-------------|
| `--temporal` | false | Enable temporal training |
| `--backbone TYPE` | lstm | See backbone list below |
| `--window-size N` | 60 | Frames per sequence |
| `--stride N` | 1 | Step between sequences |
| `--truncate-bptt N` | nil | Truncated backprop (faster training) |
| `--bptt` | false | Contiguous-BPTT training: cursors walk replays in order, GRU carry flows across chunks, per-timestep loss (GRU only; see BPTT_LOADER_DESIGN.md) |
| `--unroll N` | 80 | BPTT chunk length in frames (gradient truncation horizon) |
| `--bptt-overlap N` | 1 | Frames shared between consecutive BPTT chunks (set to frame_delay + 1) |
| `--bptt-val-files N` | 16 | Whole replays held out for the carry-threaded val pass (game-level split; val batch is capped at 8 rows) |

**Available backbones (15 total):**

| Backbone | Type | 60 FPS Ready | Notes |
|----------|------|--------------|-------|
| `mamba` | SSM | Yes | Recommended default |
| `mamba_nif` | SSM | Yes | Fastest inference (requires NIF) |
| `lstm` | Recurrent | No | Best accuracy, slow |
| `gru` | Recurrent | No | Faster than LSTM |
| `attention` | Transformer | Borderline | Also: `sliding_window` |
| `jamba` | Hybrid | Borderline | Mamba + Attention |
| `zamba` | Hybrid | Yes | Shared attention + Mamba |
| `mamba_ssd` | SSM | Yes | Mamba-2 with SSD algorithm |
| `rwkv` | Linear RNN | Yes | O(1) memory inference |
| `gla` | Linear Attn | Yes | Fast on short sequences |
| `hgrn` | Gated RNN | Yes | Hierarchical patterns |
| `decision_transformer` | Transformer | Borderline | Goal-conditioned |
| `s5` | SSM | Yes | Simplified state space |
| `liquid` | Neural ODE | Yes | Adaptive dynamics |

See [Architecture Guide](../reference/architectures/ARCHITECTURE_GUIDE.md) for detailed explanations.

### Policy Type Options

| Option | Default | Description |
|--------|---------|-------------|
| `--policy-type TYPE` | autoregressive | Policy architecture type |
| `--head TYPE` | independent | Controller head: `independent` (six parallel heads) or `autoregressive` (residual-stream conditional head, buttons→main_x→main_y→c_x→c_y→shoulder; see AUTOREGRESSIVE_HEAD_PLAN.md). Temporal only |
| `--action-horizon N` | 8 | Action prediction horizon (for chunked policies) |
| `--num-inference-steps N` | 20 | Diffusion/flow matching inference steps |
| `--kl-weight N` | 10.0 | KL divergence weight for ACT (CVAE) |

**Available policy types (4 total):**

| Policy Type | Output Format | Loss Function | Notes |
|-------------|---------------|---------------|-------|
| `autoregressive` | 6 discrete heads | Cross-entropy | Default, fastest inference |
| `diffusion` | Continuous actions | MSE noise prediction | Denoising diffusion |
| `act` | Action chunks | L1 + KL (CVAE) | Temporal ensembling |
| `flow_matching` | Continuous actions | MSE velocity | ODE-based generation |

**Example: Training with Diffusion Policy:**
```bash
mix run scripts/train_from_replays.exs \
  --policy-type diffusion \
  --action-horizon 16 \
  --num-inference-steps 20 \
  --temporal --backbone mamba
```

**Example: Training with Action Chunking (ACT):**
```bash
mix run scripts/train_from_replays.exs \
  --policy-type act \
  --action-horizon 8 \
  --kl-weight 10.0 \
  --temporal --backbone mamba
```

### Mamba-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--state-size N` | 16 | SSM state dimension |
| `--expand-factor N` | 2 | Expansion factor |
| `--conv-size N` | 4 | Convolution kernel size |
| `--num-layers N` | 2 | Number of Mamba layers |

### Attention/Hybrid Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-heads N` | 4 | Number of attention heads (parsed since 2026-09-09 — it was accepted-and-ignored before, and Trainer's private 2/32 table won) |
| `--head-dim N` | 64 | Attention head width |
| `--log-file PATH` | nil | Tee script output to a file |
| `--attention-every N` | 2 | Add attention every N layers (hybrid) |
| `--qk-layernorm` | on | Normalize Q/K before attention (stabilizes training) |
| `--no-qk-layernorm` | - | Disable QK LayerNorm |
| `--chunked-attention` | off | Use chunked attention for 20-30% memory reduction |
| `--no-chunked-attention` | - | Disable chunked attention |
| `--memory-efficient-attention` | off | Use memory-efficient attention (true O(n) memory via online softmax) |
| `--no-memory-efficient-attention` | - | Disable memory-efficient attention |
| `--chunk-size N` | 32 | Chunk size for chunked/memory-efficient attention |
| `--flash-attention-nif` | off | Use FlashAttention NIF for inference (forward-only, Ampere+ GPU) |
| `--no-flash-attention-nif` | - | Disable FlashAttention NIF |

### Liquid Neural Network Options

| Option | Default | Description |
|--------|---------|-------------|
| `--solver TYPE` | rk4 | ODE solver: euler, midpoint, rk4, dopri5 |
| `--integration-steps N` | 1 | ODE sub-steps per frame |

**Solver choice:**
- `euler` - Fastest, least accurate
- `midpoint` - Good balance for speed
- `rk4` - **Default**, good accuracy
- `dopri5` - Best accuracy, adaptive step size

See [ODE Solver Reference](../reference/ODE_SOLVER.md) for details.

### Other New Architecture Options

These options apply to multiple new architectures:

| Option | Default | Applies To | Description |
|--------|---------|------------|-------------|
| `--expand-ratio N` | 2 | hgrn, gla | State expansion ratio |
| `--attention-interval N` | 3 | zamba | Mamba blocks between attention |

### Model Architecture

| Option | Default | Description |
|--------|---------|-------------|
| `--optimizer TYPE` | adam | adam, adamw, lamb, radam |
| `--layer-norm` | false | Enable layer normalization (MLP) |
| `--no-layer-norm` | - | Disable layer normalization |
| `--residual` | false | Enable residual connections (MLP) |
| `--no-residual` | - | Disable residual connections |

### Training Features

| Option | Default | Description |
|--------|---------|-------------|
| `--val-split X` | 0.0 | Validation split (0.1 = 10%) |
| `--accumulation-steps N` | 1 | Gradient accumulation steps |
| `--lr-schedule TYPE` | constant | cosine, linear, exponential |
| `--warmup-steps N` | 1 | Learning rate warmup steps |
| `--decay-steps N` | nil | Steps for LR decay |
| `--restart-period N` | 1000 | Cosine annealing restart period (T_0) |
| `--restart-mult N` | 2 | Restart period multiplier (T_mult) |
| `--max-grad-norm X` | 1.0 | Gradient clipping norm (0 = disabled) |
| `--early-stopping` | false | Enable early stopping |
| `--patience N` | 5 | Epochs without improvement before stopping |
| `--min-delta X` | 0.01 | Minimum improvement to count as progress |
| `--save-best` | true | Save model when val_loss improves |
| `--save-every N` | nil | Save checkpoint every N epochs |
| `--save-every-batches N` | nil | Save checkpoint every N batches (for streaming) |
| `--resume PATH` | nil | Resume from checkpoint. If the checkpoint's controller head differs from `--head`, the TRUNK is transplanted: matching non-head params load, head + optimizer start fresh, config/step keep the trainer's |
| `--reinit-head` | false | With `--resume`: re-initialise the controller head from scratch while loading the trunk (the "new head params" control, AUTOREGRESSIVE_HEAD_PLAN item 9) |
| `--precision TYPE` | f32 | f32 or bf16 (FP32 is 2x faster due to XLA issues) |
| `--mixed-precision` | false | FP32 master weights + BF16 compute (not recommended) |
| `--frame-delay N` | 0 | Reaction delay (frames) on top of the causal pairing Peppi emits; 0 = causal. See the label-convention row below. |
| `--stream-chunk-size N` | nil | Load N files at a time (memory-bounded) |
| `--pipeline-chunks` | true | Prepare next chunk while training (overlaps CPU/GPU) |
| `--no-pipeline-chunks` | - | Disable chunk pipelining (sequential processing) |
| `--cache-streaming` | true | Cache chunk embeddings to disk (reuse across epochs) |
| `--no-cache-streaming` | - | Disable streaming cache |
| `--gc-every N` | 100 | Run garbage collection every N batches (0=disabled) |

### Data Augmentation

| Option | Default | Description |
|--------|---------|-------------|
| `--augment` | false | Enable data augmentation |
| `--mirror-prob X` | 0.5 | Mirror augmentation probability |
| `--noise-prob X` | 0.3 | Noise augmentation probability |
| `--noise-scale X` | 0.01 | Noise magnitude |
| `--cache-augmented` | false | Precompute augmented variants (~100x speedup) |
| `--num-noisy-variants N` | 2 | Number of noisy variants to precompute |
| `--label-smoothing X` | 0.1 | Label smoothing (prevents overconfidence) |
| `--focal-loss` | false | Enable focal loss for rare actions |
| `--prev-action` | false | Condition on previous frame's controller (training embeds frame i-1's inputs; live agent feeds back its own outputs). Enables frame-precise input sequences (dash dance, multishine). Regime is stored in the policy config — old checkpoints keep zeros. Not yet wired for --streaming. |
| `--prev-action-dropout P` | 0.0 | Zero the prev-action channel on fraction P of training frames (exposure-bias mitigation: live, the model feeds back its own outputs, which drift from teacher-forced ground truth). Try 0.1–0.3 with --prev-action. Mask is baked into the embedding cache entry — use --no-cache for a fresh mask. |
| `--scheduled-sampling P` | 0.0 | Exposure bias: on fraction P of samples, replace the LAST window position's prev-action slice with the model's OWN decoded prediction (`ExPhil.Training.ScheduledSampling`; decode pinned to the live path). Requires `--temporal` and `--prev-action`. One extra forward pass per step. Loss under this flag is a harder objective — never compare loss curves across it; judge by live runs. |
| `--ss-ramp N` | 10 | Ramp scheduled sampling 0 → P linearly over the first N epochs (drill loop; the main pipeline currently applies P flat). |
| `--action-delay N` | 0 | Standard-path twin of `--frame-delay`: reaction delay on top of the causal pairing (state[t] -> the input issued from it). 0 = causal. Composes with --prev-action. |
| `--mix-frames SPEC` | nil | Curriculum mixing: comma/glob list of drill `.frames` exports (`scripts/export_drill_frames.exs`) concatenated into training. Drill frames must be exported at the SAME `--action-delay` (the prev-action channel misaligns otherwise; a warning fires). Changes the embedding cache key. Replay mode only — corpus mode ignores it (use `--mix-corpus`). |
| `--mix-corpus DIR` | nil | Corpus-mode curriculum mixing: a snippet mini-corpus (`scripts/build_snippet_corpus.exs`, one corpus file per snippet so windows never cross snippet boundaries) whose batches are interleaved evenly into the `--corpus` training stream. Embed sizes must match. Mix files are all-train (val stays comparable to unmixed baselines). |
| `--mix-oversample N` | 1 | Passes of the mix corpus interleaved per epoch. The mix is typically a tiny fraction of the main corpus — oversample to give corrections a meaningful gradient share (e.g. 20 ≈ a few percent for a 41M-frame corpus with ~15k mix frames). |
| `--per-stage-ledge` | off | Task #25: use the real per-stage edge x (`Melee.Stages.edge_ground_position`) in the ledge-distance feature instead of the historical 85-everywhere constant (which reads "safe" at x=60 on YS when the player is offstage). Changes the embedding VALUES: existing checkpoints and corpora are calibrated to the constant, so this is for fresh v3-edge arms only, and corpus-mode training needs a corpus REBUILT with the same flag (recorded in corpus meta). |
| `--focal-gamma X` | 2.0 | Focal loss gamma (higher = focus on hard) |
| `--button-weight X` | 2.0 | Multiply button loss (fixes under-prediction) |
| `--stick-edge-weight X` | nil | Weight edge stick buckets higher (try 2.0-3.0) |
| `--neutral-weight X` | 0.25 | Per-frame loss weight for neutral (no-input) frames; action frames get 1.0. Blanket anti-passivity knob — 1.0 = unweighted (the 09-05 clean-loss arm), which raised idle 6.5x corpus at v2 scale. |
| `--transition-weight X` | nil | Per-frame loss weight for DECISION frames (controller differs from the previous frame): `max(weight, X)`. Targets *when* to change action (leaving WAIT, committing) instead of downweighting all neutral frames. Flag added 2026-09-07 (was pipeline/drill-only). |
| `--offstage-weight X` | nil | (bptt path) Per-frame loss weight for OFFSTAGE frames (subject airborne beyond the stage ledge): `max(weight, X)`. Rare-state coverage for recovery — offstage is rare in expert play, so the model gets few reps where it fails. Added 2026-09-08. |
| `--frame-delay N` (training) | 0 | **Label convention (INVARIANTS.md item 1, GOTCHA #113).** Slippi records each input on the frame whose state it produced, so the raw same-frame pair is leaked. Since 2026-09-09 `Peppi.to_training_frames` ALWAYS pairs state[t] with the input issued from it (raw controller[t+1]); N is reaction delay on top, so N=0 is the causal pairing and the leak cannot be built. Deploy law: a policy trained at reaction k plays at live `--frame-delay k+1` (`ExPhil.Data.LabelConvention`; legacy unstamped checkpoints counted delay d = reaction d-1, and the Agent translates). |

**Augmented Embedding Cache (Recommended)**

Use `--cache-augmented` to precompute augmented embedding variants for ~100x speedup:

```bash
# Fast augmented training (recommended)
mix run scripts/train_from_replays.exs \
  --augment --cache-augmented \
  --num-noisy-variants 2

# More variety with additional noisy variants
mix run scripts/train_from_replays.exs \
  --augment --cache-augmented \
  --num-noisy-variants 4
```

This precomputes multiple versions of each frame (original, mirrored, noisy variants) and randomly selects among them during training, providing similar regularization to on-the-fly augmentation.

> **Note:** Without `--cache-augmented`, `--augment` applies augmentation on-the-fly which is ~100x slower.
> See [Gotcha #40](../reference/GOTCHAS.md#40---augment-flag-bypasses-precomputed-embeddings-100x-slower) for details.

### Online Play Training

> **2026-08-03: the 18-frame target is retired.** The delay campaign
> (LATENCY_ARCHITECTURE.md) found realistic Slippi Direct = 2-4 frame
> buffer + intrinsic +2, SS-on-queue doesn't ladder past d4, and the
> production approach is the DRILL trainer's multi-delay {2,3} +
> queue-as-input + SS-on-queue recipe (below), not label-shift flags.

| Option | Default | Description |
|--------|---------|-------------|
| `--online-robust` | false | Enable online play training mode (legacy path) |
| `--stage-internals` | false | Add FoD platform heights + PS transformation to the embedding (+7 raw dims, zero-gated by stage; W4 2026-08-24 stage-blindness verdict) |
| `--frame-delay-augment` | false | Enable frame delay augmentation |
| `--frame-delay-min N` | 0 | Minimum delay frames (local play) |
| `--frame-delay-max N` | 18 | Maximum delay frames (legacy; see note above) |

### Drill trainer flags (`dagger_drill.exs` / `train_multishine_policy.exs`)

The delay-campaign and interp flags (2026-07/08). Both scripts unless noted.

| Option | Default | Description |
|--------|---------|-------------|
| `--multi-delay "2,3"` | off | Train one policy across delay rungs; per-frame delay-id tags (drill only). {2,3} is the production recipe |
| `--pipeline-offset N` | 0 | Label shift = delay + N; the measured live pipeline is +2 (drill only) |
| `--queue-depth K` | 1 | Queue-as-input: last K decoded actions as input channels (drill only) |
| `--with-delay-id` | false | Delay-id one-hot input (size 9, ids 0-8; never deploy at an untrained id — override instead) |
| `--scheduled-sampling P` + `--ss-ramp N` | off | SS-on-queue: self-sample the queue channels; THE exposure-bias fix. Never combine with `--shift-jitter` |
| `--snippet-frames GLOB` | off | Mix pre-relabeled snippet files from `snippet_mine.exs` (drill only) |
| `--probe-basin` | **ON** | Per-epoch basin mental rollout + (multishine script) fixture margins to `<out>.basin_probe.jsonl`; `--no-probe-basin` disables |
| `--probe-entries GLOB` | g+m routes | Dead-seed replays as auto-detected absorbed probe entries |
| `--reject-at N` / `--reject-on basin\|margin\|either` | off | Early-reject: halt WITHOUT export (exit 3 multishine / 6 drill) when the probe criterion holds at epoch >= N |
| `--select-by margin` + `--post-converge N` | loss/0 | Export the fattest-margin probe epoch; keep training N epochs past the loss bar (multishine script only) |

### Monitoring

| Option | Default | Description |
|--------|---------|-------------|
| `--wandb` | false | Enable Weights & Biases logging |
| `--wandb-project NAME` | exphil | W&B project name |
| `--wandb-name NAME` | nil | W&B run name (auto-generated if nil) |

### Verbosity & Reproducibility

| Option | Default | Description |
|--------|---------|-------------|
| `--quiet` | false | Minimal output (errors only), suppresses XLA/ptxas logs |
| `--verbose` | false | Debug output (timing, memory) |
| `--log-interval N` | 100 | Progress bar update frequency (every N batches) |
| `--seed N` | random | Random seed for reproducibility |

### Checkpoint Safety

| Option | Default | Description |
|--------|---------|-------------|
| `--overwrite` | false | Allow overwriting existing checkpoints |
| `--no-overwrite` | false | Fail if checkpoint exists |
| `--backup` | true | Create .bak before overwrite |
| `--no-backup` | false | Skip backup creation |
| `--backup-count N` | 3 | Number of backup versions to keep |

### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--precompute` | true | Precompute embeddings (2-3x speedup) |
| `--no-precompute` | - | Disable embedding precomputation |
| `--prefetch` | true | Prefetch batches while GPU trains |
| `--no-prefetch` | - | Disable batch prefetching |
| `--prefetch-buffer N` | 2 | Number of batches to prefetch |
| `--gradient-checkpoint` | false | Trade memory for compute |
| `--val-concurrency N` | 4 | Parallel validation batches (1=sequential) |
| `--profile` | false | Enable timing profiler (report at end) |
| `--mmap-embeddings` | false | Use memory-mapped embeddings (for datasets > RAM) |
| `--mmap-path PATH` | nil | Custom path for mmap file (auto-generated if not set) |
| `--auto-batch-size` | false | Auto-tune batch size for optimal GPU utilization |
| `--auto-batch-min N` | 32 | Minimum batch size to test |
| `--auto-batch-max N` | 4096 | Maximum batch size to test |
| `--auto-batch-backoff X` | 0.8 | Safety factor (0.8 = 20% headroom) |

> **Note:** `--prefetch` only has effect when used with `--stream-chunk-size`. In standard (non-streaming) mode, prefetching is disabled due to EXLA tensor process limitations. A warning is shown if you use `--prefetch` without streaming mode.
| `--checkpoint-every N` | 1 | Checkpoint every N layers |

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ema` | false | Enable model EMA |
| `--ema-decay X` | 0.999 | EMA decay rate |
| `--no-register` | false | Skip model registry |
| `--keep-best N` | nil | Keep best N checkpoints (prune others) |
| `--kmeans-centers PATH` | nil | K-means cluster centers for sticks |

### Embedding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--stage-mode MODE` | compact | Stage embedding: full, compact, learned |
| `--action-mode MODE` | learned | Action embedding: one_hot (399 dims) or learned (64-dim trainable) |
| `--character-mode MODE` | learned | Character embedding: one_hot (33 dims) or learned (64-dim trainable) |
| `--nana-mode MODE` | compact | Ice Climbers Nana: compact (39), enhanced (14+ID), full (449) |
| `--jumps-normalized` | true | Jumps as 1 normalized dim (false = 7-dim one-hot) |
| `--num-player-names N` | 112 | Player name dims (0 to disable) |
| `--learn-player-styles` | false | Enable style-conditional training |
| `--player-registry PATH` | nil | Save/load player registry JSON |
| `--min-player-games N` | 1 | Min games for player to be in registry |

**Embedding dimension reduction:**
Using learned embeddings dramatically reduces input dimensions:
- `--action-mode learned` saves ~670 dims (399×2 players → 2 action IDs)
- `--character-mode learned` saves ~64 dims (33×2 players → 2 char IDs)
- `--stage-mode compact` saves 57 dims (64 → 7)
- `--nana-mode enhanced` optimizes IC handling with action ID

**Note:** These optimized settings are now the **default**. Total embedding is 287 dims (vs 1204 with all one-hot). You only need to specify these flags if you want to change back to one-hot mode:

```bash
# Use one-hot mode (larger embeddings, slower training)
mix run scripts/train_from_replays.exs \
  --action-mode one_hot \
  --character-mode one_hot \
  --stage-mode full
```

### Player Style Learning

Style-conditional training learns player-specific playstyles by embedding player tags from replays. This allows the model to learn "how Plup plays Sheik" vs "how Jmook plays Sheik" - same character, different tendencies.

**Usage:**
```bash
# Enable style learning (builds registry from replay tags)
mix run scripts/train_from_replays.exs --learn-player-styles --temporal --backbone mamba

# Save registry for later use or inspection
mix run scripts/train_from_replays.exs --learn-player-styles --player-registry players.json

# Reuse saved registry (ensures consistent player IDs across runs)
mix run scripts/train_from_replays.exs --learn-player-styles --player-registry players.json
```

**How it works:**
1. Scans replay metadata for player tags (e.g., "Plup", "Jmook", "Mango")
2. Assigns each unique player a numeric ID (0 to num_player_names-1)
3. Embeds player ID as one-hot vector during training
4. Model learns player-specific tendencies through this conditioning

**Options:**
- `--min-player-games N` - Only include players with N+ games (filter rare tags)
- `--num-player-names N` - Max unique players (overflow uses hash bucketing)

**Use cases:**
- Train a model that can mimic specific player styles
- Analyze playstyle differences between players
- Condition generation: "play aggressively like Mango"

### K-means Stick Discretization

By default, stick positions are discretized into 17 uniform buckets (0-16). K-means clustering learns the actual distribution from replay data, placing more cluster centers where human inputs concentrate (cardinal/diagonal positions, specific angles for techniques).

**Benefits:**
- ~5% accuracy improvement on rare but important inputs (wavedash angles, shield drops)
- Better coverage of deadzone boundaries
- Character-specific patterns captured

**Usage:**
```bash
# Step 1: Train K-means centers from your replays
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx

# Step 2: Use centers during training
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx --temporal --backbone mamba
```

The default 21 clusters matches slippi-ai's research findings. You can also save to JSON for inspection:
```bash
cat priv/kmeans_centers.json
```

## Presets

```bash
--preset quick      # 1 epoch, 5 files, small MLP - fast iteration
--preset standard   # 10 epochs, 50 files - balanced
--preset full       # 50 epochs, Mamba, temporal - maximum quality
--preset mewtwo     # Character-optimized for Mewtwo
```

## Examples

### Full Training Run

```bash
mix run scripts/train_from_replays.exs \
  --replays /path/to/replays \
  --temporal --backbone mamba \
  --hidden 256 --window-size 60 \
  --epochs 20 --batch-size 64 \
  --lr 1e-4 --lr-schedule cosine --warmup-steps 1000 \
  --val-split 0.1 --early-stopping --patience 5 \
  --save-best --wandb --wandb-project exphil
```

### Memory-Constrained Training

Reduce model/batch size for limited GPU memory:

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 128 --window-size 30 \
  --num-layers 2 --batch-size 32 \
  --max-files 5 --epochs 3
```

Or use streaming for large datasets that don't fit in RAM (see [Streaming Data Loading](#streaming-data-loading)):

```bash
mix run scripts/train_from_replays.exs \
  --stream-chunk-size 30  # Process 30 files at a time
```

### Faster Training with Truncated BPTT

Limits how far gradients flow back through time:

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone lstm \
  --window-size 60 --truncate-bptt 20
```

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| Full BPTT | 1x | Best | Final training |
| `--truncate-bptt 30` | ~1.5x | Good | Balanced |
| `--truncate-bptt 20` | ~2x | Moderate | Prototyping |
| `--truncate-bptt 10` | ~3x | Lower | Quick experiments |

### Training with Frame Delay (for online play)

```bash
# Production delay recipe (2026-08-03): drill trainer, multi-delay {2,3}
# + queue-as-input + SS-on-queue — covers d2-d4 with one checkpoint
# (d4 via --delay-id-override 3 at play time)
mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "..." --multi-delay "2,3" --pipeline-offset 2 \
  --queue-depth 4 --with-delay-id --scheduled-sampling 0.5 --ss-ramp 10
```

| Connection | Effective delay | Notes |
|------------|-----------------|-------|
| Local / loopback Direct | 4-5 (frame-delay 2-3 + intrinsic 2) | Measured sharp, not a smear |
| Realistic remote Direct | ~4-6 | 2-4 frame Slippi buffer + intrinsic 2 |
| 18+ (old Phillip target) | — | RETIRED: SS-on-queue doesn't ladder past d4; a ranked-tier fork would need its own embedding layout |

### Gradient Accumulation

Simulate larger batch sizes on limited memory:

```bash
# Effective batch size = 32 * 4 = 128
mix run scripts/train_from_replays.exs \
  --batch-size 32 --accumulation-steps 4 --epochs 10
```

### Streaming Data Loading

Train on large datasets without running out of memory by loading files in chunks:

```bash
# Process 30 files at a time (good for 56GB RAM)
mix run scripts/train_from_replays.exs \
  --temporal --backbone jamba \
  --train-character mewtwo \
  --stream-chunk-size 30 \
  --epochs 20
```

**How it works:**
1. Files are split into chunks of N files each
2. For each epoch, iterate through all chunks
3. Each chunk: parse → embed → train → free memory → next chunk

| Chunk Size | RAM Usage | Speed | Use Case |
|------------|-----------|-------|----------|
| `30` | ~20GB | 1x | Standard GPU pods (56GB RAM) |
| `50` | ~35GB | ~1.1x | High-memory machines |
| `100` | ~70GB | ~1.15x | Large RAM servers |
| nil (default) | All data | Fastest | When data fits in RAM |

**Trade-offs:**
- ~10-20% slower due to repeated file I/O each epoch
- Validation uses training loss as proxy (no separate val set in streaming mode)
- Memory bounded by chunk size, not total dataset size

**Prefetching works with streaming mode** (fixed in commit 72e4c8f):
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone jamba \
  --stream-chunk-size 30
  # Prefetching enabled by default, works correctly
```

The prefetcher eagerly buffers batches from lazy chunk streams.

> **Why prefetch only works in streaming mode:** EXLA tensors cannot safely cross Erlang process boundaries. The streaming prefetcher uses a producer process that buffers Elixir data structures (not tensors), allowing true async batch preparation. In non-streaming mode, batches contain pre-computed EXLA tensors, so prefetching would require unsafe tensor transfers between processes.

**Automatic optimizations in streaming mode:**
- Precompute is auto-disabled (embeddings computed on-the-fly)
- This is intentional: precomputing per-chunk then discarding is wasteful
- On-the-fly embedding is faster for streaming since chunks aren't reused

**Recommended chunk sizes by RAM:**
- 32GB RAM: `--stream-chunk-size 15`
- 56GB RAM: `--stream-chunk-size 30`
- 128GB RAM: `--stream-chunk-size 60` or no streaming

## Memory Management

Training can consume significant RAM and GPU memory. This section covers how to prevent OOM errors and optimize memory usage.

### Quick Reference: When to Use Streaming

| Total Frames | RAM Needed (approx) | Recommendation |
|--------------|---------------------|----------------|
| < 200K | ~2 GB | Standard mode OK |
| 200K - 500K | 2-8 GB | Standard usually OK, streaming safer |
| 500K - 1M | 8-20 GB | Use `--stream-chunk-size 20000` |
| > 1M | 20+ GB | **Always** use `--stream-chunk-size 10000-20000` |

**Rule of thumb:** If the script shows "Total training frames: X" where X > 500K, add `--stream-chunk-size 20000` to avoid OOM during embedding.

The embedding step happens before training starts. If your terminal becomes unresponsive at "Embedding: N%" with memory at 100%, the dataset is too large for available RAM.

### Memory Usage Patterns

**Where memory goes during training:**
1. **Replay data** - Parsed frames loaded into RAM
2. **Embeddings** - Pre-computed state embeddings (if `--precompute`)
3. **Model parameters** - Network weights and optimizer state
4. **Batch tensors** - Current batch on GPU
5. **Gradient tensors** - Backprop intermediate values

### Symptoms of Memory Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| SSH freezes, can't connect | System RAM exhausted | Use `--stream-chunk-size`, reduce `--max-files` |
| Training crashes with OOM | GPU VRAM exhausted | Reduce `--batch-size` (BF16 saves VRAM but is slower) |
| Training gets progressively slower | Memory leak / fragmentation | Use `--gc-every 50` |
| Swap usage increasing | RAM pressure | Use streaming mode, reduce batch size |

### Memory Optimization Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--stream-chunk-size N` | nil | Load N files at a time instead of all |
| `--gc-every N` | 100 | Run garbage collection every N batches |
| `--batch-size N` | 64 | Smaller = less GPU memory |
| `--precision bf16` | f32 | Half precision uses ~50% less VRAM (but 2x slower) |
| `--no-precompute` | false | Compute embeddings on-the-fly (saves RAM) |

### Embedding Disk Cache

Embedding precomputation can take 1+ hours for large datasets. Enable disk caching to reuse embeddings across runs.

**Cache flags (consistent across all scripts):**

| Flag | Default | Effect |
|------|---------|--------|
| `--cache-embeddings` | false | Enable disk caching of embeddings |
| `--cache-augmented` | false | Cache augmented variants (original + mirrored + noisy) |
| `--num-noisy-variants N` | 2 | Number of noisy variants to cache |
| `--cache-dir PATH` | `cache/embeddings` | Cache directory |
| `--no-cache` | false | Ignore existing cache and recompute |

#### Training Script Examples

```bash
# Enable embedding cache (saves ~1 hour on re-runs)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --temporal --backbone mlp

# Enable AUGMENTED embedding cache (~100x speedup for --augment)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-augmented --augment \
  --num-noisy-variants 2

# Custom cache directory
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --cache-dir /workspace/cache/embeddings

# Force recompute even if cache exists
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --no-cache
```

**Cache Types:**
- `--cache-embeddings`: Caches single embedding per frame. Incompatible with `--augment`.
- `--cache-augmented`: Caches multiple variants per frame (original, mirrored, noisy). **Compatible with `--augment`** - this is the recommended way to use augmentation.

**Incompatible flags:**
- `--cache-embeddings` + `--augment`: Use `--cache-augmented` instead for fast augmentation
- `--cache-embeddings` + `--no-precompute`: Nothing to cache without precomputation
- `--cache-embeddings` + `--stream-chunk-size`: Streaming mode processes chunks on-the-fly, cache not used

#### Benchmark Script Examples

```bash
# Enable embedding cache
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings

# Custom cache directory
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings --cache-dir /workspace/my_cache

# Force recompute even if cache exists
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings --no-cache
```

**Cache key is based on:**
- Replay file list (sorted paths)
- Embedding config (action_mode, character_mode, stage_mode, etc.)
- Window size and stride (for temporal embeddings)

**Managing cache:**
```elixir
# List cached embeddings
ExPhil.Training.EmbeddingCache.list()

# Clear all cached embeddings
ExPhil.Training.EmbeddingCache.clear()

# Invalidate specific cache
ExPhil.Training.EmbeddingCache.invalidate("cache_key_here")
```

### Recommended Configurations by Hardware

**RunPod RTX 4090 (24GB VRAM, ~50GB RAM):**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden-sizes 512,256 \
  --stream-chunk-size 20000 \
  --gc-every 50 \
  --batch-size 512 \
  --seq-len 64 \
  --save-best \
  --name mamba_training \
  2>&1 | tee /workspace/logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Local GPU (8-12GB VRAM):**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --stream-chunk-size 15 \
  --gc-every 50 \
  --batch-size 32
# Add --precision bf16 if VRAM constrained (slower but uses 50% less memory)
```

**CPU-only (32GB RAM):**
```bash
mix run scripts/train_from_replays.exs \
  --backbone mlp \
  --stream-chunk-size 10 \
  --gc-every 100 \
  --batch-size 128 \
  --no-precompute
```

### Monitoring Memory During Training

**System RAM:**
```bash
# Watch memory usage
watch -n 1 'free -h'

# Or use htop
htop
```

**GPU Memory (NVIDIA):**
```bash
# Watch GPU memory
watch -n 1 'nvidia-smi'

# Or continuous monitoring
nvidia-smi -l 1
```

**From Elixir (in IEx):**
```elixir
# System memory
:erlang.memory(:total) / 1_000_000  # MB

# Force garbage collection
:erlang.garbage_collect()
```

### Troubleshooting Memory Issues

**If SSH freezes during training:**
1. The pod is likely out of system RAM
2. Use RunPod web terminal (Connect → Web Terminal) instead
3. For future runs, use `--stream-chunk-size` to bound memory

**If training crashes with CUDA OOM:**
1. Reduce `--batch-size` (try halving it)
2. Try `--precision bf16` (uses 50% less VRAM, but 2x slower due to XLA issues)
3. For temporal models, reduce `--window-size`

**If memory grows over time:**
1. Enable `--gc-every 50` for more frequent garbage collection
2. This is normal for long runs - Erlang's GC is generational
3. The `--gc-every` flag forces full collection periodically

### Technical Details

**Why streaming helps:**
- Without streaming: All replay files are parsed and loaded into RAM before training starts
- With streaming: Files are loaded in chunks, processed, then freed

**Why `--gc-every` helps:**
- BEAM's garbage collector is per-process and generational
- Long-running training can accumulate garbage in older generations
- Periodic `:erlang.garbage_collect()` forces full collection
- Default of 100 batches balances memory vs overhead (~1ms per GC)

**Prefetcher memory fix (v6911c16):**
- Previous versions materialized ALL batches into memory before training
- Now uses lazy streaming - only current batch is in memory
- This alone can reduce RAM usage by 50%+ for large datasets

## Performance Tips

### GPU Training Optimization

**See [GPU_OPTIMIZATIONS.md](../internals/GPU_OPTIMIZATIONS.md) for comprehensive GPU guide.**

Quick wins for GPU training:

```bash
# Maximum speed - larger batch with prefetch
mix run scripts/train_from_replays.exs \
  --batch-size 512 \
  --prefetch

# If GPU memory limited, use gradient accumulation
mix run scripts/train_from_replays.exs \
  --batch-size 256 \
  --accumulation-steps 2 \
  --prefetch
```

| Optimization | Flag | Impact |
|--------------|------|--------|
| Larger batch | `--batch-size 512` | Better GPU utilization |
| Prefetching | `--prefetch` | Overlap data/compute |
| Grad accumulation | `--accumulation-steps 4` | Effective larger batch |

> **Note on BF16**: Benchmarks show BF16 is actually 2x SLOWER than FP32 on RTX 4090 due to XLA issues
> (dimension misalignment, type casting overhead). FP32 is the default and recommended.

**Training time (GPU):**

| Dataset Size | Epochs | Time (RTX 4090) |
|--------------|--------|-----------------|
| 100 files | 10 | ~30 min |
| 1000 files | 10 | ~3-4 hours |
| Full dataset | 10 | ~8-10 hours |

### CPU Training Optimization

**XLA Multi-threading** (auto-enabled):
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_from_replays.exs
```

**Batch size tuning:**
- Larger batches (128, 256) reduce per-batch overhead
- Monitor RAM: `free -h` during training
- If swap increases, reduce batch size

**System tuning:**
```bash
# Lower swappiness if swap thrashing occurs
sudo sysctl vm.swappiness=10

# Increase process priority
sudo renice -n -5 -p <PID>
```

### Training Time Expectations

| Dataset Size | Epochs | Estimated Time (CPU) |
|--------------|--------|---------------------|
| 1 file (~14K frames) | 1 | ~5-6 minutes |
| 10 files (~140K frames) | 1 | ~15-20 minutes |
| 100 files (~1.4M frames) | 10 | ~3-4 hours |

*First epoch includes ~5 min JIT compilation overhead*

### Memory Usage by Config

| Config | Embedding | Training | Notes |
|--------|-----------|----------|-------|
| Full (hidden=256, window=60, files=10) | ~10GB | ~12GB | Best accuracy |
| Medium (hidden=128, window=30, files=5) | ~5GB | ~9GB | Good balance |
| Minimal (hidden=64, window=20, files=3) | ~3GB | ~5GB | Fast iteration |

## PPO Fine-tuning

After imitation learning, refine with reinforcement learning:

```bash
# Test PPO loop with mock environment
mix run scripts/train_ppo.exs --mock \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --timesteps 10000

# Full PPO training with Dolphin
mix run scripts/train_ppo.exs \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo \
  --opponent cpu3 \
  --timesteps 100000
```

## Self-Play Training

Train multiple agents against each other with Elo tracking:

```bash
# Basic self-play (4 parallel games)
mix run scripts/train_self_play.exs \
  --num-games 4 \
  --track-elo

# With custom episode length (default: 28800 frames = ~8 min)
mix run scripts/train_self_play.exs \
  --num-games 4 \
  --max-episode-frames 18000 \
  --track-elo
```

### Self-Play Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-games N` | 4 | Number of parallel games to run |
| `--game-type TYPE` | mock | Game type: mock, dolphin, libmelee |
| `--max-episode-frames N` | 28800 | Max frames per episode (~8 min at 60fps) |
| `--track-elo` | false | Enable Elo rating tracking |
| `--ppo-epochs N` | 4 | PPO update epochs per batch |
| `--clip-epsilon F` | 0.2 | PPO clipping parameter |
| `--gae-lambda F` | 0.95 | GAE lambda for advantage estimation |

## Evaluation

```bash
# Evaluate model on replay frames
mix run scripts/eval_model.exs --policy checkpoints/imitation_latest_policy.bin

# Interactive analysis with Livebook
livebook server notebooks/evaluation_dashboard.livemd --port 8080
```

The evaluation dashboard provides:
1. **Load Policy** - View config and architecture
2. **Test Inference** - Run on sample states
3. **Compare to Replays** - Accuracy vs human play
4. **Visualize Actions** - Charts of button/stick distributions

## Checkpointing

Training produces:
```
checkpoints/
├── {backbone}_{name}_{timestamp}.axon          # Full checkpoint
├── {backbone}_{name}_{timestamp}_policy.bin    # Exported policy
└── {backbone}_{name}_{timestamp}_config.json   # Training config
```

### Resume Training

```bash
mix run scripts/train_from_replays.exs --resume checkpoints/model.axon
```

Restores: model weights, optimizer state, step counter, config.

### Best Model Checkpointing

```bash
mix run scripts/train_from_replays.exs --save-best --val-split 0.1
```

Saves whenever validation loss improves.

### Batch-Interval Checkpointing

For streaming mode or long epochs, save checkpoints every N batches:

```bash
# Save every 500 batches (recommended for streaming mode)
mix run scripts/train_from_replays.exs --stream-chunk-size 30 --save-every-batches 500

# More frequent saves for large datasets
mix run scripts/train_from_replays.exs --save-every-batches 200
```

This creates `checkpoints/{name}_batch.axon` that gets overwritten at each interval.
Protects against losing progress if training crashes or is interrupted.

**When to use:**
- Streaming mode (epochs can be very long)
- Training on large datasets (>1000 batches per epoch)
- Unreliable GPU/cloud instances (preemptible VMs)

### Graceful Shutdown (Ctrl+C)

Training automatically handles SIGTERM and SIGINT (Ctrl+C) signals:

- Pressing Ctrl+C during training saves a checkpoint before exiting
- Checkpoint saved to `checkpoints/{name}_interrupt.axon`
- Resume interrupted training: `--resume checkpoints/{name}_interrupt.axon`

```bash
# Example: interrupt during training
# Press Ctrl+C, see:
#   ⚠ Received sigint - saving checkpoint before exit...
#   Saving trainer state (epoch 3, batch 450)...
#   ✓ Interrupt checkpoint saved to checkpoints/mewtwo_interrupt.axon
#   Resume with: --resume checkpoints/mewtwo_interrupt.axon

# Then resume:
mix run scripts/train_from_replays.exs --resume checkpoints/mewtwo_interrupt.axon
```

The trainer state is updated every 10 batches, so at most ~10 batches of work may be lost on interrupt.

### Config JSON Contents

Each checkpoint produces a `*_config.json` file with complete training provenance:

```json
{
  "timestamp": "2026-01-24T17:00:00Z",

  "// Training parameters": "",
  "backbone": "mamba",
  "epochs": 10,
  "batch_size": 64,
  "hidden_sizes": [256, 256],
  "temporal": true,
  "window_size": 30,

  "// Data filtering (what was trained on)": "",
  "characters": ["mewtwo"],
  "stages": null,
  "replays_dir": "/path/to/replays",
  "max_files": 100,

  "// Replay manifest (provenance)": "",
  "replay_count": 100,
  "replay_files": ["game1.slp", "game2.slp", "..."],
  "replay_manifest_hash": "sha256:abc123...",
  "character_distribution": {"mewtwo": 80000, "fox": 20000},

  "// Results": "",
  "training_frames": 125000,
  "validation_frames": 13000,
  "final_training_loss": 3.68,
  "total_time_seconds": 1847
}
```

**Key provenance fields:**

| Field | Description |
|-------|-------------|
| `characters` | Character filter used (`--characters mewtwo,fox`), null if unfiltered |
| `stages` | Stage filter used (`--stages battlefield,fd`), null if unfiltered |
| `replay_count` | Number of replay files used |
| `replay_files` | Actual file paths (if ≤500 files), for reproducibility |
| `replay_manifest_hash` | SHA256 of sorted file list, for deduplication |
| `character_distribution` | Frame counts per character in training data |

**Use cases:**
- Know exactly which replays trained a model
- Verify two models used the same data (compare hashes)
- Understand character composition of training data
- Reproduce training with same data

## Tests

```bash
mix test                                      # All tests
mix test --cover                              # With coverage
mix test test/exphil/training/imitation_test.exs  # Specific file
mix test --include slow                       # Include slow tests
```

## Interactive Setup Wizard

New to ExPhil? Use the interactive wizard to build your training command:

```bash
mix exphil.setup
```

The wizard walks you through:

1. **Goal Selection** - Quick experiment, character training, production model, or fine-tuning
2. **Character Selection** - Choose from Mewtwo, Ganondorf, Link, G&W, Zelda, Ice Climbers, or general
3. **Hardware Configuration** - Auto-detects GPU and recommends batch size
4. **Data Configuration** - Replay directory, file limits
5. **Advanced Options** - Backbone, augmentation, W&B logging

At the end, it generates a ready-to-run command and optionally executes it.

**Example session:**
```
╔════════════════════════════════════════════════════════════════╗
║                ExPhil Training Setup Wizard                    ║
╚════════════════════════════════════════════════════════════════╝

Detected Hardware:
  GPU: NVIDIA GeForce RTX 4090 (24 GB)

What would you like to do?
  [1] Quick experiment (test setup, ~5 minutes)
  [2] Train a character-specific model
  [3] Train a general-purpose model
  [4] Fine-tune an existing model

Choice [1]: 2

Which character?
  [1] Mewtwo - Floaty, teleport recovery, tail hitboxes
  [2] Ganondorf - Heavy, powerful, spacing-focused
  ...

Choice [1]: 1

GPU Memory Tier: 24GB+
  Recommended batch size: 256
Use recommended settings? [Y/n]: y

...

════════════════════════════════════════════════════════════════
                              Command
════════════════════════════════════════════════════════════════

  mix run scripts/train_from_replays.exs --train-character mewtwo --epochs 20 --batch-size 256 --temporal --backbone mamba --augment

Run this command now? [y/N]:
```

## Environment Variables

Configure defaults via environment variables (CLI args still override):

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPHIL_REPLAYS_DIR` | `./replays` | Default replay directory |
| `EXPHIL_WANDB_PROJECT` | `exphil` | Default W&B project name |
| `EXPHIL_DEFAULT_PRESET` | none | Default preset to use |

**Usage:**
```bash
# Set in shell profile (~/.bashrc or ~/.zshrc)
export EXPHIL_REPLAYS_DIR="/data/melee/replays"
export EXPHIL_WANDB_PROJECT="my-melee-ai"

# Now training uses these defaults
mix run scripts/train_from_replays.exs --epochs 10

# CLI args still override
mix run scripts/train_from_replays.exs --replays /other/path
```

## Verbosity Control

Control output verbosity with `--quiet` or `--verbose`:

| Flag | Level | Output |
|------|-------|--------|
| `--quiet` | 0 | Errors only, suppresses warnings and XLA/ptxas logs |
| (default) | 1 | Normal output with progress bars |
| `--verbose` | 2 | Debug info: timing, memory, gradients |

**Progress Bar Frequency:**
```bash
# Default is 100 batches between updates (keeps logs readable)
# For more frequent updates during debugging:
mix run scripts/train_from_replays.exs --log-interval 10

# For minimal log output (updates ~5 times per epoch on large datasets):
mix run scripts/train_from_replays.exs --log-interval 1000 --preset standard
```

**Examples:**
```bash
# Quiet mode for CI/scripted runs (suppresses XLA warnings like ptxas register spills)
mix run scripts/train_from_replays.exs --quiet --preset quick

# Verbose mode for debugging
mix run scripts/train_from_replays.exs --verbose --preset quick
```

**What `--quiet` suppresses:**
- Progress bar output (use `--log-interval N` for reduced updates)
- Warnings from Output module
- XLA/EXLA info logs (ptxas register spills, JIT compilation notices)
- Logger `:info` level messages

**Verbose output includes:**
- Per-batch timing breakdown
- GPU memory usage after each epoch
- Gradient norm statistics
- Data loading vs training time
- Cache hit rates
- Debug messages marked with `[DEBUG]`

### Progress Intervals (Programmatic)

Data processing functions accept a `:progress_interval` option to control update frequency:

```elixir
# Default: update every 10 batches
Data.precompute_frame_embeddings(dataset, show_progress: true)

# Custom: update every 50 batches (less log spam)
Data.precompute_frame_embeddings(dataset,
  show_progress: true,
  progress_interval: 50
)
```

| Function | Default Interval | Unit |
|----------|-----------------|------|
| `precompute_frame_embeddings` | 10 | batches (1000 frames each) |
| `precompute_embeddings` | 10 | chunks |
| `precompute_augmented_frame_embeddings` | 10 | batches |
| `sequences_from_frame_embeddings` | 50,000 | sequences |

## Reproducibility

Training runs can be exactly reproduced using random seeds.

### Seed Display

Every training run shows its seed in the startup banner:
```
  Model Name:  mamba_mewtwo_20260123_143052
  Seed:        1234567890 (use --seed 1234567890 to reproduce)
```

### Explicit Seed

```bash
# Reproduce a previous run exactly
mix run scripts/train_from_replays.exs --seed 1234567890 --preset quick
```

### What the seed controls

- Parameter initialization (Nx/EXLA random operations)
- Data shuffling order
- Augmentation random choices (mirror, noise)
- Train/validation split randomness

**Note:** For exact reproduction, you also need the same:
- Replay files (same files in same order)
- Hardware (GPU vs CPU may differ slightly)
- ExPhil version

## Checkpoint Safety

Protect valuable checkpoints from accidental overwrites.

### Collision Warnings

If a checkpoint already exists, you'll see a warning:
```
⚠️  Checkpoint 'checkpoints/mewtwo_v1.axon' already exists
       Size: 45.2 MB, Modified: 2026-01-23 14:30:00
       Use --overwrite to replace, or choose a different --name
```

### Overwrite Control

| Flag | Behavior |
|------|----------|
| (default) | Error if checkpoint exists |
| `--overwrite` | Allow overwriting (with backup) |
| `--no-overwrite` | Explicitly fail if exists (for CI) |

### Automatic Backups

When overwriting, the existing checkpoint is automatically backed up:

```bash
# This creates a backup before overwriting
mix run scripts/train_from_replays.exs --overwrite --name mewtwo_v1
```

Backup files:
```
checkpoints/
├── mewtwo_v1.axon        # Current (new)
├── mewtwo_v1.axon.bak    # Previous version
├── mewtwo_v1.axon.bak.1  # Two versions ago
└── mewtwo_v1.axon.bak.2  # Three versions ago
```

### Backup Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backup` | true | Create .bak before overwrite |
| `--no-backup` | - | Skip backup (faster) |
| `--backup-count N` | 3 | Number of backup versions to keep |

**Example:**
```bash
# Keep 5 backup versions
mix run scripts/train_from_replays.exs --overwrite --backup-count 5

# Skip backups (for ephemeral training)
mix run scripts/train_from_replays.exs --overwrite --no-backup
```

## Embedding Options

### Stage Embedding Mode

Control how stages are embedded:

| Mode | Dims | Description |
|------|------|-------------|
| `--stage-mode full` | 64 | One-hot for all 64 stages (default) |
| `--stage-mode compact` | 7 | One-hot for 6 competitive + "other" |
| `--stage-mode learned` | 1 | Stage ID with trainable embedding |

```bash
# Save 57 dimensions with compact mode
mix run scripts/train_from_replays.exs --stage-mode compact

# Learned embedding (most compact)
mix run scripts/train_from_replays.exs --stage-mode learned
```

### Player Name Embedding

Control player name embedding dimensions:

| Option | Dims | Description |
|--------|------|-------------|
| (default) | 112 | slippi-ai compatible |
| `--num-player-names 0` | 0 | Disable (saves 112 dims) |
| `--num-player-names N` | N | Custom size |

```bash
# Disable player names to save dimensions
mix run scripts/train_from_replays.exs --num-player-names 0
```

**Note:** Existing models trained with 112 dims require `--num-player-names 112` for inference compatibility.


<!-- flag-reference:start (generated by ExPhil.Training.Config.FlagDocs — do not edit) -->

## Flag reference (generated)

Every training flag `scripts/train.exs` accepts, straight from the parser table
(`ExPhil.Training.Config.Parser.flag_table/0`) and `Config.defaults/0`.
Regenerate: `mix run -e 'ExPhil.Training.Config.FlagDocs.write!()'`.

| flag | type | default | description |
|---|---|---|---|
| `--replays` | string | `"./replays"` | Directory containing .slp files |
| `--replay-dir` | string | `"./replays"` | _(undocumented)_ |
| `--corpus` | string | `nil` | Pre-built MmapCorpus dir (`scripts/build_corpus.exs`) — skips parse/embed, trains straight off disk; overrides `--replays`. Temporal only. RAM is O(batch), so corpus size is unbounded (built for the 4,461-game fox_il_v2 run) |
| `--epochs` | int | `10` | Number of training epochs |
| `--batch-size` | int | `64` | Batch size |
| `--max-files` | optional int | `nil` | Limit number of replay files |
| `--skip-errors` | flag | `true` | Continue past bad replay files |
| `--fail-fast` | flag | `nil` | Stop on first error |
| `--show-errors` | flag | `true` | Show individual file errors |
| `--hide-errors` | flag | `nil` | Hide individual file errors |
| `--error-log` | string | `nil` | Log errors to file |
| `--checkpoint` | string | `nil` | _(undocumented)_ |
| `--player` | int | `1` | _(undocumented)_ |
| `--train-character` | atom | `nil` | 1x (filtered) |
| `--select-character-port` | flag | `false` | With `--train-character`: imitate that character's actual port per file (singles → their port, dittos → port 1). Without it the streaming loader imitates port 1 regardless (the fox_gen_v1 43%-non-fox corpus bug, E1). Streaming pipeline only |
| `--dual-port` | flag | `false` | Maximum data, mixed characters |
| `--balance-characters` | flag | `false` | Weight sampling by inverse char frequency |
| `--wandb` | flag | `false` | Enable Weights & Biases logging |
| `--wandb-project` | string | `"exphil"` | W&B project name |
| `--wandb-name` | string | `nil` | W&B run name (auto-generated if nil) |
| `--temporal` | flag | `false` | Enable temporal training |
| `--backbone` | atom | `:sliding_window` | See backbone list below |
| `--policy-type` | atom | `:autoregressive` | Policy architecture type |
| `--head` | atom | `:independent` | Controller head: `independent` (six parallel heads) or `autoregressive` (residual-stream conditional head, buttons→main_x→main_y→c_x→c_y→shoulder; see AUTOREGRESSIVE_HEAD_PLAN.md). Temporal only |
| `--action-horizon` | int | `8` | Action prediction horizon (for chunked policies) |
| `--num-inference-steps` | int | `20` | Diffusion/flow matching inference steps |
| `--kl-weight` | float | `10.0` | KL divergence weight for ACT (CVAE) |
| `--window-size` | int | `60` | Frames per sequence |
| `--stride` | int | `5` | Step between sequences |
| `--num-layers` | int | `2` | Number of Mamba layers |
| `--attention-every` | int | `nil` | Add attention every N layers (hybrid) |
| `--pre-norm` | flag | `true` | _(undocumented)_ |
| `--no-pre-norm` | flag | `nil` | _(undocumented)_ |
| `--qk-layernorm` | flag | `true` | Normalize Q/K before attention (stabilizes training) |
| `--no-qk-layernorm` | flag | `nil` | Disable QK LayerNorm |
| `--chunked-attention` | flag | `false` | Use chunked attention for 20-30% memory reduction |
| `--no-chunked-attention` | flag | `nil` | Disable chunked attention |
| `--chunk-size` | int | `32` | Chunk size for chunked/memory-efficient attention |
| `--memory-efficient-attention` | flag | `false` | Use memory-efficient attention (true O(n) memory via online softmax) |
| `--no-memory-efficient-attention` | flag | `nil` | Disable memory-efficient attention |
| `--flash-attention-nif` | flag | `false` | Use FlashAttention NIF for inference (forward-only, Ampere+ GPU) |
| `--no-flash-attention-nif` | flag | `nil` | Disable FlashAttention NIF |
| `--state-size` | int | `16` | SSM state dimension |
| `--expand-factor` | int | `2` | Expansion factor |
| `--conv-size` | int | `4` | Convolution kernel size |
| `--truncate-bptt` | optional int | `nil` | Truncated backprop (faster training) |
| `--bptt` | flag | `false` | Contiguous-BPTT training: cursors walk replays in order, GRU carry flows across chunks, per-timestep loss (GRU only; see BPTT_LOADER_DESIGN.md) |
| `--unroll` | int | `80` | BPTT chunk length in frames (gradient truncation horizon) |
| `--bptt-overlap` | int | `1` | Frames shared between consecutive BPTT chunks (set to frame_delay + 1) |
| `--bptt-val-files` | int | `16` | Whole replays held out for the carry-threaded val pass (game-level split; val batch is capped at 8 rows) |
| `--mixed-precision` | flag | `false` | FP32 master weights + BF16 compute (not recommended) |
| `--frame-delay` | int | `nil` | Training alias for --label-delay on every loader. Live Dolphin --frame-delay N remains reaction delay N-1. |
| `--label-delay` | int | `nil` | Reaction delay on top of causal state[t]/controller[t+1] pairing. Defaults to 0 after resolution; standard, streaming, and BPTT loaders apply it once. CLI overrides YAML, preset, and resume values. Live Dolphin delay numbering is unchanged. |
| `--num-heads` | int | `4` | Number of attention heads (parsed since 2026-09-09 — it was accepted-and-ignored before, and Trainer's private 2/32 table won) |
| `--head-dim` | int | `64` | Attention head width |
| `--log-file` | string | `nil` | Tee script output to a file |
| `--frame-delay-augment` | flag | `false` | Additional delay jitter on top of --label-delay; non-temporal standard loader only. Other loaders reject this flag. |
| `--frame-delay-min` | int | `0` | Minimum additional delay jitter when augmenting |
| `--frame-delay-max` | int | `18` | Maximum additional delay jitter when augmenting |
| `--stage-internals` | flag | `false` | Add FoD platform heights + PS transformation to the embedding (+7 raw dims, zero-gated by stage; W4 2026-08-24 stage-blindness verdict) |
| `--action-frame-buckets` | int | `0` | Bucketized action-frame one-hot per player: N dims, frames 0..N-2 individually, N-1 = at-or-beyond (0 = off, the historical scalar-only layout). The jab-chain lever (V2_PREP 7b): the 1/60 scalar could not carve the expert's frame-6 cliff. Layout key: stamped in the checkpoint, rebuilt by the Agent. Try 24. |
| `--early-stopping` | flag | `false` | Enable early stopping |
| `--patience` | int | `5` | Epochs without improvement before stopping |
| `--min-delta` | float | `0.01` | Minimum improvement to count as progress |
| `--save-best` | flag | `true` | Save model when val_loss improves |
| `--save-every` | optional int | `nil` | Save checkpoint every N epochs |
| `--save-every-batches` | optional int | `nil` | Save checkpoint every N batches (for streaming) |
| `--lr` | float | `0.0001` | Learning rate (alias: `--learning-rate`) |
| `--learning-rate` | float | `0.0001` | _(undocumented)_ |
| `--lr-schedule` | atom | `:constant` | cosine, linear, exponential |
| `--warmup-steps` | optional int | `1` | Learning rate warmup steps |
| `--decay-steps` | optional int | `nil` | Steps for LR decay |
| `--restart-period` | int | `1000` | Cosine annealing restart period (T_0) |
| `--restart-mult` | float | `2` | Restart period multiplier (T_mult) |
| `--max-grad-norm` | float | `1.0` | Gradient clipping norm (0 = disabled) |
| `--resume` | string | `nil` | Resume from checkpoint. If the checkpoint's controller head differs from `--head`, the TRUNK is transplanted: matching non-head params load, head + optimizer start fresh, config/step keep the trainer's |
| `--reinit-head` | flag | `false` | With `--resume`: re-initialise the controller head from scratch while loading the trunk (the "new head params" control, AUTOREGRESSIVE_HEAD_PLAN item 9) |
| `--name` | string | `nil` | Custom checkpoint name |
| `--accumulation-steps` | int | `1` | Gradient accumulation steps |
| `--val-split` | float | `0.1` | Validation split (0.1 = 10%) |
| `--augment` | flag | `false` | Enable data augmentation |
| `--mirror-prob` | float | `0.5` | Mirror augmentation probability |
| `--noise-prob` | float | `0.3` | Noise augmentation probability |
| `--noise-scale` | float | `0.01` | Noise magnitude |
| `--label-smoothing` | float | `0.1` | Label smoothing (prevents overconfidence) |
| `--dropout` | float | `0.0` | Dropout rate |
| `--focal-loss` | flag | `true` | Enable focal loss for rare actions |
| `--prev-action` | flag | `false` | Condition on previous frame's controller (training embeds frame i-1's inputs; live agent feeds back its own outputs). Enables frame-precise input sequences (dash dance, multishine). Regime is stored in the policy config — old checkpoints keep zeros. Not yet wired for --streaming. |
| `--no-prev-action` | neg flag | `false` | _(undocumented)_ |
| `--prev-action-dropout` | float | `0.0` | Zero the prev-action channel on fraction P of training frames (exposure-bias mitigation: live, the model feeds back its own outputs, which drift from teacher-forced ground truth). Try 0.1–0.3 with --prev-action. Mask is baked into the embedding cache entry — use --no-cache for a fresh mask. |
| `--scheduled-sampling` | float | `0.0` | Exposure bias: on fraction P of samples, replace the LAST window position's prev-action slice with the model's OWN decoded prediction (`ExPhil.Training.ScheduledSampling`; decode pinned to the live path). Requires `--temporal` and `--prev-action`. One extra forward pass per step. Loss under this flag is a harder objective — never compare loss curves across it; judge by live runs. |
| `--ss-ramp` | int | `10` | Ramp scheduled sampling 0 → P linearly over the first N epochs (drill loop; the main pipeline currently applies P flat). |
| `--mix-frames` | string | `nil` | Curriculum mixing: comma/glob list of drill `.frames` exports (`scripts/export_drill_frames.exs`) concatenated into training. Drill frames must be exported at the SAME `--action-delay` (the prev-action channel misaligns otherwise; a warning fires). Changes the embedding cache key. Replay mode only — corpus mode ignores it (use `--mix-corpus`). |
| `--mix-corpus` | string | `nil` | Corpus-mode curriculum mixing: a snippet mini-corpus (`scripts/build_snippet_corpus.exs`, one corpus file per snippet so windows never cross snippet boundaries) whose batches are interleaved evenly into the `--corpus` training stream. Embed sizes must match. Mix files are all-train (val stays comparable to unmixed baselines). |
| `--mix-oversample` | int | `1` | Passes of the mix corpus interleaved per epoch. The mix is typically a tiny fraction of the main corpus — oversample to give corrections a meaningful gradient share (e.g. 20 ≈ a few percent for a 41M-frame corpus with ~15k mix frames). |
| `--per-stage-ledge` | flag | `false` | Task #25: use the real per-stage edge x (`Melee.Stages.edge_ground_position`) in the ledge-distance feature instead of the historical 85-everywhere constant (which reads "safe" at x=60 on YS when the player is offstage). Changes the embedding VALUES: existing checkpoints and corpora are calibrated to the constant, so this is for fresh v3-edge arms only, and corpus-mode training needs a corpus REBUILT with the same flag (recorded in corpus meta). |
| `--action-delay` | int | `nil` | Training alias for --label-delay on every loader. Conflicting explicit aliases are rejected. |
| `--no-focal-loss` | neg flag | `true` | _(undocumented)_ |
| `--focal-gamma` | float | `3.0` | Focal loss gamma (higher = focus on hard) |
| `--button-weight` | float | `2.0` | Multiply button loss (fixes under-prediction) |
| `--stick-edge-weight` | float | `2.0` | Weight edge stick buckets higher (try 2.0-3.0) |
| `--entropy-weight` | float | `0.01` | _(undocumented)_ |
| `--neutral-weight` | float | `0.25` | Per-frame loss weight for neutral (no-input) frames; action frames get 1.0. Blanket anti-passivity knob — 1.0 = unweighted (the 09-05 clean-loss arm), which raised idle 6.5x corpus at v2 scale. |
| `--transition-weight` | float | `nil` | Per-frame loss weight for DECISION frames (controller differs from the previous frame): `max(weight, X)`. Targets *when* to change action (leaving WAIT, committing) instead of downweighting all neutral frames. Flag added 2026-09-07 (was pipeline/drill-only). |
| `--offstage-weight` | float | `nil` | (bptt path) Per-frame loss weight for OFFSTAGE frames (subject airborne beyond the stage ledge): `max(weight, X)`. Rare-state coverage for recovery — offstage is rare in expert play, so the model gets few reps where it fails. Added 2026-09-08. |
| `--awbc` | flag | `false` | _(undocumented)_ |
| `--awbc-reward` | atom | `:shine` | _(undocumented)_ |
| `--awbc-beta` | float | `nil` | _(undocumented)_ |
| `--awbc-shuffle` | flag | `false` | _(undocumented)_ |
| `--head-normalize` | flag | `false` | _(undocumented)_ |
| `--no-head-normalize` | neg flag | `false` | _(undocumented)_ |
| `--action-oversample` | float | `3.0` | _(undocumented)_ |
| `--lazy-sequences` | flag | `true` | _(undocumented)_ |
| `--use-batch` | flag | `false` | _(undocumented)_ |
| `--no-register` | flag | `false` | Skip model registry |
| `--keep-best` | optional int | `nil` | Keep best N checkpoints (prune others) |
| `--ema` | flag | `false` | Enable model EMA |
| `--ema-decay` | float | `0.999` | EMA decay rate |
| `--precompute` | flag | `true` | Precompute embeddings (2-3x speedup) |
| `--no-precompute` | flag | `false` | Disable embedding precomputation |
| `--cache-embeddings` | flag | `true` | Enable disk caching of embeddings |
| `--no-cache` | flag | `false` | Ignore existing cache and recompute |
| `--cache-dir` | string | `"cache/embeddings"` | Cache directory |
| `--cache-augmented` | flag | `false` | Precompute augmented variants (~100x speedup) |
| `--num-noisy-variants` | int | `2` | Number of noisy variants to precompute |
| `--prefetch` | flag | `false` | Prefetch batches while GPU trains |
| `--no-prefetch` | flag | `nil` | Disable batch prefetching |
| `--gradient-checkpoint` | flag | `false` | Trade memory for compute |
| `--checkpoint-every` | int | `1` | Checkpoint every N layers |
| `--prefetch-buffer` | int | `2` | Number of batches to prefetch |
| `--layer-norm` | flag | `false` | Enable layer normalization (MLP) |
| `--no-layer-norm` | flag | `nil` | Disable layer normalization |
| `--residual` | flag | `false` | Enable residual connections (MLP) |
| `--no-residual` | flag | `nil` | Disable residual connections |
| `--optimizer` | atom | `:adam` | adam, adamw, lamb, radam |
| `--dry-run` | flag | `false` | Validate config without training |
| `--character` | atom list | `[]` | _(undocumented)_ |
| `--characters` | atom list | `[]` | Filter replays by character |
| `--stage` | atom list | `[]` | _(undocumented)_ |
| `--stages` | atom list | `[]` | Filter replays by stage |
| `--kmeans-centers` | string | `nil` | K-means cluster centers for sticks |
| `--stream-chunk-size` | optional int | `nil` | Load N files at a time (memory-bounded) |
| `--pipeline-chunks` | flag | `true` | Prepare next chunk while training (overlaps CPU/GPU) |
| `--no-pipeline-chunks` | flag | `nil` | Disable chunk pipelining (sequential processing) |
| `--cache-streaming` | flag | `true` | Cache chunk embeddings to disk (reuse across epochs) |
| `--no-cache-streaming` | flag | `nil` | Disable streaming cache |
| `--num-player-names` | optional int | `112` | Player name dims (0 to disable) |
| `--learn-player-styles` | flag | `false` | Enable style-conditional training |
| `--no-learn-player-styles` | flag | `nil` | _(undocumented)_ |
| `--player-registry` | string | `nil` | Save/load player registry JSON |
| `--min-player-games` | optional int | `1` | Min games for player to be in registry |
| `--log-interval` | optional int | `100` | Progress bar update frequency (every N batches) |
| `--seed` | optional int | `nil` | Random seed for reproducibility |
| `--overwrite` | flag | `false` | Allow overwriting existing checkpoints |
| `--no-overwrite` | flag | `nil` | Fail if checkpoint exists |
| `--backup` | flag | `true` | Create .bak before overwrite |
| `--no-backup` | flag | `nil` | Skip backup creation |
| `--backup-count` | optional int | `3` | Number of backup versions to keep |
| `--skip-duplicates` | flag | `true` | Skip duplicate replay files by hash |
| `--no-skip-duplicates` | flag | `nil` | Include all files even if duplicates |
| `--min-quality` | optional int | `nil` | Minimum quality score (0-100) for replays |
| `--show-quality-stats` | flag | `false` | Show quality distribution after filtering |
| `--gc-every` | optional int | `100` | Run garbage collection every N batches (0=disabled) |
| `--profile` | flag | `false` | Enable timing profiler (report at end) |
| `--val-concurrency` | optional int | `4` | Parallel validation batches (1=sequential) |
| `--mmap-embeddings` | flag or string | `false` | Use memory-mapped embeddings (for datasets > RAM) |
| `--mmap-path` | string | `nil` | Custom path for mmap file (auto-generated if not set) |
| `--auto-batch-size` | flag | `false` | Auto-tune batch size for optimal GPU utilization |
| `--auto-batch-min` | optional int | `32` | Minimum batch size to test |
| `--auto-batch-max` | optional int | `4096` | Maximum batch size to test |
| `--auto-batch-backoff` | float | `0.8` | Safety factor (0.8 = 20% headroom) |
| `--action-mode` | special | — | Action embedding: one_hot (399 dims) or learned (64-dim trainable) |
| `--action-mode-learned` | special | — | _(undocumented)_ |
| `--action-mode-one-hot` | special | — | _(undocumented)_ |
| `--button-pos-weight` | special | — | _(undocumented)_ |
| `--character-mode` | special | — | Character embedding: one_hot (33 dims) or learned (64-dim trainable) |
| `--character-mode-learned` | special | — | _(undocumented)_ |
| `--character-mode-one-hot` | special | — | _(undocumented)_ |
| `--config` | special | — | YAML config file path |
| `--hidden-sizes` | special | — | _(undocumented)_ |
| `--jumps-normalized` | special | — | Jumps as 1 normalized dim (false = 7-dim one-hot) |
| `--nana-mode` | special | — | Ice Climbers Nana: compact (39), enhanced (14+ID), full (449) |
| `--no-jumps-normalized` | special | — | _(undocumented)_ |
| `--online-robust` | special | — | Enable online play training mode (legacy path) |
| `--precision` | special | — | f32 or bf16 (FP32 is 2x faster due to XLA issues) |
| `--preset` | special | — | Training preset (quick, standard, full, mewtwo) |
| `--quiet` | special | — | Minimal output (errors only), suppresses XLA/ptxas logs |
| `--stage-mode` | special | — | Stage embedding: full, compact, learned |
| `--stage-mode-compact` | special | — | _(undocumented)_ |
| `--stage-mode-full` | special | — | _(undocumented)_ |
| `--stage-mode-learned` | special | — | _(undocumented)_ |
| `--verbose` | special | — | Debug output (timing, memory) |

<!-- flag-reference:end -->
