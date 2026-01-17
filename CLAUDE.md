# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, designed to create high-ELO playable bots for lower-tier Melee characters.

## Target Characters (Initial Focus)
1. **Mewtwo** - Unique physics, teleport recovery, tail hurtbox mechanics
2. **Mr. Game & Watch** - No L-cancel, random hammer, unique shield
3. **Link** - Projectiles, tether recovery, bomb tech
4. **Ganondorf** - Slow but powerful, requires precise spacing/timing
5. **Zelda** - Transform mechanic, Din's Fire zoning, Lightning Kick spacing

## Project Structure

```
exphil/
├── lib/
│   ├── exphil/
│   │   ├── application.ex          # OTP Application
│   │   ├── networks/               # Neural network architectures
│   │   │   ├── policy.ex           # Policy network (actor)
│   │   │   ├── value.ex            # Value function (critic)
│   │   │   ├── transformer.ex      # Transformer-like temporal attention
│   │   │   └── recurrent.ex        # LSTM/GRU alternatives
│   │   ├── embeddings/             # Game state → tensor conversion
│   │   │   ├── player.ex           # Player state embedding
│   │   │   ├── game.ex             # Full game state embedding
│   │   │   ├── controller.ex       # Controller action embedding
│   │   │   └── character.ex        # Character-specific embeddings
│   │   ├── training/               # Training loops
│   │   │   ├── imitation.ex        # Behavioral cloning from replays
│   │   │   ├── ppo.ex              # Proximal Policy Optimization
│   │   │   ├── vtrace.ex           # V-trace for off-policy correction
│   │   │   └── distributed.ex      # Multi-GPU/node training
│   │   ├── agents/                 # Agent implementations
│   │   │   ├── basic.ex            # Single-step agent
│   │   │   └── delayed.ex          # Agent with frame delay handling
│   │   ├── rewards/                # Reward computation
│   │   │   ├── standard.ex         # KO/damage differential
│   │   │   └── shaped.ex           # Shaped rewards (approach, combo)
│   │   ├── data/                   # Data pipeline
│   │   │   ├── replay_parser.ex    # Parse .slp files (via Peppi)
│   │   │   ├── dataset.ex          # Training dataset management
│   │   │   └── batch.ex            # Batching utilities
│   │   └── eval/                   # Evaluation
│   │       ├── metrics.ex          # Performance metrics
│   │       └── matchups.ex         # Character matchup tracking
│   └── exphil_bridge/              # Python interop
│       ├── libmelee_port.ex        # libmelee communication
│       └── dolphin.ex              # Dolphin process management
├── priv/
│   └── python/
│       ├── melee_bridge.py         # Python-side libmelee wrapper
│       └── replay_converter.py     # Convert .slp to training format
├── test/
├── config/
│   ├── config.exs
│   ├── dev.exs
│   └── prod.exs
├── scripts/
│   ├── train_imitation.exs         # Imitation learning script
│   ├── train_rl.exs                # RL fine-tuning script
│   └── eval.exs                    # Evaluation script
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/                      # Livebook notebooks for analysis
├── mix.exs
└── CLAUDE.md
```

## Implementation Plan

### Phase 1: Foundation (Weeks 1-3)
Core infrastructure and Python bridge

#### 1.1 Project Setup
- [x] Initialize Mix project with proper deps (Nx, Axon, EXLA)
- [x] Configure EXLA backend (CPU w/ XLA optimizations, ~2000x faster)
- [x] Set up Pythonx or Erlang Port for libmelee communication
- [x] Create basic OTP supervision tree

#### 1.2 Data Pipeline
- [x] Port Peppi replay parser bindings (or use Pythonx)
- [x] Implement game state type structs (mirror slippi-ai's types.py)
- [x] Create training data format (efficient tensor storage)
- [x] Build dataset streaming for large replay collections

#### 1.3 Python Bridge
- [x] Implement libmelee wrapper in Python
- [x] Create Elixir Port for bidirectional communication
- [x] Handle game state serialization/deserialization
- [ ] Test with Dolphin + Slippi

### Phase 2: Embeddings & Networks (Weeks 4-6)
Neural network architecture in Nx/Axon

#### 2.1 State Embeddings (Port from slippi-ai/embed.py)
- [x] Player embedding: position, action, damage, character, etc.
- [x] Stage embedding: ID, platform positions
- [x] Item embedding (Link bombs, Peach turnips, Mr. Saturn, Bob-ombs)
- [x] One-hot and continuous embedding primitives

#### 2.2 Controller Embeddings
- [x] Button embedding (8 legal buttons as Bernoulli)
- [x] Stick embedding (discretized axis positions)
- [x] Autoregressive structure for sampling

#### 2.3 Network Architecture
- [x] Implement MLP backbone
- [x] Implement LSTM/GRU recurrent layers (`ExPhil.Networks.Recurrent`)
- [x] Transformer-like temporal attention (`ExPhil.Networks.Attention`)
  - Sliding window attention (O(K²) efficient)
  - Hybrid LSTM + attention architecture
  - Sinusoidal positional encoding
- [x] Policy head with controller output
- [x] Value head for RL
- [x] Temporal policy (`Policy.build_temporal/1`) integrating attention

### Phase 3: Training Infrastructure (Weeks 7-10)
Imitation learning and RL training loops

#### 3.1 Imitation Learning
- [x] Behavioral cloning loss (cross-entropy on actions)
- [ ] Value function bootstrapping
- [x] Temporal/sequence training with attention backbones
- [x] Wandb integration for metrics logging
- [x] Checkpointing and model saving

#### 3.2 Reinforcement Learning
- [x] PPO implementation in Axon
- [x] PPO training script (`scripts/train_ppo.exs`)
- [ ] V-trace for off-policy correction
- [x] Reward computation (KO diff, damage ratio)
- [ ] Teacher KL regularization (stay close to imitation policy)
- [ ] Self-play infrastructure

#### 3.3 Distributed Training
- [ ] Data parallelism across GPUs
- [ ] Gradient accumulation
- [ ] Async evaluation during training

### Phase 4: Character-Specific Features (Weeks 11-14)
Tailored embeddings and rewards per character

#### 4.1 Mewtwo
- [ ] Teleport recovery state tracking
- [ ] Tail hurtbox considerations in reward
- [ ] Confusion (side-B) combo reward shaping

#### 4.2 Mr. Game & Watch
- [ ] Handle no L-cancel (simplifies action space)
- [ ] Bucket (down-B) projectile absorption tracking
- [ ] Hammer RNG awareness in value function

#### 4.3 Link
- [ ] Bomb tracking and self-damage handling
- [ ] Projectile zoning reward shaping
- [ ] Tether recovery state machine

#### 4.4 Ganondorf
- [ ] Spacing-focused reward shaping
- [ ] Combo optimization (stomp chains)
- [ ] Edge guarding emphasis

#### 4.5 Zelda
- [ ] Transform state machine (Zelda ↔ Sheik)
- [ ] Din's Fire zoning and tracking
- [ ] Lightning Kick sweetspot spacing rewards

### Phase 5: Evaluation & Deployment (Weeks 15-18)
Testing, optimization, and Slippi integration

#### 5.1 Evaluation Framework
- [ ] CPU opponent benchmarks (Level 9)
- [ ] Self-play ELO tracking
- [ ] Human evaluation sessions
- [ ] Matchup-specific metrics

#### 5.2 Slippi Online Integration
- [ ] Handle 18+ frame delay properly
- [ ] Buffer donation support
- [ ] Stable netplay performance (< 2ms inference)

#### 5.3 Optimization
- [ ] Inference optimization for real-time play
- [ ] Model quantization (INT8 if needed)
- [ ] Frame skip strategies for slower hardware

## Key Improvements Over slippi-ai

### 1. Modern Architecture (Decision Transformer-inspired)
```elixir
# Instead of pure LSTM, use temporal attention
defmodule ExPhil.Networks.TemporalTransformer do
  # Self-attention over recent N frames
  # Better at capturing long-range dependencies
  # More parallelizable than LSTM during training
end
```

### 2. Character-Specific Modules
```elixir
# Character-aware embeddings and rewards
defmodule ExPhil.Embeddings.Character.Mewtwo do
  # Mewtwo-specific state features:
  # - Tail hurtbox extension
  # - Teleport charge state
  # - Shadow Ball charge level
end
```

### 3. Elixir Concurrency for Evaluation
```elixir
# Parallel self-play games using BEAM
defmodule ExPhil.Eval.Arena do
  use GenServer

  # Run many parallel games for faster evaluation
  # Natural fit for Elixir's concurrency model
end
```

### 4. World Model (Future Enhancement)
Inspired by Dreamer v3/v4:
- Learn environment dynamics model
- Train policy in "imagination" (faster than real rollouts)
- Better sample efficiency

## Current Research to Incorporate

### Decision Transformers
- Treat RL as sequence modeling
- Condition on desired return
- May improve sample efficiency

### Graph Neural Networks for Game State
- Model relationships between entities
- Better generalization across characters/stages

### Multi-Agent Training
- Train against diverse opponent pool
- Prevents overfitting to self-play strategies

## Dependencies (mix.exs)

```elixir
defp deps do
  [
    # ML Core
    {:nx, "~> 0.9"},
    {:axon, "~> 0.7"},
    {:exla, "~> 0.9"},  # GPU backend
    {:polaris, "~> 0.1"},  # Optimizers

    # Data
    {:explorer, "~> 0.9"},  # DataFrames
    {:rustler, "~> 0.33"},  # For Peppi bindings if needed

    # Python Interop
    {:pythonx, "~> 0.3"},  # Or use Erlport

    # Observability
    {:telemetry, "~> 1.2"},
    {:wandb, github: "..."},  # Wandb Elixir client if available

    # Dev
    {:kino, "~> 0.14", only: :dev},  # Livebook
  ]
end
```

## Getting Started

```bash
# Install dependencies
mix deps.get

# Configure GPU (in config/config.exs)
# config :exla, :clients, default: [platform: :cuda]

# Download replay dataset
mix exphil.download_replays

# Parse replays to training format
mix exphil.parse_replays --input ./replays --output ./parsed

# Start imitation training
mix exphil.train --mode imitation --character mewtwo

# Continue with RL
mix exphil.train --mode rl --checkpoint ./checkpoints/latest.axon
```

## Open Questions

1. **Python Bridge Strategy**: Port vs Pythonx vs NIFs?
   - Port: Simple but serialization overhead
   - Pythonx: In-process Python, lower overhead
   - NIFs: Maximum performance but complex

2. **Replay Parser**: Use Peppi directly or port to Elixir?
   - Peppi is Rust, could use Rustler
   - Or use Pythonx to call py-slippi

3. **Distributed Training**:
   - Single node multi-GPU first
   - Consider libcluster for multi-node later

4. **Inference Latency Target**:
   - Need < 2ms for smooth 60 FPS play
   - May need INT8 quantization or model distillation

## Implementation Notes

### Current Status (January 2025)

**Completed modules:**
- `ExPhil.Networks.Policy` - 6-head autoregressive policy + temporal variants
- `ExPhil.Networks.Value` - Value function with GAE computation
- `ExPhil.Networks.ActorCritic` - Combined actor-critic with PPO loss
- `ExPhil.Networks.Recurrent` - LSTM/GRU layers for temporal processing
- `ExPhil.Networks.Attention` - Sliding window & hybrid attention mechanisms
- `ExPhil.Embeddings.Primitives` - One-hot, float, bool embedding utilities
- `ExPhil.Embeddings.Player` - Player state embedding (446 dims base)
- `ExPhil.Embeddings.Game` - Full game state embedding (~1991 dims)
- `ExPhil.Embeddings.Controller` - Controller action embedding
- `ExPhil.Training.Imitation` - Behavioral cloning trainer (single-frame + temporal)
- `ExPhil.Training.Data` - Dataset batching with sequence support
- `ExPhil.Training.PPO` - PPO trainer with clipped objective
- `ExPhil.Rewards` - Reward computation (damage, KO, combo, recovery)
- `ExPhil.Data.ReplayParser` - Slippi replay parsing via py-slippi
- `ExPhil.Data.Dataset` - Dataset management and batching
- `ExPhil.Bridge.*` - Game state structs (GameState, Player, ControllerState, etc.)
- `ExPhil.Bridge.Supervisor` - DynamicSupervisor for MeleePort bridge processes
- `ExPhil.Agents.Supervisor` - DynamicSupervisor for inference agents
- `ExPhil.Agents.Agent` - GenServer holding trained policy for inference (single-frame + temporal)
- `ExPhil.Agents` - High-level facade for agent management
- `ExPhil.Telemetry` - Telemetry events and metrics collector
- `ExPhil.Integrations.Wandb` - Weights & Biases experiment tracking

**Test coverage:** 580 tests passing

### Next Steps (Priority Order)

#### Immediate (Ready Now)
1. **Temporal training baseline** - Train with `--temporal --backbone sliding_window`
   - Compare loss curves and action quality vs single-frame MLP
   - Expected: better combo recognition, reaction timing

2. **Larger dataset training** - Train on full replay collection
   - Use `--max-files 100` or more with multiple epochs
   - Monitor with Wandb: `--wandb --wandb-project exphil`

3. **PPO fine-tuning in mock mode** - Test the PPO training loop
   - `mix run scripts/train_ppo.exs --mock --pretrained checkpoints/imitation_latest_policy.bin`
   - Verify gradient flow and loss computation

#### Short-term (Requires Setup)
4. **Dolphin integration testing** - Connect MeleePort to real Dolphin
   - Install Slippi Dolphin, configure paths
   - Test `scripts/play_dolphin.exs` against CPU

5. **Frame delay training** - Add delay simulation to imitation learning
   - Critical for realistic Slippi online play (18+ frames)
   - Modify embedding to include past N frames

6. **Self-play infrastructure** - Train agent vs agent
   - Use BEAM concurrency for parallel games
   - Prevents overfitting to human replay patterns

#### Medium-term
7. **Character-specific rewards** - Mewtwo, G&W, Link specialization
8. **Distributed training** - Multi-GPU with EXLA
9. **Model quantization** - INT8 for <2ms inference

### Technical Gotchas

#### 1. Polaris.Updates.apply_updates nil issue
`Polaris.Updates.apply_updates/2` is a `defn` with a default nil parameter (`state \\ nil`).
In Nx 0.10.0, calling it directly fails during lazy container traversal.

**Fix:** Wrap with `Nx.Defn.jit/1`:
```elixir
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)
new_params = apply_updates_fn.(params, updates)
```

#### 2. Axon training vs inference mode
- `mode: :train` returns `%{prediction: ..., state: ...}` for stateful layers (dropout)
- `mode: :inference` returns predictions directly
- For gradient computation, use `mode: :inference` to avoid pattern matching complexity

#### 3. Nx.to_number with {1} shaped tensors
`Nx.to_number/1` only works on scalar tensors (shape `{}`). When `Nx.slice` returns
a single element, it has shape `{1}` which must be squeezed first:
```elixir
tensor |> Nx.squeeze() |> Nx.to_number()
```

#### 4. Axon.ModelState deprecation
Pass full `%Axon.ModelState{}` to predict functions, not just the `.data` map:
```elixir
# Good
predict_fn.(model_state, input)

# Deprecated (triggers warning)
predict_fn.(model_state.data, input)
```

#### 5. JIT compilation time
First batch takes 2-5 minutes on CPU for large models (1991 input dims).
This is normal - subsequent batches are fast after compilation.

#### 6. EXLA Backend Configuration
EXLA provides ~2000x speedup over Nx.BinaryBackend. Configured in `config/config.exs`:
```elixir
config :nx, default_backend: EXLA.Backend
config :exla, default_client: :host  # CPU with XLA optimizations
```

For CUDA GPU support (if available):
```elixir
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

#### 7. EXLA/Defn.Expr tensor mismatch in closures
When using `Nx.Defn.value_and_grad` with closures that capture tensors, you get:
```
cannot invoke Nx function because it relies on two incompatible tensor implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Copy ALL captured tensors before using them in gradient computation:
```elixir
# Copy batch data
states = Nx.backend_copy(states)
actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

# Also copy model parameters to avoid closure capture issues
model_state = deep_backend_copy(trainer.policy_params)
```

#### 8. Elixir struct pattern matching order
Structs are maps, so `is_map(%Nx.Tensor{})` returns `true`. When using guards with
struct patterns, the struct clause MUST come before the `is_map` guard:
```elixir
# CORRECT order - struct patterns first
defp deep_backend_copy(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor)
defp deep_backend_copy(%Axon.ModelState{data: data} = state) do
  %{state | data: deep_backend_copy(data)}
end
defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
end
defp deep_backend_copy(other), do: other
```

#### 9. EXLA tensor serialization in checkpoints
EXLA tensors contain device buffer references that are process/session-specific.
When serializing checkpoints with `:erlang.term_to_binary/1`, these buffers become
invalid after the training process ends.

**Fix:** Always convert tensors to `Nx.BinaryBackend` before saving:
```elixir
def save_checkpoint(trainer, path) do
  checkpoint = %{
    policy_params: to_binary_backend(trainer.policy_params),
    optimizer_state: to_binary_backend(trainer.optimizer_state),
    # ... other fields
  }
  File.write(path, :erlang.term_to_binary(checkpoint))
end

defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
defp to_binary_backend(%Axon.ModelState{data: data, state: state} = ms) do
  %{ms | data: to_binary_backend(data), state: to_binary_backend(state)}
end
defp to_binary_backend(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
end
defp to_binary_backend(other), do: other
```

**Symptoms of stale EXLA buffers:**
- `ArgumentError: unable to get buffer. It may belong to another node`
- `ArgumentError: decode failed, none of the variant types could be decoded`

#### 10. Policy architecture mismatch on load
When loading a policy, the model architecture must match exactly. If training used
`hidden_sizes: [64, 64]` but loading uses default `[512, 512]`, you get shape errors.

**Fix:** Always save and restore `hidden_sizes`, `embed_size`, `dropout` in policy config:
```elixir
# In export_policy
config = %{
  embed_size: trainer.config[:embed_size],
  hidden_sizes: trainer.config[:hidden_sizes],
  dropout: trainer.config[:dropout],
  # ... other config
}
```

#### 11. Dynamic sequence length in attention models causes infinite JIT compilation
When building attention models with `shape: {nil, nil, embed_size}` (dynamic batch AND
sequence length), XLA tries to compile kernels that handle arbitrary sequence lengths.
This can take 30+ minutes or never complete.

**Symptoms:**
- Training process stuck at "Step 4: Training..." with high CPU, no progress
- JIT compilation takes >10 minutes for attention models

**Fix:** Use concrete sequence length (`:seq_len` option) in attention model builders:
```elixir
# GOOD - concrete seq_len (fast JIT, ~2-3 minutes)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: 30  # Defaults to window_size, which is correct for training
)

# BAD - dynamic seq_len (can hang indefinitely)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: nil  # Forces dynamic shape - avoid for training
)
```

The attention modules now default `seq_len` to `window_size` for training efficiency.
For inference with variable-length sequences, pass `seq_len: nil` explicitly and
expect longer JIT compilation on first run.

#### 12. Pre-computed tensors in Axon.nx closures need BinaryBackend
When pre-computing tensors (like attention masks) that are captured in `Axon.nx`
closures, they must be in `Nx.BinaryBackend` to avoid EXLA/Defn.Expr mismatch.

**Symptom:**
```
cannot invoke Nx function because it relies on two incompatible tensor
implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Convert pre-computed tensors to BinaryBackend before capture:
```elixir
# GOOD - explicit BinaryBackend
mask = window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
Axon.nx(input, fn tensor ->
  # mask is now safely inlined into the defn expression
  scaled_dot_product_attention(query, key, value, mask: mask)
end)

# BAD - EXLA tensor captured in closure
mask = window_mask(seq_len, window_size)  # Uses default EXLA backend
Axon.nx(input, fn tensor ->
  # CRASH: mask is EXLA but tensor is Defn.Expr during tracing
  scaled_dot_product_attention(query, key, value, mask: mask)
end)
```

### Performance Tips

#### CPU Training Optimization

**XLA Multi-threading (2-3x speedup):**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_from_replays.exs
```

**Batch size tuning:**
- Larger batches (128, 256) reduce per-batch overhead
- Monitor RAM usage: `free -h` during training
- If swap usage increases, reduce batch size

**System tuning (optional):**
```bash
# Lower swappiness if swap thrashing occurs
sudo sysctl vm.swappiness=10

# Check process priority (nice value)
ps -o nice,pid,comm -p <PID>

# Increase priority (requires sudo for negative values)
sudo renice -n -5 -p <PID>
```

#### Training Time Expectations

| Dataset Size | Epochs | Estimated Time (CPU) |
|--------------|--------|---------------------|
| 1 file (~14K frames) | 1 | ~5-6 minutes |
| 10 files (~140K frames) | 1 | ~15-20 minutes |
| 100 files (~1.4M frames) | 10 | ~3-4 hours |

*First epoch includes ~5 min JIT compilation overhead*

#### Temporal Training Overhead

Temporal training (`--temporal`) has additional embedding overhead:
- Each sequence of `window_size` frames must be embedded
- Pre-computed embeddings happen during Step 2 (visible progress)
- Embedding time scales with: `sequences × window_size × embed_time`

Example for 1 replay file (~14K frames, window=30):
- Sequences created: ~13,848
- Total embeddings: ~415,000 (sequences × window)
- Embedding time: ~20-30 minutes on CPU

The `Data.precompute_embeddings/2` function does this once during dataset
creation rather than per-batch, preventing silent hangs during training.

#### GPU Support

Currently using CPU with XLA optimizations. For NVIDIA GPU:
```elixir
# config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

Intel Iris Xe / AMD integrated GPUs have limited EXLA support.

### Architecture Decisions

#### Embedding Structure
- **Player embedding:** 446 dimensions (position, action one-hot, character, stocks, etc.)
- **Game embedding:** ~1991 dimensions (2 players + stage + optional projectiles + name ID)
- **Controller embedding:** 8 buttons + 4 stick axes + 1 shoulder = 13 dimensions

#### Policy Network
- MLP backbone with configurable hidden sizes (default: [512, 512])
- 6 output heads: buttons (8 Bernoulli), main_x/y, c_x/y (17-way categorical each), shoulder (5-way)
- Autoregressive sampling during inference

#### Training
- AdamW optimizer with weight decay
- Imitation learning first, then PPO fine-tuning
- Gradient clipping for stability

### Running Training

```bash
# Single-frame imitation learning (baseline)
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100

# Temporal training with sliding window attention
mix run scripts/train_from_replays.exs --temporal --backbone sliding_window \
  --window-size 60 --epochs 10

# Temporal training with hybrid LSTM + attention
mix run scripts/train_from_replays.exs --temporal --backbone hybrid

# Temporal training with LSTM only
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# With all options
mix run scripts/train_from_replays.exs \
  --replays /path/to/replays \
  --epochs 5 \
  --batch-size 64 \
  --player-port 1 \
  --temporal \
  --backbone sliding_window \
  --window-size 60 \
  --stride 1
```

**Temporal training tradeoffs:**
- Slower per-epoch (sequences are larger than single frames)
- Better at learning temporal patterns (combos, reactions, habits)
- Recommended after establishing baseline with single-frame training

### PPO Fine-tuning

```bash
# Test PPO loop with mock environment (no Dolphin needed)
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

### Playing and Evaluation

```bash
# Evaluate model on replay frames
mix run scripts/eval_model.exs --policy checkpoints/imitation_latest_policy.bin

# Play against CPU in Dolphin
mix run scripts/play_dolphin.exs \
  --policy checkpoints/imitation_latest_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso

# Interactive analysis with Livebook
livebook server notebooks/evaluation_dashboard.livemd
```

### Temporal Inference in Agents

Agents automatically detect temporal policies and handle frame buffering:

```elixir
# Load a temporal policy - agent auto-detects temporal config
{:ok, agent} = ExPhil.Agents.start_with_policy(:my_agent, "checkpoints/temporal_policy.bin")

# Check temporal config
config = ExPhil.Agents.Agent.get_config(agent)
# => %{temporal: true, backbone: :sliding_window, window_size: 60, buffer_size: 0, ...}

# Each get_action call adds to the frame buffer
{:ok, action} = ExPhil.Agents.get_action(:my_agent, game_state)

# Reset buffer when starting a new game
ExPhil.Agents.Agent.reset_buffer(agent)
```

**How temporal inference works:**
1. Each `get_action` call embeds the game state and adds it to a rolling buffer
2. If buffer has fewer than `window_size` frames, pad with first frame (warmup)
3. Build sequence tensor `[1, window_size, embed_size]`
4. Feed to temporal policy network
5. Sample action from output logits

**Policy export includes temporal metadata:**
```elixir
# When exporting, temporal config is preserved
Imitation.export_policy(trainer, "policy.bin")
# Saved config includes: temporal, backbone, window_size, num_heads, head_dim, etc.
```

### Test Commands

```bash
# Run all tests
mix test

# Run with coverage
mix test --cover

# Run specific test file
mix test test/exphil/training/imitation_test.exs

# Run slow tests (tagged @tag :slow)
mix test --include slow
```

## References

- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference
- [libmelee](https://github.com/altf4/libmelee) - Game interface
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Sequence modeling for RL
- [DreamerV3](https://danijar.com/project/dreamerv3/) - World model RL
- [Nx](https://github.com/elixir-nx/nx) - Numerical Elixir
- [Axon](https://github.com/elixir-nx/axon) - Neural networks for Elixir
