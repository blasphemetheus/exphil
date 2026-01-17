# ExPhil Architecture

## Overview

ExPhil reimplements slippi-ai's architecture in Elixir, with enhancements from modern ML research. This document details the technical architecture and key design decisions.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ExPhil System                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Training Pipeline                               │ │
│  │                                                                        │ │
│  │   ┌──────────┐    ┌──────────────┐    ┌──────────────┐                │ │
│  │   │  Replay  │───→│   Parser     │───→│   Dataset    │                │ │
│  │   │  Files   │    │ (Peppi/SLP)  │    │  (Tensors)   │                │ │
│  │   │  (.slp)  │    │              │    │              │                │ │
│  │   └──────────┘    └──────────────┘    └──────┬───────┘                │ │
│  │                                              │                         │ │
│  │                    ┌─────────────────────────┴──────────────────┐      │ │
│  │                    │                                            │      │ │
│  │                    ▼                                            ▼      │ │
│  │   ┌────────────────────────────┐    ┌────────────────────────────┐    │ │
│  │   │    Imitation Learning      │    │    Reinforcement Learning   │    │ │
│  │   │                            │    │                             │    │ │
│  │   │  • Behavioral Cloning      │    │  • PPO Updates              │    │ │
│  │   │  • Teacher Forcing         │    │  • V-Trace Correction       │    │ │
│  │   │  • Value Pretraining       │    │  • Teacher KL Regularization│    │ │
│  │   │                            │    │  • Self-Play                │    │ │
│  │   └─────────────┬──────────────┘    └──────────────┬──────────────┘    │ │
│  │                 │                                   │                  │ │
│  │                 └───────────────┬───────────────────┘                  │ │
│  │                                 │                                      │ │
│  │                                 ▼                                      │ │
│  │                    ┌───────────────────────┐                          │ │
│  │                    │    Trained Model      │                          │ │
│  │                    │    (.axon checkpoint) │                          │ │
│  │                    └───────────┬───────────┘                          │ │
│  │                                │                                      │ │
│  └────────────────────────────────┼──────────────────────────────────────┘ │
│                                   │                                        │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                    Inference Pipeline                                  │ │
│  │                                │                                      │ │
│  │                                ▼                                      │ │
│  │   ┌────────────────────────────────────────────────────────────────┐ │ │
│  │   │                        Agent                                    │ │ │
│  │   │                                                                 │ │ │
│  │   │  ┌─────────────┐   ┌────────────┐   ┌─────────────────────┐   │ │ │
│  │   │  │   State     │──→│   Policy   │──→│   Controller        │   │ │ │
│  │   │  │  Embedding  │   │   Network  │   │   Output            │   │ │ │
│  │   │  │             │   │            │   │                     │   │ │ │
│  │   │  └─────────────┘   └────────────┘   └──────────┬──────────┘   │ │ │
│  │   │                                                │              │ │ │
│  │   └────────────────────────────────────────────────┼──────────────┘ │ │
│  │                                                    │                │ │
│  │                           ┌────────────────────────┘                │ │
│  │                           │                                         │ │
│  │                           ▼                                         │ │
│  │   ┌────────────────────────────────────────────────────────────┐   │ │
│  │   │                  Python Bridge                              │   │ │
│  │   │                                                             │   │ │
│  │   │   ┌─────────────┐        ┌────────────────────────────┐    │   │ │
│  │   │   │   Elixir    │◀──────▶│   libmelee                 │    │   │ │
│  │   │   │   Port/NIF  │        │   (Python)                 │    │   │ │
│  │   │   │             │        │                            │    │   │ │
│  │   │   └─────────────┘        └─────────────┬──────────────┘    │   │ │
│  │   │                                        │                   │   │ │
│  │   └────────────────────────────────────────┼───────────────────┘   │ │
│  │                                            │                       │ │
│  └────────────────────────────────────────────┼───────────────────────┘ │
│                                               │                         │
└───────────────────────────────────────────────┼─────────────────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │   Dolphin + Slippi     │
                                   │   (Game Emulator)      │
                                   └────────────────────────┘
```

## Neural Network Architecture

### Comparison: slippi-ai vs ExPhil

| Component | slippi-ai | ExPhil | Rationale |
|-----------|-----------|--------|-----------|
| Framework | TensorFlow + Sonnet | Nx + Axon | Elixir-native, GPU via EXLA |
| Backbone | LSTM/GRU | Transformer-LSTM Hybrid | Better long-range dependencies |
| Attention | None | Temporal Self-Attention | Captures multi-frame patterns |
| Training | Single GPU | Distributed (planned) | BEAM concurrency |
| Inference | Python | Elixir | Lower latency, better scheduling |

### Network Structure

```
Input: Game State (t-N to t)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    State Embedding Layer                     │
│                                                              │
│  Player0: [pos, action, damage, char, shield, jumps, ...]   │
│  Player1: [pos, action, damage, char, shield, jumps, ...]   │
│  Stage:   [id, platform_positions, randall]                 │
│  Items:   [type, pos, state] × N                            │
│  Prev Action: [buttons, sticks, shoulder]                   │
│                                                              │
│  Total: ~400-600 dimensional embedding                      │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Temporal Encoder                          │
│                                                              │
│  Option A: Pure LSTM (slippi-ai default)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LSTM(hidden=128) → LayerNorm → ResidualConnection  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Option B: Transformer-Like (ExPhil enhancement)            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  for layer in 1..N:                                  │   │
│  │    x = x + LSTM(LayerNorm(x))      # Recurrence     │   │
│  │    x = x + MLP(LayerNorm(x))       # FFW            │   │
│  │  # Optional: add self-attention over last K frames  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Output: 128-256 dimensional hidden state                   │
└─────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│    Policy Head      │  │    Value Head       │
│                     │  │                     │
│  Autoregressive:    │  │  Linear(hidden, 1)  │
│  1. Buttons (8)     │  │                     │
│  2. Main X (16)     │  │  Output: V(s)       │
│  3. Main Y (16)     │  │                     │
│  4. C-Stick X (16)  │  └─────────────────────┘
│  5. C-Stick Y (16)  │
│  6. Shoulder (4)    │
│                     │
│  Each conditioned   │
│  on previous        │
└─────────────────────┘
```

### Autoregressive Controller Head

slippi-ai uses autoregressive sampling for the controller output. Each component is conditioned on previous samples:

```elixir
defmodule ExPhil.Networks.ControllerHead do
  @moduledoc """
  Autoregressive controller head.
  Samples: buttons → main_x → main_y → c_x → c_y → shoulder
  """

  def sample(hidden, prev_action, temperature \\ 1.0) do
    # Sample buttons (8 independent Bernoulli)
    buttons_logits = button_mlp(hidden)
    buttons = sample_bernoulli(buttons_logits, temperature)

    # Condition on buttons, sample main stick X
    main_x_input = concat([hidden, embed(buttons)])
    main_x_logits = main_x_mlp(main_x_input)
    main_x = sample_categorical(main_x_logits, temperature)

    # Continue for remaining components...
    # Each subsequent sample is conditioned on all previous
  end
end
```

## Training Pipeline

### Phase 1: Imitation Learning

Goal: Learn to mimic human play from replay data

```
Loss = CrossEntropy(predicted_action, human_action) + β * MSE(predicted_value, discounted_return)
```

Key considerations:
- **Frame Delay**: Train with same delay as online play (18+ frames)
- **Action Space**: Discretize continuous sticks to ~16 positions per axis
- **Value Pretraining**: Bootstrap value function for faster RL

### Phase 2: Reinforcement Learning (PPO)

Goal: Improve beyond human level through self-play

```
L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
L_value = (V(s) - V_target)²
L_teacher_kl = KL(π_policy || π_teacher)  # Stay close to imitation policy

Loss = -L_clip + c1 * L_value + c2 * L_teacher_kl
```

Key differences from slippi-ai:
- Consider **Decision Transformer** framing: condition on desired return
- Consider **World Model**: predict next state for imagination-based training

## Reward Design

### Standard Rewards (from slippi-ai)

```elixir
defmodule ExPhil.Rewards.Standard do
  @damage_ratio 0.01

  def compute(game_state, prev_state) do
    # KO difference (primary signal)
    ko_diff = count_kos(game_state.p1) - count_kos(game_state.p0)

    # Damage dealt/received (secondary signal)
    damage_dealt = max(game_state.p1.percent - prev_state.p1.percent, 0)
    damage_taken = max(game_state.p0.percent - prev_state.p0.percent, 0)
    damage_diff = @damage_ratio * (damage_dealt - damage_taken)

    ko_diff + damage_diff
  end
end
```

### Character-Specific Shaped Rewards

For lower-tier characters, we may need additional reward shaping:

```elixir
defmodule ExPhil.Rewards.Mewtwo do
  @moduledoc "Mewtwo-specific reward shaping"

  # Reward successful teleport recoveries
  def teleport_recovery_bonus(state, prev_state) do
    if recovering?(prev_state) and on_stage?(state) do
      0.05
    else
      0.0
    end
  end

  # Reward confusion → aerial combos
  def confusion_combo_bonus(state) do
    if opponent_confused?(state) and following_up?(state) do
      0.02
    else
      0.0
    end
  end
end
```

## Python Bridge Design

### Option 1: Erlang Port (Simple, Isolated)

```elixir
defmodule ExPhil.Bridge.LibmeleePort do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    port = Port.open({:spawn, "python3 priv/python/melee_bridge.py"}, [:binary, :exit_status])
    {:ok, %{port: port}}
  end

  def get_game_state(port_pid) do
    GenServer.call(port_pid, :get_game_state)
  end

  def send_controller(port_pid, controller) do
    GenServer.cast(port_pid, {:send_controller, controller})
  end
end
```

### Option 2: Pythonx (In-Process, Lower Latency)

```elixir
defmodule ExPhil.Bridge.Pythonx do
  def init do
    Pythonx.init()
    # Execute Python code to set up libmelee
    Pythonx.execute("""
    import melee
    console = melee.Console(path="/path/to/slippi")
    controller = melee.Controller(console=console, port=1)
    """)
  end

  def get_game_state do
    Pythonx.execute("console.step()")
    |> parse_game_state()
  end
end
```

## Inference Optimization

Target: < 2ms per frame for 60 FPS play

### Strategies

1. **Model Quantization**: INT8 inference with EXLA
2. **Batching**: Process multiple potential futures in parallel
3. **JIT Compilation**: Nx.Defn compiles to optimized XLA

```elixir
defmodule ExPhil.Agent.Optimized do
  import Nx.Defn

  # JIT-compiled inference
  defn predict(model, state) do
    Axon.predict(model, state)
  end

  # Warm up compilation before game starts
  def warmup(model) do
    dummy_state = create_dummy_state()
    predict(model, dummy_state)
  end
end
```

## Testing Strategy

### Unit Tests
- Embedding correctness (compare with slippi-ai outputs)
- Network forward pass shapes
- Reward computation edge cases

### Integration Tests
- Full training loop on small dataset
- Python bridge communication
- Checkpoint save/load

### Evaluation Tests
- Play against Level 9 CPU
- Self-play ELO tracking
- Regression tests (don't get worse)

## Monitoring & Observability

### Training Metrics (Wandb)
- Imitation loss
- Value loss
- PPO objective
- KL divergence from teacher
- Reward statistics

### Inference Metrics (Telemetry)
- Inference latency (p50, p95, p99)
- Frame drops
- Memory usage
- GPU utilization
