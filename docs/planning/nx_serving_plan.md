# Nx.Serving for ExPhil Policy Inference

## Overview

Use `Nx.Serving` to wrap the trained policy network for real-time Dolphin gameplay. Nx.Serving provides auto-batching, multi-GPU partitioning, and distributed inference — exactly what's needed for scaling self-play training.

## Why Nx.Serving

Current inference path (`play_dolphin_async.exs`) runs one game instance with one model call per frame. This is:
- Batch size 1 on GPU — massively underutilizing the 5090
- Single process — can't scale to multi-game self-play
- Synchronous — blocks on each inference call

Nx.Serving solves all three:
- **Auto-batching**: Multiple game instances send requests, Serving batches them automatically
- **Multi-GPU**: `partitions: true` distributes across GPUs
- **Distributed**: Built-in Erlang distribution for multi-node clusters

## Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Dolphin 1  │  │  Dolphin 2  │  │  Dolphin N  │
│  (game)     │  │  (game)     │  │  (game)     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
   batched_run      batched_run      batched_run
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌──────────────────┐
              │  PolicyServing   │
              │  batch_size: 8   │
              │  timeout: 10ms   │
              │                  │
              │  preprocess:     │
              │    game_state    │
              │    → embedding   │
              │    → Nx.Batch    │
              │                  │
              │  compute:        │
              │    predict_fn    │
              │    (JIT/GPU)     │
              │                  │
              │  postprocess:    │
              │    logits        │
              │    → action      │
              └──────────────────┘
```

## Implementation Plan

### Phase 1: Basic Serving (single GPU, single node)

```elixir
defmodule ExPhil.Inference.PolicyServing do
  def create(checkpoint_path, opts \\ []) do
    # Load model
    {predict_fn, params} = load_policy(checkpoint_path)
    embed_config = Keyword.get(opts, :embed_config, ExPhil.Embeddings.config())

    Nx.Serving.new(fn serving_opts ->
      Nx.Defn.jit(
        fn batch -> predict_fn.(params, batch) end,
        Keyword.merge(serving_opts, on_conflict: :reuse)
      )
    end)
    |> Nx.Serving.client_preprocessing(fn game_state ->
      embedding = ExPhil.Embeddings.Game.embed_state(game_state, embed_config)
      {Nx.Batch.stack([embedding]), %{}}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _server}, _client ->
      ExPhil.Networks.Policy.Sampling.sample_action(result)
    end)
  end

  def start_link(checkpoint_path, opts \\ []) do
    serving = create(checkpoint_path, opts)

    Nx.Serving.start_link(
      serving: serving,
      name: Keyword.get(opts, :name, __MODULE__),
      batch_size: Keyword.get(opts, :batch_size, 8),
      batch_timeout: Keyword.get(opts, :batch_timeout, 10)
    )
  end

  def predict(game_state, name \\ __MODULE__) do
    Nx.Serving.batched_run(name, game_state)
  end
end
```

### Phase 2: Multi-GPU (partitions)

```elixir
# With partitions: true, Serving auto-distributes across GPUs
Nx.Serving.start_link(
  serving: serving,
  name: PolicyServing,
  batch_size: 8,
  batch_timeout: 10,
  partitions: true  # auto-detects GPUs
)
```

### Phase 3: Distributed (multi-node self-play)

```elixir
# Node A (GPU node): runs PolicyServing
# Node B (CPU node): runs Dolphin instances

# On Node B:
Nx.Serving.batched_run(PolicyServing, game_state,
  &Nx.backend_copy(&1, Nx.BinaryBackend)  # serialize for network
)
```

## Constraints

- **Latency budget**: 16.67ms per frame at 60fps. With batch_timeout: 10ms, we have 6.67ms for compute + overhead.
- **Memory**: Serving keeps model params in GPU memory permanently. Fine for inference (much smaller than training).
- **Temporal models**: For Mamba/LSTM, need to manage hidden state across frames. The Serving's stateless model needs state passed explicitly or stored per-game.

## Temporal State Management

For sequence models (Mamba, LSTM), each game instance maintains its own hidden state:

```elixir
# Each game stores its running state
defmodule GameState do
  defstruct [:hidden_state, :frame_buffer]
end

# Client preprocessing includes the hidden state
|> Nx.Serving.client_preprocessing(fn {game_state, hidden} ->
  embedding = embed(game_state)
  input = %{embedding: embedding, hidden: hidden}
  {Nx.Batch.stack([input]), %{}}
end)
```

## Performance Targets

| Scenario | Latency | Throughput |
|----------|---------|------------|
| 1 game, batch 1 | ~2ms | 500 fps |
| 8 games, batch 8 | ~5ms | 1600 fps |
| 8 games, 2 GPUs | ~3ms | 3200 fps |

## Dependencies

- Trained model checkpoint (`.axon` or `.bin`)
- ExPhil.Embeddings for game state embedding
- ExPhil.Networks.Policy.Sampling for action sampling
- Dolphin bridge (ExPhil.Bridge) for game communication
