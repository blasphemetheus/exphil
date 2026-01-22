#!/usr/bin/env elixir
# Benchmark script for comparing embedding dimension configurations
#
# Evaluates speed and memory tradeoffs of:
# - One-hot vs learned action embeddings
# - Compact vs enhanced vs full Nana mode
# - Various 512-dim optimized configurations
#
# Usage:
#   mix run scripts/benchmark_embeddings.exs [--iterations 1000] [--batch-size 32]

defmodule EmbeddingBenchmark do
  @moduledoc """
  Benchmarks different embedding configurations for speed and memory.
  """

  alias ExPhil.Embeddings.{Game, Player}
  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Output

  @default_iterations 1000
  @default_batch_size 32

  # ============================================================================
  # Configuration Presets to Benchmark
  # ============================================================================

  @configs %{
    # Current default (1204 dims)
    "current_default" => %{
      description: "Current default (one-hot actions, compact Nana)",
      player: %Player{
        action_mode: :one_hot,
        nana_mode: :compact,
        jumps_normalized: true
      }
    },

    # Learned actions only (408 dims)
    "learned_actions" => %{
      description: "Learned actions, compact Nana",
      player: %Player{
        action_mode: :learned,
        nana_mode: :compact,
        jumps_normalized: true
      }
    },

    # Enhanced Nana with learned actions (~250 dims continuous + 4 IDs)
    "enhanced_nana" => %{
      description: "Learned actions + enhanced Nana (4 action IDs)",
      player: %Player{
        action_mode: :learned,
        nana_mode: :enhanced,
        jumps_normalized: true
      }
    },

    # Full Nana mode for maximum IC detail
    "full_nana" => %{
      description: "One-hot actions + full Nana (~900 dims)",
      player: %Player{
        action_mode: :one_hot,
        nana_mode: :full,
        jumps_normalized: true
      }
    },

    # 512-target configuration (when stage + IC features implemented)
    "target_512" => %{
      description: "512-dim target (enhanced Nana + competitive stage)",
      player: %Player{
        action_mode: :learned,
        nana_mode: :enhanced,
        jumps_normalized: true
      },
      # TODO: Add when implemented
      # stage_mode: :competitive,
      # num_player_names: 0,
      # with_ic_features: true
    }
  }

  # ============================================================================
  # Benchmark Functions
  # ============================================================================

  def run(opts \\ []) do
    iterations = Keyword.get(opts, :iterations, @default_iterations)
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)

    Output.banner("Embedding Dimension Benchmark")
    Output.config([
      {"Iterations", iterations},
      {"Batch size", batch_size}
    ])

    results = Enum.map(@configs, fn {name, config_opts} ->
      benchmark_config(name, config_opts, iterations, batch_size)
    end)

    print_results(results)
    results
  end

  defp benchmark_config(name, config_opts, iterations, batch_size) do
    Output.puts("\n[#{name}] Benchmarking #{config_opts.description}")

    player_config = config_opts.player
    game_config = %Game{player: player_config}

    # Measure embedding size
    embed_size = Game.embedding_size(game_config)
    continuous_size = Game.continuous_embedding_size(game_config)

    # Create mock game states for benchmarking
    mock_states = create_mock_states(batch_size)

    # Benchmark embedding speed
    {embed_time_us, _} = :timer.tc(fn ->
      for _ <- 1..iterations do
        Enum.map(mock_states, fn state ->
          Game.embed(state, nil, 1, config: game_config)
        end)
      end
    end)

    embed_ms = embed_time_us / 1000
    embed_per_state_us = embed_time_us / (iterations * batch_size)

    # Benchmark batch embedding
    {batch_time_us, _} = :timer.tc(fn ->
      for _ <- 1..iterations do
        Game.embed_states_fast(mock_states, 1, config: game_config)
      end
    end)

    batch_ms = batch_time_us / 1000
    batch_per_state_us = batch_time_us / (iterations * batch_size)

    # Build and benchmark policy network
    action_embed_size = if player_config.action_mode == :learned, do: 64, else: nil
    num_action_ids = Game.num_action_ids(game_config)
    policy = Policy.build(
      embed_size: embed_size,
      action_embed_size: action_embed_size,
      num_action_ids: num_action_ids
    )
    {init_fn, predict_fn} = Axon.build(policy)
    params = init_fn.(Nx.template({1, embed_size}, :f32), Axon.ModelState.empty())

    # Benchmark inference
    input = Nx.broadcast(0.5, {batch_size, embed_size})
    {inference_time_us, _} = :timer.tc(fn ->
      for _ <- 1..iterations do
        predict_fn.(params, input)
      end
    end)

    inference_ms = inference_time_us / 1000
    inference_per_batch_us = inference_time_us / iterations

    %{
      name: name,
      description: config_opts.description,
      embed_size: embed_size,
      continuous_size: continuous_size,
      action_ids: embed_size - continuous_size,
      embed_total_ms: embed_ms,
      embed_per_state_us: embed_per_state_us,
      batch_total_ms: batch_ms,
      batch_per_state_us: batch_per_state_us,
      inference_total_ms: inference_ms,
      inference_per_batch_us: inference_per_batch_us,
      fps_60_ready: inference_per_batch_us < 16_666  # 60 FPS = 16.67ms per frame
    }
  end

  defp create_mock_states(count) do
    # Create mock game states with IC players for realistic benchmarking
    for _ <- 1..count do
      %ExPhil.Bridge.GameState{
        frame: :rand.uniform(10000),
        stage: 31,  # Battlefield
        players: %{
          1 => mock_player(with_nana: true),
          2 => mock_player(with_nana: false)
        },
        projectiles: [],
        items: [],
        distance: 50.0
      }
    end
  end

  defp mock_player(opts) do
    nana = if opts[:with_nana] do
      %ExPhil.Bridge.Nana{
        x: :rand.uniform() * 100 - 50,
        y: :rand.uniform() * 50,
        percent: :rand.uniform() * 100,
        stock: :rand.uniform(4),
        facing: Enum.random([true, false]),
        action: :rand.uniform(398)
      }
    else
      nil
    end

    %ExPhil.Bridge.Player{
      x: :rand.uniform() * 100 - 50,
      y: :rand.uniform() * 50,
      percent: :rand.uniform() * 150,
      stock: :rand.uniform(4),
      facing: Enum.random([true, false]),
      action: :rand.uniform(398),
      action_frame: :rand.uniform(60),
      character: :rand.uniform(32),
      invulnerable: false,
      jumps_left: :rand.uniform(6),
      on_ground: Enum.random([true, false]),
      shield_strength: :rand.uniform() * 60,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nana,
      controller_state: nil
    }
  end

  defp print_results(results) do
    Output.puts("\n" <> String.duplicate("=", 80))
    Output.puts("BENCHMARK RESULTS")
    Output.puts(String.duplicate("=", 80))

    # Header
    IO.puts("")
    IO.puts(String.pad_trailing("Config", 20) <>
            String.pad_leading("Dims", 8) <>
            String.pad_leading("Cont", 8) <>
            String.pad_leading("IDs", 6) <>
            String.pad_leading("Embed μs", 12) <>
            String.pad_leading("Infer μs", 12) <>
            String.pad_leading("60 FPS", 8))
    IO.puts(String.duplicate("-", 80))

    Enum.each(results, fn r ->
      fps_ready = if r.fps_60_ready, do: "✓", else: "✗"
      IO.puts(
        String.pad_trailing(r.name, 20) <>
        String.pad_leading("#{r.embed_size}", 8) <>
        String.pad_leading("#{r.continuous_size}", 8) <>
        String.pad_leading("#{r.action_ids}", 6) <>
        String.pad_leading("#{Float.round(r.embed_per_state_us, 1)}", 12) <>
        String.pad_leading("#{Float.round(r.inference_per_batch_us, 1)}", 12) <>
        String.pad_leading(fps_ready, 8)
      )
    end)

    IO.puts(String.duplicate("-", 80))
    Output.puts("\nLegend:")
    Output.puts("  Dims: Total embedding dimensions")
    Output.puts("  Cont: Continuous features (without action IDs)")
    Output.puts("  IDs: Number of action IDs for learned embedding")
    Output.puts("  Embed μs: Microseconds per state embedding")
    Output.puts("  Infer μs: Microseconds per batch inference")
    Output.puts("  60 FPS: Can run inference at 60 FPS (< 16.67ms)")
  end
end

# Parse command line args
{opts, _, _} = OptionParser.parse(System.argv(), strict: [
  iterations: :integer,
  batch_size: :integer
])

# Run benchmark
EmbeddingBenchmark.run(opts)
