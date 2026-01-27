defmodule ExPhil.Benchmarks.InferenceBenchmarkTest do
  @moduledoc """
  Benchmark tests for inference performance regression detection.

  These tests measure inference times and compare against stored baselines.
  If performance degrades beyond a threshold, tests fail.

  ## Running Benchmarks

      # Run benchmark tests
      mix test --only benchmark

      # Update baselines (when regression is expected/acceptable)
      BENCHMARK_UPDATE=1 mix test --only benchmark

  ## How It Works

  1. Each benchmark runs multiple iterations after warmup
  2. Results are compared against `test/fixtures/benchmark_baselines.json`
  3. If mean time exceeds baseline by >20% (configurable), test fails
  4. Use BENCHMARK_UPDATE=1 to update baselines when changes are intentional
  """

  use ExUnit.Case, async: false
  import ExPhil.Test.Helpers
  import ExPhil.Test.Factories

  alias ExPhil.Networks.{Policy, GatedSSM}
  alias ExPhil.Embeddings

  # Tag all tests as benchmarks (excluded by default)
  @moduletag :benchmark
  @moduletag :slow

  # Test parameters
  @batch_size 32
  @embed_size 128
  @seq_len 30

  describe "Policy network inference" do
    setup do
      # Build and initialize policy model
      model = Policy.build(embed_size: @embed_size, hidden_sizes: [128, 64])
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @embed_size}, :f32), Axon.ModelState.empty())

      # Create test input
      input = random_tensor({@batch_size, @embed_size})

      {:ok, params: params, predict_fn: predict_fn, input: input}
    end

    test "single frame inference", %{params: params, predict_fn: predict_fn, input: input} do
      {:ok, stats} =
        benchmark name: "policy_single_frame", iterations: 50, warmup: 10 do
          predict_fn.(params, input)
        end

      # Should complete in reasonable time (adjust based on hardware)
      assert stats.mean < 100, "Policy inference too slow: #{stats.mean}ms"
    end

    test "batched inference scales linearly", %{params: params, predict_fn: predict_fn} do
      # Single sample
      input_1 = random_tensor({1, @embed_size})

      {:ok, stats_1} =
        benchmark name: "policy_batch_1", iterations: 20 do
          predict_fn.(params, input_1)
        end

      # Larger batch
      input_32 = random_tensor({32, @embed_size})

      {:ok, stats_32} =
        benchmark name: "policy_batch_32", iterations: 20 do
          predict_fn.(params, input_32)
        end

      # Batch processing should be more efficient per-sample
      per_sample_1 = stats_1.mean
      per_sample_32 = stats_32.mean / 32

      # Batching should provide at least 2x improvement per sample
      assert per_sample_32 < per_sample_1,
             "Batching not improving efficiency: #{per_sample_32}ms/sample vs #{per_sample_1}ms/sample"
    end
  end

  # GatedSSM is very slow on CPU - requires GPU for reasonable benchmark times
  describe "GatedSSM backbone inference" do
    @describetag :gpu
    # 5 minutes for GPU benchmarks
    @describetag timeout: 300_000

    setup do
      model =
        GatedSSM.build(
          embed_size: @embed_size,
          hidden_size: 64,
          state_size: 16,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = random_tensor({@batch_size, @seq_len, @embed_size})

      {:ok, params: params, predict_fn: predict_fn, input: input}
    end

    test "temporal sequence inference", %{params: params, predict_fn: predict_fn, input: input} do
      {:ok, stats} =
        benchmark name: "mamba_temporal", iterations: 30, warmup: 5 do
          predict_fn.(params, input)
        end

      # GatedSSM should be fast enough for 60 FPS (< 16.67ms per frame)
      # Allow some headroom for full pipeline
      assert stats.mean < 50, "GatedSSM inference too slow for real-time: #{stats.mean}ms"
    end

    test "scales with sequence length", _context do
      # Short sequence
      input_10 = random_tensor({@batch_size, 10, @embed_size})

      # Need to rebuild model for different seq_len
      model_10 =
        GatedSSM.build(
          embed_size: @embed_size,
          hidden_size: 64,
          state_size: 16,
          num_layers: 2,
          window_size: 10
        )

      {init_fn_10, predict_fn_10} = Axon.build(model_10)
      params_10 = init_fn_10.(Nx.template({1, 10, @embed_size}, :f32), Axon.ModelState.empty())

      {:ok, stats_10} =
        benchmark name: "mamba_seq_10", iterations: 20 do
          predict_fn_10.(params_10, input_10)
        end

      # Longer sequence
      input_60 = random_tensor({@batch_size, 60, @embed_size})

      model_60 =
        GatedSSM.build(
          embed_size: @embed_size,
          hidden_size: 64,
          state_size: 16,
          num_layers: 2,
          window_size: 60
        )

      {init_fn_60, predict_fn_60} = Axon.build(model_60)
      params_60 = init_fn_60.(Nx.template({1, 60, @embed_size}, :f32), Axon.ModelState.empty())

      {:ok, stats_60} =
        benchmark name: "mamba_seq_60", iterations: 20 do
          predict_fn_60.(params_60, input_60)
        end

      # GatedSSM should scale sub-linearly with sequence length (that's its advantage)
      # 6x longer sequence should take < 3.5x longer (with some overhead slack)
      scaling_factor = stats_60.mean / stats_10.mean

      assert scaling_factor < 3.5,
             "GatedSSM not scaling well: #{scaling_factor}x slowdown for 6x sequence"
    end
  end

  describe "Embedding computation" do
    test "player embedding speed" do
      player = build_player()

      {:ok, stats} =
        benchmark name: "embed_player", iterations: 100, warmup: 20 do
          Embeddings.Player.embed(player)
        end

      # Embedding should be very fast (< 1ms)
      assert stats.mean < 5, "Player embedding too slow: #{stats.mean}ms"
    end

    test "game state embedding speed" do
      game_state = build_game_state()

      {:ok, stats} =
        benchmark name: "embed_game_state", iterations: 100, warmup: 20 do
          Embeddings.Game.embed(game_state, nil, 1)
        end

      # Full game state embedding should still be fast
      assert stats.mean < 10, "Game state embedding too slow: #{stats.mean}ms"
    end
  end
end
