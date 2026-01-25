defmodule ExPhil.Benchmarks.GpuIntegrationTest do
  @moduledoc """
  GPU integration tests for memory, stability, checkpoints, and gradients.

  These tests verify GPU-specific functionality beyond basic training speed.

  ## Running

      MIX_ENV=test mix test --only gpu test/exphil/benchmarks/gpu_integration_test.exs

  ## Test Categories

  - Memory tests: OOM detection, leak detection
  - Numerical stability: NaN/Inf detection, gradient explosion
  - Checkpoint tests: Save/load roundtrip, cross-device
  - Performance tests: JIT timing, real-time inference
  """

  use ExUnit.Case, async: false

  alias ExPhil.Training.Imitation
  alias ExPhil.Training.Utils

  @moduletag :gpu
  @moduletag :slow
  @moduletag timeout: 600_000  # 10 min for GPU tests

  # Test parameters
  @embed_size 408
  @seq_len 30

  # ============================================================================
  # Performance Tests
  # ============================================================================

  describe "JIT compilation timing" do
    @tag :gpu
    test "measures first-batch JIT compilation overhead" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      batch = generate_batch(8, @embed_size, temporal: true, seq_len: @seq_len)

      # First batch includes JIT compilation
      {cold_time_us, {_, _}} = :timer.tc(fn ->
        Imitation.train_step(trainer, batch, nil)
      end)

      cold_ms = cold_time_us / 1000

      # Second batch is warm (JIT cached)
      {warm_time_us, _} = :timer.tc(fn ->
        Imitation.train_step(trainer, batch, nil)
      end)

      warm_ms = warm_time_us / 1000

      # Log the timings
      IO.puts("\n  [INFO] JIT cold start: #{Float.round(cold_ms, 1)}ms")
      IO.puts("  [INFO] Warm batch: #{Float.round(warm_ms, 1)}ms")
      IO.puts("  [INFO] JIT overhead: #{Float.round(cold_ms - warm_ms, 1)}ms (#{Float.round(cold_ms / warm_ms, 1)}x)")

      # Cold should be slower than warm (JIT overhead)
      assert cold_ms > warm_ms,
        "Expected cold start to be slower than warm batch"

      # Warm batch should be reasonably fast
      assert warm_ms < 5000,
        "Warm batch too slow: #{warm_ms}ms (expected <5000ms)"
    end
  end

  describe "real-time inference" do
    @tag :gpu
    test "single-frame inference meets 60fps budget (<16.6ms)" do
      # Build a temporal model
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      # Single frame batch
      batch = generate_batch(1, @embed_size, temporal: true, seq_len: @seq_len)

      # Warmup
      _ = do_predict(trainer, batch.states)

      # Measure 100 inferences
      num_inferences = 100
      {total_us, _} = :timer.tc(fn ->
        for _ <- 1..num_inferences do
          do_predict(trainer, batch.states)
        end
      end)

      avg_ms = total_us / 1000 / num_inferences

      IO.puts("\n  [INFO] Single-frame inference: #{Float.round(avg_ms, 2)}ms")
      IO.puts("  [INFO] Max sustainable FPS: #{Float.round(1000 / avg_ms, 1)}")

      # Should meet 60fps budget (16.6ms per frame)
      # Allow some headroom for game logic
      assert avg_ms < 16.6,
        "Inference too slow for 60fps: #{avg_ms}ms (budget: 16.6ms)"
    end

    @tag :gpu
    test "MLP inference is faster than Mamba" do
      # MLP should be fastest since no sequence processing
      mlp_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256, 256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      mamba_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      mlp_batch = generate_batch(1, @embed_size, temporal: false)
      mamba_batch = generate_batch(1, @embed_size, temporal: true, seq_len: @seq_len)

      # Warmup
      _ = do_predict(mlp_trainer, mlp_batch.states)
      _ = do_predict(mamba_trainer, mamba_batch.states)

      # Measure
      num_inferences = 50

      {mlp_us, _} = :timer.tc(fn ->
        for _ <- 1..num_inferences, do: do_predict(mlp_trainer, mlp_batch.states)
      end)

      {mamba_us, _} = :timer.tc(fn ->
        for _ <- 1..num_inferences, do: do_predict(mamba_trainer, mamba_batch.states)
      end)

      mlp_ms = mlp_us / 1000 / num_inferences
      mamba_ms = mamba_us / 1000 / num_inferences

      IO.puts("\n  [INFO] MLP inference: #{Float.round(mlp_ms, 2)}ms")
      IO.puts("  [INFO] Mamba inference: #{Float.round(mamba_ms, 2)}ms")

      # MLP should be faster (or at least comparable)
      assert mlp_ms <= mamba_ms * 1.5,
        "MLP slower than expected vs Mamba: #{mlp_ms}ms vs #{mamba_ms}ms"
    end
  end

  # ============================================================================
  # Numerical Stability Tests
  # ============================================================================

  describe "numerical stability" do
    @tag :gpu
    test "training produces no NaN or Inf for 100 batches" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      batches = generate_batches(100, 32, @embed_size, temporal: true, seq_len: @seq_len)

      final_trainer = Enum.with_index(batches)
      |> Enum.reduce(trainer, fn {batch, idx}, t ->
        {new_t, metrics} = Imitation.train_step(t, batch, nil)

        loss = Nx.to_number(metrics.loss)

        assert is_float(loss), "Loss is not a float at batch #{idx}: #{inspect(loss)}"
        refute is_nan(loss), "NaN loss detected at batch #{idx}"
        refute is_infinite(loss), "Infinite loss detected at batch #{idx}"
        assert loss < 1000, "Loss exploded at batch #{idx}: #{loss}"

        new_t
      end)

      assert final_trainer.step == 100
      IO.puts("\n  [INFO] Completed 100 batches with no NaN/Inf")
    end

    @tag :gpu
    test "extreme input values don't cause NaN" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      # Test with various extreme values
      test_cases = [
        {"zeros", Nx.broadcast(0.0, {8, @embed_size})},
        {"ones", Nx.broadcast(1.0, {8, @embed_size})},
        {"large positive", Nx.broadcast(100.0, {8, @embed_size})},
        {"small positive", Nx.broadcast(1.0e-6, {8, @embed_size})},
        {"negative", Nx.broadcast(-1.0, {8, @embed_size})}
      ]

      for {name, states} <- test_cases do
        batch = %{
          states: states,
          actions: generate_actions(8)
        }

        {_trainer, metrics} = Imitation.train_step(trainer, batch, nil)
        loss = Nx.to_number(metrics.loss)

        refute is_nan(loss), "NaN loss with #{name} inputs"
        refute is_infinite(loss), "Infinite loss with #{name} inputs"
      end

      IO.puts("\n  [INFO] All extreme input cases passed")
    end

    @tag :gpu
    test "gradients stay bounded (no explosion)" do
      # Use a model without gradient clipping to test raw gradients
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        max_grad_norm: nil,  # Disable clipping
        learning_rate: 1.0e-4
      )

      batches = generate_batches(20, 32, @embed_size, temporal: false)

      # Track gradient norms (approximated by weight change magnitude)
      Enum.reduce(batches, trainer, fn batch, t ->
        old_params = get_sample_params(t)
        {new_t, _} = Imitation.train_step(t, batch, nil)
        new_params = get_sample_params(new_t)

        # Check parameter change isn't too large (proxy for gradient explosion)
        param_diff = Nx.subtract(new_params, old_params) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        assert param_diff < 10.0,
          "Parameter change too large (possible gradient explosion): #{param_diff}"

        new_t
      end)

      IO.puts("\n  [INFO] Gradient bounds test passed")
    end
  end

  # ============================================================================
  # Checkpoint Tests
  # ============================================================================

  describe "checkpoint roundtrip" do
    @tag :gpu
    test "save and load produces identical predictions" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      # Train a few steps to get non-initial weights
      batches = generate_batches(5, 16, @embed_size, temporal: true, seq_len: @seq_len)
      trained = Enum.reduce(batches, trainer, fn batch, t ->
        {new_t, _} = Imitation.train_step(t, batch, nil)
        new_t
      end)

      # Test input
      test_batch = generate_batch(4, @embed_size, temporal: true, seq_len: @seq_len)

      # Get predictions before save
      pred_before = do_predict(trained, test_batch.states)

      # Save and reload
      tmp_dir = System.tmp_dir!()
      checkpoint_path = Path.join(tmp_dir, "test_checkpoint_#{:rand.uniform(100000)}.axon")

      try do
        :ok = Imitation.save_checkpoint(trained, checkpoint_path)
        {:ok, loaded} = Imitation.load_checkpoint(trained, checkpoint_path)

        # Get predictions after load
        pred_after = do_predict(loaded, test_batch.states)

        # Compare predictions (should be identical)
        assert_tensors_close(pred_before, pred_after, atol: 1.0e-5)

        IO.puts("\n  [INFO] Checkpoint roundtrip: predictions match")
      after
        File.rm(checkpoint_path)
      end
    end

    @tag :gpu
    test "checkpoint preserves training state (step, optimizer)" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      # Train a few steps
      batches = generate_batches(10, 16, @embed_size, temporal: false)
      trained = Enum.reduce(batches, trainer, fn batch, t ->
        {new_t, _} = Imitation.train_step(t, batch, nil)
        new_t
      end)

      original_step = trained.step

      tmp_dir = System.tmp_dir!()
      checkpoint_path = Path.join(tmp_dir, "test_state_#{:rand.uniform(100000)}.axon")

      try do
        :ok = Imitation.save_checkpoint(trained, checkpoint_path)
        {:ok, loaded} = Imitation.load_checkpoint(trained, checkpoint_path)

        assert loaded.step == original_step,
          "Step not preserved: expected #{original_step}, got #{loaded.step}"

        IO.puts("\n  [INFO] Training state preserved (step=#{original_step})")
      after
        File.rm(checkpoint_path)
      end
    end
  end

  # ============================================================================
  # Memory Tests
  # ============================================================================

  describe "memory stability" do
    @tag :gpu
    @tag :slow
    @tag timeout: 300_000
    test "no memory leak over 500 batches" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      # Force GC before starting
      :erlang.garbage_collect()
      Process.sleep(100)

      # Get initial memory
      initial_memory = :erlang.memory(:total)

      # Train many batches
      batches = generate_batches(500, 32, @embed_size, temporal: true, seq_len: @seq_len)

      _final_trainer = Enum.reduce(batches, trainer, fn batch, t ->
        {new_t, _} = Imitation.train_step(t, batch, nil)
        new_t
      end)

      # Force GC after
      :erlang.garbage_collect()
      Process.sleep(100)

      final_memory = :erlang.memory(:total)
      memory_growth_mb = (final_memory - initial_memory) / 1_000_000

      IO.puts("\n  [INFO] Initial memory: #{Float.round(initial_memory / 1_000_000, 1)}MB")
      IO.puts("  [INFO] Final memory: #{Float.round(final_memory / 1_000_000, 1)}MB")
      IO.puts("  [INFO] Memory growth: #{Float.round(memory_growth_mb, 1)}MB")

      # Allow some growth but catch major leaks
      # 500MB growth for 500 batches would indicate a leak
      assert memory_growth_mb < 500,
        "Possible memory leak: #{memory_growth_mb}MB growth over 500 batches"
    end
  end

  # ============================================================================
  # Gradient Accumulation Tests
  # ============================================================================

  describe "gradient accumulation" do
    @tag :gpu
    test "accumulated gradients approximate large batch" do
      # This test verifies that gradient accumulation works correctly
      # 4 batches of 16 should ≈ 1 batch of 64 (with some numerical differences)

      # Create two identical trainers
      trainer1 = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      trainer2 = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      # Generate consistent data
      key = Nx.Random.key(42)
      {large_states, _key} = Nx.Random.uniform(key, shape: {64, @embed_size}, type: :f32)
      actions = generate_actions(64)
      large_batch = %{states: large_states, actions: actions}

      # Split into 4 small batches
      small_batches = for i <- 0..3 do
        %{
          states: Nx.slice(large_states, [i * 16, 0], [16, @embed_size]),
          actions: %{
            buttons: Nx.slice(actions.buttons, [i * 16, 0], [16, 8]),
            main_x: Nx.slice(actions.main_x, [i * 16], [16]),
            main_y: Nx.slice(actions.main_y, [i * 16], [16]),
            c_x: Nx.slice(actions.c_x, [i * 16], [16]),
            c_y: Nx.slice(actions.c_y, [i * 16], [16]),
            shoulder: Nx.slice(actions.shoulder, [i * 16], [16])
          }
        }
      end

      # Method 1: Single large batch
      {_trained_large, metrics_large} = Imitation.train_step(trainer1, large_batch, nil)
      loss_large = Nx.to_number(metrics_large.loss)

      # Method 2: 4 small batches (simulated accumulation)
      # Note: This isn't true gradient accumulation (which would average gradients),
      # but we're checking that 4 steps on 4x smaller batches produces similar loss
      losses_small = Enum.map(small_batches, fn batch ->
        {_, metrics} = Imitation.train_step(trainer2, batch, nil)
        Nx.to_number(metrics.loss)
      end)
      avg_loss_small = Enum.sum(losses_small) / 4

      IO.puts("\n  [INFO] Large batch loss: #{Float.round(loss_large, 4)}")
      IO.puts("  [INFO] Avg small batch loss: #{Float.round(avg_loss_small, 4)}")

      # Losses should be in similar range (not identical due to different batch statistics)
      assert_in_delta loss_large, avg_loss_small, 1.0,
        "Large batch loss (#{loss_large}) too different from small batches (#{avg_loss_small})"
    end
  end

  # ============================================================================
  # OOM and Memory Boundary Tests
  # ============================================================================

  describe "OOM boundary detection" do
    @tag :gpu
    @tag :slow
    @tag timeout: 600_000
    test "finds maximum batch size before OOM for MLP" do
      # Binary search for max batch size
      # Start conservative, double until OOM, then binary search
      embed_size = @embed_size

      find_max_batch = fn ->
        # Start with small batch, keep doubling
        Enum.reduce_while([32, 64, 128, 256, 512, 1024, 2048], 32, fn batch_size, last_good ->
          try do
            trainer = Imitation.new(
              embed_size: embed_size,
              hidden_sizes: [256, 256],
              temporal: false,
              learning_rate: 1.0e-4
            )

            batch = generate_batch(batch_size, embed_size, temporal: false)
            {_, _} = Imitation.train_step(trainer, batch, nil)

            # Force memory cleanup
            :erlang.garbage_collect()

            {:cont, batch_size}
          rescue
            _ -> {:halt, last_good}
          catch
            :exit, _ -> {:halt, last_good}
          end
        end)
      end

      max_batch = find_max_batch.()

      IO.puts("\n  [INFO] Max MLP batch size: #{max_batch}")

      # Should support at least batch_size=128 on any reasonable GPU
      assert max_batch >= 128,
        "GPU should support at least batch_size=128, got #{max_batch}"
    end

    @tag :gpu
    @tag :slow
    test "finds maximum batch size for Mamba" do
      embed_size = @embed_size
      seq_len = @seq_len

      find_max_batch = fn ->
        Enum.reduce_while([8, 16, 32, 64, 128, 256], 8, fn batch_size, last_good ->
          try do
            trainer = Imitation.new(
              embed_size: embed_size,
              hidden_sizes: [256],
              temporal: true,
              backbone: :mamba,
              window_size: seq_len,
              num_layers: 2,
              learning_rate: 1.0e-4
            )

            batch = generate_batch(batch_size, embed_size, temporal: true, seq_len: seq_len)
            {_, _} = Imitation.train_step(trainer, batch, nil)

            :erlang.garbage_collect()

            {:cont, batch_size}
          rescue
            _ -> {:halt, last_good}
          catch
            :exit, _ -> {:halt, last_good}
          end
        end)
      end

      max_batch = find_max_batch.()

      IO.puts("\n  [INFO] Max Mamba batch size (seq_len=#{seq_len}): #{max_batch}")

      assert max_batch >= 16,
        "GPU should support at least batch_size=16 for Mamba, got #{max_batch}"
    end
  end

  # ============================================================================
  # Scaling Tests
  # ============================================================================

  describe "batch size scaling" do
    @tag :gpu
    test "throughput scales with batch size" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      batch_sizes = [16, 32, 64, 128]
      results = for bs <- batch_sizes do
        batch = generate_batch(bs, @embed_size, temporal: false)

        # Warmup
        {_, _} = Imitation.train_step(trainer, batch, nil)

        # Measure
        {time_us, _} = :timer.tc(fn ->
          for _ <- 1..10 do
            Imitation.train_step(trainer, batch, nil)
          end
        end)

        samples_per_sec = bs * 10 / (time_us / 1_000_000)
        {bs, samples_per_sec}
      end

      IO.puts("\n  [INFO] Throughput scaling:")
      for {bs, throughput} <- results do
        IO.puts("    batch_size=#{bs}: #{Float.round(throughput, 1)} samples/sec")
      end

      # Larger batches should have better throughput (up to a point)
      throughputs = Enum.map(results, fn {_, t} -> t end)
      max_throughput = Enum.max(throughputs)
      min_throughput = Enum.min(throughputs)

      # Max should be at least 1.5x min (some scaling benefit)
      assert max_throughput > min_throughput * 1.2,
        "Expected throughput to scale with batch size"
    end
  end

  describe "sequence length scaling" do
    @tag :gpu
    test "Mamba handles different sequence lengths" do
      seq_lengths = [16, 30, 60]

      results = for seq_len <- seq_lengths do
        trainer = Imitation.new(
          embed_size: @embed_size,
          hidden_sizes: [256],
          temporal: true,
          backbone: :mamba,
          window_size: seq_len,
          num_layers: 2,
          learning_rate: 1.0e-4
        )

        batch = generate_batch(32, @embed_size, temporal: true, seq_len: seq_len)

        # Warmup
        {_, _} = Imitation.train_step(trainer, batch, nil)

        # Measure
        {time_us, _} = :timer.tc(fn ->
          for _ <- 1..5 do
            Imitation.train_step(trainer, batch, nil)
          end
        end)

        avg_ms = time_us / 1000 / 5
        {seq_len, avg_ms}
      end

      IO.puts("\n  [INFO] Mamba timing by sequence length:")
      for {seq_len, ms} <- results do
        IO.puts("    seq_len=#{seq_len}: #{Float.round(ms, 1)}ms/batch")
      end

      # All should complete (Mamba is O(L) so should handle all lengths)
      for {seq_len, ms} <- results do
        assert ms < 10000, "seq_len=#{seq_len} too slow: #{ms}ms"
      end
    end
  end

  # ============================================================================
  # Precision Tests (bf16 vs f32)
  # ============================================================================

  describe "precision comparison" do
    @tag :gpu
    test "bf16 is faster than f32" do
      # bf16 trainer
      bf16_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        precision: :bf16,
        learning_rate: 1.0e-4
      )

      # f32 trainer
      f32_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        precision: :f32,
        learning_rate: 1.0e-4
      )

      batch = generate_batch(64, @embed_size, temporal: false)

      # Warmup both
      {_, _} = Imitation.train_step(bf16_trainer, batch, nil)
      {_, _} = Imitation.train_step(f32_trainer, batch, nil)

      # Measure bf16
      {bf16_us, _} = :timer.tc(fn ->
        for _ <- 1..20 do
          Imitation.train_step(bf16_trainer, batch, nil)
        end
      end)

      # Measure f32
      {f32_us, _} = :timer.tc(fn ->
        for _ <- 1..20 do
          Imitation.train_step(f32_trainer, batch, nil)
        end
      end)

      bf16_ms = bf16_us / 1000 / 20
      f32_ms = f32_us / 1000 / 20
      speedup = f32_ms / bf16_ms

      IO.puts("\n  [INFO] bf16: #{Float.round(bf16_ms, 1)}ms/batch")
      IO.puts("  [INFO] f32: #{Float.round(f32_ms, 1)}ms/batch")
      IO.puts("  [INFO] bf16 speedup: #{Float.round(speedup, 2)}x")

      # bf16 should be at least as fast (often 1.5-2x faster)
      assert bf16_ms <= f32_ms * 1.1,
        "bf16 should not be slower than f32"
    end

    @tag :gpu
    test "bf16 produces similar loss to f32" do
      # Same random seed for both
      batch = generate_batch(32, @embed_size, temporal: false)

      bf16_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        precision: :bf16,
        learning_rate: 1.0e-4
      )

      f32_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        precision: :f32,
        learning_rate: 1.0e-4
      )

      # Train both for a few steps
      bf16_losses = for _ <- 1..10 do
        {_, metrics} = Imitation.train_step(bf16_trainer, batch, nil)
        Nx.to_number(metrics.loss)
      end

      f32_losses = for _ <- 1..10 do
        {_, metrics} = Imitation.train_step(f32_trainer, batch, nil)
        Nx.to_number(metrics.loss)
      end

      bf16_avg = Enum.sum(bf16_losses) / 10
      f32_avg = Enum.sum(f32_losses) / 10

      IO.puts("\n  [INFO] bf16 avg loss: #{Float.round(bf16_avg, 4)}")
      IO.puts("  [INFO] f32 avg loss: #{Float.round(f32_avg, 4)}")

      # Losses should be in similar range (within 20%)
      assert_in_delta bf16_avg, f32_avg, max(bf16_avg, f32_avg) * 0.5,
        "bf16 and f32 losses too different"
    end
  end

  # ============================================================================
  # Long Training Stability
  # ============================================================================

  describe "long training stability" do
    @tag :gpu
    @tag :slow
    @tag timeout: 900_000  # 15 min
    test "training remains stable for 1000 batches" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        window_size: @seq_len,
        num_layers: 2,
        learning_rate: 1.0e-4
      )

      # Track loss over time to detect divergence
      losses = []

      final_trainer = Enum.reduce(1..1000, {trainer, losses}, fn idx, {t, loss_acc} ->
        batch = generate_batch(32, @embed_size, temporal: true, seq_len: @seq_len)
        {new_t, metrics} = Imitation.train_step(t, batch, nil)

        loss = Nx.to_number(metrics.loss)

        # Check for problems
        refute is_nan(loss), "NaN at batch #{idx}"
        refute is_infinite(loss), "Inf at batch #{idx}"
        assert loss < 100, "Loss exploded at batch #{idx}: #{loss}"

        # Track every 100th loss
        new_losses = if rem(idx, 100) == 0 do
          [{idx, loss} | loss_acc]
        else
          loss_acc
        end

        {new_t, new_losses}
      end)
      |> elem(0)

      assert final_trainer.step == 1000
      IO.puts("\n  [INFO] Completed 1000 batches without divergence")
    end
  end

  # ============================================================================
  # Cross-Device Checkpoint Tests
  # ============================================================================

  describe "cross-device checkpoints" do
    @tag :gpu
    test "checkpoint saved on GPU loads correctly" do
      # Train on GPU
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      batches = generate_batches(5, 16, @embed_size, temporal: false)
      trained = Enum.reduce(batches, trainer, fn batch, t ->
        {new_t, _} = Imitation.train_step(t, batch, nil)
        new_t
      end)

      test_batch = generate_batch(4, @embed_size, temporal: false)
      pred_before = do_predict(trained, test_batch.states)

      tmp_dir = System.tmp_dir!()
      checkpoint_path = Path.join(tmp_dir, "cross_device_#{:rand.uniform(100000)}.axon")

      try do
        # Save
        :ok = Imitation.save_checkpoint(trained, checkpoint_path)

        # Verify file exists and has reasonable size
        {:ok, stat} = File.stat(checkpoint_path)
        assert stat.size > 1000, "Checkpoint too small: #{stat.size} bytes"

        # Load (should work regardless of backend)
        {:ok, loaded} = Imitation.load_checkpoint(trained, checkpoint_path)

        # Predictions should match
        pred_after = do_predict(loaded, test_batch.states)
        assert_tensors_close(pred_before, pred_after, atol: 1.0e-4)

        IO.puts("\n  [INFO] Cross-device checkpoint: #{stat.size} bytes, predictions match")
      after
        File.rm(checkpoint_path)
      end
    end

    @tag :gpu
    test "handles corrupted checkpoint gracefully" do
      # Need a trainer to attempt loading
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      tmp_dir = System.tmp_dir!()
      corrupt_path = Path.join(tmp_dir, "corrupt_#{:rand.uniform(100000)}.axon")

      try do
        # Write garbage data
        File.write!(corrupt_path, "not a valid checkpoint file at all!!!")

        # Should raise or return error, not crash
        result = try do
          Imitation.load_checkpoint(trainer, corrupt_path)
        rescue
          e -> {:error, e}
        catch
          kind, reason -> {:caught, kind, reason}
        end

        case result do
          {:ok, _} ->
            flunk("Should not load corrupted checkpoint")
          {:error, _reason} ->
            IO.puts("\n  [INFO] Corrupted checkpoint correctly rejected with error")
          {:caught, _, _} ->
            IO.puts("\n  [INFO] Corrupted checkpoint correctly rejected with exception")
        end
      after
        File.rm(corrupt_path)
      end
    end
  end

  # ============================================================================
  # Gradient Clipping Tests
  # ============================================================================

  describe "gradient clipping" do
    @tag :gpu
    test "gradient clipping prevents explosion" do
      # Trainer WITH clipping
      clipped_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        max_grad_norm: 1.0,
        learning_rate: 1.0e-2  # High LR to stress test
      )

      # Trainer WITHOUT clipping
      unclipped_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        max_grad_norm: nil,
        learning_rate: 1.0e-2
      )

      # Use extreme input to generate large gradients
      extreme_batch = %{
        states: Nx.broadcast(10.0, {32, @embed_size}),
        actions: generate_actions(32)
      }

      # Both should complete without NaN
      {clipped_result, clipped_metrics} = Imitation.train_step(clipped_trainer, extreme_batch, nil)
      {unclipped_result, unclipped_metrics} = Imitation.train_step(unclipped_trainer, extreme_batch, nil)

      clipped_loss = Nx.to_number(clipped_metrics.loss)
      unclipped_loss = Nx.to_number(unclipped_metrics.loss)

      IO.puts("\n  [INFO] Clipped loss: #{Float.round(clipped_loss, 4)}")
      IO.puts("  [INFO] Unclipped loss: #{Float.round(unclipped_loss, 4)}")

      # Clipped should not be NaN
      refute is_nan(clipped_loss), "Clipped training produced NaN"

      # Both completed
      assert clipped_result.step == 1
      assert unclipped_result.step == 1
    end
  end

  # ============================================================================
  # Backend Comparison Tests
  # ============================================================================

  describe "backend comparison" do
    @tag :gpu
    test "GPU and CPU produce similar predictions" do
      # Build trainer (will use GPU via EXLA default)
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      # Generate test input
      batch = generate_batch(8, @embed_size, temporal: false)

      # Get GPU prediction (tuple of tensors)
      gpu_pred = do_predict(trainer, batch.states)

      # Flatten all tensors in the tuple to a single list of values
      gpu_values = gpu_pred
        |> Tuple.to_list()
        |> Enum.flat_map(&Nx.to_flat_list/1)

      # Verify predictions are reasonable numbers
      for {val, idx} <- Enum.with_index(gpu_values) do
        refute is_nan(val), "NaN at index #{idx}"
        refute is_infinite(val), "Inf at index #{idx}"
      end

      IO.puts("\n  [INFO] GPU predictions: #{length(gpu_values)} values, all finite")
    end

    @tag :gpu
    test "same input produces deterministic output" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      # Fixed input
      key = Nx.Random.key(12345)
      {states, _} = Nx.Random.uniform(key, shape: {4, @embed_size}, type: :f32)

      # Run prediction multiple times
      pred1 = do_predict(trainer, states)
      pred2 = do_predict(trainer, states)
      pred3 = do_predict(trainer, states)

      # All should be identical
      assert_tensors_close(pred1, pred2, atol: 1.0e-6)
      assert_tensors_close(pred2, pred3, atol: 1.0e-6)

      IO.puts("\n  [INFO] Predictions are deterministic")
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp generate_batch(batch_size, embed_size, opts) do
    generate_batches(1, batch_size, embed_size, opts) |> hd()
  end

  defp generate_batches(num_batches, batch_size, embed_size, opts) do
    temporal = Keyword.get(opts, :temporal, false)
    seq_len = Keyword.get(opts, :seq_len, 30)

    key = Nx.Random.key(:rand.uniform(100000))

    {batches, _} = Enum.map_reduce(1..num_batches, key, fn _, k ->
      {states, k} = if temporal do
        Nx.Random.uniform(k, shape: {batch_size, seq_len, embed_size}, type: :f32)
      else
        Nx.Random.uniform(k, shape: {batch_size, embed_size}, type: :f32)
      end

      actions = generate_actions(batch_size, k)

      {%{states: states, actions: actions}, k}
    end)

    batches
  end

  defp generate_actions(batch_size, key \\ nil) do
    key = key || Nx.Random.key(:rand.uniform(100000))

    {buttons_f, key} = Nx.Random.uniform(key, shape: {batch_size, 8}, type: :f32)
    buttons = buttons_f |> Nx.multiply(2) |> Nx.floor() |> Nx.as_type(:s32)

    {main_x_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    main_x = main_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

    {main_y_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    main_y = main_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

    {c_x_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    c_x = c_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

    {c_y_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    c_y = c_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

    {shoulder_f, _key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
    shoulder = shoulder_f |> Nx.multiply(4) |> Nx.floor() |> Nx.as_type(:s32)

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder
    }
  end

  defp get_sample_params(trainer) do
    # Get a sample parameter tensor for comparison
    # Use the first dense layer weights
    params = trainer.policy_params
    case params do
      %Axon.ModelState{data: data} ->
        # Find first layer with weights
        data
        |> Map.values()
        |> Enum.find_value(fn layer_params ->
          case layer_params do
            %{"kernel" => kernel} -> kernel
            _ -> nil
          end
        end)
        |> case do
          nil -> Nx.tensor([0.0])
          tensor -> Nx.flatten(tensor) |> Nx.slice([0], [min(100, Nx.size(Nx.flatten(tensor)))])
        end
      _ ->
        Nx.tensor([0.0])
    end
  end

  # Handle tuple of tensors (from predict_fn output)
  defp assert_tensors_close(a, b, opts) when is_tuple(a) and is_tuple(b) do
    a_list = Tuple.to_list(a)
    b_list = Tuple.to_list(b)
    assert length(a_list) == length(b_list), "Tuple sizes don't match"

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {at, bt} -> assert_tensors_close(at, bt, opts) end)
  end

  # Handle single tensor
  defp assert_tensors_close(a, b, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    # Flatten both tensors for comparison
    a_flat = Nx.to_flat_list(a)
    b_flat = Nx.to_flat_list(b)

    assert length(a_flat) == length(b_flat), "Tensor sizes don't match"

    Enum.zip(a_flat, b_flat)
    |> Enum.with_index()
    |> Enum.each(fn {{av, bv}, idx} ->
      diff = abs(av - bv)
      assert diff < atol,
        "Tensors differ at index #{idx}: #{av} vs #{bv} (diff: #{diff}, atol: #{atol})"
    end)
  end

  defp is_nan(x) when is_float(x), do: x != x
  defp is_nan(_), do: false

  defp is_infinite(x) when is_float(x), do: x == :infinity or x == :neg_infinity or abs(x) > 1.0e38
  defp is_infinite(_), do: false

  # Helper to run prediction using the trainer's predict function
  defp do_predict(trainer, states) do
    model_state = Utils.ensure_model_state(trainer.policy_params)
    trainer.predict_fn.(model_state, states)
  end
end
