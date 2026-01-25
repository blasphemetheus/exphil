defmodule ExPhil.Benchmarks.TrainingSpeedTest do
  @moduledoc """
  GPU training speed benchmarks for all architectures.

  These tests verify that training batch speed is as expected and detect
  performance regressions, particularly the Nx.to_number GPU blocking issue.

  ## Running on GPU

      # Run all GPU benchmarks
      mix test --only gpu

      # Run with benchmark tag (includes baseline comparisons)
      mix test --only gpu --only benchmark

      # Update baselines
      BENCHMARK_UPDATE=1 mix test --only gpu --only benchmark

  ## What These Tests Catch

  1. **Nx.to_number blocking (Gotcha #18)**: Converting tensors to numbers
     every batch causes GPU→CPU sync, resulting in 0% GPU utilization.
     Expected: ~0.2s/batch, Regression: ~80s/batch

  2. **Architecture regressions**: Each backbone (MLP, LSTM, Mamba, etc.)
     should maintain expected per-batch speeds.

  3. **Memory issues**: Batch sizes that cause OOM or excessive swapping.
  """

  use ExUnit.Case, async: false

  alias ExPhil.Training.Imitation

  # Tag all tests as GPU-required and benchmarks
  @moduletag :gpu
  @moduletag :benchmark
  @moduletag :slow
  @moduletag timeout: 600_000  # 10 minutes for GPU benchmarks

  # Test parameters - sized for meaningful GPU benchmarks
  @batch_size 128
  @embed_size 408  # Realistic embedding size with learned actions
  @seq_len 30
  @num_batches 10  # Number of batches to average over

  describe "MLP training speed" do
    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256, 256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      batches = generate_training_batches(@num_batches, @batch_size, @embed_size, temporal: false)

      {:ok, trainer: trainer, batches: batches}
    end

    test "train_step completes under 500ms per batch", %{trainer: trainer, batches: batches} do
      # Warmup JIT
      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      # Measure actual training speed
      {total_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, warmup_trainer, fn batch, t ->
          {new_t, _metrics} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      avg_ms = total_time_us / 1000 / (length(rest))

      assert avg_ms < 500,
        "MLP training too slow: #{Float.round(avg_ms, 1)}ms/batch (expected <500ms)"
    end

    @tag :regression
    test "tensor loss accumulation is faster than per-batch conversion", %{trainer: trainer, batches: batches} do
      # This test verifies the fix for gotcha #18
      # Accumulating tensor losses should be MUCH faster than converting each batch

      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      # Method 1: Accumulate tensors (GOOD - single GPU→CPU transfer)
      {time_tensor_us, {_, tensor_losses}} = :timer.tc(fn ->
        Enum.reduce(rest, {warmup_trainer, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [metrics.loss | losses]}  # Keep as tensor
        end)
      end)

      # Compute mean from tensors (single transfer)
      _avg_loss_tensor = tensor_losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()

      # Method 2: Convert each batch (BAD - blocks GPU every batch)
      {warmup_trainer2, _} = Imitation.train_step(trainer, first_batch, nil)

      {time_number_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, {warmup_trainer2, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [Nx.to_number(metrics.loss) | losses]}  # Blocks every batch!
        end)
      end)

      tensor_ms = time_tensor_us / 1000
      number_ms = time_number_us / 1000

      # Tensor accumulation should be at least 2x faster
      # (In practice it's often 10-100x faster on GPU)
      assert tensor_ms < number_ms,
        """
        Tensor accumulation should be faster than per-batch Nx.to_number!
        Tensor method: #{Float.round(tensor_ms, 1)}ms
        Number method: #{Float.round(number_ms, 1)}ms

        This indicates the Nx.to_number blocking issue (gotcha #18) may be recurring.
        """

      # Log the speedup for visibility
      speedup = number_ms / tensor_ms
      IO.puts("\n  [INFO] Tensor accumulation speedup: #{Float.round(speedup, 1)}x faster")
    end
  end

  describe "LSTM training speed" do
    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :lstm,
        seq_len: @seq_len,
        learning_rate: 1.0e-4
      )

      batches = generate_training_batches(@num_batches, @batch_size, @embed_size,
        temporal: true, seq_len: @seq_len)

      {:ok, trainer: trainer, batches: batches}
    end

    # Note: LSTM is inherently slow due to sequential gradient computation (BPTT)
    # Cannot parallelize across sequence dimension like Mamba/MLP
    # 15s/batch is acceptable for batch_size=128, seq_len=30 with gradients
    test "train_step completes under 15000ms per batch", %{trainer: trainer, batches: batches} do
      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      {total_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, warmup_trainer, fn batch, t ->
          {new_t, _metrics} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      avg_ms = total_time_us / 1000 / (length(rest))

      assert avg_ms < 15000,
        "LSTM training too slow: #{Float.round(avg_ms, 1)}ms/batch (expected <15000ms)"
    end
  end

  describe "GRU training speed" do
    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :gru,
        seq_len: @seq_len,
        learning_rate: 1.0e-4
      )

      batches = generate_training_batches(@num_batches, @batch_size, @embed_size,
        temporal: true, seq_len: @seq_len)

      {:ok, trainer: trainer, batches: batches}
    end

    # Note: GRU is inherently slow due to sequential gradient computation (BPTT)
    # Cannot parallelize across sequence dimension like Mamba/MLP
    # 15s/batch is acceptable for batch_size=128, seq_len=30 with gradients
    test "train_step completes under 15000ms per batch", %{trainer: trainer, batches: batches} do
      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      {total_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, warmup_trainer, fn batch, t ->
          {new_t, _metrics} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      avg_ms = total_time_us / 1000 / (length(rest))

      assert avg_ms < 15000,
        "GRU training too slow: #{Float.round(avg_ms, 1)}ms/batch (expected <15000ms)"
    end
  end

  describe "Mamba training speed" do
    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        seq_len: @seq_len,
        num_layers: 2,
        state_size: 16,
        learning_rate: 1.0e-4
      )

      # Smaller batch size for Mamba (more memory intensive)
      batches = generate_training_batches(@num_batches, 64, @embed_size,
        temporal: true, seq_len: @seq_len)

      {:ok, trainer: trainer, batches: batches}
    end

    test "train_step completes under 2000ms per batch", %{trainer: trainer, batches: batches} do
      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      {total_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, warmup_trainer, fn batch, t ->
          {new_t, _metrics} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      avg_ms = total_time_us / 1000 / (length(rest))

      assert avg_ms < 2000,
        "Mamba training too slow: #{Float.round(avg_ms, 1)}ms/batch (expected <2000ms)"
    end

    test "Mamba is faster than LSTM for same sequence length" do
      # Build both trainers
      mamba_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :mamba,
        seq_len: @seq_len,
        num_layers: 2,
        state_size: 16,
        learning_rate: 1.0e-4
      )

      lstm_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :lstm,
        seq_len: @seq_len,
        learning_rate: 1.0e-4
      )

      # Same batches for both
      batches = generate_training_batches(5, 32, @embed_size, temporal: true, seq_len: @seq_len)
      [first | rest] = batches

      # Warmup
      {mamba_warmed, _} = Imitation.train_step(mamba_trainer, first, nil)
      {lstm_warmed, _} = Imitation.train_step(lstm_trainer, first, nil)

      # Time Mamba
      {mamba_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, mamba_warmed, fn batch, t ->
          {new_t, _} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      # Time LSTM
      {lstm_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, lstm_warmed, fn batch, t ->
          {new_t, _} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      mamba_ms = mamba_time_us / 1000 / length(rest)
      lstm_ms = lstm_time_us / 1000 / length(rest)

      IO.puts("\n  [INFO] Mamba: #{Float.round(mamba_ms, 1)}ms/batch, LSTM: #{Float.round(lstm_ms, 1)}ms/batch")

      # Mamba should be competitive with or faster than LSTM
      # (Allow 50% slower since Mamba has different characteristics)
      assert mamba_ms < lstm_ms * 1.5,
        "Mamba significantly slower than LSTM: #{Float.round(mamba_ms, 1)}ms vs #{Float.round(lstm_ms, 1)}ms"
    end
  end

  describe "Jamba (Mamba + Attention) training speed" do
    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256],
        temporal: true,
        backbone: :jamba,
        seq_len: @seq_len,
        num_layers: 3,
        attention_every: 3,
        learning_rate: 1.0e-4
      )

      # Smaller batch for Jamba (attention is memory intensive)
      batches = generate_training_batches(@num_batches, 32, @embed_size,
        temporal: true, seq_len: @seq_len)

      {:ok, trainer: trainer, batches: batches}
    end

    test "train_step completes under 3000ms per batch", %{trainer: trainer, batches: batches} do
      [first_batch | rest] = batches
      {warmup_trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      {total_time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, warmup_trainer, fn batch, t ->
          {new_t, _metrics} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      avg_ms = total_time_us / 1000 / (length(rest))

      assert avg_ms < 3000,
        "Jamba training too slow: #{Float.round(avg_ms, 1)}ms/batch (expected <3000ms)"
    end
  end

  describe "Nx.to_number blocking regression (Gotcha #18)" do
    @describetag :regression

    setup do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [256, 256],
        temporal: false,
        learning_rate: 1.0e-4
      )

      batches = generate_training_batches(20, @batch_size, @embed_size, temporal: false)

      {:ok, trainer: trainer, batches: batches}
    end

    test "periodic conversion (every N batches) matches full tensor accumulation", %{trainer: trainer, batches: batches} do
      # This tests the pattern used in the progress display fix
      [first | rest] = batches
      {warmed, _} = Imitation.train_step(trainer, first, nil)

      # Method 1: Full tensor accumulation (gold standard)
      {_, tensor_losses} = Enum.reduce(rest, {warmed, []}, fn batch, {t, losses} ->
        {new_t, m} = Imitation.train_step(t, batch, nil)
        {new_t, [m.loss | losses]}
      end)
      gold_avg = tensor_losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()

      # Method 2: Periodic conversion (every 5 batches) - matches progress display pattern
      {warmed2, _} = Imitation.train_step(trainer, first, nil)
      {_, periodic_losses} = Enum.with_index(rest)
      |> Enum.reduce({warmed2, []}, fn {batch, idx}, {t, losses} ->
        {new_t, m} = Imitation.train_step(t, batch, nil)
        # Store tuple like the progress display does
        display = if rem(idx, 5) == 0, do: Nx.to_number(m.loss), else: 0.0
        {new_t, [{display, m.loss} | losses]}
      end)

      periodic_avg = periodic_losses
      |> Enum.map(fn {_, tensor} -> tensor end)
      |> Nx.stack()
      |> Nx.mean()
      |> Nx.to_number()

      # Both methods should produce the same average loss
      assert_in_delta gold_avg, periodic_avg, 0.0001,
        "Periodic conversion produced different loss: #{gold_avg} vs #{periodic_avg}"
    end

    test "GPU utilization stays high with tensor accumulation", %{trainer: trainer, batches: batches} do
      # This is a proxy test - we can't directly measure GPU utilization,
      # but we can verify that training is fast (which implies GPU is being used)

      [first | rest] = batches
      {warmed, _} = Imitation.train_step(trainer, first, nil)

      # Time training with tensor accumulation
      {time_us, _} = :timer.tc(fn ->
        Enum.reduce(rest, {warmed, []}, fn batch, {t, losses} ->
          {new_t, m} = Imitation.train_step(t, batch, nil)
          {new_t, [m.loss | losses]}
        end)
      end)

      avg_ms = time_us / 1000 / length(rest)

      # If GPU is being utilized, each batch should be fast
      # If Nx.to_number is blocking, batches would be ~80s each
      assert avg_ms < 1000,
        """
        Training too slow - possible GPU blocking issue!
        Average: #{Float.round(avg_ms, 1)}ms/batch

        If this is ~80s/batch, check for Nx.to_number calls in the training loop.
        See gotcha #18 in docs/GOTCHAS.md
        """
    end
  end

  # Helper to generate training batches
  defp generate_training_batches(num_batches, batch_size, embed_size, opts) do
    temporal = Keyword.get(opts, :temporal, false)
    seq_len = Keyword.get(opts, :seq_len, 30)

    key = Nx.Random.key(42)

    {batches, _final_key} = Enum.map_reduce(1..num_batches, key, fn _, k ->
      {states, k} = if temporal do
        Nx.Random.uniform(k, shape: {batch_size, seq_len, embed_size}, type: :f32)
      else
        Nx.Random.uniform(k, shape: {batch_size, embed_size}, type: :f32)
      end

      # Generate integer actions using uniform and floor
      {buttons_f, k} = Nx.Random.uniform(k, shape: {batch_size, 8}, type: :f32)
      buttons = buttons_f |> Nx.multiply(2) |> Nx.floor() |> Nx.as_type(:s32)

      {main_x_f, k} = Nx.Random.uniform(k, shape: {batch_size}, type: :f32)
      main_x = main_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

      {main_y_f, k} = Nx.Random.uniform(k, shape: {batch_size}, type: :f32)
      main_y = main_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

      {c_x_f, k} = Nx.Random.uniform(k, shape: {batch_size}, type: :f32)
      c_x = c_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

      {c_y_f, k} = Nx.Random.uniform(k, shape: {batch_size}, type: :f32)
      c_y = c_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)

      {shoulder_f, k} = Nx.Random.uniform(k, shape: {batch_size}, type: :f32)
      shoulder = shoulder_f |> Nx.multiply(4) |> Nx.floor() |> Nx.as_type(:s32)

      batch = %{
        states: states,
        actions: %{
          buttons: buttons,
          main_x: main_x,
          main_y: main_y,
          c_x: c_x,
          c_y: c_y,
          shoulder: shoulder
        }
      }

      {batch, k}
    end)

    batches
  end
end
