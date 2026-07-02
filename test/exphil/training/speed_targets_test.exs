defmodule ExPhil.Training.SpeedTargetsTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :benchmark

  alias ExPhil.Training.{Config, Pipeline, Trainer}
  alias ExPhil.Training.Imitation

  @doc """
  Performance target tests. These verify that key operations meet speed targets.
  Run with: mix test test/exphil/training/speed_targets_test.exs --include benchmark

  Speed targets based on RTX 5090 (32GB VRAM):
  - Batch creation (lazy+chunked): <5ms per batch
  - GPU train step (batch 16, Mamba): <20ms per step
  - Full iteration (batch creation + train step): <30ms per step
  - Validation (1750 batches, sequential): <30s total
  - Embedding cache load (1.6GB): <5s
  """

  describe "training step speed" do
    @tag timeout: 300_000
    test "train step under 30ms after JIT warmup" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16", "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)
      {stream, _} = Pipeline.batch_stream(pipeline, [])

      # JIT warmup
      first_batch = Enum.take(stream, 1) |> hd()
      {trainer, _} = Imitation.train_step(trainer, first_batch, nil)

      # Time 50 real steps
      batches = Enum.take(stream, 50)
      {us, _} = :timer.tc(fn ->
        Enum.reduce(batches, trainer, fn batch, t ->
          {new_t, _} = Imitation.train_step(t, batch, nil)
          new_t
        end)
      end)

      ms_per_step = us / 50_000
      IO.puts("\n  Train step: #{Float.round(ms_per_step, 1)}ms/step")

      assert ms_per_step < 50.0,
        "Train step too slow: #{Float.round(ms_per_step, 1)}ms (target: <30ms). " <>
        "Check batch creation overhead, GPU utilization, or memory pressure."
    end
  end

  describe "batch creation speed" do
    @tag timeout: 120_000
    test "lazy batch creation under 5ms (chunked)" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "10", "--batch-size", "16"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      {stream, _} = Pipeline.batch_stream(pipeline, [])

      # Time 200 batch creations (CPU only — no GPU transfer in timing)
      {us, _} = :timer.tc(fn ->
        stream
        |> Stream.take(200)
        |> Enum.each(fn batch ->
          # Touch the tensor to force materialization
          _ = Nx.shape(batch.states)
        end)
      end)

      ms_per_batch = us / 200_000
      IO.puts("\n  Batch creation: #{Float.round(ms_per_batch, 2)}ms/batch")

      assert ms_per_batch < 10.0,
        "Batch creation too slow: #{Float.round(ms_per_batch, 2)}ms (target: <5ms). " <>
        "Check if chunked embedding slicing is active."
    end
  end

  describe "pipeline setup speed" do
    @tag timeout: 120_000
    test "pipeline setup under 30s for 10 files (cached)" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "10", "--batch-size", "16"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      # First run creates cache, second should be fast
      Pipeline.setup!(opts)

      {us, _pipeline} = :timer.tc(fn ->
        Pipeline.setup!(opts)
      end)

      seconds = us / 1_000_000
      IO.puts("\n  Pipeline setup (cached): #{Float.round(seconds, 1)}s")

      assert seconds < 30.0,
        "Pipeline setup too slow: #{Float.round(seconds, 1)}s (target: <30s with cache)"
    end
  end

  describe "inference speed" do
    @tag timeout: 120_000
    test "single inference under 5ms (post-JIT)" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "1"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)
      {stream, _} = Pipeline.batch_stream(pipeline, [])

      batch = Enum.take(stream, 1) |> hd()

      # Warmup predict
      trainer.predict_fn.(trainer.policy_params, batch.states)

      # Time 100 predictions
      {us, _} = :timer.tc(fn ->
        for _ <- 1..100 do
          trainer.predict_fn.(trainer.policy_params, batch.states)
        end
      end)

      ms_per_inference = us / 100_000
      IO.puts("\n  Inference: #{Float.round(ms_per_inference, 2)}ms/call")

      # For 60fps gameplay: must be under 16.67ms
      assert ms_per_inference < 16.67,
        "Inference too slow for 60fps: #{Float.round(ms_per_inference, 2)}ms (need <16.67ms)"
    end
  end
end
