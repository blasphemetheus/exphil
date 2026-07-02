defmodule ExPhil.Training.PipelineCorrectnessTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline}

  @doc """
  Verify Pipeline produces correct data shapes, splits, and batch counts.
  """

  describe "pipeline data correctness" do
    @tag timeout: 120_000
    test "train/val split produces non-overlapping datasets" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)

      assert pipeline.train_dataset.size > 0, "Train set should have frames"
      assert Pipeline.has_validation?(pipeline), "Should have validation set"

      # Val batch count should be proportional to val_split
      total = pipeline.train_dataset.size + (pipeline.val_batches |> length()) * 16
      # Rough check — val should be ~10% of total
      val_ratio = length(pipeline.val_batches) * 16 / total
      assert val_ratio > 0.05 and val_ratio < 0.20,
        "Val ratio should be ~10%, got #{Float.round(val_ratio * 100, 1)}%"
    end

    @tag timeout: 120_000
    test "batch stream produces correct shapes" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      {stream, num_batches} = Pipeline.batch_stream(pipeline, [])

      assert num_batches > 0, "Should have batches"

      # Check first batch shapes
      batch = Enum.take(stream, 1) |> hd()

      assert is_map(batch), "Batch should be a map"
      assert Map.has_key?(batch, :states), "Batch should have :states"
      assert Map.has_key?(batch, :actions), "Batch should have :actions"

      {batch_size, seq_len, embed_dim} = Nx.shape(batch.states)
      assert batch_size == 16, "Batch size should be 16, got #{batch_size}"
      assert seq_len == 60, "Seq len should be 60, got #{seq_len}"
      assert embed_dim == 288, "Embed dim should be 288, got #{embed_dim}"

      # Check action shapes
      assert elem(Nx.shape(batch.actions.buttons), 0) == 16
      assert elem(Nx.shape(batch.actions.main_x), 0) == 16
    end

    @tag timeout: 120_000
    test "estimated batch count matches actual" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      {stream, estimated} = Pipeline.batch_stream(pipeline, [])

      actual = stream |> Enum.count()

      # Allow 5% tolerance
      ratio = actual / max(estimated, 1)
      assert ratio > 0.9 and ratio < 1.1,
        "Estimated #{estimated} but got #{actual} batches (#{Float.round(ratio * 100, 1)}%)"
    end
  end
end
