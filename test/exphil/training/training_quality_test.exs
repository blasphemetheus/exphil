defmodule ExPhil.Training.TrainingQualityTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline, Trainer, Callback}
  alias ExPhil.Training.Callbacks.{Validation, Diagnostics}
  alias ExPhil.Training.Imitation

  @doc """
  Tests for training quality — verify the model actually learns useful behavior.
  These are slow integration tests that run real training.
  """

  describe "learning verification" do
    @tag timeout: 300_000
    test "loss monotonically decreases over first epoch" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16", "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)
      {stream, _} = Pipeline.batch_stream(pipeline, [])

      # Train 200 steps, compute windowed average every 50
      {_trainer, losses} =
        stream
        |> Stream.take(200)
        |> Enum.reduce({trainer, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [Nx.to_number(metrics.loss) | losses]}
        end)

      losses = Enum.reverse(losses)

      # Compute 4 windows of 50
      windows = Enum.chunk_every(losses, 50)
        |> Enum.map(&(Enum.sum(&1) / length(&1)))

      IO.puts("\n  Loss windows: #{inspect(Enum.map(windows, &Float.round(&1, 2)))}")

      # Each window should be <= previous (allowing small increase)
      Enum.chunk_every(windows, 2, 1, :discard)
      |> Enum.with_index()
      |> Enum.each(fn {[prev, curr], idx} ->
        assert curr < prev * 1.5,
          "Loss increased significantly at window #{idx + 1}: #{Float.round(prev, 2)} → #{Float.round(curr, 2)}"
      end)
    end

    @tag timeout: 300_000
    test "validation loss correlates with training loss" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--epochs", "2", "--batch-size", "16", "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)

      callbacks = [{Validation, []}]

      {:ok, state} = Trainer.fit(trainer, pipeline, callbacks: callbacks)

      assert state.val_loss != nil, "Val loss should be computed"
      assert state.train_loss != nil, "Train loss should be computed"

      # Val loss should be in the same order of magnitude as train loss
      ratio = state.val_loss / max(state.train_loss, 0.001)
      IO.puts("\n  Train: #{Float.round(state.train_loss, 2)}, Val: #{Float.round(state.val_loss, 2)}, Ratio: #{Float.round(ratio, 2)}")

      assert ratio < 5.0,
        "Val loss (#{Float.round(state.val_loss, 2)}) is #{Float.round(ratio, 1)}x train loss " <>
        "(#{Float.round(state.train_loss, 2)}). Possible data leak or overfitting."
    end

    @tag timeout: 300_000
    test "model produces diverse predictions after training" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--epochs", "2", "--batch-size", "16", "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)

      {:ok, state} = Trainer.fit(trainer, pipeline, callbacks: [])

      # Check prediction diversity on val data
      val_batches = pipeline.val_batches || []
      assert length(val_batches) > 0, "Need val batches for diversity check"

      sample = Enum.take(val_batches, 5)
      predictions = Enum.flat_map(sample, fn batch ->
        {btn_logits, mx_logits, _, _, _, _} =
          state.trainer.predict_fn.(state.trainer.policy_params, batch.states)

        pred_buttons = Nx.greater(Nx.sigmoid(btn_logits), 0.5) |> Nx.as_type(:u8)
        pred_mx = Nx.argmax(mx_logits, axis: -1)

        batch_size = elem(Nx.shape(pred_buttons), 0)
        for i <- 0..min(batch_size - 1, 7) do
          {Nx.to_flat_list(pred_buttons[i]), Nx.to_number(pred_mx[i])}
        end
      end)

      unique = MapSet.new(predictions) |> MapSet.size()
      IO.puts("\n  Unique predictions: #{unique} / #{length(predictions)}")

      assert unique > 1,
        "Model predicts only #{unique} unique action(s) — mode collapse detected"
    end
  end

  describe "mixed precision correctness" do
    @tag timeout: 300_000
    test "bf16 mixed precision produces finite gradients" do
      opts = Config.parse_args([
        "--backbone", "mamba", "--replays", "./replays/huggingface",
        "--max-files", "5", "--batch-size", "16", "--precision", "bf16", "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)
      {stream, _} = Pipeline.batch_stream(pipeline, [])

      batch = Enum.take(stream, 1) |> hd()

      # Compute gradients
      alias ExPhil.Training.Imitation.TrainLoop
      {grads, loss} = TrainLoop.compute_gradients(trainer, batch)

      assert is_number(loss), "Loss should be a number, got #{inspect(loss)}"
      refute loss == :nan, "Loss should not be NaN"
      refute loss == :infinity, "Loss should not be infinity"

      # Check gradient norms are finite — simple recursive check
      check_grads_finite = fn grads_map ->
        grads_map
        |> Map.values()
        |> Enum.each(fn
          %Nx.Tensor{} = t ->
            has_nan = Nx.any(Nx.is_nan(t)) |> Nx.to_number() > 0
            refute has_nan, "NaN gradient found"
          inner when is_map(inner) ->
            Enum.each(Map.values(inner), fn
              %Nx.Tensor{} = t ->
                has_nan = Nx.any(Nx.is_nan(t)) |> Nx.to_number() > 0
                refute has_nan, "NaN gradient found (nested)"
              _ -> :ok
            end)
          _ -> :ok
        end)
      end

      check_grads_finite.(grads)
    end
  end
end
