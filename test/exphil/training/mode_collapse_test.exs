defmodule ExPhil.Training.ModeCollapseTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline, Trainer, TrainingState, Callback}
  alias ExPhil.Training.Imitation

  @doc """
  Smoke test: train 200 steps on 5 files and verify the model doesn't collapse
  to predicting a single action for all inputs.

  Mode collapse symptoms:
  - Action diversity = 1 (same prediction for every input)
  - All button predictions identical (all 0 or all 1)
  - Loss plateaus at a high value

  This test catches bad default configurations (e.g., bf16 causing numerical
  instability, LR schedules with broken warmup).
  """

  describe "mode collapse detection" do
    @tag timeout: 300_000
    test "model learns diverse actions after 200 steps" do
      opts = Config.parse_args([
        "--backbone", "mamba",
        "--replays", "./replays/huggingface",
        "--max-files", "5",
        "--epochs", "1",
        "--batch-size", "16",
        "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)

      # Train 200 steps
      {batch_stream, _} = Pipeline.batch_stream(pipeline, [])

      {trainer, losses} =
        batch_stream
        |> Stream.take(200)
        |> Enum.reduce({trainer, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          loss = Nx.to_number(metrics.loss)
          {new_t, [loss | losses]}
        end)

      losses = Enum.reverse(losses)

      # Check 1: Loss should decrease
      first_10_avg = Enum.take(losses, 10) |> Enum.sum() |> Kernel./(10)
      last_10_avg = Enum.take(losses, -10) |> Enum.sum() |> Kernel./(10)

      assert last_10_avg < first_10_avg,
        "Loss should decrease: first 10 avg=#{Float.round(first_10_avg, 2)}, " <>
        "last 10 avg=#{Float.round(last_10_avg, 2)}"

      # Check 2: No NaN losses
      nan_count = Enum.count(losses, &(not is_number(&1) or &1 != &1))
      assert nan_count == 0, "Found #{nan_count} NaN losses"

      # Check 3: Action diversity > 1
      # Run predictions on a few val batches and check diversity
      val_batches = pipeline.val_batches || []
      if length(val_batches) >= 3 do
        sample = Enum.take(val_batches, 3)

        combos = Enum.flat_map(sample, fn batch ->
          {buttons_logits, mx_logits, my_logits, _, _, _} =
            trainer.predict_fn.(trainer.policy_params, batch.states)

          pred_buttons = Nx.greater(Nx.sigmoid(buttons_logits), 0.5) |> Nx.as_type(:u8)
          pred_mx = Nx.argmax(mx_logits, axis: -1)
          pred_my = Nx.argmax(my_logits, axis: -1)

          batch_size = elem(Nx.shape(pred_buttons), 0)
          for i <- 0..min(batch_size - 1, 15) do
            {Nx.to_flat_list(pred_buttons[i]), Nx.to_number(pred_mx[i]), Nx.to_number(pred_my[i])}
          end
        end)

        unique_combos = MapSet.new(combos) |> MapSet.size()
        assert unique_combos > 1,
          "Action diversity should be > 1 after 200 steps, got #{unique_combos} unique combos (mode collapse)"
      end
    end
  end
end
