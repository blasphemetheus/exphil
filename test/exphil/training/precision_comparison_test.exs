defmodule ExPhil.Training.PrecisionComparisonTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline, Trainer}
  alias ExPhil.Training.Imitation

  @doc """
  Compare bf16 vs f32 training on the same data with the same seed.
  Identifies if bf16 causes numerical instability (vanishing gradients,
  mode collapse) for the current model size (3.7M params).
  """

  describe "bf16 vs f32 precision" do
    @tag timeout: 600_000
    test "bf16 and f32 both learn (loss decreases, no NaN)" do
      results = for precision <- [:f32, :bf16] do
        opts = Config.parse_args([
          "--backbone", "mamba",
          "--replays", "./replays/huggingface",
          "--max-files", "5",
          "--epochs", "1",
          "--batch-size", "16",
          "--precision", to_string(precision),
          "--lr-schedule", "constant",
          "--seed", "42"
        ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

        pipeline = Pipeline.setup!(opts)
        trainer = Trainer.new(pipeline, opts)

        {batch_stream, _} = Pipeline.batch_stream(pipeline, [])

        {_trainer, losses} =
          batch_stream
          |> Stream.take(100)
          |> Enum.reduce({trainer, []}, fn batch, {t, losses} ->
            {new_t, metrics} = Imitation.train_step(t, batch, nil)
            loss = Nx.to_number(metrics.loss)
            {new_t, [loss | losses]}
          end)

        losses = Enum.reverse(losses)
        {precision, losses}
      end

      [{:f32, f32_losses}, {:bf16, bf16_losses}] = results

      # Both should have no NaN
      f32_nans = Enum.count(f32_losses, &(not is_number(&1)))
      bf16_nans = Enum.count(bf16_losses, &(not is_number(&1)))

      assert f32_nans == 0, "f32 had #{f32_nans} NaN losses"
      assert bf16_nans == 0, "bf16 had #{bf16_nans} NaN losses"

      # Both should decrease
      f32_first = Enum.take(f32_losses, 10) |> Enum.sum() |> Kernel./(10)
      f32_last = Enum.take(f32_losses, -10) |> Enum.sum() |> Kernel./(10)
      bf16_first = Enum.take(bf16_losses, 10) |> Enum.sum() |> Kernel./(10)
      bf16_last = Enum.take(bf16_losses, -10) |> Enum.sum() |> Kernel./(10)

      IO.puts("\n  f32:  #{Float.round(f32_first, 2)} → #{Float.round(f32_last, 2)}")
      IO.puts("  bf16: #{Float.round(bf16_first, 2)} → #{Float.round(bf16_last, 2)}")

      assert f32_last < f32_first, "f32 loss should decrease"
      assert bf16_last < bf16_first,
        "bf16 loss should decrease (#{Float.round(bf16_first, 2)} → #{Float.round(bf16_last, 2)}). " <>
        "If this fails, bf16 causes numerical instability for this model."

      # Final losses should be in the same ballpark (within 2x)
      ratio = bf16_last / max(f32_last, 0.001)
      assert ratio < 3.0,
        "bf16 loss (#{Float.round(bf16_last, 2)}) is #{Float.round(ratio, 1)}x worse than " <>
        "f32 (#{Float.round(f32_last, 2)}). bf16 may be unsuitable for this model."
    end
  end
end
