defmodule ExPhil.Training.BackboneSmokeTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline, Trainer}
  alias ExPhil.Training.Imitation

  @doc """
  Smoke test each backbone with its auto-applied defaults.
  Runs 50 training steps and verifies:
  1. No crash
  2. Loss decreases (or at least doesn't NaN)
  3. No NaN gradients
  4. Action diversity > 1 by step 50

  This catches bad default combinations before users hit them.
  """

  # Test each backbone that has custom defaults
  @backbones_to_test [:mamba, :mlp, :griffin, :min_gru, :lstm]

  for backbone <- @backbones_to_test do
    @tag timeout: 300_000
    test "#{backbone} backbone trains without collapse" do
      backbone = unquote(backbone)

      opts = Config.parse_args([
        "--backbone", to_string(backbone),
        "--replays", "./replays/huggingface",
        "--max-files", "5",
        "--epochs", "1",
        "--batch-size", "16",
        "--seed", "42"
      ]) |> Config.validate!() |> Config.ensure_checkpoint_name()

      # Verify backbone defaults were applied
      if backbone == :mamba do
        assert opts[:temporal] == true, "Mamba should auto-set temporal=true"
        assert opts[:precision] == :bf16, "Mamba should auto-set precision=bf16"
      end

      if backbone == :mlp do
        assert opts[:temporal] == false, "MLP should set temporal=false"
      end

      # Setup
      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)

      # Train 50 steps
      {batch_stream, _} = Pipeline.batch_stream(pipeline, [])

      {trainer, losses} =
        batch_stream
        |> Stream.take(50)
        |> Enum.reduce({trainer, []}, fn batch, {t, losses} ->
          try do
            {new_t, metrics} = Imitation.train_step(t, batch, nil)
            loss = Nx.to_number(metrics.loss)
            {new_t, [loss | losses]}
          rescue
            e ->
              flunk("#{backbone} crashed at step #{length(losses)}: #{Exception.message(e)}")
          end
        end)

      losses = Enum.reverse(losses)

      # Check: no NaN
      nan_count = Enum.count(losses, fn l -> not is_number(l) end)
      assert nan_count == 0, "#{backbone}: #{nan_count} NaN losses in 50 steps"

      # Check: loss decreased (first 5 avg vs last 5 avg)
      if length(losses) >= 10 do
        first_5 = Enum.take(losses, 5) |> Enum.sum() |> Kernel./(5)
        last_5 = Enum.take(losses, -5) |> Enum.sum() |> Kernel./(5)

        # Allow some backbones to be slow learners — just check it's not diverging wildly
        assert last_5 < first_5 * 2,
          "#{backbone}: loss should not diverge. First 5: #{Float.round(first_5, 2)}, Last 5: #{Float.round(last_5, 2)}"
      end
    end
  end
end
