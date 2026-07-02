defmodule ExPhil.Training.NxBatchEquivalenceTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :training
  @moduletag :gpu

  alias ExPhil.Training.{Config, Pipeline, Trainer}
  alias ExPhil.Training.Imitation

  @doc """
  Verify that Nx.Batch (lazy) and eager (Nx.stack) batch assembly produce
  identical training results. Same seed, same data, same model — only
  the batch assembly method differs.
  """

  describe "Nx.Batch equivalence" do
    @tag timeout: 300_000
    test "eager and lazy batch produce same loss values" do
      base_args = [
        "--backbone", "mamba",
        "--replays", "./replays/huggingface",
        "--max-files", "5",
        "--epochs", "1",
        "--batch-size", "16",
        "--seed", "123"
      ]

      # Run with eager batching (default)
      eager_opts = Config.parse_args(base_args)
        |> Config.validate!()
        |> Config.ensure_checkpoint_name()

      eager_pipeline = Pipeline.setup!(eager_opts)
      eager_trainer = Trainer.new(eager_pipeline, eager_opts)

      {eager_stream, _} = Pipeline.batch_stream(eager_pipeline, [])
      {_trainer, eager_losses} =
        eager_stream
        |> Stream.take(20)
        |> Enum.reduce({eager_trainer, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [Nx.to_number(metrics.loss) | losses]}
        end)

      eager_losses = Enum.reverse(eager_losses)

      # Run with Nx.Batch (lazy)
      batch_opts = Config.parse_args(base_args ++ ["--use-batch"])
        |> Config.validate!()
        |> Config.ensure_checkpoint_name()

      batch_pipeline = Pipeline.setup!(batch_opts)
      batch_trainer = Trainer.new(batch_pipeline, batch_opts)

      {batch_stream, _} = Pipeline.batch_stream(batch_pipeline, [])
      {_trainer, batch_losses} =
        batch_stream
        |> Stream.take(20)
        |> Enum.reduce({batch_trainer, []}, fn batch, {t, losses} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [Nx.to_number(metrics.loss) | losses]}
        end)

      batch_losses = Enum.reverse(batch_losses)

      # Compare losses — should be very close (floating point tolerance)
      Enum.zip(eager_losses, batch_losses)
      |> Enum.with_index()
      |> Enum.each(fn {{eager, batch}, idx} ->
        assert_in_delta eager, batch, 0.1,
          "Loss diverged at step #{idx}: eager=#{eager}, batch=#{batch}"
      end)
    end
  end
end
