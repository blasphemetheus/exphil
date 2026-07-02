defmodule ExPhil.Training.Callbacks.TestEval do
  @moduledoc "Evaluate on held-out test set at end of training."

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Imitation, Output}

  @impl true
  def init(_opts), do: %{}

  @impl true
  def on_train_end(state, cb) do
    test_ds = state.pipeline.test_dataset

    if test_ds && test_ds != nil do
      Output.puts("\n  Evaluating on test set...")
      # Use val batches creation logic but for test data
      alias ExPhil.Training.Data
      test_batches =
        if state.opts[:temporal] do
          Data.batched_sequences(test_ds,
            batch_size: state.opts[:batch_size] || 32,
            shuffle: false, lazy: true, gpu: false,
            window_size: state.opts[:window_size] || 60,
            stride: state.opts[:stride] || 5
          )
        else
          Data.batched(test_ds, batch_size: state.opts[:batch_size] || 32, shuffle: false)
        end
        |> Enum.to_list()

      if test_batches != [] do
        metrics = Imitation.evaluate(state.trainer, test_batches, max_concurrency: 1)
        Output.puts("  Test loss: #{Float.round(metrics.loss * 1.0, 4)}")
        state = ExPhil.Training.TrainingState.put_meta(state, :test_loss, metrics.loss)
        {:cont, state, cb}
      else
        {:cont, state, cb}
      end
    else
      {:cont, state, cb}
    end
  end
end
