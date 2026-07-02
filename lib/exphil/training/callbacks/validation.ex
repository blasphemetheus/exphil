defmodule ExPhil.Training.Callbacks.Validation do
  @moduledoc """
  Runs validation after each epoch and stores val_loss in TrainingState.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.Imitation

  @impl true
  def init(_opts), do: %{}

  @impl true
  def on_epoch_end(state, cb) do
    val_batches = state.pipeline.val_batches

    val_loss =
      if val_batches && val_batches != [] do
        metrics = Imitation.evaluate(state.trainer, val_batches, max_concurrency: 1)
        :erlang.garbage_collect()
        metrics.loss
      else
        state.train_loss
      end

    state = %{state | val_loss: val_loss}
    {:cont, state, cb}
  end
end
