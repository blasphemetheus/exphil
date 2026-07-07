defmodule ExPhil.Training.Callbacks.Validation do
  @moduledoc """
  Runs validation after each epoch and stores val_loss in TrainingState.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Imitation, Output}

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

    # Nx.to_number returns :nan/:infinity ATOMS for non-finite losses. The
    # trainer halts on non-finite TRAIN loss, but a val-only NaN would
    # otherwise flow through silently (Checkpoint skips it via is_number,
    # EarlyStopping counts it as no-improvement) — make it loud.
    if not is_number(val_loss) do
      Output.warning(
        "Validation loss is non-finite (#{inspect(val_loss)}) at epoch #{state.epoch} — " <>
          "model may be diverging; best-checkpoint saving is suspended for this epoch"
      )
    end

    state = %{state | val_loss: val_loss}
    {:cont, state, cb}
  end
end
