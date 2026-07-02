defmodule ExPhil.Training.Callbacks.EarlyStopping do
  @moduledoc """
  Stop training when validation loss stops improving.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.Output

  @impl true
  def init(opts) do
    %{
      patience: Keyword.get(opts, :patience, 5),
      min_delta: Keyword.get(opts, :min_delta, 0.0),
      best_loss: nil,
      wait: 0
    }
  end

  @impl true
  def on_epoch_end(state, cb) do
    loss = state.val_loss || state.train_loss

    if is_number(loss) do
      improved = cb.best_loss == nil or loss < cb.best_loss - cb.min_delta

      if improved do
        {:cont, state, %{cb | best_loss: loss, wait: 0}}
      else
        new_wait = cb.wait + 1

        if new_wait >= cb.patience do
          Output.puts("  Early stopping: no improvement for #{cb.patience} epochs")
          {:halt, %{state | halt: true}, %{cb | wait: new_wait}}
        else
          {:cont, state, %{cb | wait: new_wait}}
        end
      end
    else
      {:cont, state, cb}
    end
  end
end
