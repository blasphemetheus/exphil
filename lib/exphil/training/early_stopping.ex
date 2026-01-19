defmodule ExPhil.Training.EarlyStopping do
  @moduledoc """
  Early stopping callback for training loops.

  Tracks validation loss and signals when training should stop
  if no improvement is seen for a specified number of epochs.

  ## Usage

      # Initialize state
      state = EarlyStopping.init(patience: 5, min_delta: 0.01)

      # After each epoch, check if training should stop
      {state, :continue} = EarlyStopping.check(state, val_loss)
      # or
      {state, :stop} = EarlyStopping.check(state, val_loss)

  ## Options

  - `:patience` - Number of epochs to wait for improvement (default: 5)
  - `:min_delta` - Minimum change in loss to qualify as improvement (default: 0.01)
  """

  defstruct [
    :patience,
    :min_delta,
    :best_loss,
    :epochs_without_improvement,
    :best_epoch
  ]

  @type t :: %__MODULE__{
    patience: pos_integer(),
    min_delta: float(),
    best_loss: float() | nil,
    epochs_without_improvement: non_neg_integer(),
    best_epoch: pos_integer() | nil
  }

  @doc """
  Initialize early stopping state.

  ## Options

  - `:patience` - Number of epochs to wait for improvement (default: 5)
  - `:min_delta` - Minimum improvement to reset patience counter (default: 0.01)

  ## Examples

      iex> state = EarlyStopping.init(patience: 3, min_delta: 0.005)
      iex> state.patience
      3

  """
  @spec init(keyword()) :: t()
  def init(opts \\ []) do
    %__MODULE__{
      patience: Keyword.get(opts, :patience, 5),
      min_delta: Keyword.get(opts, :min_delta, 0.01),
      best_loss: nil,
      epochs_without_improvement: 0,
      best_epoch: nil
    }
  end

  @doc """
  Check if training should stop based on validation loss.

  Returns `{updated_state, :continue}` if training should continue,
  or `{updated_state, :stop}` if patience has been exhausted.

  ## Examples

      iex> state = EarlyStopping.init(patience: 2)
      iex> {state, :continue} = EarlyStopping.check(state, 1.0)  # epoch 1, new best
      iex> {state, :continue} = EarlyStopping.check(state, 1.1)  # epoch 2, no improvement
      iex> {state, :stop} = EarlyStopping.check(state, 1.2)      # epoch 3, patience exhausted

  """
  @spec check(t(), float()) :: {t(), :continue | :stop}
  def check(%__MODULE__{best_loss: nil} = state, val_loss) do
    # First epoch - always continue, set baseline
    new_state = %{state |
      best_loss: val_loss,
      best_epoch: 1,
      epochs_without_improvement: 0
    }
    {new_state, :continue}
  end

  def check(%__MODULE__{} = state, val_loss) do
    improvement = state.best_loss - val_loss

    if improvement >= state.min_delta do
      # Significant improvement - reset counter
      new_state = %{state |
        best_loss: val_loss,
        best_epoch: (state.best_epoch || 0) + state.epochs_without_improvement + 1,
        epochs_without_improvement: 0
      }
      {new_state, :continue}
    else
      # No significant improvement
      new_epochs_without = state.epochs_without_improvement + 1

      if new_epochs_without >= state.patience do
        # Patience exhausted - stop training
        new_state = %{state | epochs_without_improvement: new_epochs_without}
        {new_state, :stop}
      else
        # Still have patience left - continue
        new_state = %{state | epochs_without_improvement: new_epochs_without}
        {new_state, :continue}
      end
    end
  end

  @doc """
  Get a summary of the early stopping state.

  Returns a map with status information useful for logging.

  ## Examples

      iex> state = EarlyStopping.init(patience: 5)
      iex> {state, _} = EarlyStopping.check(state, 1.0)
      iex> EarlyStopping.summary(state)
      %{best_loss: 1.0, best_epoch: 1, epochs_without_improvement: 0, patience_remaining: 5}

  """
  @spec summary(t()) :: map()
  def summary(%__MODULE__{} = state) do
    %{
      best_loss: state.best_loss,
      best_epoch: state.best_epoch,
      epochs_without_improvement: state.epochs_without_improvement,
      patience_remaining: state.patience - state.epochs_without_improvement
    }
  end

  @doc """
  Check if the current validation loss is a new best.

  This can be used to decide whether to save a checkpoint.

  ## Examples

      iex> state = EarlyStopping.init()
      iex> {state, _} = EarlyStopping.check(state, 1.0)
      iex> EarlyStopping.is_best?(state, 0.9)
      true
      iex> EarlyStopping.is_best?(state, 1.1)
      false

  """
  @spec is_best?(t(), float()) :: boolean()
  def is_best?(%__MODULE__{best_loss: nil}, _val_loss), do: true
  def is_best?(%__MODULE__{best_loss: best}, val_loss), do: val_loss < best

  @doc """
  Format a status message for logging.

  ## Examples

      iex> state = %EarlyStopping{best_loss: 1.0, epochs_without_improvement: 2, patience: 5}
      iex> EarlyStopping.status_message(state)
      "best=1.0, no improvement for 2/5 epochs"

  """
  @spec status_message(t()) :: String.t()
  def status_message(%__MODULE__{best_loss: nil}), do: "initializing..."
  def status_message(%__MODULE__{} = state) do
    if state.epochs_without_improvement == 0 do
      "new best loss: #{Float.round(state.best_loss, 4)}"
    else
      "best=#{Float.round(state.best_loss, 4)}, no improvement for #{state.epochs_without_improvement}/#{state.patience} epochs"
    end
  end
end
