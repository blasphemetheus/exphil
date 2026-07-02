defmodule ExPhil.Training.TrainingState do
  @moduledoc """
  Shared state that flows through the training loop and all callbacks.

  This struct is the single source of truth during training. Callbacks can
  read any field and update fields relevant to their domain. The training
  loop updates `:trainer`, `:epoch`, `:step`, and `:batch_metrics` — everything
  else is managed by callbacks.

  ## Fields

  ### Set by training loop
  - `:trainer` — The `Imitation` trainer struct (model params, optimizer state)
  - `:epoch` — Current epoch number (1-indexed)
  - `:step` — Global step count across all epochs
  - `:batch_metrics` — Metrics from the most recent batch (`%{loss: float}`)
  - `:epochs` — Total epochs configured

  ### Set by callbacks
  - `:val_loss` — Validation loss (set by Validation callback)
  - `:train_loss` — Average training loss for current epoch
  - `:history` — List of `%{epoch, train_loss, val_loss, time_seconds}` per epoch
  - `:best_val_loss` — Best validation loss seen so far (Checkpoint callback)
  - `:halt` — Set to `true` by EarlyStopping to stop training
  - `:epoch_time` — Seconds elapsed in current epoch
  - `:epoch_losses` — Accumulated losses for current epoch (list of floats)

  ### Set at init
  - `:pipeline` — Reference to the `Pipeline` struct
  - `:opts` — Training options (keyword list)
  """

  @type t :: %__MODULE__{
          trainer: struct() | nil,
          pipeline: struct() | nil,
          epoch: non_neg_integer(),
          epochs: non_neg_integer(),
          step: non_neg_integer(),
          batch_idx: non_neg_integer(),
          batch_metrics: map(),
          val_loss: float() | nil,
          train_loss: float() | nil,
          best_val_loss: float() | nil,
          history: [map()],
          epoch_losses: [float()],
          epoch_time: non_neg_integer(),
          halt: boolean(),
          opts: keyword(),
          event_counts: map(),
          meta: map()
        }

  defstruct [
    trainer: nil,
    pipeline: nil,
    epoch: 0,
    epochs: 0,
    step: 0,
    batch_idx: 0,
    batch_metrics: %{},
    val_loss: nil,
    train_loss: nil,
    best_val_loss: nil,
    history: [],
    epoch_losses: [],
    epoch_time: 0,
    halt: false,
    opts: [],
    # Event counts for filter predicates ({:every, n}, {:after, n})
    event_counts: %{},
    meta: %{}
  ]

  @doc "Get a value from the extensible meta map."
  def get_meta(%__MODULE__{meta: meta}, key, default \\ nil) do
    Map.get(meta, key, default)
  end

  @doc "Put a value in the extensible meta map."
  def put_meta(%__MODULE__{meta: meta} = state, key, value) do
    %{state | meta: Map.put(meta, key, value)}
  end

  @doc "Update a value in the extensible meta map."
  def update_meta(%__MODULE__{meta: meta} = state, key, default, fun) do
    %{state | meta: Map.update(meta, key, default, fun)}
  end

  # ============================================================================
  # Metric Accumulators (Axon-inspired)
  # ============================================================================

  @doc """
  Update a running average metric.

  Stores `{sum, count}` in meta and returns the current average.

  ## Example

      state = TrainingState.running_average(state, :train_loss, batch_loss)
      avg = TrainingState.get_metric(state, :train_loss)
  """
  def running_average(state, key, value) when is_number(value) do
    update_meta(state, {:metric, key}, {value, 1}, fn {sum, count} ->
      {sum + value, count + 1}
    end)
  end

  def running_average(state, _key, _value), do: state

  @doc """
  Update a running sum metric.
  """
  def running_sum(state, key, value) when is_number(value) do
    update_meta(state, {:metric, key}, 0.0, fn sum -> sum + value end)
  end

  def running_sum(state, _key, _value), do: state

  @doc """
  Get the current value of an accumulated metric.

  For running_average, returns the mean. For running_sum, returns the sum.
  """
  def get_metric(state, key) do
    case get_meta(state, {:metric, key}) do
      {sum, count} when count > 0 -> sum / count
      {_, 0} -> 0.0
      sum when is_number(sum) -> sum
      nil -> nil
    end
  end

  @doc """
  Reset a metric accumulator (call at epoch start).
  """
  def reset_metric(state, key) do
    put_meta(state, {:metric, key}, nil)
  end
end
