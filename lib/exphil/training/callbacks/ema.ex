defmodule ExPhil.Training.Callbacks.EMA do
  @moduledoc """
  Exponential Moving Average of model weights.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{EMA, Output}

  @impl true
  def init(opts) do
    %{
      decay: Keyword.get(opts, :decay, 0.999),
      ema: nil
    }
  end

  @impl true
  def on_train_begin(state, cb) do
    ema = EMA.new(state.trainer.policy_params, decay: cb.decay)
    Output.puts("  EMA enabled (decay=#{cb.decay})")
    {:cont, ExPhil.Training.TrainingState.put_meta(state, :ema, ema), %{cb | ema: ema}}
  end

  @impl true
  def on_epoch_end(state, cb) do
    ema = EMA.update(cb.ema, state.trainer.policy_params)
    {:cont, ExPhil.Training.TrainingState.put_meta(state, :ema, ema), %{cb | ema: ema}}
  end
end
