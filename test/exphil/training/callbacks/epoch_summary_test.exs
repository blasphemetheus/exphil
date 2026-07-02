defmodule ExPhil.Training.Callbacks.EpochSummaryTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Callbacks.EpochSummary
  alias ExPhil.Training.TrainingState

  describe "on_epoch_end/2" do
    test "does not crash with minimal state" do
      cb = EpochSummary.init([])

      state = %TrainingState{
        epoch: 1,
        epochs: 5,
        train_loss: 2.5,
        val_loss: 2.3,
        best_val_loss: nil,
        epoch_time: 10,
        history: []
      }

      {:cont, _state, _cb} = EpochSummary.on_epoch_end(state, cb)
    end

    test "handles nil val_loss" do
      cb = EpochSummary.init([])

      state = %TrainingState{
        epoch: 1,
        epochs: 5,
        train_loss: 2.5,
        val_loss: nil,
        best_val_loss: nil,
        epoch_time: 10,
        history: []
      }

      {:cont, _state, _cb} = EpochSummary.on_epoch_end(state, cb)
    end

    test "shows sparkline with history" do
      cb = EpochSummary.init([])

      state = %TrainingState{
        epoch: 3,
        epochs: 5,
        train_loss: 1.5,
        val_loss: 1.6,
        best_val_loss: 1.6,
        epoch_time: 10,
        history: [
          %{train_loss: 3.0, val_loss: 2.8},
          %{train_loss: 2.0, val_loss: 1.9}
        ]
      }

      {:cont, _state, _cb} = EpochSummary.on_epoch_end(state, cb)
    end
  end
end
