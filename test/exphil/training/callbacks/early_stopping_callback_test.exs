defmodule ExPhil.Training.Callbacks.EarlyStoppingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Callbacks.EarlyStopping
  alias ExPhil.Training.TrainingState

  describe "init/1" do
    test "default patience is 5" do
      cb = EarlyStopping.init([])
      assert cb.patience == 5
      assert cb.best_loss == nil
      assert cb.wait == 0
    end

    test "accepts custom patience" do
      cb = EarlyStopping.init(patience: 3, min_delta: 0.001)
      assert cb.patience == 3
      assert cb.min_delta == 0.001
    end
  end

  describe "on_epoch_end/2" do
    test "continues when loss improves" do
      cb = EarlyStopping.init(patience: 3)
      state = %TrainingState{val_loss: 1.0}

      {:cont, _state, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.best_loss == 1.0
      assert cb.wait == 0

      state = %TrainingState{val_loss: 0.8}
      {:cont, _state, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.best_loss == 0.8
      assert cb.wait == 0
    end

    test "increments wait when loss doesn't improve" do
      cb = EarlyStopping.init(patience: 3)

      state = %TrainingState{val_loss: 1.0}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)

      state = %TrainingState{val_loss: 1.1}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.wait == 1

      state = %TrainingState{val_loss: 1.2}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.wait == 2
    end

    test "halts after patience exhausted" do
      cb = EarlyStopping.init(patience: 2)

      state = %TrainingState{val_loss: 1.0}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)

      state = %TrainingState{val_loss: 1.1}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)

      state = %TrainingState{val_loss: 1.2}
      {:halt, state, _cb} = EarlyStopping.on_epoch_end(state, cb)
      assert state.halt == true
    end

    test "resets wait on improvement" do
      cb = EarlyStopping.init(patience: 3)

      state = %TrainingState{val_loss: 1.0}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)

      state = %TrainingState{val_loss: 1.1}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.wait == 1

      # Improvement resets
      state = %TrainingState{val_loss: 0.5}
      {:cont, _, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.wait == 0
      assert cb.best_loss == 0.5
    end

    test "handles nil val_loss gracefully" do
      cb = EarlyStopping.init(patience: 3)
      state = %TrainingState{val_loss: nil, train_loss: nil}

      {:cont, _state, cb} = EarlyStopping.on_epoch_end(state, cb)
      assert cb.wait == 0
    end
  end
end
