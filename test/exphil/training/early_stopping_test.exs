defmodule ExPhil.Training.EarlyStoppingTest do
  use ExUnit.Case, async: true
  doctest ExPhil.Training.EarlyStopping

  alias ExPhil.Training.EarlyStopping

  describe "init/1" do
    test "uses default values" do
      state = EarlyStopping.init()

      assert state.patience == 5
      assert state.min_delta == 0.01
      assert state.best_loss == nil
      assert state.epochs_without_improvement == 0
    end

    test "accepts custom options" do
      state = EarlyStopping.init(patience: 10, min_delta: 0.001)

      assert state.patience == 10
      assert state.min_delta == 0.001
    end
  end

  describe "check/2" do
    test "first check sets baseline and continues" do
      state = EarlyStopping.init()
      {new_state, action} = EarlyStopping.check(state, 1.5)

      assert action == :continue
      assert new_state.best_loss == 1.5
      assert new_state.best_epoch == 1
      assert new_state.epochs_without_improvement == 0
    end

    test "improvement resets counter" do
      state = EarlyStopping.init(patience: 3, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)
      # No improvement (0.05 < min_delta would be)
      {state, _} = EarlyStopping.check(state, 0.95)
      # Big improvement
      {state, _} = EarlyStopping.check(state, 0.5)

      assert state.best_loss == 0.5
      assert state.epochs_without_improvement == 0
    end

    test "no improvement increments counter" do
      state = EarlyStopping.init(patience: 5)
      {state, _} = EarlyStopping.check(state, 1.0)
      # Worse
      {state, _} = EarlyStopping.check(state, 1.1)
      # Still worse than best
      {state, _} = EarlyStopping.check(state, 1.05)

      assert state.best_loss == 1.0
      assert state.epochs_without_improvement == 2
    end

    test "stops when patience exhausted" do
      state = EarlyStopping.init(patience: 2)
      {state, :continue} = EarlyStopping.check(state, 1.0)
      {state, :continue} = EarlyStopping.check(state, 1.1)
      {state, :stop} = EarlyStopping.check(state, 1.2)

      assert state.epochs_without_improvement == 2
    end
  end

  describe "is_best?/2" do
    test "returns true for first check" do
      state = EarlyStopping.init()
      assert EarlyStopping.is_best?(state, 1.0)
    end

    test "returns true when loss is lower" do
      state = EarlyStopping.init()
      {state, _} = EarlyStopping.check(state, 1.0)

      assert EarlyStopping.is_best?(state, 0.9)
      refute EarlyStopping.is_best?(state, 1.0)
      refute EarlyStopping.is_best?(state, 1.1)
    end
  end
end
