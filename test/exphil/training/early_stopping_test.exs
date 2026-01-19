defmodule ExPhil.Training.EarlyStoppingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.EarlyStopping

  describe "init/1" do
    test "initializes with default values" do
      state = EarlyStopping.init()
      assert state.patience == 5
      assert state.min_delta == 0.01
      assert state.best_loss == nil
      assert state.epochs_without_improvement == 0
      assert state.best_epoch == nil
    end

    test "accepts custom patience" do
      state = EarlyStopping.init(patience: 10)
      assert state.patience == 10
    end

    test "accepts custom min_delta" do
      state = EarlyStopping.init(min_delta: 0.005)
      assert state.min_delta == 0.005
    end

    test "accepts multiple options" do
      state = EarlyStopping.init(patience: 3, min_delta: 0.1)
      assert state.patience == 3
      assert state.min_delta == 0.1
    end
  end

  describe "check/2" do
    test "first epoch sets baseline and continues" do
      state = EarlyStopping.init()
      {new_state, decision} = EarlyStopping.check(state, 1.0)

      assert decision == :continue
      assert new_state.best_loss == 1.0
      assert new_state.best_epoch == 1
      assert new_state.epochs_without_improvement == 0
    end

    test "improvement resets counter" do
      state = EarlyStopping.init(min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      # Improve by more than min_delta
      {state, decision} = EarlyStopping.check(state, 0.98)

      assert decision == :continue
      assert state.best_loss == 0.98
      assert state.epochs_without_improvement == 0
    end

    test "no improvement increments counter" do
      state = EarlyStopping.init(patience: 5, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      # No improvement (loss same or higher)
      {state, decision} = EarlyStopping.check(state, 1.0)

      assert decision == :continue
      assert state.best_loss == 1.0
      assert state.epochs_without_improvement == 1
    end

    test "small improvement below min_delta counts as no improvement" do
      state = EarlyStopping.init(patience: 5, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      # Improve by less than min_delta
      {state, decision} = EarlyStopping.check(state, 0.995)

      assert decision == :continue
      assert state.best_loss == 1.0  # Not updated
      assert state.epochs_without_improvement == 1
    end

    test "stops when patience exhausted" do
      state = EarlyStopping.init(patience: 2, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)  # epoch 1

      {state, :continue} = EarlyStopping.check(state, 1.0)  # epoch 2, no improvement (1/2)
      {state, :stop} = EarlyStopping.check(state, 1.0)  # epoch 3, no improvement (2/2) -> stop

      assert state.epochs_without_improvement == 2
    end

    test "improvement after some epochs resets counter" do
      state = EarlyStopping.init(patience: 3, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      # No improvement for 2 epochs
      {state, :continue} = EarlyStopping.check(state, 1.0)
      {state, :continue} = EarlyStopping.check(state, 1.0)
      assert state.epochs_without_improvement == 2

      # Then improve
      {state, :continue} = EarlyStopping.check(state, 0.9)
      assert state.epochs_without_improvement == 0
      assert state.best_loss == 0.9
    end

    test "patience of 1 stops immediately after no improvement" do
      state = EarlyStopping.init(patience: 1, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      {_state, :stop} = EarlyStopping.check(state, 1.0)
    end

    test "large patience allows many non-improving epochs" do
      state = EarlyStopping.init(patience: 10, min_delta: 0.01)
      {state, _} = EarlyStopping.check(state, 1.0)

      # 9 epochs without improvement should continue
      state = Enum.reduce(1..9, state, fn _, acc ->
        {new_state, :continue} = EarlyStopping.check(acc, 1.0)
        new_state
      end)

      assert state.epochs_without_improvement == 9

      # 10th epoch without improvement should stop
      {_state, :stop} = EarlyStopping.check(state, 1.0)
    end
  end

  describe "summary/1" do
    test "returns summary map" do
      state = EarlyStopping.init(patience: 5)
      {state, _} = EarlyStopping.check(state, 1.0)
      {state, _} = EarlyStopping.check(state, 1.1)  # no improvement

      summary = EarlyStopping.summary(state)

      assert summary.best_loss == 1.0
      assert summary.best_epoch == 1
      assert summary.epochs_without_improvement == 1
      assert summary.patience_remaining == 4
    end
  end

  describe "is_best?/2" do
    test "returns true for first check" do
      state = EarlyStopping.init()
      assert EarlyStopping.is_best?(state, 1.0)
    end

    test "returns true when better than best" do
      state = EarlyStopping.init()
      {state, _} = EarlyStopping.check(state, 1.0)

      assert EarlyStopping.is_best?(state, 0.9)
    end

    test "returns false when worse than best" do
      state = EarlyStopping.init()
      {state, _} = EarlyStopping.check(state, 1.0)

      refute EarlyStopping.is_best?(state, 1.1)
    end

    test "returns false when equal to best" do
      state = EarlyStopping.init()
      {state, _} = EarlyStopping.check(state, 1.0)

      refute EarlyStopping.is_best?(state, 1.0)
    end
  end

  describe "status_message/1" do
    test "shows initializing for nil best_loss" do
      state = EarlyStopping.init()
      assert EarlyStopping.status_message(state) == "initializing..."
    end

    test "shows new best when no epochs without improvement" do
      state = EarlyStopping.init()
      {state, _} = EarlyStopping.check(state, 1.0)

      message = EarlyStopping.status_message(state)
      assert message =~ "new best"
      assert message =~ "1.0"
    end

    test "shows patience status when epochs without improvement" do
      state = EarlyStopping.init(patience: 5)
      {state, _} = EarlyStopping.check(state, 1.0)
      {state, _} = EarlyStopping.check(state, 1.1)

      message = EarlyStopping.status_message(state)
      assert message =~ "best=1.0"
      assert message =~ "1/5"
    end
  end

  describe "integration scenarios" do
    test "typical training scenario with eventual convergence" do
      state = EarlyStopping.init(patience: 3, min_delta: 0.1)

      # Losses that decrease then plateau
      # Improvements: 5->4 (1.0), 4->3.5 (0.5), 3.5->3.45 (0.05 < 0.1 = no improvement)
      losses = [5.0, 4.0, 3.5, 3.45, 3.44, 3.43, 3.42, 3.41]

      {final_state, epochs, stopped} =
        Enum.reduce_while(Enum.with_index(losses, 1), {state, 0, false}, fn {loss, epoch}, {s, _, _} ->
          {new_state, decision} = EarlyStopping.check(s, loss)
          case decision do
            :stop -> {:halt, {new_state, epoch, true}}
            :continue -> {:cont, {new_state, epoch, false}}
          end
        end)

      assert stopped == true
      assert epochs == 6  # Stops at epoch 6 (3 epochs without improvement after epoch 3)
      assert final_state.best_loss == 3.5  # Last significant improvement
    end

    test "continuously improving model never stops" do
      state = EarlyStopping.init(patience: 2, min_delta: 0.1)

      # Always improving losses
      losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]

      {_final_state, epochs, stopped} =
        Enum.reduce_while(losses, {state, 0, false}, fn loss, {s, epoch, _} ->
          {new_state, decision} = EarlyStopping.check(s, loss)
          case decision do
            :stop -> {:halt, {new_state, epoch + 1, true}}
            :continue -> {:cont, {new_state, epoch + 1, false}}
          end
        end)

      assert stopped == false
      assert epochs == length(losses)
    end
  end
end
