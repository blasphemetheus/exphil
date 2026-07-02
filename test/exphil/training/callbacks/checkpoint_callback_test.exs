defmodule ExPhil.Training.Callbacks.CheckpointTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Callbacks.Checkpoint
  alias ExPhil.Training.TrainingState

  describe "init/1" do
    test "defaults" do
      cb = Checkpoint.init([])
      assert cb.save_best == true
      assert cb.save_every == nil
      assert cb.save_every_batches == nil
      assert cb.best_val_loss == nil
    end

    test "custom options" do
      cb = Checkpoint.init(save_best: false, save_every: 5, checkpoint_path: "test.axon")
      assert cb.save_best == false
      assert cb.save_every == 5
      assert cb.checkpoint_path == "test.axon"
    end
  end

  describe "on_epoch_end/2" do
    test "tracks best val loss" do
      cb = Checkpoint.init(save_best: false)
      state = %TrainingState{val_loss: 2.0, opts: []}

      {:cont, state, _cb} = Checkpoint.on_epoch_end(state, cb)
      # best_val_loss not set when save_best is false and no checkpoint
      assert state.best_val_loss == nil
    end

    test "does not crash without checkpoint path" do
      cb = Checkpoint.init(save_best: true, checkpoint_path: nil)
      state = %TrainingState{val_loss: 1.5, epoch: 1, opts: []}

      {:cont, _state, _cb} = Checkpoint.on_epoch_end(state, cb)
    end
  end

  describe "on_batch_end/2" do
    test "no-op without save_every_batches" do
      cb = Checkpoint.init([])
      state = %TrainingState{step: 100, opts: []}

      {:cont, _state, _cb} = Checkpoint.on_batch_end(state, cb)
    end
  end
end
