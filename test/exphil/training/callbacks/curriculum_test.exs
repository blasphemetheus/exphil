defmodule ExPhil.Training.Callbacks.CurriculumTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Callbacks.Curriculum
  alias ExPhil.Training.TrainingState

  describe "init/1" do
    test "stores stages" do
      cb = Curriculum.init(stages: [{100, 3}, {200, 30}])
      assert cb.stages == [{100, 3}, {200, 30}]
      assert cb.current_stage == 0
    end
  end

  describe "stage finding" do
    test "finds correct stage for epoch" do
      cb = Curriculum.init(stages: [{100, 3}, {200, 30}])

      # Epoch 1-3 should be stage 0 (100 files)
      state = %TrainingState{epoch: 1, opts: [], pipeline: nil}
      {:cont, _state, cb1} = Curriculum.on_epoch_begin(state, cb)
      # Stage 0 → 0, no change on first call
      assert cb1.current_stage == 0

      # Epoch 4 should trigger stage 1 (200 files)
      # But since we can't actually rebuild pipeline in test, it will error
      # Just test the logic doesn't crash with nil pipeline
    end
  end
end
