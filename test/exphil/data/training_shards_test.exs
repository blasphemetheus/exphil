defmodule ExPhil.Data.TrainingShardsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.TrainingShards

  describe "action packing" do
    test "round-trips discretized actions exactly (targets tolerate no drift)" do
      actions = [
        %{buttons: %{a: true, b: false, x: true, y: false, z: true, l: false, r: true, d_up: false},
          main_x: 0, main_y: 15, c_x: 8, c_y: 3, shoulder: 3},
        %{buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
          main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0},
        %{buttons: %{a: false, b: true, x: false, y: true, z: false, l: true, r: false, d_up: true},
          main_x: 15, main_y: 0, c_x: 1, c_y: 14, shoulder: 1}
      ]

      packed = TrainingShards.__pack_actions__(actions)
      # 6 bytes per frame
      assert byte_size(packed) == 6 * length(actions)
      assert TrainingShards.__unpack_actions__(packed) == actions
    end

    test "every button bit is independent" do
      for b <- [:a, :b, :x, :y, :z, :l, :r, :d_up] do
        buttons = Map.new([:a, :b, :x, :y, :z, :l, :r, :d_up], fn k -> {k, k == b} end)
        a = %{buttons: buttons, main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0}
        assert [^a] = TrainingShards.__unpack_actions__(TrainingShards.__pack_actions__([a]))
      end
    end
  end

  describe "manifest" do
    test "empty dir yields empty stats" do
      dir = Path.join(System.tmp_dir!(), "ts_empty_#{System.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      assert TrainingShards.stats(dir) == %{files: 0, frames: 0, sequences: 0}
      File.rm_rf!(dir)
    end
  end
end
