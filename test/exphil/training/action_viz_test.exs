defmodule ExPhil.Training.ActionVizTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.ActionViz

  describe "new/1" do
    test "creates empty visualizer" do
      viz = ActionViz.new()
      assert viz.total == 0
      assert viz.grid_size == 17
      assert Enum.sort(Map.keys(viz.button_counts)) == [:a, :b, :d_up, :l, :r, :x, :y, :z]
    end

    test "accepts custom grid size" do
      viz = ActionViz.new(grid_size: 9)
      assert viz.grid_size == 9
    end
  end

  describe "record/2" do
    test "records button presses from list" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [:a, :b], main_x: 8, main_y: 8})

      assert viz.total == 1
      assert viz.button_counts[:a] == 1
      assert viz.button_counts[:b] == 1
      assert viz.button_counts[:x] == 0
    end

    test "records button presses from bitmask" do
      viz =
        ActionViz.new()
        # Bitmask: bit 0 = A, bit 1 = B
        |> ActionViz.record(%{buttons: 0b00000011, main_x: 8, main_y: 8})

      assert viz.button_counts[:a] == 1
      assert viz.button_counts[:b] == 1
      assert viz.button_counts[:x] == 0
    end

    test "records stick positions" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{main_x: 0, main_y: 16, c_x: 8, c_y: 8})

      assert viz.main_stick_counts == %{{0, 16} => 1}
      assert viz.c_stick_counts == %{{8, 8} => 1}
    end

    test "records shoulder position" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{shoulder: 3})

      assert viz.shoulder_counts == %{3 => 1}
    end

    test "accumulates multiple records" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [:a], main_x: 8, main_y: 8})
        |> ActionViz.record(%{buttons: [:a, :b], main_x: 8, main_y: 8})
        |> ActionViz.record(%{buttons: [:b], main_x: 0, main_y: 0})

      assert viz.total == 3
      assert viz.button_counts[:a] == 2
      assert viz.button_counts[:b] == 2
      assert viz.main_stick_counts[{8, 8}] == 2
      assert viz.main_stick_counts[{0, 0}] == 1
    end
  end

  describe "record_batch/2" do
    test "records multiple actions" do
      actions = [
        %{buttons: [:a], main_x: 8, main_y: 8},
        %{buttons: [:b], main_x: 0, main_y: 16},
        %{buttons: [:a], main_x: 16, main_y: 0}
      ]

      viz = ActionViz.new() |> ActionViz.record_batch(actions)

      assert viz.total == 3
      assert viz.button_counts[:a] == 2
      assert viz.button_counts[:b] == 1
    end
  end

  describe "button_rates/1" do
    test "calculates button press rates" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [:a]})
        |> ActionViz.record(%{buttons: [:a]})
        |> ActionViz.record(%{buttons: [:b]})
        |> ActionViz.record(%{buttons: []})

      rates = ActionViz.button_rates(viz)

      assert rates[:a] == 0.5
      assert rates[:b] == 0.25
      assert rates[:x] == 0.0
    end

    test "handles empty visualizer" do
      viz = ActionViz.new()
      rates = ActionViz.button_rates(viz)
      assert rates[:a] == 0.0
    end
  end

  describe "to_stats/1" do
    test "returns stats map" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [:a], main_x: 8, main_y: 8})
        |> ActionViz.record(%{buttons: [:a], main_x: 8, main_y: 8})

      stats = ActionViz.to_stats(viz)

      assert stats.total_samples == 2
      assert stats.button_rates[:a] == 1.0
      assert stats.stick_entropy.main >= 0.0
    end
  end

  describe "print_summary/2" do
    import ExUnit.CaptureIO

    test "handles empty visualizer" do
      viz = ActionViz.new()

      output =
        capture_io(:stderr, fn ->
          ActionViz.print_summary(viz)
        end)

      assert output =~ "No actions recorded"
    end

    test "prints button distribution" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [:a], main_x: 8, main_y: 8})

      output =
        capture_io(:stderr, fn ->
          ActionViz.print_summary(viz, show_sticks: false, show_shoulder: false)
        end)

      assert output =~ "Action Distribution"
      assert output =~ "Buttons:"
      assert output =~ "A"
      assert output =~ "100.0%"
    end

    test "prints stick heatmaps" do
      viz =
        ActionViz.new()
        |> ActionViz.record(%{buttons: [], main_x: 8, main_y: 8})

      output =
        capture_io(:stderr, fn ->
          ActionViz.print_summary(viz, show_shoulder: false)
        end)

      # Side-by-side headers
      assert output =~ "Main Stick"
      assert output =~ "C-Stick"
      # Neutral percentage
      assert output =~ "Neutral:"
    end
  end
end
