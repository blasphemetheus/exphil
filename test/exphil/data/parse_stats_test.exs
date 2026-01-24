defmodule ExPhil.Data.ParseStatsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.ParseStats

  describe "new/0" do
    test "creates empty stats" do
      stats = ParseStats.new()
      assert stats.total_frames == 0
      assert stats.valid_frames == 0
      assert stats.dropped_no_player == 0
      assert stats.warnings == []
    end
  end

  describe "track_extraction/4" do
    test "tracks valid frames" do
      frames = [
        %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}},
        %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}}
      ]

      {valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert length(valid) == 2
      assert stats.total_frames == 2
      assert stats.valid_frames == 2
      assert stats.dropped_no_player == 0
    end

    test "tracks missing player" do
      frames = [
        %{players: %{2 => %{controller: %{}}}},  # Missing port 1
        %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}}
      ]

      {valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert length(valid) == 1
      assert stats.total_frames == 2
      assert stats.valid_frames == 1
      assert stats.dropped_no_player == 1
    end

    test "tracks missing opponent" do
      frames = [
        %{players: %{1 => %{controller: %{}}}},  # Missing port 2
        %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}}
      ]

      {valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert length(valid) == 1
      assert stats.dropped_no_opponent == 1
    end

    test "tracks missing controller" do
      frames = [
        %{players: %{1 => %{controller: nil}, 2 => %{controller: %{}}}},
        %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}}
      ]

      {valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert length(valid) == 1
      assert stats.dropped_no_controller == 1
    end

    test "adds warning when no valid frames" do
      frames = [
        %{players: %{2 => %{controller: %{}}}},
        %{players: %{2 => %{controller: %{}}}}
      ]

      {valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert length(valid) == 0
      assert stats.warnings != []
      assert Enum.any?(stats.warnings, &String.contains?(&1, "No valid frames"))
    end

    test "adds warning when majority missing player" do
      frames = Enum.map(1..10, fn i ->
        if i <= 6 do
          %{players: %{2 => %{controller: %{}}}}  # Missing port 1
        else
          %{players: %{1 => %{controller: %{}}, 2 => %{controller: %{}}}}
        end
      end)

      {_valid, stats} = ParseStats.track_extraction(frames, 1, 2)

      assert Enum.any?(stats.warnings, &String.contains?(&1, "wrong port"))
    end
  end

  describe "merge/1" do
    test "combines multiple stats" do
      stats1 = %ParseStats{
        total_frames: 100,
        valid_frames: 90,
        dropped_no_player: 10,
        dropped_no_opponent: 0,
        dropped_no_controller: 0,
        dropped_delay_cutoff: 0,
        warnings: ["warn1"]
      }

      stats2 = %ParseStats{
        total_frames: 200,
        valid_frames: 180,
        dropped_no_player: 5,
        dropped_no_opponent: 15,
        dropped_no_controller: 0,
        dropped_delay_cutoff: 0,
        warnings: ["warn2"]
      }

      merged = ParseStats.merge([stats1, stats2])

      assert merged.total_frames == 300
      assert merged.valid_frames == 270
      assert merged.dropped_no_player == 15
      assert merged.dropped_no_opponent == 15
      assert merged.warnings == ["warn1", "warn2"]
    end
  end

  describe "drop_percentage/1" do
    test "calculates correctly" do
      stats = %ParseStats{
        total_frames: 100,
        valid_frames: 75,
        dropped_no_player: 0,
        dropped_no_opponent: 0,
        dropped_no_controller: 0,
        dropped_delay_cutoff: 0,
        warnings: []
      }

      assert ParseStats.drop_percentage(stats) == 25.0
    end

    test "handles zero total" do
      stats = ParseStats.new()
      assert ParseStats.drop_percentage(stats) == 0.0
    end
  end

  describe "has_issues?/1" do
    test "returns true when warnings present" do
      stats = %{ParseStats.new() | warnings: ["problem"]}
      assert ParseStats.has_issues?(stats)
    end

    test "returns true when no valid frames" do
      stats = %{ParseStats.new() | total_frames: 10, valid_frames: 0}
      assert ParseStats.has_issues?(stats)
    end

    test "returns false when OK" do
      stats = %{ParseStats.new() | total_frames: 10, valid_frames: 10}
      refute ParseStats.has_issues?(stats)
    end
  end

  describe "format_summary/1" do
    test "includes all relevant info" do
      stats = %ParseStats{
        total_frames: 100,
        valid_frames: 80,
        dropped_no_player: 10,
        dropped_no_opponent: 5,
        dropped_no_controller: 5,
        dropped_delay_cutoff: 0,
        warnings: ["Test warning"]
      }

      summary = ParseStats.format_summary(stats)

      assert summary =~ "Total frames: 100"
      assert summary =~ "Valid frames: 80"
      assert summary =~ "missing player: 10"
      assert summary =~ "missing opponent: 5"
      assert summary =~ "Test warning"
    end
  end
end
