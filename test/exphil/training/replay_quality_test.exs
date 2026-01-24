defmodule ExPhil.Training.ReplayQualityTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.ReplayQuality

  # Helper to build a test replay
  defp good_replay(overrides \\ %{}) do
    base = %{
      frames: 5000,
      players: [
        %{
          damage_dealt: 150,
          damage_taken: 120,
          stocks_lost: 3,
          sd_count: 0,
          input_frames: 1500,
          unique_actions: 55,
          is_cpu: false
        },
        %{
          damage_dealt: 120,
          damage_taken: 150,
          stocks_lost: 4,
          sd_count: 0,
          input_frames: 1400,
          unique_actions: 50,
          is_cpu: false
        }
      ]
    }

    deep_merge(base, overrides)
  end

  defp deep_merge(base, overrides) when is_map(base) and is_map(overrides) do
    Map.merge(base, overrides, fn
      _k, v1, v2 when is_map(v1) and is_map(v2) -> deep_merge(v1, v2)
      _k, v1, v2 when is_list(v1) and is_list(v2) ->
        Enum.zip_with(v1, v2, &deep_merge/2)
      _k, _v1, v2 -> v2
    end)
  end

  defp deep_merge(base, override), do: override || base

  describe "score/1" do
    test "returns high score for good replay" do
      replay = good_replay()
      score = ReplayQuality.score(replay)

      assert is_integer(score)
      assert score >= 80
    end

    test "rejects replay that is too short" do
      replay = good_replay(%{frames: 500})
      assert ReplayQuality.score(replay) == :rejected
    end

    test "rejects replay that is too long" do
      replay = good_replay(%{frames: 20000})
      assert ReplayQuality.score(replay) == :rejected
    end

    test "rejects replay with CPU player" do
      replay = good_replay()
      players = [
        %{damage_dealt: 100, damage_taken: 50, is_cpu: true},
        %{damage_dealt: 50, damage_taken: 100, is_cpu: false}
      ]
      replay = %{replay | players: players}

      assert ReplayQuality.score(replay) == :rejected
    end

    test "rejects replay with zero engagement" do
      replay = good_replay()
      players = [
        %{damage_dealt: 0, damage_taken: 0},
        %{damage_dealt: 0, damage_taken: 0}
      ]
      replay = %{replay | players: players}

      assert ReplayQuality.score(replay) == :rejected
    end

    test "gives lower score for short game" do
      short_replay = good_replay(%{frames: 2000})
      ideal_replay = good_replay(%{frames: 5000})

      short_score = ReplayQuality.score(short_replay)
      ideal_score = ReplayQuality.score(ideal_replay)

      assert short_score < ideal_score
    end

    test "gives lower score for low damage" do
      low_damage = good_replay()
      players = Enum.map(low_damage.players, fn p ->
        %{p | damage_dealt: 30, damage_taken: 30}
      end)
      low_damage = %{low_damage | players: players}

      high_damage = good_replay()

      assert ReplayQuality.score(low_damage) < ReplayQuality.score(high_damage)
    end

    test "gives lower score for low input activity" do
      low_activity = good_replay()
      players = Enum.map(low_activity.players, fn p ->
        %{p | input_frames: 200}  # 4% of 5000 frames
      end)
      low_activity = %{low_activity | players: players}

      high_activity = good_replay()

      assert ReplayQuality.score(low_activity) < ReplayQuality.score(high_activity)
    end

    test "gives lower score for high SD rate" do
      high_sd = good_replay()
      players = Enum.map(high_sd.players, fn p ->
        %{p | sd_count: 3, stocks_lost: 4}  # 75% SD rate
      end)
      high_sd = %{high_sd | players: players}

      low_sd = good_replay()

      assert ReplayQuality.score(high_sd) < ReplayQuality.score(low_sd)
    end

    test "gives lower score for low action diversity" do
      low_diversity = good_replay()
      players = Enum.map(low_diversity.players, fn p ->
        %{p | unique_actions: 10}
      end)
      low_diversity = %{low_diversity | players: players}

      high_diversity = good_replay()

      assert ReplayQuality.score(low_diversity) < ReplayQuality.score(high_diversity)
    end
  end

  describe "passes?/2" do
    test "returns true for good replay with default threshold" do
      replay = good_replay()
      assert ReplayQuality.passes?(replay)
    end

    test "returns false for rejected replay" do
      replay = good_replay(%{frames: 500})
      refute ReplayQuality.passes?(replay)
    end

    test "respects custom min_score" do
      replay = good_replay()

      # Should pass with low threshold
      assert ReplayQuality.passes?(replay, min_score: 50)

      # Should fail with very high threshold
      refute ReplayQuality.passes?(replay, min_score: 101)
    end
  end

  describe "analyze/1" do
    test "returns detailed breakdown for good replay" do
      replay = good_replay()
      analysis = ReplayQuality.analyze(replay)

      assert analysis.status == :passed
      assert analysis.score > 0
      assert analysis.quality in ["Excellent", "Good", "Fair", "Poor"]

      assert Map.has_key?(analysis.breakdown, :length)
      assert Map.has_key?(analysis.breakdown, :damage)
      assert Map.has_key?(analysis.breakdown, :activity)
      assert Map.has_key?(analysis.breakdown, :sd_rate)
      assert Map.has_key?(analysis.breakdown, :diversity)
    end

    test "returns rejection reason for bad replay" do
      replay = good_replay(%{frames: 500})
      analysis = ReplayQuality.analyze(replay)

      assert analysis.status == :rejected
      assert analysis.reason =~ "Too short"
      assert analysis.score == 0
    end

    test "breakdown scores sum to total" do
      replay = good_replay()
      analysis = ReplayQuality.analyze(replay)

      breakdown_sum = Enum.reduce(analysis.breakdown, 0, fn {_k, v}, acc ->
        acc + v.score
      end)

      assert analysis.score == breakdown_sum
    end
  end

  describe "batch_score/1" do
    test "returns statistics for multiple replays" do
      replays = [
        good_replay(),
        good_replay(%{frames: 500}),  # Will be rejected
        good_replay(%{frames: 2500})   # Lower score
      ]

      stats = ReplayQuality.batch_score(replays)

      assert stats.total == 3
      assert stats.passed == 2
      assert stats.rejected == 1
      assert stats.mean_score > 0
    end

    test "handles empty list" do
      stats = ReplayQuality.batch_score([])

      assert stats.total == 0
      assert stats.passed == 0
      assert stats.rejected == 0
      assert stats.mean_score == 0
    end
  end

  describe "print_analysis/1" do
    import ExUnit.CaptureIO

    test "prints analysis for good replay" do
      replay = good_replay()

      output = capture_io(:stderr, fn ->
        ReplayQuality.print_analysis(replay)
      end)

      assert output =~ "Replay Quality:"
      assert output =~ "/100"
    end

    test "prints rejection reason for bad replay" do
      replay = good_replay(%{frames: 500})

      output = capture_io(:stderr, fn ->
        ReplayQuality.print_analysis(replay)
      end)

      assert output =~ "REJECTED"
      assert output =~ "Too short"
    end
  end
end
