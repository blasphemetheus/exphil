defmodule ExPhil.League.MatchSchedulerTest do
  use ExUnit.Case, async: true

  alias ExPhil.League.{MatchScheduler, ArchitectureEntry}

  describe "round_robin/2" do
    test "generates all pairs for 3 architectures" do
      schedule = MatchScheduler.round_robin([:a, :b, :c], shuffle: false)

      assert length(schedule) == 3
      assert {:a, :b} in schedule
      assert {:a, :c} in schedule
      assert {:b, :c} in schedule
    end

    test "generates correct number of pairs for 4 architectures" do
      schedule = MatchScheduler.round_robin([:a, :b, :c, :d], shuffle: false)

      # 4 choose 2 = 6 pairs
      assert length(schedule) == 6
    end

    test "generates correct number of pairs for 6 architectures" do
      schedule = MatchScheduler.round_robin([:a, :b, :c, :d, :e, :f], shuffle: false)

      # 6 choose 2 = 15 pairs
      assert length(schedule) == 15
    end

    test "respects matches_per_pair option" do
      schedule = MatchScheduler.round_robin([:a, :b, :c], matches_per_pair: 3, shuffle: false)

      # 3 pairs * 3 matches each = 9
      assert length(schedule) == 9
      assert Enum.count(schedule, &(&1 == {:a, :b})) == 3
    end

    test "shuffles by default" do
      schedule1 = MatchScheduler.round_robin([:a, :b, :c, :d, :e, :f])
      schedule2 = MatchScheduler.round_robin([:a, :b, :c, :d, :e, :f])

      # With 15 pairs, very unlikely to be in same order
      # (though theoretically possible, so we use refute instead of assert)
      refute schedule1 == schedule2
    end

    test "handles single architecture" do
      schedule = MatchScheduler.round_robin([:a])

      assert schedule == []
    end

    test "handles empty list" do
      schedule = MatchScheduler.round_robin([])

      assert schedule == []
    end
  end

  describe "skill_based/2" do
    setup do
      # Create architectures with different Elo ratings
      {:ok, low} = ArchitectureEntry.new(id: :low, architecture: :mlp, elo: 900)
      {:ok, mid1} = ArchitectureEntry.new(id: :mid1, architecture: :lstm, elo: 1000)
      {:ok, mid2} = ArchitectureEntry.new(id: :mid2, architecture: :gru, elo: 1050)
      {:ok, high} = ArchitectureEntry.new(id: :high, architecture: :mamba, elo: 1200)

      {:ok, archs: [low, mid1, mid2, high]}
    end

    test "generates requested number of matches", %{archs: archs} do
      schedule = MatchScheduler.skill_based(archs, num_matches: 10)

      assert length(schedule) == 10
    end

    test "all matchups are valid pairs", %{archs: archs} do
      schedule = MatchScheduler.skill_based(archs, num_matches: 20)
      ids = Enum.map(archs, & &1.id)

      for {p1, p2} <- schedule do
        assert p1 in ids
        assert p2 in ids
        assert p1 != p2
      end
    end

    test "prefers similar Elo matchups within range", %{archs: archs} do
      schedule = MatchScheduler.skill_based(archs, num_matches: 100, elo_range: 100)

      # Count matchups between :low (900) and :high (1200)
      extreme_matchups =
        Enum.count(schedule, fn {p1, p2} ->
          :low in [p1, p2] and :high in [p1, p2]
        end)

      # Most matches should be between closer ratings
      # :low and :high are 300 points apart, outside the range
      # So extreme matchups should be relatively rare (fallback only)
      assert extreme_matchups < 30
    end
  end

  describe "swiss_rounds/2" do
    setup do
      architectures =
        Enum.map(1..8, fn i ->
          {:ok, entry} =
            ArchitectureEntry.new(
              id: :"arch_#{i}",
              architecture: :mlp,
              elo: 1000 + i * 10
            )

          entry
        end)

      {:ok, archs: architectures}
    end

    test "generates requested number of rounds", %{archs: archs} do
      rounds = MatchScheduler.swiss_rounds(archs, num_rounds: 3)

      assert length(rounds) == 3
    end

    test "each round has correct number of matchups", %{archs: archs} do
      rounds = MatchScheduler.swiss_rounds(archs, num_rounds: 3)

      for round <- rounds do
        # 8 architectures = 4 matchups per round
        assert length(round) == 4
      end
    end

    test "all architectures appear in each round", %{archs: archs} do
      ids = Enum.map(archs, & &1.id) |> MapSet.new()
      [round1 | _] = MatchScheduler.swiss_rounds(archs, num_rounds: 1)

      participants =
        round1
        |> Enum.flat_map(fn {p1, p2} -> [p1, p2] end)
        |> MapSet.new()

      assert participants == ids
    end
  end

  describe "bracket/2" do
    test "generates first round for power of 2 participants" do
      [first_round] = MatchScheduler.bracket([:a, :b, :c, :d])

      assert length(first_round) == 2
    end

    test "pads to next power of 2 with byes" do
      [first_round] = MatchScheduler.bracket([:a, :b, :c])

      # 3 -> 4, so 2 matchups
      assert length(first_round) == 2

      # One matchup should include a bye
      bye_matchups =
        Enum.count(first_round, fn {p1, p2} ->
          p1 == :bye or p2 == :bye
        end)

      assert bye_matchups == 1
    end

    test "shuffles seeds when requested" do
      # With 8 participants, shuffling should change the bracket
      bracket1 = MatchScheduler.bracket(Enum.map(1..8, &:"arch_#{&1}"), shuffle_seeds: false)
      bracket2 = MatchScheduler.bracket(Enum.map(1..8, &:"arch_#{&1}"), shuffle_seeds: true)

      # Unshuffled should have predictable first matchup
      [{p1_1, p2_1} | _] = hd(bracket1)
      [{p1_2, p2_2} | _] = hd(bracket2)

      # At least one should be different when shuffled
      # (very high probability with 8 participants)
      assert {p1_1, p2_1} != {p1_2, p2_2} or
               bracket1 != bracket2
    end
  end

  describe "pfsp/3" do
    test "generates requested number of opponents" do
      candidates = [
        %{id: :a, win_rate: 0.6},
        %{id: :b, win_rate: 0.4},
        %{id: :c, win_rate: 0.5}
      ]

      opponents = MatchScheduler.pfsp(:target, candidates, num_matches: 10)

      assert length(opponents) == 10
    end

    test "excludes self from candidates" do
      candidates = [
        %{id: :target, win_rate: 0.5},
        %{id: :a, win_rate: 0.6},
        %{id: :b, win_rate: 0.4}
      ]

      opponents = MatchScheduler.pfsp(:target, candidates, num_matches: 20)

      refute :target in opponents
    end

    test "prioritizes low win-rate opponents" do
      candidates = [
        %{id: :easy, win_rate: 0.9},
        %{id: :hard, win_rate: 0.1}
      ]

      opponents =
        MatchScheduler.pfsp(:target, candidates,
          num_matches: 100,
          exploit_factor: 0.9
        )

      hard_count = Enum.count(opponents, &(&1 == :hard))
      easy_count = Enum.count(opponents, &(&1 == :easy))

      # Hard opponent (low win rate) should appear more often
      assert hard_count > easy_count
    end

    test "returns empty list with no valid candidates" do
      candidates = [%{id: :target, win_rate: 0.5}]

      opponents = MatchScheduler.pfsp(:target, candidates, num_matches: 10)

      assert opponents == []
    end
  end

  describe "diverse/2" do
    setup do
      architectures =
        Enum.map(1..4, fn i ->
          {:ok, entry} =
            ArchitectureEntry.new(
              id: :"arch_#{i}",
              architecture: :mlp,
              elo: 1000 + i * 50
            )

          entry
        end)

      {:ok, archs: architectures}
    end

    test "generates requested number of matches", %{archs: archs} do
      schedule = MatchScheduler.diverse(archs, num_matches: 20)

      assert length(schedule) == 20
    end

    test "all matchups are valid", %{archs: archs} do
      schedule = MatchScheduler.diverse(archs, num_matches: 20)
      ids = Enum.map(archs, & &1.id)

      for {p1, p2} <- schedule do
        assert p1 in ids
        assert p2 in ids
        assert p1 != p2
      end
    end
  end

  describe "matches_for_confidence/2" do
    test "returns reasonable estimate for small league" do
      matches = MatchScheduler.matches_for_confidence(4)

      # 4 architectures = 6 pairs, ~30 games each
      assert matches > 100
      assert matches < 300
    end

    test "scales with number of architectures" do
      small = MatchScheduler.matches_for_confidence(4)
      large = MatchScheduler.matches_for_confidence(8)

      # 8 has 28 pairs vs 6 pairs, so should need more matches
      assert large > small
    end
  end

  describe "estimated_duration/2" do
    test "calculates duration for schedule" do
      schedule = [{:a, :b}, {:a, :c}, {:b, :c}]

      duration =
        MatchScheduler.estimated_duration(schedule,
          avg_match_frames: 4500,
          parallel_games: 1
        )

      # 3 matches * 4500 frames / 60 fps = 225 seconds
      assert_in_delta duration, 225.0, 1.0
    end

    test "accounts for parallelism" do
      schedule = [{:a, :b}, {:a, :c}, {:b, :c}, {:c, :d}]

      serial = MatchScheduler.estimated_duration(schedule, parallel_games: 1)
      parallel = MatchScheduler.estimated_duration(schedule, parallel_games: 2)

      # Parallel should be roughly half (with some rounding)
      assert parallel < serial
      assert_in_delta parallel, serial / 2, 50.0
    end
  end
end
