defmodule ExPhil.LeagueTest do
  use ExUnit.Case

  alias ExPhil.League
  alias ExPhil.League.ArchitectureEntry

  # Start a fresh league for each test
  setup do
    name = :"league_#{System.unique_integer([:positive])}"
    {:ok, _pid} = League.start_link(name: name, game_type: :mock)

    {:ok, league: name}
  end

  describe "start_link/1" do
    test "starts with default configuration" do
      name = :"league_test_#{System.unique_integer([:positive])}"
      {:ok, pid} = League.start_link(name: name)

      assert Process.alive?(pid)
    end

    test "starts with custom configuration" do
      name = :"league_custom_#{System.unique_integer([:positive])}"
      {:ok, _pid} = League.start_link(
        name: name,
        game_type: :mock,
        stocks: 3
      )

      # Just verify it started
      assert GenServer.whereis(name) != nil
    end
  end

  describe "register_architecture/5" do
    test "registers a new architecture", %{league: league} do
      result = League.register_architecture(
        league,
        :mamba_mewtwo,
        nil,  # model
        %{},  # params
        %{architecture: :mamba}
      )

      assert result == :ok
    end

    test "rejects duplicate registration", %{league: league} do
      League.register_architecture(league, :mamba_mewtwo, nil, %{}, %{architecture: :mamba})

      result = League.register_architecture(
        league,
        :mamba_mewtwo,
        nil,
        %{},
        %{architecture: :mamba}
      )

      assert result == {:error, :already_registered}
    end

    test "allows multiple different architectures", %{league: league} do
      assert :ok == League.register_architecture(league, :mamba_mewtwo, nil, %{}, %{architecture: :mamba})
      assert :ok == League.register_architecture(league, :lstm_mewtwo, nil, %{}, %{architecture: :lstm})
      assert :ok == League.register_architecture(league, :mlp_mewtwo, nil, %{}, %{architecture: :mlp})
    end
  end

  describe "register_entry/2" do
    test "registers an ArchitectureEntry struct", %{league: league} do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      result = League.register_entry(league, entry)

      assert result == :ok
    end

    test "rejects duplicate entry", %{league: league} do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)

      League.register_entry(league, entry)
      result = League.register_entry(league, entry)

      assert result == {:error, :already_registered}
    end
  end

  describe "unregister_architecture/2" do
    test "removes registered architecture", %{league: league} do
      League.register_architecture(league, :mamba_mewtwo, nil, %{}, %{architecture: :mamba})

      result = League.unregister_architecture(league, :mamba_mewtwo)

      assert result == :ok
      assert {:error, :not_found} == League.get_architecture(league, :mamba_mewtwo)
    end

    test "returns error for unknown architecture", %{league: league} do
      result = League.unregister_architecture(league, :unknown)

      assert result == {:error, :not_found}
    end
  end

  describe "get_architecture/2" do
    test "returns registered architecture", %{league: league} do
      League.register_architecture(league, :mamba_mewtwo, nil, %{}, %{architecture: :mamba})

      {:ok, entry} = League.get_architecture(league, :mamba_mewtwo)

      assert entry.id == :mamba_mewtwo
      assert entry.architecture == :mamba
    end

    test "returns error for unknown architecture", %{league: league} do
      result = League.get_architecture(league, :unknown)

      assert result == {:error, :not_found}
    end
  end

  describe "list_architectures/1" do
    test "returns empty list initially", %{league: league} do
      result = League.list_architectures(league)

      assert result == []
    end

    test "returns all registered architectures", %{league: league} do
      League.register_architecture(league, :mamba_mewtwo, nil, %{}, %{architecture: :mamba})
      League.register_architecture(league, :lstm_mewtwo, nil, %{}, %{architecture: :lstm})

      result = League.list_architectures(league)

      assert length(result) == 2
      ids = Enum.map(result, & &1.id)
      assert :mamba_mewtwo in ids
      assert :lstm_mewtwo in ids
    end
  end

  describe "get_leaderboard/2" do
    test "returns empty leaderboard initially", %{league: league} do
      result = League.get_leaderboard(league)

      assert result == []
    end

    test "returns architectures sorted by Elo", %{league: league} do
      # Register with different initial Elos
      {:ok, low} = ArchitectureEntry.new(id: :low, architecture: :mlp, elo: 900)
      {:ok, high} = ArchitectureEntry.new(id: :high, architecture: :mamba, elo: 1200)
      {:ok, mid} = ArchitectureEntry.new(id: :mid, architecture: :lstm, elo: 1000)

      League.register_entry(league, low)
      League.register_entry(league, high)
      League.register_entry(league, mid)

      leaderboard = League.get_leaderboard(league)

      assert length(leaderboard) == 3
      assert hd(leaderboard).id == :high
      assert List.last(leaderboard).id == :low
    end

    test "respects limit parameter", %{league: league} do
      for i <- 1..10 do
        {:ok, entry} = ArchitectureEntry.new(
          id: :"arch_#{i}",
          architecture: :mlp,
          elo: 1000 + i * 10
        )
        League.register_entry(league, entry)
      end

      leaderboard = League.get_leaderboard(league, 3)

      assert length(leaderboard) == 3
    end

    test "includes statistics", %{league: league} do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)
      League.register_entry(league, entry)

      [first | _] = League.get_leaderboard(league)

      assert Map.has_key?(first, :elo)
      assert Map.has_key?(first, :wins)
      assert Map.has_key?(first, :losses)
      assert Map.has_key?(first, :win_rate)
      assert Map.has_key?(first, :games_played)
    end
  end

  describe "update_params/3" do
    test "updates architecture params", %{league: league} do
      {:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)
      League.register_entry(league, entry)

      new_params = %{layer1: %{weight: :updated}}
      result = League.update_params(league, :mamba_mewtwo, new_params)

      assert result == :ok

      {:ok, updated} = League.get_architecture(league, :mamba_mewtwo)
      assert updated.params == new_params
      assert updated.generation == 1
    end

    test "returns error for unknown architecture", %{league: league} do
      result = League.update_params(league, :unknown, %{})

      assert result == {:error, :not_found}
    end
  end

  describe "advance_generation/1" do
    test "increments generation counter", %{league: league} do
      assert League.get_generation(league) == 0

      {:ok, 1} = League.advance_generation(league)
      assert League.get_generation(league) == 1

      {:ok, 2} = League.advance_generation(league)
      assert League.get_generation(league) == 2
    end
  end

  describe "get_stats/1" do
    test "returns initial stats", %{league: league} do
      stats = League.get_stats(league)

      assert stats.num_architectures == 0
      assert stats.total_matches == 0
      assert stats.generation == 0
    end

    test "tracks architecture count", %{league: league} do
      League.register_architecture(league, :arch1, nil, %{}, %{architecture: :mlp})
      League.register_architecture(league, :arch2, nil, %{}, %{architecture: :lstm})

      stats = League.get_stats(league)

      assert stats.num_architectures == 2
    end
  end

  describe "experience pool" do
    test "get_experiences returns empty initially", %{league: league} do
      result = League.get_experiences(league)

      assert result == []
    end

    test "clear_experiences clears the pool", %{league: league} do
      # Can't easily add experiences without running matches,
      # but we can verify clear works
      result = League.clear_experiences(league)

      assert result == :ok
      assert League.get_experiences(league) == []
    end
  end

  describe "get_match_history/3" do
    test "returns empty history initially", %{league: league} do
      League.register_architecture(league, :arch1, nil, %{}, %{architecture: :mlp})

      history = League.get_match_history(league, :arch1)

      assert history == []
    end
  end
end
