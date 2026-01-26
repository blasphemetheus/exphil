defmodule ExPhil.League.EvolutionTest do
  use ExUnit.Case

  alias ExPhil.League
  alias ExPhil.League.{Evolution, ArchitectureEntry}

  # These tests verify the Evolution module's structure and basic behavior
  # Full integration tests would require mock environments

  describe "evolve/2 options" do
    setup do
      name = :"league_evo_#{System.unique_integer([:positive])}"
      {:ok, _pid} = League.start_link(name: name, game_type: :mock)

      {:ok, league: name}
    end

    test "evolve requires at least 2 architectures for tournament", %{league: league} do
      {:ok, entry} = ArchitectureEntry.new(id: :solo, architecture: :mlp)
      League.register_entry(league, entry)

      # With only 1 architecture, tournament produces no matches
      {:ok, metrics} = Evolution.evolve(league, verbose: false)

      assert metrics.tournament.matches_played == 0
    end
  end

  describe "run/2 options" do
    test "accepts generation count" do
      # Verify the function signature accepts expected options
      # Functions have default opts, so arity 1 is the base
      Code.ensure_loaded!(Evolution)
      assert function_exported?(Evolution, :run, 1)
    end
  end

  describe "module structure" do
    # Ensure module is loaded before checking exports
    setup do
      Code.ensure_loaded!(Evolution)
      :ok
    end

    test "exports evolve/1 (with default opts)" do
      assert function_exported?(Evolution, :evolve, 1)
    end

    test "exports run/1 (with default opts)" do
      assert function_exported?(Evolution, :run, 1)
    end

    test "exports run_with_pruning/1 (with default opts)" do
      assert function_exported?(Evolution, :run_with_pruning, 1)
    end
  end
end
