defmodule ExPhil.Training.SelfPlay.LeagueTrainerTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.SelfPlay.LeagueTrainer

  # Tests take 20-150 seconds each due to model initialization
  @moduletag :slow
  @moduletag :self_play

  # Error may be a string or a ConfigError struct
  defp error_message(%{message: msg}), do: msg
  defp error_message(msg) when is_binary(msg), do: msg

  describe "new/1 validation" do
    test "creates trainer with default mock game type" do
      assert {:ok, trainer} = LeagueTrainer.new()

      assert trainer.config.game_type == :mock
      assert trainer.mode == :simple_mix
    end

    test "creates trainer with explicit mock game type" do
      assert {:ok, trainer} = LeagueTrainer.new(game_type: :mock)

      assert trainer.config.game_type == :mock
    end

    test "requires dolphin_config when game_type is :dolphin" do
      result = LeagueTrainer.new(game_type: :dolphin)

      assert {:error, error} = result
      assert error_message(error) =~ "dolphin_config"
    end

    test "requires dolphin_path in dolphin_config" do
      result =
        LeagueTrainer.new(
          game_type: :dolphin,
          dolphin_config: %{iso_path: "/path/to/iso"}
        )

      assert {:error, error} = result
      assert error_message(error) =~ "dolphin_path"
    end

    test "requires iso_path in dolphin_config" do
      result =
        LeagueTrainer.new(
          game_type: :dolphin,
          dolphin_config: %{dolphin_path: "/path/to/dolphin"}
        )

      assert {:error, error} = result
      assert error_message(error) =~ "iso_path"
    end

    test "accepts valid dolphin_config" do
      assert {:ok, trainer} =
               LeagueTrainer.new(
                 game_type: :dolphin,
                 dolphin_config: %{
                   dolphin_path: "/path/to/dolphin",
                   iso_path: "/path/to/melee.iso",
                   character: "mewtwo",
                   stage: "final_destination"
                 }
               )

      assert trainer.config.game_type == :dolphin
      assert trainer.config.dolphin_config.dolphin_path == "/path/to/dolphin"
    end
  end

  describe "new/1 modes" do
    test "creates simple_mix trainer" do
      assert {:ok, trainer} = LeagueTrainer.new(mode: :simple_mix)

      assert trainer.mode == :simple_mix
      assert map_size(trainer.agents) == 1
      assert Map.has_key?(trainer.agents, :main)
    end

    # League mode creates multiple agents, slow on CPU
    @tag timeout: 180_000
    test "creates league trainer with multiple agents" do
      assert {:ok, trainer} =
               LeagueTrainer.new(
                 mode: :league,
                 num_main_agents: 2,
                 num_main_exploiters: 1,
                 num_league_exploiters: 1
               )

      assert trainer.mode == :league
      # 2 main + 1 main_exploiter + 1 league_exploiter = 4
      assert map_size(trainer.agents) == 4
    end
  end

  describe "new/1 config" do
    test "accepts custom rollout_length" do
      assert {:ok, trainer} = LeagueTrainer.new(rollout_length: 4096)

      assert trainer.config.rollout_length == 4096
    end

    test "accepts custom learning_rate" do
      assert {:ok, trainer} = LeagueTrainer.new(learning_rate: 1.0e-4)

      assert trainer.config.learning_rate == 1.0e-4
    end

    test "accepts custom num_parallel_games" do
      assert {:ok, trainer} = LeagueTrainer.new(num_parallel_games: 4)

      assert trainer.config.num_parallel_games == 4
    end

    test "accepts custom opponent_mix" do
      custom_mix = %{current: 0.5, historical: 0.2, cpu: 0.2, random: 0.1}

      assert {:ok, trainer} = LeagueTrainer.new(opponent_mix: custom_mix)

      assert trainer.config.opponent_mix == custom_mix
    end
  end

  describe "get_best_agent/1" do
    test "returns main agent for simple_mix mode" do
      {:ok, trainer} = LeagueTrainer.new(mode: :simple_mix)

      {agent_id, agent} = LeagueTrainer.get_best_agent(trainer)

      assert agent_id == :main
      assert agent.type == :simple
    end

    # League mode creates multiple agents, slow on CPU
    @tag timeout: 180_000
    test "returns highest elo agent for league mode" do
      {:ok, trainer} =
        LeagueTrainer.new(
          mode: :league,
          num_main_agents: 2
        )

      # Both agents start with same elo, so first one is returned
      {agent_id, agent} = LeagueTrainer.get_best_agent(trainer)

      assert agent.type == :main
      assert is_atom(agent_id)
    end
  end

  describe "agent creation" do
    test "simple_mix creates single simple agent" do
      {:ok, trainer} = LeagueTrainer.new(mode: :simple_mix)

      assert Map.has_key?(trainer.agents, :main)
      assert trainer.agents.main.type == :simple
      assert trainer.agents.main.elo == 1000
    end

    # League mode creates multiple agents, slow on CPU
    @tag timeout: 180_000
    test "league mode creates correct agent types" do
      {:ok, trainer} =
        LeagueTrainer.new(
          mode: :league,
          num_main_agents: 2,
          num_main_exploiters: 1,
          num_league_exploiters: 1
        )

      # Check main agents
      assert Map.has_key?(trainer.agents, :main_1)
      assert Map.has_key?(trainer.agents, :main_2)
      assert trainer.agents.main_1.type == :main
      assert trainer.agents.main_2.type == :main

      # Check exploiters
      assert Map.has_key?(trainer.agents, :main_exploiter_1)
      assert trainer.agents.main_exploiter_1.type == :main_exploiter

      assert Map.has_key?(trainer.agents, :league_exploiter_1)
      assert trainer.agents.league_exploiter_1.type == :league_exploiter
    end
  end

  describe "ppo_trainers" do
    # League mode creates multiple agents, slow on CPU
    @tag timeout: 180_000
    test "creates ppo trainer for each agent" do
      {:ok, trainer} =
        LeagueTrainer.new(
          mode: :league,
          num_main_agents: 2
        )

      # Should have PPO trainer for each agent
      assert map_size(trainer.ppo_trainers) == map_size(trainer.agents)

      for {agent_id, _agent} <- trainer.agents do
        assert Map.has_key?(trainer.ppo_trainers, agent_id)
      end
    end
  end

  describe "metrics" do
    test "initializes with zero metrics" do
      {:ok, trainer} = LeagueTrainer.new()

      assert trainer.metrics.total_timesteps == 0
      assert trainer.metrics.episodes == 0
      assert trainer.metrics.avg_reward == 0.0
    end

    test "starts at iteration 0" do
      {:ok, trainer} = LeagueTrainer.new()

      assert trainer.iteration == 0
    end
  end
end
