defmodule ExPhil.Training.SelfPlay.LeagueTrainer do
  @moduledoc """
  League training for self-play reinforcement learning.

  Supports two training modes:

  ## Simple Mix Mode (Default)

  Train a single agent against a mix of opponents:
  - 40% current self-play
  - 30% historical checkpoints
  - 20% CPU opponents
  - 10% random historical

  ```elixir
  {:ok, trainer} = LeagueTrainer.new(
    mode: :simple_mix,
    pretrained: "checkpoints/imitation.bin"
  )

  LeagueTrainer.train(trainer, total_timesteps: 1_000_000)
  ```

  ## League Mode (AlphaStar-style)

  Train multiple agent types with different objectives:

  - **Main Agents**: Train to beat the entire league
  - **Main Exploiters**: Find weaknesses in main agents
  - **League Exploiters**: Find weaknesses in anyone

  ```elixir
  {:ok, trainer} = LeagueTrainer.new(
    mode: :league,
    num_main_agents: 3,
    num_main_exploiters: 2,
    num_league_exploiters: 2
  )

  LeagueTrainer.train(trainer, total_timesteps: 10_000_000)
  ```

  ## Architecture

  ```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         League Trainer                                   │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                      Agent Pool                                  │    │
  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │    │
  │  │  │  Main 1  │ │  Main 2  │ │Exploiter1│ │Exploiter2│  ...       │    │
  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                              │                                           │
  │                              ▼                                           │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                   Matchmaking                                    │    │
  │  │   - Main vs League (all opponents)                              │    │
  │  │   - Exploiter vs Main (find weaknesses)                         │    │
  │  │   - League Exploiter vs Anyone                                  │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                              │                                           │
  │                              ▼                                           │
  │  ┌─────────────────┐ ┌─────────────────┐                                │
  │  │   Game 1        │ │   Game 2        │   Parallel Games               │
  │  │  (P1 vs P2)     │ │  (P1 vs P2)     │                                │
  │  └─────────────────┘ └─────────────────┘                                │
  │                              │                                           │
  │                              ▼                                           │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                   Experience Buffer                              │    │
  │  │   Collected from all games, batched for PPO updates             │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                              │                                           │
  │                              ▼                                           │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                    PPO Trainer                                   │    │
  │  │   Updates each agent's policy based on collected experience     │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
  ```
  """

  alias ExPhil.Training.PPO
  alias ExPhil.Training.SelfPlay.{OpponentPool, SelfPlayEnv}

  require Logger

  defstruct [
    :mode,                 # :simple_mix or :league
    :agents,               # Map of agent_id => agent_state
    :opponent_pool,        # OpponentPool for opponent selection
    :ppo_trainers,         # Map of agent_id => PPO trainer
    :config,               # Training configuration
    :metrics,              # Training metrics
    :iteration             # Current training iteration
  ]

  @type agent_type :: :main | :main_exploiter | :league_exploiter | :simple
  @type t :: %__MODULE__{}

  # Agent type configurations
  @agent_types %{
    main: %{
      description: "Main agent - trains against entire league",
      opponent_selection: :league_wide,
      pfsp_weight: 0.5  # Prioritized fictitious self-play weight
    },
    main_exploiter: %{
      description: "Main exploiter - finds weaknesses in main agents",
      opponent_selection: :main_agents_only,
      pfsp_weight: 0.8
    },
    league_exploiter: %{
      description: "League exploiter - finds weaknesses in anyone",
      opponent_selection: :league_wide,
      pfsp_weight: 0.9
    },
    simple: %{
      description: "Simple agent - practical mix training",
      opponent_selection: :simple_mix,
      pfsp_weight: 0.0
    }
  }

  @default_config %{
    # Training
    rollout_length: 2048,
    num_epochs: 10,
    batch_size: 64,
    learning_rate: 3.0e-4,

    # League
    num_main_agents: 1,
    num_main_exploiters: 0,
    num_league_exploiters: 0,

    # Checkpointing
    snapshot_interval: 10,      # Snapshot to pool every N iterations
    checkpoint_interval: 50,    # Save full checkpoint every N iterations
    checkpoint_dir: "checkpoints/league",

    # Parallel games
    num_parallel_games: 2,

    # Game environment
    game_type: :mock,           # :mock or :dolphin
    dolphin_config: nil,        # Required if game_type is :dolphin
                                # %{dolphin_path: "...", iso_path: "...", character: "fox", stage: "final_destination"}

    # Opponent mix (for simple_mix mode)
    opponent_mix: %{
      current: 0.4,
      historical: 0.3,
      cpu: 0.2,
      random: 0.1
    }
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Create a new league trainer.

  ## Options
    - `:mode` - Training mode: `:simple_mix` or `:league` (default: :simple_mix)
    - `:pretrained` - Path to pretrained policy for initialization
    - `:embed_size` - Embedding size (default: auto from pretrained)
    - `:num_main_agents` - Number of main agents in league mode
    - `:num_main_exploiters` - Number of main exploiters
    - `:num_league_exploiters` - Number of league exploiters
    - `:num_parallel_games` - Parallel games to run (default: 2)
    - `:opponent_mix` - Opponent sampling weights for simple_mix mode
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts \\ []) do
    mode = Keyword.get(opts, :mode, :simple_mix)
    config = Map.merge(@default_config, Map.new(opts))

    # Validate dolphin_config when using Dolphin
    if config.game_type == :dolphin do
      cond do
        is_nil(config.dolphin_config) ->
          {:error, "dolphin_config is required when game_type is :dolphin"}

        is_nil(config.dolphin_config[:dolphin_path]) ->
          {:error, "dolphin_config.dolphin_path is required"}

        is_nil(config.dolphin_config[:iso_path]) ->
          {:error, "dolphin_config.iso_path is required"}

        true ->
          do_new(mode, config, opts)
      end
    else
      do_new(mode, config, opts)
    end
  end

  defp do_new(mode, config, opts) do

    # Initialize opponent pool
    {:ok, opponent_pool} = OpponentPool.new(
      config: config.opponent_mix,
      cpu_levels: [5, 6, 7, 8, 9]
    )

    # Create agents based on mode
    agents = create_agents(mode, config, opts)

    # Create PPO trainers for each agent
    ppo_trainers = create_ppo_trainers(agents, config, opts)

    trainer = %__MODULE__{
      mode: mode,
      agents: agents,
      opponent_pool: opponent_pool,
      ppo_trainers: ppo_trainers,
      config: config,
      metrics: init_metrics(),
      iteration: 0
    }

    {:ok, trainer}
  end

  @doc """
  Run training loop.

  ## Options
    - `:total_timesteps` - Total environment steps (default: 1_000_000)
    - `:callback` - Function called each iteration with metrics
  """
  @spec train(t(), keyword()) :: {:ok, t()}
  def train(%__MODULE__{} = trainer, opts \\ []) do
    total_timesteps = Keyword.get(opts, :total_timesteps, 1_000_000)
    callback = Keyword.get(opts, :callback, fn _, _ -> :ok end)

    steps_per_iteration = trainer.config.rollout_length * trainer.config.num_parallel_games
    num_iterations = div(total_timesteps, steps_per_iteration)

    Logger.info("""
    Starting #{trainer.mode} training:
      Total timesteps: #{total_timesteps}
      Steps per iteration: #{steps_per_iteration}
      Iterations: #{num_iterations}
      Agents: #{map_size(trainer.agents)}
      Parallel games: #{trainer.config.num_parallel_games}
    """)

    final_trainer = Enum.reduce(1..num_iterations, trainer, fn iteration, acc ->
      # Run one training iteration
      {:ok, new_trainer, metrics} = train_iteration(acc)

      # Callback
      callback.(new_trainer, metrics)

      # Periodic snapshots and checkpoints
      new_trainer = maybe_snapshot(new_trainer, iteration)
      new_trainer = maybe_checkpoint(new_trainer, iteration)

      %{new_trainer | iteration: iteration}
    end)

    {:ok, final_trainer}
  end

  @doc """
  Get the best agent from the trainer.
  """
  @spec get_best_agent(t()) :: {atom(), map()}
  def get_best_agent(%__MODULE__{agents: agents}) do
    # Return agent with highest Elo (or first main agent)
    agents
    |> Enum.filter(fn {_id, agent} -> agent.type in [:main, :simple] end)
    |> Enum.max_by(fn {_id, agent} -> agent.elo end, fn -> hd(Map.to_list(agents)) end)
  end

  @doc """
  Export the best agent's policy.
  """
  @spec export_best_policy(t(), String.t()) :: :ok
  def export_best_policy(%__MODULE__{} = trainer, path) do
    {agent_id, _agent} = get_best_agent(trainer)
    ppo_trainer = Map.get(trainer.ppo_trainers, agent_id)

    PPO.export_policy(ppo_trainer, path)
  end

  # ============================================================================
  # Training Loop
  # ============================================================================

  defp train_iteration(%__MODULE__{} = trainer) do
    # For each agent, collect experience and update
    {new_trainers, all_metrics} = trainer.agents
    |> Enum.map(fn {agent_id, agent} ->
      # Select opponent for this agent
      opponent = select_opponent(trainer, agent)

      # Collect rollout
      ppo_trainer = Map.get(trainer.ppo_trainers, agent_id)
      rollout = collect_rollout(ppo_trainer, agent, opponent, trainer.config)

      # Update PPO (returns {trainer, metrics} tuple)
      {updated_ppo, metrics} = PPO.update(ppo_trainer, rollout)

      # Update Elo based on results
      updated_agent = update_agent_elo(agent, rollout)

      {agent_id, updated_ppo, updated_agent, metrics}
    end)
    |> Enum.reduce({%{}, %{}}, fn {id, ppo, _agent, metrics}, {ppo_acc, metrics_acc} ->
      {
        Map.put(ppo_acc, id, ppo),
        Map.put(metrics_acc, id, metrics)
      }
    end)

    # Update trainer state
    new_agents = Enum.reduce(trainer.agents, %{}, fn {id, _}, acc ->
      Map.put(acc, id, Enum.find_value(all_metrics, fn {aid, _} -> if aid == id, do: Map.get(trainer.agents, id) end) || Map.get(trainer.agents, id))
    end)

    new_trainer = %{trainer |
      ppo_trainers: new_trainers,
      agents: new_agents,
      metrics: aggregate_metrics(all_metrics)
    }

    {:ok, new_trainer, new_trainer.metrics}
  end

  defp select_opponent(trainer, agent) do
    case agent.type do
      :simple ->
        # Simple mix - sample from opponent pool
        OpponentPool.sample(trainer.opponent_pool)

      :main ->
        # Main agent - sample from entire league with PFSP
        sample_from_league(trainer, agent, :league_wide)

      :main_exploiter ->
        # Main exploiter - only target main agents
        sample_from_league(trainer, agent, :main_agents_only)

      :league_exploiter ->
        # League exploiter - sample from entire league with high PFSP
        sample_from_league(trainer, agent, :league_wide)
    end
  end

  defp sample_from_league(trainer, agent, selection_mode) do
    type_config = Map.get(@agent_types, agent.type)
    pfsp_weight = type_config.pfsp_weight

    # Get eligible opponents
    eligible = case selection_mode do
      :main_agents_only ->
        trainer.agents
        |> Enum.filter(fn {_id, a} -> a.type == :main end)

      :league_wide ->
        Map.to_list(trainer.agents)
    end

    if length(eligible) == 0 do
      # Fallback to opponent pool
      OpponentPool.sample(trainer.opponent_pool)
    else
      # Prioritized sampling based on win rate
      if :rand.uniform() < pfsp_weight do
        # PFSP: prioritize opponents we struggle against
        prioritized_sample(eligible, agent, trainer)
      else
        # Uniform sampling
        {opponent_id, _opponent} = Enum.random(eligible)
        ppo = Map.get(trainer.ppo_trainers, opponent_id)
        {:league, %{type: :league, params: ppo.params, id: opponent_id}}
      end
    end
  end

  defp prioritized_sample(eligible, _agent, trainer) do
    # Weight by inverse win rate (harder opponents get higher weight)
    weights = Enum.map(eligible, fn {id, _opponent} ->
      win_rate = OpponentPool.get_win_rate(trainer.opponent_pool, to_string(id))
      # Inverse: lower win rate = higher weight
      {id, 1.0 - win_rate + 0.1}
    end)

    total = Enum.sum(Enum.map(weights, &elem(&1, 1)))
    r = :rand.uniform() * total

    {selected_id, _} = Enum.reduce_while(weights, {nil, 0.0}, fn {id, w}, {_, cumsum} ->
      new_cumsum = cumsum + w
      if r <= new_cumsum do
        {:halt, {id, new_cumsum}}
      else
        {:cont, {id, new_cumsum}}
      end
    end)

    ppo = Map.get(trainer.ppo_trainers, selected_id)
    {:league, %{type: :league, params: ppo.params, id: selected_id}}
  end

  defp collect_rollout(ppo_trainer, _agent, opponent, config) do
    # Build environment options
    env_opts = [
      p1_policy: {ppo_trainer.model, ppo_trainer.params},
      p2_policy: opponent_to_policy(opponent),
      game_type: config.game_type
    ]

    # Add dolphin_config if using Dolphin
    env_opts = if config.game_type == :dolphin do
      Keyword.put(env_opts, :dolphin_config, config.dolphin_config)
    else
      env_opts
    end

    # Create environment with opponent
    {:ok, env} = SelfPlayEnv.new(env_opts)

    # Collect steps
    {:ok, env, experiences} = SelfPlayEnv.collect_steps(env, config.rollout_length)

    # Shutdown environment to release Dolphin resources
    SelfPlayEnv.shutdown(env)

    # Convert to rollout format for PPO
    experiences_to_rollout(experiences)
  end

  defp opponent_to_policy({:cpu, %{level: _level}}) do
    :cpu
  end

  defp opponent_to_policy({:current, %{params: params}}) do
    # Need to get model - for now assume same architecture
    {nil, params}  # Will be handled by SelfPlayEnv
  end

  defp opponent_to_policy({:historical, %{params: params}}) do
    {nil, params}
  end

  defp opponent_to_policy({:league, %{params: params}}) do
    {nil, params}
  end

  defp experiences_to_rollout(experiences) do
    %{
      states: experiences |> Enum.map(& &1.state) |> Nx.stack(),
      actions: stack_actions(Enum.map(experiences, & &1.action)),
      rewards: experiences |> Enum.map(& &1.reward) |> Nx.tensor(type: :f32),
      dones: experiences |> Enum.map(& if(&1.done, do: 1.0, else: 0.0)) |> Nx.tensor(type: :f32),
      values: experiences |> Enum.map(& Nx.to_number(&1.value)) |> Kernel.++([0.0]) |> Nx.tensor(type: :f32),
      log_probs: experiences |> Enum.map(& Nx.to_number(&1.log_prob)) |> Nx.tensor(type: :f32)
    }
  end

  defp stack_actions(actions) do
    keys = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]
    Map.new(keys, fn key ->
      {key, actions |> Enum.map(&Map.get(&1, key)) |> Nx.stack()}
    end)
  end

  defp update_agent_elo(agent, rollout) do
    # Simple Elo update based on episode rewards
    total_reward = Nx.sum(rollout.rewards) |> Nx.to_number()
    elo_change = total_reward * 10  # Scale factor

    %{agent | elo: agent.elo + elo_change}
  end

  # ============================================================================
  # Agent Creation
  # ============================================================================

  defp create_agents(:simple_mix, _config, _opts) do
    %{
      main: %{
        id: :main,
        type: :simple,
        elo: 1000,
        created_at: System.system_time(:second)
      }
    }
  end

  defp create_agents(:league, config, _opts) do
    agents = %{}

    # Main agents
    agents = Enum.reduce(1..config.num_main_agents, agents, fn i, acc ->
      id = :"main_#{i}"
      Map.put(acc, id, %{
        id: id,
        type: :main,
        elo: 1000,
        created_at: System.system_time(:second)
      })
    end)

    # Main exploiters
    agents = Enum.reduce(1..config.num_main_exploiters, agents, fn i, acc ->
      id = :"main_exploiter_#{i}"
      Map.put(acc, id, %{
        id: id,
        type: :main_exploiter,
        elo: 1000,
        created_at: System.system_time(:second)
      })
    end)

    # League exploiters
    Enum.reduce(1..config.num_league_exploiters, agents, fn i, acc ->
      id = :"league_exploiter_#{i}"
      Map.put(acc, id, %{
        id: id,
        type: :league_exploiter,
        elo: 1000,
        created_at: System.system_time(:second)
      })
    end)
  end

  defp create_ppo_trainers(agents, config, opts) do
    Map.new(agents, fn {id, _agent} ->
      ppo_opts = [
        embed_size: Keyword.get(opts, :embed_size, 1991),
        pretrained_path: Keyword.get(opts, :pretrained),
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        num_epochs: config.num_epochs
      ]

      {id, PPO.new(ppo_opts)}
    end)
  end

  # ============================================================================
  # Checkpointing
  # ============================================================================

  defp maybe_snapshot(trainer, iteration) do
    if rem(iteration, trainer.config.snapshot_interval) == 0 do
      # Snapshot best agent to opponent pool
      {agent_id, _agent} = get_best_agent(trainer)
      ppo = Map.get(trainer.ppo_trainers, agent_id)

      version = "v#{iteration}"
      new_pool = OpponentPool.snapshot(
        OpponentPool.set_current(trainer.opponent_pool, ppo.params),
        version
      )

      Logger.info("Snapshotted #{agent_id} as #{version}")
      %{trainer | opponent_pool: new_pool}
    else
      trainer
    end
  end

  defp maybe_checkpoint(trainer, iteration) do
    if rem(iteration, trainer.config.checkpoint_interval) == 0 do
      dir = trainer.config.checkpoint_dir
      File.mkdir_p!(dir)

      # Save each agent
      Enum.each(trainer.ppo_trainers, fn {id, ppo} ->
        path = Path.join(dir, "#{id}_iter#{iteration}.axon")
        PPO.save_checkpoint(ppo, path)
      end)

      Logger.info("Saved checkpoint at iteration #{iteration}")
    end

    trainer
  end

  # ============================================================================
  # Metrics
  # ============================================================================

  defp init_metrics do
    %{
      total_timesteps: 0,
      episodes: 0,
      avg_reward: 0.0,
      avg_episode_length: 0.0,
      policy_loss: 0.0,
      value_loss: 0.0,
      entropy: 0.0
    }
  end

  defp aggregate_metrics(agent_metrics) do
    if map_size(agent_metrics) == 0 do
      init_metrics()
    else
      # Average across agents
      values = Map.values(agent_metrics)
      n = length(values)

      %{
        total_timesteps: Enum.sum(Enum.map(values, & &1[:timesteps] || 0)),
        episodes: Enum.sum(Enum.map(values, & &1[:episodes] || 0)),
        avg_reward: Enum.sum(Enum.map(values, & &1[:avg_reward] || 0)) / n,
        avg_episode_length: Enum.sum(Enum.map(values, & &1[:avg_episode_length] || 0)) / n,
        policy_loss: Enum.sum(Enum.map(values, & &1[:policy_loss] || 0)) / n,
        value_loss: Enum.sum(Enum.map(values, & &1[:value_loss] || 0)) / n,
        entropy: Enum.sum(Enum.map(values, & &1[:entropy] || 0)) / n
      }
    end
  end
end
