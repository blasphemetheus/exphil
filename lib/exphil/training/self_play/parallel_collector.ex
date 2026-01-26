defmodule ExPhil.Training.SelfPlay.ParallelCollector do
  @moduledoc """
  Collects experience from multiple parallel self-play games.

  Uses Elixir's concurrency to run N games simultaneously, collecting
  experience in parallel and aggregating rollouts for PPO training.

  ## Architecture

      ┌─────────────────────────────────────────────────────────────────┐
      │                     ParallelCollector                           │
      │                                                                 │
      │  ┌───────────┐  ┌───────────┐       ┌───────────┐             │
      │  │  Game 1   │  │  Game 2   │  ...  │  Game N   │             │
      │  │ SelfPlay  │  │ SelfPlay  │       │ SelfPlay  │             │
      │  │   Env     │  │   Env     │       │   Env     │             │
      │  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘             │
      │        │              │                   │                    │
      │        └──────────────┴───────────────────┘                    │
      │                       │                                        │
      │                       ▼                                        │
      │        ┌─────────────────────────────────┐                    │
      │        │      Aggregated Rollouts        │                    │
      │        │  (ready for PPO.update batch)   │                    │
      │        └─────────────────────────────────┘                    │
      │                                                                │
      └─────────────────────────────────────────────────────────────────┘

  ## Usage

      # Create collector with 2 parallel games
      {:ok, collector} = ParallelCollector.new(
        num_envs: 2,
        policy: trained_policy,
        opponent_pool: pool,
        rollout_length: 128
      )

      # Collect experience from all games
      {:ok, collector, rollouts} = ParallelCollector.collect_rollouts(collector)
      # rollouts = %{states: tensor, actions: tensor, rewards: tensor, ...}

      # Update policy and sync to all environments
      {:ok, collector} = ParallelCollector.update_policy(collector, new_params)

  """

  alias ExPhil.Training.SelfPlay.{SelfPlayEnv, OpponentPool}
  require Logger

  defstruct [
    # List of SelfPlayEnv instances
    :envs,
    # Number of parallel environments
    :num_envs,
    # Current policy {model, params}
    :policy,
    # OpponentPool for opponent sampling
    :opponent_pool,
    # Steps per rollout collection
    :rollout_length,
    # Additional configuration
    :config,
    # Collection statistics
    :stats
  ]

  @type t :: %__MODULE__{}

  @default_config %{
    game_type: :mock,
    p1_port: 1,
    p2_port: 2,
    # Timeout per game step (ms)
    timeout: 60_000,
    # Use async collection (true) or sequential (false)
    async: true
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Create a new parallel collector.

  ## Options
    - `:num_envs` - Number of parallel environments (default: 2)
    - `:policy` - Current policy as `{model, params}` (required)
    - `:opponent_pool` - OpponentPool for sampling opponents (required)
    - `:rollout_length` - Steps per rollout collection (default: 128)
    - `:game_type` - `:mock` or `:dolphin` (default: :mock)
    - `:dolphin_config` - Config for Dolphin mode (map with :path, :iso, etc.)
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) do
    num_envs = Keyword.get(opts, :num_envs, 2)
    policy = Keyword.fetch!(opts, :policy)
    opponent_pool = Keyword.fetch!(opts, :opponent_pool)
    rollout_length = Keyword.get(opts, :rollout_length, 128)
    config = Map.merge(@default_config, Map.new(Keyword.get(opts, :config, [])))
    dolphin_config = Keyword.get(opts, :dolphin_config, %{})

    # Create N environments with different opponents
    envs = create_environments(num_envs, policy, opponent_pool, config, dolphin_config)

    case envs do
      {:error, reason} ->
        {:error, reason}

      envs when is_list(envs) ->
        collector = %__MODULE__{
          envs: envs,
          num_envs: num_envs,
          policy: policy,
          opponent_pool: opponent_pool,
          rollout_length: rollout_length,
          config: config,
          stats: init_stats()
        }

        Logger.info("[ParallelCollector] Created #{num_envs} parallel environments")
        {:ok, collector}
    end
  end

  @doc """
  Collect rollouts from all parallel environments.

  Returns aggregated experience ready for PPO training.
  """
  @spec collect_rollouts(t()) :: {:ok, t(), map()} | {:error, term()}
  def collect_rollouts(%__MODULE__{} = collector) do
    start_time = System.monotonic_time(:millisecond)

    # Collect from all environments
    {results, updated_envs} =
      if collector.config.async do
        collect_async(collector)
      else
        collect_sequential(collector)
      end

    # Aggregate rollouts from all environments
    rollouts = aggregate_rollouts(results)

    # Update statistics
    end_time = System.monotonic_time(:millisecond)
    elapsed = end_time - start_time

    stats = update_stats(collector.stats, results, elapsed)

    updated_collector = %{collector | envs: updated_envs, stats: stats}

    Logger.debug(
      "[ParallelCollector] Collected #{collector.rollout_length} steps from #{collector.num_envs} envs in #{elapsed}ms"
    )

    {:ok, updated_collector, rollouts}
  end

  @doc """
  Update policy in all environments.

  Called after PPO update to sync new parameters.
  """
  @spec update_policy(t(), map()) :: {:ok, t()}
  def update_policy(%__MODULE__{} = collector, new_params) do
    {model, _old_params} = collector.policy
    new_policy = {model, new_params}

    # Update all environments with new policy
    updated_envs =
      Enum.map(collector.envs, fn env ->
        SelfPlayEnv.update_p1_policy(env, new_policy)
      end)

    {:ok, %{collector | policy: new_policy, envs: updated_envs}}
  end

  @doc """
  Resample opponents for all environments.

  Called periodically to diversify training.
  """
  @spec resample_opponents(t()) :: {:ok, t()}
  def resample_opponents(%__MODULE__{} = collector) do
    updated_envs =
      Enum.map(collector.envs, fn env ->
        {_type, opponent} = OpponentPool.sample(collector.opponent_pool)
        SelfPlayEnv.update_opponent(env, opponent)
      end)

    {:ok, %{collector | envs: updated_envs}}
  end

  @doc """
  Get current statistics.
  """
  @spec get_stats(t()) :: map()
  def get_stats(%__MODULE__{stats: stats}), do: stats

  @doc """
  Reset all environments.
  """
  @spec reset(t()) :: {:ok, t()}
  def reset(%__MODULE__{} = collector) do
    updated_envs =
      Enum.map(collector.envs, fn env ->
        case SelfPlayEnv.reset(env) do
          {:ok, env} -> env
          _error -> env
        end
      end)

    {:ok, %{collector | envs: updated_envs, stats: init_stats()}}
  end

  @doc """
  Shutdown all environments.
  """
  @spec shutdown(t()) :: :ok
  def shutdown(%__MODULE__{envs: envs}) do
    Enum.each(envs, &SelfPlayEnv.shutdown/1)
    :ok
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp create_environments(num_envs, policy, opponent_pool, config, dolphin_config) do
    results =
      for i <- 1..num_envs do
        # Sample opponent for this environment
        {_type, opponent} = OpponentPool.sample(opponent_pool)

        # Determine P2 policy
        {p2_policy, p2_cpu_level} =
          case opponent do
            %{type: :cpu, level: level} -> {:cpu, level}
            %{type: :current, params: params} -> {params, nil}
            %{type: :historical, params: params} -> {params, nil}
            _ -> {:cpu, 7}
          end

        env_opts = [
          p1_policy: policy,
          p2_policy: p2_policy,
          p2_cpu_level: p2_cpu_level,
          game_type: config.game_type,
          dolphin_config: Map.put(dolphin_config, :instance_id, i),
          p1_port: config.p1_port,
          p2_port: config.p2_port
        ]

        case SelfPlayEnv.new(env_opts) do
          {:ok, env} -> {:ok, env}
          error -> error
        end
      end

    # Check for errors
    case Enum.find(results, fn {status, _} -> status == :error end) do
      {:error, reason} -> {:error, reason}
      nil -> Enum.map(results, fn {:ok, env} -> env end)
    end
  end

  defp collect_async(%__MODULE__{} = collector) do
    # Use Task.async_stream for parallel collection
    tasks =
      Enum.map(collector.envs, fn env ->
        Task.async(fn ->
          collect_from_env(env, collector.rollout_length)
        end)
      end)

    # Await all tasks with timeout
    results = Task.await_many(tasks, collector.config.timeout)

    # Separate results and updated environments
    {rollouts, updated_envs} = Enum.unzip(results)

    {rollouts, updated_envs}
  end

  defp collect_sequential(%__MODULE__{} = collector) do
    results =
      Enum.map(collector.envs, fn env ->
        collect_from_env(env, collector.rollout_length)
      end)

    {rollouts, updated_envs} = Enum.unzip(results)
    {rollouts, updated_envs}
  end

  defp collect_from_env(env, rollout_length) do
    # Collect rollout_length steps from this environment
    {env, trajectory} = collect_trajectory(env, rollout_length, [])

    rollout = trajectory_to_rollout(trajectory)
    {rollout, env}
  end

  defp collect_trajectory(env, 0, acc), do: {env, Enum.reverse(acc)}

  defp collect_trajectory(env, remaining, acc) do
    case SelfPlayEnv.step(env) do
      {:ok, updated_env, experience} ->
        if experience.done do
          # Episode ended, reset and continue
          case SelfPlayEnv.reset(updated_env) do
            {:ok, reset_env} ->
              collect_trajectory(reset_env, remaining - 1, [experience | acc])

            _error ->
              # Can't reset, return what we have
              {updated_env, Enum.reverse([experience | acc])}
          end
        else
          collect_trajectory(updated_env, remaining - 1, [experience | acc])
        end
    end
  end

  defp trajectory_to_rollout(trajectory) when length(trajectory) == 0 do
    %{
      states: [],
      actions: %{},
      rewards: [],
      values: [],
      log_probs: [],
      dones: []
    }
  end

  defp trajectory_to_rollout(trajectory) do
    # Stack all experience into tensors
    states = Enum.map(trajectory, & &1.state)
    rewards = Enum.map(trajectory, & &1.reward)
    dones = Enum.map(trajectory, & &1.done)
    values = Enum.map(trajectory, & &1.value)
    log_probs = Enum.map(trajectory, & &1.log_prob)

    # Actions are maps with button/stick keys
    actions =
      if length(trajectory) > 0 and is_map(hd(trajectory).action) do
        action_keys = Map.keys(hd(trajectory).action)

        Map.new(action_keys, fn key ->
          {key, Enum.map(trajectory, fn exp -> Map.get(exp.action, key) end)}
        end)
      else
        %{}
      end

    %{
      states: states,
      actions: actions,
      rewards: rewards,
      values: values,
      log_probs: log_probs,
      dones: dones
    }
  end

  defp aggregate_rollouts(rollouts) when length(rollouts) == 0 do
    empty_rollout()
  end

  defp aggregate_rollouts(rollouts) do
    # Filter out empty rollouts
    non_empty = Enum.filter(rollouts, fn r -> length(r.states) > 0 end)

    if length(non_empty) == 0 do
      empty_rollout()
    else
      # Concatenate all data across environments
      states = Enum.flat_map(non_empty, & &1.states)
      rewards = Enum.flat_map(non_empty, & &1.rewards)
      dones = Enum.flat_map(non_empty, & &1.dones)
      values = Enum.flat_map(non_empty, & &1.values)
      log_probs = Enum.flat_map(non_empty, & &1.log_probs)

      # Merge action maps
      actions =
        if length(non_empty) > 0 do
          action_keys = Map.keys(hd(non_empty).actions)

          Map.new(action_keys, fn key ->
            {key, Enum.flat_map(non_empty, fn r -> Map.get(r.actions, key, []) end)}
          end)
        else
          %{}
        end

      # Convert to tensors for PPO
      %{
        states: states_to_tensor(states),
        actions: actions_to_tensors(actions),
        rewards: Nx.tensor(rewards, type: :f32),
        values: Nx.tensor(values, type: :f32),
        log_probs: Nx.tensor(log_probs, type: :f32),
        dones: Nx.tensor(Enum.map(dones, fn d -> if d, do: 1.0, else: 0.0 end), type: :f32)
      }
    end
  end

  defp empty_rollout do
    %{
      states: Nx.tensor([], type: :f32),
      actions: %{},
      rewards: Nx.tensor([], type: :f32),
      values: Nx.tensor([], type: :f32),
      log_probs: Nx.tensor([], type: :f32),
      dones: Nx.tensor([], type: :f32)
    }
  end

  defp states_to_tensor([]), do: Nx.tensor([], type: :f32)

  defp states_to_tensor(states) when is_list(states) do
    # States are already embedded tensors, stack them
    if is_struct(hd(states), Nx.Tensor) do
      Nx.stack(states)
    else
      # States are raw GameState structs, need embedding
      # This case shouldn't happen in normal operation
      Logger.warning("States not pre-embedded, performance will be degraded")
      Nx.tensor([], type: :f32)
    end
  end

  defp actions_to_tensors(actions) when map_size(actions) == 0, do: %{}

  defp actions_to_tensors(actions) do
    Map.new(actions, fn {key, values} ->
      {key, Nx.tensor(values, type: :s32)}
    end)
  end

  defp init_stats do
    %{
      total_steps: 0,
      total_episodes: 0,
      total_rewards: 0.0,
      collection_time_ms: 0,
      collections: 0
    }
  end

  defp update_stats(stats, rollouts, elapsed_ms) do
    total_steps = Enum.reduce(rollouts, 0, fn r, acc -> acc + length(r.states) end)

    episodes_completed =
      Enum.reduce(rollouts, 0, fn r, acc ->
        acc + Enum.count(r.dones, & &1)
      end)

    total_reward =
      Enum.reduce(rollouts, 0.0, fn r, acc ->
        acc + Enum.sum(r.rewards)
      end)

    %{
      stats
      | total_steps: stats.total_steps + total_steps,
        total_episodes: stats.total_episodes + episodes_completed,
        total_rewards: stats.total_rewards + total_reward,
        collection_time_ms: stats.collection_time_ms + elapsed_ms,
        collections: stats.collections + 1
    }
  end
end
