defmodule ExPhil.SelfPlay.Supervisor do
  @moduledoc """
  Top-level supervisor for the self-play training infrastructure.

  Supervises all self-play components and provides a unified API
  for starting, stopping, and managing self-play training.

  ## Architecture

      ┌──────────────────────────────────────────────────────────────────────┐
      │                     SelfPlay.Supervisor                               │
      │                       (Supervisor)                                    │
      │                                                                       │
      │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────────┐│
      │  │    Registry    │ │ Population     │ │   Experience              ││
      │  │  (GameRegistry)│ │   Manager      │ │     Collector             ││
      │  └────────────────┘ └────────────────┘ └────────────────────────────┘│
      │                                                                       │
      │  ┌────────────────────────────────────────────────────────────────┐  │
      │  │                   GamePoolSupervisor                            │  │
      │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       ┌──────────┐     │  │
      │  │  │GameRunner│ │GameRunner│ │GameRunner│  ...  │GameRunner│     │  │
      │  │  │  game_1  │ │  game_2  │ │  game_3  │       │  game_N  │     │  │
      │  │  └──────────┘ └──────────┘ └──────────┘       └──────────┘     │  │
      │  └────────────────────────────────────────────────────────────────┘  │
      │                                                                       │
      │  ┌────────────────────────────────────────────────────────────────┐  │
      │  │                      Matchmaker                                 │  │
      │  │               (Elo ratings, match scheduling)                   │  │
      │  └────────────────────────────────────────────────────────────────┘  │
      │                                                                       │
      └──────────────────────────────────────────────────────────────────────┘

  ## Usage

      # Start the supervisor
      {:ok, sup} = ExPhil.SelfPlay.Supervisor.start_link(
        num_games: 4,
        batch_size: 2048
      )

      # Access components
      pool = ExPhil.SelfPlay.Supervisor.game_pool()
      manager = ExPhil.SelfPlay.Supervisor.population_manager()
      collector = ExPhil.SelfPlay.Supervisor.experience_collector()

      # High-level training API
      ExPhil.SelfPlay.Supervisor.set_policy(model, params)
      {:ok, batch} = ExPhil.SelfPlay.Supervisor.collect_batch(2048)
      ExPhil.SelfPlay.Supervisor.snapshot_policy()

  """

  use Supervisor

  alias ExPhil.SelfPlay.{
    GamePoolSupervisor,
    PopulationManager,
    ExperienceCollector,
    Matchmaker
  }

  require Logger

  @registry_name ExPhil.SelfPlay.GameRegistry
  @pool_name ExPhil.SelfPlay.GamePool
  @population_name ExPhil.SelfPlay.Population
  @collector_name ExPhil.SelfPlay.Collector
  @matchmaker_name ExPhil.SelfPlay.Match

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the SelfPlay supervisor and all child processes.

  ## Options
    - `:name` - Supervisor name (default: `__MODULE__`)
    - `:num_games` - Initial number of games to start (default: 0)
    - `:batch_size` - Experience batch size (default: 2048)
    - `:max_history_size` - Max historical policies (default: 20)
    - `:game_type` - Game type for auto-started games (default: :mock)
    - `:start_matchmaker` - Whether to start matchmaker (default: true)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    Supervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Returns the GamePoolSupervisor PID.
  """
  def game_pool, do: Process.whereis(@pool_name)

  @doc """
  Returns the PopulationManager PID.
  """
  def population_manager, do: Process.whereis(@population_name)

  @doc """
  Returns the ExperienceCollector PID.
  """
  def experience_collector, do: Process.whereis(@collector_name)

  @doc """
  Returns the Matchmaker PID.
  """
  def matchmaker, do: Process.whereis(@matchmaker_name)

  @doc """
  Sets the current policy for training.
  """
  def set_policy(model, params) do
    PopulationManager.set_current(population_manager(), model, params)
  end

  @doc """
  Updates policy parameters (model unchanged).
  """
  def update_policy_params(params) do
    PopulationManager.update_params(population_manager(), params)

    # Update all games with new policy
    GamePoolSupervisor.update_all_learners(game_pool(), :current)
  end

  @doc """
  Snapshots the current policy to history.
  """
  def snapshot_policy do
    PopulationManager.snapshot(population_manager())
  end

  @doc """
  Gets the current policy.
  """
  def get_policy do
    PopulationManager.get_current(population_manager())
  end

  @doc """
  Samples an opponent from the population.
  """
  def sample_opponent(opts \\ []) do
    PopulationManager.sample_opponent(population_manager(), opts)
  end

  @doc """
  Collects a batch of experiences.

  Blocks until `size` experiences are available.
  """
  def collect_batch(size, timeout \\ 60_000) do
    ExperienceCollector.get_batch(experience_collector(), size, timeout)
  end

  @doc """
  Gets all available experiences.
  """
  def get_all_experiences do
    ExperienceCollector.get_all(experience_collector())
  end

  @doc """
  Flushes experiences and returns remaining.
  """
  def flush_experiences do
    ExperienceCollector.flush(experience_collector())
  end

  @doc """
  Starts a new game.
  """
  def start_game(opts \\ []) do
    opts = Keyword.merge([
      population_manager: population_manager(),
      experience_collector: experience_collector()
    ], opts)

    # Default policy IDs if not provided
    opts = Keyword.put_new(opts, :p1_policy_id, :current)
    opts = Keyword.put_new(opts, :p2_policy_id, sample_opponent_id())

    GamePoolSupervisor.start_game(game_pool(), opts)
  end

  @doc """
  Starts multiple games.
  """
  def start_games(count, opts \\ []) do
    base_opts = Keyword.merge([
      population_manager: population_manager(),
      experience_collector: experience_collector(),
      p1_policy_id: :current,
      p2_policy_id: :cpu  # Default to CPU opponent
    ], opts)

    GamePoolSupervisor.start_games(game_pool(), count, base_opts)
  end

  @doc """
  Stops a game.
  """
  def stop_game(game_id) do
    GamePoolSupervisor.stop_game(game_pool(), game_id)
  end

  @doc """
  Stops all games.
  """
  def stop_all_games do
    GamePoolSupervisor.stop_all_games(game_pool())
  end

  @doc """
  Lists all active games.
  """
  def list_games do
    GamePoolSupervisor.list_games(game_pool())
  end

  @doc """
  Gets game counts by status.
  """
  def count_games do
    GamePoolSupervisor.count_games(game_pool())
  end

  @doc """
  Steps all games once.
  """
  def step_all_games do
    GamePoolSupervisor.step_all(game_pool())
  end

  @doc """
  Collects N steps from all games.
  """
  def collect_steps(n) do
    GamePoolSupervisor.collect_all_steps(game_pool(), n)
  end

  @doc """
  Resamples opponents for all games.
  """
  def resample_all_opponents do
    games = list_games()

    Enum.each(games, fn %{id: game_id} ->
      {:ok, {policy_id, _policy}} = sample_opponent()
      case ExPhil.SelfPlay.GameRunner.whereis(game_id) do
        nil -> :ok
        pid -> ExPhil.SelfPlay.GameRunner.swap_policy(pid, :p2, policy_id)
      end
    end)

    :ok
  end

  @doc """
  Gets overall training stats.
  """
  def get_stats do
    %{
      games: count_games(),
      collector: ExperienceCollector.get_stats(experience_collector()),
      population: PopulationManager.get_stats(population_manager()),
      matchmaker: if(matchmaker(), do: Matchmaker.get_stats(matchmaker()), else: nil)
    }
  end

  @doc """
  Reports a game result to matchmaker.
  """
  def report_game_result(p1_id, p2_id, result) do
    if matchmaker() do
      Matchmaker.report_result(matchmaker(), p1_id, p2_id, result)
    else
      :ok
    end
  end

  @doc """
  Gets Elo leaderboard.
  """
  def get_leaderboard(limit \\ 10) do
    if matchmaker() do
      Matchmaker.get_leaderboard(matchmaker(), limit)
    else
      []
    end
  end

  # ============================================================================
  # Supervisor Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    batch_size = Keyword.get(opts, :batch_size, 2048)
    max_history_size = Keyword.get(opts, :max_history_size, 20)
    start_matchmaker = Keyword.get(opts, :start_matchmaker, true)

    children = [
      # Registry for game lookup
      {Registry, keys: :unique, name: @registry_name},

      # Population Manager
      {PopulationManager, [
        name: @population_name,
        max_history_size: max_history_size
      ]},

      # Experience Collector
      {ExperienceCollector, [
        name: @collector_name,
        batch_size: batch_size
      ]},

      # Game Pool Supervisor
      {GamePoolSupervisor, [
        name: @pool_name
      ]}
    ]

    # Conditionally add matchmaker
    children = if start_matchmaker do
      children ++ [{Matchmaker, [name: @matchmaker_name]}]
    else
      children
    end

    Logger.info("[SelfPlay.Supervisor] Starting with batch_size=#{batch_size}, max_history=#{max_history_size}")

    Supervisor.init(children, strategy: :one_for_one)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp sample_opponent_id do
    case sample_opponent() do
      {:ok, {policy_id, _}} -> policy_id
      _ -> {:cpu, 7}
    end
  end
end
