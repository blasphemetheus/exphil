defmodule ExPhil.SelfPlay.GamePoolSupervisor do
  @moduledoc """
  DynamicSupervisor for managing a pool of GameRunner processes.

  Handles starting, stopping, and monitoring game instances across the
  self-play training cluster.

  ## Architecture

      ┌────────────────────────────────────────────────────────────────────┐
      │                     GamePoolSupervisor                              │
      │                    (DynamicSupervisor)                              │
      │                                                                     │
      │  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐        │
      │  │  GameRunner  │  │  GameRunner  │  ...  │  GameRunner  │        │
      │  │   game_1     │  │   game_2     │       │   game_N     │        │
      │  └──────────────┘  └──────────────┘       └──────────────┘        │
      │                                                                     │
      └────────────────────────────────────────────────────────────────────┘

  ## Usage

      # Start the supervisor
      {:ok, pool} = GamePoolSupervisor.start_link(name: MyPool)

      # Start games
      {:ok, game_id} = GamePoolSupervisor.start_game(MyPool, [
        p1_policy_id: :current,
        p2_policy_id: :historical_v5,
        game_type: :mock
      ])

      # List all games
      games = GamePoolSupervisor.list_games(MyPool)
      # => [%{id: "game_1", status: :playing, ...}, ...]

      # Stop a specific game
      :ok = GamePoolSupervisor.stop_game(MyPool, "game_1")

      # Stop all games
      :ok = GamePoolSupervisor.stop_all_games(MyPool)

  """

  use DynamicSupervisor

  alias ExPhil.SelfPlay.GameRunner

  require Logger

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the GamePoolSupervisor.

  ## Options
    - `:name` - Name for the supervisor (default: `__MODULE__`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    DynamicSupervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Starts a new game in the pool.

  Returns `{:ok, game_id}` on success.

  ## Options
    - `:game_id` - Explicit game ID (optional, auto-generated if not provided)
    - `:p1_policy_id` - Policy ID for P1 (required)
    - `:p2_policy_id` - Policy ID for P2 (required)
    - `:game_type` - `:mock` or `:dolphin` (default: :mock)
    - All other GameRunner options
  """
  def start_game(pool, opts) do
    game_id = Keyword.get_lazy(opts, :game_id, &generate_game_id/0)
    opts = Keyword.put(opts, :game_id, game_id)

    spec = {GameRunner, opts}

    case DynamicSupervisor.start_child(pool, spec) do
      {:ok, _pid} ->
        Logger.debug("[GamePoolSupervisor] Started game #{game_id}")
        {:ok, game_id}

      {:ok, _pid, _info} ->
        {:ok, game_id}

      {:error, reason} = error ->
        Logger.error("[GamePoolSupervisor] Failed to start game: #{inspect(reason)}")
        error
    end
  end

  @doc """
  Starts multiple games in the pool concurrently.

  Returns a list of `{:ok, game_id}` or `{:error, reason}` tuples.
  """
  def start_games(pool, count, base_opts) do
    1..count
    |> Enum.map(fn i ->
      game_id = "game_#{System.unique_integer([:positive])}_#{i}"
      opts = Keyword.put(base_opts, :game_id, game_id)

      Task.async(fn ->
        start_game(pool, opts)
      end)
    end)
    |> Task.await_many(30_000)
  end

  @doc """
  Stops a game by its ID.
  """
  def stop_game(pool, game_id) do
    case GameRunner.whereis(game_id) do
      nil ->
        {:error, :not_found}

      pid ->
        DynamicSupervisor.terminate_child(pool, pid)
    end
  end

  @doc """
  Stops all games in the pool.
  """
  def stop_all_games(pool) do
    list_games(pool)
    |> Enum.each(fn %{id: game_id} ->
      stop_game(pool, game_id)
    end)

    :ok
  end

  @doc """
  Lists all games in the pool with their status.
  """
  def list_games(pool) do
    DynamicSupervisor.which_children(pool)
    |> Enum.map(fn {_, pid, _, _} ->
      if is_pid(pid) and Process.alive?(pid) do
        try do
          GameRunner.get_status(pid)
        catch
          :exit, _ -> nil
        end
      else
        nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.map(fn status ->
      %{id: status.game_id, status: status.status, frame: status.frame_count}
    end)
  end

  @doc """
  Returns the count of games in each status.
  """
  def count_games(pool) do
    games = list_games(pool)

    %{
      total: length(games),
      waiting: Enum.count(games, &(&1.status == :waiting)),
      playing: Enum.count(games, &(&1.status == :playing)),
      finished: Enum.count(games, &(&1.status == :finished))
    }
  end

  @doc """
  Gets all game runner PIDs.
  """
  def get_runners(pool) do
    DynamicSupervisor.which_children(pool)
    |> Enum.map(fn {_, pid, _, _} -> pid end)
    |> Enum.filter(&(is_pid(&1) and Process.alive?(&1)))
  end

  @doc """
  Steps all games once and collects experiences.
  """
  def step_all(pool) do
    get_runners(pool)
    |> Task.async_stream(
      fn pid ->
        GameRunner.step(pid)
      end,
      timeout: 30_000,
      ordered: false
    )
    |> Enum.map(fn
      {:ok, result} -> result
      {:exit, reason} -> {:error, reason}
    end)
  end

  @doc """
  Collects N steps from all games in parallel.
  """
  def collect_all_steps(pool, n) do
    get_runners(pool)
    |> Task.async_stream(
      fn pid ->
        GameRunner.collect_steps(pid, n)
      end,
      timeout: 120_000,
      ordered: false
    )
    |> Enum.flat_map(fn
      {:ok, {:ok, experiences}} -> experiences
      _ -> []
    end)
  end

  @doc """
  Resets all games in the pool.
  """
  def reset_all(pool) do
    get_runners(pool)
    |> Enum.each(fn pid ->
      GameRunner.reset(pid)
    end)

    :ok
  end

  @doc """
  Updates the opponent policy for all games.
  """
  def update_all_opponents(pool, policy_id) do
    get_runners(pool)
    |> Enum.each(fn pid ->
      GameRunner.swap_policy(pid, :p2, policy_id)
    end)

    :ok
  end

  @doc """
  Updates the learner policy for all games.
  """
  def update_all_learners(pool, policy_id) do
    get_runners(pool)
    |> Enum.each(fn pid ->
      GameRunner.swap_policy(pid, :p1, policy_id)
    end)

    :ok
  end

  # ============================================================================
  # DynamicSupervisor Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(
      strategy: :one_for_one,
      max_restarts: 10,
      max_seconds: 60
    )
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp generate_game_id do
    "game_#{System.unique_integer([:positive, :monotonic])}"
  end
end
