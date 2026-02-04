defmodule ExPhil.League do
  @moduledoc """
  GenServer managing an architecture competition league.

  The League coordinates competitions between different neural network architectures,
  tracking Elo ratings, managing tournaments, and collecting match experiences for
  self-play training.

  ## Architecture

  ```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           League Manager                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │                    Architecture Registry                          │ │
  │  │  %{                                                               │ │
  │  │    :mamba_mewtwo => %ArchitectureEntry{elo: 1450, ...},          │ │
  │  │    :lstm_mewtwo => %ArchitectureEntry{elo: 1380, ...},           │ │
  │  │    :attention_mewtwo => %ArchitectureEntry{elo: 1420, ...}       │ │
  │  │  }                                                                │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │                      Match Runner                                 │ │
  │  │  - Load both policies                                            │ │
  │  │  - Execute match (MockEnv or Dolphin)                            │ │
  │  │  - Collect experiences                                           │ │
  │  │  - Update Elo ratings                                            │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │                    Experience Pool                                │ │
  │  │  Collected match experiences for PPO training                    │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  ```

  ## Usage

      # Start the league
      {:ok, league} = League.start_link(game_type: :mock)

      # Register architectures
      League.register_architecture(league, :mamba_mewtwo, model, params, config)
      League.register_architecture(league, :lstm_mewtwo, model, params, config)

      # Run tournament
      {:ok, results} = League.run_tournament(league, matches_per_pair: 10)

      # Get leaderboard
      leaderboard = League.get_leaderboard(league)

      # Run a single match
      {:ok, result} = League.run_match(league, :mamba_mewtwo, :lstm_mewtwo)

  """

  use GenServer

  alias ExPhil.League.ArchitectureEntry
  alias ExPhil.SelfPlay.Elo
  alias ExPhil.MockEnv
  alias ExPhil.Error.LeagueError

  require Logger

  defstruct [
    # Map of arch_id => ArchitectureEntry
    :architectures,
    # List of MatchResult records
    :match_history,
    # Collected experiences for training
    :experience_pool,
    # Current league generation
    :generation,
    # League configuration
    :config,
    # Global statistics
    :stats
  ]

  @type match_result :: %{
          p1_id: atom(),
          p2_id: atom(),
          winner: :p1 | :p2 | :draw,
          p1_stocks: non_neg_integer(),
          p2_stocks: non_neg_integer(),
          frames: non_neg_integer(),
          experiences: [map()],
          elo_change: {float(), float()},
          timestamp: integer()
        }

  @type t :: %__MODULE__{}

  @default_config %{
    # :mock or :dolphin
    game_type: :mock,
    # Required if game_type == :dolphin
    dolphin_config: nil,
    # Stocks per game
    stocks: 4,
    # 8 minutes in frames (at 60fps)
    time_limit: 8 * 60 * 60,
    # Whether to collect experiences during matches
    collect_experiences: true,
    # Max experiences to keep
    experience_pool_size: 100_000
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start the league manager.

  ## Options

  - `:game_type` - `:mock` or `:dolphin` (default: :mock)
  - `:dolphin_config` - Dolphin configuration (required if game_type is :dolphin)
  - `:stocks` - Stocks per game (default: 4)
  - `:name` - Process name (default: __MODULE__)

  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Register a new architecture in the league.

  ## Parameters

  - `arch_id` - Unique identifier for the architecture
  - `model` - Compiled Axon model
  - `params` - Trained parameters
  - `config` - Architecture configuration

  """
  @spec register_architecture(GenServer.server(), atom(), Axon.t(), map(), map()) ::
          :ok | {:error, term()}
  def register_architecture(league, arch_id, model, params, config) do
    GenServer.call(league, {:register_architecture, arch_id, model, params, config})
  end

  @doc """
  Register an architecture from an ArchitectureEntry struct.
  """
  @spec register_entry(GenServer.server(), ArchitectureEntry.t()) :: :ok | {:error, term()}
  def register_entry(league, %ArchitectureEntry{} = entry) do
    GenServer.call(league, {:register_entry, entry})
  end

  @doc """
  Unregister an architecture from the league.
  """
  @spec unregister_architecture(GenServer.server(), atom()) :: :ok | {:error, term()}
  def unregister_architecture(league, arch_id) do
    GenServer.call(league, {:unregister_architecture, arch_id})
  end

  @doc """
  Get an architecture entry by ID.
  """
  @spec get_architecture(GenServer.server(), atom()) ::
          {:ok, ArchitectureEntry.t()} | {:error, LeagueError.t()}
  def get_architecture(league, arch_id) do
    GenServer.call(league, {:get_architecture, arch_id})
  end

  @doc """
  List all registered architectures.
  """
  @spec list_architectures(GenServer.server()) :: [ArchitectureEntry.t()]
  def list_architectures(league) do
    GenServer.call(league, :list_architectures)
  end

  @doc """
  Run a tournament (round-robin or partial).

  ## Options

  - `:matches_per_pair` - Number of matches between each pair (default: 10)
  - `:arch_ids` - Specific architectures to include (default: all)

  ## Returns

  List of match results.
  """
  @spec run_tournament(GenServer.server(), keyword()) :: {:ok, [match_result()]}
  def run_tournament(league, opts \\ []) do
    GenServer.call(league, {:run_tournament, opts}, :infinity)
  end

  @doc """
  Run a single match between two architectures.
  """
  @spec run_match(GenServer.server(), atom(), atom()) :: {:ok, match_result()} | {:error, term()}
  def run_match(league, p1_id, p2_id) do
    GenServer.call(league, {:run_match, p1_id, p2_id}, :infinity)
  end

  @doc """
  Get the leaderboard sorted by Elo rating.
  """
  @spec get_leaderboard(GenServer.server(), non_neg_integer()) :: [map()]
  def get_leaderboard(league, limit \\ 10) do
    GenServer.call(league, {:get_leaderboard, limit})
  end

  @doc """
  Get match history for an architecture.
  """
  @spec get_match_history(GenServer.server(), atom(), non_neg_integer()) :: [match_result()]
  def get_match_history(league, arch_id, limit \\ 20) do
    GenServer.call(league, {:get_match_history, arch_id, limit})
  end

  @doc """
  Get all collected experiences from the pool.
  """
  @spec get_experiences(GenServer.server()) :: [map()]
  def get_experiences(league) do
    GenServer.call(league, :get_experiences)
  end

  @doc """
  Clear the experience pool.
  """
  @spec clear_experiences(GenServer.server()) :: :ok
  def clear_experiences(league) do
    GenServer.call(league, :clear_experiences)
  end

  @doc """
  Update an architecture's parameters (after training).
  """
  @spec update_params(GenServer.server(), atom(), map()) :: :ok | {:error, term()}
  def update_params(league, arch_id, new_params) do
    GenServer.call(league, {:update_params, arch_id, new_params})
  end

  @doc """
  Advance the league to the next generation.
  """
  @spec advance_generation(GenServer.server()) :: {:ok, non_neg_integer()}
  def advance_generation(league) do
    GenServer.call(league, :advance_generation)
  end

  @doc """
  Get current league generation.
  """
  @spec get_generation(GenServer.server()) :: non_neg_integer()
  def get_generation(league) do
    GenServer.call(league, :get_generation)
  end

  @doc """
  Get league statistics.
  """
  @spec get_stats(GenServer.server()) :: map()
  def get_stats(league) do
    GenServer.call(league, :get_stats)
  end

  @doc """
  Save all architecture checkpoints to a directory.
  """
  @spec save_all_checkpoints(GenServer.server(), String.t()) :: :ok
  def save_all_checkpoints(league, dir) do
    GenServer.call(league, {:save_all_checkpoints, dir})
  end

  @doc """
  Load architectures from a checkpoint directory.
  """
  @spec load_checkpoints(GenServer.server(), String.t()) :: {:ok, non_neg_integer()}
  def load_checkpoints(league, dir) do
    GenServer.call(league, {:load_checkpoints, dir})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    config = Map.merge(@default_config, Map.new(opts))

    state = %__MODULE__{
      architectures: %{},
      match_history: [],
      experience_pool: [],
      generation: 0,
      config: config,
      stats: init_stats()
    }

    Logger.info("[League] Started with game_type=#{config.game_type}")

    {:ok, state}
  end

  @impl true
  def handle_call({:register_architecture, arch_id, model, params, config}, _from, state) do
    if Map.has_key?(state.architectures, arch_id) do
      {:reply, {:error, LeagueError.new(:already_registered, agent_id: to_string(arch_id))}, state}
    else
      arch_config = Map.get(config, :architecture, :mlp)
      character = Map.get(config, :character, :mewtwo)

      case ArchitectureEntry.new(
             id: arch_id,
             architecture: arch_config,
             character: character,
             model: model,
             params: params,
             config: config
           ) do
        {:ok, entry} ->
          new_state = %{
            state
            | architectures: Map.put(state.architectures, arch_id, entry),
              stats: update_stats(state.stats, :architecture_registered)
          }

          Logger.info("[League] Registered architecture #{arch_id} (#{arch_config})")
          {:reply, :ok, new_state}

        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end

  @impl true
  def handle_call({:register_entry, %ArchitectureEntry{} = entry}, _from, state) do
    if Map.has_key?(state.architectures, entry.id) do
      {:reply, {:error, LeagueError.new(:already_registered, agent_id: to_string(entry.id))}, state}
    else
      new_state = %{
        state
        | architectures: Map.put(state.architectures, entry.id, entry),
          stats: update_stats(state.stats, :architecture_registered)
      }

      Logger.info("[League] Registered architecture #{entry.id} (#{entry.architecture})")
      {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:unregister_architecture, arch_id}, _from, state) do
    if Map.has_key?(state.architectures, arch_id) do
      new_state = %{state | architectures: Map.delete(state.architectures, arch_id)}
      Logger.info("[League] Unregistered architecture #{arch_id}")
      {:reply, :ok, new_state}
    else
      {:reply, {:error, LeagueError.new(:not_found, agent_id: to_string(arch_id))}, state}
    end
  end

  @impl true
  def handle_call({:get_architecture, arch_id}, _from, state) do
    case Map.get(state.architectures, arch_id) do
      nil -> {:reply, {:error, LeagueError.new(:not_found, agent_id: to_string(arch_id))}, state}
      entry -> {:reply, {:ok, entry}, state}
    end
  end

  @impl true
  def handle_call(:list_architectures, _from, state) do
    {:reply, Map.values(state.architectures), state}
  end

  @impl true
  def handle_call({:run_tournament, opts}, _from, state) do
    matches_per_pair = Keyword.get(opts, :matches_per_pair, 10)
    arch_ids = Keyword.get(opts, :arch_ids, Map.keys(state.architectures))

    # Generate round-robin pairs
    pairs = for p1 <- arch_ids, p2 <- arch_ids, p1 < p2, do: {p1, p2}

    Logger.info(
      "[League] Running tournament: #{length(pairs)} pairs, #{matches_per_pair} matches each"
    )

    # Run all matches
    {results, new_state} =
      Enum.reduce(pairs, {[], state}, fn {p1_id, p2_id}, {results_acc, state_acc} ->
        # Run multiple matches per pair
        {pair_results, updated_state} =
          Enum.reduce(1..matches_per_pair, {[], state_acc}, fn _i, {pair_acc, s} ->
            case do_run_match(s, p1_id, p2_id) do
              {:ok, result, new_state} ->
                {[result | pair_acc], new_state}

              {:error, _reason} ->
                {pair_acc, s}
            end
          end)

        {pair_results ++ results_acc, updated_state}
      end)

    Logger.info("[League] Tournament complete: #{length(results)} matches played")

    {:reply, {:ok, results}, new_state}
  end

  @impl true
  def handle_call({:run_match, p1_id, p2_id}, _from, state) do
    case do_run_match(state, p1_id, p2_id) do
      {:ok, result, new_state} -> {:reply, {:ok, result}, new_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:get_leaderboard, limit}, _from, state) do
    leaderboard =
      state.architectures
      |> Map.values()
      |> Enum.map(fn entry ->
        %{
          id: entry.id,
          architecture: entry.architecture,
          character: entry.character,
          elo: entry.elo,
          generation: entry.generation,
          wins: entry.stats.wins,
          losses: entry.stats.losses,
          draws: entry.stats.draws,
          win_rate: ArchitectureEntry.win_rate(entry),
          games_played: ArchitectureEntry.games_played(entry)
        }
      end)
      |> Enum.sort_by(& &1.elo, :desc)
      |> Enum.take(limit)

    {:reply, leaderboard, state}
  end

  @impl true
  def handle_call({:get_match_history, arch_id, limit}, _from, state) do
    history =
      state.match_history
      |> Enum.filter(fn m -> m.p1_id == arch_id or m.p2_id == arch_id end)
      |> Enum.take(limit)

    {:reply, history, state}
  end

  @impl true
  def handle_call(:get_experiences, _from, state) do
    {:reply, state.experience_pool, state}
  end

  @impl true
  def handle_call(:clear_experiences, _from, state) do
    {:reply, :ok, %{state | experience_pool: []}}
  end

  @impl true
  def handle_call({:update_params, arch_id, new_params}, _from, state) do
    case Map.get(state.architectures, arch_id) do
      nil ->
        {:reply, {:error, LeagueError.new(:not_found, agent_id: to_string(arch_id))}, state}

      entry ->
        updated_entry = ArchitectureEntry.update_from_training(entry, new_params)
        new_state = %{state | architectures: Map.put(state.architectures, arch_id, updated_entry)}
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call(:advance_generation, _from, state) do
    new_gen = state.generation + 1
    Logger.info("[League] Advanced to generation #{new_gen}")
    {:reply, {:ok, new_gen}, %{state | generation: new_gen}}
  end

  @impl true
  def handle_call(:get_generation, _from, state) do
    {:reply, state.generation, state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats =
      Map.merge(state.stats, %{
        num_architectures: map_size(state.architectures),
        total_matches: length(state.match_history),
        experience_pool_size: length(state.experience_pool),
        generation: state.generation
      })

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:save_all_checkpoints, dir}, _from, state) do
    File.mkdir_p!(dir)

    # Save each architecture
    Enum.each(state.architectures, fn {arch_id, entry} ->
      # Save params
      params_path = Path.join(dir, "#{arch_id}_params.bin")
      save_params(entry.params, params_path)

      # Save metadata
      metadata_path = Path.join(dir, "#{arch_id}_metadata.json")
      metadata = ArchitectureEntry.to_metadata(entry)
      File.write!(metadata_path, Jason.encode!(metadata, pretty: true))

      Logger.debug("[League] Saved checkpoint for #{arch_id}")
    end)

    # Save league state
    league_state_path = Path.join(dir, "league_state.json")

    league_state = %{
      generation: state.generation,
      stats: state.stats,
      architecture_ids: Map.keys(state.architectures)
    }

    File.write!(league_state_path, Jason.encode!(league_state, pretty: true))

    Logger.info("[League] Saved #{map_size(state.architectures)} checkpoints to #{dir}")

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:load_checkpoints, dir}, _from, state) do
    # Load league state
    league_state_path = Path.join(dir, "league_state.json")

    {generation, arch_ids} =
      if File.exists?(league_state_path) do
        league_state = File.read!(league_state_path) |> Jason.decode!()
        {league_state["generation"] || 0, league_state["architecture_ids"] || []}
      else
        # Discover from files
        metadata_files = Path.wildcard(Path.join(dir, "*_metadata.json"))

        ids =
          Enum.map(metadata_files, fn f ->
            f
            |> Path.basename()
            |> String.replace("_metadata.json", "")
            |> String.to_atom()
          end)

        {0, ids}
      end

    # Load each architecture
    {architectures, loaded_count} =
      Enum.reduce(arch_ids, {state.architectures, 0}, fn arch_id, {acc, count} ->
        metadata_path = Path.join(dir, "#{arch_id}_metadata.json")
        params_path = Path.join(dir, "#{arch_id}_params.bin")

        if File.exists?(metadata_path) and File.exists?(params_path) do
          metadata = File.read!(metadata_path) |> Jason.decode!()
          params = load_params(params_path)

          {:ok, entry} = ArchitectureEntry.from_metadata(metadata)
          entry = %{entry | params: params}
          {Map.put(acc, entry.id, entry), count + 1}
        else
          Logger.warning("[League] Missing files for #{arch_id}")
          {acc, count}
        end
      end)

    new_state = %{state | architectures: architectures, generation: generation}

    Logger.info("[League] Loaded #{loaded_count} architectures from #{dir}")

    {:reply, {:ok, loaded_count}, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_run_match(state, p1_id, p2_id) do
    p1_entry = Map.get(state.architectures, p1_id)
    p2_entry = Map.get(state.architectures, p2_id)

    cond do
      is_nil(p1_entry) ->
        {:error, {:not_found, p1_id}}

      is_nil(p2_entry) ->
        {:error, {:not_found, p2_id}}

      not ArchitectureEntry.trained?(p1_entry) ->
        {:error, {:not_trained, p1_id}}

      not ArchitectureEntry.trained?(p2_entry) ->
        {:error, {:not_trained, p2_id}}

      true ->
        # Run the match
        {result, experiences} = run_game(state.config, p1_entry, p2_entry)

        # Update Elo ratings
        elo_result =
          case result.winner do
            :p1 -> :win
            :p2 -> :loss
            :draw -> :draw
          end

        k1 = Elo.dynamic_k_factor(ArchitectureEntry.games_played(p1_entry))
        k2 = Elo.dynamic_k_factor(ArchitectureEntry.games_played(p2_entry))

        {new_p1_elo, new_p2_elo} =
          Elo.update(
            p1_entry.elo,
            p2_entry.elo,
            elo_result,
            k_factor_a: k1,
            k_factor_b: k2
          )

        elo_change = {new_p1_elo - p1_entry.elo, new_p2_elo - p2_entry.elo}

        # Update entries
        p1_result =
          if result.winner == :p1,
            do: :win,
            else: if(result.winner == :p2, do: :loss, else: :draw)

        p2_result =
          if result.winner == :p2,
            do: :win,
            else: if(result.winner == :p1, do: :loss, else: :draw)

        updated_p1 =
          p1_entry
          |> ArchitectureEntry.update_elo(new_p1_elo)
          |> ArchitectureEntry.record_result(p1_result, result.frames)

        updated_p2 =
          p2_entry
          |> ArchitectureEntry.update_elo(new_p2_elo)
          |> ArchitectureEntry.record_result(p2_result, result.frames)

        # Build match result
        match_result = %{
          p1_id: p1_id,
          p2_id: p2_id,
          winner: result.winner,
          p1_stocks: result.p1_stocks,
          p2_stocks: result.p2_stocks,
          frames: result.frames,
          experiences: if(state.config.collect_experiences, do: experiences, else: []),
          elo_change: elo_change,
          timestamp: System.system_time(:second)
        }

        # Update state
        new_architectures =
          state.architectures
          |> Map.put(p1_id, updated_p1)
          |> Map.put(p2_id, updated_p2)

        # Add experiences to pool (with size limit)
        new_pool =
          if state.config.collect_experiences do
            (experiences ++ state.experience_pool)
            |> Enum.take(state.config.experience_pool_size)
          else
            state.experience_pool
          end

        new_state = %{
          state
          | architectures: new_architectures,
            match_history: [match_result | Enum.take(state.match_history, 999)],
            experience_pool: new_pool,
            stats: update_stats(state.stats, :match_completed)
        }

        Logger.debug(
          "[League] Match: #{p1_id} vs #{p2_id} -> #{result.winner} " <>
            "(stocks: #{result.p1_stocks}-#{result.p2_stocks}, " <>
            "Elo: #{Float.round(elem(elo_change, 0), 1)}/#{Float.round(elem(elo_change, 1), 1)})"
        )

        {:ok, match_result, new_state}
    end
  end

  defp run_game(config, p1_entry, p2_entry) do
    case config.game_type do
      :mock ->
        run_mock_game(config, p1_entry, p2_entry)

      :dolphin ->
        run_dolphin_game(config, p1_entry, p2_entry)
    end
  end

  defp run_mock_game(config, p1_entry, p2_entry) do
    # Initialize mock game
    game =
      MockEnv.Game.new(
        stocks: config.stocks,
        p1_character: p1_entry.character,
        p2_character: p2_entry.character
      )

    # Build policy functions
    p1_policy = build_policy_fn(p1_entry)
    p2_policy = build_policy_fn(p2_entry)

    # Run game loop
    {final_game, experiences} =
      run_game_loop(
        game,
        p1_policy,
        p2_policy,
        config.time_limit,
        []
      )

    # Determine winner
    p1_stocks = MockEnv.Game.get_stocks(final_game, :p1)
    p2_stocks = MockEnv.Game.get_stocks(final_game, :p2)

    winner =
      cond do
        p1_stocks > p2_stocks -> :p1
        p2_stocks > p1_stocks -> :p2
        true -> :draw
      end

    result = %{
      winner: winner,
      p1_stocks: p1_stocks,
      p2_stocks: p2_stocks,
      frames: MockEnv.Game.get_frame(final_game)
    }

    {result, experiences}
  end

  defp run_game_loop(game, p1_policy, p2_policy, time_limit, experiences) do
    frame = MockEnv.Game.get_frame(game)

    if MockEnv.Game.is_over?(game) or frame >= time_limit do
      {game, Enum.reverse(experiences)}
    else
      # Get game state
      state = MockEnv.Game.get_state(game)

      # Get actions from policies
      p1_action = p1_policy.(state, :p1)
      p2_action = p2_policy.(state, :p2)

      # Step the game
      new_game = MockEnv.Game.step(game, p1_action, p2_action)

      # Collect experience (for P1 - can be extended to both)
      experience = %{
        state: state,
        action: p1_action,
        reward: compute_reward(game, new_game, :p1),
        done: MockEnv.Game.is_over?(new_game),
        frame: frame
      }

      run_game_loop(new_game, p1_policy, p2_policy, time_limit, [experience | experiences])
    end
  end

  defp run_dolphin_game(_config, _p1_entry, _p2_entry) do
    # Dolphin integration - placeholder for now
    # This would use MeleePort to run actual games
    raise "Dolphin game execution not yet implemented"
  end

  defp build_policy_fn(entry) do
    fn state, _player ->
      # Use the trained model to predict action
      if entry.model && map_size(entry.params) > 0 do
        # Embed state and run inference
        embedded = embed_state(state)
        predict_action(entry.model, entry.params, embedded)
      else
        # Random action fallback
        random_action()
      end
    end
  end

  defp embed_state(state) do
    # Simple embedding - actual implementation would use ExPhil.Embeddings
    # For now, flatten state to tensor
    state
    |> Map.values()
    |> Enum.flat_map(fn
      v when is_number(v) -> [v]
      v when is_list(v) -> List.flatten(v)
      _ -> [0.0]
    end)
    |> Nx.tensor(type: :f32)
    # Add batch dim
    |> Nx.new_axis(0)
  end

  defp predict_action(model, params, state) do
    # Run inference
    output = Axon.predict(model, params, %{"state" => state})

    # Sample action from policy output
    # For now, return a simple action struct
    %{
      buttons: sample_buttons(output),
      main_x: sample_stick(output, :main_x),
      main_y: sample_stick(output, :main_y),
      c_x: sample_stick(output, :c_x),
      c_y: sample_stick(output, :c_y),
      shoulder: 0.0
    }
  end

  defp sample_buttons(output) do
    case output do
      %{"buttons" => buttons} -> Nx.argmax(buttons, axis: -1) |> Nx.to_number()
      _ -> 0
    end
  end

  defp sample_stick(output, key) do
    key_str = to_string(key)

    case output do
      %{^key_str => stick} -> Nx.argmax(stick, axis: -1) |> Nx.to_number() |> normalize_stick()
      _ -> 0.5
    end
  end

  defp normalize_stick(bucket) do
    # Convert bucket index to -1..1 range
    bucket / 16 * 2 - 1
  end

  defp random_action do
    %{
      buttons: 0,
      main_x: :rand.uniform() * 2 - 1,
      main_y: :rand.uniform() * 2 - 1,
      c_x: 0.0,
      c_y: 0.0,
      shoulder: 0.0
    }
  end

  defp compute_reward(old_game, new_game, player) do
    # Simple reward: damage dealt - damage taken + stock differential
    old_state = MockEnv.Game.get_state(old_game)
    new_state = MockEnv.Game.get_state(new_game)

    opponent = if player == :p1, do: :p2, else: :p1

    damage_dealt = new_state[opponent].damage - old_state[opponent].damage
    damage_taken = new_state[player].damage - old_state[player].damage
    stock_change = (new_state[player].stocks - old_state[player].stocks) * 100

    damage_dealt - damage_taken + stock_change
  end

  defp init_stats do
    %{
      matches_played: 0,
      architectures_registered: 0,
      total_frames_played: 0
    }
  end

  defp update_stats(stats, :match_completed) do
    %{stats | matches_played: stats.matches_played + 1}
  end

  defp update_stats(stats, :architecture_registered) do
    %{stats | architectures_registered: stats.architectures_registered + 1}
  end

  defp save_params(params, path) do
    # Convert to binary backend for serialization
    params_binary =
      params
      |> Enum.map(fn {k, v} ->
        v_binary = Nx.backend_transfer(v, Nx.BinaryBackend)
        {k, v_binary}
      end)
      |> Map.new()

    binary = :erlang.term_to_binary(params_binary)
    File.write!(path, binary)
  end

  defp load_params(path) do
    binary = File.read!(path)
    :erlang.binary_to_term(binary)
  end
end
