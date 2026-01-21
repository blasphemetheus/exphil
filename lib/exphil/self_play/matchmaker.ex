defmodule ExPhil.SelfPlay.Matchmaker do
  @moduledoc """
  GenServer managing Elo ratings and matchmaking for self-play training.

  Tracks player ratings, schedules matches, and supports various
  matchmaking strategies.

  ## Matchmaking Strategies

  - `:self_play` - Current policy vs itself
  - `:historical` - Current vs random historical policy
  - `:skill_based` - Match players within Elo range
  - `:exploiter` - Prioritize opponents with low win rate (PFSP)

  ## Architecture

      ┌────────────────────────────────────────────────────────────────────┐
      │                        Matchmaker                                   │
      │                                                                     │
      │  ┌────────────────────────────────────────────────────────────┐    │
      │  │                    Elo Ratings                              │    │
      │  │  %{                                                        │    │
      │  │    "current" => %{rating: 1200, wins: 10, losses: 5, ...}, │    │
      │  │    "v42" => %{rating: 1150, wins: 8, losses: 7, ...},      │    │
      │  │    ...                                                      │    │
      │  │  }                                                          │    │
      │  └────────────────────────────────────────────────────────────┘    │
      │                            │                                        │
      │                            ▼                                        │
      │  ┌────────────────────────────────────────────────────────────┐    │
      │  │                 Matchmaking Engine                          │    │
      │  │                                                             │    │
      │  │  Strategies:                                                │    │
      │  │  - self_play: Latest vs itself                             │    │
      │  │  - historical: Latest vs past version                      │    │
      │  │  - skill_based: Similar Elo ratings                        │    │
      │  │  - exploiter: Target weak opponents (PFSP)                 │    │
      │  └────────────────────────────────────────────────────────────┘    │
      │                                                                     │
      └────────────────────────────────────────────────────────────────────┘

  ## Usage

      {:ok, mm} = Matchmaker.start_link(k_factor: 32)

      # Request a match for a game runner
      {:ok, match} = Matchmaker.request_match(mm, "game_1", strategy: :skill_based)
      # => %{match_id: "m_123", p1: "current", p2: "v40", p1_rating: 1200, p2_rating: 1180}

      # Report result
      :ok = Matchmaker.report_result(mm, "current", "v40", :win)

      # Get leaderboard
      leaderboard = Matchmaker.get_leaderboard(mm, 10)

  """

  use GenServer

  alias ExPhil.SelfPlay.Elo

  require Logger

  defstruct [
    :ratings,              # Map of policy_id => rating info
    :pending_matches,      # Map of match_id => match info
    :completed_matches,    # List of completed match records
    :k_factor,
    :elo_range,            # Range for skill-based matching
    :strategy_weights,
    :stats
  ]

  @type strategy :: :self_play | :historical | :skill_based | :exploiter | :random
  @type result :: :win | :loss | :draw

  @default_opts %{
    k_factor: 32,
    elo_range: 100,  # Match within +/- this range for skill_based
    strategy_weights: %{
      self_play: 0.3,
      historical: 0.3,
      skill_based: 0.2,
      exploiter: 0.2
    }
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the Matchmaker.

  ## Options
    - `:k_factor` - Elo K-factor (default: 32)
    - `:elo_range` - Range for skill-based matching (default: 100)
    - `:strategy_weights` - Weights for different strategies
    - `:name` - Name for the GenServer
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Requests a match for a game.

  Returns match info including opponent and ratings.

  ## Options
    - `:strategy` - Matchmaking strategy (default: weighted random)
    - `:player_id` - ID of the requesting player (default: :current)
  """
  def request_match(matchmaker, game_id, opts \\ []) do
    GenServer.call(matchmaker, {:request_match, game_id, opts})
  end

  @doc """
  Reports a match result.

  Updates Elo ratings for both players.
  """
  def report_result(matchmaker, p1_id, p2_id, result) when result in [:win, :loss, :draw] do
    GenServer.call(matchmaker, {:report_result, p1_id, p2_id, result})
  end

  @doc """
  Reports a match result with game stats.
  """
  def report_result(matchmaker, p1_id, p2_id, result, stats) do
    GenServer.call(matchmaker, {:report_result_with_stats, p1_id, p2_id, result, stats})
  end

  @doc """
  Gets the rating for a specific policy.
  """
  def get_rating(matchmaker, policy_id) do
    GenServer.call(matchmaker, {:get_rating, policy_id})
  end

  @doc """
  Gets all ratings.
  """
  def get_ratings(matchmaker) do
    GenServer.call(matchmaker, :get_ratings)
  end

  @doc """
  Gets the leaderboard (top N by rating).
  """
  def get_leaderboard(matchmaker, limit \\ 10) do
    GenServer.call(matchmaker, {:get_leaderboard, limit})
  end

  @doc """
  Gets matchmaker statistics.
  """
  def get_stats(matchmaker) do
    GenServer.call(matchmaker, :get_stats)
  end

  @doc """
  Registers a new policy in the rating system.
  """
  def register_policy(matchmaker, policy_id, initial_rating \\ nil) do
    GenServer.call(matchmaker, {:register_policy, policy_id, initial_rating})
  end

  @doc """
  Gets win rate between two policies.
  """
  def get_win_rate(matchmaker, p1_id, p2_id) do
    GenServer.call(matchmaker, {:get_win_rate, p1_id, p2_id})
  end

  @doc """
  Gets match history for a policy.
  """
  def get_match_history(matchmaker, policy_id, limit \\ 20) do
    GenServer.call(matchmaker, {:get_match_history, policy_id, limit})
  end

  @doc """
  Resets all ratings.
  """
  def reset(matchmaker) do
    GenServer.call(matchmaker, :reset)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    state = %__MODULE__{
      ratings: %{},
      pending_matches: %{},
      completed_matches: [],
      k_factor: Keyword.get(opts, :k_factor, @default_opts.k_factor),
      elo_range: Keyword.get(opts, :elo_range, @default_opts.elo_range),
      strategy_weights: Keyword.get(opts, :strategy_weights, @default_opts.strategy_weights),
      stats: init_stats()
    }

    Logger.debug("[Matchmaker] Started with k_factor=#{state.k_factor}")

    {:ok, state}
  end

  @impl true
  def handle_call({:request_match, game_id, opts}, _from, state) do
    strategy = Keyword.get(opts, :strategy, sample_strategy(state))
    player_id = Keyword.get(opts, :player_id, :current)

    # Ensure player is registered
    state = ensure_registered(state, player_id)

    # Find opponent based on strategy
    {opponent_id, new_state} = find_opponent(state, player_id, strategy)

    # Create match record
    match_id = "m_#{System.unique_integer([:positive, :monotonic])}"
    p1_rating = get_policy_rating(new_state, player_id)
    p2_rating = get_policy_rating(new_state, opponent_id)

    match = %{
      match_id: match_id,
      game_id: game_id,
      p1: player_id,
      p2: opponent_id,
      p1_rating: p1_rating,
      p2_rating: p2_rating,
      strategy: strategy,
      created_at: System.system_time(:second)
    }

    # Track pending match
    new_state = %{new_state |
      pending_matches: Map.put(new_state.pending_matches, match_id, match)
    }

    {:reply, {:ok, match}, new_state}
  end

  @impl true
  def handle_call({:report_result, p1_id, p2_id, result}, _from, state) do
    state = ensure_registered(state, p1_id)
    state = ensure_registered(state, p2_id)

    {new_state, rating_changes} = update_ratings(state, p1_id, p2_id, result)

    # Record match
    match_record = %{
      p1: p1_id,
      p2: p2_id,
      result: result,
      rating_changes: rating_changes,
      timestamp: System.system_time(:second)
    }

    new_state = %{new_state |
      completed_matches: [match_record | Enum.take(new_state.completed_matches, 999)],
      stats: update_stats(new_state.stats, result)
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:report_result_with_stats, p1_id, p2_id, result, game_stats}, _from, state) do
    state = ensure_registered(state, p1_id)
    state = ensure_registered(state, p2_id)

    {new_state, rating_changes} = update_ratings(state, p1_id, p2_id, result)

    match_record = %{
      p1: p1_id,
      p2: p2_id,
      result: result,
      rating_changes: rating_changes,
      game_stats: game_stats,
      timestamp: System.system_time(:second)
    }

    new_state = %{new_state |
      completed_matches: [match_record | Enum.take(new_state.completed_matches, 999)],
      stats: update_stats(new_state.stats, result)
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:get_rating, policy_id}, _from, state) do
    case Map.get(state.ratings, normalize_id(policy_id)) do
      nil -> {:reply, {:error, :not_found}, state}
      rating_info -> {:reply, {:ok, rating_info}, state}
    end
  end

  @impl true
  def handle_call(:get_ratings, _from, state) do
    {:reply, state.ratings, state}
  end

  @impl true
  def handle_call({:get_leaderboard, limit}, _from, state) do
    leaderboard = state.ratings
    |> Enum.map(fn {id, info} -> Map.put(info, :id, id) end)
    |> Enum.sort_by(& &1.rating, :desc)
    |> Enum.take(limit)

    {:reply, leaderboard, state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = Map.merge(state.stats, %{
      num_policies: map_size(state.ratings),
      pending_matches: map_size(state.pending_matches),
      completed_matches: length(state.completed_matches)
    })
    {:reply, stats, state}
  end

  @impl true
  def handle_call({:register_policy, policy_id, initial_rating}, _from, state) do
    id = normalize_id(policy_id)
    rating = initial_rating || Elo.initial_rating()

    if Map.has_key?(state.ratings, id) do
      {:reply, {:error, :already_registered}, state}
    else
      rating_info = %{
        rating: rating,
        wins: 0,
        losses: 0,
        draws: 0,
        games_played: 0,
        registered_at: System.system_time(:second)
      }

      new_state = %{state | ratings: Map.put(state.ratings, id, rating_info)}
      {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:get_win_rate, p1_id, p2_id}, _from, state) do
    p1_key = normalize_id(p1_id)
    p2_key = normalize_id(p2_id)

    matches = state.completed_matches
    |> Enum.filter(fn m ->
      (m.p1 == p1_key and m.p2 == p2_key) or (m.p1 == p2_key and m.p2 == p1_key)
    end)

    if length(matches) == 0 do
      {:reply, {:ok, 0.5}, state}  # Unknown, assume 50%
    else
      wins = Enum.count(matches, fn m ->
        (m.p1 == p1_key and m.result == :win) or (m.p2 == p1_key and m.result == :loss)
      end)

      win_rate = wins / length(matches)
      {:reply, {:ok, win_rate}, state}
    end
  end

  @impl true
  def handle_call({:get_match_history, policy_id, limit}, _from, state) do
    id = normalize_id(policy_id)

    history = state.completed_matches
    |> Enum.filter(fn m -> m.p1 == id or m.p2 == id end)
    |> Enum.take(limit)

    {:reply, history, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    new_state = %{state |
      ratings: %{},
      pending_matches: %{},
      completed_matches: [],
      stats: init_stats()
    }
    {:reply, :ok, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp normalize_id(id) when is_atom(id), do: Atom.to_string(id)
  defp normalize_id(id) when is_binary(id), do: id
  defp normalize_id({type, val}), do: "#{type}_#{val}"
  defp normalize_id(id), do: inspect(id)

  defp ensure_registered(state, policy_id) do
    id = normalize_id(policy_id)

    if Map.has_key?(state.ratings, id) do
      state
    else
      rating_info = %{
        rating: Elo.initial_rating(),
        wins: 0,
        losses: 0,
        draws: 0,
        games_played: 0,
        registered_at: System.system_time(:second)
      }

      %{state | ratings: Map.put(state.ratings, id, rating_info)}
    end
  end

  defp get_policy_rating(state, policy_id) do
    id = normalize_id(policy_id)
    case Map.get(state.ratings, id) do
      nil -> Elo.initial_rating()
      info -> info.rating
    end
  end

  defp sample_strategy(state) do
    weights = state.strategy_weights
    total = Enum.sum(Map.values(weights))

    if total == 0 do
      :self_play
    else
      r = :rand.uniform() * total

      weights
      |> Enum.reduce_while({0.0, :self_play}, fn {strategy, weight}, {cumsum, _} ->
        new_cumsum = cumsum + weight
        if r <= new_cumsum do
          {:halt, {new_cumsum, strategy}}
        else
          {:cont, {new_cumsum, strategy}}
        end
      end)
      |> elem(1)
    end
  end

  defp find_opponent(state, player_id, :self_play) do
    # Self-play: opponent is the same policy
    {player_id, state}
  end

  defp find_opponent(state, player_id, :historical) do
    # Find historical versions (not current)
    player_key = normalize_id(player_id)

    historical = state.ratings
    |> Map.keys()
    |> Enum.filter(& &1 != player_key and String.starts_with?(&1, "v"))

    if length(historical) > 0 do
      {Enum.random(historical), state}
    else
      # No historical, fall back to self-play
      {player_id, state}
    end
  end

  defp find_opponent(state, player_id, :skill_based) do
    player_rating = get_policy_rating(state, player_id)
    player_key = normalize_id(player_id)

    # Find policies within Elo range
    candidates = state.ratings
    |> Enum.filter(fn {id, info} ->
      id != player_key and abs(info.rating - player_rating) <= state.elo_range
    end)

    if length(candidates) > 0 do
      {id, _info} = Enum.random(candidates)
      {id, state}
    else
      # No one in range, fall back to closest
      find_closest_opponent(state, player_id, player_rating)
    end
  end

  defp find_opponent(state, player_id, :exploiter) do
    # PFSP: prioritize opponents we have low win rate against
    player_key = normalize_id(player_id)

    # Get win rates against all opponents
    opponents_with_rates = state.ratings
    |> Map.keys()
    |> Enum.filter(& &1 != player_key)
    |> Enum.map(fn opp_id ->
      matches = state.completed_matches
      |> Enum.filter(fn m ->
        (m.p1 == player_key and m.p2 == opp_id) or
        (m.p2 == player_key and m.p1 == opp_id)
      end)

      win_rate = if length(matches) > 0 do
        wins = Enum.count(matches, fn m ->
          (m.p1 == player_key and m.result == :win) or
          (m.p2 == player_key and m.result == :loss)
        end)
        wins / length(matches)
      else
        0.5  # Unknown
      end

      {opp_id, win_rate}
    end)

    if length(opponents_with_rates) > 0 do
      # Weight by inverse win rate (prioritize hard opponents)
      weighted = Enum.map(opponents_with_rates, fn {id, rate} ->
        {id, 1.0 - rate + 0.1}  # +0.1 to avoid zero weight
      end)

      total = Enum.sum(Enum.map(weighted, &elem(&1, 1)))
      r = :rand.uniform() * total

      {selected, _} = Enum.reduce_while(weighted, {nil, 0.0}, fn {id, w}, {_, cumsum} ->
        new_cumsum = cumsum + w
        if r <= new_cumsum do
          {:halt, {id, new_cumsum}}
        else
          {:cont, {id, new_cumsum}}
        end
      end)

      {selected || player_id, state}
    else
      {player_id, state}
    end
  end

  defp find_opponent(state, player_id, :random) do
    player_key = normalize_id(player_id)

    candidates = state.ratings
    |> Map.keys()
    |> Enum.filter(& &1 != player_key)

    if length(candidates) > 0 do
      {Enum.random(candidates), state}
    else
      {player_id, state}
    end
  end

  defp find_closest_opponent(state, player_id, player_rating) do
    player_key = normalize_id(player_id)

    closest = state.ratings
    |> Enum.filter(fn {id, _} -> id != player_key end)
    |> Enum.min_by(fn {_id, info} -> abs(info.rating - player_rating) end, fn -> nil end)

    case closest do
      {id, _info} -> {id, state}
      nil -> {player_id, state}
    end
  end

  defp update_ratings(state, p1_id, p2_id, result) do
    p1_key = normalize_id(p1_id)
    p2_key = normalize_id(p2_id)

    p1_info = Map.get(state.ratings, p1_key)
    p2_info = Map.get(state.ratings, p2_key)

    # Calculate K-factors based on games played
    k1 = Elo.dynamic_k_factor(p1_info.games_played)
    k2 = Elo.dynamic_k_factor(p2_info.games_played)

    # Update ratings
    {new_p1_rating, new_p2_rating} = Elo.update(
      p1_info.rating,
      p2_info.rating,
      result,
      k_factor_a: k1,
      k_factor_b: k2
    )

    # Update record counts
    {p1_wins, p1_losses, p1_draws} = update_record(p1_info, result, :p1)
    {p2_wins, p2_losses, p2_draws} = update_record(p2_info, result, :p2)

    new_p1_info = %{p1_info |
      rating: new_p1_rating,
      wins: p1_wins,
      losses: p1_losses,
      draws: p1_draws,
      games_played: p1_info.games_played + 1
    }

    new_p2_info = %{p2_info |
      rating: new_p2_rating,
      wins: p2_wins,
      losses: p2_losses,
      draws: p2_draws,
      games_played: p2_info.games_played + 1
    }

    new_state = %{state |
      ratings: state.ratings
      |> Map.put(p1_key, new_p1_info)
      |> Map.put(p2_key, new_p2_info)
    }

    rating_changes = %{
      p1: new_p1_rating - p1_info.rating,
      p2: new_p2_rating - p2_info.rating
    }

    {new_state, rating_changes}
  end

  defp update_record(info, result, :p1) do
    case result do
      :win -> {info.wins + 1, info.losses, info.draws}
      :loss -> {info.wins, info.losses + 1, info.draws}
      :draw -> {info.wins, info.losses, info.draws + 1}
    end
  end

  defp update_record(info, result, :p2) do
    case result do
      :win -> {info.wins, info.losses + 1, info.draws}  # P1 won, P2 lost
      :loss -> {info.wins + 1, info.losses, info.draws}  # P1 lost, P2 won
      :draw -> {info.wins, info.losses, info.draws + 1}
    end
  end

  defp init_stats do
    %{
      total_matches: 0,
      p1_wins: 0,
      p2_wins: 0,
      draws: 0
    }
  end

  defp update_stats(stats, result) do
    case result do
      :win -> %{stats | total_matches: stats.total_matches + 1, p1_wins: stats.p1_wins + 1}
      :loss -> %{stats | total_matches: stats.total_matches + 1, p2_wins: stats.p2_wins + 1}
      :draw -> %{stats | total_matches: stats.total_matches + 1, draws: stats.draws + 1}
    end
  end
end
