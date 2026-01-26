defmodule ExPhil.League.MatchScheduler do
  @moduledoc """
  Tournament scheduling for architecture league competitions.

  Provides various scheduling strategies for organizing matches between
  architectures, from simple round-robin to skill-based matchmaking.

  ## Scheduling Strategies

  - **Round-robin**: Every architecture plays every other architecture
  - **Swiss-style**: Pair architectures with similar records
  - **Skill-based**: Match by Elo rating proximity
  - **Bracket**: Elimination tournament style

  ## Usage

      # Generate round-robin schedule
      schedule = MatchScheduler.round_robin([:mamba, :lstm, :attention])
      # => [{:mamba, :lstm}, {:mamba, :attention}, {:lstm, :attention}]

      # Generate skill-based matchups
      matchups = MatchScheduler.skill_based(architectures, num_matches: 10)

      # Generate Swiss-style rounds
      rounds = MatchScheduler.swiss_rounds(architectures, num_rounds: 5)

  """

  alias ExPhil.League.ArchitectureEntry

  require Logger

  @type arch_id :: atom()
  @type matchup :: {arch_id(), arch_id()}
  @type schedule :: [matchup()]

  # ============================================================================
  # Scheduling Strategies
  # ============================================================================

  @doc """
  Generate a round-robin schedule where every architecture plays every other.

  ## Options

  - `:matches_per_pair` - Number of matches per pair (default: 1)
  - `:shuffle` - Randomize match order (default: true)

  ## Example

      MatchScheduler.round_robin([:a, :b, :c], matches_per_pair: 2)
      # => [{:a, :b}, {:a, :b}, {:a, :c}, {:a, :c}, {:b, :c}, {:b, :c}]

  """
  @spec round_robin([arch_id()], keyword()) :: schedule()
  def round_robin(arch_ids, opts \\ []) do
    matches_per_pair = Keyword.get(opts, :matches_per_pair, 1)
    shuffle = Keyword.get(opts, :shuffle, true)

    # Generate all pairs
    pairs = for p1 <- arch_ids, p2 <- arch_ids, p1 < p2, do: {p1, p2}

    # Expand by matches_per_pair
    schedule = List.duplicate(pairs, matches_per_pair) |> List.flatten()

    if shuffle do
      Enum.shuffle(schedule)
    else
      schedule
    end
  end

  @doc """
  Generate skill-based matchups pairing architectures with similar Elo ratings.

  ## Parameters

  - `architectures` - List of ArchitectureEntry structs
  - `opts` - Options

  ## Options

  - `:num_matches` - Total number of matches to generate (default: 10)
  - `:elo_range` - Maximum Elo difference for pairing (default: 200)
  - `:allow_rematches` - Allow same pair multiple times (default: true)

  """
  @spec skill_based([ArchitectureEntry.t()], keyword()) :: schedule()
  def skill_based(architectures, opts \\ []) when is_list(architectures) do
    num_matches = Keyword.get(opts, :num_matches, 10)
    elo_range = Keyword.get(opts, :elo_range, 200)
    allow_rematches = Keyword.get(opts, :allow_rematches, true)

    # Sort by Elo
    sorted = Enum.sort_by(architectures, & &1.elo)
    arch_ids = Enum.map(sorted, & &1.id)

    # Build Elo lookup
    elo_map = Map.new(architectures, &{&1.id, &1.elo})

    generate_skill_matches(arch_ids, elo_map, elo_range, num_matches, allow_rematches, [])
  end

  @doc """
  Generate Swiss-style tournament rounds.

  In Swiss-style:
  - Each round pairs architectures with similar win records
  - No rematches within the tournament
  - Continues for specified number of rounds

  ## Parameters

  - `architectures` - List of ArchitectureEntry structs
  - `opts` - Options

  ## Options

  - `:num_rounds` - Number of rounds (default: calculated from num architectures)

  ## Returns

  List of rounds, each containing matchups for that round.

  """
  @spec swiss_rounds([ArchitectureEntry.t()], keyword()) :: [[matchup()]]
  def swiss_rounds(architectures, opts \\ []) when is_list(architectures) do
    n = length(architectures)
    default_rounds = max(1, ceil(:math.log2(n)))
    num_rounds = Keyword.get(opts, :num_rounds, default_rounds)

    # Initial standings (all 0-0)
    standings = Map.new(architectures, &{&1.id, {0, 0}})

    generate_swiss_rounds(architectures, standings, num_rounds, [], MapSet.new())
  end

  @doc """
  Generate single-elimination bracket.

  ## Parameters

  - `arch_ids` - List of architecture IDs (should be power of 2)
  - `opts` - Options

  ## Options

  - `:shuffle_seeds` - Randomize seeding (default: false)

  ## Returns

  List of rounds, where each round has half the matchups of the previous.

  """
  @spec bracket([arch_id()], keyword()) :: [[matchup()]]
  def bracket(arch_ids, opts \\ []) do
    shuffle_seeds = Keyword.get(opts, :shuffle_seeds, false)

    # Pad to power of 2 if needed
    n = length(arch_ids)
    target_size = next_power_of_2(n)

    # Add byes if needed
    padded =
      if n < target_size do
        arch_ids ++ List.duplicate(:bye, target_size - n)
      else
        arch_ids
      end

    # Optionally shuffle
    seeded = if shuffle_seeds, do: Enum.shuffle(padded), else: padded

    # Generate first round
    first_round = pair_bracket(seeded)

    # For subsequent rounds, we'd need results - return just first round structure
    [first_round]
  end

  @doc """
  Generate a prioritized fictitious self-play (PFSP) schedule.

  PFSP prioritizes matches against opponents the architecture struggles against,
  promoting faster learning on weaknesses.

  ## Parameters

  - `arch_id` - Architecture to generate matches for
  - `candidates` - List of potential opponents with win rate data
  - `opts` - Options

  ## Options

  - `:num_matches` - Number of matches to generate (default: 10)
  - `:exploit_factor` - Higher = more focus on weak matchups (default: 0.8)

  """
  @spec pfsp(arch_id(), [%{id: arch_id(), win_rate: float()}], keyword()) :: [arch_id()]
  def pfsp(arch_id, candidates, opts \\ []) do
    num_matches = Keyword.get(opts, :num_matches, 10)
    exploit_factor = Keyword.get(opts, :exploit_factor, 0.8)

    # Filter out self
    opponents = Enum.reject(candidates, &(&1.id == arch_id))

    if length(opponents) == 0 do
      []
    else
      # Weight by inverse win rate (lower win rate = higher priority)
      weighted =
        Enum.map(opponents, fn opp ->
          # Win rate against this opponent (from our perspective)
          weight = (1.0 - opp.win_rate) * exploit_factor + (1 - exploit_factor) * 0.5
          {opp.id, max(weight, 0.01)}
        end)

      # Sample with replacement based on weights
      sample_weighted(weighted, num_matches)
    end
  end

  @doc """
  Generate a diverse schedule mixing multiple strategies.

  ## Parameters

  - `architectures` - List of ArchitectureEntry structs
  - `opts` - Options

  ## Options

  - `:num_matches` - Total matches to generate (default: 20)
  - `:strategy_weights` - Weight for each strategy (default: balanced)

  ## Strategy Weights

  - `:round_robin` - Standard round-robin matches
  - `:skill_based` - Elo-proximity matchups
  - `:random` - Random pairings
  - `:exploiter` - PFSP-style targeted matchups

  """
  @spec diverse([ArchitectureEntry.t()], keyword()) :: schedule()
  def diverse(architectures, opts \\ []) do
    num_matches = Keyword.get(opts, :num_matches, 20)

    weights =
      Keyword.get(opts, :strategy_weights, %{
        round_robin: 0.3,
        skill_based: 0.3,
        random: 0.2,
        exploiter: 0.2
      })

    arch_ids = Enum.map(architectures, & &1.id)

    # Generate matches using each strategy
    schedule =
      Enum.flat_map(1..num_matches, fn _ ->
        strategy = sample_strategy(weights)
        generate_match(strategy, architectures, arch_ids)
      end)

    Enum.take(schedule, num_matches)
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  @doc """
  Estimate number of matches needed for statistically significant Elo differences.

  Uses binomial confidence intervals to estimate when rankings are reliable.
  """
  @spec matches_for_confidence(non_neg_integer(), float()) :: non_neg_integer()
  def matches_for_confidence(num_architectures, confidence \\ 0.95) do
    # Rule of thumb: ~30 games per pair for reasonable confidence
    # Adjusted by number of architectures
    base_matches = 30
    pairs = div(num_architectures * (num_architectures - 1), 2)

    ceil(base_matches * pairs * (1 + (1 - confidence)))
  end

  @doc """
  Calculate expected duration of a tournament.

  ## Parameters

  - `schedule` - List of matchups
  - `avg_match_frames` - Average frames per match (default: 4500 ~= 75 seconds)
  - `parallel_games` - Number of games running in parallel (default: 1)

  ## Returns

  Estimated duration in seconds.
  """
  @spec estimated_duration(schedule(), keyword()) :: float()
  def estimated_duration(schedule, opts \\ []) do
    avg_frames = Keyword.get(opts, :avg_match_frames, 4500)
    parallel = Keyword.get(opts, :parallel_games, 1)
    fps = 60

    total_matches = length(schedule)
    matches_per_batch = parallel
    batches = ceil(total_matches / matches_per_batch)

    batches * (avg_frames / fps)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp generate_skill_matches(_arch_ids, _elo_map, _range, 0, _allow, acc), do: Enum.reverse(acc)

  defp generate_skill_matches(arch_ids, elo_map, elo_range, remaining, allow_rematches, acc) do
    # Pick a random architecture
    p1 = Enum.random(arch_ids)
    p1_elo = Map.get(elo_map, p1, 1000)

    # Find opponents within Elo range
    candidates =
      arch_ids
      |> Enum.filter(fn id ->
        id != p1 and abs(Map.get(elo_map, id, 1000) - p1_elo) <= elo_range
      end)

    if length(candidates) == 0 do
      # No one in range, pick random
      p2 = Enum.random(Enum.reject(arch_ids, &(&1 == p1)))
      match = if p1 < p2, do: {p1, p2}, else: {p2, p1}

      if allow_rematches or match not in acc do
        generate_skill_matches(arch_ids, elo_map, elo_range, remaining - 1, allow_rematches, [
          match | acc
        ])
      else
        generate_skill_matches(arch_ids, elo_map, elo_range, remaining, allow_rematches, acc)
      end
    else
      p2 = Enum.random(candidates)
      match = if p1 < p2, do: {p1, p2}, else: {p2, p1}

      if allow_rematches or match not in acc do
        generate_skill_matches(arch_ids, elo_map, elo_range, remaining - 1, allow_rematches, [
          match | acc
        ])
      else
        generate_skill_matches(arch_ids, elo_map, elo_range, remaining, allow_rematches, acc)
      end
    end
  end

  defp generate_swiss_rounds(_archs, _standings, 0, rounds, _played), do: Enum.reverse(rounds)

  defp generate_swiss_rounds(architectures, standings, remaining, rounds, played) do
    # Sort by record (wins - losses), then by Elo for tiebreaker
    sorted =
      Enum.sort_by(architectures, fn arch ->
        {wins, losses} = Map.get(standings, arch.id, {0, 0})
        {-(wins - losses), -arch.elo}
      end)

    # Pair adjacent entries
    {round_matchups, new_played} = pair_swiss(sorted, played, [])

    # Update standings (simulate 50/50 for now since we don't have results)
    new_standings = update_standings_simulated(standings, round_matchups)

    generate_swiss_rounds(
      architectures,
      new_standings,
      remaining - 1,
      [round_matchups | rounds],
      new_played
    )
  end

  defp pair_swiss([], _played, acc), do: {Enum.reverse(acc), MapSet.new()}

  defp pair_swiss([_single], played, acc) do
    # Odd number, one gets a bye
    {Enum.reverse(acc), played}
  end

  defp pair_swiss([a, b | rest], played, acc) do
    match = if a.id < b.id, do: {a.id, b.id}, else: {b.id, a.id}

    if MapSet.member?(played, match) do
      # Already played, try to find alternative
      case find_alternative(a, rest, played) do
        {:ok, alt, remaining} ->
          new_match = if a.id < alt.id, do: {a.id, alt.id}, else: {alt.id, a.id}
          pair_swiss([b | remaining], MapSet.put(played, new_match), [new_match | acc])

        :none ->
          # Can't avoid rematch
          pair_swiss(rest, MapSet.put(played, match), [match | acc])
      end
    else
      pair_swiss(rest, MapSet.put(played, match), [match | acc])
    end
  end

  defp find_alternative(_arch, [], _played), do: :none

  defp find_alternative(arch, [candidate | rest], played) do
    match = if arch.id < candidate.id, do: {arch.id, candidate.id}, else: {candidate.id, arch.id}

    if MapSet.member?(played, match) do
      find_alternative(arch, rest, played)
    else
      {:ok, candidate, rest}
    end
  end

  defp update_standings_simulated(standings, matchups) do
    # Simulate 50/50 results for planning purposes
    Enum.reduce(matchups, standings, fn {p1, p2}, acc ->
      {w1, l1} = Map.get(acc, p1, {0, 0})
      {w2, l2} = Map.get(acc, p2, {0, 0})

      # Randomly assign winner for simulation
      if :rand.uniform() > 0.5 do
        acc
        |> Map.put(p1, {w1 + 1, l1})
        |> Map.put(p2, {w2, l2 + 1})
      else
        acc
        |> Map.put(p1, {w1, l1 + 1})
        |> Map.put(p2, {w2 + 1, l2})
      end
    end)
  end

  defp pair_bracket(list) do
    list
    |> Enum.chunk_every(2)
    |> Enum.map(fn
      [a, b] when a != :bye and b != :bye -> {a, b}
      [a, :bye] -> {a, :bye}
      [:bye, b] -> {:bye, b}
      [a] -> {a, :bye}
    end)
  end

  defp next_power_of_2(n) do
    :math.pow(2, :math.ceil(:math.log2(max(n, 1)))) |> round()
  end

  defp sample_weighted(weighted, n) do
    total = Enum.sum(Enum.map(weighted, &elem(&1, 1)))

    Enum.map(1..n, fn _ ->
      r = :rand.uniform() * total

      Enum.reduce_while(weighted, {nil, 0.0}, fn {id, w}, {_, cumsum} ->
        new_cumsum = cumsum + w

        if r <= new_cumsum do
          {:halt, {id, new_cumsum}}
        else
          {:cont, {id, new_cumsum}}
        end
      end)
      |> elem(0)
    end)
  end

  defp sample_strategy(weights) do
    total = Enum.sum(Map.values(weights))
    r = :rand.uniform() * total

    weights
    |> Enum.reduce_while({nil, 0.0}, fn {strategy, w}, {_, cumsum} ->
      new_cumsum = cumsum + w

      if r <= new_cumsum do
        {:halt, {strategy, new_cumsum}}
      else
        {:cont, {strategy, new_cumsum}}
      end
    end)
    |> elem(0)
  end

  defp generate_match(:round_robin, _architectures, arch_ids) do
    # Random pair
    [p1, p2 | _] = Enum.shuffle(arch_ids)
    if p1 < p2, do: [{p1, p2}], else: [{p2, p1}]
  end

  defp generate_match(:skill_based, architectures, _arch_ids) do
    skill_based(architectures, num_matches: 1)
  end

  defp generate_match(:random, _architectures, arch_ids) do
    [p1, p2 | _] = Enum.shuffle(arch_ids)
    if p1 < p2, do: [{p1, p2}], else: [{p2, p1}]
  end

  defp generate_match(:exploiter, architectures, arch_ids) do
    # Pick random architecture and generate PFSP match
    arch = Enum.random(arch_ids)

    candidates =
      Enum.map(architectures, fn a ->
        %{id: a.id, win_rate: ArchitectureEntry.win_rate(a)}
      end)

    case pfsp(arch, candidates, num_matches: 1) do
      [opponent] ->
        if arch < opponent, do: [{arch, opponent}], else: [{opponent, arch}]

      [] ->
        # Fallback to random
        [p1, p2 | _] = Enum.shuffle(arch_ids)
        if p1 < p2, do: [{p1, p2}], else: [{p2, p1}]
    end
  end
end
