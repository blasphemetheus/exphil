defmodule ExPhil.Training.ReplayQuality do
  @moduledoc """
  Replay quality scoring for training data curation.

  Scores replays on a 0-100 scale based on multiple quality factors:
  - Game length (balanced matches)
  - Damage dealt (both players engaged)
  - Input activity (no AFK players)
  - SD rate (low self-destruct rate)
  - Action diversity (varied gameplay)

  ## Usage

      # Score a parsed replay
      score = ReplayQuality.score(replay_data)

      # Check if replay passes quality threshold
      if ReplayQuality.passes?(replay_data, min_score: 60) do
        # Use for training
      end

      # Get detailed breakdown
      details = ReplayQuality.analyze(replay_data)

  ## Quality Thresholds

  | Score | Quality | Recommendation |
  |-------|---------|----------------|
  | 80+   | Excellent | High-priority training data |
  | 60-79 | Good | Standard training data |
  | 40-59 | Fair | Use with lower weight |
  | <40   | Poor | Consider rejecting |

  ## Hard Filters (Auto-Reject)

  Some replays are rejected regardless of score:
  - Duration < 1000 frames (~17 seconds)
  - Duration > 15000 frames (> 4 minutes, likely timeout)
  - CPU players detected
  - Both players dealt 0 damage
  """

  alias ExPhil.Training.Output

  # Frame counts
  @min_frames 1000        # ~17 seconds, absolute minimum
  @ideal_min_frames 3000  # ~50 seconds, ideal range start
  @ideal_max_frames 8000  # ~133 seconds, ideal range end
  @max_frames 15000       # ~4 minutes, timeout threshold

  # Quality thresholds
  @excellent_score 80
  @good_score 60
  @fair_score 40

  # Scoring weights (total = 100)
  @length_weight 25
  @damage_weight 20
  @activity_weight 20
  @sd_weight 15
  @diversity_weight 20

  # Input activity thresholds
  @min_input_activity 0.10    # At least 10% of frames have inputs
  @good_input_activity 0.25   # 25% input activity is healthy

  # SD rate thresholds
  @low_sd_rate 0.05           # < 5% is normal
  @high_sd_rate 0.15          # > 15% is concerning

  # Minimum damage for engagement
  @min_damage 20
  @good_damage 100

  @doc """
  Score a replay's quality on a 0-100 scale.

  ## Input Format

  Accepts a map with the following structure:
  ```
  %{
    frames: integer(),           # Total frame count
    players: [
      %{
        damage_dealt: float(),   # Total damage dealt
        damage_taken: float(),   # Total damage taken
        stocks_lost: integer(),  # How many stocks lost
        sd_count: integer(),     # Self-destruct count
        input_frames: integer(), # Frames with non-neutral input
        unique_actions: integer(), # Distinct action states used
        is_cpu: boolean()        # True if CPU player
      },
      ...
    ]
  }
  ```

  Returns score from 0-100, or `:rejected` for hard-fail conditions.
  """
  @spec score(map()) :: non_neg_integer() | :rejected
  def score(replay) do
    case hard_filter(replay) do
      :pass -> compute_score(replay)
      :reject -> :rejected
    end
  end

  @doc """
  Check if replay passes quality threshold.

  ## Options

    * `:min_score` - Minimum acceptable score (default: 60)

  """
  @spec passes?(map(), keyword()) :: boolean()
  def passes?(replay, opts \\ []) do
    min_score = Keyword.get(opts, :min_score, @good_score)

    case score(replay) do
      :rejected -> false
      s when s >= min_score -> true
      _ -> false
    end
  end

  @doc """
  Analyze a replay and return detailed quality breakdown.

  Returns a map with individual component scores and explanations.
  """
  @spec analyze(map()) :: map()
  def analyze(replay) do
    case hard_filter(replay) do
      :reject ->
        %{
          status: :rejected,
          reason: rejection_reason(replay),
          score: 0
        }

      :pass ->
        length_score = score_length(replay)
        damage_score = score_damage(replay)
        activity_score = score_activity(replay)
        sd_score = score_sd_rate(replay)
        diversity_score = score_diversity(replay)

        total = length_score + damage_score + activity_score + sd_score + diversity_score

        %{
          status: :passed,
          score: total,
          quality: quality_label(total),
          breakdown: %{
            length: %{score: length_score, max: @length_weight, frames: replay[:frames]},
            damage: %{score: damage_score, max: @damage_weight, details: damage_details(replay)},
            activity: %{score: activity_score, max: @activity_weight, details: activity_details(replay)},
            sd_rate: %{score: sd_score, max: @sd_weight, details: sd_details(replay)},
            diversity: %{score: diversity_score, max: @diversity_weight, details: diversity_details(replay)}
          }
        }
    end
  end

  @doc """
  Print a quality analysis summary.
  """
  @spec print_analysis(map()) :: :ok
  def print_analysis(replay) do
    analysis = analyze(replay)

    case analysis.status do
      :rejected ->
        Output.puts_raw("")
        Output.puts_raw("  " <> Output.colorize("Replay Quality: REJECTED", :red))
        Output.puts_raw("  Reason: #{analysis.reason}")

      :passed ->
        color = score_color(analysis.score)
        Output.puts_raw("")
        Output.puts_raw("  " <> Output.colorize("Replay Quality: #{analysis.score}/100 (#{analysis.quality})", color))
        Output.puts_raw("")

        # Breakdown
        for {name, data} <- analysis.breakdown do
          pct = round(data.score / data.max * 100)
          bar_width = 20
          filled = round(pct / 100 * bar_width)
          bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)
          name_str = name |> to_string() |> String.capitalize() |> String.pad_trailing(10)
          Output.puts_raw("    #{name_str} #{Output.colorize(bar, score_color(pct))} #{data.score}/#{data.max}")
        end
    end

    :ok
  end

  @doc """
  Score multiple replays and return statistics.
  """
  @spec batch_score([map()]) :: map()
  def batch_score(replays) do
    results = Enum.map(replays, &score/1)

    passed = Enum.reject(results, &(&1 == :rejected))
    rejected = Enum.count(results, &(&1 == :rejected))

    %{
      total: length(replays),
      passed: length(passed),
      rejected: rejected,
      mean_score: if(passed != [], do: Enum.sum(passed) / length(passed), else: 0),
      min_score: if(passed != [], do: Enum.min(passed), else: 0),
      max_score: if(passed != [], do: Enum.max(passed), else: 0),
      excellent_count: Enum.count(passed, &(&1 >= @excellent_score)),
      good_count: Enum.count(passed, &(&1 >= @good_score and &1 < @excellent_score)),
      fair_count: Enum.count(passed, &(&1 >= @fair_score and &1 < @good_score)),
      poor_count: Enum.count(passed, &(&1 < @fair_score))
    }
  end

  # ============================================================
  # Hard Filters
  # ============================================================

  defp hard_filter(replay) do
    cond do
      # Duration checks
      (replay[:frames] || 0) < @min_frames -> :reject
      (replay[:frames] || 0) > @max_frames -> :reject

      # CPU check
      has_cpu?(replay) -> :reject

      # Zero engagement check
      zero_engagement?(replay) -> :reject

      true -> :pass
    end
  end

  defp has_cpu?(replay) do
    players = replay[:players] || []
    Enum.any?(players, fn p -> p[:is_cpu] == true end)
  end

  defp zero_engagement?(replay) do
    players = replay[:players] || []

    Enum.all?(players, fn p ->
      (p[:damage_dealt] || 0) == 0 and (p[:damage_taken] || 0) == 0
    end)
  end

  defp rejection_reason(replay) do
    frames = replay[:frames] || 0

    cond do
      frames < @min_frames -> "Too short (#{frames} frames, min: #{@min_frames})"
      frames > @max_frames -> "Too long (#{frames} frames, max: #{@max_frames})"
      has_cpu?(replay) -> "CPU player detected"
      zero_engagement?(replay) -> "No damage dealt by either player"
      true -> "Unknown"
    end
  end

  # ============================================================
  # Scoring Components
  # ============================================================

  defp compute_score(replay) do
    length_score = score_length(replay)
    damage_score = score_damage(replay)
    activity_score = score_activity(replay)
    sd_score = score_sd_rate(replay)
    diversity_score = score_diversity(replay)

    length_score + damage_score + activity_score + sd_score + diversity_score
  end

  # Game length scoring (25 points)
  defp score_length(replay) do
    frames = replay[:frames] || 0

    cond do
      frames >= @ideal_min_frames and frames <= @ideal_max_frames ->
        @length_weight  # Full points for ideal range

      frames >= @min_frames and frames < @ideal_min_frames ->
        # Below ideal - partial points
        round(@length_weight * 0.6)

      frames > @ideal_max_frames and frames <= @max_frames ->
        # Above ideal - slight penalty
        round(@length_weight * 0.8)

      true ->
        0
    end
  end

  # Damage scoring (20 points)
  defp score_damage(replay) do
    players = replay[:players] || []

    damages = Enum.map(players, fn p ->
      max(p[:damage_dealt] || 0, p[:damage_taken] || 0)
    end)

    min_damage = Enum.min(damages, fn -> 0 end)

    cond do
      min_damage >= @good_damage -> @damage_weight
      min_damage >= @min_damage -> round(@damage_weight * 0.5)
      min_damage > 0 -> round(@damage_weight * 0.2)
      true -> 0
    end
  end

  # Input activity scoring (20 points)
  defp score_activity(replay) do
    players = replay[:players] || []
    frames = max(replay[:frames] || 1, 1)

    activities = Enum.map(players, fn p ->
      input_frames = p[:input_frames] || 0
      input_frames / frames
    end)

    min_activity = Enum.min(activities, fn -> 0 end)

    cond do
      min_activity >= @good_input_activity -> @activity_weight
      min_activity >= @min_input_activity -> round(@activity_weight * 0.6)
      min_activity > 0 -> round(@activity_weight * 0.2)
      true -> 0
    end
  end

  # SD rate scoring (15 points - lower is better)
  defp score_sd_rate(replay) do
    players = replay[:players] || []

    sd_rates = Enum.map(players, fn p ->
      stocks_lost = max(p[:stocks_lost] || 0, 1)
      sd_count = p[:sd_count] || 0
      sd_count / stocks_lost
    end)

    max_sd_rate = Enum.max(sd_rates, fn -> 0 end)

    cond do
      max_sd_rate <= @low_sd_rate -> @sd_weight
      max_sd_rate <= @high_sd_rate -> round(@sd_weight * 0.5)
      true -> 0
    end
  end

  # Action diversity scoring (20 points)
  defp score_diversity(replay) do
    players = replay[:players] || []

    diversities = Enum.map(players, fn p ->
      p[:unique_actions] || 0
    end)

    min_diversity = Enum.min(diversities, fn -> 0 end)

    cond do
      min_diversity >= 50 -> @diversity_weight
      min_diversity >= 30 -> round(@diversity_weight * 0.7)
      min_diversity >= 15 -> round(@diversity_weight * 0.4)
      true -> 0
    end
  end

  # ============================================================
  # Details Helpers
  # ============================================================

  defp damage_details(replay) do
    players = replay[:players] || []

    players
    |> Enum.with_index()
    |> Enum.map(fn {p, i} ->
      {"P#{i + 1}", "dealt: #{round(p[:damage_dealt] || 0)}, taken: #{round(p[:damage_taken] || 0)}"}
    end)
    |> Map.new()
  end

  defp activity_details(replay) do
    players = replay[:players] || []
    frames = max(replay[:frames] || 1, 1)

    players
    |> Enum.with_index()
    |> Enum.map(fn {p, i} ->
      activity = (p[:input_frames] || 0) / frames * 100
      {"P#{i + 1}", "#{Float.round(activity, 1)}% active"}
    end)
    |> Map.new()
  end

  defp sd_details(replay) do
    players = replay[:players] || []

    players
    |> Enum.with_index()
    |> Enum.map(fn {p, i} ->
      sd_count = p[:sd_count] || 0
      stocks_lost = p[:stocks_lost] || 0
      {"P#{i + 1}", "#{sd_count}/#{stocks_lost} SDs"}
    end)
    |> Map.new()
  end

  defp diversity_details(replay) do
    players = replay[:players] || []

    players
    |> Enum.with_index()
    |> Enum.map(fn {p, i} ->
      actions = p[:unique_actions] || 0
      {"P#{i + 1}", "#{actions} unique actions"}
    end)
    |> Map.new()
  end

  defp quality_label(score) do
    cond do
      score >= @excellent_score -> "Excellent"
      score >= @good_score -> "Good"
      score >= @fair_score -> "Fair"
      true -> "Poor"
    end
  end

  defp score_color(score) when score >= 80, do: :green
  defp score_color(score) when score >= 60, do: :cyan
  defp score_color(score) when score >= 40, do: :yellow
  defp score_color(_), do: :red

  # ============================================================
  # Peppi Integration
  # ============================================================

  @doc """
  Convert a parsed Peppi replay to quality scoring format.

  Extracts quality metrics from the Peppi ParsedReplay struct:
  - Frame count from metadata or frames list
  - Per-player damage, stocks, SD rate, input activity, action diversity

  ## Example

      {:ok, replay} = Peppi.parse("game.slp")
      quality_data = ReplayQuality.from_parsed_replay(replay)
      score = ReplayQuality.score(quality_data)

  """
  @spec from_parsed_replay(map() | struct()) :: map()
  def from_parsed_replay(%{frames: frames, metadata: metadata}) do
    frame_count = length(frames)
    players_meta = metadata.players || []

    # Build per-player stats by analyzing frames
    player_stats = compute_player_stats(frames, players_meta)

    %{
      frames: frame_count,
      players: player_stats
    }
  end

  defp compute_player_stats(frames, players_meta) do
    # Initialize accumulators for each player
    num_players = length(players_meta)
    if num_players == 0, do: [], else: do_compute_player_stats(frames, num_players, players_meta)
  end

  defp do_compute_player_stats(frames, num_players, _players_meta) do
    # Track per-player metrics across all frames
    initial = for i <- 0..(num_players - 1), into: %{} do
      {i, %{
        prev_percent: 0.0,
        prev_stock: 4,
        damage_dealt: 0.0,
        damage_taken: 0.0,
        stocks_lost: 0,
        sd_count: 0,
        input_frames: 0,
        actions_seen: MapSet.new()
      }}
    end

    # Process each frame
    final = Enum.reduce(frames, initial, fn frame, acc ->
      players = frame.players || []

      Enum.reduce(Enum.with_index(players), acc, fn {player_data, idx}, acc2 ->
        # Handle both {port, player} tuples and bare player structs
        player = case player_data do
          {_port, p} -> p
          p -> p
        end

        if idx >= num_players do
          acc2
        else
          prev = Map.get(acc2, idx, %{prev_percent: 0.0, prev_stock: 4, damage_dealt: 0.0,
                                       damage_taken: 0.0, stocks_lost: 0, sd_count: 0,
                                       input_frames: 0, actions_seen: MapSet.new()})

          current_percent = player.percent || 0.0
          current_stock = player.stock || 4
          action = player.action

          # Damage taken = percent increase
          damage_increase = max(0, current_percent - prev.prev_percent)

          # Stock lost detection (stock decreased and percent reset)
          stock_lost = if current_stock < prev.prev_stock and current_percent < 30, do: 1, else: 0

          # SD detection: stock lost while near blast zone or in certain actions
          # Simplified: count stock losses where opponent didn't deal recent damage
          # For now, estimate SD as stock loss when percent was < 50
          sd = if stock_lost > 0 and prev.prev_percent < 50, do: 1, else: 0

          # Input activity: non-neutral stick or any button pressed
          has_input = has_active_input?(player.controller)

          updated = %{
            prev_percent: if(current_stock < prev.prev_stock, do: 0.0, else: current_percent),
            prev_stock: current_stock,
            damage_dealt: prev.damage_dealt,  # Updated from opponent's perspective
            damage_taken: prev.damage_taken + damage_increase,
            stocks_lost: prev.stocks_lost + stock_lost,
            sd_count: prev.sd_count + sd,
            input_frames: prev.input_frames + if(has_input, do: 1, else: 0),
            actions_seen: MapSet.put(prev.actions_seen, action)
          }

          Map.put(acc2, idx, updated)
        end
      end)
    end)

    # Cross-reference damage: P1's damage_dealt = P2's damage_taken
    for i <- 0..(num_players - 1) do
      stats = Map.get(final, i, %{})
      opponent_idx = rem(i + 1, num_players)
      opponent = Map.get(final, opponent_idx, %{damage_taken: 0.0})

      %{
        damage_dealt: opponent.damage_taken,
        damage_taken: stats.damage_taken,
        stocks_lost: stats.stocks_lost,
        sd_count: stats.sd_count,
        input_frames: stats.input_frames,
        unique_actions: MapSet.size(stats.actions_seen),
        is_cpu: false  # Peppi metadata doesn't expose this directly
      }
    end
  end

  defp has_active_input?(nil), do: false
  defp has_active_input?(controller) do
    # Check for non-neutral stick
    main_x = controller.main_stick_x || 0.0
    main_y = controller.main_stick_y || 0.0
    c_x = controller.c_stick_x || 0.0
    c_y = controller.c_stick_y || 0.0

    stick_active = abs(main_x) > 0.3 or abs(main_y) > 0.3 or
                   abs(c_x) > 0.3 or abs(c_y) > 0.3

    # Check for button presses
    button_active = controller.button_a or controller.button_b or
                    controller.button_x or controller.button_y or
                    controller.button_z or controller.button_l or
                    controller.button_r

    stick_active or button_active
  end
end
