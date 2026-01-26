defmodule ExPhil.Data.ParseStats do
  @moduledoc """
  Statistics tracking for replay parsing.

  Tracks why frames are dropped during parsing/conversion, helping diagnose
  issues like wrong port selection, disconnected players, or corrupted data.

  ## Usage

      stats = ParseStats.new()
      {frames, stats} = ParseStats.track_frames(raw_frames, stats)
      ParseStats.print_summary(stats)

  """

  defstruct [
    :total_frames,
    :valid_frames,
    :dropped_no_player,
    :dropped_no_opponent,
    :dropped_no_controller,
    :dropped_delay_cutoff,
    :warnings
  ]

  @type t :: %__MODULE__{
          total_frames: non_neg_integer(),
          valid_frames: non_neg_integer(),
          dropped_no_player: non_neg_integer(),
          dropped_no_opponent: non_neg_integer(),
          dropped_no_controller: non_neg_integer(),
          dropped_delay_cutoff: non_neg_integer(),
          warnings: [String.t()]
        }

  @doc "Create new stats tracker"
  def new do
    %__MODULE__{
      total_frames: 0,
      valid_frames: 0,
      dropped_no_player: 0,
      dropped_no_opponent: 0,
      dropped_no_controller: 0,
      dropped_delay_cutoff: 0,
      warnings: []
    }
  end

  @doc "Track frame extraction with detailed drop reasons"
  def track_extraction(frames, player_port, opponent_port, opts \\ []) do
    stats = new()
    delay = Keyword.get(opts, :frame_delay, 0)

    {valid_frames, stats} =
      Enum.reduce(frames, {[], stats}, fn frame, {acc, stats} ->
        stats = %{stats | total_frames: stats.total_frames + 1}

        player = Map.get(frame.players, player_port)
        opponent = Map.get(frame.players, opponent_port)

        cond do
          player == nil ->
            {acc, %{stats | dropped_no_player: stats.dropped_no_player + 1}}

          opponent == nil ->
            {acc, %{stats | dropped_no_opponent: stats.dropped_no_opponent + 1}}

          player.controller == nil or get_controller(player) == nil ->
            {acc, %{stats | dropped_no_controller: stats.dropped_no_controller + 1}}

          true ->
            {[frame | acc], %{stats | valid_frames: stats.valid_frames + 1}}
        end
      end)

    # Adjust for delay cutoff
    stats =
      if delay > 0 do
        cutoff = min(delay, stats.valid_frames)
        %{stats | dropped_delay_cutoff: cutoff, valid_frames: stats.valid_frames - cutoff}
      else
        stats
      end

    # Add warnings for suspicious patterns
    stats = add_warnings(stats, player_port, opponent_port)

    {Enum.reverse(valid_frames), stats}
  end

  defp get_controller(%{controller: c}), do: c
  defp get_controller(_), do: nil

  defp add_warnings(stats, player_port, opponent_port) do
    warnings = []

    # Warn if no valid frames
    warnings =
      if stats.valid_frames == 0 and stats.total_frames > 0 do
        [
          "No valid frames extracted - check player port (requested port #{player_port})"
          | warnings
        ]
      else
        warnings
      end

    # Warn if majority of frames dropped due to missing player
    warnings =
      if stats.dropped_no_player > stats.total_frames * 0.5 do
        pct = round(stats.dropped_no_player / stats.total_frames * 100)
        ["#{pct}% frames missing player #{player_port} - wrong port?" | warnings]
      else
        warnings
      end

    # Warn if majority dropped due to missing opponent
    warnings =
      if stats.dropped_no_opponent > stats.total_frames * 0.5 do
        pct = round(stats.dropped_no_opponent / stats.total_frames * 100)
        ["#{pct}% frames missing opponent #{opponent_port} - wrong port or 1P mode?" | warnings]
      else
        warnings
      end

    %{stats | warnings: warnings}
  end

  @doc "Merge multiple stats together"
  def merge(stats_list) when is_list(stats_list) do
    Enum.reduce(stats_list, new(), fn stats, acc ->
      %__MODULE__{
        total_frames: acc.total_frames + stats.total_frames,
        valid_frames: acc.valid_frames + stats.valid_frames,
        dropped_no_player: acc.dropped_no_player + stats.dropped_no_player,
        dropped_no_opponent: acc.dropped_no_opponent + stats.dropped_no_opponent,
        dropped_no_controller: acc.dropped_no_controller + stats.dropped_no_controller,
        dropped_delay_cutoff: acc.dropped_delay_cutoff + stats.dropped_delay_cutoff,
        warnings: acc.warnings ++ stats.warnings
      }
    end)
  end

  @doc "Check if stats indicate problems"
  def has_issues?(%__MODULE__{} = stats) do
    stats.warnings != [] or stats.valid_frames == 0
  end

  @doc "Get drop percentage"
  def drop_percentage(%__MODULE__{total_frames: 0}), do: 0.0

  def drop_percentage(%__MODULE__{} = stats) do
    dropped = stats.total_frames - stats.valid_frames
    Float.round(dropped / stats.total_frames * 100, 1)
  end

  @doc "Format stats as a summary string"
  def format_summary(%__MODULE__{} = stats) do
    lines = [
      "Parse Stats:",
      "  Total frames: #{stats.total_frames}",
      "  Valid frames: #{stats.valid_frames} (#{100 - drop_percentage(stats)}%)"
    ]

    # Add drop reasons if any
    drop_lines =
      [
        {stats.dropped_no_player, "missing player"},
        {stats.dropped_no_opponent, "missing opponent"},
        {stats.dropped_no_controller, "missing controller"},
        {stats.dropped_delay_cutoff, "delay cutoff"}
      ]
      |> Enum.filter(fn {count, _} -> count > 0 end)
      |> Enum.map(fn {count, reason} -> "    - #{reason}: #{count}" end)

    lines =
      if drop_lines != [] do
        lines ++ ["  Dropped frames:"] ++ drop_lines
      else
        lines
      end

    # Add warnings
    lines =
      if stats.warnings != [] do
        warning_lines = Enum.map(stats.warnings, &("  ⚠ " <> &1))
        lines ++ ["  Warnings:"] ++ warning_lines
      else
        lines
      end

    Enum.join(lines, "\n")
  end

  @doc "Print summary to output"
  def print_summary(%__MODULE__{} = stats, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)
    output_module = Keyword.get(opts, :output, ExPhil.Training.Output)

    # Always show warnings
    if stats.warnings != [] do
      Enum.each(stats.warnings, fn warning ->
        output_module.warning(warning)
      end)
    end

    # Show full stats if verbose or if there were significant drops
    if verbose or drop_percentage(stats) > 10 do
      output_module.puts(format_summary(stats))
    end
  end
end
