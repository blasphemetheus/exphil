#!/usr/bin/env elixir
# Analyze replay data for action distributions and potential training issues
#
# USAGE:
#   mix run scripts/analyze_replays.exs --replays ~/replays/mewtwo
#   mix run scripts/analyze_replays.exs --replays ~/replays/mewtwo --player-port 1
#   mix run scripts/analyze_replays.exs --replays ~/replays/mewtwo --top-actions 20

alias ExPhil.Training.Output
alias ExPhil.Data.Peppi

defmodule ReplayAnalyzer do
  @default_opts [
    top_actions: 15,
    player_port: nil,  # nil = auto-detect
    show_stages: true,
    show_positions: true
  ]

  def run(args) do
    opts = parse_args(args)

    Output.banner("Replay Data Analyzer")
    Output.puts("")

    # Find replays
    Output.step(1, 4, "Finding replays")
    replay_files = find_replays(opts[:replays])
    Output.puts("  Found #{length(replay_files)} replay files")
    Output.puts("")

    # Parse replays
    Output.step(2, 4, "Parsing replays")
    {replays, parse_errors} = parse_all_replays(replay_files)
    Output.puts("  Successfully parsed: #{length(replays)}")
    if parse_errors > 0, do: Output.puts("  Parse errors: #{parse_errors}")
    Output.puts("")

    # Analyze
    Output.step(3, 4, "Analyzing data")
    analysis = analyze_replays(replays, opts)
    Output.puts("")

    # Report
    Output.step(4, 4, "Generating report")
    print_report(analysis, opts)
  end

  defp parse_args(args) do
    {opts, _rest, _invalid} = OptionParser.parse(args,
      strict: [
        replays: :string,
        player_port: :integer,
        top_actions: :integer,
        show_stages: :boolean,
        show_positions: :boolean,
        character: :string,
        help: :boolean
      ],
      aliases: [
        r: :replays,
        p: :player_port,
        n: :top_actions,
        c: :character,
        h: :help
      ]
    )

    if opts[:help] do
      print_help()
      System.halt(0)
    end

    unless opts[:replays] do
      Output.error("Missing required --replays argument")
      print_help()
      System.halt(1)
    end

    Keyword.merge(@default_opts, opts)
  end

  defp print_help do
    IO.puts("""
    Replay Data Analyzer

    Analyze action distributions and data characteristics for training.

    USAGE:
      mix run scripts/analyze_replays.exs --replays <path> [options]

    OPTIONS:
      -r, --replays <path>       Path to replay directory (required)
      -p, --player-port <1|2>    Analyze specific port (default: auto-detect character)
      -c, --character <name>     Filter to specific character (e.g., mewtwo, ganondorf)
      -n, --top-actions <N>      Show top N actions (default: 15)
      --no-show-stages           Hide stage breakdown
      --no-show-positions        Hide position analysis

    EXAMPLES:
      mix run scripts/analyze_replays.exs -r ~/replays/mewtwo
      mix run scripts/analyze_replays.exs -r ~/replays -c mewtwo -n 20
    """)
  end

  defp find_replays(dir) do
    unless File.dir?(dir) do
      Output.error("Directory not found: #{dir}")
      System.halt(1)
    end

    Path.wildcard(Path.join(dir, "**/*.slp"))
  end

  defp parse_all_replays(files) do
    total = length(files)

    results = files
    |> Enum.with_index(1)
    |> Enum.map(fn {file, idx} ->
      if rem(idx, 10) == 0 or idx == total do
        Output.progress_bar(idx, total, label: "Parsing")
      end

      case parse_replay(file) do
        {:ok, data} -> {:ok, data}
        {:error, _} -> :error
      end
    end)

    Output.progress_done()

    successes = results |> Enum.filter(&match?({:ok, _}, &1)) |> Enum.map(fn {:ok, d} -> d end)
    errors = results |> Enum.count(&(&1 == :error))

    {successes, errors}
  end

  defp parse_replay(file) do
    case Peppi.parse(file) do
      {:ok, replay} -> {:ok, normalize_replay(replay)}
      {:error, reason} -> {:error, reason}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp normalize_replay(replay) do
    # Extract player info from metadata
    players = replay.metadata.players
    p1 = Enum.find(players, &(&1.port == 1)) || Enum.at(players, 0)
    p2 = Enum.find(players, &(&1.port == 2)) || Enum.find(players, &(&1.port == 3)) || Enum.at(players, 1)

    %{
      file: replay.metadata.path,
      p1_character: p1 && p1.character_name,
      p2_character: p2 && p2.character_name,
      p1_port: p1 && p1.port,
      p2_port: p2 && p2.port,
      stage: replay.metadata.stage,
      duration: replay.metadata.duration_frames,
      frames: replay.frames
    }
  end

  defp analyze_replays(replays, opts) do
    # Collect all frame data
    all_frames = collect_frames(replays, opts)

    %{
      total_replays: length(replays),
      total_frames: length(all_frames),
      action_counts: count_actions(all_frames),
      button_counts: count_buttons(all_frames),
      stick_positions: analyze_stick_positions(all_frames),
      stage_counts: count_stages(replays),
      character_counts: count_characters(replays),
      position_stats: analyze_positions(all_frames),
      game_outcomes: analyze_outcomes(replays, opts)
    }
  end

  defp collect_frames(replays, opts) do
    player_port = opts[:player_port]
    target_char = opts[:character] && String.downcase(opts[:character])

    Enum.flat_map(replays, fn replay ->
      # Determine which port to use
      port = cond do
        player_port -> player_port
        target_char -> find_character_port(replay, target_char)
        true -> 1  # Default to port 1
      end

      extract_player_frames(replay, port)
    end)
  end

  defp find_character_port(replay, target_char) do
    p1_char = to_string(replay.p1_character || "")
    p2_char = to_string(replay.p2_character || "")

    cond do
      String.downcase(p1_char) |> String.contains?(target_char) -> replay.p1_port || 1
      String.downcase(p2_char) |> String.contains?(target_char) -> replay.p2_port || 2
      true -> replay.p1_port || 1
    end
  end

  defp extract_player_frames(replay, port) do
    frames = replay.frames || []

    Enum.flat_map(frames, fn frame ->
      case frame do
        %Peppi.GameFrame{players: players} ->
          player = Map.get(players, port)
          if player do
            ctrl = player.controller || %{}
            [%{
              action_state: player.action,
              position_x: player.x,
              position_y: player.y,
              buttons: extract_buttons(ctrl),
              main_stick_x: (ctrl.main_stick_x || 0.5) - 0.5,  # Normalize to -0.5 to 0.5
              main_stick_y: (ctrl.main_stick_y || 0.5) - 0.5,
              c_stick_x: (ctrl.c_stick_x || 0.5) - 0.5,
              c_stick_y: (ctrl.c_stick_y || 0.5) - 0.5,
              stocks: player.stock,
              percent: player.percent
            }]
          else
            []
          end
        _ ->
          []
      end
    end)
    |> Enum.reject(&is_nil(&1.action_state))
  end

  defp extract_buttons(ctrl) do
    buttons = []
    buttons = if ctrl.button_a, do: [:A | buttons], else: buttons
    buttons = if ctrl.button_b, do: [:B | buttons], else: buttons
    buttons = if ctrl.button_x, do: [:X | buttons], else: buttons
    buttons = if ctrl.button_y, do: [:Y | buttons], else: buttons
    buttons = if ctrl.button_z, do: [:Z | buttons], else: buttons
    buttons = if ctrl.button_l, do: [:L | buttons], else: buttons
    buttons = if ctrl.button_r, do: [:R | buttons], else: buttons
    buttons = if ctrl.l_trigger > 0.3, do: [:L_ANALOG | buttons], else: buttons
    buttons = if ctrl.r_trigger > 0.3, do: [:R_ANALOG | buttons], else: buttons
    buttons
  end

  defp count_actions(frames) do
    frames
    |> Enum.map(& &1.action_state)
    |> Enum.reject(&is_nil/1)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_action, count} -> -count end)
  end

  defp count_buttons(frames) do
    frames
    |> Enum.flat_map(fn frame ->
      case frame.buttons do
        buttons when is_list(buttons) -> buttons
        buttons when is_map(buttons) -> Map.keys(buttons) |> Enum.filter(&buttons[&1])
        _ -> []
      end
    end)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_button, count} -> -count end)
  end

  defp analyze_stick_positions(frames) do
    main_x = frames |> Enum.map(& &1.main_stick_x) |> Enum.reject(&is_nil/1)
    main_y = frames |> Enum.map(& &1.main_stick_y) |> Enum.reject(&is_nil/1)

    %{
      main_x_mean: safe_mean(main_x),
      main_y_mean: safe_mean(main_y),
      main_x_std: safe_std(main_x),
      main_y_std: safe_std(main_y),
      # Check for directional bias
      right_bias: Enum.count(main_x, &(&1 > 0.3)) / max(length(main_x), 1),
      left_bias: Enum.count(main_x, &(&1 < -0.3)) / max(length(main_x), 1),
      up_bias: Enum.count(main_y, &(&1 > 0.3)) / max(length(main_y), 1),
      down_bias: Enum.count(main_y, &(&1 < -0.3)) / max(length(main_y), 1)
    }
  end

  defp safe_mean([]), do: 0.0
  defp safe_mean(list), do: Enum.sum(list) / length(list)

  defp safe_std([]), do: 0.0
  defp safe_std(list) do
    mean = safe_mean(list)
    variance = list |> Enum.map(&(:math.pow(&1 - mean, 2))) |> safe_mean()
    :math.sqrt(variance)
  end

  defp count_stages(replays) do
    replays
    |> Enum.map(&(&1[:stage] || &1["stage"]))
    |> Enum.reject(&is_nil/1)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_stage, count} -> -count end)
  end

  defp count_characters(replays) do
    (Enum.map(replays, &(&1[:p1_character] || &1["p1_character"])) ++
     Enum.map(replays, &(&1[:p2_character] || &1["p2_character"])))
    |> Enum.reject(&is_nil/1)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_char, count} -> -count end)
  end

  defp analyze_positions(frames) do
    xs = frames |> Enum.map(& &1.position_x) |> Enum.reject(&is_nil/1)
    ys = frames |> Enum.map(& &1.position_y) |> Enum.reject(&is_nil/1)

    if length(xs) > 0 do
      %{
        x_mean: safe_mean(xs),
        y_mean: safe_mean(ys),
        x_min: Enum.min(xs),
        x_max: Enum.max(xs),
        y_min: Enum.min(ys),
        y_max: Enum.max(ys),
        # Offstage analysis (rough - depends on stage)
        offstage_left: Enum.count(xs, &(&1 < -70)) / length(xs) * 100,
        offstage_right: Enum.count(xs, &(&1 > 70)) / length(xs) * 100,
        airborne: Enum.count(ys, &(&1 > 10)) / length(ys) * 100
      }
    else
      %{}
    end
  end

  defp analyze_outcomes(replays, opts) do
    player_port = opts[:player_port] || 1

    results = Enum.map(replays, fn replay ->
      # Try to determine winner from final stocks
      p1_stocks = get_final_stocks(replay, 1)
      p2_stocks = get_final_stocks(replay, 2)

      cond do
        is_nil(p1_stocks) or is_nil(p2_stocks) -> :unknown
        p1_stocks > p2_stocks -> if player_port == 1, do: :win, else: :loss
        p2_stocks > p1_stocks -> if player_port == 2, do: :win, else: :loss
        true -> :draw
      end
    end)

    %{
      wins: Enum.count(results, &(&1 == :win)),
      losses: Enum.count(results, &(&1 == :loss)),
      draws: Enum.count(results, &(&1 == :draw)),
      unknown: Enum.count(results, &(&1 == :unknown))
    }
  end

  defp get_final_stocks(replay, port) do
    frames = replay.frames || []

    case List.last(frames) do
      nil -> nil
      %Peppi.GameFrame{players: players} ->
        player = Map.get(players, port)
        player && player.stock
      _ -> nil
    end
  end

  # ============================================================================
  # Report Printing
  # ============================================================================

  defp print_report(analysis, opts) do
    Output.puts("")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("REPLAY ANALYSIS REPORT")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("")

    # Summary
    Output.puts("Summary:")
    Output.puts("  Total replays: #{analysis.total_replays}")
    Output.puts("  Total frames: #{analysis.total_frames}")
    Output.puts("  Avg frames/replay: #{div(analysis.total_frames, max(analysis.total_replays, 1))}")
    Output.puts("")

    # Game outcomes
    if analysis.game_outcomes.wins + analysis.game_outcomes.losses > 0 do
      total_known = analysis.game_outcomes.wins + analysis.game_outcomes.losses
      win_rate = analysis.game_outcomes.wins / total_known * 100
      Output.puts("Game Outcomes:")
      Output.puts("  Wins: #{analysis.game_outcomes.wins}")
      Output.puts("  Losses: #{analysis.game_outcomes.losses}")
      Output.puts("  Win Rate: #{Float.round(win_rate, 1)}%")
      if analysis.game_outcomes.unknown > 0 do
        Output.puts("  Unknown: #{analysis.game_outcomes.unknown}")
      end
      Output.puts("")
    end

    # Action distribution
    Output.puts("Top #{opts[:top_actions]} Actions:")
    Output.puts("-" |> String.duplicate(40))

    total_actions = analysis.action_counts |> Enum.map(fn {_, c} -> c end) |> Enum.sum()

    analysis.action_counts
    |> Enum.take(opts[:top_actions])
    |> Enum.each(fn {action, count} ->
      pct = count / total_actions * 100
      bar_len = round(pct / 2)
      bar = String.duplicate("█", bar_len)
      Output.puts("  #{String.pad_trailing(to_string(action), 25)} #{String.pad_leading(Integer.to_string(count), 8)} (#{Float.round(pct, 1)}%) #{bar}")
    end)
    Output.puts("")

    # Check for problematic actions
    print_action_warnings(analysis.action_counts, total_actions)

    # Stick analysis
    if map_size(analysis.stick_positions) > 0 do
      Output.puts("Stick Position Analysis:")
      stick = analysis.stick_positions
      Output.puts("  Main stick X mean: #{Float.round(stick.main_x_mean, 3)}")
      Output.puts("  Main stick Y mean: #{Float.round(stick.main_y_mean, 3)}")
      Output.puts("")
      Output.puts("  Directional bias:")
      Output.puts("    Right (>0.3): #{Float.round(stick.right_bias * 100, 1)}%")
      Output.puts("    Left (<-0.3): #{Float.round(stick.left_bias * 100, 1)}%")
      Output.puts("    Up (>0.3): #{Float.round(stick.up_bias * 100, 1)}%")
      Output.puts("    Down (<-0.3): #{Float.round(stick.down_bias * 100, 1)}%")

      # Warn about bias
      if abs(stick.right_bias - stick.left_bias) > 0.1 do
        if stick.right_bias > stick.left_bias do
          Output.warning("Right stick bias detected (#{Float.round((stick.right_bias - stick.left_bias) * 100, 1)}% more right)")
        else
          Output.warning("Left stick bias detected (#{Float.round((stick.left_bias - stick.right_bias) * 100, 1)}% more left)")
        end
      end
      Output.puts("")
    end

    # Position analysis
    if opts[:show_positions] && map_size(analysis.position_stats) > 0 do
      pos = analysis.position_stats
      Output.puts("Position Analysis:")
      Output.puts("  X range: #{Float.round(pos.x_min, 1)} to #{Float.round(pos.x_max, 1)}")
      Output.puts("  Y range: #{Float.round(pos.y_min, 1)} to #{Float.round(pos.y_max, 1)}")
      Output.puts("  Offstage left: #{Float.round(pos.offstage_left, 1)}%")
      Output.puts("  Offstage right: #{Float.round(pos.offstage_right, 1)}%")
      Output.puts("  Airborne: #{Float.round(pos.airborne, 1)}%")

      if pos.offstage_left + pos.offstage_right > 20 do
        Output.warning("High offstage time (#{Float.round(pos.offstage_left + pos.offstage_right, 1)}%) - model may learn risky offstage behavior")
      end
      Output.puts("")
    end

    # Stage breakdown
    if opts[:show_stages] && length(analysis.stage_counts) > 0 do
      Output.puts("Stages:")
      Enum.each(analysis.stage_counts, fn {stage, count} ->
        Output.puts("  #{stage}: #{count}")
      end)
      Output.puts("")
    end

    # Character breakdown
    if length(analysis.character_counts) > 0 do
      Output.puts("Characters:")
      Enum.each(analysis.character_counts, fn {char, count} ->
        Output.puts("  #{char}: #{count}")
      end)
      Output.puts("")
    end

    Output.puts("=" |> String.duplicate(60))
  end

  defp print_action_warnings(_action_counts, 0), do: Output.puts("")
  defp print_action_warnings(action_counts, total) do
    # Look for potentially problematic patterns
    action_map = Map.new(action_counts)

    # Check for high roll usage
    roll_actions = [:roll_forward, :roll_backward, :ROLL_F, :ROLL_B,
                    "RollForward", "RollBackward", 232, 233]
    roll_count = roll_actions |> Enum.map(&Map.get(action_map, &1, 0)) |> Enum.sum()
    roll_pct = roll_count / total * 100

    if roll_pct > 5 do
      Output.warning("High roll usage (#{Float.round(roll_pct, 1)}%) - may lead to predictable defensive patterns")
    end

    # Check for high airdodge
    airdodge_actions = [:escape_air, :ESCAPE_AIR, "EscapeAir", 236]
    airdodge_count = airdodge_actions |> Enum.map(&Map.get(action_map, &1, 0)) |> Enum.sum()
    airdodge_pct = airdodge_count / total * 100

    if airdodge_pct > 8 do
      Output.warning("High airdodge usage (#{Float.round(airdodge_pct, 1)}%)")
    end

    # Check for dead/respawn states (indicates deaths)
    dead_actions = [:dead_down, :dead_left, :dead_right, :dead_up, :rebirth, :rebirth_wait,
                    :DEAD_DOWN, :DEAD_LEFT, :DEAD_RIGHT, :DEAD_UP, :REBIRTH, :REBIRTH_WAIT,
                    0, 1, 2, 3, 4, 5]
    dead_count = dead_actions |> Enum.map(&Map.get(action_map, &1, 0)) |> Enum.sum()
    dead_pct = dead_count / total * 100

    if dead_pct > 2 do
      Output.puts("  Note: #{Float.round(dead_pct, 1)}% of frames are death/respawn states")
    end

    Output.puts("")
  end
end

ReplayAnalyzer.run(System.argv())
