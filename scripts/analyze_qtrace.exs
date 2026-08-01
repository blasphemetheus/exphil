# Analyze an EXPHIL_QUEUE_TRACE=1 log: per-game decode->apply latency and
# B-press behavior. Replay-free — works when the mainline replay writer
# dies (its usual state) and under netplay, where the 2026-08-01 own-port
# fix makes the applied= stream trustworthy.
#
#   mix run scripts/analyze_qtrace.exs eval_runs/direct_rematch.log
#   mix run scripts/analyze_qtrace.exs LOG --max-lag 12 --game 2
#
# Per game (segmented on backward frame jumps = new-game resets; verified
# 2026-08-01 that libmelee never re-delivers earlier frames mid-game):
#   * lag-agreement curve: % of frames where applied inputs == the decision
#     from N frames earlier. A sharp peak at N = measured decision->apply
#     latency (expect --frame-delay + 2); a flat curve = smear or a
#     wrong-port applied stream (pre-fix symptom).
#   * B run-length histograms, decision stream vs applied stream: healthy
#     multishine = many short runs; long runs = hold-B degeneration.
#
# Pure text processing — no Nx/EXLA, safe to run beside a live play beam.

alias ExPhil.Training.Output

{opts, args, _} =
  OptionParser.parse(System.argv(), strict: [max_lag: :integer, game: :integer])

log_path = List.first(args) || raise "usage: analyze_qtrace.exs LOG [--max-lag N] [--game N]"
max_lag = opts[:max_lag] || 10

parse_line = fn line ->
  case Regex.run(~r/\[qtrace\] f(-?\d+) applied=(\S+)((?: \S+)*)/, line) do
    [_, f, applied, slots] ->
      {String.to_integer(f), applied, slots |> String.split(" ", trim: true) |> List.first()}

    _ ->
      nil
  end
end

# Segment into games on backward frame jumps
games =
  log_path
  |> File.stream!()
  |> Stream.map(parse_line)
  |> Stream.reject(&is_nil/1)
  |> Enum.reduce([], fn {f, _, _} = entry, acc ->
    case acc do
      [[{prev_f, _, _} | _] = cur | rest] when f >= prev_f ->
        [[entry | cur] | rest]

      _ ->
        [[entry] | acc]
    end
  end)
  |> Enum.map(&Enum.reverse/1)
  |> Enum.reverse()

run_lengths = fn flags ->
  flags
  |> Enum.chunk_by(& &1)
  |> Enum.filter(&hd/1)
  |> Enum.map(&length/1)
  |> Enum.frequencies()
  |> Enum.sort()
end

fmt_hist = fn hist ->
  Enum.map_join(hist, " ", fn {len, count} -> "#{count}x#{len}" end)
end

Output.banner("qtrace analysis: #{log_path}")
Output.puts("#{length(games)} game(s) found")

games
|> Enum.with_index(1)
|> Enum.filter(fn {_, idx} -> opts[:game] in [nil, idx] end)
|> Enum.each(fn {entries, idx} ->
  in_game = Enum.filter(entries, fn {f, _, _} -> f >= 0 end)
  n = length(in_game)

  Output.puts("")
  Output.puts("=== game #{idx}: #{n} traced frames (of #{length(entries)} incl. countdown)")

  if n < 120 do
    Output.warning("too short to analyze, skipping")
  else
    decisions = Map.new(in_game, fn {f, _, slot1} -> {f, slot1} end)
    applieds = Map.new(in_game, fn {f, applied, _} -> {f, applied} end)
    frames = in_game |> Enum.map(&elem(&1, 0))
    {min_f, max_f} = Enum.min_max(frames)

    curve =
      for lag <- 0..max_lag do
        {match, total} =
          Enum.reduce((min_f + lag)..max_f, {0, 0}, fn f, {m, t} ->
            case {applieds[f], decisions[f - lag]} do
              {nil, _} -> {m, t}
              {_, nil} -> {m, t}
              {same, same} -> {m + 1, t + 1}
              _ -> {m, t + 1}
            end
          end)

        {lag, if(total > 0, do: 100.0 * match / total, else: 0.0), total}
      end

    {peak_lag, peak_pct, _} = Enum.max_by(curve, &elem(&1, 1))

    Enum.each(curve, fn {lag, pct, total} ->
      marker = if lag == peak_lag, do: "  <-- PEAK", else: ""
      Output.puts("  lag #{String.pad_leading(to_string(lag), 2)}: " <>
        "#{:erlang.float_to_binary(pct, decimals: 1)}% (#{total} frames)#{marker}")
    end)

    verdict =
      cond do
        peak_pct >= 90.0 -> "sharp — decision->apply latency is #{peak_lag} frames"
        peak_pct >= 75.0 -> "moderate peak at #{peak_lag} — some jitter/smear"
        true -> "FLAT — smear, or the applied stream is not the bot's own port"
      end

    Output.puts("  verdict: #{verdict} (expect --frame-delay + 2)")

    d_flags = Enum.map(min_f..max_f, fn f -> match?("B" <> _, decisions[f] || "") end)
    a_flags = Enum.map(min_f..max_f, fn f -> match?("B" <> _, applieds[f] || "") end)
    d_hist = run_lengths.(d_flags)
    a_hist = run_lengths.(a_flags)

    d_presses = d_hist |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    d_long = d_hist |> Enum.filter(fn {len, _} -> len > 30 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    minutes = n / 3600.0

    Output.puts("")
    Output.puts("  B press cycles: #{d_presses} (#{Float.round(d_presses / max(minutes, 0.01), 1)}/min); " <>
      "#{d_long} hold-B runs >30f")
    Output.puts("  decision B-runs: #{fmt_hist.(d_hist)}")
    Output.puts("  applied  B-runs: #{fmt_hist.(a_hist)}")
  end
end)
