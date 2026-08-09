#!/usr/bin/env elixir
# Behavior metrics for GENERALIST bots — the eval the shine counters can't
# give (HANDOFF_2026-08-08 queue item #4). Scores full-game play from
# replays: stocks, damage, action diversity, input rate, movement.
#
# USAGE:
#   mix run scripts/analyze_behavior.exs --replays eval_runs/0809_fox_il_v2_cpu --port 1
#   mix run scripts/analyze_behavior.exs --replays dir_a,dir_b --port 1
#
# Bot is --port (default 1; eval_live_protocol puts the policy on port 1,
# CPU on port 2). One table row per replay + a per-directory mean row so
# n>=3 protocol runs read as one number.

alias ExPhil.Data.Peppi
alias ExPhil.Interp.ReplayStats
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, quiet: :boolean]
  )

if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

dirs = String.split(opts[:replays] || raise("--replays DIR[,DIR...] required"), ",")
bot_port = opts[:port] || 1
opp_port = if bot_port == 1, do: 2, else: 1

Output.banner("Behavior Analyzer")

analyze_replay = fn path ->
  case Peppi.parse(path, player_port: bot_port) do
    {:ok, replay} ->
      frames = replay.frames

      if frames == [] do
        nil
      else
        n = length(frames)

        get = fn frame, port -> Map.get(frame.players, port) end

        first = List.first(frames)
        last = List.last(frames)

        start_stocks = fn port -> (get.(first, port) || %{stock: 4}).stock || 4 end
        end_stocks = fn port -> (get.(last, port) || %{stock: 0}).stock || 0 end

        # Damage dealt: sum of positive percent deltas on the opponent
        # (resets to 0 on stock loss don't count as negative damage)
        {damage_dealt, _} =
          Enum.reduce(frames, {0.0, nil}, fn frame, {acc, prev} ->
            cur = (get.(frame, opp_port) || %{percent: nil}).percent

            acc =
              if is_number(cur) and is_number(prev) and cur > prev,
                do: acc + (cur - prev),
                else: acc

            {acc, cur}
          end)

        # Bot action-state diversity + input rate + movement span
        bot_frames = Enum.map(frames, &get.(&1, bot_port)) |> Enum.reject(&is_nil/1)
        actions = Enum.map(bot_frames, & &1.action) |> Enum.reject(&is_nil/1)
        distinct_actions = actions |> Enum.uniq() |> length()

        # Input events: frames where the controller state changes (button
        # or stick) — an APM-like rate without double-counting held inputs
        input_changes =
          bot_frames
          |> Enum.map(& &1.controller)
          |> Enum.chunk_every(2, 1, :discard)
          |> Enum.count(fn [a, b] -> a != b and b != nil end)

        # Shield behavior via the canonical module (task #27): fraction of
        # frames shielding + observed hard shield breaks (entry into 205..211)
        shield = ReplayStats.shield_stats(Enum.map(actions, &trunc(&1 * 1.0)))

        # Death classification: a stock decrement whose PRE-death percent is
        # under 20 is an SD (walked/fell off), not a KO
        {sd_deaths, ko_deaths, _} =
          Enum.reduce(frames, {0, 0, nil}, fn frame, {sd, ko, prev} ->
            p = get.(frame, bot_port)

            case {prev, p} do
              {%{stock: ps, percent: pp}, %{stock: cs}}
              when is_integer(ps) and is_integer(cs) and cs < ps ->
                if is_number(pp) and pp < 20.0,
                  do: {sd + 1, ko, p},
                  else: {sd, ko + 1, p}

              _ ->
                {sd, ko, p}
            end
          end)

        xs = bot_frames |> Enum.map(& &1.x) |> Enum.reject(&is_nil/1)
        offstage =
          bot_frames
          |> Enum.count(fn p ->
            is_number(p.x) and is_number(p.y) and (abs(p.x) > 85.0 or p.y < -10.0)
          end)

        minutes = n / 3600

        %{
          file: Path.basename(path),
          seconds: Float.round(n / 60, 1),
          stocks_lost: start_stocks.(bot_port) - end_stocks.(bot_port),
          sd_deaths: sd_deaths,
          ko_deaths: ko_deaths,
          stocks_taken: start_stocks.(opp_port) - end_stocks.(opp_port),
          shield_pct: Float.round(shield.shield_frac * 100, 1),
          shieldbreaks: shield.breaks,
          damage_dealt: Float.round(damage_dealt, 1),
          distinct_actions: distinct_actions,
          inputs_per_min: Float.round(input_changes / max(minutes, 1.0e-9), 0),
          x_span: if(xs == [], do: 0.0, else: Float.round(Enum.max(xs) - Enum.min(xs), 0)),
          offstage_pct: Float.round(offstage / max(n, 1) * 100, 1)
        }
      end

    _ ->
      nil
  end
end

for dir <- dirs do
  files = Path.wildcard(Path.join(dir, "**/*.slp")) |> Enum.sort()
  Output.puts("")
  Output.puts("== #{dir} (#{length(files)} replays, bot port #{bot_port})")

  rows = files |> Enum.map(analyze_replay) |> Enum.reject(&is_nil/1)

  keys = [
    :seconds, :stocks_lost, :sd_deaths, :ko_deaths, :stocks_taken, :damage_dealt,
    :shield_pct, :shieldbreaks, :distinct_actions, :inputs_per_min, :x_span, :offstage_pct
  ]

  header = ~w(file secs stk_lost SD KO stk_taken dmg shield% breaks actions inp/min x_span offstage%)
  Output.puts("  " <> Enum.join(header, "  "))

  row_line = fn label, r ->
    Output.puts("  " <> label <> "  " <> Enum.map_join(keys, "  ", &"#{r[&1]}"))
  end

  for r <- rows, do: row_line.(r.file, r)

  if rows != [] do
    means =
      Map.new(keys, fn k ->
        {k, Float.round(Enum.sum(Enum.map(rows, &(&1[k] * 1.0))) / length(rows), 1)}
      end)

    row_line.("MEAN", means)
  end
end
