# Engagement + jab-discipline scan over .slp files (2026-09-29).
#
# Answers two human impressions with numbers, per game, for one port:
#   "the Mamba is a camper"      → range / approach / laser features
#   "multi-jab comes out too much" → jab1/jab2/jab3 rates, jab2-per-jab1,
#                                    A press-edges per jab1
# Same features on expert corpus games give the human band.
#
#   mix run scripts/engagement_scan.exs --label NAME --port 1 --out FILE.jsonl DIR_OR_FILE...
#   mix run scripts/engagement_scan.exs --label corpus --character fox --sample 60 --out FILE.jsonl DIR
#
# --port N        subject port (bot sessions: 1). With --character, the port
#                 is the first port playing that character instead (corpus).
# --sample N      random sample of N files from the given dirs (seeded).
# Prints a per-label summary table (median over games) at the end.
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [label: :string, port: :integer, character: :string, sample: :integer, out: :string,
             far: :float, seed: :integer])

label = opts[:label] || "scan"
far = opts[:far] || 60.0
files =
  paths
  |> Enum.flat_map(fn p -> if File.dir?(p), do: Path.wildcard(Path.join(p, "**/*.slp")), else: [p] end)
  |> Enum.sort()

files =
  case opts[:sample] do
    nil -> files
    n -> :rand.seed(:exsss, {opts[:seed] || 929, 1, 1}); Enum.take_random(files, n)
  end

if files == [], do: raise("no .slp files")

# Melee action-state ids (libmelee numbering): jab1 44, jab2 45, jab3 46,
# rapid jab 47–49; Fox blaster 341–348 (ground pull/charge/fire/end, air same).
jab1 = 44
jab2 = 45
jab3 = 46
laser_ids = MapSet.new(341..348)
actionable = fn p -> p.on_ground and (p.hitstun_frames_left || 0) == 0 and (p.action || 0) in [14, 20, 21, 22, 23, 24, 25] end
# 14 WAIT, 20 DASH, 21 RUN, 22 RUN_DIRECT?, 23 RUN_BRAKE, 24 TURNING, 25 TURNING_RUN — "standing/moving, free"

features = fn states, controllers, port, opp ->
  ctrl = List.to_tuple(controllers)
  n = length(states)
  minutes = n / 3600
  arr = List.to_tuple(states)
  dist = fn s -> abs(s.players[port].x - s.players[opp].x) end
  dists = Enum.map(states, dist)
  sorted = Enum.sort(dists)
  median_dist = Enum.at(sorted, div(n, 2))
  far_frac = Enum.count(dists, &(&1 > far)) / n

  # approaches: subject actionable + moving toward opponent, counted once per
  # contiguous run (a dash toward counts once until it stops closing)
  {approaches, _} =
    Enum.reduce(1..(n - 1)//1, {0, false}, fn i, {acc, in_run} ->
      s = elem(arr, i)
      p = s.players[port]
      closing = dist.(s) < dist.(elem(arr, i - 1)) - 0.5
      free = actionable.(p) and (p.action || 0) in [20, 21]
      cond do
        closing and free and not in_run -> {acc + 1, true}
        closing and free -> {acc, true}
        true -> {acc, false}
      end
    end)

  # opponent-side approaches, same rule, for "who initiates"
  {opp_approaches, _} =
    Enum.reduce(1..(n - 1)//1, {0, false}, fn i, {acc, in_run} ->
      s = elem(arr, i)
      p = s.players[opp]
      closing = dist.(s) < dist.(elem(arr, i - 1)) - 0.5
      free = actionable.(p) and (p.action || 0) in [20, 21]
      cond do
        closing and free and not in_run -> {acc + 1, true}
        closing and free -> {acc, true}
        true -> {acc, false}
      end
    end)

  # action-entry counts (transitions into the state)
  entries = fn pred ->
    Enum.count(1..(n - 1)//1, fn i ->
      a = elem(arr, i).players[port].action || 0
      b = elem(arr, i - 1).players[port].action || 0
      pred.(a) and not pred.(b)
    end)
  end

  lasers = entries.(&MapSet.member?(laser_ids, &1))
  j1 = entries.(&(&1 == jab1))
  j2 = entries.(&(&1 == jab2))
  j3 = entries.(&(&1 == jab3))

  # A press-edges while in jab1 (the re-press that makes jab2)
  # subject controller per frame (Peppi causal pairing: the input produced on that frame)
  a_down = fn i -> case elem(ctrl, i) do %{button_a: v} -> v == true; _ -> false end end
  a_edges_in_jab1 =
    Enum.count(1..(n - 1)//1, fn i ->
      s = elem(arr, i)
      (s.players[port].action || 0) == jab1 and a_down.(i) and not a_down.(i - 1)
    end)

  # time to first hit per subject stock: frames from a stock's first frame
  # until the opponent's percent first rises
  {ttfh, _, _} =
    Enum.reduce(1..(n - 1)//1, {[], elem(arr, 0).players[port].stock, 0}, fn i, {acc, stock, start} ->
      s = elem(arr, i)
      cur = s.players[port].stock
      cond do
        cur != stock -> {acc, cur, i}
        start != nil and s.players[opp].percent > elem(arr, i - 1).players[opp].percent ->
          {[i - start | acc], stock, nil}
        true -> {acc, stock, start}
      end
    end)

  %{
    frames: n,
    median_distance: Float.round(median_dist * 1.0, 1),
    far_frac: Float.round(far_frac, 3),
    approaches_per_min: Float.round(approaches / minutes, 2),
    opp_approaches_per_min: Float.round(opp_approaches / minutes, 2),
    initiative_share: Float.round(approaches / max(1, approaches + opp_approaches), 3),
    lasers_per_min: Float.round(lasers / minutes, 2),
    jab1_per_min: Float.round(j1 / minutes, 2),
    jab2_per_min: Float.round(j2 / minutes, 2),
    jab3_per_min: Float.round(j3 / minutes, 2),
    jab2_per_jab1: if(j1 > 0, do: Float.round(j2 / j1, 3), else: nil),
    a_repress_per_jab1: if(j1 > 0, do: Float.round(a_edges_in_jab1 / j1, 3), else: nil),
    median_frames_to_first_hit: if(ttfh == [], do: nil, else: Enum.at(Enum.sort(ttfh), div(length(ttfh), 2)))
  }
end

Output.banner("Engagement scan: #{label}")
Output.puts("#{length(files)} files")

rows =
  files
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {path, i} ->
    with {:ok, meta} <- Peppi.metadata(path),
         port when is_integer(port) <-
           (case opts[:character] do
              nil -> opts[:port] || 1
              c -> (Enum.find(meta.players, &(String.downcase(&1.character_name || "") == String.downcase(c))) || %{port: nil}).port
            end),
         opp when is_integer(opp) <- (Enum.find(meta.players, &(&1.port != port)) || %{port: nil}).port,
         {:ok, replay} <- Peppi.parse(path, player_port: port),
         frames when length(frames) > 600 <-
           replay
           |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
           |> Enum.reject(&(&1.game_state.frame < 0)) do
      states = Enum.map(frames, & &1.game_state)
      controllers = Enum.map(frames, & &1.controller)
      subj = Enum.find(meta.players, &(&1.port == port))
      oppp = Enum.find(meta.players, &(&1.port == opp))
      row = %{label: label, path: path, port: port, character: subj && subj.character_name,
              opponent: oppp && oppp.character_name, stage: meta.stage}
      row = Map.merge(row, features.(states, controllers, port, opp))
      if rem(i, 10) == 0, do: Output.puts("  #{i}/#{length(files)}")
      [row]
    else
      other ->
        Output.warning("skip #{Path.basename(path)}: #{inspect(other) |> String.slice(0, 80)}")
        []
    end
  end)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Enum.map_join(rows, "", &(Jason.encode!(&1) <> "\n")))
  Output.success("#{length(rows)} rows -> #{out}")
end

med = fn key ->
  vals = rows |> Enum.map(&Map.get(&1, key)) |> Enum.reject(&is_nil/1) |> Enum.sort()
  if vals == [], do: "-", else: Enum.at(vals, div(length(vals), 2))
end

Output.puts("median over #{length(rows)} games (#{label}):")
for k <- [:median_distance, :far_frac, :approaches_per_min, :opp_approaches_per_min, :initiative_share,
          :lasers_per_min, :jab1_per_min, :jab2_per_min, :jab3_per_min, :jab2_per_jab1,
          :a_repress_per_jab1, :median_frames_to_first_hit] do
  Output.puts("  #{String.pad_trailing(to_string(k), 28)} #{med.(k)}")
end
Output.puts("opponents: #{rows |> Enum.map(& &1.opponent) |> Enum.frequencies() |> inspect()}")
