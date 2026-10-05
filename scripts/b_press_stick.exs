# Offstage B presses: where was the stick on the press frame? (2026-10-05)
#
# A special's identity is decided by the main stick on the frame B goes down
# (up-B needs y high enough on THAT frame; a stick that arrives a frame later
# gives a laser or an illusion). The recovery-means scorecard showed the bots
# dying from ledge height after one move with almost no up-B in the sequence,
# so: for every B press while offstage, record the stick on the press frame
# and the next three, the special entered within 4 frames, and the height.
#
#   mix run scripts/b_press_stick.exs --label L [--bot-port 1] GAME.slp ...        # bot games
#   mix run scripts/b_press_stick.exs --label expert --expert [--max-games 150]    # expert Fox on FD
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, files, bad} =
  OptionParser.parse(System.argv(),
    strict: [label: :string, bot_port: :integer, expert: :boolean, max_games: :integer, split: :string, min_bytes: :integer, out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
label = opts[:label] || "b_press"
edge = GA.stage_edge(32)

games =
  if opts[:expert] do
    split = (opts[:split] || "checkpoints/coh_base/split.json") |> File.read!() |> Jason.decode!()
    (split["validation"] ++ split["train"])
    |> Task.async_stream(fn p -> with {:ok, m} <- Peppi.metadata(p), do: {p, m} end, max_concurrency: 16, timeout: 60_000)
    |> Enum.flat_map(fn
      {:ok, {p, %{stage: 32, players: [_, _] = ps}}} ->
        case Enum.filter(ps, &(String.downcase(&1.character_name || "") == "fox")) do
          [own] -> [{p, own.port, Enum.find(ps, &(&1.port != own.port)).port}]
          _ -> []
        end
      _ -> []
    end)
    |> Enum.take(opts[:max_games] || 150)
  else
    bot_port = opts[:bot_port] || 1
    files
    |> Enum.filter(&(File.stat!(&1).size >= (opts[:min_bytes] || 150_000)))
    |> Enum.flat_map(fn path ->
      {:ok, meta} = Peppi.metadata(path)
      own = Enum.find(meta.players, &(&1.port == bot_port))
      opp = Enum.find(meta.players, &(&1.port != bot_port))
      if meta.stage == 32 and own && opp, do: [{path, own.port, opp.port}], else: []
    end)
  end

Output.banner("Offstage B presses: #{label} (#{length(games)} games)")

special = fn a ->
  cond do
    a in 350..352 -> :side_b
    a in 353..356 -> :up_b
    a in 341..348 -> :laser
    a in 360..368 -> :shine
    true -> nil
  end
end

presses =
  games
  |> Task.async_stream(fn {path, own_port, opp_port} ->
    {:ok, replay} = Peppi.parse(path, player_port: own_port)
    frames =
      replay
      |> Peppi.to_training_frames(player_port: own_port, opponent_port: opp_port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&{&1.game_state.players[own_port], &1.controller})
      |> List.to_tuple()
    n = tuple_size(frames)

    for i <- 4..(n - 5),
        {p, c} = elem(frames, i),
        {_, c0} = elem(frames, i - 1),
        c.button_b and not c0.button_b,
        not p.on_ground, (p.action || 99) > 13, (p.hitstun_frames_left || 0) == 0,
        abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0 do
      sticks = for j <- -3..3, do: (elem(frames, i + j) |> elem(1) |> then(&{Float.round(&1.main_stick.x - 0.5, 2), Float.round(&1.main_stick.y - 0.5, 2)}))
      result = Enum.find_value(0..4, fn j -> special.(elem(elem(frames, i + j), 0).action || 0) end)
      toward = if (p.facing || 1) * (p.x || 0.0) < 0, do: :facing_stage, else: :facing_out
      %{game: Path.basename(path), frame: i, y: Float.round((p.y || 0.0) * 1.0, 1), jumps: p.jumps_left || 0,
        sticks: sticks, result: result || :none, facing: toward}
    end
  end, max_concurrency: 8, timeout: 300_000)
  |> Enum.flat_map(fn {:ok, ps} -> ps end)

height = fn y -> cond do y > 0.0 -> :high; y > -20.0 -> :ledge; y > -60.0 -> :low; true -> :deep end end
# stick zone on the press frame: up if y >= 0.33 (≈ the up-special threshold on a 0..1 stick), side if |x| >= 0.33
zone = fn {x, y} -> cond do y >= 0.33 -> :up; y <= -0.33 -> :down; abs(x) >= 0.33 -> :side; true -> :neutral end end
# sticks = frames -3..3 around the press; index 3 is the press frame
at_press = fn s -> Enum.at(s, 3) end
# did the stick reach UP within 3 frames after a press that was not up on the press frame?
late_up = fn s -> zone.(at_press.(s)) != :up and Enum.any?(Enum.drop(s, 4), &(zone.(&1) == :up)) end
# for presses that are UP on the press frame: how many frames before the press was the stick already up?
lead_up = fn s -> Enum.take(s, 3) |> Enum.reverse() |> Enum.take_while(&(zone.(&1) == :up)) |> length() end

pct = fn n, d -> if d == 0, do: "-", else: "#{Float.round(100 * n / d, 1)}%" end
by = fn list, key -> list |> Enum.frequencies_by(key) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.map_join("  ", fn {k, v} -> "#{k} #{v} (#{pct.(v, length(list))})" end) end

Output.puts("RESULT #{label}: #{length(presses)} offstage B presses")
Output.puts("RESULT #{label} result: " <> by.(presses, & &1.result))
Output.puts("RESULT #{label} stick zone on press frame: " <> by.(presses, &zone.(at_press.(&1.sticks))))
ups = Enum.filter(presses, &(zone.(at_press.(&1.sticks)) == :up))
Output.puts("RESULT #{label} UP presses (#{length(ups)}): frames the stick was already up before the press: " <> by.(ups, &lead_up.(&1.sticks)) <>
  "  (0 = stick arrives ON the press frame)")
Output.puts("RESULT #{label} late-up (stick not up on press, up within 3 f): #{pct.(Enum.count(presses, &late_up.(&1.sticks)), length(presses))}")
for h <- [:high, :ledge, :low, :deep] do
  ps = Enum.filter(presses, &(height.(&1.y) == h))
  Output.puts("RESULT #{label} #{h} (#{length(ps)}): result " <> by.(ps, & &1.result) <> "  | zone " <> by.(ps, &zone.(at_press.(&1.sticks))) <>
    "  | late-up #{pct.(Enum.count(ps, &late_up.(&1.sticks)), length(ps))}  | facing " <> by.(ps, & &1.facing))
end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{label: label, presses: Enum.map(presses, &%{&1 | sticks: Enum.map(&1.sticks, fn {x, y} -> [x, y] end)})}, pretty: true))
end
