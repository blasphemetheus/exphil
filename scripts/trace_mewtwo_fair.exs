# One-line scoreboard for the Mewtwo fair-chain drill: how many fairs came
# out, and what fraction of fair landings were L-cancelled?
#
#   mix run scripts/trace_mewtwo_fair.exs path/to/Game.slp [more.slp ...]
#
# L-cancel detection: fair landing lag is action 71 (LANDING_AIR_F); an
# L-cancel halves the lag, so short runs of consecutive 71 frames (<= 15)
# count as cancelled.

paths = System.argv()

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/trace_mewtwo_fair.exs <replay.slp> [...]")
  System.halt(1)
end

for path <- paths do
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)

  actions =
    replay
    |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(fn f -> trunc(f.game_state.players[1].action) end)

  pairs = Enum.chunk_every(actions, 2, 1, :discard)

  fairs = Enum.count(pairs, fn [a, b] -> a != 66 and b == 66 end)
  jumps = Enum.count(pairs, fn [a, b] -> a != 24 and b == 24 end)

  # Aerial mix: which attack comes out is decided by stick position at the
  # press frame — wrong-aerial share measures stick accuracy, not intent
  aerial_names = %{65 => "nair", 66 => "fair", 67 => "bair", 68 => "uair", 69 => "dair"}

  aerial_mix =
    aerial_names
    |> Enum.map(fn {id, name} ->
      n = Enum.count(pairs, fn [a, b] -> a != id and b == id end)
      {name, n}
    end)
    |> Enum.reject(fn {_, n} -> n == 0 end)
    |> Enum.map(fn {name, n} -> "#{name}=#{n}" end)
    |> Enum.join(" ")

  landing_runs =
    actions
    |> Enum.chunk_by(& &1)
    |> Enum.filter(fn [a | _] -> a == 71 end)
    |> Enum.map(&length/1)

  landings = length(landing_runs)
  cancelled = Enum.count(landing_runs, &(&1 <= 15))

  lcancel_pct =
    if landings > 0, do: Float.round(100.0 * cancelled / landings, 1), else: 0.0

  frames = length(actions)
  rate = if frames > 0, do: Float.round(fairs * 1000 / frames, 1), else: 0.0

  IO.puts(
    "#{Path.basename(path)}: #{frames}f | jumps=#{jumps} fairs=#{fairs} " <>
      "fair_landings=#{landings} L-cancelled=#{cancelled} (#{lcancel_pct}%) | " <>
      "#{rate} fairs/1k frames | aerials: #{aerial_mix}"
  )

  # Structured row for cross-iteration plots (logs/drill_scores.jsonl)
  File.mkdir_p!("logs")

  row =
    Jason.encode!(%{
      drill: "mewtwo_fair",
      replay: Path.basename(path),
      scored_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      frames: frames,
      jumps: jumps,
      fairs: fairs,
      fair_landings: landings,
      lcancelled: cancelled,
      lcancel_pct: lcancel_pct,
      fairs_per_1k: rate
    })

  File.write!("logs/drill_scores.jsonl", row <> "\n", [:append])
end
