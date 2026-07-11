# Tech-chase scoreboard: how many opponent knockdowns did port 1 convert
# into a punish within the chase window?
#
#   mix run scripts/trace_tech_chase.exs replay.slp [...]
#
# Knockdown = opponent entering down-bounce/tech states (183/191/199-201).
# Conversion = opponent entering hitstun/grabbed states (75-91, 223-232)
# within 90 frames of the knockdown resolving.

paths = System.argv()

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/trace_tech_chase.exs <replay.slp> [...]")
  System.halt(1)
end

knockdown_entries = [183, 191, 199, 200, 201]
punish_states = Enum.to_list(75..91) ++ Enum.to_list(223..232)

for path <- paths do
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)

  opp_actions =
    replay
    |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(fn f -> trunc((f.game_state.players[2] && f.game_state.players[2].action) || 0) end)

  frames = length(opp_actions)

  # Knockdown events: transitions INTO a knockdown state from outside the
  # whole lifecycle (rolls chain into getups — count the lifecycle once)
  lifecycle = MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])

  knockdowns =
    opp_actions
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.filter(fn {[a, b], _i} ->
      b in knockdown_entries and not MapSet.member?(lifecycle, a)
    end)
    |> Enum.map(fn {_pair, i} -> i + 1 end)

  conversions =
    Enum.count(knockdowns, fn kd_idx ->
      # Find where the lifecycle ends after this knockdown, then look for a
      # punish within 90 frames of that
      rest = Enum.drop(opp_actions, kd_idx)

      exit_offset =
        Enum.find_index(rest, fn a -> not MapSet.member?(lifecycle, a) end) || length(rest)

      rest
      |> Enum.drop(exit_offset)
      |> Enum.take(90)
      |> Enum.any?(fn a -> a in punish_states end)
    end)

  rate =
    if knockdowns == [], do: 0.0, else: Float.round(100.0 * conversions / length(knockdowns), 1)

  IO.puts(
    "#{Path.basename(path)}: #{frames}f | knockdowns=#{length(knockdowns)} " <>
      "conversions=#{conversions} (#{rate}%)"
  )

  File.mkdir_p!("logs")

  row =
    Jason.encode!(%{
      drill: "mewtwo_techchase",
      replay: Path.basename(path),
      scored_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      frames: frames,
      knockdowns: length(knockdowns),
      conversions: conversions,
      conversion_pct: rate
    })

  File.write!("logs/drill_scores.jsonl", row <> "\n", [:append])
end
