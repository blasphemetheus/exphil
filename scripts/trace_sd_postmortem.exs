# SD post-mortem: for each of port 1's deaths, characterize the walk-off —
# where was the opponent when P1 crossed the edge (edgeguard chase vs
# pathological run-off), what was P1 doing before, and did it try to recover?
#
#   mix run scripts/trace_sd_postmortem.exs replay.slp [...]

paths = System.argv()

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/trace_sd_postmortem.exs <replay.slp> [...]")
  System.halt(1)
end

# FD edge
edge_x = 85.57

for path <- paths do
  {:ok, replay} = ExPhil.Data.Peppi.parse(path)

  frames =
    replay
    |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))

  rows =
    Enum.map(frames, fn f ->
      p1 = f.game_state.players[1]
      p2 = f.game_state.players[2]

      %{
        frame: f.game_state.frame,
        stock: (p1 && p1.stock) || 0,
        x: (p1 && p1.x) || 0.0,
        y: (p1 && p1.y) || 0.0,
        action: trunc((p1 && p1.action) || 0),
        facing: (p1 && p1.facing) || 0,
        opp_x: (p2 && p2.x) || 0.0,
        opp_offstage: p2 != nil and (abs(p2.x) > edge_x or p2.y < -5),
        stick_x: f.controller.main_stick.x,
        jump: f.controller.button_x or f.controller.button_y,
        b: f.controller.button_b
      }
    end)

  # Death = stock decrement
  deaths =
    rows
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [a, b] -> b.stock < a.stock and a.stock > 0 end)
    |> Enum.map(fn [a, _b] -> a.frame end)

  IO.puts("== #{Path.basename(path)}: #{length(rows)} frames, #{length(deaths)} deaths of P1 ==")

  for death_frame <- deaths do
    # The window leading to death
    window = Enum.filter(rows, &(&1.frame >= death_frame - 240 and &1.frame <= death_frame))

    # First frame past the edge in this window (the walk-off moment), if any
    crossing = Enum.find(window, &(abs(&1.x) > edge_x))

    case crossing do
      nil ->
        # Died without leaving via the side (e.g. star KO from a hit)
        last = List.last(window)
        IO.puts(
          "  death@#{death_frame}: no edge crossing in last 4s (killed off top/hit?) " <>
            "last pos=(#{Float.round(last.x, 1)}, #{Float.round(last.y, 1)})"
        )

      c ->
        after_cross = Enum.filter(window, &(&1.frame > c.frame))
        n_after = max(length(after_cross), 1)
        # Recovery effort: jump/B presses and inward stick after crossing
        jumps = Enum.count(after_cross, & &1.jump)
        bs = Enum.count(after_cross, & &1.b)

        inward =
          Enum.count(after_cross, fn r ->
            (r.x > 0 and r.stick_x < 0.4) or (r.x < 0 and r.stick_x > 0.6)
          end)

        # What was it doing 1-2s before the crossing?
        before = Enum.filter(window, &(&1.frame < c.frame and &1.frame >= c.frame - 90))

        run_actions =
          before
          |> Enum.frequencies_by(& &1.action)
          |> Enum.sort_by(fn {_, n} -> -n end)
          |> Enum.take(3)

        IO.puts(
          "  death@#{death_frame}: crossed edge at x=#{Float.round(c.x, 1)} " <>
            "opp_x=#{Float.round(c.opp_x, 1)} opp_offstage=#{c.opp_offstage} | " <>
            "pre-cross actions: #{inspect(run_actions)} | " <>
            "recovery: jumps=#{jumps} B=#{bs} inward_stick=#{inward}/#{n_after}f"
        )
    end
  end

  # Shield stats for the shield question
  shield_frames = Enum.count(rows, &(&1.action in [178, 179, 180]))
  IO.puts("  shield frames: #{shield_frames}/#{length(rows)} (#{Float.round(100.0 * shield_frames / max(length(rows), 1), 1)}%)")
end
