# Closed-loop B hazard at the Illusion state, on the bot's OWN states
# (2026-10-10 13:25; needs a set rolled with --keep-actual). State S =
# airborne, within -30..+10 of the edge, facing out, B released on the bot's
# previous own input; split by the bot's previous stick (toward the edge /
# centred / toward centre) and by jumps left. Per cell: frames, the bot's B
# press rate (its own next input), the label's B rate on the labelled frames
# of the cell, the gated share. Teacher-forced on the expert's states the
# model's P(B) is the expert's (DecisionMap Q1); this is the same number on
# the states the bot actually reaches.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_edge_b_hazard.exs SET.frames
[path] = System.argv()
set = path |> File.read!() |> :erlang.binary_to_term()
edge = ExPhil.Sim.GA.stage_edge(32)

# stick x in -1..1 units, signed toward the near edge (+ = outward). The sim
# fires a side special at |x| >= ~0.6 (measured 10-10 on the bot's own presses:
# 10 / 13 Illusions at 0.6-0.7, 266 / 360 at >= 0.8, lasers / shines below),
# so the Illusion-capable cell is >= 0.6, not the probe's 0.33 deadzone.
sideb = 0.6
sx = fn c, p ->
  ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
  sign = if (p.x || 0.0) >= 0, do: 1, else: -1
  ((ms[:x] || 0.5) - 0.5) * 2 * sign
end
toward = fn c, p ->
  x = sx.(c, p)
  cond do
    x >= sideb -> :toward_edge
    x <= -sideb -> :toward_center
    abs(x) < 0.33 -> :x0
    true -> :tilt
  end
end

state? = fn f ->
  p = f.game_state.players[1]
  sign = if (p.x || 0.0) >= 0, do: 1, else: -1
  d = abs(p.x || 0.0) - edge
  d > -30 and d < 10 and (p.facing || 1) * sign > 0 and p.on_ground != true
end

rows =
  for frames <- set.frame_lists, [prev, cur] <- Enum.chunk_every(frames, 2, 1, :discard),
      cur.game_state.frame == prev.game_state.frame + 1, prev[:actual] != nil, cur[:actual] != nil,
      state?.(cur), not prev.actual.button_b do
    p = cur.game_state.players[1]
    labelled = cur[:input_only] != true
    %{
      stick: toward.(prev.actual, p),
      jumps: min(p.jumps_left || 0, 1),
      bot_b: cur.actual.button_b == true,
      labelled: labelled,
      gated: cur[:gated] == true,
      label_b: labelled and cur.controller.button_b == true,
      bot_jump: (cur.actual.button_x or cur.actual.button_y) and not (prev.actual.button_x or prev.actual.button_y),
      label_jump: labelled and (cur.controller.button_x or cur.controller.button_y) and not (prev.actual.button_x or prev.actual.button_y),
      # the deadly joint: B with the stick (bot's next input) outward past the side-B threshold
      bot_illusion_out: cur.actual.button_b == true and sx.(cur.actual, p) >= sideb,
      label_illusion_out: labelled and cur.controller.button_b == true and sx.(cur.controller, p) >= sideb
    }
  end

pct = fn n, d -> if d == 0, do: "-", else: :io_lib.format("~.4f", [n / d]) |> to_string() end
IO.puts("RESULT edge B hazard #{Path.basename(path)}: state frames #{length(rows)} (airborne, -30..+10 of the edge, facing out, B up at t-1)")
IO.puts("  cell (prev stick, jumps): frames | bot P(B) | bot P(B & stick->edge) | labelled | label P(B) | label P(B & stick->edge) | gated share | bot P(jump) | label P(jump)")

for stick <- [:toward_edge, :tilt, :x0, :toward_center], j <- [1, 0] do
  cell = Enum.filter(rows, &(&1.stick == stick and &1.jumps == j))
  n = length(cell)
  lab = Enum.filter(cell, & &1.labelled)
  IO.puts("  #{String.pad_trailing("#{stick} j#{j}", 18)} #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(cell, & &1.bot_b), n)} | #{pct.(Enum.count(cell, & &1.bot_illusion_out), n)} | #{String.pad_leading("#{length(lab)}", 6)} | #{pct.(Enum.count(lab, & &1.label_b), length(lab))} | #{pct.(Enum.count(lab, & &1.label_illusion_out), length(lab))} | #{pct.(Enum.count(cell, & &1.gated), n)} | #{pct.(Enum.count(cell, & &1.bot_jump), n)} | #{pct.(Enum.count(lab, & &1.label_jump), length(lab))}")
end

n = length(rows)
lab = Enum.filter(rows, & &1.labelled)
IO.puts("  all                #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(rows, & &1.bot_b), n)} | #{pct.(Enum.count(rows, & &1.bot_illusion_out), n)} | #{String.pad_leading("#{length(lab)}", 6)} | #{pct.(Enum.count(lab, & &1.label_b), length(lab))} | #{pct.(Enum.count(lab, & &1.label_illusion_out), length(lab))} | #{pct.(Enum.count(rows, & &1.gated), n)} | #{pct.(Enum.count(rows, & &1.bot_jump), n)} | #{pct.(Enum.count(lab, & &1.label_jump), length(lab))}")
