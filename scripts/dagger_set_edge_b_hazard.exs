# Closed-loop B hazard at the Illusion state, on the bot's OWN states
# (2026-10-10 14:00; needs a set rolled with --keep-actual). State S =
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

toward = fn c, p ->
  ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
  x = (ms[:x] || 0.5) - 0.5
  sign = if (p.x || 0.0) >= 0, do: 1, else: -1
  cond do
    abs(x) < 0.165 -> :x0
    x * sign > 0 -> :toward_edge
    true -> :toward_center
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
      # the deadly joint: B with the stick (bot's next input) toward the edge
      bot_illusion_out: cur.actual.button_b == true and toward.(cur.actual, p) == :toward_edge,
      label_illusion_out: labelled and cur.controller.button_b == true and toward.(cur.controller, p) == :toward_edge
    }
  end

pct = fn n, d -> if d == 0, do: "-", else: :io_lib.format("~.4f", [n / d]) |> to_string() end
IO.puts("RESULT edge B hazard #{Path.basename(path)}: state frames #{length(rows)} (airborne, -30..+10 of the edge, facing out, B up at t-1)")
IO.puts("  cell (prev stick, jumps): frames | bot P(B) | bot P(B & stick->edge) | labelled | label P(B) | label P(B & stick->edge) | gated share | bot P(jump) | label P(jump)")

for stick <- [:toward_edge, :x0, :toward_center], j <- [1, 0] do
  cell = Enum.filter(rows, &(&1.stick == stick and &1.jumps == j))
  n = length(cell)
  lab = Enum.filter(cell, & &1.labelled)
  IO.puts("  #{String.pad_trailing("#{stick} j#{j}", 18)} #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(cell, & &1.bot_b), n)} | #{pct.(Enum.count(cell, & &1.bot_illusion_out), n)} | #{String.pad_leading("#{length(lab)}", 6)} | #{pct.(Enum.count(lab, & &1.label_b), length(lab))} | #{pct.(Enum.count(lab, & &1.label_illusion_out), length(lab))} | #{pct.(Enum.count(cell, & &1.gated), n)} | #{pct.(Enum.count(cell, & &1.bot_jump), n)} | #{pct.(Enum.count(lab, & &1.label_jump), length(lab))}")
end

n = length(rows)
lab = Enum.filter(rows, & &1.labelled)
IO.puts("  all                #{String.pad_leading("#{n}", 6)} | #{pct.(Enum.count(rows, & &1.bot_b), n)} | #{pct.(Enum.count(rows, & &1.bot_illusion_out), n)} | #{String.pad_leading("#{length(lab)}", 6)} | #{pct.(Enum.count(lab, & &1.label_b), length(lab))} | #{pct.(Enum.count(lab, & &1.label_illusion_out), length(lab))} | #{pct.(Enum.count(rows, & &1.gated), n)} | #{pct.(Enum.count(rows, & &1.bot_jump), n)} | #{pct.(Enum.count(lab, & &1.label_jump), length(lab))}")
