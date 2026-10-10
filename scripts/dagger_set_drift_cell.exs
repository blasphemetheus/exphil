# Drift-cell coverage of a DAgger set (2026-10-10, labeler v5 `:air` gate):
# frames airborne over the stage with the double jump spent, by distance
# inside the edge — how many, how many labelled (the lever's dose), the bot's
# own stick outward share there (`:actual`, sets rolled with --keep-actual)
# and the label's. Also the on-stage double-jump spends by band (labelled?).
# Run on the UNSPLIT set (the split duplicates prefixes).
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_drift_cell.exs data/silent_fall/sim_dagger_expert_r8a.frames
path = List.first(System.argv()) || raise("usage: dagger_set_drift_cell.exs SET.frames")
m = File.read!(path) |> :erlang.binary_to_term()
edge = 85.57
bands = ["-15..0", "-30..-15", "-50..-30", "-80..-50", "<-80"]
band = fn d -> cond do d > -15 -> "-15..0"; d > -30 -> "-30..-15"; d > -50 -> "-50..-30"; d > -80 -> "-80..-50"; true -> "<-80" end end
fr = List.flatten(m.frame_lists)
pct = fn a, b -> if b == 0, do: "-", else: :erlang.float_to_binary(100 * a / b, decimals: 1) <> " %" end
out? = fn f, c -> p = f.game_state.players[1]; s = if p.x >= 0, do: 1, else: -1; (c.main_stick.x - 0.5) * 2 * s >= 0.6 end

cell = Enum.filter(fr, fn f -> p = f.game_state.players[1]; p.on_ground != true and abs(p.x) < edge and (p.jumps_left || 0) == 0 and (p.hitstun_frames_left || 0) == 0 and (p.action || 0) > 13 and p.action != 35 end)
IO.puts("#{Path.basename(path)}: #{length(fr)} frames, #{length(m.frame_lists)} lists; airborne over the stage, no jump, actionable: #{length(cell)}")
for k <- bands do
  xs = Enum.filter(cell, fn f -> band.(abs(f.game_state.players[1].x) - edge) == k end)
  lab = Enum.filter(xs, &(&1[:input_only] != true))
  bot_out = Enum.count(xs, fn f -> out?.(f, f[:actual] || f.controller) end)
  lab_out = Enum.count(lab, fn f -> out?.(f, f.controller) end)
  IO.puts("  #{String.pad_trailing(k, 9)} n=#{String.pad_leading(to_string(length(xs)), 6)}  labelled #{String.pad_leading(to_string(length(lab)), 6)} (#{pct.(length(lab), length(xs))})  bot stick outward >=0.6 #{pct.(bot_out, length(xs))}  label outward #{pct.(lab_out, length(lab))}")
end
drift = Enum.filter(cell, fn f -> d = abs(f.game_state.players[1].x) - edge; d > -80 and d <= -15 end)
dl = Enum.count(drift, &(&1[:input_only] != true))
IO.puts("RESULT drift cell -80..-15: #{length(drift)} frames, labelled #{dl} (#{pct.(dl, length(drift))})  of #{Enum.count(fr, &(&1[:input_only] != true))} labelled frames in the set")

# on-stage double-jump spends (jumps 1 -> 0 while airborne over the stage), by band
pairs = Enum.flat_map(m.frame_lists, fn l -> Enum.chunk_every(l, 2, 1, :discard) end)
dj = Enum.filter(pairs, fn [a, b] -> pa = a.game_state.players[1]; pb = b.game_state.players[1]; pa.on_ground != true and abs(pa.x) < edge and (pa.jumps_left || 0) == 1 and (pb.jumps_left || 0) == 0 and a.game_state.frame + 1 == b.game_state.frame and (pa.hitstun_frames_left || 0) == 0 end)
IO.puts("on-stage double-jump spends: #{length(dj)}; by band (n / labelled / bot stick outward at the spend):")
for k <- bands do
  xs = Enum.filter(dj, fn [a, _] -> band.(abs(a.game_state.players[1].x) - edge) == k end)
  lab = Enum.count(xs, fn [a, _] -> a[:input_only] != true end)
  tow = Enum.count(xs, fn [a, b] -> out?.(a, b[:actual] || b.controller) end)
  IO.puts("  #{String.pad_trailing(k, 9)} #{String.pad_leading(to_string(length(xs)), 5)} / #{String.pad_leading(to_string(lab), 5)} / #{pct.(tow, length(xs))}")
end
