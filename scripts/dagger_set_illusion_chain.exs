# The on-stage Illusion chain in a DAgger set rolled with --keep-actual
# (2026-10-10 13:40, INPUT_COHERENCE "13:15"): for every frame where the
# bot's own input pressed B (edge vs its previous own input) near the edge,
# facing out, airborne (the carried side-B precursor), report
#   * whether that frame is labelled or gated (no expert row within the
#     coverage gate -> the press is never supervised), and the label's
#     stick zone / buttons when labelled (what the expert would do instead);
#   * walking back up to 30 frames in the same run: the FIRST frame where a
#     label disagrees with the bot's input (button edge or stick zone), and
#     what kind — the failure state before the Illusion.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_illusion_chain.exs SET.frames
[path] = System.argv()
set = path |> File.read!() |> :erlang.binary_to_term()
edge = ExPhil.Sim.GA.stage_edge(32)

zone = fn c ->
  ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
  x = (ms[:x] || 0.5) - 0.5
  y = (ms[:y] || 0.5) - 0.5
  cond do
    abs(x) < 0.165 and abs(y) < 0.165 -> :neutral
    abs(y) >= abs(x) and y > 0 -> :up
    abs(y) >= abs(x) -> :down
    true -> :side
  end
end

# stick x sign relative to the edge the player is near (+ = toward it)
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

buttons = [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]
pressed = fn c -> Enum.filter(buttons, &Map.get(c, &1)) end

precursor? = fn f ->
  p = f.game_state.players[1]
  sign = if (p.x || 0.0) >= 0, do: 1, else: -1
  d = abs(p.x || 0.0) - edge
  d > -30 and d < 10 and (p.facing || 1) * sign > 0 and p.on_ground != true
end

b_edge? = fn prev, cur -> cur[:actual] != nil and prev[:actual] != nil and cur.actual.button_b and not prev.actual.button_b end

labelled? = fn f -> f[:input_only] != true end

disagree = fn f ->
  # label vs the bot's own input on a labelled frame
  if labelled?.(f) and f[:actual] do
    c = f.controller
    a = f.actual
    p = f.game_state.players[1]
    cond do
      Enum.any?([:button_x, :button_y], &(Map.get(a, &1) and not Map.get(c, &1))) -> :jump_pressed_label_holds
      Map.get(a, :button_b) and not Map.get(c, :button_b) -> :b_pressed_label_holds
      toward.(a, p) == :toward_edge and toward.(c, p) != :toward_edge -> {:stick_edge_label, toward.(c, p)}
      zone.(a) != zone.(c) -> {:zone, zone.(a), zone.(c)}
      true -> nil
    end
  end
end

rows =
  for frames <- set.frame_lists, {prev, cur, i} <- (frames |> Enum.chunk_every(2, 1, :discard) |> Enum.with_index() |> Enum.map(fn {[a, b], i} -> {a, b, i + 1} end)),
      cur.game_state.frame == prev.game_state.frame + 1, precursor?.(cur), b_edge?.(prev, cur) do
    p = cur.game_state.players[1]
    back = frames |> Enum.slice(max(i - 30, 0)..(i - 1)) |> Enum.reverse()
    first_dis = Enum.find_value(back, fn f -> d = disagree.(f); d && {f.game_state.frame, d} end)
    lab_back = Enum.count(back, labelled?)
    gated_back = Enum.count(back, &(&1[:gated] == true))
    %{
      labelled: labelled?.(cur),
      gated: cur[:gated] == true,
      status: cond do
        labelled?.(cur) -> :labelled
        cur[:gated] == true -> :gated
        true -> :prefix
      end,
      gated_back: gated_back,
      label_zone: if(labelled?.(cur), do: zone.(cur.controller)),
      label_toward: if(labelled?.(cur), do: toward.(cur.controller, p)),
      label_buttons: if(labelled?.(cur), do: pressed.(cur.controller)),
      actual_toward: toward.(cur.actual, p),
      action: p.action, jumps: p.jumps_left, y: Float.round((p.y || 0.0) * 1.0, 1),
      d: Float.round(abs(p.x || 0.0) - edge, 1),
      frames_since_dis: first_dis && cur.game_state.frame - elem(first_dis, 0),
      dis: first_dis && elem(first_dis, 1),
      labelled_back: lab_back
    }
  end

all_rows = rows
tally = fn list, f -> list |> Enum.map(f) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.map(fn {k, v} -> "#{inspect(k)} #{v}" end) |> Enum.join(", ") end
q = fn l, p -> if l == [], do: "-", else: Enum.at(l, min(trunc(p * length(l)), length(l) - 1)) end
IO.puts("RESULT illusion chain #{Path.basename(path)}: airborne near-edge facing-out B presses by the bot: #{length(all_rows)}  by stick: #{tally.(all_rows, & &1.actual_toward)}")

for {name, rows} <- [{"ALL", all_rows}, {"STICK TOWARD EDGE (the Illusion off the stage)", Enum.filter(all_rows, &(&1.actual_toward == :toward_edge))}] do
n = length(rows)
IO.puts("== #{name}: n=#{n}")
lab = Enum.filter(rows, & &1.labelled)
IO.puts("  press frame: #{tally.(rows, & &1.status)}  | in the 30 before: labelled median #{q.(Enum.sort(Enum.map(rows, & &1.labelled_back)), 0.5)}, gated median #{q.(Enum.sort(Enum.map(rows, & &1.gated_back)), 0.5)}, prefix median #{q.(Enum.sort(Enum.map(rows, &(30 - &1.labelled_back - &1.gated_back))), 0.5)}")
IO.puts("  bot stick at the press: #{tally.(rows, & &1.actual_toward)}  action: #{tally.(rows, & &1.action)}  jumps: #{tally.(rows, & &1.jumps)}")
IO.puts("  label at the press (what the expert does instead): zone #{tally.(lab, & &1.label_zone)}  x: #{tally.(lab, & &1.label_toward)}  buttons: #{tally.(lab, & &1.label_buttons)}")
IO.puts("  first label disagreement in the 30 frames before: #{tally.(rows, & &1.dis)}")
fs = rows |> Enum.map(& &1.frames_since_dis) |> Enum.reject(&is_nil/1) |> Enum.sort()
IO.puts("  frames from that first disagreement to the B press q25/50/75: #{q.(fs, 0.25)}/#{q.(fs, 0.5)}/#{q.(fs, 0.75)}  (none within 30 f: #{Enum.count(rows, &is_nil(&1.dis))})")
IO.puts("  y at press q50 #{q.(Enum.sort(Enum.map(rows, & &1.y)), 0.5)}  d past edge q50 #{q.(Enum.sort(Enum.map(rows, & &1.d)), 0.5)}")
end
