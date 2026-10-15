# Onstage-cell dose + label content of a DAgger set rolled with --keep-actual
# (2026-10-15, labeler v6 `:onstage` gate — the "count the dose before
# launching" check, feedback 10-10). Per cell (offstage / near-edge / drift /
# shield / lying / knockdown-other / threat): frames in the set, how many got
# a label, and the bot's own edge rate vs the label's for the inputs the live
# look found deficient (OOS release 8 vs 27 %; lying longer after a miss;
# in-threat smash / grab). The lever is empty where the label's rate equals
# the bot's — the gate line pools the three deficits.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_onstage_cells.exs SET.frames [SET2.frames ...]
# (several sets are pooled in memory — a round's parts can't be concatenated
# on disk past 4 GB uncompressed, GOTCHA: term_to_binary [:compressed] wraps)
for app <- [:nx], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L

paths = System.argv()
if paths == [], do: raise("usage: dagger_set_onstage_cells.exs SET.frames [SET2.frames ...]")
path = Enum.join(Enum.map(paths, &Path.basename/1), "+")
m = %{frame_lists: Enum.flat_map(paths, fn p -> (p |> File.read!() |> :erlang.binary_to_term()).frame_lists end)}
edge = 85.5
fr = List.flatten(m.frame_lists)
pct = fn a, b -> if b == 0, do: "-", else: :erlang.float_to_binary(100 * a / b, decimals: 1) <> " %" end

cell = fn f ->
  p = f.game_state.players[1]
  a = p.action || 0
  d = abs(p.x || 0.0) - edge
  cond do
    a in 178..182 -> :shield
    a in [184, 192] -> :lying
    a in 183..198 -> :knockdown_other
    p.on_ground != true and (d > 0 or (p.y || 0.0) < -5.0) -> :offstage
    d > -L.near_edge() -> :near_edge
    L.onstage_cell(p, f.game_state.players[2]) == :threat -> :threat
    p.on_ground != true and (p.jumps_left || 0) == 0 -> :drift
    true -> :other
  end
end

sx = fn c -> ((c.main_stick[:x] || 0.5) - 0.5) * 2 end
sy = fn c -> ((c.main_stick[:y] || 0.5) - 0.5) * 2 end
cx = fn c -> ((c.c_stick[:x] || 0.5) - 0.5) * 2 end
cy = fn c -> ((c.c_stick[:y] || 0.5) - 0.5) * 2 end
on = %{
  "release" => &(not L.shielding?(&1)),
  "jump" => &(&1.button_x or &1.button_y),
  "grab" => &(&1.button_z or (&1.button_a and L.shielding?(&1))),
  "roll" => &(abs(sx.(&1)) >= 0.6),
  "spot" => &(sy.(&1) <= -0.6),
  "a" => & &1.button_a,
  "z" => & &1.button_z,
  "b" => & &1.button_b,
  "smash" => &(abs(cx.(&1)) >= 0.6 or abs(cy.(&1)) >= 0.6),
  "shield" => &L.shielding?/1,
  "any_input" => &(abs(sx.(&1)) >= 0.6 or abs(sy.(&1)) >= 0.6 or &1.button_a or &1.button_b or &1.button_x or &1.button_y or &1.button_z)
}
events = %{shield: ~w(release jump grab roll spot), lying: ~w(any_input a), threat: ~w(a z smash b jump shield)}

# edge = on at t, off at t-1. actual: the bot's press vs its real previous press (the frame before it in the
# list); label: the label vs the label chain's prev (prev_controller on a labelled frame IS the previous label)
pairs = Enum.flat_map(m.frame_lists, fn l -> Enum.chunk_every(l, 2, 1, :discard) end)
actual_of = fn f -> f[:actual] || f.controller end
labelled? = fn f -> f[:input_only] != true end

by_cell = Enum.group_by(fr, cell)
IO.puts("#{Path.basename(path)}: #{length(fr)} frames, #{length(m.frame_lists)} lists, #{Enum.count(fr, labelled?)} labelled")
for c <- [:offstage, :near_edge, :drift, :shield, :lying, :knockdown_other, :threat, :other] do
  xs = by_cell[c] || []
  IO.puts("  #{String.pad_trailing(to_string(c), 16)} n=#{String.pad_leading(to_string(length(xs)), 7)}  labelled #{String.pad_leading(to_string(Enum.count(xs, labelled?)), 7)} (#{pct.(Enum.count(xs, labelled?), length(xs))})")
end

gate =
  for {c, evs} <- events, into: %{} do
    ps = Enum.filter(pairs, fn [_, b] -> cell.(b) == c and labelled?.(b) end)
    n = length(ps)
    line =
      Enum.map(evs, fn e ->
        f = on[e]
        act = Enum.count(ps, fn [a, b] -> f.(actual_of.(b)) and not f.(actual_of.(a)) end)
        lab = Enum.count(ps, fn [_, b] -> f.(b.controller) and not f.(b.prev_controller) end)
        {e, act, lab}
      end)
    IO.puts("  #{c} (labelled pairs n=#{n}): " <> Enum.map_join(line, "  ", fn {e, a, l} -> "#{e} bot #{pct.(a, n)} | label #{pct.(l, n)}" end))
    {c, Map.new(line, fn {e, a, l} -> {e, {a, l, n}} end)}
  end

# the three deficits, as "label edge rate / bot edge rate" (> 1 = the label does it more)
ratio = fn {a, l, _} -> if a == 0, do: nil, else: Float.round(l / a, 2) end
rel = ratio.(gate[:shield]["release"])
gu = ratio.(gate[:lying]["any_input"])
sm = ratio.(gate[:threat]["smash"])
gr = ratio.(gate[:threat]["z"])
IO.puts("RESULT onstage cells: labelled shield #{elem(gate[:shield]["release"], 2)} lying #{elem(gate[:lying]["any_input"], 2)} threat #{elem(gate[:threat]["smash"], 2)} pairs; " <>
  "label/bot edge ratio: shield release #{inspect(rel)}  lying getup #{inspect(gu)}  threat smash #{inspect(sm)}  threat grab(Z) #{inspect(gr)}")
