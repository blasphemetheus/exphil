# Carried-off cells of the expert labeler (2026-10-16). Every mixed arm this
# labeler produced shares one signature — carried-off died 0.74–0.86, carried
# side-B x2–x3, side-B latency 0 (dag6w, dag7w, dag9o) — so this measures the
# per-frame press hazard exactly there: offstage in hitstun ("carried") and in
# the frames after hitstun ends, by jump state, for four input sources:
#
#   expert   the held-out expert's own press (24 FD Fox games)
#   tf-label the labeler's teacher-forced label on those expert frames
#   bot      the bot's own press on its DAgger states (--set parts, keep-actual)
#   label    the labeler's chained label on those states
#
#   mix run --no-compile scripts/labeler_carried_cells.exs --split HELD.json --index IDX.bin \
#     [--games 24] [--set data/silent_fall/sim_dagger_expert_r9o_2091.frames ...] [--out report.json]
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [split: :string, index: :string, games: :integer, set: [:string, :keep], out: :string])
index = L.load(opts[:index] || raise("--index required"))
edge = GA.stage_edge(32)
:rand.seed(:exsss, {1, 2, 3})
Output.banner("Labeler carried-off cells (#{index.n} rows, window #{index.window})")

sx = fn c -> ((c.main_stick || %{})[:x] || 0.5) - 0.5 end
sy = fn c -> ((c.main_stick || %{})[:y] || 0.5) - 0.5 end
b? = fn c -> c.button_b == true end
jump? = fn c -> c.button_x == true or c.button_y == true end
events = [
  {"b", fn c, p -> b?.(c) and not b?.(p) end},
  {"side_b", fn c, p -> b?.(c) and not b?.(p) and abs(sx.(c)) >= 0.25 and abs(sx.(c)) > abs(sy.(c)) end},
  {"up_b", fn c, p -> b?.(c) and not b?.(p) and sy.(c) >= 0.25 and sy.(c) >= abs(sx.(c)) end},
  {"jump", fn c, p -> jump?.(c) and not jump?.(p) end},
  {"airdodge", fn c, p -> L.shielding?(c) and not L.shielding?(p) end},
  {"a", fn c, p -> c.button_a == true and not (p.button_a == true) end}
]

# cell of frame t given the frames since hitstun ended: L.carried_cell/3 (hit
# cells carried / post<1-5|6-15|16-30>_<inhand|spent>; free offstage cells
# off_<band>_<inhand|spent> with the readout's height bands; onstage origins
# near_ground / drift_<inhand|spent>). Shared with label_edge_origins.exs.
tag = &L.tag_cells(&1, edge)
cells = L.carried_cells()
hazards = fn rows, src ->
  by = Enum.group_by(Enum.filter(rows, & &1.cell), & &1.cell)
  for c <- cells, into: %{} do
    xs = by[c] || []
    {c, Map.new(events, fn {e, f} -> {e, {Enum.count(xs, fn r -> f.(r[src], r[:"prev_#{src}"]) end), length(xs)}} end)}
  end
end

# ---- expert held-out: actual + teacher-forced label ---------------------------
expert_rows =
  if opts[:split] do
    files = opts[:split] |> File.read!() |> Jason.decode!() |> Map.fetch!("validation")
    games =
      files
      |> Enum.flat_map(fn path ->
        case Peppi.metadata(path) do
          {:ok, %{stage: 32} = meta} ->
            own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
            opp = own && Enum.find(meta.players, &(&1.port != own.port))
            if own && opp, do: [{path, own.port, opp.port}], else: []
          _ -> []
        end
      end)
      |> Enum.take(opts[:games] || 24)

    Enum.flat_map(games, fn {path, own, opp} ->
      {:ok, replay} = Peppi.parse(path, player_port: own)
      frames = replay |> Peppi.to_training_frames(player_port: own, opponent_port: opp, remap_ports: true) |> Enum.reject(&(&1.game_state.frame < 0))
      pairs = frames |> Enum.chunk_every(2, 1, :discard) |> Enum.filter(fn [f0, f1] -> f1.game_state.frame == f0.game_state.frame + 1 end)
      lab? = fn [_, f1] -> L.labelable?(f1.game_state.players[1], edge, index.window, f1.game_state.players[2]) end
      labelable = Enum.filter(pairs, lab?)
      states = Enum.map(labelable, fn [f0, f1] -> {f1.game_state.players[1], f0.controller, f1.game_state.players[2], f0.game_state.players[1]} end)
      labels = Map.new(Enum.zip(Enum.map(labelable, fn [_, f1] -> f1.game_state.frame end), L.label_batch(index, states, edge)))
      pairs
      |> Enum.map(fn [f0, f1] ->
        lab = labels[f1.game_state.frame]
        %{p: f1.game_state.players[1], expert: f1.controller, prev_expert: f0.controller, tf: lab, prev_tf: f0.controller}
      end)
      |> tag.()
    end)
  else
    []
  end

# ---- DAgger sets: bot's press + chained label ----------------------------------
set_rows =
  Enum.flat_map(Keyword.get_values(opts, :set), fn path ->
    m = path |> File.read!() |> :erlang.binary_to_term()
    Enum.flat_map(m.frame_lists, fn l ->
      l
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] ->
        labelled = b[:input_only] != true
        %{p: b.game_state.players[1], labelled: labelled,
          bot: b[:actual] || b.controller, prev_bot: a[:actual] || a.controller,
          label: if(labelled, do: b.controller), prev_label: if(labelled, do: b.prev_controller)}
      end)
      |> tag.()
    end)
  end)

pct = fn {k, n} -> if n == 0, do: "   -  ", else: String.pad_leading("#{Float.round(100 * k / n, 2)}", 6) end
report = %{}
report =
  if expert_rows != [] do
    # both over LABELABLE frames only (the first cut compared the expert over
    # every cell frame — ledge hangs 252..263 included, where the expert
    # presses jump 0.5 %/f — against the label over labelable ones: a 4.6x
    # "inflation" at off_ledge_inhand that was the denominator)
    lab_rows = Enum.filter(expert_rows, & &1.tf)
    e = hazards.(lab_rows, :expert)
    t = hazards.(lab_rows, :tf)
    Output.puts("held-out expert: #{length(expert_rows)} frames, #{Enum.count(expert_rows, & &1.cell)} in cells, #{Enum.count(lab_rows, & &1.cell)} labelable in cells; per-frame edge hazard % on labelable frames (expert | tf-label)")
    for c <- cells do
      n = elem(e[c]["b"], 1)
      Output.puts("  #{String.pad_trailing(c, 17)} n=#{String.pad_leading(to_string(n), 6)}  " <> Enum.map_join(events, "  ", fn {ev, _} -> "#{ev} #{pct.(e[c][ev])} | #{pct.(t[c][ev])}" end))
    end
    Map.merge(report, %{expert: e, tf_label: t})
  else
    report
  end
report =
  if set_rows != [] do
    lr = Enum.filter(set_rows, & &1.labelled)
    bo = hazards.(lr, :bot)
    la = hazards.(lr, :label)
    Output.puts("DAgger sets: #{length(set_rows)} frames, #{length(lr)} labelled, #{Enum.count(lr, & &1.cell)} labelled in cells; per-frame edge hazard % (bot | label)")
    for c <- cells do
      n = elem(bo[c]["b"], 1)
      Output.puts("  #{String.pad_trailing(c, 17)} n=#{String.pad_leading(to_string(n), 6)}  " <> Enum.map_join(events, "  ", fn {ev, _} -> "#{ev} #{pct.(bo[c][ev])} | #{pct.(la[c][ev])}" end))
    end
    Map.merge(report, %{bot: bo, label: la})
  else
    report
  end

r = fn m, c, e -> with {k, n} when n > 0 <- m[c][e], do: Float.round(100 * k / n, 2), else: (_ -> nil) end
Output.puts("RESULT carried side_b hazard %: expert #{r.(report[:expert], "carried", "side_b")} tf-label #{r.(report[:tf_label], "carried", "side_b")} bot #{r.(report[:bot], "carried", "side_b")} label #{r.(report[:label], "carried", "side_b")}; " <>
  "post1-5 inhand side_b: expert #{r.(report[:expert], "post1-5_inhand", "side_b")} tf-label #{r.(report[:tf_label], "post1-5_inhand", "side_b")} bot #{r.(report[:bot], "post1-5_inhand", "side_b")} label #{r.(report[:label], "post1-5_inhand", "side_b")}; " <>
  "off_high_inhand airdodge: expert #{r.(report[:expert], "off_high_inhand", "airdodge")} tf-label #{r.(report[:tf_label], "off_high_inhand", "airdodge")} bot #{r.(report[:bot], "off_high_inhand", "airdodge")} label #{r.(report[:label], "off_high_inhand", "airdodge")}; " <>
  "near_ground side_b: expert #{r.(report[:expert], "near_ground", "side_b")} tf-label #{r.(report[:tf_label], "near_ground", "side_b")} bot #{r.(report[:bot], "near_ground", "side_b")} label #{r.(report[:label], "near_ground", "side_b")}")

if opts[:out] do
  enc = fn m -> m && Map.new(m, fn {c, evs} -> {c, Map.new(evs, fn {e, {k, n}} -> {e, %{edges: k, frames: n}} end)} end) end
  File.write!(opts[:out], Jason.encode!(Map.new(report, fn {k, v} -> {k, enc.(v)} end), pretty: true))
  Output.puts("-> #{opts[:out]}")
end
