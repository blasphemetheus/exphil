# Origin of the chained labels' off-expert edges (2026-10-16). dag9o's second
# roll made the r9o regression real while labeler_carried_cells.exs showed the
# teacher-forced label faithful on the EXPERT's states (every cell within x2);
# the gap is on the BOT's states: the chained label airdodged 0.51 %/f at
# off_high_inhand where the tf-label presses 0.0. This traces each label edge
# in a DAgger set back to its mechanism. Per labelled frame of the keep-actual
# parts, conditioned on the set's own chain prev (`prev_controller`):
#
#   stored   the edge the set carries (b.controller vs b.prev_controller — the
#            v6 chain as trained on)
#   abs      expected edge hazard re-labelling the same state with THIS index
#            under the v6 absolute rule (mean over the k neighbours: a change
#            row with the button down and the prev up)
#   edge     the same under the v7 edge rule (row_controller_edges/4)
#   lift     share of the abs-edge neighbours that are mid-hold rows (the
#            expert HOLDING the button, lifted onto a prev that has it up) —
#            what v7 removes
#   cross    share of the abs-edge neighbours whose own coarse cell (from the
#            index row: height band / jump / hitstun / grounded) differs from
#            the query's — an off-manifold borrow
#   botprev  share of the cell's frames whose chain prev is the bot's own press
#            (run start after a gate reset or a --max-run cut), and the share
#            of the stored edges that sit on those frames
#   held     share of botprev frames where the label keeps a button the bot was
#            holding (hold-copy of the bot's press: no edge, but a label)
#   d2       median squared distance to the nearest expert row
#
#   mix run --no-compile scripts/label_edge_origins.exs --index IDX.bin \
#     --set data/silent_fall/sim_dagger_expert_r9o_2091.frames [--set ...] \
#     [--ref eval_runs/1001_queue/labeler_carried_cells_v7.json] [--out report.json]
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [index: :string, set: [:string, :keep], ref: :string, out: :string, limit: :integer])
index = L.load(opts[:index] || raise("--index required"))
edge = GA.stage_edge(32)
Output.banner("Label edge origins (#{index.n} rows, window #{index.window})")
if index.prev_labels == nil, do: raise("a v7 index (prev_labels) is needed for the edge rule")

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
# the button each event rides on, for the hold-copy count
held = %{"b" => b?, "side_b" => b?, "up_b" => b?, "jump" => jump?, "airdodge" => &L.shielding?/1, "a" => fn c -> c.button_a == true end}
event_names = Enum.map(events, &elem(&1, 0))

# ---- index columns for the neighbours' coarse cell --------------------------
col = fn name ->
  {_, scale, w} = Enum.find(L.dims(index.window), &(elem(&1, 0) == name))
  i = Enum.find_index(index.dims, &(&1 == name))
  index.x |> Nx.slice_along_axis(i, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.multiply(scale / w) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list() |> List.to_tuple()
end
cy = col.(:y); cd = col.(:dist); cj = col.(:jumps); ch = col.(:hitstun); cg = col.(:grounded)
coarse = fn y, dist, jumps, hit, grounded ->
  inhand = if(jumps > 0.5, do: "inhand", else: "spent")
  band = cond do y > 0.0 -> "high"; y > -20.0 -> "ledge"; y > -60.0 -> "low"; true -> "deep" end
  cond do
    dist > 0.0 and not grounded and hit -> "carried"
    dist > 0.0 and not grounded -> "off_" <> band <> "_" <> inhand
    hit -> "hit"
    grounded and dist > -L.near_edge() -> "near_ground"
    not grounded and dist > -L.air_reach() -> "drift_" <> inhand
    true -> "other"
  end
end
row_coarse = fn j -> coarse.(elem(cy, j), elem(cd, j), elem(cj, j), elem(ch, j) > 0.5, elem(cg, j) > 0.5) end
p_coarse = fn p -> coarse.(p.y || 0.0, abs(p.x || 0.0) - edge, p.jumps_left || 0, (p.hitstun_frames_left || 0) > 0, p.on_ground == true) end

# ---- the sets' labelled frames in cells ---------------------------------------
rows =
  Enum.flat_map(Keyword.get_values(opts, :set), fn path ->
    m = path |> File.read!() |> :erlang.binary_to_term()
    Output.puts("#{path}: #{length(m.frame_lists)} lists")
    Enum.flat_map(m.frame_lists, fn l ->
      l
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] ->
        %{p: b.game_state.players[1], opp: b.game_state.players[2], prev_p: b[:prev_player] || a.game_state.players[1],
          labelled: b[:input_only] != true, label: b.controller, prev: b.prev_controller,
          bot: b[:actual] || b.controller, botprev: a[:input_only] == true}
      end)
      |> L.tag_cells(edge)
    end)
  end)
  |> Enum.filter(&(&1.labelled and &1.cell))
rows = if opts[:limit], do: Enum.take(rows, opts[:limit]), else: rows
Output.puts("#{length(rows)} labelled frames in cells (#{Enum.count(rows, & &1.botprev)} with the bot's press as chain prev)")

# ---- k-NN with this index, both rules over every neighbour --------------------
states = Enum.map(rows, &{&1.p, &1.prev, &1.opp, &1.prev_p})
queries = Enum.map(states, &(L.query_features(&1, edge) |> L.vector(index.window)))
n_chunks = div(length(queries) + 255, 256)
{idx, d2} =
  queries
  |> Enum.chunk_every(256)
  |> Enum.with_index()
  |> Enum.map(fn {q, i} ->
    if rem(i, 50) == 0, do: IO.write(:stderr, "\r  kNN #{i}/#{n_chunks}\e[K")
    L.nearest_d2(index, q, index.k)
  end)
  |> Enum.reduce({[], []}, fn {i, d}, {is, ds} -> {[Nx.to_list(i) | is], [Nx.to_list(d) | ds]} end)
IO.write(:stderr, "\n")
idx = idx |> Enum.reverse() |> List.flatten() |> Enum.chunk_every(index.k)
d2 = d2 |> Enum.reverse() |> List.flatten()

k = index.k
bi = fn true -> 1; _ -> 0 end
rows =
  Enum.zip_with([rows, idx, d2], fn [r, ns, d] ->
    t = L.toward(r.p)
    qc = p_coarse.(r.p)
    per =
      Enum.map(ns, fn j ->
        if elem(index.hold, j) == 1 do
          nil
        else
          row = elem(index.labels, j)
          {L.row_controller(row, t, r.prev), L.row_controller_edges(row, elem(index.prev_labels, j), t, r.prev), row_coarse.(j)}
        end
      end)
    stats =
      Map.new(events, fn {e, f} ->
        stored = f.(r.label, r.prev)
        {n_abs, n_edge, n_lift, n_cross, from} =
          Enum.reduce(per, {0, 0, 0, 0, %{}}, fn
            nil, acc -> acc
            {abs_c, edge_c, nc_cell}, {na, ne, nl, nc, from} ->
              ea = f.(abs_c, r.prev)
              ee = f.(edge_c, r.prev)
              cross = nc_cell != qc
              from = if ea and cross, do: Map.update(from, nc_cell, 1, &(&1 + 1)), else: from
              {na + bi.(ea), ne + bi.(ee), nl + bi.(ea and not ee), nc + bi.(ea and cross), from}
          end)
        held_copy = r.botprev and held[e].(r.label) and held[e].(r.prev)
        {e, %{stored: stored, abs: n_abs / k, edge: n_edge / k, lift: n_lift, cross: n_cross, n_abs: n_abs, held: held_copy, from: from}}
      end)
    Map.merge(r, %{stats: stats, d2: d})
  end)

# ---- aggregate by cell x event --------------------------------------------------
ref = if opts[:ref], do: opts[:ref] |> File.read!() |> Jason.decode!()
ref_pct = fn src, c, e ->
  with %{} = m <- ref && ref[src] && ref[src][c] && ref[src][c][e], n when n > 0 <- m["frames"], do: 100 * m["edges"] / n, else: (_ -> nil)
end
fmt = fn v -> if v == nil, do: "   -  ", else: String.pad_leading(:erlang.float_to_binary(v / 1, decimals: 2), 6) end
by = Enum.group_by(rows, & &1.cell)
report =
  for c <- L.carried_cells(), into: %{} do
    xs = by[c] || []
    n = length(xs)
    nb = Enum.count(xs, & &1.botprev)
    med = if xs == [], do: nil, else: xs |> Enum.map(& &1.d2) |> Enum.sort() |> Enum.at(div(n, 2))
    evs =
      for e <- event_names, into: %{} do
        s = Enum.map(xs, & &1.stats[e])
        stored = Enum.count(s, & &1.stored)
        stored_bp = Enum.count(Enum.zip(xs, s), fn {r, st} -> r.botprev and st.stored end)
        n_abs = Enum.sum(Enum.map(s, & &1.n_abs))
        # where the cross-cell edge neighbours live (coarse cell -> count)
        from = Enum.reduce(s, %{}, fn st, acc -> Map.merge(acc, st.from, fn _, a, b -> a + b end) end)
        {e, %{frames: n, stored: stored, stored_botprev: stored_bp, abs: Enum.sum(Enum.map(s, & &1.abs)), edge: Enum.sum(Enum.map(s, & &1.edge)),
              lift: Enum.sum(Enum.map(s, & &1.lift)), cross: Enum.sum(Enum.map(s, & &1.cross)), n_abs: n_abs, held: Enum.count(s, & &1.held),
              from: from, expert: ref_pct.("expert", c, e), tf_label: ref_pct.("tf_label", c, e)}}
      end
    {c, %{frames: n, botprev: nb, d2_median: med, events: evs}}
  end

pct = fn k, n -> if n == 0, do: nil, else: 100 * k / n end
from_s = fn v -> v.from |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(3) |> Enum.map_join(",", fn {c, n} -> "#{c} #{n}" end) end
Output.puts("per cell x event: hazard %/frame expert | tf-label (held-out, --ref) | stored chain | abs-rule | edge-rule ; of the abs edges: lift % cross % ; botprev frames % (stored edges on them %) ; held-copy % of botprev ; d2 median")
for c <- L.carried_cells(), r = report[c], r.frames > 0 do
  Output.puts("#{c}  n=#{r.frames}  botprev #{fmt.(pct.(r.botprev, r.frames))} %  d2 #{Float.round(r.d2_median, 3)}")
  for e <- event_names, v = r.events[e], v.stored > 0 or v.abs > 0.05 or v.edge > 0.05 or (v.expert || 0.0) > 0.0 do
    Output.puts("    #{String.pad_trailing(e, 9)} expert #{fmt.(v.expert)} | tf #{fmt.(v.tf_label)} | stored #{fmt.(pct.(v.stored, r.frames))} (#{v.stored}) | abs #{fmt.(pct.(v.abs, r.frames))} | edge #{fmt.(pct.(v.edge, r.frames))} ; lift #{fmt.(pct.(v.lift, v.n_abs))} cross #{fmt.(pct.(v.cross, v.n_abs))} ; stored on botprev #{fmt.(pct.(v.stored_botprev, v.stored))} % ; held #{fmt.(pct.(v.held, r.botprev))} % ; from #{from_s.(v)}")
  end
end

line = fn c, e ->
  r = report[c]
  v = r.events[e]
  "#{c}/#{e}: n=#{r.frames} expert #{fmt.(v.expert)} tf #{fmt.(v.tf_label)} stored #{fmt.(pct.(v.stored, r.frames))} abs #{fmt.(pct.(v.abs, r.frames))} edge #{fmt.(pct.(v.edge, r.frames))} lift #{fmt.(pct.(v.lift, v.n_abs))} cross #{fmt.(pct.(v.cross, v.n_abs))} botprev-share-of-stored #{fmt.(pct.(v.stored_botprev, v.stored))} held #{fmt.(pct.(v.held, r.botprev))} from [#{from_s.(v)}]"
end
for {c, e} <- [{"off_high_inhand", "airdodge"}, {"off_ledge_inhand", "airdodge"}, {"off_high_inhand", "jump"}, {"off_ledge_inhand", "jump"}, {"carried", "side_b"}, {"post1-5_inhand", "side_b"}, {"near_ground", "side_b"}, {"off_deep_spent", "b"}] do
  Output.puts("RESULT " <> line.(c, e))
end

if opts[:out] do
  File.write!(opts[:out], Jason.encode!(report, pretty: true))
  Output.puts("-> #{opts[:out]}")
end
