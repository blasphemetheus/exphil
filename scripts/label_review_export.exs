# Label review export (2026-10-16, Bradley: "visualize what is happening with
# labeling and dagger and the teacher … so I can use my own taste to see that
# looks right"). Writes a directory the labels page (priv/viewer/labels/) plays
# one run at a time: per run a MSLTRACE1 clip of the bot's (or expert's) own
# state trajectory and a sidecar with, per frame, the cell, the player's real
# press, the label, the chain prev, the gate, and the k nearest expert rows the
# label was sampled from (hold / press, their own cell, d²).
#
# Sources (any number, each a "source" in the manifest):
#   --set  PATH.frames   a keep-actual DAgger set: bot states + chained labels
#                        (+ --index: the neighbours are recomputed with it)
#   --slp  GLOB/.json    expert replays (a split json's "validation" list or a
#                        glob of .slp): expert states, expert press, and the
#                        teacher-forced label from --index (required)
#   --max-runs N         per source (default 80), --min-cell-frames M (default 8:
#                        a run must have that many frames in a readout cell)
#
#   mix run --no-compile scripts/label_review_export.exs --index IDX.bin \
#     --set data/silent_fall/sim_dagger_expert_r9o_2091.frames --slp data/silent_fall/heldout_fd_fox_split.json \
#     --out eval_runs/1016_label_review
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Data.Peppi
alias ExPhil.Interp.ActionNames
alias ExPhil.Sim.{GA, Trace}
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [index: :string, set: [:string, :keep], slp: [:string, :keep], out: :string, max_runs: :integer, min_cell_frames: :integer, games: :integer, max_d2: :float])
out = opts[:out] || raise("--out required")
index = if opts[:index], do: L.load(opts[:index])
edge = GA.stage_edge(32)
max_runs = opts[:max_runs] || 80
min_cell = opts[:min_cell_frames] || 8
:rand.seed(:exsss, {7, 7, 7})
File.mkdir_p!(out)
Output.banner("Label review export -> #{out}" <> if(index, do: " (index #{index.n} rows, #{index.window})", else: ""))

ctrl = fn
  nil -> nil
  c ->
    ms = c.main_stick || %{x: 0.5, y: 0.5}
    cs = c.c_stick || %{x: 0.5, y: 0.5}
    %{ms: [Float.round((ms[:x] || 0.5) / 1, 3), Float.round((ms[:y] || 0.5) / 1, 3)], cs: [Float.round((cs[:x] || 0.5) / 1, 3), Float.round((cs[:y] || 0.5) / 1, 3)],
      a: c.button_a == true, b: c.button_b == true, x: c.button_x == true, y: c.button_y == true, z: c.button_z == true,
      l: c.button_l == true, r: c.button_r == true, sh: Float.round(max(c.l_shoulder || 0.0, c.r_shoulder || 0.0) / 1, 2)}
end

# the readout events (labeler_carried_cells.exs), as edges of a press over its prev
sx = fn c -> ((c.main_stick || %{})[:x] || 0.5) - 0.5 end
sy = fn c -> ((c.main_stick || %{})[:y] || 0.5) - 0.5 end
b? = fn c -> c.button_b == true end
jump? = fn c -> c.button_x == true or c.button_y == true end
events = [
  {"side_b", fn c, p -> b?.(c) and not b?.(p) and abs(sx.(c)) >= 0.25 and abs(sx.(c)) > abs(sy.(c)) end},
  {"up_b", fn c, p -> b?.(c) and not b?.(p) and sy.(c) >= 0.25 and sy.(c) >= abs(sx.(c)) end},
  {"b", fn c, p -> b?.(c) and not b?.(p) end},
  {"jump", fn c, p -> jump?.(c) and not jump?.(p) end},
  {"airdodge", fn c, p -> L.shielding?(c) and not L.shielding?(p) end},
  {"a", fn c, p -> c.button_a == true and not (p.button_a == true) end}
]
edges_of = fn c, p -> if c && p, do: events |> Enum.filter(fn {_, f} -> f.(c, p) end) |> Enum.map(&elem(&1, 0)), else: [] end

# neighbour rows' own coarse cell, from the index columns
{row_cell, press_of} =
  if index do
    col = fn name ->
      {_, scale, w} = Enum.find(L.dims(index.window), &(elem(&1, 0) == name))
      i = Enum.find_index(index.dims, &(&1 == name))
      index.x |> Nx.slice_along_axis(i, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.multiply(scale / w) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list() |> List.to_tuple()
    end
    cy = col.(:y); cd = col.(:dist); cj = col.(:jumps); ch = col.(:hitstun); cg = col.(:grounded)
    off? = Enum.find_index(index.dims, &(&1 == :offstage)) && col.(:offstage)
    cell = fn j ->
      y = elem(cy, j); dist = elem(cd, j); grounded = elem(cg, j) > 0.5; hit = elem(ch, j) > 0.5
      off = if(off?, do: elem(off?, j) > 0.5, else: dist > 0.0 and not grounded)
      inhand = if(elem(cj, j) > 0.5, do: "inhand", else: "spent")
      band = cond do y > 0.0 -> "high"; y > -20.0 -> "ledge"; y > -60.0 -> "low"; true -> "deep" end
      cond do
        off and hit -> "carried"
        off -> "off_#{band}_#{inhand}"
        hit -> "hit"
        grounded and dist > -L.near_edge() -> "near_ground"
        not grounded and dist > -L.air_reach() -> "drift_#{inhand}"
        true -> "onstage"
      end
    end
    # what the row presses (edges over its own prev row) or holds
    press = fn j ->
      if elem(index.hold, j) == 1 do
        "hold"
      else
        row = elem(index.labels, j); prow = elem(index.prev_labels, j)
        names = [{4, "A"}, {5, "B"}, {6, "X"}, {7, "Y"}, {8, "Z"}, {9, "L"}, {10, "R"}]
        pressed = for {c, n} <- names, Enum.at(row, c) > 0.5 and Enum.at(prow, c) < 0.5, do: n
        held = for {c, n} <- names, Enum.at(row, c) > 0.5 and Enum.at(prow, c) > 0.5, do: n
        stick = "#{Float.round(Enum.at(row, 0) - 0.5, 2)},#{Float.round(Enum.at(row, 1) - 0.5, 2)}"
        (if(pressed == [], do: "", else: "press " <> Enum.join(pressed)) <> if(held == [], do: "", else: " hold " <> Enum.join(held)) <> " stick " <> stick) |> String.trim()
      end
    end
    {cell, press}
  else
    {nil, nil}
  end

# neighbours for a list of label states ({p, prev, opp, prev_p})
neighbours = fn states ->
  if index && states != [] do
    queries = Enum.map(states, &(L.query_features(&1, edge) |> L.vector(index.window)))
    queries
    |> Enum.chunk_every(256)
    |> Enum.flat_map(fn q ->
      {idx, d2} = L.nearest_d2k(index, q, index.k)
      Enum.zip_with(Nx.to_list(idx), Nx.to_list(d2), fn ns, ds ->
        Enum.zip_with(ns, ds, fn j, d -> %{hold: elem(index.hold, j) == 1, cell: row_cell.(j), press: press_of.(j), d2: Float.round(d / 1, 3)} end)
      end)
    end)
  else
    List.duplicate(nil, length(states))
  end
end

write_run = fn source, id, frames, label ->
  # frames: [%{gs, p, opp, prev_p, actual, label, prev, input_only, gated, cell}]
  states = frames |> Enum.filter(& &1.label) |> Enum.map(&{&1.p, &1.prev, &1.opp, &1.prev_p})
  nbs = Map.new(Enum.zip(frames |> Enum.with_index() |> Enum.filter(fn {f, _} -> f.label end) |> Enum.map(&elem(&1, 1)), neighbours.(states)))
  rows =
    frames
    |> Enum.with_index()
    |> Enum.map(fn {f, i} ->
      %{i: i, frame: f.gs.frame, cell: f.cell, action: f.p.action, name: ActionNames.name(f.p.action || 0), af: f.p.action_frame,
        x: Float.round((f.p.x || 0.0) / 1, 1), y: Float.round((f.p.y || 0.0) / 1, 1), dist: Float.round((abs(f.p.x || 0.0) - edge) / 1, 1),
        jumps: f.p.jumps_left, hit: f.p.hitstun_frames_left, ground: f.p.on_ground == true,
        input_only: f.input_only, gated: f.gated, actual: ctrl.(f.actual), label: ctrl.(f.label), prev: ctrl.(f.prev),
        actual_edges: edges_of.(f.actual, f.prev_actual), label_edges: edges_of.(f.label, f.prev), nbs: nbs[i]}
    end)
  label_events = rows |> Enum.flat_map(fn r -> Enum.map(r.label_edges, &%{i: r.i, event: &1, cell: r.cell, who: "label"}) end)
  actual_events = rows |> Enum.flat_map(fn r -> Enum.map(r.actual_edges, &%{i: r.i, event: &1, cell: r.cell, who: "actual"}) end)
  cells = rows |> Enum.map(& &1.cell) |> Enum.reject(&is_nil/1) |> Enum.frequencies()
  d2s = rows |> Enum.map(fn r -> r.nbs && hd(r.nbs).d2 end) |> Enum.reject(&is_nil/1)
  summary = %{id: id, source: source, label: label, frames: length(rows), labelled: Enum.count(rows, & &1.label), gated: Enum.count(rows, & &1.gated),
              cells: cells, label_events: label_events, actual_events: actual_events,
              d2_median: if(d2s == [], do: nil, else: d2s |> Enum.sort() |> Enum.at(div(length(d2s), 2)))}
  Trace.from_game_states(Enum.map(frames, & &1.gs), label: label) |> Trace.write!(Path.join(out, "#{id}.msltrace.json"))
  File.write!(Path.join(out, "#{id}.json"), Jason.encode!(Map.put(summary, :rows, rows)))
  summary |> Map.put(:trace, "#{id}.msltrace.json") |> Map.put(:sidecar, "#{id}.json")
end

interesting? = fn frames -> Enum.count(frames, & &1.cell) >= min_cell end

# ---- DAgger sets ---------------------------------------------------------------
set_runs =
  Enum.flat_map(Keyword.get_values(opts, :set), fn path ->
    m = path |> File.read!() |> :erlang.binary_to_term()
    source = Path.basename(path, ".frames")
    Output.puts("#{path}: #{length(m.frame_lists)} lists")
    m.frame_lists
    |> Enum.with_index()
    |> Enum.map(fn {l, li} ->
      frames =
        [nil | l]
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] ->
          labelled = b[:input_only] != true
          %{gs: b.game_state, p: b.game_state.players[1], opp: b.game_state.players[2], prev_p: b[:prev_player] || (a && a.game_state.players[1]),
            actual: b[:actual] || b.controller, prev_actual: a && (a[:actual] || a.controller),
            label: if(labelled, do: b.controller), prev: b.prev_controller, input_only: not labelled, gated: b[:gated] == true}
        end)
        |> L.tag_cells(edge)
      {li, frames}
    end)
    |> Enum.filter(fn {_, frames} -> interesting?.(frames) end)
    |> Enum.shuffle()
    |> Enum.take(max_runs)
    |> Enum.sort_by(&elem(&1, 0))
    # sim rollouts put the bot on port 1 (drill/ppo/ga: attacker = port 1); live .slp sets lost the port at remap
    |> Enum.map(fn {li, frames} -> write_run.(source, "#{source}_l#{li}", frames, "#{source} list #{li}" <> if(String.starts_with?(source, "sim_"), do: " (bot on sim port 1)", else: " (bot's real port not recorded)")) end)
  end)

# ---- expert replays, teacher-forced --------------------------------------------
slp_runs =
  Enum.flat_map(Keyword.get_values(opts, :slp), fn spec ->
    if index == nil, do: raise("--slp needs --index (the teacher-forced label)")
    files = if String.ends_with?(spec, ".json"), do: spec |> File.read!() |> Jason.decode!() |> Map.fetch!("validation"), else: Path.wildcard(spec)
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
      |> Enum.take(opts[:games] || 6)
    source = "expert_" <> Path.basename(spec, ".json")
    Output.puts("#{spec}: #{length(games)} Fox FD games")
    runs =
      Enum.flat_map(games, fn {path, own, opp} ->
        {:ok, replay} = Peppi.parse(path, player_port: own)
        frames = replay |> Peppi.to_training_frames(player_port: own, opponent_port: opp, remap_ports: true) |> Enum.reject(&(&1.game_state.frame < 0))
        # one run per offstage excursion: labelable frames with 60 f of context before, cut at 90 f past the last labelable frame
        pairs = frames |> Enum.chunk_every(2, 1, :discard) |> Enum.filter(fn [f0, f1] -> f1.game_state.frame == f0.game_state.frame + 1 end)
        rows =
          Enum.map(pairs, fn [f0, f1] ->
            p = f1.game_state.players[1]
            %{gs: f1.game_state, p: p, opp: f1.game_state.players[2], prev_p: f0.game_state.players[1], actual: f1.controller, prev_actual: f0.controller,
              lab: L.labelable?(p, edge, index.window, f1.game_state.players[2]) and (abs(p.x || 0.0) > edge - L.air_reach() or p.on_ground != true), prev: f0.controller}
          end)
          |> L.tag_cells(edge)
        # excursions = maximal stretches where a cell is set, padded 60 f before / 30 after
        idxs = rows |> Enum.with_index() |> Enum.filter(fn {r, _} -> r.cell end) |> Enum.map(&elem(&1, 1))
        groups = Enum.chunk_while(idxs, [], fn i, acc -> if acc == [] or i - hd(acc) <= 30, do: {:cont, [i | acc]}, else: {:cont, Enum.reverse(acc), [i]} end, fn acc -> {:cont, Enum.reverse(acc), []} end)
        groups
        |> Enum.map(fn g -> {max(List.first(g) - 60, 0), min(List.last(g) + 30, length(rows) - 1)} end)
        |> Enum.map(fn {a, b} ->
          seg = Enum.slice(rows, a..b)
          # teacher-forced: the label on the expert's own state with the expert's own prev
          lab_states = seg |> Enum.filter(& &1.lab) |> Enum.map(&{&1.p, &1.prev, &1.opp, &1.prev_p})
          labels = L.label_batch(index, lab_states, edge, max_d2: opts[:max_d2])
          {seg, _} = Enum.map_reduce(seg, labels, fn r, ls -> if r.lab, do: {Map.merge(r, %{label: hd(ls), input_only: false, gated: hd(ls) == nil}), tl(ls)}, else: {Map.merge(r, %{label: nil, input_only: true, gated: false}), ls} end)
          {Path.basename(path, ".slp"), a, seg, {own, opp}}
        end)
      end)
      |> Enum.filter(fn {_, _, seg, _} -> interesting?.(seg) end)
      |> Enum.shuffle()
      |> Enum.take(max_runs)
    Enum.map(runs, fn {game, a, seg, {own, opp}} ->
      # ids are file names and URL paths: keep [A-Za-z0-9_-]
      id = "#{source}_#{String.slice(game, 0, 24)}_f#{a}" |> String.replace(~r/[^A-Za-z0-9_-]+/, "_")
      # the real ports: the embedding/viewer put the subject in slot 1, but port order decides same-frame ties in Melee
      write_run.(source, id, seg, "#{game} @#{a} (teacher-forced; expert really on port #{own}, opponent port #{opp})")
    end)
  end)

manifest = %{created: DateTime.utc_now() |> DateTime.to_iso8601(), index: opts[:index], runs: set_runs ++ slp_runs, cells: L.carried_cells(), events: Enum.map(events, &elem(&1, 0))}
File.write!(Path.join(out, "manifest.json"), Jason.encode!(manifest))
Output.puts("RESULT label review: #{length(set_runs)} DAgger runs + #{length(slp_runs)} expert excursions -> #{out}/manifest.json")
