# Counterfactual COVERAGE MAP of a multishine policy (interp instrument #2,
# HANDOFF_2026-09-12 §3). The goal ("multishine from ANY state vs ANY
# opponent") measured offline: stream the fixture through the agent with
# TEACHER-FORCED history (Agent.observe/4 with the recorded input, probe:
# true), read the button head's probabilities at every loop state, and
# repeat with one counterfactual perturbation applied to EVERY frame of
# the stream (own/pair position, opponent distance, opponent action,
# facing, mirror, percents, opponent character). A cell's p(correct) is
# how sure the policy is of the fixture's B/X on the loop states under
# that perturbation; the baseline cell is the unperturbed fixture. Cells
# far below baseline are where DAgger rollouts should be aimed.
#
#   mix run scripts/probe_ms_coverage_map.exs --policy checkpoints/ms_g23a_ep57.bin \
#     [--fixture test/fixtures/replays/fox_multishine_closed_d1.slp] [--port 1] \
#     [--limit 600] [--delay-id 0] [--temperature 1.0] [--offset N|auto] \
#     [--axes pair_x,opp_dx,opp_action,facing,mirror,own_pct,opp_pct,opp_char] \
#     [--out eval_runs/<dir>/coverage_map.json]
#
# "Correct" per loop state = the fixture's ISSUED B/X majority at label
# offset N (the policy emits for t+latency; --offset auto picks the offset
# the baseline tracks best and reports all four). Independent-head
# checkpoints only (the AR head has no per-button marginal to read).

alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      fixture: :string,
      port: :integer,
      limit: :integer,
      delay_id: :integer,
      temperature: :float,
      offset: :string,
      axes: :string,
      out: :string
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
port = opts[:port] || 1
limit = opts[:limit] || 600
temperature = opts[:temperature] || 1.0
offset_opt = opts[:offset] || "auto"

{:ok, replay} = ExPhil.Data.Peppi.parse(fixture, player_port: port)

frames =
  replay
  |> ExPhil.Data.Peppi.to_training_frames(
    player_port: port,
    opponent_port: if(port == 1, do: 2, else: 1),
    remap_ports: true
  )
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.take(limit)

arr = List.to_tuple(frames)
n = tuple_size(arr)

Output.banner("Multishine counterfactual coverage map (teacher-forced live path)")

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Fixture", "#{Path.basename(fixture)} (#{n} frames)"},
  {"Delay-id", opts[:delay_id] || 0},
  {"Temperature", temperature},
  {"Label offset", offset_opt}
])

{:ok, agent} =
  Agent.start_link(
    policy_path: policy_path,
    deterministic: false,
    temperature: temperature,
    delay_id: opts[:delay_id] || 0,
    allow_untrained_delay_id: true
  )

Agent.warmup(agent)

# ---------------------------------------------------------------------------
# Loop states (parsed action_frame numbering, same keys as the expert table)
# ---------------------------------------------------------------------------
loop_keys = [
  {361, 1, true},
  {361, 2, true},
  {361, 3, true},
  {24, 0, true},
  {24, 1, true},
  {24, 2, true},
  {365, 1, false},
  {365, 2, false},
  {365, 3, false},
  {366, 0, false},
  {366, 1, false}
]

key_of = fn gs ->
  p = gs.players[1]
  {trunc(p.action || 0), trunc(p.action_frame || 0), p.on_ground == true}
end

loop_idx =
  frames
  |> Enum.with_index()
  |> Enum.filter(fn {f, _} -> key_of.(f.game_state) in loop_keys end)
  |> Enum.map(&elem(&1, 1))

Output.puts("#{length(loop_idx)} loop-state frames across #{length(loop_keys)} keys")

# Fixture's issued B/X majority per key at offset o
target_for = fn o ->
  loop_idx
  |> Enum.group_by(fn i -> key_of.(elem(arr, i).game_state) end)
  |> Map.new(fn {key, idxs} ->
    cs = for i <- idxs, i + o < n, do: elem(arr, i + o).controller
    m = length(cs)

    if m == 0 do
      {key, nil}
    else
      b = Enum.count(cs, & &1.button_b) / m
      x = Enum.count(cs, & &1.button_x) / m
      {key, %{b: b >= 0.5, x: x >= 0.5, b_rate: b, x_rate: x}}
    end
  end)
end

# Offsets 0..6: reaction delay k tracks offset k (frames[i+k].controller = raw[i+k+1]);
# physical drill ids reach 5 (09-12), and ep57's id 0 sat at offset 2.
targets_by_offset = Map.new(0..6, fn o -> {o, target_for.(o)} end)

# ---------------------------------------------------------------------------
# Perturbations: each is {axis, label, fun(game_state) -> game_state}
# applied to EVERY frame of the stream (a counterfactual game).
# ---------------------------------------------------------------------------
base_p1 = hd(frames).game_state.players[1]
base_p2 = hd(frames).game_state.players[2]
Output.puts("baseline: p1 x=#{base_p1.x} facing=#{base_p1.facing} pct=#{base_p1.percent} char=#{base_p1.character}; p2 x=#{base_p2.x} action=#{trunc(base_p2.action)} pct=#{base_p2.percent} char=#{base_p2.character}")

upd = fn gs, port_, f -> %{gs | players: Map.update!(gs.players, port_, f)} end
clamp = fn x -> max(-85.0, min(85.0, x)) end

pair_x =
  for dx <- [-20.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0] do
    {:pair_x, "both +#{trunc(dx)} (p1 x=#{clamp.(base_p1.x + dx)})",
     fn gs ->
       gs
       |> upd.(1, fn p -> %{p | x: clamp.(p.x + dx)} end)
       |> upd.(2, fn p -> %{p | x: clamp.(p.x + dx)} end)
     end}
  end

opp_dx =
  for d <- [-40.0, -15.0, 8.0, 20.0, 40.0, 80.0, 160.0] do
    {:opp_dx, "opp at p1.x#{if d >= 0, do: "+", else: ""}#{trunc(d)}",
     fn gs ->
       p1x = gs.players[1].x
       upd.(gs, 2, fn p -> %{p | x: clamp.(p1x + d)} end)
     end}
  end

# {action id, label, airborne?}
opp_actions = [
  {20, "dash", false},
  {21, "run", false},
  {24, "jumpsquat", false},
  {25, "jump (air)", true},
  {29, "fall (air)", true},
  {179, "shield", false},
  {44, "jab", false},
  {63, "usmash", false},
  {212, "grab", false},
  {75, "hitstun", false},
  {361, "shine", false}
]

opp_action =
  for {id, name, air?} <- opp_actions do
    {:opp_action, "opp #{name} (#{id})",
     fn gs ->
       upd.(gs, 2, fn p ->
         %{p | action: id, action_frame: 2, on_ground: not air?, y: if(air?, do: 25.0, else: p.y)}
       end)
     end}
  end

facing = [
  {:facing, "p1 facing flipped (#{-base_p1.facing})",
   fn gs -> upd.(gs, 1, fn p -> %{p | facing: -p.facing} end) end},
  {:facing, "p2 facing flipped",
   fn gs -> upd.(gs, 2, fn p -> %{p | facing: -p.facing} end) end}
]

mirror = [
  {:mirror, "stage mirrored (x -> -x, both facings flipped)",
   fn gs ->
     gs
     |> upd.(1, fn p -> %{p | x: -p.x, facing: -p.facing} end)
     |> upd.(2, fn p -> %{p | x: -p.x, facing: -p.facing} end)
   end}
]

own_pct =
  for pct <- [30.0, 60.0, 100.0, 150.0] do
    {:own_pct, "p1 #{trunc(pct)}%", fn gs -> upd.(gs, 1, fn p -> %{p | percent: pct} end) end}
  end

opp_pct =
  for pct <- [30.0, 60.0, 100.0, 150.0] do
    {:opp_pct, "p2 #{trunc(pct)}%", fn gs -> upd.(gs, 2, fn p -> %{p | percent: pct} end) end}
  end

# Parser's character numbering (the fixture's Fox is #{base_p2.character});
# labels are ids only — read them against the Player embedding's table.
opp_char =
  for id <- [0, 2, 7, 16, 18, 22], id != base_p2.character do
    {:opp_char, "p2 character id #{id}", fn gs -> upd.(gs, 2, fn p -> %{p | character: id} end) end}
  end

all_axes = %{
  pair_x: pair_x,
  opp_dx: opp_dx,
  opp_action: opp_action,
  facing: facing,
  mirror: mirror,
  own_pct: own_pct,
  opp_pct: opp_pct,
  opp_char: opp_char
}

axes =
  case opts[:axes] do
    nil -> Map.keys(all_axes)
    s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_atom/1)
  end

cells = [{:baseline, "baseline (unperturbed fixture)", & &1}] ++ Enum.flat_map(axes, &Map.get(all_axes, &1, []))

# ---------------------------------------------------------------------------
# One cell = one teacher-forced pass over the stream; probe at loop frames.
# Returns %{key => [%{b: pB, x: pX}]}.
# ---------------------------------------------------------------------------
loop_set = MapSet.new(loop_idx)

run_cell = fn perturb ->
  Agent.reset_buffer(agent)

  frames
  |> Enum.with_index()
  |> Enum.reduce(%{}, fn {f, i}, acc ->
    gs = perturb.(f.game_state)
    probe? = MapSet.member?(loop_set, i)

    case Agent.observe(agent, gs, f.controller, player_port: 1, probe: probe?, temperature: temperature) do
      {:ok, %{buttons: b}} ->
        key = key_of.(f.game_state)
        Map.update(acc, key, [%{b: Enum.at(b, 1), x: Enum.at(b, 2)}], &[%{b: Enum.at(b, 1), x: Enum.at(b, 2)} | &1])

      :ok ->
        acc

      {:error, reason} ->
        raise "observe failed at frame #{i}: #{inspect(reason)}"
    end
  end)
end

# p(correct) of one probe vs a target (independent sigmoids -> product)
p_correct = fn %{b: pb, x: px}, %{b: tb, x: tx} ->
  (if tb, do: pb, else: 1.0 - pb) * if tx, do: px, else: 1.0 - px
end

score_cell = fn probes, targets ->
  per_key =
    for key <- loop_keys, ps = probes[key], ps != nil, t = targets[key], t != nil, into: %{} do
      {key, Enum.sum(Enum.map(ps, &p_correct.(&1, t))) / length(ps)}
    end

  vals = Map.values(per_key)

  %{
    per_key: per_key,
    mean: if(vals == [], do: nil, else: Enum.sum(vals) / length(vals)),
    min: if(vals == [], do: nil, else: Enum.min(vals)),
    min_key: if(vals == [], do: nil, else: per_key |> Enum.min_by(&elem(&1, 1)) |> elem(0))
  }
end

# ---------------------------------------------------------------------------
# Baseline + offset selection
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("baseline pass (#{n} frames, probing #{length(loop_idx)})...")
base_probes = run_cell.(& &1)

by_offset = Map.new(0..6, fn o -> {o, score_cell.(base_probes, targets_by_offset[o])} end)

Output.puts("baseline p(correct) by label offset: " <>
  Enum.map_join(0..6, "  ", fn o -> "off#{o}=#{Float.round((by_offset[o].mean || 0.0) * 1.0, 3)}" end))

offset =
  case offset_opt do
    "auto" -> Enum.max_by(0..6, fn o -> by_offset[o].mean || 0.0 end)
    s -> String.to_integer(s)
  end

targets = targets_by_offset[offset]
Output.puts("using label offset #{offset}")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
fmt = fn v -> if v == nil, do: "  -  ", else: String.pad_leading(:erlang.float_to_binary(v * 1.0, decimals: 3), 5) end
kfmt = fn {a, af, g} -> "#{a}/#{af}#{if g, do: "g", else: "a"}" end

rows =
  cells
  |> Enum.with_index()
  |> Enum.map(fn {{axis, label, perturb}, ci} ->
    IO.write(:stderr, "\r  cell #{ci + 1}/#{length(cells)}: #{label}\e[K")
    probes = if axis == :baseline, do: base_probes, else: run_cell.(perturb)
    s = score_cell.(probes, targets)
    %{axis: axis, label: label, mean: s.mean, min: s.min, min_key: s.min_key, per_key: s.per_key}
  end)

IO.write(:stderr, "\r\e[K")

base_mean = hd(rows).mean

Output.puts("")
Output.puts("| axis | cell | p(correct) mean | delta vs baseline | min | at |")
Output.puts("|---|---|---:|---:|---:|---|")

for r <- rows do
  d = if r.mean && base_mean, do: r.mean - base_mean, else: nil
  Output.puts("| #{r.axis} | #{r.label} | #{fmt.(r.mean)} | #{if d, do: (if d >= 0, do: "+", else: "") <> fmt.(d), else: "  -  "} | #{fmt.(r.min)} | #{if r.min_key, do: kfmt.(r.min_key), else: "-"} |")
end

Output.puts("")
Output.puts("Per-key detail (columns = loop states #{Enum.map_join(loop_keys, " ", kfmt)}):")

for r <- rows do
  Output.puts(String.pad_trailing(String.slice(r.label, 0, 44), 45) <> Enum.map_join(loop_keys, " ", fn k -> fmt.(r.per_key[k]) end))
end

Output.puts("")
Output.puts("Reading: baseline is the fixture-like state the policy was trained on. A row whose mean")
Output.puts("drops well below baseline is a state family the policy has NOT covered — aim DAgger rollouts")
Output.puts("there (opponent action/distance rows are the CPU-gate failures; pair_x rows are stage position).")

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))

  File.write!(
    out,
    Jason.encode!(
      %{
        policy: policy_path,
        fixture: fixture,
        frames: n,
        loop_frames: length(loop_idx),
        delay_id: opts[:delay_id] || 0,
        temperature: temperature,
        label_offset: offset,
        baseline_by_offset: Map.new(by_offset, fn {o, s} -> {o, s.mean} end),
        targets: Map.new(targets, fn {k, v} -> {kfmt.(k), v} end),
        rows:
          Enum.map(rows, fn r ->
            %{r | min_key: r.min_key && kfmt.(r.min_key), per_key: Map.new(r.per_key, fn {k, v} -> {kfmt.(k), v} end)}
          end)
      },
      pretty: true
    )
  )

  Output.success("wrote #{out}")
end

GenServer.stop(agent)
