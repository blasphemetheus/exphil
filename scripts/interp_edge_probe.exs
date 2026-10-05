# Edge interp probe (2026-10-05): does the policy walk, wavedash or illusion
# OFF the stage, and does that read where the edge is?
#
#   mix run scripts/interp_edge_probe.exs --policy P --label L \
#     [--split SPLIT.json] [--games 24] [--batch 256] [--out FILE.json]
#
# Over held-out expert ONSTAGE actionable frames (teacher-forced), bucketed
# by distance to the nearer edge (near < 20 u / mid < 60 / far) and by
# whether the player FACES that edge:
#
#  Q1 walk-off hazards — model P(B press), P(main stick toward the edge,
#                        side zone), their product (illusion-off proxy) vs
#                        the expert's label rates on the same frames.
#  Q2 edge use         — on near/facing-edge frames, own-x dims (and the
#                        facing dims) of the whole window set to a far
#                        frame's values; change in the hazards. Control:
#                        opponent-percent dims.
#
# NO-MIX LAW: run only with no other live beam.
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Attribution}
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Sim.GA
alias ExPhil.Training.{Data, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, split: :string, games: :integer, out: :string, batch: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
split = opts[:split] || Path.join(Path.dirname(policy), "split.json")
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation") |> Enum.take(opts[:games] || 24)
batch = opts[:batch] || 256

Output.banner("Interp edge probe: #{label}")

heads = Activations.load_heads(policy)
config = heads.config
window = heads.window
embed_size = Map.fetch!(config, :embed_size)
truthy? = fn v -> v == true or v == "true" end
json_cfg =
  case File.read(Path.join(Path.dirname(policy), "model_config.json")) do
    {:ok, s} -> Jason.decode!(s)
    _ -> %{}
  end
flag = fn key -> truthy?.(Map.get(config, key, Map.get(json_cfg, to_string(key), false))) end
use_prev? = flag.(:use_prev_action)
quant? = flag.(:prev_action_quantize)
stage_internals? = flag.(:stage_internals)
with_projectiles? = not (Map.get(json_cfg, "with_projectiles") in [false, "false"])
axis_buckets = Map.get(config, :axis_buckets, 16)
button_events? = flag.(:button_events)
stick_events? = flag.(:stick_events)
Output.config([{"window", window}, {"embed", embed_size}, {"prev-action", use_prev?}, {"quantized channel", quant?},
  {"event heads", "buttons #{button_events?} sticks #{stick_events?}"}, {"games", length(files)}])
embed_cfg = %{ExPhil.Embeddings.Game.Config.default() | stage_internals: stage_internals?, with_projectiles: with_projectiles?}

# ---- embed the held-out games exactly as training did ---------------------------
games =
  Enum.map(files, fn path ->
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    opp = Enum.find(meta.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(path, player_port: own.port)
    frames =
      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))

    ds = Data.from_frames(frames)
    ds = %{ds | embed_config: embed_cfg}
    ds = Data.precompute_frame_embeddings(ds, use_prev_action: use_prev?, prev_action_quantize: quant?, show_progress: false)
    emb = ds.embedded_frames
    emb = if is_list(emb), do: Nx.concatenate(emb), else: emb
    {n, ^embed_size} = Nx.shape(emb)
    actions = Enum.map(frames, &Data.controller_to_action(&1.controller, axis_buckets: axis_buckets))
    players = Enum.map(frames, & &1.game_state.players[1])
    Output.puts("  #{Path.basename(path)}: #{n} frames (stage #{meta.stage})")
    %{emb: Nx.backend_transfer(emb, EXLA.Backend), actions: List.to_tuple(actions), players: List.to_tuple(players), n: n,
      edge: GA.stage_edge(meta.stage)}
  end)

dims = Attribution.discover_dims(embed_cfg, use_prev_action: use_prev?)
[prev_off, 13] = Attribution.prev_action_dim_range(config: embed_cfg)
x_dims = dims.own_x
facing_dims = dims.own_facing
ctrl_dims = dims.opp_percent
Output.puts("own-x dims #{inspect(x_dims)}  facing dims #{inspect(facing_dims)}  control (opp percent) dims #{inspect(ctrl_dims)}")

# ---- forward ----------------------------------------------------------------------
predict = heads.predict_fn
params = heads.params

with_states = fn tf, s, mode ->
  s =
    if mode == :no_stick_history do
      idx = Nx.iota({Nx.axis_size(s, 2)})
      keep = Nx.logical_or(Nx.less(idx, prev_off + 8), Nx.greater_equal(idx, prev_off + 12))
      Nx.multiply(s, Nx.as_type(keep, Nx.type(s)))
    else
      s
    end

  if button_events? or stick_events? do
    (fn tf, s ->
      last = s |> Nx.slice_along_axis(Nx.axis_size(s, 1) - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      idx = Nx.iota({Nx.axis_size(s, 2)})
      keep = Nx.logical_or(Nx.less(idx, prev_off), Nx.greater_equal(idx, prev_off + 13))
      inputs = Map.put(tf, "state_sequence", Nx.multiply(s, Nx.as_type(keep, Nx.type(s))))
      inputs = if button_events?, do: Map.put(inputs, "prev_buttons", Nx.slice_along_axis(last, prev_off, 8, axis: 1)), else: inputs
      if stick_events? do
        buckets =
          last |> Nx.slice_along_axis(prev_off + 8, 4, axis: 1) |> Nx.as_type(:f32)
          |> Nx.divide(2.0) |> Nx.add(0.5) |> Nx.multiply(axis_buckets) |> Nx.floor()
          |> Nx.clip(0, axis_buckets - 1) |> Nx.as_type(:s64)
        buckets = if mode == :tf, do: buckets, else: Nx.broadcast(Nx.tensor(div(axis_buckets, 2), type: :s64), Nx.shape(buckets))
        Map.put(inputs, "prev_sticks", buckets)
      else
        inputs
      end
    end).(tf, s)
  else
    Map.put(tf, "state_sequence", s)
  end
end

# zone probabilities from the main-stick heads (marginals; x and y treated as independent)
n_classes = axis_buckets + 1
centers = Nx.tensor(for k <- 0..(n_classes - 1), do: (k + 0.5) / axis_buckets - 0.5)
up_mask = Nx.greater_equal(centers, 0.33) |> Nx.as_type(:f32)
down_mask = Nx.less_equal(centers, -0.33) |> Nx.as_type(:f32)
side_mask = Nx.greater_equal(Nx.abs(centers), 0.33) |> Nx.as_type(:f32)
zone_probs = fn {b, mx, my, _cx, _cy, _sh} ->
  px = Axon.Activations.softmax(mx)
  py = Axon.Activations.softmax(my)
  p_up = Nx.sum(Nx.multiply(py, up_mask), axes: [1])
  p_down = Nx.sum(Nx.multiply(py, down_mask), axes: [1])
  p_y_mid = Nx.subtract(1.0, Nx.add(p_up, p_down))
  p_x_side = Nx.sum(Nx.multiply(px, side_mask), axes: [1])
  %{up: p_up, down: p_down, side: Nx.multiply(p_y_mid, p_x_side),
    neutral: Nx.multiply(p_y_mid, Nx.subtract(1.0, p_x_side)), b: Nx.sigmoid(b)[[.., 1]]}
end

window_of = fn s -> Nx.slice_along_axis(Enum.at(games, s.game).emb, s.t - window + 1, window, axis: 0) end
forward = fn chunk, states, mode ->
  tgt = Data.actions_to_tensors(Enum.map(chunk, & &1.action)) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
  predict.(params, with_states.(Heads.tf_inputs(tgt), states, mode)) |> zone_probs.()
end
to_rows = fn chunk, zp ->
  lists = Map.new(zp, fn {k, v} -> {k, Nx.to_flat_list(v)} end)
  chunk |> Enum.with_index() |> Enum.map(fn {s, i} -> Map.merge(s, Map.new(lists, fn {k, l} -> {k, Enum.at(l, i)} end)) end)
end

run = fn list, states_fn, mode ->
  list
  |> Enum.chunk_every(batch)
  |> Enum.flat_map(fn chunk -> to_rows.(chunk, forward.(chunk, states_fn.(chunk), mode)) end)
end
windows = fn chunk -> chunk |> Enum.map(window_of) |> Nx.stack() end

mean = fn l, k -> if l == [], do: nil, else: Float.round(Enum.sum(Enum.map(l, & &1[k])) / length(l), 3) end
share = fn l, pred -> if l == [], do: nil, else: Float.round(Enum.count(l, pred) / length(l), 3) end

# ---- samples: onstage, actionable ------------------------------------------------
actionable? = fn p -> (p.hitstun_frames_left || 0) == 0 and (p.action || 99) > 13 and (p.action || 0) != 35 end
onstage? = fn p, edge -> abs(p.x || 0.0) <= edge and (p.y || 0.0) >= -5.0 end
dist_band = fn d -> cond do d < 20.0 -> :near; d < 60.0 -> :mid; true -> :far end end
# stick x on the frame, signed toward the nearer edge (+ = toward it)
toward_edge = fn a, x ->
  cx = (a.main_x + 0.5) / axis_buckets - 0.5
  if x >= 0, do: cx, else: -cx
end

samples =
  games
  |> Enum.with_index()
  |> Enum.flat_map(fn {g, gi} ->
    for t <- (window - 1)..(g.n - 2),
        p = elem(g.players, t),
        onstage?.(p, g.edge) and actionable?.(p) do
      a = elem(g.actions, t)
      prev = elem(g.actions, t - 1)
      x = p.x || 0.0
      %{game: gi, t: t, action: a, x: x, sign: if(x >= 0, do: 1, else: -1),
        dist: dist_band.(g.edge - abs(x)),
        facing_edge: (p.facing || 1) * x > 0,
        press: a.buttons.b and not prev.buttons.b, prev_b: prev.buttons.b,
        stick_edge: toward_edge.(a, x) >= 0.33, prev_stick_edge: toward_edge.(prev, x) >= 0.33}
    end
  end)
Output.puts("#{length(samples)} onstage actionable frames; near/facing-edge " <>
  "#{Enum.count(samples, &(&1.dist == :near and &1.facing_edge))}")

# P(stick toward the edge) from the main_x head: mass on the side buckets pointing at the nearer edge
edge_mass = fn {_b, mx, _my, _cx, _cy, _sh}, signs ->
  px = Axon.Activations.softmax(mx)
  pos = Nx.greater_equal(centers, 0.33) |> Nx.as_type(:f32)
  neg = Nx.less_equal(centers, -0.33) |> Nx.as_type(:f32)
  s = Nx.reshape(signs, {:auto, 1}) |> Nx.as_type(:f32)
  mask = Nx.add(Nx.multiply(pos, Nx.greater(s, 0)), Nx.multiply(neg, Nx.less(s, 0)))
  Nx.sum(Nx.multiply(px, mask), axes: [1])
end
forward_edge = fn chunk, states, mode ->
  tgt = Data.actions_to_tensors(Enum.map(chunk, & &1.action)) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
  out = predict.(params, with_states.(Heads.tf_inputs(tgt), states, mode))
  signs = Nx.tensor(Enum.map(chunk, & &1.sign))
  %{b: Nx.sigmoid(elem(out, 0))[[.., 1]], edge: edge_mass.(out, signs)}
end
run_edge = fn list, states_fn, mode ->
  list
  |> Enum.chunk_every(batch)
  |> Enum.flat_map(fn chunk ->
    zp = forward_edge.(chunk, states_fn.(chunk), mode)
    lists = Map.new(zp, fn {k, v} -> {k, Nx.to_flat_list(v)} end)
    chunk |> Enum.with_index() |> Enum.map(fn {smp, i} -> Map.merge(smp, Map.new(lists, fn {k, l} -> {k, Enum.at(l, i)} end)) end)
  end)
end

rows = run_edge.(samples, windows, :tf)
buckets = for d <- [:near, :mid, :far], f <- [true, false], do: {d, f}
line = fn l ->
  nb = Enum.reject(l, & &1.prev_b)
  # illusion-off proxy: B press with the stick toward the edge
  "n=#{length(l)} P(B press) #{mean.(nb, :b)}|#{share.(nb, & &1.press)}  P(stick→edge) #{mean.(l, :edge)}|#{share.(l, & &1.stick_edge)}  " <>
    "press∧stick→edge #{if nb == [], do: "-", else: Float.round(Enum.sum(Enum.map(nb, &(&1.b * &1.edge))) / length(nb), 4)}|#{share.(nb, &(&1.press and &1.stick_edge))}"
end
for {d, f} <- buckets do
  l = Enum.filter(rows, &(&1.dist == d and &1.facing_edge == f))
  Output.puts("RESULT #{label} Q1 #{d}/#{if f, do: "facing-edge", else: "facing-in"} model|expert: " <> line.(l))
end

# Q1b: the conditional that fires an illusion off the edge — B press with the
# stick ALREADY toward the edge (the head sees the previous stick)
for {d, f} <- buckets do
  l = Enum.filter(rows, &(&1.dist == d and &1.facing_edge == f and &1.prev_stick_edge and not &1.prev_b))
  Output.puts("RESULT #{label} Q1b #{d}/#{if f, do: "facing-edge", else: "facing-in"} P(B press | prev stick→edge) model|expert: #{mean.(l, :b)}|#{share.(l, & &1.press)} (n=#{length(l)})")
end

# Q2: ablations on near/facing-edge frames
near = Enum.filter(samples, &(&1.dist == :near and &1.facing_edge))
far = Enum.filter(samples, &(&1.dist == :far))
far_facing_in = Enum.filter(samples, &(&1.dist == :near and not &1.facing_edge))
:rand.seed(:exsss, {9, 9, 9})
ablate = fn list, donors, dims_to_swap ->
  run_edge.(list, fn chunk ->
    states = windows.(chunk)
    donor = chunk |> Enum.map(fn _ -> window_of.(Enum.random(donors)) end) |> Nx.stack()
    mask = Nx.tensor(Enum.map(0..(embed_size - 1), fn d -> if d in dims_to_swap, do: 1.0, else: 0.0 end)) |> Nx.backend_transfer(EXLA.Backend)
    Nx.add(Nx.multiply(states, Nx.subtract(1.0, mask)), Nx.multiply(donor, mask))
  end, :tf)
end
q2 =
  if near != [] and far != [] and far_facing_in != [] do
    base = Enum.filter(rows, &(&1.dist == :near and &1.facing_edge))
    x_far = ablate.(near, far, x_dims)
    face_in = ablate.(near, far_facing_in, facing_dims)
    ctrl = ablate.(near, far, ctrl_dims)
    f = fn l -> "P(B press) #{mean.(Enum.reject(l, & &1.prev_b), :b)}  P(stick→edge) #{mean.(l, :edge)}" end
    Output.puts("RESULT #{label} Q2 near/facing-edge frames (n=#{length(base)}): base #{f.(base)}  ->  x:=far #{f.(x_far)}  |  facing:=in #{f.(face_in)}  |  control #{f.(ctrl)}")
    %{base: %{b: mean.(base, :b), edge: mean.(base, :edge)}, x_far: %{b: mean.(x_far, :b), edge: mean.(x_far, :edge)},
      facing_in: %{b: mean.(face_in, :b), edge: mean.(face_in, :edge)}, control: %{b: mean.(ctrl, :b), edge: mean.(ctrl, :edge)}}
  end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  q1 = Map.new(buckets, fn {d, f} ->
    l = Enum.filter(rows, &(&1.dist == d and &1.facing_edge == f))
    nb = Enum.reject(l, & &1.prev_b)
    {"#{d}/#{if f, do: "facing_edge", else: "facing_in"}", %{n: length(l), model_b: mean.(nb, :b), expert_b: share.(nb, & &1.press),
       model_edge: mean.(l, :edge), expert_edge: share.(l, & &1.stick_edge)}}
  end)
  File.write!(out, Jason.encode!(%{label: label, frames: length(rows), q1: q1, q2: q2}, pretty: true))
end
