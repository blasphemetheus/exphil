# Recovery AIM probe (2026-10-08, INPUT_COHERENCE "10-08 12:00"): once the
# double jump is spent, does the model move the stick UP where the expert
# does? Closed-loop the bot aims up at 1/4 the expert rate (up-onset 3.4 %
# vs 12.3 % per 3 f) while P(B | stick up) is above the expert — the aim is
# the missing half of the Firefox. Teacher-forced over held-out expert
# offstage frames with the jump spent, below the ledge, previous stick NOT
# up: expert share of "stick up this frame" (the aim onset hazard) vs the
# model P(up) from the main_y head (hold/change collapsed on the previous
# bucket). Same ≈ the loop states differ (distribution shift, the DAgger
# case); lower = the aim was never learned (weighting / label).
#
#   mix run scripts/recovery_aim_probe.exs --policy P --label L [--games 24]
#
# Shares the embedding/forward code of scripts/interp_recovery_probe.exs.
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

Output.banner("Interp recovery probe: #{label}")

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
y_dims = dims.own_y
ctrl_dims = dims.opp_percent
Output.puts("own-y dims #{inspect(y_dims)}  control (opp percent) dims #{inspect(ctrl_dims)}")

# ---- samples: offstage, actionable (not stunned, not dead, not helpless) --------
height = fn y -> cond do y > 0.0 -> :high; y > -20.0 -> :ledge; y > -60.0 -> :low; true -> :deep end end
offstage? = fn p, edge -> p.on_ground != true and (p.action || 99) > 13 and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0) end
actionable? = fn p -> (p.hitstun_frames_left || 0) == 0 and (p.action || 0) != 35 end
zone_of = fn a ->
  cx = (a.main_x + 0.5) / axis_buckets - 0.5
  cy = (a.main_y + 0.5) / axis_buckets - 0.5
  cond do cy >= 0.33 -> :up; cy <= -0.33 -> :down; abs(cx) >= 0.33 -> :side; true -> :neutral end
end

samples =
  games
  |> Enum.with_index()
  |> Enum.flat_map(fn {g, gi} ->
    for t <- (window - 1)..(g.n - 5),
        p = elem(g.players, t),
        offstage?.(p, g.edge) and actionable?.(p) do
      a = elem(g.actions, t)
      prev = elem(g.actions, t - 1)
      press? = a.buttons.b and not prev.buttons.b
      b_soon? = Enum.any?(0..3, fn j -> elem(g.actions, t + j).buttons.b and not elem(g.actions, t + j - 1).buttons.b end)
      jump_press? = (a.buttons.x and not prev.buttons.x) or (a.buttons.y and not prev.buttons.y)
      %{game: gi, t: t, action: a, press: press?, b_soon: b_soon?, prev_b: prev.buttons.b, prev_zone: zone_of.(prev), height: height.(p.y || 0.0), jumps: p.jumps_left || 0, zone: zone_of.(a),
        jump_press: jump_press?, prev_jump_held: prev.buttons.x or prev.buttons.y, airborne: p.on_ground != true}
    end
  end)
presses = Enum.filter(samples, & &1.press)
Output.puts("#{length(samples)} offstage actionable frames; #{length(presses)} B presses: " <>
  inspect(Enum.frequencies_by(presses, & &1.height)))

# ---- forward ----------------------------------------------------------------------
predict = heads.predict_fn
# stick_duration checkpoints (2026-10-06) carry a 7th (duration) head and need
# the hold age of the previous main-stick pair ("prev_age", read off the
# window's prev-action slots); the probes read the six main heads.
stick_duration? = Map.get(config, :stick_duration, Map.get(json_cfg, "stick_duration")) != nil
predict = fn p, i ->
  case predict.(p, i) do
    {b, mx, my, cx, cy, sh, _dur} -> {b, mx, my, cx, cy, sh}
    out -> out
  end
end
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
        inputs = Map.put(inputs, "prev_sticks", buckets)
        if stick_duration?,
          do: Map.put(inputs, "prev_age", ExPhil.Training.Imitation.Loss.prev_age_from_window(s, prev_off, axis_buckets)),
          else: inputs
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
  pb = Nx.sigmoid(b)
  %{up: p_up, down: p_down, side: Nx.multiply(p_y_mid, p_x_side),
    neutral: Nx.multiply(p_y_mid, Nx.subtract(1.0, p_x_side)), b: pb[[.., 1]],
    jump: Nx.reduce_max(pb[[.., 2..3]], axes: [1])}
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

# ---- Q9: the aim onset once the jump is spent --------------------------------------
rows = run.(samples, windows, :tf)
aim = Enum.filter(rows, &(&1.jumps == 0 and &1.height in [:low, :deep] and &1.prev_zone != :up and &1.airborne))
by = fn l, k -> Enum.filter(l, &(&1.height == k)) end
Output.puts("RESULT #{label} Q9 aim onset once the jump is spent (prev stick not up) expert share up | model P(up) (n): " <>
  Enum.map_join([:low, :deep], "  ", fn h ->
    l = by.(aim, h)
    "#{h} #{share.(l, &(&1.zone == :up))}|#{mean.(l, :up)} (#{length(l)})"
  end))
# the same with a jump in hand (the expert aims rarely here — it jumps)
in_hand = Enum.filter(rows, &(&1.jumps > 0 and &1.height in [:low, :deep] and &1.prev_zone != :up and &1.airborne))
Output.puts("RESULT #{label} Q9 aim onset with a jump in hand (control) expert share up | model P(up) (n): " <>
  Enum.map_join([:low, :deep], "  ", fn h ->
    l = by.(in_hand, h)
    "#{h} #{share.(l, &(&1.zone == :up))}|#{mean.(l, :up)} (#{length(l)})"
  end))
# by previous zone: from side (the drift) vs from neutral
Output.puts("RESULT #{label} Q9 aim onset (jump spent, low+deep) by previous zone expert share up | model P(up) (n): " <>
  Enum.map_join([:side, :neutral, :down], "  ", fn z ->
    l = Enum.filter(aim, &(&1.prev_zone == z))
    "#{z} #{share.(l, &(&1.zone == :up))}|#{mean.(l, :up)} (#{length(l)})"
  end))
# and the model P(up) on the frames where the expert DID aim vs did not
did = Enum.filter(aim, &(&1.zone == :up)); not_did = Enum.reject(aim, &(&1.zone == :up))
Output.puts("RESULT #{label} Q9 model P(up) where the expert aimed | where it did not: #{mean.(did, :up)} (#{length(did)}) | #{mean.(not_did, :up)} (#{length(not_did)})")
if opts[:out] do
  File.write!(opts[:out], Jason.encode!(%{label: label, n: length(aim), expert_up: share.(aim, &(&1.zone == :up)), model_up: mean.(aim, :up)}))
end
