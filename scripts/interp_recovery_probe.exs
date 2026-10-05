# Recovery interp probe (2026-10-05): is the up-B stick emitted where the
# expert emits it, and does that read the player's height?
#
#   mix run scripts/interp_recovery_probe.exs --policy P --label L \
#     [--split SPLIT.json] [--games 24] [--batch 256] [--out FILE.json]
#
# Over held-out expert OFFSTAGE frames (teacher-forced, same embedding as
# training), three readouts on the AR head's own distributions:
#
#  Q1 press joint   — on frames where the expert presses B (edge), with the
#                     head teacher-forced on that press: P(stick up / side /
#                     neutral) from the main-stick heads vs the expert's
#                     actual stick zone, by height band. The live finding
#                     (scripts/b_press_stick.exs): bots press B with a
#                     neutral stick 20-28 % of the time, the expert 2.5 %.
#  Q2 B emission    — on actionable offstage frames, P(B) where the expert
#                     presses B within 3 frames vs where it does not.
#  Q3 height use    — at low/deep press frames, the own-y dims of the whole
#                     window are set to a high-band value (and vice versa);
#                     change in P(up) / P(side). Control: opponent-percent
#                     dims swapped the same way.
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
      %{game: gi, t: t, action: a, press: press?, b_soon: b_soon?, prev_b: prev.buttons.b, prev_zone: zone_of.(prev), height: height.(p.y || 0.0), jumps: p.jumps_left || 0, zone: zone_of.(a)}
    end
  end)
presses = Enum.filter(samples, & &1.press)
Output.puts("#{length(samples)} offstage actionable frames; #{length(presses)} B presses: " <>
  inspect(Enum.frequencies_by(presses, & &1.height)))

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

# Q1 + Q2 in one pass over all offstage frames
rows = run.(samples, windows, :tf)
press_rows = Enum.filter(rows, & &1.press)
# Q1b: the same press frames with the head's previous-stick input centred, and
# with the own-stick history zeroed over the whole window as well
q1b = %{neutral_prev_stick: run.(presses, windows, :neutral_prev_stick), no_stick_history: run.(presses, windows, :no_stick_history)}

Output.puts("RESULT #{label} Q1 press joint (teacher-forced on the expert's B press) expert zone share | model P: " <>
  Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
    l = Enum.filter(press_rows, &(&1.height == h))
    "#{h} n=#{length(l)} up #{share.(l, &(&1.zone == :up))}|#{mean.(l, :up)} side #{share.(l, &(&1.zone == :side))}|#{mean.(l, :side)} neutral #{share.(l, &(&1.zone == :neutral))}|#{mean.(l, :neutral)}"
  end))
# the expert's own up presses: how much up does the model put there?
Output.puts("RESULT #{label} Q1 on expert UP presses, model P(up) by height: " <>
  Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
    l = Enum.filter(press_rows, &(&1.height == h and &1.zone == :up))
    "#{h} n=#{length(l)} P(up) #{mean.(l, :up)} P(side) #{mean.(l, :side)} P(neutral) #{mean.(l, :neutral)}"
  end))
for {mode, mrows} <- q1b do
  Output.puts("RESULT #{label} Q1b #{mode}: on expert UP presses, model P(up)/P(side)/P(neutral) by height: " <>
    Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
      l = Enum.filter(mrows, &(&1.height == h and &1.zone == :up))
      "#{h} n=#{length(l)} #{mean.(l, :up)}/#{mean.(l, :side)}/#{mean.(l, :neutral)}"
    end))
end
# press probability: frames where B was NOT held on t-1; expert presses at t vs does not
Output.puts("RESULT #{label} Q2 P(B press) on offstage frames with B up at t-1, expert presses at t | does not: " <>
  Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
    l = Enum.filter(rows, &(&1.height == h and not &1.prev_b))
    yes = Enum.filter(l, & &1.press)
    no = Enum.reject(l, & &1.press)
    "#{h} #{mean.(yes, :b)} (n=#{length(yes)}) | #{mean.(no, :b)} (n=#{length(no)}; expert rate #{share.(l, & &1.press)})"
  end))

# Q2b: the stick is already up (the expert's up-B setup): per-frame B-press hazard
Output.puts("RESULT #{label} Q2b P(B press) with stick up at t-1 and B up at t-1, model mean | expert hazard, by height: " <>
  Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
    l = Enum.filter(rows, &(&1.height == h and &1.prev_zone == :up and not &1.prev_b))
    "#{h} #{mean.(l, :b)} | #{share.(l, & &1.press)} (n=#{length(l)})"
  end))

# Q2c: does the button head condition on the stick zone? B-press rate by the
# zone of the previous stick (the head's own input) on offstage frames, B up at t-1
Output.puts("RESULT #{label} Q2c P(B press) by previous stick zone, model mean | expert rate (n): " <>
  Enum.map_join([:up, :side, :down, :neutral], "  ", fn z ->
    l = Enum.filter(rows, &(&1.prev_zone == z and not &1.prev_b))
    "#{z} #{mean.(l, :b)} | #{share.(l, & &1.press)} (#{length(l)})"
  end))

# Q4: the stick-UP event offstage (expert goes from not-up to up, no B held): the
# decision the expert makes before an up-B. Model P(up) there vs on frames where
# the expert keeps the stick not-up, by height; then the y ablation on the event frames.
up_events = Enum.filter(samples, &(&1.zone == :up and &1.prev_zone != :up and not &1.prev_b))
stay_down = Enum.filter(samples, &(&1.zone != :up and &1.prev_zone != :up and not &1.prev_b))
ev_rows = Enum.filter(rows, &(&1.zone == :up and &1.prev_zone != :up and not &1.prev_b))
sd_rows = Enum.filter(rows, &(&1.zone != :up and &1.prev_zone != :up and not &1.prev_b))
Output.puts("RESULT #{label} Q4 stick-up events offstage, model P(up) on the event frame | on stay-down frames (expert event rate), by height: " <>
  Enum.map_join([:high, :ledge, :low, :deep], "  ", fn h ->
    e = Enum.filter(ev_rows, &(&1.height == h))
    d = Enum.filter(sd_rows, &(&1.height == h))
    "#{h} #{mean.(e, :up)} (n=#{length(e)}) | #{mean.(d, :up)} (n=#{length(d)}; rate #{Float.round(length(e) / max(length(e) + length(d), 1), 3)})"
  end))
Output.puts("RESULT #{label} Q4 by jumps left on low+deep event frames, model P(up): " <>
  Enum.map_join([0, 1], "  ", fn j ->
    e = Enum.filter(ev_rows, &(&1.height in [:low, :deep] and &1.jumps == j))
    d = Enum.filter(sd_rows, &(&1.height in [:low, :deep] and &1.jumps == j))
    "jumps=#{j} event #{mean.(e, :up)} (n=#{length(e)}) | stay #{mean.(d, :up)} (n=#{length(d)})"
  end))
_ = {up_events, stay_down}

# Q3: height ablation at press frames — own-y dims of the whole window set to a donor's
low_presses = Enum.filter(presses, &(&1.height in [:low, :deep]))
high_presses = Enum.filter(presses, &(&1.height == :high))
:rand.seed(:exsss, {5, 5, 5})

ablate = fn list, donors, dims_to_swap ->
  swap_idx = Nx.tensor(dims_to_swap)
  run.(list, fn chunk ->
    states = windows.(chunk)
    donor = chunk |> Enum.map(fn _ -> window_of.(Enum.random(donors)) end) |> Nx.stack()
    mask = Nx.tensor(Enum.map(0..(embed_size - 1), fn d -> if d in dims_to_swap, do: 1.0, else: 0.0 end)) |> Nx.backend_transfer(EXLA.Backend)
    _ = swap_idx
    Nx.add(Nx.multiply(states, Nx.subtract(1.0, mask)), Nx.multiply(donor, mask))
  end, :tf)
end

q3 =
  if low_presses != [] and high_presses != [] do
    base_low = Enum.filter(press_rows, &(&1.height in [:low, :deep]))
    base_high = Enum.filter(press_rows, &(&1.height == :high))
    low_to_high = ablate.(low_presses, high_presses, y_dims)
    low_ctrl = ablate.(low_presses, high_presses, ctrl_dims)
    high_to_low = ablate.(high_presses, low_presses, y_dims)
    high_ctrl = ablate.(high_presses, low_presses, ctrl_dims)
    ev_low = Enum.filter(up_events, &(&1.height in [:low, :deep]))
    ev_high = Enum.filter(up_events, &(&1.height == :high))
    ev = if ev_low != [] and ev_high != [] do
      base = Enum.filter(ev_rows, &(&1.height in [:low, :deep]))
      to_high = ablate.(ev_low, ev_high, y_dims)
      ctrl = ablate.(ev_low, ev_high, ctrl_dims)
      Output.puts("RESULT #{label} Q4 y ablation on low+deep stick-up event frames, P(up): base #{mean.(base, :up)} (n=#{length(base)}) -> y:=high #{mean.(to_high, :up)}, control #{mean.(ctrl, :up)}")
      %{base: mean.(base, :up), y_to_high: mean.(to_high, :up), control: mean.(ctrl, :up), n: length(base)}
    end
    r = %{
      q4_event_ablation: ev,
      low_base: %{up: mean.(base_low, :up), side: mean.(base_low, :side), n: length(base_low)},
      low_y_to_high: %{up: mean.(low_to_high, :up), side: mean.(low_to_high, :side)},
      low_ctrl: %{up: mean.(low_ctrl, :up), side: mean.(low_ctrl, :side)},
      high_base: %{up: mean.(base_high, :up), side: mean.(base_high, :side), n: length(base_high)},
      high_y_to_low: %{up: mean.(high_to_low, :up), side: mean.(high_to_low, :side)},
      high_ctrl: %{up: mean.(high_ctrl, :up), side: mean.(high_ctrl, :side)}
    }
    Output.puts("RESULT #{label} Q3 height ablation at press frames, P(up)/P(side): " <>
      "low+deep base #{r.low_base.up}/#{r.low_base.side} (n=#{r.low_base.n}) -> y:=high #{r.low_y_to_high.up}/#{r.low_y_to_high.side}, control #{r.low_ctrl.up}/#{r.low_ctrl.side}  |  " <>
      "high base #{r.high_base.up}/#{r.high_base.side} (n=#{r.high_base.n}) -> y:=low #{r.high_y_to_low.up}/#{r.high_y_to_low.side}, control #{r.high_ctrl.up}/#{r.high_ctrl.side}")
    r
  else
    Output.warning("Q3 skipped: no low/high press frames")
    nil
  end

# Q5 (10-05, the silent fall): the bot's decided-trip deaths are input-free
# falls with the double jump in hand, and its hazard of resuming input DECAYS
# with the length of the silence closed-loop (0.20 -> 0.09 per 3 f) where the
# expert's rises (0.17 -> 0.23). Teacher-forced: on expert offstage frames whose
# previous input was neutral, P(any input now) model vs expert, binned by how
# long the expert had already been silent. Flat/calibrated here = the decay is
# closed-loop (states the expert never produces); decaying here = structural.
silent? = fn a -> zone_of.(a) == :neutral and not a.buttons.b and not a.buttons.x and not a.buttons.y end
silence_len = fn g, t ->
  Enum.reduce_while(1..60, 0, fn k, acc ->
    if t - k >= 0 and silent?.(elem(g.actions, t - k)), do: {:cont, acc + 1}, else: {:halt, acc}
  end)
end
cliff? = fn p -> (p.action || 0) in 252..263 end
q5_samples =
  samples
  # not on the ledge (CLIFF_* is airborne + offstage + input-free for dozens of frames)
  |> Enum.reject(fn s -> cliff?.(elem(Enum.at(games, s.game).players, s.t)) end)
  |> Enum.filter(fn s -> silent?.(elem(Enum.at(games, s.game).actions, s.t - 1)) end)
  |> Enum.map(fn s ->
    g = Enum.at(games, s.game)
    Map.merge(s, %{k: silence_len.(g, s.t), active: not silent?.(s.action), below: s.height != :high})
  end)
k_bin = fn k -> cond do k <= 3 -> "1-3"; k <= 6 -> "4-6"; k <= 12 -> "7-12"; k <= 24 -> "13-24"; k <= 48 -> "25-48"; true -> "49+" end end
k_bins = ["1-3", "4-6", "7-12", "13-24", "25-48", "49+"]
q5_rows = run.(q5_samples, windows, :tf)
p_active = fn r -> 1.0 - r.neutral * (1.0 - r.b) * (1.0 - r.jump) end
q5 =
  Map.new([true, false], fn below ->
    {if(below, do: :below_stage, else: :above_stage),
     Map.new(k_bins, fn bin ->
       l = Enum.filter(q5_rows, &(&1.below == below and k_bin.(&1.k) == bin))
       {bin, %{n: length(l), expert: share.(l, & &1.active),
               model: if(l == [], do: nil, else: Float.round(Enum.sum(Enum.map(l, p_active)) / length(l), 3)),
               model_up: mean.(l, :up), expert_up: share.(l, &(&1.zone == :up)),
               model_jump: mean.(l, :jump), expert_jump: share.(l, &(&1.action.buttons.x or &1.action.buttons.y))}}
     end)}
  end)
for {where, bins} <- q5 do
  Output.puts("RESULT #{label} Q5 P(any input | silent so far k frames) #{where} model|expert (n): " <>
    Enum.map_join(k_bins, "  ", fn bin -> b = bins[bin]; "k#{bin} #{b.model}|#{b.expert} (#{b.n})" end))
  Output.puts("RESULT #{label} Q5 P(stick up) / P(jump) by silence #{where} model|expert: " <>
    Enum.map_join(k_bins, "  ", fn bin -> b = bins[bin]; "k#{bin} up #{b.model_up}|#{b.expert_up} jump #{b.model_jump}|#{b.expert_jump}" end))
end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  summary = %{
    q5: q5,
    label: label, frames: length(rows), presses: length(press_rows),
    q1: Map.new([:high, :ledge, :low, :deep], fn h ->
      l = Enum.filter(press_rows, &(&1.height == h))
      {h, %{n: length(l), expert_up: share.(l, &(&1.zone == :up)), model_up: mean.(l, :up), expert_side: share.(l, &(&1.zone == :side)),
            model_side: mean.(l, :side), expert_neutral: share.(l, &(&1.zone == :neutral)), model_neutral: mean.(l, :neutral)}}
    end),
    q1b: Map.new(q1b, fn {mode, mrows} ->
      {mode, Map.new([:high, :ledge, :low, :deep], fn h ->
        l = Enum.filter(mrows, &(&1.height == h and &1.zone == :up))
        {h, %{n: length(l), up: mean.(l, :up), side: mean.(l, :side), neutral: mean.(l, :neutral)}}
      end)}
    end),
    q2: Map.new([:high, :ledge, :low, :deep], fn h ->
      l = Enum.filter(rows, &(&1.height == h and not &1.prev_b))
      {h, %{press: mean.(Enum.filter(l, & &1.press), :b), no_press: mean.(Enum.reject(l, & &1.press), :b), expert_rate: share.(l, & &1.press)}}
    end),
    q3: q3
  }
  File.write!(out, Jason.encode!(summary, pretty: true))
end
