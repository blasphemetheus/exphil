# Silent-fall probe ON THE BOT'S OWN STATES (2026-10-05 19:10). The
# recovery probe measures the policy on EXPERT offstage frames, where it is
# calibrated through 24 frames of silence. The bot's deaths happen in states
# the expert never produces — 30-50 frames of input-free fall with the
# double jump in hand. Those states exist by the thousand in the sim-DAgger
# export (scripts/sim_recovery_dagger.exs --only-silent: every trip frame is
# one where the policy's own press was neutral, each trip carrying 90
# frames of its own context). This probe embeds them like training, feeds
# the policy teacher-forced with a NEUTRAL previous input (what the bot had
# actually pressed), and reads P(any input) by how long the silence has
# lasted — then ablates what might be driving it:
#   af1       own action_frame := 1 frame in the last 12 window frames
#   shallow   own y := -20 (ledge band) across the window
#   static    the whole window replaced by the last frame (no trajectory history)
#   jumps0    own jumps_left := 0 (what would it do without the jump?)
#
#   mix run scripts/interp_silent_fall_probe.exs --policy P --label L \
#     [--frames data/silent_fall/sim_dagger_r1_silent.frames] [--batch 256] [--out FILE.json]
alias ExPhil.Bridge.ControllerState
alias ExPhil.Interp.{Activations, Attribution}
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Training.{Data, MixFrames, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(), strict: [policy: :string, label: :string, frames: :string, batch: :integer, out: :string, split: :string, donor_games: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
frames_path = opts[:frames] || "data/silent_fall/sim_dagger_r1_silent.frames"
batch = opts[:batch] || 256

Output.banner("Silent-fall probe on bot states: #{label}")

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
embed_cfg = %{ExPhil.Embeddings.Game.Config.default() | stage_internals: stage_internals?, with_projectiles: with_projectiles?}

# ---- the bot's silent-fall trips -------------------------------------------------
{lists, _} = MixFrames.load_lists(frames_path, label_delay: 0)
neutral = ControllerState.neutral()

# every frame's prev is NEUTRAL (the bot was silent): the export's prev is the
# previous label, which is the expert's, not what the bot pressed
lists = Enum.map(lists, fn l -> Enum.map(l, &Map.put(&1, :prev_controller, neutral)) end)
Output.puts("#{length(lists)} trips, #{lists |> List.flatten() |> Enum.count(&(&1[:input_only] != true))} silent frames")

ds = Data.from_frame_lists(lists, embed_config: embed_cfg)
ds = Data.precompute_frame_embeddings(ds, use_prev_action: use_prev?, prev_action_quantize: quant?, show_progress: false)
emb = ds.embedded_frames
emb = if is_list(emb), do: Nx.concatenate(emb), else: emb
{_n, ^embed_size} = Nx.shape(emb)
emb = Nx.backend_transfer(emb, EXLA.Backend)
frames_t = ds.frames |> List.to_tuple()
starts = ds.metadata.sequence_starts

# samples: silent trip frames with at least `window` frames of history inside the same segment
samples =
  for i <- 0..(tuple_size(frames_t) - 1),
      f = elem(frames_t, i),
      f[:input_only] != true,
      i - elem(starts, i) >= window - 1,
      p = f.game_state.players[1],
      do: %{i: i, k: i - Enum.find_index((elem(starts, i))..i, fn j -> elem(frames_t, j)[:input_only] != true end) - elem(starts, i) + 1,
            y: p.y || 0.0, jumps: p.jumps_left || 0, action: p.action || 0}

Output.puts("#{length(samples)} silent frames with a full window; silence length k quartiles #{inspect(samples |> Enum.map(& &1.k) |> Enum.sort() |> then(fn l -> [Enum.at(l, div(length(l), 4)), Enum.at(l, div(length(l), 2)), Enum.at(l, div(3 * length(l), 4))] end))}")

dims = Attribution.discover_dims(embed_cfg, use_prev_action: use_prev?)
[prev_off, 13] = Attribution.prev_action_dim_range(config: embed_cfg)
af_dims = dims[:own_action_frame] || []
y_dims = dims.own_y
jump_dims = dims[:own_jumps] || []
Output.puts("dims: action_frame #{inspect(af_dims)} y #{inspect(y_dims)} jumps #{inspect(jump_dims)}")

# ---- forward (same head-input protocol as the recovery probe) --------------------
predict = heads.predict_fn
params = heads.params

with_states = fn tf, s ->
  if button_events? or stick_events? do
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
      Map.put(inputs, "prev_sticks", buckets)
    else
      inputs
    end
  else
    Map.put(tf, "state_sequence", s)
  end
end

n_classes = axis_buckets + 1
centers = Nx.tensor(for k <- 0..(n_classes - 1), do: (k + 0.5) / axis_buckets - 0.5)
side_mask = Nx.greater_equal(Nx.abs(centers), 0.33) |> Nx.as_type(:f32)
up_mask = Nx.greater_equal(centers, 0.33) |> Nx.as_type(:f32)
down_mask = Nx.less_equal(centers, -0.33) |> Nx.as_type(:f32)
probs = fn {b, mx, my, _cx, _cy, _sh} ->
  px = Axon.Activations.softmax(mx)
  py = Axon.Activations.softmax(my)
  p_up = Nx.sum(Nx.multiply(py, up_mask), axes: [1])
  p_down = Nx.sum(Nx.multiply(py, down_mask), axes: [1])
  p_y_mid = Nx.subtract(1.0, Nx.add(p_up, p_down))
  p_x_side = Nx.sum(Nx.multiply(px, side_mask), axes: [1])
  pb = Nx.sigmoid(b)
  neutral_p = Nx.multiply(p_y_mid, Nx.subtract(1.0, p_x_side))
  %{neutral: neutral_p, up: p_up, b: pb[[.., 1]], jump: Nx.reduce_max(pb[[.., 2..3]], axes: [1])}
end

window_of = fn s -> Nx.slice_along_axis(emb, s.i - window + 1, window, axis: 0) end
# teacher-forced target = neutral (what the bot pressed); only used to shape the head inputs
neutral_action = Data.controller_to_action(neutral, axis_buckets: axis_buckets)

run = fn list, states_fn ->
  list
  |> Enum.chunk_every(batch)
  |> Enum.flat_map(fn chunk ->
    tgt = Data.actions_to_tensors(List.duplicate(neutral_action, length(chunk))) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
    out = predict.(params, with_states.(Heads.tf_inputs(tgt), states_fn.(chunk))) |> probs.()
    lists = Map.new(out, fn {k, v} -> {k, Nx.to_flat_list(v)} end)
    chunk |> Enum.with_index() |> Enum.map(fn {s, i} -> Map.merge(s, Map.new(lists, fn {k, l} -> {k, Enum.at(l, i)} end)) end)
  end)
end

windows = fn chunk -> chunk |> Enum.map(window_of) |> Nx.stack() end
p_active = fn r -> 1.0 - r.neutral * (1.0 - r.b) * (1.0 - r.jump) end
mean = fn l, f -> if l == [], do: nil, else: Float.round(Enum.sum(Enum.map(l, f)) / length(l), 3) end

set_dims = fn dims_to_set, value, last_frames ->
  fn chunk ->
    states = windows.(chunk)
    {_b, w, d} = Nx.shape(states)
    dim_mask = Nx.tensor(Enum.map(0..(d - 1), fn i -> if i in dims_to_set, do: 1.0, else: 0.0 end)) |> Nx.reshape({1, 1, d})
    time_mask = Nx.tensor(Enum.map(0..(w - 1), fn t -> if t >= w - last_frames, do: 1.0, else: 0.0 end)) |> Nx.reshape({1, w, 1})
    mask = Nx.multiply(dim_mask, time_mask) |> Nx.backend_transfer(EXLA.Backend)
    Nx.add(Nx.multiply(states, Nx.subtract(1.0, mask)), Nx.multiply(mask, value))
  end
end

static = fn chunk ->
  states = windows.(chunk)
  last = Nx.slice_along_axis(states, window - 1, 1, axis: 1)
  Nx.broadcast(last, Nx.shape(states))
end

# the embedded value of y = -20 / jumps = 0: embed a reference state and read the dims
ref_emb = fn changes ->
  gs = elem(frames_t, hd(samples).i).game_state
  gs = %{gs | players: Map.update!(gs.players, 1, &struct(&1, changes))}
  ExPhil.Embeddings.Game.embed(gs, neutral, 1, config: embed_cfg) |> Nx.backend_transfer(Nx.BinaryBackend)
end
y20 = ref_emb.(%{y: -20.0})
j0 = ref_emb.(%{jumps_left: 0})
val_of = fn e, ds_ -> if ds_ == [], do: 0.0, else: Nx.to_number(e[hd(ds_)]) end

k_bin = fn k -> cond do k <= 6 -> "1-6"; k <= 12 -> "7-12"; k <= 24 -> "13-24"; k <= 48 -> "25-48"; true -> "49+" end end
k_bins = ["1-6", "7-12", "13-24", "25-48", "49+"]

variants = [
  {"base", windows},
  {"af1", set_dims.(af_dims, 1 / 60.0, 12)},
  {"shallow", set_dims.(y_dims, val_of.(y20, y_dims), window)},
  {"jumps0", set_dims.(jump_dims, val_of.(j0, jump_dims), window)},
  {"static", static}
]

y_bin = fn y -> cond do y > -20 -> "ledge(>-20)"; y > -40 -> "-20..-40"; y > -60 -> "-40..-60"; y > -90 -> "-60..-90"; true -> "<-90" end end
y_bins = ["ledge(>-20)", "-20..-40", "-40..-60", "-60..-90", "<-90"]

base_rows = run.(samples, windows)
Output.puts("RESULT #{label} silent-fall probe base P(any input) by the bot's y: " <>
  Enum.map_join(y_bins, "  ", fn b -> l = Enum.filter(base_rows, &(y_bin.(&1.y) == b)); "#{b} #{mean.(l, p_active)} up #{mean.(l, & &1.up)} (n=#{length(l)}, jumps≥1 #{if l == [], do: "-", else: Float.round(Enum.count(l, &(&1.jumps >= 1)) / length(l), 2)})" end))

results =
  Map.new(variants, fn {name, fun} ->
    rows = run.(samples, fun)
    by_k = Map.new(k_bins, fn b -> l = Enum.filter(rows, &(k_bin.(&1.k) == b)); {b, %{n: length(l), p: mean.(l, p_active), up: mean.(l, & &1.up), jump: mean.(l, & &1.jump), b: mean.(l, & &1.b)}} end)
    Output.puts("RESULT #{label} silent-fall probe #{String.pad_trailing(name, 8)} P(any input) by silence k: " <>
      Enum.map_join(k_bins, "  ", fn b -> r = by_k[b]; "k#{b} #{r.p} (n=#{r.n})" end) <>
      "  | overall up #{mean.(rows, & &1.up)} jump #{mean.(rows, & &1.jump)} B #{mean.(rows, & &1.b)}")
    {name, by_k}
  end)

# ---- donor swaps (19:40): which dim group separates the bot's silent states from
# the expert's at the same depth? For each bot frame, one group's dims (over
# the whole window) are replaced by an EXPERT silent offstage frame's window
# from the same depth band; the group whose swap restores P(input) is the one
# carrying the silence.
donor_games =
  case opts[:split] && File.read(opts[:split]) do
    {:ok, s} -> s |> Jason.decode!() |> Map.fetch!("validation")
    _ -> Path.join(Path.dirname(policy), "split.json") |> File.read!() |> Jason.decode!() |> Map.fetch!("validation")
  end
  |> Enum.take(opts[:donor_games] || 12)

alias ExPhil.Data.Peppi
edge = ExPhil.Sim.GA.stage_edge(32)
silent_ctrl? = &ExPhil.Training.SilentFallWeighting.neutral?/1

donors =
  donor_games
  |> Enum.flat_map(fn path ->
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    opp = own && Enum.find(meta.players, &(&1.port != own.port))

    if own == nil or opp == nil or meta.stage != 32 do
      []
    else
      {:ok, replay} = Peppi.parse(path, player_port: own.port)
      frames = replay |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true) |> Enum.reject(&(&1.game_state.frame < 0))
      gds = Data.from_frames(frames, embed_config: embed_cfg)
      gds = Data.precompute_frame_embeddings(gds, use_prev_action: use_prev?, prev_action_quantize: quant?, show_progress: false)
      gemb = gds.embedded_frames
      gemb = if is_list(gemb), do: Nx.concatenate(gemb), else: gemb
      gemb = Nx.backend_transfer(gemb, EXLA.Backend)
      ft = List.to_tuple(frames)

      for i <- (window - 1)..(tuple_size(ft) - 1),
          f = elem(ft, i),
          p = f.game_state.players[1],
          p.on_ground != true and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -12.0) and (p.y || 0.0) < 0.0,
          (p.action || 0) > 13 and (p.action || 0) != 35 and (p.action || 0) not in 252..263,
          silent_ctrl?.(elem(ft, i - 1).controller),
          do: %{emb: gemb, i: i, y: p.y || 0.0}
    end
  end)

donor_by_band = Enum.group_by(donors, &y_bin.(&1.y))
Output.puts("expert donors (silent offstage frames) by band: #{inspect(Map.new(donor_by_band, fn {k, v} -> {k, length(v)} end))}")

swap_group = fn group_dims ->
  fn chunk ->
    states = windows.(chunk)
    donor =
      chunk
      |> Enum.map(fn s ->
        pool = donor_by_band[y_bin.(s.y)] || donors
        d = Enum.random(pool)
        Nx.slice_along_axis(d.emb, d.i - window + 1, window, axis: 0)
      end)
      |> Nx.stack()
    {_b, _w, dsz} = Nx.shape(states)
    mask = Nx.tensor(Enum.map(0..(dsz - 1), fn i -> if i in group_dims, do: 1.0, else: 0.0 end)) |> Nx.reshape({1, 1, dsz}) |> Nx.backend_transfer(EXLA.Backend)
    Nx.add(Nx.multiply(states, Nx.subtract(1.0, mask)), Nx.multiply(donor, mask))
  end
end

all_but_prev = Enum.to_list(0..(embed_size - 1)) -- Enum.to_list(prev_off..(prev_off + 12))
groups =
  [own_x: dims.own_x, own_y: dims.own_y, own_facing: dims.own_facing, own_speeds: dims.own_speeds, own_action: dims.own_action,
   own_jumps: dims[:own_jumps] || [], opp_position: dims.opp_position, opp_action: dims.opp_action, opp_facing: dims.opp_facing,
   opp_percent: dims.opp_percent, own_position: dims.own_position,
   everything_but_prev: all_but_prev]

swaps =
  if donors == [] do
    Output.warning("no expert donors; swaps skipped")
    %{}
  else
    Map.new(groups, fn {name, gdims} ->
      rows = run.(samples, swap_group.(gdims))
      p = mean.(rows, p_active)
      Output.puts("RESULT #{label} silent-fall donor swap #{String.pad_trailing(to_string(name), 20)} P(any input) #{p}  up #{mean.(rows, & &1.up)} jump #{mean.(rows, & &1.jump)} B #{mean.(rows, & &1.b)}  (#{length(gdims)} dims)")
      {name, p}
    end)
  end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{label: label, frames: frames_path, n: length(samples), results: results, swaps: swaps}, pretty: true))
end
