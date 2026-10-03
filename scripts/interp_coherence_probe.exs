# Interp readout for the input-coherence program (2026-10-02): does the
# prev-action channel model stop reading the game state, and is "change the
# input now" represented in its trunk even though the head rarely emits it?
#
#   mix run scripts/interp_coherence_probe.exs --policy P --label L \
#     [--split SPLIT.json] [--games 16] [--stride 4] [--out FILE.json]
#
# Over held-out expert windows (teacher-forced, same embedding as training):
#
#  Q1 attribution  — |grad x input| of the expert target's log-likelihood,
#                    grouped into prev-action dims vs game-state dims, last
#                    frame vs history; plus ablation KL (prev slot zeroed on
#                    every frame / last-frame game state swapped). Split by
#                    change frames (target != previous input) vs hold frames.
#  Q2 probes       — linear probe on the trunk state for "this is a change
#                    frame" (and change within 3 / 6 frames), by-replay
#                    split, vs the input floor (last-frame embedding) and a
#                    label-shuffled control; next to the HEAD's own
#                    teacher-forced change recall on the same frames.
#  Q5 R button     — P(R) / P(L) on frames where the expert presses L or R,
#                    and what the R logit's saliency keys on.
#
# NO-MIX LAW: run only with no other live beam.
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Attribution, Probe}
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Training.{Data, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, split: :string, games: :integer, stride: :integer, out: :string, batch: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
split = opts[:split] || Path.join(Path.dirname(policy), "split.json")
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation") |> Enum.take(opts[:games] || 16)
stride = opts[:stride] || 4
batch = opts[:batch] || 256

Output.banner("Interp coherence probe: #{label}")

heads = Activations.load_heads(policy)
trunk = Activations.load_trunk(policy)
config = heads.config
window = heads.window
embed_size = Map.fetch!(config, :embed_size)
truthy? = fn v -> v == true or v == "true" end
# the exported .bin config lacks the data flags (prev_action_quantize,
# with_projectiles); the training script's model_config.json next to it has them
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
Output.config([{"window", window}, {"embed", embed_size}, {"prev-action", use_prev?}, {"quantized channel", quant?},
  {"stage internals", stage_internals?}, {"projectiles", with_projectiles?}, {"games", length(files)}, {"stride", stride}])
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
    Output.puts("  #{Path.basename(path)}: #{n} frames")
    %{emb: Nx.backend_transfer(emb, EXLA.Backend), actions: List.to_tuple(actions), n: n, file: path}
  end)

[prev_off, 13] = Attribution.prev_action_dim_range(config: embed_cfg)
prev_dims = Enum.to_list(prev_off..(prev_off + 12))
state_dims = Enum.to_list(0..(embed_size - 1)) -- prev_dims
Output.puts("prev-action dims #{prev_off}..#{prev_off + 12}; #{length(state_dims)} game-state dims")

same? = fn a, b -> a == b end
changed = fn a, b -> %{
  any: not same?.(a, b), buttons: a.buttons != b.buttons,
  main: a.main_x != b.main_x or a.main_y != b.main_y,
  c: a.c_x != b.c_x or a.c_y != b.c_y, shoulder: a.shoulder != b.shoulder} end

# decision frames per game: t in (window-1)..(n-8) so change-within-6 is defined
samples =
  games
  |> Enum.with_index()
  |> Enum.flat_map(fn {g, gi} ->
    for t <- Range.new(window - 1, g.n - 8, stride) do
      a = elem(g.actions, t)
      p = elem(g.actions, t - 1)
      within = fn k -> Enum.any?(1..k, fn j -> not same?.(elem(g.actions, t + j), a) end) end
      %{game: gi, t: t, action: a, prev: p, change: changed.(p, a), within3: within.(3), within6: within.(6)}
    end
  end)
Output.puts("#{length(samples)} decision windows; change frames #{Enum.count(samples, & &1.change.any)}")

# teacher-forced log-likelihood of the expert target (the training objective)
logp = fn {b, mx, my, cx, cy, sh}, tgt ->
  lsm = fn l, y -> l |> Axon.Activations.log_softmax(axis: -1) |> Nx.take_along_axis(Nx.new_axis(y, 1), axis: 1) |> Nx.squeeze(axes: [1]) end
  yb = Nx.as_type(tgt.buttons, :f32)
  lb = Nx.sum(Nx.add(Nx.multiply(yb, Axon.Activations.log_sigmoid(b)), Nx.multiply(Nx.subtract(1.0, yb), Axon.Activations.log_sigmoid(Nx.negate(b)))), axes: [1])
  [lb, lsm.(mx, tgt.main_x), lsm.(my, tgt.main_y), lsm.(cx, tgt.c_x), lsm.(cy, tgt.c_y), lsm.(sh, tgt.shoulder)] |> Enum.reduce(&Nx.add/2)
end

predict = heads.predict_fn
params = heads.params

# Forward inputs from the teacher-forced map + a window. Event-head models
# (`--button-events` / `--stick-events`) mirror Loss.policy_forward_inputs/4:
# the trunk sees the 13-dim prev slot ZEROED and the heads get the last
# position's previous buttons / stick buckets as their own inputs — so for
# them "prev slot zeroed" also neutralises the head's prev inputs, and the
# state gradient flows only through the masked trunk.
button_events? = flag.(:button_events)
stick_events? = flag.(:stick_events)
Output.puts("event heads: buttons #{button_events?} sticks #{stick_events?}")
with_states =
  if button_events? or stick_events? do
    fn tf, s ->
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
    end
  else
    fn tf, s -> Map.put(tf, "state_sequence", s) end
  end

grad_fn = fn p, s, tf, tgt ->
  Nx.Defn.grad(s, fn s2 -> Nx.sum(logp.(predict.(p, with_states.(tf, s2)), tgt)) end)
end
grad_jit = Nx.Defn.jit(grad_fn, compiler: EXLA)
# Q5: gradient of the R-button logit alone (jitted ONCE — a fresh closure per batch would recompile)
r_grad_jit =
  Nx.Defn.jit(fn p, s, tf -> Nx.Defn.grad(s, fn s2 -> Nx.sum(elem(predict.(p, with_states.(tf, s2)), 0)[[.., 6]]) end) end,
    compiler: EXLA)

probs = fn {b, mx, my, cx, cy, sh} ->
  %{buttons: Nx.sigmoid(b), main_x: Axon.Activations.softmax(mx), main_y: Axon.Activations.softmax(my),
    c_x: Axon.Activations.softmax(cx), c_y: Axon.Activations.softmax(cy), shoulder: Axon.Activations.softmax(sh)}
end
kl_cat = fn p, q -> Nx.sum(Nx.multiply(p, Nx.log(Nx.divide(Nx.add(p, 1.0e-9), Nx.add(q, 1.0e-9)))), axes: [-1]) end
kl_bern = fn p, q ->
  Nx.add(kl_cat.(p, q), kl_cat.(Nx.subtract(1.0, p), Nx.subtract(1.0, q)))
end
kl_total = fn a, b ->
  [kl_bern.(a.buttons, b.buttons), kl_cat.(a.main_x, b.main_x), kl_cat.(a.main_y, b.main_y),
   kl_cat.(a.c_x, b.c_x), kl_cat.(a.c_y, b.c_y), kl_cat.(a.shoulder, b.shoulder)] |> Enum.reduce(&Nx.add/2)
end

prev_mask = Nx.tensor(Enum.map(0..(embed_size - 1), fn d -> if d in prev_dims, do: 0.0, else: 1.0 end)) |> Nx.backend_transfer(EXLA.Backend)
prev_idx = Nx.tensor(prev_dims)
state_idx = Nx.tensor(state_dims)

decode_change? = fn {b, mx, my, cx, cy, sh}, prev_t ->
  # deterministic decode vs the previous input: would the head change anything?
  btn = Nx.greater(b, 0.0) |> Nx.as_type(:s64)
  bchg = Nx.any(Nx.not_equal(btn, prev_t.buttons), axes: [1])
  cat = fn l, y -> Nx.not_equal(Nx.argmax(l, axis: -1), y) end
  [bchg, cat.(mx, prev_t.main_x), cat.(my, prev_t.main_y), cat.(cx, prev_t.c_x), cat.(cy, prev_t.c_y), cat.(sh, prev_t.shoulder)]
  |> Enum.reduce(&Nx.logical_or/2)
end

:rand.seed(:exsss, {7, 7, 7})
n_batches = div(length(samples) + batch - 1, batch)

rows =
  samples
  |> Enum.chunk_every(batch)
  |> Enum.with_index()
  |> Enum.flat_map(fn {chunk, bi} ->
    if rem(bi, 10) == 0, do: Output.puts("  batch #{bi + 1}/#{n_batches}")
    m = length(chunk)
    # gather windows {m, window, embed} per game then concatenate
    states =
      chunk
      |> Enum.map(fn s -> Nx.slice_along_axis(Enum.at(games, s.game).emb, s.t - window + 1, window, axis: 0) end)
      |> Nx.stack()
    tgt = Data.actions_to_tensors(Enum.map(chunk, & &1.action)) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
    prv = Data.actions_to_tensors(Enum.map(chunk, & &1.prev)) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
    tf = Heads.tf_inputs(tgt)

    out = predict.(params, with_states.(tf, states))
    p0 = probs.(out)
    ll = logp.(out, tgt)

    # Q1a saliency
    g = grad_jit.(params, states, tf, tgt)
    sal = Nx.abs(Nx.multiply(g, states))
    gabs = Nx.abs(g)
    last = fn t -> t |> Nx.slice_along_axis(window - 1, 1, axis: 1) |> Nx.squeeze(axes: [1]) end
    hist = fn t -> t |> Nx.slice_along_axis(0, window - 1, axis: 1) |> Nx.sum(axes: [1]) end
    part = fn t, idx -> t |> Nx.take(idx, axis: 1) |> Nx.sum(axes: [1]) end
    shares = fn t ->
      lp = part.(last.(t), prev_idx); ls = part.(last.(t), state_idx)
      hp = part.(hist.(t), prev_idx); hs = part.(hist.(t), state_idx)
      tot = [lp, ls, hp, hs] |> Enum.reduce(&Nx.add/2) |> Nx.max(1.0e-12)
      %{last_prev: Nx.divide(lp, tot), last_state: Nx.divide(ls, tot), hist_prev: Nx.divide(hp, tot), hist_state: Nx.divide(hs, tot)}
    end
    sh_gx = shares.(sal)
    sh_g = shares.(gabs)

    # Q1b ablation KL: prev slot zeroed on every frame; last-frame game state swapped with another window's
    p_noprev = probs.(predict.(params, with_states.(tf, Nx.multiply(states, prev_mask))))
    perm = Enum.shuffle(0..(m - 1)) |> Nx.tensor()
    donor_last = states |> Nx.take(perm, axis: 0) |> last.()
    own_last = last.(states)
    swapped_last = Nx.add(Nx.multiply(donor_last, prev_mask), Nx.multiply(own_last, Nx.subtract(1.0, prev_mask)))
    states_swap = Nx.put_slice(states, [0, window - 1, 0], Nx.new_axis(swapped_last, 1))
    p_swap = probs.(predict.(params, with_states.(tf, states_swap)))
    kl_noprev = kl_total.(p0, p_noprev)
    kl_swap = kl_total.(p0, p_swap)

    # Q2: trunk activation + input floor; head's own change decision
    acts = trunk.predict_fn.(trunk.params, states)
    head_chg = decode_change?.(out, prv)

    # Q5: R / L probabilities and R-logit saliency shares
    pr = p0.buttons[[.., 6]]
    pl = p0.buttons[[.., 5]]
    r_logit_grad = r_grad_jit.(params, states, tf)
    r_sal = Nx.abs(Nx.multiply(r_logit_grad, states))
    r_sh = shares.(r_sal)
    # top single dims for the R logit (mean over this batch)
    r_dim_mean = r_sal |> Nx.sum(axes: [1]) |> Nx.mean(axes: [0])

    to_l = fn t -> Nx.to_flat_list(t) end
    cols = %{ll: to_l.(ll), gx_lp: to_l.(sh_gx.last_prev), gx_ls: to_l.(sh_gx.last_state), gx_hp: to_l.(sh_gx.hist_prev), gx_hs: to_l.(sh_gx.hist_state),
      g_lp: to_l.(sh_g.last_prev), g_ls: to_l.(sh_g.last_state), g_hp: to_l.(sh_g.hist_prev), g_hs: to_l.(sh_g.hist_state),
      kl_noprev: to_l.(kl_noprev), kl_swap: to_l.(kl_swap), head_chg: to_l.(head_chg), pr: to_l.(pr), pl: to_l.(pl),
      r_lp: to_l.(r_sh.last_prev), r_ls: to_l.(r_sh.last_state), r_hp: to_l.(r_sh.hist_prev), r_hs: to_l.(r_sh.hist_state)}
    acts_l = Nx.to_batched(acts, 1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))
    floor_l = Nx.to_batched(own_last, 1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))

    chunk
    |> Enum.with_index()
    |> Enum.map(fn {s, i} ->
      s
      |> Map.merge(Map.new(cols, fn {k, v} -> {k, Enum.at(v, i)} end))
      |> Map.put(:act, Enum.at(acts_l, i))
      |> Map.put(:floor, Enum.at(floor_l, i))
      |> Map.put(:r_dim_mean, if(i == 0, do: r_dim_mean, else: nil))
    end)
  end)

mean = fn l -> if l == [], do: nil, else: Float.round(Enum.sum(l) / length(l), 4) end
r3 = fn v -> if is_float(v), do: Float.round(v, 3), else: v end
chg = Enum.filter(rows, & &1.change.any)
hold = Enum.reject(rows, & &1.change.any)

# ---- Q1 report ------------------------------------------------------------------
report_shares = fn title, rs, pre ->
  Output.puts("RESULT #{label} #{title} n=#{length(rs)}  last-frame prev #{mean.(Enum.map(rs, & &1[:"#{pre}_lp"]))}  last-frame state #{mean.(Enum.map(rs, & &1[:"#{pre}_ls"]))}  " <>
    "history prev #{mean.(Enum.map(rs, & &1[:"#{pre}_hp"]))}  history state #{mean.(Enum.map(rs, & &1[:"#{pre}_hs"]))}")
end
report_shares.("Q1 saliency |grad x input| shares, CHANGE frames:", chg, :gx)
report_shares.("Q1 saliency |grad x input| shares, HOLD frames:  ", hold, :gx)
report_shares.("Q1 saliency |grad| shares, CHANGE frames:        ", chg, :g)
report_shares.("Q1 saliency |grad| shares, HOLD frames:          ", hold, :g)
Output.puts("RESULT #{label} Q1 ablation KL (nats): prev slot zeroed  change #{mean.(Enum.map(chg, & &1.kl_noprev))}  hold #{mean.(Enum.map(hold, & &1.kl_noprev))}   |   " <>
  "last-frame state swapped  change #{mean.(Enum.map(chg, & &1.kl_swap))}  hold #{mean.(Enum.map(hold, & &1.kl_swap))}")
Output.puts("RESULT #{label} Q1 teacher-forced log-lik: change #{mean.(Enum.map(chg, & &1.ll))}  hold #{mean.(Enum.map(hold, & &1.ll))}")

# ---- Q2 report ------------------------------------------------------------------
n_games = length(games)
eval_games = Enum.to_list(max(n_games - 4, 1)..(n_games - 1))
{tr, ev} = Enum.split_with(rows, &(&1.game not in eval_games))
head_recall = mean.(Enum.map(chg, &if(&1.head_chg == 1, do: 1.0, else: 0.0)))
head_false = mean.(Enum.map(hold, &if(&1.head_chg == 1, do: 1.0, else: 0.0)))
Output.puts("RESULT #{label} Q2 HEAD teacher-forced decode: change recall #{head_recall}  false-change on hold #{head_false}")

probe = fn feature, name, lab_fn ->
  x = Nx.stack(Enum.map(tr, & &1[feature])); xe = Nx.stack(Enum.map(ev, & &1[feature]))
  y = Nx.tensor(Enum.map(tr, &lab_fn.(&1)), type: :s64); ye = Nx.tensor(Enum.map(ev, &lab_fn.(&1)), type: :s64)
  res = Probe.fit_eval(x, y, xe, ye, 2)
  ys = y |> Nx.to_flat_list() |> Enum.shuffle() |> Nx.tensor(type: :s64)
  ctrl = Probe.fit_eval(x, ys, xe, ye, 2)
  Output.puts("RESULT #{label} Q2 probe #{name} on #{feature}: balanced acc #{r3.(res.balanced_accuracy)} (majority #{r3.(res.majority_baseline)}, shuffled-label control #{r3.(ctrl.balanced_accuracy)}, n_eval #{res.n_eval})")
  %{feature: feature, label: name, balanced_accuracy: res.balanced_accuracy, control: ctrl.balanced_accuracy, majority: res.majority_baseline}
end
b = fn v -> if v, do: 1, else: 0 end
probes =
  for {name, lab} <- [{"change now", & b.(&1.change.any)}, {"button change now", & b.(&1.change.buttons)}, {"main-stick change now", & b.(&1.change.main)},
                      {"change within 3", & b.(&1.within3)}, {"change within 6", & b.(&1.within6)}],
      feature <- [:act, :floor] do
    probe.(feature, name, lab)
  end

# ---- Q5 report ------------------------------------------------------------------
l_press = Enum.filter(rows, &(&1.action.buttons.l and not &1.prev.buttons.l))
r_press = Enum.filter(rows, &(&1.action.buttons.r and not &1.prev.buttons.r))
l_hold = Enum.filter(rows, &(&1.action.buttons.l and &1.prev.buttons.l))
neither = Enum.filter(rows, &(not &1.action.buttons.l and not &1.action.buttons.r and not &1.prev.buttons.l and not &1.prev.buttons.r))
q5 = fn title, rs -> Output.puts("RESULT #{label} Q5 #{title} n=#{length(rs)}: P(L) #{mean.(Enum.map(rs, & &1.pl))}  P(R) #{mean.(Enum.map(rs, & &1.pr))}") end
q5.("expert presses L (edge)   ", l_press)
q5.("expert presses R (edge)   ", r_press)
q5.("expert holds L            ", l_hold)
q5.("expert no L/R, none before", neither)
r_hi = Enum.filter(neither, &(&1.pr > 0.2))
Output.puts("RESULT #{label} Q5 unprompted R (P(R)>0.2 with no expert L/R) #{length(r_hi)}/#{length(neither)} frames; R-logit saliency shares: " <>
  "last prev #{mean.(Enum.map(r_hi, & &1.r_lp))} last state #{mean.(Enum.map(r_hi, & &1.r_ls))} hist prev #{mean.(Enum.map(r_hi, & &1.r_hp))} hist state #{mean.(Enum.map(r_hi, & &1.r_hs))}")
r_dims = rows |> Enum.map(& &1.r_dim_mean) |> Enum.reject(&is_nil/1) |> Nx.stack() |> Nx.mean(axes: [0])
top = r_dims |> Nx.to_flat_list() |> Enum.with_index() |> Enum.sort_by(&(-elem(&1, 0))) |> Enum.take(8)
Output.puts("RESULT #{label} Q5 R-logit top dims (dim:mean|grad x input|): " <> Enum.map_join(top, "  ", fn {v, d} -> "#{d}#{if d in prev_dims, do: "(prev#{d - prev_off})", else: ""}:#{Float.round(v, 4)}" end))

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  strip = fn rs -> Enum.map(rs, &Map.drop(&1, [:act, :floor, :r_dim_mean, :action, :prev])) end
  File.write!(out, Jason.encode!(%{label: label, policy: policy, n: length(rows), probes: probes,
    summary: %{change_n: length(chg), hold_n: length(hold), head_change_recall: head_recall, head_false_change: head_false},
    rows: strip.(rows)}, pretty: false))
  Output.puts("-> #{out}")
end
