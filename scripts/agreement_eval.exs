# Agreement eval (2026-10-21): how often the model's top guess is what the
# expert actually pressed, per head and per OPTION FAMILY, split by situation.
# Bradley: "we want how often the top guess matches what the expert pressed
# per situation ... whether you pressed L or R isn't as important as L
# cancelled ... recognize that this is a similar option."
#
#   mix run scripts/agreement_eval.exs --policy P [--split SPLIT.json] [--games 24]
#     [--stride 2] [--batch 256] [--out FILE.json] [--rows FILE.jsonl] [--top 40]
#
# Teacher-forced over held-out expert games (the same embedding + forward as
# the validation loss; interp_coherence_probe.exs is the template): at every
# frame the model sees the real history and we read its argmax per head.
#
# Levels, coarse to fine (a coarser level forgives differences that do not
# change what happens in the game):
#   families   jump (X|Y)  attack (A)  special (B)  grab (Z)  trigger (L|R digital
#              or analog shoulder)  main stick 9-way direction  c stick 5-way
#   heads      buttons exact (all 8), main/c stick bucket exact and ±1, shoulder
#   joint      every head exact (the training objective's argmax)
# Change frames (the real input differs from the previous frame's) are
# reported apart from hold frames: holds are ~80 % of frames and trivially
# easy, so "overall" numbers hide everything. On change frames we also ask
# whether the model CHANGES at all (change recall) and on hold frames whether
# it wrongly changes (false change).
# Surprise = −log p(real input) per frame; the top-N most surprising frames
# are listed with their state — the seed of the "engine" readout.
# True option-level agreement (dash-dance vs wavedash vs shield) needs the
# resulting action state, i.e. a simulated line; this is the input level only.
#
# NO-MIX LAW: run only with no other live beam.

alias ExPhil.Training.Output
alias ExPhil.Training.Data
alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, ActionNames}
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Situations

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, split: :string, games: :integer, stride: :integer, batch: :integer, out: :string, rows: :string, top: :integer])

if bad != [], do: raise("bad options: #{inspect(bad)}")
policy = opts[:policy] || raise("--policy required")
split = opts[:split] || Path.join(Path.dirname(policy), "split.json")
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation") |> Enum.take(opts[:games] || 24)
stride = opts[:stride] || 2
batch = opts[:batch] || 256
top_n = opts[:top] || 40

Output.banner("Agreement eval: #{Path.basename(Path.dirname(policy))}/#{Path.basename(policy)}")

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
shoulder_buckets = Map.get(config, :shoulder_buckets, 4)
button_events? = flag.(:button_events)
stick_events? = flag.(:stick_events)
stick_duration? = Map.get(config, :stick_duration, Map.get(json_cfg, "stick_duration")) not in [nil, ""]

Output.config([
  {"window", window}, {"embed", embed_size}, {"prev-action", use_prev?}, {"quantized channel", quant?},
  {"stage internals", stage_internals?}, {"projectiles", with_projectiles?}, {"event heads", "buttons #{button_events?} sticks #{stick_events?}"},
  {"games", length(files)}, {"stride", stride}, {"batch", batch}
])

embed_cfg = %{ExPhil.Embeddings.Game.Config.default() | stage_internals: stage_internals?, with_projectiles: with_projectiles?}

# ---- embed the held-out games exactly as training did; label each frame's situation ----
games =
  files
  |> Enum.with_index()
  |> Enum.map(fn {path, gi} ->
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
    actions = Enum.map(frames, &Data.controller_to_action(&1.controller, axis_buckets: axis_buckets, shoulder_buckets: shoulder_buckets))

    # situation labels per frame (ego = port 1 after the remap) + the state the page shows
    {sits, _} =
      Enum.map_reduce(frames, Situations.new_context(), fn f, ctx ->
        {ctx2, set} = Situations.fold(ctx, f.game_state, 1)
        {set, ctx2}
      end)

    states =
      Enum.map(frames, fn f ->
        p = f.game_state.players[1]
        %{frame: f.game_state.frame, action: p.action, af: p.action_frame, x: p.x, y: p.y, ground: p.on_ground, hit: p.hitstun_frames_left, pct: p.percent}
      end)

    Output.puts("  #{gi + 1}/#{length(files)} #{Path.basename(path)}: #{n} frames")
    %{emb: Nx.backend_transfer(emb, EXLA.Backend), actions: List.to_tuple(actions), sits: List.to_tuple(sits), states: List.to_tuple(states), n: n, file: path, game: Path.basename(path, ".slp")}
  end)

[prev_off, 13] = ExPhil.Interp.Attribution.prev_action_dim_range(config: embed_cfg)

# ---- the teacher-forced forward (mirrors Loss.policy_forward_inputs/4 for event heads) ----
predict = heads.predict_fn
# semi-Markov main stick (stick_duration C): the main-stick logits are trained ONLY on decision frames
# (pair changes / button edge / hold age a multiple of C); elsewhere the live agent keeps the previous
# pair. The eval applies the same rule, and scores the duration head on decision frames.
dur_cap = (case Map.get(config, :stick_duration, Map.get(json_cfg, "stick_duration")) do v when is_integer(v) -> v; v when is_binary(v) and v != "" -> String.to_integer(v); _ -> nil end)

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

        inputs = Map.put(inputs, "prev_sticks", buckets)
        if stick_duration?, do: Map.put(inputs, "prev_age", ExPhil.Training.Imitation.Loss.prev_age_from_window(s, prev_off, axis_buckets)), else: inputs
      else
        inputs
      end
    end
  else
    fn tf, s -> Map.put(tf, "state_sequence", s) end
  end

params = heads.params

# argmax per head + −log p(target); one jitted program per batch shape
decode =
  Nx.Defn.jit(
    fn p, s, tf, tgt ->
      inputs = with_states.(tf, s)
      {b, mx, my, cx, cy, sh, dur} =
        case predict.(p, inputs) do
          {b, mx, my, cx, cy, sh, d} -> {b, mx, my, cx, cy, sh, d}
          {b, mx, my, cx, cy, sh} -> {b, mx, my, cx, cy, sh, nil}
        end
      lsm = fn l, y -> l |> Axon.Activations.log_softmax(axis: -1) |> Nx.take_along_axis(Nx.new_axis(y, 1), axis: 1) |> Nx.squeeze(axes: [1]) end
      yb = Nx.as_type(tgt.buttons, :f32)
      lb = Nx.sum(Nx.add(Nx.multiply(yb, Axon.Activations.log_sigmoid(b)), Nx.multiply(Nx.subtract(1.0, yb), Axon.Activations.log_sigmoid(Nx.negate(b)))), axes: [1])
      nll = [lb, lsm.(mx, tgt.main_x), lsm.(my, tgt.main_y), lsm.(cx, tgt.c_x), lsm.(cy, tgt.c_y), lsm.(sh, tgt.shoulder)] |> Enum.reduce(&Nx.add/2) |> Nx.negate()
      am = fn l -> Nx.argmax(l, axis: -1) |> Nx.as_type(:s64) end
      pb = Nx.sigmoid(b)
      %{
        buttons: Nx.greater(b, 0.0) |> Nx.as_type(:s64),
        p_buttons: pb,
        main_x: am.(mx), main_y: am.(my), c_x: am.(cx), c_y: am.(cy), shoulder: am.(sh),
        p_main: Nx.reduce_max(Axon.Activations.softmax(mx), axes: [-1]),
        p_shoulder_on: Nx.subtract(1.0, Axon.Activations.softmax(sh)[[.., 0]]),
        nll: nll
      }
      |> then(fn m ->
        if dur != nil and dur_cap do
          %{mask: mask, duration: dur_t} = ExPhil.Training.Imitation.Loss.stick_decision_targets(inputs, tgt, dur_cap)
          Map.merge(m, %{mask: mask, dur: Nx.argmax(dur, axis: -1) |> Nx.as_type(:s64), dur_t: Nx.as_type(dur_t, :s64)})
        else
          m
        end
      end)
    end,
    compiler: EXLA
  )

# ---- option families (the same function on the prediction and the real input) ----
button_order = [:a, :b, :x, :y, :z, :l, :r, :d_up]
# a stick bucket is "pushed" past the deadzone when ≥ 3 buckets from centre (≈ 0.19 of half-range; Melee's deadzone is 0.2875)
centre = div(axis_buckets, 2)
dir = fn b -> cond do b >= centre + 3 -> 1; b <= centre - 3 -> -1; true -> 0 end end
dir_name = %{{0, 0} => "neutral", {1, 0} => "right", {-1, 0} => "left", {0, 1} => "up", {0, -1} => "down",
  {1, 1} => "up-right", {-1, 1} => "up-left", {1, -1} => "down-right", {-1, -1} => "down-left"}

families = fn a ->
  b = a.buttons
  %{
    jump: b.x or b.y, attack: b.a, special: b.b, grab: b.z,
    trigger: b.l or b.r or a.shoulder > 0,
    main: dir_name[{dir.(a.main_x), dir.(a.main_y)}],
    c: (case {dir.(a.c_x), dir.(a.c_y)} do {0, 0} -> "neutral"; {1, _} -> "right"; {-1, _} -> "left"; {0, 1} -> "up"; {0, -1} -> "down" end)
  }
end

family_keys = [:jump, :attack, :special, :grab, :trigger, :main, :c]

pred_action = fn d, i ->
  bs = Enum.zip(button_order, Enum.map(0..7, fn k -> d.buttons[i][k] == 1 end)) |> Map.new()
  %{buttons: bs, main_x: d.main_x[i], main_y: d.main_y[i], c_x: d.c_x[i], c_y: d.c_y[i], shoulder: d.shoulder[i]}
end

bool_action = fn a -> %{a | buttons: Map.new(a.buttons, fn {k, v} -> {k, v == true} end)} end

# press edges in a family (a family that is OFF before and ON now)
edges = fn fam, pfam -> Enum.filter([:jump, :attack, :special, :grab, :trigger], fn k -> fam[k] and not pfam[k] end) end

# decision frames: t in (window-1)..(n-1), stride; prev at t-1
samples =
  games
  |> Enum.with_index()
  |> Enum.flat_map(fn {g, gi} ->
    for t <- Range.new(window - 1, g.n - 1, stride) do
      %{game: gi, t: t, action: bool_action.(elem(g.actions, t)), prev: bool_action.(elem(g.actions, t - 1))}
    end
  end)

Output.puts("#{length(samples)} decision frames over #{length(games)} games")
n_batches = div(length(samples) + batch - 1, batch)

rows =
  samples
  |> Enum.chunk_every(batch)
  |> Enum.with_index()
  |> Enum.flat_map(fn {chunk, bi} ->
    if rem(bi, 20) == 0, do: Output.puts("  batch #{bi + 1}/#{n_batches}")

    states =
      chunk
      |> Enum.map(fn s -> Nx.slice_along_axis(Enum.at(games, s.game).emb, s.t - window + 1, window, axis: 0) end)
      |> Nx.stack()

    tgt = Data.actions_to_tensors(Enum.map(chunk, & &1.action)) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
    tf = Heads.tf_inputs(tgt)
    d = decode.(params, states, tf, tgt) |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, Nx.BinaryBackend)} end)
    d = Map.new(d, fn {k, v} -> {k, Nx.to_list(v)} end)

    chunk
    |> Enum.with_index()
    |> Enum.map(fn {s, i} ->
      g = Enum.at(games, s.game)
      pa = %{
        buttons: Enum.zip(button_order, Enum.map(Enum.at(d.buttons, i), &(&1 == 1))) |> Map.new(),
        main_x: Enum.at(d.main_x, i), main_y: Enum.at(d.main_y, i), c_x: Enum.at(d.c_x, i), c_y: Enum.at(d.c_y, i), shoulder: Enum.at(d.shoulder, i)
      }
      ra = s.action
      pr = s.prev
      decision = if d[:mask], do: Enum.at(d.mask, i) == 1.0, else: true
      pa = if decision, do: pa, else: %{pa | main_x: pr.main_x, main_y: pr.main_y}
      dur_ok = decision and d[:dur] != nil and Enum.at(d.dur, i) == Enum.at(d.dur_t, i)
      dur_pm1 = decision and d[:dur] != nil and abs(Enum.at(d.dur, i) - Enum.at(d.dur_t, i)) <= 1
      pb = Enum.at(d.p_buttons, i)
      probs = %{jump: max(Enum.at(pb, 2), Enum.at(pb, 3)), attack: Enum.at(pb, 0), special: Enum.at(pb, 1), grab: Enum.at(pb, 4),
        trigger: Enum.max([Enum.at(pb, 5), Enum.at(pb, 6), Enum.at(d.p_shoulder_on, i)])}
      pf = families.(pa)
      rf = families.(ra)
      prf = families.(pr)
      fam_ok = Map.new(family_keys, fn k -> {k, pf[k] == rf[k]} end)
      btn_ok = Enum.all?(button_order, &(pa.buttons[&1] == ra.buttons[&1]))
      pm1 = fn a, b -> abs(a - b) <= 1 end
      joint = btn_ok and pa.main_x == ra.main_x and pa.main_y == ra.main_y and pa.c_x == ra.c_x and pa.c_y == ra.c_y and pa.shoulder == ra.shoulder
      changed = ra != pr
      model_changed = pa != pr
      st = elem(g.states, s.t)

      %{
        game: s.game, t: s.t, frame: st.frame, action: st.action, name: ActionNames.name(st.action || 0), af: st.af,
        ground: st.ground, hit: st.hit, y: st.y, x: st.x,
        sits: elem(g.sits, s.t) |> MapSet.to_list(),
        changed: changed, model_changed: model_changed, decision: decision, dur_ok: dur_ok, dur_pm1: dur_pm1, p: probs,
        real_edges: edges.(rf, prf), label_edges: edges.(pf, prf),
        real: rf, pred: pf, real_sticks: {ra.main_x, ra.main_y}, pred_sticks: {pa.main_x, pa.main_y},
        m: %{
          joint: joint, buttons: btn_ok,
          main_bucket: pa.main_x == ra.main_x and pa.main_y == ra.main_y,
          main_pm1: pm1.(pa.main_x, ra.main_x) and pm1.(pa.main_y, ra.main_y),
          c_bucket: pa.c_x == ra.c_x and pa.c_y == ra.c_y,
          shoulder: pa.shoulder == ra.shoulder,
          families: Enum.all?(family_keys, &fam_ok[&1]),
          jump: fam_ok.jump, attack: fam_ok.attack, special: fam_ok.special, grab: fam_ok.grab, trigger: fam_ok.trigger, main_dir: fam_ok.main, c_dir: fam_ok.c
        },
        nll: Enum.at(d.nll, i), p_main: Enum.at(d.p_main, i)
      }
    end)
  end)

# ---- aggregate -----------------------------------------------------------
metric_keys = [:joint, :buttons, :main_bucket, :main_pm1, :c_bucket, :shoulder, :families, :jump, :attack, :special, :grab, :trigger, :main_dir, :c_dir]
pct = fn xs -> if xs == [], do: nil, else: Float.round(100 * Enum.count(xs, & &1) / length(xs), 1) end
mean = fn xs -> if xs == [], do: nil, else: Float.round(Enum.sum(xs) / length(xs), 3) end

summ = fn rs ->
  Map.new(metric_keys, fn k -> {k, pct.(Enum.map(rs, & &1.m[k]))} end)
  |> Map.merge(%{
    n: length(rs),
    surprise: mean.(Enum.map(rs, & &1.nll)),
    change_recall: (ch = Enum.filter(rs, & &1.changed); pct.(Enum.map(ch, & &1.model_changed))),
    false_change: (ho = Enum.reject(rs, & &1.changed); pct.(Enum.map(ho, & &1.model_changed)))
  })
end

by = fn rs, key_fn ->
  rs |> Enum.flat_map(fn r -> Enum.map(List.wrap(key_fn.(r)), &{&1, r}) end)
  |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
  |> Enum.map(fn {k, v} -> {k, summ.(v)} end)
  |> Enum.sort_by(fn {_, s} -> -s.n end)
end

decision_rows = Enum.filter(rows, & &1.decision)
dur_summary = %{n: length(decision_rows), exact: pct.(Enum.map(decision_rows, & &1.dur_ok)), pm1: pct.(Enum.map(decision_rows, & &1.dur_pm1))}

# press calibration: does P(family) rise on the frames where the expert pressed it? AUC = P(press frame
# scores above a comparable no-press frame); 0.5 = no information, 1 = perfect ranking. Negatives are
# frames where the family was OFF on the previous frame and stayed off (a press was possible).
auc = fn pos, neg ->
  if pos == [] or neg == [] do nil else
    ranked = (Enum.map(pos, &{&1, 1}) ++ Enum.map(neg, &{&1, 0})) |> Enum.sort_by(&elem(&1, 0))
    {sum_ranks, _} = Enum.reduce(ranked, {0.0, 1}, fn {_, y}, {acc, r} -> {if(y == 1, do: acc + r, else: acc), r + 1} end)
    np = length(pos); nn = length(neg)
    Float.round((sum_ranks - np * (np + 1) / 2) / (np * nn), 3)
  end
end
calibration =
  Enum.map([:jump, :attack, :special, :grab, :trigger], fn k ->
    pos = Enum.filter(rows, &(k in &1.real_edges))
    neg = Enum.filter(rows, fn r -> not r.real[k] end)
    %{family: k, n_press: length(pos), n_no_press: length(neg), auc: auc.(Enum.map(pos, & &1.p[k]), Enum.map(neg, & &1.p[k])),
      mean_p_on_press: mean.(Enum.map(pos, & &1.p[k])), mean_p_no_press: mean.(Enum.map(neg, & &1.p[k])),
      press_frames_above_p50: (med = (neg |> Enum.map(& &1.p[k]) |> Enum.sort() |> then(fn l -> if l == [], do: 1.0, else: Enum.at(l, div(length(l), 2)) end)); pct.(Enum.map(pos, &(&1.p[k] > med))))}
  end)

hold = Enum.reject(rows, & &1.changed)
change = Enum.filter(rows, & &1.changed)
with_edge = Enum.filter(rows, &(&1.real_edges != []))

# per press-edge family: on frames where the expert PRESSED jump/attack/..., did the model press the same family?
edge_table =
  Enum.map([:jump, :attack, :special, :grab, :trigger], fn k ->
    rs = Enum.filter(rows, &(k in &1.real_edges))
    %{family: k, n: length(rs), model_same_press: pct.(Enum.map(rs, &(k in &1.label_edges))), model_any_press: pct.(Enum.map(rs, &(&1.label_edges != []))),
      model_family_on: pct.(Enum.map(rs, & &1.m[k]))}
  end)

# false presses: model press edge where the expert made none
no_edge = Enum.filter(rows, &(&1.real_edges == []))
false_press = pct.(Enum.map(no_edge, &(&1.label_edges != [])))

top =
  rows
  |> Enum.sort_by(& -&1.nll)
  |> Enum.take(top_n)
  |> Enum.map(fn r ->
    g = Enum.at(games, r.game)
    %{game: g.game, frame: r.frame, nll: Float.round(r.nll, 2), state: "#{r.name} f#{r.af}#{if r.ground, do: " grounded", else: " air"} y=#{Float.round(r.y * 1.0, 1)} hit=#{r.hit}",
      sits: r.sits, real: r.real, model: r.pred, real_edges: r.real_edges, model_edges: r.label_edges}
  end)

report = %{
  policy: policy, split: split, games: Enum.map(games, & &1.game), stride: stride, frames: length(rows),
  levels: "families = jump/attack/special/grab/trigger on-off + main 9-way + c 5-way; heads = exact buckets; joint = every head exact",
  overall: summ.(rows), hold: summ.(hold), change: summ.(change), with_real_press_edge: summ.(with_edge),
  press_edges: edge_table, false_press_rate_on_no_edge_frames: false_press,
  decision_frames: summ.(decision_rows), duration_head: dur_summary, stick_duration: dur_cap, press_calibration: calibration,
  by_situation: by.(rows, & &1.sits) |> Enum.map(fn {k, s} -> Map.put(s, :situation, k) end),
  by_situation_change_frames: by.(change, & &1.sits) |> Enum.map(fn {k, s} -> Map.put(s, :situation, k) end),
  by_action: by.(rows, & &1.name) |> Enum.take(40) |> Enum.map(fn {k, s} -> Map.put(s, :action, k) end),
  by_action_change_frames: by.(change, & &1.name) |> Enum.take(40) |> Enum.map(fn {k, s} -> Map.put(s, :action, k) end),
  most_surprising: top,
  created: DateTime.utc_now() |> DateTime.to_iso8601()
}

out = opts[:out] || "eval_runs/agreement/#{Path.basename(Path.dirname(policy))}.json"
File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(report, pretty: true))

if opts[:rows] do
  File.write!(opts[:rows], Enum.map_join(rows, "\n", fn r ->
    Jason.encode!(%{g: r.game, t: r.t, frame: r.frame, name: r.name, changed: r.changed, model_changed: r.model_changed, real: r.real, pred: r.pred,
      real_edges: r.real_edges, model_edges: r.label_edges, nll: Float.round(r.nll, 3), m: r.m, sits: r.sits, decision: r.decision, p: Map.new(r.p, fn {k, v} -> {k, Float.round(v, 3)} end)})
  end))
end

line = fn label, s ->
  Output.puts("RESULT #{label}: n #{s.n}  joint #{s.joint}%  buttons #{s.buttons}%  families #{s.families}%  main dir #{s.main_dir}% (bucket #{s.main_bucket}%, ±1 #{s.main_pm1}%)  c dir #{s.c_dir}%  trigger #{s.trigger}%  jump #{s.jump}%  attack #{s.attack}%  special #{s.special}%  surprise #{s.surprise}  change recall #{s.change_recall}%  false change #{s.false_change}%")
end

line.("ALL frames", report.overall)
line.("HOLD frames (real input unchanged)", report.hold)
line.("CHANGE frames (real input changed)", report.change)
line.("frames with a real press edge", report.with_real_press_edge)
line.("DECISION frames (stick head trained here)", report.decision_frames)
Output.puts("RESULT duration head (semi-Markov cap #{dur_cap}): n #{dur_summary.n}  exact #{dur_summary.exact}%  ±1 #{dur_summary.pm1}%")
Enum.each(calibration, fn c -> Output.puts("RESULT press calibration #{c.family}: AUC #{c.auc}  mean P on press frames #{c.mean_p_on_press} vs no-press #{c.mean_p_no_press}  press frames above the no-press median #{c.press_frames_above_p50}%  (n #{c.n_press} / #{c.n_no_press})") end)
Enum.each(edge_table, fn e -> Output.puts("RESULT press edge #{e.family}: n #{e.n}  model pressed the same family #{e.model_same_press}%  model pressed anything #{e.model_any_press}%  family on #{e.model_family_on}%") end)
Output.puts("RESULT false press (model press edge on a frame with no real press): #{false_press}% of #{length(no_edge)}")
Output.puts("RESULT by situation (change frames, n ≥ 200):")
Enum.each(report.by_situation_change_frames, fn s -> if s.n >= 200, do: Output.puts("  #{String.pad_trailing(to_string(s.situation), 24)} n #{String.pad_leading(to_string(s.n), 6)}  families #{s.families}%  joint #{s.joint}%  main dir #{s.main_dir}%  jump #{s.jump}%  attack #{s.attack}%  change recall #{s.change_recall}%  surprise #{s.surprise}") end)
Output.puts("RESULT by action state (change frames, top 25):")
Enum.each(Enum.take(report.by_action_change_frames, 25), fn s -> Output.puts("  #{String.pad_trailing(to_string(s.action), 24)} n #{String.pad_leading(to_string(s.n), 6)}  families #{s.families}%  joint #{s.joint}%  main dir #{s.main_dir}%  change recall #{s.change_recall}%  surprise #{s.surprise}") end)
Output.puts("RESULT most surprising (top 8 of #{top_n} in the json):")
Enum.each(Enum.take(top, 8), fn t -> Output.puts("  #{t.game} f#{t.frame} nll #{t.nll} #{t.state} real #{inspect(t.real_edges)} #{t.real.main} | model #{inspect(t.model_edges)} #{t.model.main}") end)
Output.puts("RESULT agreement report -> #{out}")
