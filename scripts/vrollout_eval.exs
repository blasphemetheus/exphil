# Plan (c) — V-rollout selector, OFFLINE eval (09-01, v2 trunk-V design).
#
# The critic ladder scored candidates with a bilinear S(s,a) and stayed
# under the wire-live bar on v1.2-ARrefit (+3.6/+1.0 over mode-of-N,
# eval_runs/0831_critic_refit). This replaces the scorer with the G3b
# dynamics model: each candidate action is HELD for k frames of open-loop
# rollout in embed space (e'_norm = e_norm + f([e_norm, a])), the imagined
# embeds are appended to the row's REAL embed window (last window-k real +
# k predicted), that window runs through the POLICY'S OWN TRUNK, and V —
# an MLP on trunk features -> discounted return-to-go — scores the result:
#
#   score(s, a) = V( trunk( [e_{t-49..t} ; f^1..k(norm(e_t), a)] ) )
#
# Design history (both negatives recorded in git): V-on-raw-embeds is
# hopeless — linear anti-correlates (rank 0.406), MLP is exactly chance
# (0.503). The 08-31 critic's V reached 0.593 ONLY with trunk features,
# hence the imagined-window-through-trunk construction.
#
# Metrics on held-out replays' DECISION rows (same rules as the ladder):
#   sampling pass@1 / selector pass@1 / oracle pass@K, shuffled-STATE
#   control (candidates scored from another row's state — the ladder's
#   "generic action-frequency preference" control), V R^2 + pair-rank.
#
#   mix run scripts/vrollout_eval.exs \
#     --policy checkpoints/fox_gen_v1.2_ARrefit_policy.bin \
#     --dynamics checkpoints/dynamics_fox_v11AR.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --train-files 25 --eval-files 8 --k 16 --rollout-k 10 \
#     --out eval_runs/0901_vrollout/RESULTS.md
#
# Options: --char-id (2) · --temperature (0.5) · --gamma (0.99) ·
#   --horizon (600) · --seed (20260901) · --max-rows (3000) ·
#   --train-stride (4)
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.AdvantageWeighting
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, dynamics: :string, replays: :string,
             train_files: :integer, eval_files: :integer, k: :integer,
             rollout_k: :integer, char_id: :integer, temperature: :float,
             gamma: :float, horizon: :integer, seed: :integer,
             max_rows: :integer, train_stride: :integer, out: :string]
  )

policy = opts[:policy] || raise "--policy required"
dyn_path = opts[:dynamics] || raise "--dynamics required"
glob = opts[:replays] || raise "--replays required"
n_train = opts[:train_files] || 25
n_eval = opts[:eval_files] || 8
k = opts[:k] || 16
rollout_k = opts[:rollout_k] || 10
char_id = opts[:char_id] || 2
temperature = opts[:temperature] || 0.5
gamma = opts[:gamma] || 0.99
horizon = opts[:horizon] || 600
seed = opts[:seed] || 20_260_901
max_rows = opts[:max_rows] || 3000
train_stride = opts[:train_stride] || 4

Output.banner("Plan (c) V-rollout selector — offline eval (trunk-V)")

dyn = dyn_path |> File.read!() |> :erlang.binary_to_term()
d = dyn.config.embed_dim
hidden = dyn.config.hidden

dyn_model =
  Axon.input("x", shape: {nil, d + 13})
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(d)

{_, dyn_predict} = Axon.build(dyn_model, mode: :inference)
mu = dyn.mu
sd = dyn.sd
norm = fn e -> Nx.divide(Nx.subtract(e, mu), sd) end
denorm = fn e_norm -> Nx.add(Nx.multiply(e_norm, sd), mu) end

trunk = Activations.load_trunk(policy)
heads = Activations.load_heads_only(policy)
config = Map.get(trunk, :config, %{})
window = trunk.window
h = trunk.hidden_size

if rollout_k >= window, do: raise("rollout_k must be < window")

Output.config([
  {"Policy", Path.basename(policy)},
  {"Dynamics", "#{Path.basename(dyn_path)} (R2 #{Float.round(dyn.config.r2, 3)}, cos@#{dyn.config.k} #{Float.round(dyn.config.cos_at_k, 3)})"},
  {"Files", "#{n_train} train / #{n_eval} eval"},
  {"K candidates / rollout k", "#{k} / #{rollout_k}"},
  {"Trunk hidden (V input)", h},
  {"Temperature", temperature}
])

# ---- files -----------------------------------------------------------------
files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take((n_train + n_eval) * 3)

resolve = fn path ->
  case Peppi.metadata(path) do
    {:ok, meta} ->
      case Enum.filter(meta.players, &(&1.character == char_id)) do
        [%{port: p}] -> {:ok, p}
        _ -> :skip
      end

    _ -> :skip
  end
end

picked =
  files
  |> Enum.flat_map(fn f ->
    case resolve.(f) do
      {:ok, p} -> [{f, p}]
      _ -> []
    end
  end)
  |> Enum.take(n_train + n_eval)

{eval_pairs, train_pairs} = Enum.split(picked, n_eval)
Output.puts("  #{length(train_pairs)} train / #{length(eval_pairs)} eval files")

embeds_and_frames = fn {path, port} ->
  opp = if port == 1, do: 2, else: 1

  with {:ok, replay} <- Peppi.parse(path) do
    frames =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
      |> Enum.reject(&(&1.game_state.frame < 0))

    if length(frames) < 120 do
      nil
    else
      ds = Activations.embed_frames(frames, config)

      emb =
        case Nx.rank(ds.embedded_frames) do
          2 -> ds.embedded_frames
          3 -> ds.embedded_frames[[.., 0, ..]]
        end

      {Nx.backend_copy(emb, Nx.BinaryBackend), frames, port}
    end
  else
    _ -> nil
  end
end

trunk_feats = fn emb, indices ->
  indices
  |> Enum.chunk_every(256)
  |> Enum.map(fn ts ->
    wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
    trunk.predict_fn.(trunk.params, Nx.stack(wins))
  end)
  |> Nx.concatenate(axis: 0)
end

rtg_of = fn frames, port ->
  frames
  |> AdvantageWeighting.standard_rewards(port)
  |> AdvantageWeighting.return_to_go(gamma, horizon)
end

# ---- V: MLP on TRUNK features -> return-to-go ------------------------------
Output.puts("Building V training set (trunk features at stride #{train_stride})...")

{v_feats, v_y} =
  train_pairs
  |> Enum.map(embeds_and_frames)
  |> Enum.reject(&is_nil/1)
  |> Enum.map(fn {emb, frames, port} ->
    n = Nx.axis_size(emb, 0)
    idx = Enum.to_list((window - 1)..(n - 1)//train_stride)
    rtg = rtg_of.(frames, port) |> List.to_tuple()

    {trunk_feats.(emb, idx) |> Nx.backend_copy(Nx.BinaryBackend),
     Nx.tensor(Enum.map(idx, &elem(rtg, &1)), type: :f32)}
  end)
  |> Enum.unzip()

v_feats = Nx.concatenate(v_feats, axis: 0)
v_y = Nx.concatenate(v_y, axis: 0)
nv = Nx.axis_size(v_feats, 0)
Output.puts("  #{nv} V-training rows")

v_mu = Nx.mean(v_feats, axes: [0])
v_sd = Nx.max(Nx.standard_deviation(v_feats, axes: [0]), 1.0e-3)
vnorm = fn f -> Nx.divide(Nx.subtract(f, v_mu), v_sd) end

v_model =
  Axon.input("f", shape: {nil, h})
  |> Axon.dense(256, activation: :relu)
  |> Axon.dense(256, activation: :relu)
  |> Axon.dense(1)

v_batch = 2048
v_key = Nx.Random.key(seed + 1)
{v_perm, _} = Nx.Random.shuffle(v_key, Nx.iota({nv}))
v_x_s = Nx.take(vnorm.(v_feats), v_perm, axis: 0)
v_y_s = Nx.take(v_y, v_perm, axis: 0) |> Nx.reshape({nv, 1})
nb = div(nv, v_batch)

v_data =
  Stream.map(0..(nb - 1), fn i ->
    {%{"f" => Nx.slice_along_axis(v_x_s, i * v_batch, v_batch, axis: 0)},
     Nx.slice_along_axis(v_y_s, i * v_batch, v_batch, axis: 0)}
  end)

v_loop = Axon.Loop.trainer(v_model, :mean_squared_error, Polaris.Optimizers.adam(learning_rate: 1.0e-3))
v_state = Axon.Loop.run(v_loop, v_data, %{}, epochs: 3, compiler: EXLA)
{_, v_predict} = Axon.build(v_model, mode: :inference)

v_of = fn feats ->
  m = Nx.axis_size(feats, 0)
  v_predict.(v_state, %{"f" => vnorm.(feats)}) |> Nx.reshape({m})
end

# ---- eval rows: decision frames, candidates, embeds, real windows ----------
Output.puts("Building eval rows...")
key = Nx.Random.key(seed)
keep_real = window - rollout_k

{re0, rwin, rfeats, rsamp, rmatch, rrtg, _key} =
  eval_pairs
  |> Enum.map(embeds_and_frames)
  |> Enum.reject(&is_nil/1)
  |> Enum.with_index()
  |> Enum.reduce({[], [], [], [], [], [], key}, fn {{emb, frames, port}, fi},
                                                  {ae, aw, af, as_, am, ar, key} ->
    n = Nx.axis_size(emb, 0)
    sits = ExPhil.Situations.label_states(Enum.map(frames, & &1.game_state), port, as: :set)
    prevs = [nil | Enum.map(Enum.drop(frames, -1), & &1.controller)]
    dl = CriticFeatures.decision_labels()

    dec_idx =
      Enum.zip([Enum.with_index(frames), sits, prevs])
      |> Enum.filter(fn {{f, i}, set, prev} ->
        i >= window - 1 and set != nil and not MapSet.disjoint?(set, dl) and
          prev != nil and not CriticFeatures.controller_match?(prev, f.controller)
      end)
      |> Enum.map(fn {{_f, i}, _s, _p} -> i end)

    if dec_idx == [] do
      {ae, aw, af, as_, am, ar, key}
    else
      feats = trunk_feats.(emb, dec_idx)
      dec_frames = Enum.map(dec_idx, &Enum.at(frames, &1))
      sub = Nx.Random.fold_in(key, fi)
      {samples, match, _} = CriticFeatures.sample_candidates(heads, feats, dec_frames, k, temperature, sub)

      rtg = rtg_of.(frames, port) |> List.to_tuple()
      e0 = Nx.take(emb, Nx.tensor(dec_idx), axis: 0)

      wins =
        dec_idx
        |> Enum.map(&Nx.slice_along_axis(emb, &1 - keep_real + 1, keep_real, axis: 0))
        |> Nx.stack()

      {[Nx.backend_copy(e0, Nx.BinaryBackend) | ae],
       [Nx.backend_copy(wins, Nx.BinaryBackend) | aw],
       [Nx.backend_copy(feats, Nx.BinaryBackend) | af],
       [samples | as_], [match | am],
       [Nx.tensor(Enum.map(dec_idx, &elem(rtg, &1)), type: :f32) | ar], key}
    end
  end)

cat = fn l -> Nx.concatenate(Enum.reverse(l), axis: 0) end
e0 = cat.(re0)
wins_real = cat.(rwin)
feats0 = cat.(rfeats)
samples = cat.(rsamp)
match = cat.(rmatch)
rtg_rows = cat.(rrtg)

m = min(Nx.axis_size(e0, 0), max_rows)
slice = fn t -> Nx.slice_along_axis(t, 0, m, axis: 0) end
e0 = slice.(e0)
wins_real = slice.(wins_real)
feats0 = slice.(feats0)
samples = slice.(samples)
match = slice.(match)
rtg_rows = slice.(rtg_rows)
Output.puts("  #{m} decision rows")

# ---- V sanity on eval rows (real trunk features) ---------------------------
v_pred = v_of.(feats0)
err = Nx.subtract(v_pred, rtg_rows)
ss_res = Nx.sum(Nx.pow(err, 2))
ss_tot = Nx.sum(Nx.pow(Nx.subtract(rtg_rows, Nx.mean(rtg_rows)), 2))
v_r2 = 1.0 - Nx.to_number(ss_res) / max(Nx.to_number(ss_tot), 1.0e-9)

{pi, key} = Nx.Random.randint(key, 0, m, shape: {20_000})
{pj, key} = Nx.Random.randint(key, 0, m, shape: {20_000})
dy = Nx.subtract(Nx.take(rtg_rows, pi), Nx.take(rtg_rows, pj))
dp = Nx.subtract(Nx.take(v_pred, pi), Nx.take(v_pred, pj))
inf = Nx.greater(Nx.abs(dy), 1.0e-6)
agree = Nx.logical_and(Nx.equal(Nx.sign(dy), Nx.sign(dp)), inf)
v_rank = Nx.to_number(Nx.sum(agree)) / max(Nx.to_number(Nx.sum(inf)), 1)

Output.puts("  V on eval decision rows: R^2 #{Float.round(v_r2, 3)}, pair-rank #{Float.round(v_rank, 3)}")

# ---- rollout scoring: imagined window through the trunk --------------------
Output.puts("Scoring #{m} rows x #{k} candidates (rollout #{rollout_k}, imagined windows)...")

chunk = 128

score_rows = fn e0_src, wins_src ->
  0..(m - 1)
  |> Enum.to_list()
  |> Enum.chunk_every(chunk)
  |> Enum.map(fn idxs ->
    ids = Nx.tensor(idxs)
    c = length(idxs)
    ce = Nx.take(e0_src, ids, axis: 0)
    cw = Nx.take(wins_src, ids, axis: 0)
    ca = Nx.take(samples, ids, axis: 0)

    flat_e = norm.(ce) |> Nx.new_axis(1) |> Nx.broadcast({c, k, d}) |> Nx.reshape({c * k, d})
    flat_a = Nx.reshape(ca, {c * k, 13})

    {_, traj_rev} =
      Enum.reduce(1..rollout_k, {flat_e, []}, fn _j, {cur, acc} ->
        nxt = Nx.add(cur, dyn_predict.(dyn.params, %{"x" => Nx.concatenate([cur, flat_a], axis: 1)}))
        {nxt, [nxt | acc]}
      end)

    traj = traj_rev |> Enum.reverse() |> Nx.stack(axis: 1)
    traj_raw = denorm.(traj)

    wexp =
      cw
      |> Nx.new_axis(1)
      |> Nx.broadcast({c, k, keep_real, d})
      |> Nx.reshape({c * k, keep_real, d})

    full = Nx.concatenate([wexp, traj_raw], axis: 1)

    feats =
      0..(c * k - 1)
      |> Enum.chunk_every(512)
      |> Enum.map(fn sub ->
        trunk.predict_fn.(trunk.params, Nx.take(full, Nx.tensor(sub), axis: 0))
      end)
      |> Nx.concatenate(axis: 0)

    v_of.(feats) |> Nx.reshape({c, k})
  end)
  |> Nx.concatenate(axis: 0)
end

scores = score_rows.(e0, wins_real)

# shuffled-STATE control: same candidates, another row's state + window
{perm, _key} = Nx.Random.shuffle(key, Nx.iota({m}))
scores_shuf = score_rows.(Nx.take(e0, perm, axis: 0), Nx.take(wins_real, perm, axis: 0))

pick = fn sc ->
  sel = Nx.argmax(sc, axis: 1)
  Nx.to_number(Nx.mean(Nx.as_type(Nx.take_along_axis(match, Nx.new_axis(sel, 1), axis: 1), :f32)))
end

sampling_p1 = Nx.to_number(Nx.mean(Nx.as_type(match, :f32)))
selector_p1 = pick.(scores)
shuffled_p1 = pick.(scores_shuf)
oracle = Nx.to_number(Nx.mean(Nx.as_type(Nx.greater(Nx.sum(match, axes: [1]), 0), :f32)))

pctf = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
f3 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

gap =
  if oracle - sampling_p1 > 1.0e-9,
    do: pctf.((selector_p1 - sampling_p1) / (oracle - sampling_p1)),
    else: "—"

report = """
# Plan (c) V-rollout selector — offline RESULTS (trunk-V design)

Policy #{Path.basename(policy)}, dynamics #{Path.basename(dyn_path)}
(1-step R^2 #{f3.(dyn.config.r2)}, cos@#{dyn.config.k} #{f3.(dyn.config.cos_at_k)}).
V = 256x256 MLP on TRUNK features -> RTG (gamma=#{gamma}, horizon #{horizon}f),
#{nv} rows from #{length(train_pairs)} train replays. Scoring: candidate held
#{rollout_k} frames of open-loop embed rollout; imagined window (#{keep_real}
real + #{rollout_k} predicted) -> trunk -> V. Eval: #{m} decision rows from
#{length(eval_pairs)} held-out replays, K=#{k} coherent candidates at T=#{temperature}.

| metric | value |
|---|---:|
| V held-out R^2 (decision rows, real feats) | #{f3.(v_r2)} |
| V pair-rank acc (chance 0.5) | #{f3.(v_rank)} |
| sampling pass@1 | #{pctf.(sampling_p1)}% |
| **V-rollout selector pass@1** | **#{pctf.(selector_p1)}%** |
| selector, shuffled-STATE control | #{pctf.(shuffled_p1)}% |
| oracle pass@#{k} | #{pctf.(oracle)}% |

Gap recovered: #{gap}% of (oracle - sampling).

Design history (git): V-on-raw-embeds failed twice — linear ANTI-correlated
(rank 0.406), MLP exactly chance (0.503); trunk features are where RTG is
readable (the 08-31 critic's 0.593 was trunk+raw phi).

Reference (same corpus, direct bilinear critic on ARrefit,
eval_runs/0831_critic_refit): sampling 1.3 / mode-of-N 7.2 / selector 10.8 /
oracle 13.9 (different row rule — compare within-table margins only).

Caveats: candidate held constant through the rollout (no reactive policy in
the imagination); imagined embeds are approximate trunk inputs; match-rate
is a mode-seeking proxy — the live gate (frozen-input <= 0.20, 7/8 cap,
F1 airdodge, F2 commitment, F3 approach_delta) stays the real judge before
any decode change.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
