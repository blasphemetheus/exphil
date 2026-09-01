# Plan (c) — V-rollout selector, OFFLINE eval (09-01, v3: discriminative
# scorer on imagined-rollout trunk features).
#
# Scorer: each candidate action is HELD for k frames of open-loop rollout in
# embed space (G3b dynamics, e'_norm = e_norm + f([e_norm, a])); the imagined
# embeds are appended to the row's REAL embed window (window-k real + k
# predicted) and run through the POLICY'S OWN TRUNK; a small MLP g on those
# trunk features scores the candidate:
#
#   score(s, a) = g( trunk( [e_{t-(w-k)+1..t} ; f^1..k(norm(e_t), a)] ) )
#
# g is trained DISCRIMINATIVELY (binary CE: master's action = 1, K policy
# samples = 0, on train replays' decision rows) — NOT to predict return.
#
# Design history (all negatives in git, 09-01):
#   v1  V-on-raw-embeds, linear ridge      -> ANTI-correlated (rank 0.406)
#   v1b V-on-raw-embeds, MLP               -> exactly chance (0.503)
#   v2  V-on-trunk-feats(imagined), RTG    -> V works (rank 0.554) but the
#       selector doesn't move (0.5 vs 0.7 sampling; shuffled identical):
#       a prediction-trained V cannot see the consequence differences a
#       10-frame held action makes. Hence discriminative g.
#
# Metrics on held-out replays' DECISION rows: sampling pass@1 / selector
# pass@1 / oracle pass@K, shuffled-STATE control (same candidates scored
# from a permuted row's state+window), master-vs-sample score separation.
#
#   mix run scripts/vrollout_eval.exs \
#     --policy checkpoints/fox_gen_v1.2_ARrefit_policy.bin \
#     --dynamics checkpoints/dynamics_fox_v11AR.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --train-files 25 --eval-files 8 --k 16 --rollout-k 10 \
#     --out eval_runs/0901_vrollout/RESULTS.md
#
# Options: --char-id (2) · --temperature (0.5) · --seed (20260901) ·
#   --max-rows (3000) · --train-rows (6000) · --epochs (2)
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Data.Peppi
alias ExPhil.Embeddings.Controller, as: ControllerEmbed
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, dynamics: :string, replays: :string,
             train_files: :integer, eval_files: :integer, k: :integer,
             rollout_k: :integer, char_id: :integer, temperature: :float,
             seed: :integer, max_rows: :integer, train_rows: :integer,
             epochs: :integer, out: :string]
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
seed = opts[:seed] || 20_260_901
max_rows = opts[:max_rows] || 3000
train_rows_cap = opts[:train_rows] || 6000
g_epochs = opts[:epochs] || 2

Output.banner("Plan (c) V-rollout selector — offline eval v3 (discriminative)")

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
keep_real = window - rollout_k
if rollout_k >= window, do: raise("rollout_k must be < window")

Output.config([
  {"Policy", Path.basename(policy)},
  {"Dynamics", "#{Path.basename(dyn_path)} (R2 #{Float.round(dyn.config.r2, 3)}, cos@#{dyn.config.k} #{Float.round(dyn.config.cos_at_k, 3)})"},
  {"Files", "#{n_train} train / #{n_eval} eval"},
  {"K candidates / rollout k", "#{k} / #{rollout_k}"},
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

# ---- row builder: decision rows w/ candidates, master action, windows ------
build_rows = fn pairs, cap, key ->
  {re0, rwin, rsamp, rmast, rmatch, _key} =
    pairs
    |> Enum.map(embeds_and_frames)
    |> Enum.reject(&is_nil/1)
    |> Enum.with_index()
    |> Enum.reduce({[], [], [], [], [], key}, fn {{emb, frames, port}, fi},
                                                {ae, aw, as_, amst, am, key} ->
      _ = port
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
        {ae, aw, as_, amst, am, key}
      else
        feats = trunk_feats.(emb, dec_idx)
        dec_frames = Enum.map(dec_idx, &Enum.at(frames, &1))
        sub = Nx.Random.fold_in(key, fi)
        {samples, match, _} = CriticFeatures.sample_candidates(heads, feats, dec_frames, k, temperature, sub)
        a_master = ControllerEmbed.embed_continuous_batch(Enum.map(dec_frames, & &1.controller))

        e0 = Nx.take(emb, Nx.tensor(dec_idx), axis: 0)

        wins =
          dec_idx
          |> Enum.map(&Nx.slice_along_axis(emb, &1 - keep_real + 1, keep_real, axis: 0))
          |> Nx.stack()

        {[Nx.backend_copy(e0, Nx.BinaryBackend) | ae],
         [Nx.backend_copy(wins, Nx.BinaryBackend) | aw],
         [samples | as_],
         [Nx.backend_copy(a_master, Nx.BinaryBackend) | amst],
         [match | am], key}
      end
    end)

  cat = fn l -> Nx.concatenate(Enum.reverse(l), axis: 0) end
  rows = %{e0: cat.(re0), wins: cat.(rwin), samples: cat.(rsamp), master: cat.(rmast), match: cat.(rmatch)}
  m = min(Nx.axis_size(rows.e0, 0), cap)
  Map.new(rows, fn {kk, t} -> {kk, Nx.slice_along_axis(t, 0, m, axis: 0)} end)
end

key = Nx.Random.key(seed)
Output.puts("Building train rows...")
tr = build_rows.(train_pairs, train_rows_cap, Nx.Random.fold_in(key, 1))
Output.puts("  #{Nx.axis_size(tr.e0, 0)} train decision rows")
Output.puts("Building eval rows...")
ev = build_rows.(eval_pairs, max_rows, Nx.Random.fold_in(key, 2))
m_ev = Nx.axis_size(ev.e0, 0)
Output.puts("  #{m_ev} eval decision rows")

# ---- imagined-window trunk features for a set of actions -------------------
# rows: %{e0 {n,d}, wins {n,w-k,d}}; actions {n, J, 13} -> feats {n, J, h}
imagined_feats = fn e0_src, wins_src, actions ->
  n = Nx.axis_size(e0_src, 0)
  j = Nx.axis_size(actions, 1)

  0..(n - 1)
  |> Enum.to_list()
  |> Enum.chunk_every(128)
  |> Enum.map(fn idxs ->
    ids = Nx.tensor(idxs)
    c = length(idxs)
    ce = Nx.take(e0_src, ids, axis: 0)
    cw = Nx.take(wins_src, ids, axis: 0)
    ca = Nx.take(actions, ids, axis: 0)

    flat_e = norm.(ce) |> Nx.new_axis(1) |> Nx.broadcast({c, j, d}) |> Nx.reshape({c * j, d})
    flat_a = Nx.reshape(ca, {c * j, 13})

    {_, traj_rev} =
      Enum.reduce(1..rollout_k, {flat_e, []}, fn _s, {cur, acc} ->
        nxt = Nx.add(cur, dyn_predict.(dyn.params, %{"x" => Nx.concatenate([cur, flat_a], axis: 1)}))
        {nxt, [nxt | acc]}
      end)

    traj_raw = traj_rev |> Enum.reverse() |> Nx.stack(axis: 1) |> denorm.()

    wexp =
      cw
      |> Nx.new_axis(1)
      |> Nx.broadcast({c, j, keep_real, d})
      |> Nx.reshape({c * j, keep_real, d})

    full = Nx.concatenate([wexp, traj_raw], axis: 1)

    0..(c * j - 1)
    |> Enum.chunk_every(512)
    |> Enum.map(fn sub ->
      trunk.predict_fn.(trunk.params, Nx.take(full, Nx.tensor(sub), axis: 0))
    end)
    |> Nx.concatenate(axis: 0)
    |> Nx.reshape({c, j, h})
  end)
  |> Nx.concatenate(axis: 0)
end

# ---- g: discriminative scorer on imagined feats ----------------------------
Output.puts("Computing imagined features for g training (master + #{k} negatives)...")
tr_actions = Nx.concatenate([Nx.new_axis(tr.master, 1), tr.samples], axis: 1)
tr_feats = imagined_feats.(tr.e0, tr.wins, tr_actions)
n_tr = Nx.axis_size(tr_feats, 0)

g_mu = Nx.mean(Nx.reshape(tr_feats, {n_tr * (k + 1), h}), axes: [0])
g_sd = Nx.max(Nx.standard_deviation(Nx.reshape(tr_feats, {n_tr * (k + 1), h}), axes: [0]), 1.0e-3)
gnorm = fn f -> Nx.divide(Nx.subtract(f, g_mu), g_sd) end

g_model =
  Axon.input("f", shape: {nil, h})
  |> Axon.dense(256, activation: :relu)
  |> Axon.dense(256, activation: :relu)
  |> Axon.dense(1, activation: :sigmoid)

flat_x = gnorm.(Nx.reshape(tr_feats, {n_tr * (k + 1), h}))

labels =
  Nx.concatenate([Nx.broadcast(1.0, {n_tr, 1}), Nx.broadcast(0.0, {n_tr, k})], axis: 1)
  |> Nx.reshape({n_tr * (k + 1), 1})

n_flat = n_tr * (k + 1)
g_key = Nx.Random.key(seed + 7)
{g_perm, _} = Nx.Random.shuffle(g_key, Nx.iota({n_flat}))
flat_x = Nx.take(flat_x, g_perm, axis: 0)
labels = Nx.take(labels, g_perm, axis: 0)

g_batch = 2048
nb = div(n_flat, g_batch)

g_data =
  Stream.map(0..(nb - 1), fn i ->
    {%{"f" => Nx.slice_along_axis(flat_x, i * g_batch, g_batch, axis: 0)},
     Nx.slice_along_axis(labels, i * g_batch, g_batch, axis: 0)}
  end)

Output.puts("Training g (binary CE, #{n_flat} rows, #{g_epochs} epochs)...")
g_loop = Axon.Loop.trainer(g_model, :binary_cross_entropy, Polaris.Optimizers.adam(learning_rate: 1.0e-3))
g_state = Axon.Loop.run(g_loop, g_data, %{}, epochs: g_epochs, compiler: EXLA)
{_, g_predict} = Axon.build(g_model, mode: :inference)

g_of = fn feats3 ->
  n = Nx.axis_size(feats3, 0)
  j = Nx.axis_size(feats3, 1)
  g_predict.(g_state, %{"f" => gnorm.(Nx.reshape(feats3, {n * j, h}))}) |> Nx.reshape({n, j})
end

# ---- eval ------------------------------------------------------------------
Output.puts("Scoring eval rows (real + shuffled-state control)...")
ev_actions = Nx.concatenate([Nx.new_axis(ev.master, 1), ev.samples], axis: 1)
ev_feats = imagined_feats.(ev.e0, ev.wins, ev_actions)
ev_scores_all = g_of.(ev_feats)
master_scores = ev_scores_all[[.., 0]]
scores = ev_scores_all[[.., 1..k]]

{perm, _key} = Nx.Random.shuffle(key, Nx.iota({m_ev}))
ev_feats_shuf = imagined_feats.(Nx.take(ev.e0, perm, axis: 0), Nx.take(ev.wins, perm, axis: 0), ev.samples)
scores_shuf = g_of.(ev_feats_shuf)

match = ev.match

pick = fn sc ->
  sel = Nx.argmax(sc, axis: 1)
  Nx.to_number(Nx.mean(Nx.as_type(Nx.take_along_axis(match, Nx.new_axis(sel, 1), axis: 1), :f32)))
end

sampling_p1 = Nx.to_number(Nx.mean(Nx.as_type(match, :f32)))
selector_p1 = pick.(scores)
shuffled_p1 = pick.(scores_shuf)
oracle = Nx.to_number(Nx.mean(Nx.as_type(Nx.greater(Nx.sum(match, axes: [1]), 0), :f32)))

# master-top1 among K+1 (chance 1/(K+1)); mean score separation
master_top1 =
  Nx.to_number(Nx.mean(Nx.as_type(Nx.equal(Nx.argmax(ev_scores_all, axis: 1), 0), :f32)))

sep_master = Nx.to_number(Nx.mean(master_scores))
sep_samples = Nx.to_number(Nx.mean(scores))

pctf = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end
f3 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

gap =
  if oracle - sampling_p1 > 1.0e-9,
    do: pctf.((selector_p1 - sampling_p1) / (oracle - sampling_p1)),
    else: "—"

report = """
# Plan (c) V-rollout selector — offline RESULTS (v3, discriminative g)

Policy #{Path.basename(policy)}, dynamics #{Path.basename(dyn_path)}
(1-step R^2 #{f3.(dyn.config.r2)}, cos@#{dyn.config.k} #{f3.(dyn.config.cos_at_k)}).
Scorer g = 256x256 MLP on trunk(imagined window: #{keep_real} real + #{rollout_k}
predicted embeds under the held candidate), trained discriminatively
(binary CE, master=1 vs #{k} policy samples=0) on #{n_tr} train decision rows
from #{length(train_pairs)} replays. Eval: #{m_ev} decision rows from
#{length(eval_pairs)} held-out replays, K=#{k} coherent candidates at T=#{temperature}.

| metric | value |
|---|---:|
| sampling pass@1 | #{pctf.(sampling_p1)}% |
| **rollout-g selector pass@1** | **#{pctf.(selector_p1)}%** |
| selector, shuffled-STATE control | #{pctf.(shuffled_p1)}% |
| oracle pass@#{k} | #{pctf.(oracle)}% |
| master top-1 among K+1 (chance #{pctf.(1 / (k + 1))}%) | #{pctf.(master_top1)}% |
| mean g: master / samples | #{f3.(sep_master)} / #{f3.(sep_samples)} |

Gap recovered: #{gap}% of (oracle - sampling).

Design history (git): v1 linear V-on-embeds ANTI-correlated (0.406); v1b MLP
V-on-embeds chance (0.503); v2 RTG-trained V on imagined trunk feats works
as a value fn (rank 0.554) but cannot separate candidates (selector 0.5 vs
sampling 0.7) -> v3 trains the scorer discriminatively on the same features.

Reference (same corpus, direct bilinear critic on ARrefit,
eval_runs/0831_critic_refit): sampling 1.3 / mode-of-N 7.2 / selector 10.8 /
oracle 13.9 (different row rule — compare within-table margins only).

Caveats: candidate held constant through the rollout; imagined embeds are
approximate trunk inputs; match-rate is a mode-seeking proxy — the live
gate (frozen-input <= 0.20, 7/8 cap, F1 airdodge, F2 commitment, F3
approach_delta) stays the real judge before any decode change.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
