# G3b spike — learned one-step dynamics model (INTERP_GEN_V1 G3b; the
# CycleSim successor). Trains f: (embed_t ⊕ action_t) -> embed_{t+1} on
# real replays and measures whether open-loop rollouts hold together long
# enough to serve the V-rollout selector (plan step (c), 08-31).
#
# Rollout eval is TEACHER-FORCED on actions (ground-truth action sequence,
# predicted states) — that is the engine's actual job: score a GIVEN
# candidate action's consequences a few frames out.
#
# Spike gate (declared before running): held-out 1-step R^2 > 0.9 on the
# continuous dims AND k=10 open-loop cosine similarity > 0.8 -> good enough
# to build V-rollouts on; else record what drifts first and stop.
#
#   mix run scripts/dynamics_spike.exs \
#     --policy checkpoints/fox_gen_v1.1_AR_20260831_080100_policy.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --limit-files 60 --epochs 2 --out eval_runs/0831_dynamics_spike/RESULTS.md
#
# Options: --limit-files (60) · --val-files (8, held out by file) ·
#   --epochs (2) · --batch-size (1024) · --hidden (512) · --k (10) ·
#   --char-id (2) · --seed (20260831) · --out PATH
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, limit_files: :integer, val_files: :integer,
             epochs: :integer, batch_size: :integer, hidden: :integer, k: :integer,
             char_id: :integer, seed: :integer, out: :string, save: :string]
  )

policy = opts[:policy] || raise "--policy required (for the embed config)"
glob = opts[:replays] || raise "--replays required"
limit_files = opts[:limit_files] || 60
val_n = opts[:val_files] || 8
epochs = opts[:epochs] || 2
batch = opts[:batch_size] || 1024
hidden = opts[:hidden] || 512
k_steps = opts[:k] || 10
char_id = opts[:char_id] || 2
seed = opts[:seed] || 20_260_831

trunk = Activations.load_trunk(policy)
policy_config = Map.get(trunk, :config)

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files * 3)

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
  |> Enum.take(limit_files)

{val_pairs, train_pairs} = Enum.split(picked, val_n)

Output.banner("G3b dynamics spike")

Output.config([
  {"Embed config from", Path.basename(policy)},
  {"Train files", length(train_pairs)},
  {"Val files (held out)", length(val_pairs)},
  {"Epochs", epochs},
  {"k-step rollout", k_steps}
])

# Per-replay {embeds {n,d}, actions {n,13}} — consecutive-frame pairs never
# cross a replay boundary.
load_replay = fn {path, p} ->
  o = if p == 1, do: 2, else: 1

  with {:ok, replay} <- Peppi.parse(path, player_port: p) do
    frames =
      replay
      |> Peppi.to_training_frames(player_port: p, opponent_port: o, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))

    if length(frames) < 120 do
      nil
    else
      # embed_frames returns the DATASET with :embedded_frames set (a
      # stacked {n, d} tensor; {n, variants, d} on multi-delay configs).
      ds = Activations.embed_frames(frames, policy_config)

      embeds =
        case Nx.rank(ds.embedded_frames) do
          2 -> ds.embedded_frames
          3 -> ds.embedded_frames[[.., 0, ..]]
        end

      acts =
        frames
        |> Enum.map(& &1.controller)
        |> ExPhil.Embeddings.Controller.embed_continuous_batch()

      {Nx.backend_copy(embeds, Nx.BinaryBackend), Nx.backend_copy(acts, Nx.BinaryBackend)}
    end
  else
    _ -> nil
  end
end

Output.puts("Embedding train replays...")
train_data = train_pairs |> Enum.map(load_replay) |> Enum.reject(&is_nil/1)
Output.puts("Embedding val replays...")
val_data = val_pairs |> Enum.map(load_replay) |> Enum.reject(&is_nil/1)

d = train_data |> hd() |> elem(0) |> Nx.axis_size(1)
Output.puts("  embed dim #{d}; #{length(train_data)} train / #{length(val_data)} val replays")

# Normalization from train set (per-dim mean/std) — MSE in normalized space
# so binary/one-hot dims don't drown the continuous physics dims.
all_train = Nx.concatenate(Enum.map(train_data, &elem(&1, 0)), axis: 0)
mu = Nx.mean(all_train, axes: [0])
sd = Nx.max(Nx.standard_deviation(all_train, axes: [0]), 1.0e-3)
norm = fn x -> Nx.divide(Nx.subtract(x, mu), sd) end

pairs_of = fn {e, a} ->
  n = Nx.axis_size(e, 0)
  x = Nx.concatenate([norm.(Nx.slice_along_axis(e, 0, n - 1, axis: 0)),
                      Nx.slice_along_axis(a, 0, n - 1, axis: 0)], axis: 1)
  y = norm.(Nx.slice_along_axis(e, 1, n - 1, axis: 0))
  {x, y}
end

# Single concatenate, not an incremental reduce: the reduce version copies
# the whole accumulated array every iteration and the superseded device
# buffers stay pinned by BEAM refs until GC — exhausted the EXLA pool at
# 52 files (OOM at 552MB with 32GB free, 2026-08-31).
{txs, tys} = train_data |> Enum.map(pairs_of) |> Enum.unzip()
{tx, ty} = {Nx.concatenate(txs, axis: 0), Nx.concatenate(tys, axis: 0)}

{vxs, vys} = val_data |> Enum.map(pairs_of) |> Enum.unzip()
{vx, vy} = {Nx.concatenate(vxs, axis: 0), Nx.concatenate(vys, axis: 0)}

n_train = Nx.axis_size(tx, 0)
Output.puts("  #{n_train} train pairs, #{Nx.axis_size(vx, 0)} val pairs")

model =
  Axon.input("x", shape: {nil, d + 13})
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(d)

# Residual target: predict the DELTA (next - current) — identity is the
# dominant component of frame-to-frame dynamics; make the net learn what
# CHANGES. y_hat = x_state + f(x); implemented by training on (y - x_state).
x_state = fn x -> Nx.slice_along_axis(x, 0, d, axis: 1) end
t_delta = fn x, y -> Nx.subtract(y, x_state.(x)) end

batches = fn x, y, key ->
  n = Nx.axis_size(x, 0)
  {perm, _} = Nx.Random.shuffle(key, Nx.iota({n}))
  x = Nx.take(x, perm, axis: 0)
  y = Nx.take(y, perm, axis: 0)
  nb = div(n, batch)

  Stream.map(0..(nb - 1), fn i ->
    {Nx.slice_along_axis(x, i * batch, batch, axis: 0),
     Nx.slice_along_axis(y, i * batch, batch, axis: 0)}
  end)
end

loop =
  model
  |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 1.0e-3))

key = Nx.Random.key(seed)

data =
  batches.(tx, ty, key)
  |> Stream.map(fn {x, y} -> {%{"x" => x}, t_delta.(x, y)} end)

state = Axon.Loop.run(loop, data, %{}, epochs: epochs, compiler: EXLA)

{_, predict_fn} = Axon.build(model, mode: :inference)
step = fn x -> Nx.add(x_state.(x), predict_fn.(state, %{"x" => x})) end

# ---- eval: 1-step R^2 on val -----------------------------------------------
pred = Nx.add(x_state.(vx), predict_fn.(state, %{"x" => vx}))
err = Nx.subtract(pred, vy)
ss_res = Nx.sum(Nx.pow(err, 2))
ss_tot = Nx.sum(Nx.pow(Nx.subtract(vy, Nx.mean(vy, axes: [0])), 2))
r2 = 1.0 - Nx.to_number(ss_res) / max(Nx.to_number(ss_tot), 1.0e-9)

# ---- eval: k-step open-loop rollout (ground-truth actions) ------------------
cos = fn a, b ->
  num = Nx.sum(Nx.multiply(a, b))
  den = Nx.multiply(Nx.LinAlg.norm(a), Nx.LinAlg.norm(b))
  Nx.to_number(num) / max(Nx.to_number(den), 1.0e-9)
end

rollouts =
  val_data
  |> Enum.take(4)
  |> Enum.flat_map(fn {e, a} ->
    n = Nx.axis_size(e, 0)
    starts = Enum.take_every(60..(n - k_steps - 2)//1, 120) |> Enum.take(25)

    Enum.map(starts, fn s ->
      e0 = norm.(e[s..s])

      {_, sims} =
        Enum.reduce(1..k_steps, {e0, []}, fn j, {cur, acc} ->
          x = Nx.concatenate([cur, a[(s + j - 1)..(s + j - 1)]], axis: 1)
          nxt = step.(x)
          truth = norm.(e[(s + j)..(s + j)])
          {nxt, [{j, cos.(Nx.squeeze(nxt), Nx.squeeze(truth))} | acc]}
        end)

      Map.new(sims)
    end)
  end)

k_curve =
  Map.new(1..k_steps, fn j ->
    vals = Enum.map(rollouts, & &1[j])
    {j, Enum.sum(vals) / length(vals)}
  end)

f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

curve_rows =
  Enum.map_join(1..k_steps, "\n", fn j -> "| #{j} | #{f2.(k_curve[j])} |" end)

gate =
  if r2 > 0.9 and k_curve[k_steps] > 0.8,
    do: "PASS — build V-rollouts on this",
    else: "FAIL — record what drifts, do not build on it yet"

report = """
# G3b dynamics spike — RESULTS

f: (embed_t ⊕ action_13) -> delta(embed), #{hidden}x#{hidden} MLP,
#{length(train_data)} train / #{length(val_data)} held-out replays
(#{n_train} pairs), normalized space, #{epochs} epochs.

- **held-out 1-step R^2: #{f2.(r2)}** (gate > 0.9)
- **k-step open-loop cosine (ground-truth actions, #{length(rollouts)} rollouts):**

| k | cos sim |
|---|---:|
#{curve_rows}

**Gate (declared pre-run): #{gate}**

Caveats: single-perspective embeds (opponent modeled only through the
embedding's opponent block); cosine in normalized space; no collision/
blast-zone hard constraints — a V-rollout consumer must treat rollouts
as SHORT-horizon (k <= #{k_steps}) texture, not simulation truth
(GOTCHA: this is the learned rung of the fidelity ladder; headless
Dolphin savestates are the mechanics-true audit — INTERP_GEN_V1 G3b).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end

# --save: persist the fitted model for downstream consumers (the V-rollout
# selector). Params/mu/sd on BinaryBackend (GOTCHA #1); model config is
# enough to rebuild the Axon graph exactly.
if save = opts[:save] do
  File.mkdir_p!(Path.dirname(save))

  to_bin = fn t -> Nx.backend_copy(t, Nx.BinaryBackend) end

  blob = %{
    params: Nx.Defn.Composite.traverse(state, to_bin),
    mu: to_bin.(mu),
    sd: to_bin.(sd),
    config: %{embed_dim: d, hidden: hidden, action_dim: 13, residual: true,
              policy: policy, r2: r2, cos_at_k: k_curve[k_steps], k: k_steps}
  }

  File.write!(save, :erlang.term_to_binary(blob))
  Output.success("saved dynamics model -> #{save}")
end
