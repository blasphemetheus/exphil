# D2 critic — step 2: train V(s) and the selector S(s, a); report by RANKING.
#
# Inputs: one or more extracts from critic_extract.exs. Split is BY REPLAY
# (frames within a game are correlated; never split within one).
#
#   V(s)    = w^T phi_std(s) + b          ridge, closed form
#   S(s, a) = a^T W phi_std(s) + v^T a    bilinear, softmax over {master} U {K samples}
#
# Metrics (held-out replays), all on DECISION rows unless --all-rows:
#   V:  R^2 on return-to-go; pairwise ranking accuracy (P(V ranks the
#       higher-return state higher) over random pairs); shuffled-target control.
#   S:  top-1 among K+1 (master included; chance 1/(K+1)); and the Leg S
#       cash-in — among the K SAMPLES only:
#         sampling pass@1  = mean match rate of a random sample
#         selector pass@1  = match rate of the argmax-scored sample
#         oracle pass@K    = any-match rate (the ceiling)
#       plus a shuffled-label control for the selector.
#
# Usage:
#   mix run scripts/critic_train.exs --data cache/critic/fox_gen_v1_ep10_erickfm40.nx \
#     --out checkpoints/critic_fox_gen_v1_ep10.bin --report eval_runs/0829_critic/train.md
#
# Options:
#   --data PATH[,PATH]   extracts (required)
#   --eval-frac F        fraction of replays held out (default 0.25)
#   --l2 F               ridge / selector L2 (default 1e-3)
#   --steps N            selector SGD steps (default 2000)
#   --lr F               selector learning rate (default 0.05)
#   --all-rows           train/eval on every row, not just decision rows
#   --out PATH           save critic params (BinaryBackend, term_to_binary)
#   --report PATH        markdown report
#   --seed N
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Training.Output

defmodule CriticTrain do
  import Nx.Defn

  # ---- V: ridge --------------------------------------------------------------

  # X {n,d} standardized (with bias column appended by caller), y {n}
  def ridge(x, y, l2) do
    d = Nx.axis_size(x, 1)
    # X'X and X'y on the default (device) backend; the {d,d} solve on Binary
    xtx = Nx.dot(Nx.transpose(x), x)
    xty = Nx.dot(Nx.transpose(x), y)
    reg = Nx.multiply(Nx.eye(d), l2 * Nx.axis_size(x, 0))

    prev_backend = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)
    a = Nx.backend_copy(Nx.add(xtx, reg), Nx.BinaryBackend)
    b = Nx.backend_copy(xty, Nx.BinaryBackend)
    w = Nx.LinAlg.solve(a, b)
    Nx.default_backend(prev_backend)
    w
  end

  def r2(pred, y) do
    ss_res = Nx.sum(Nx.pow(Nx.subtract(y, pred), 2))
    ss_tot = Nx.sum(Nx.pow(Nx.subtract(y, Nx.mean(y)), 2))
    Nx.to_number(Nx.subtract(1.0, Nx.divide(ss_res, Nx.max(ss_tot, 1.0e-9))))
  end

  # P(V ranks higher the state with the higher return) over random pairs
  def pair_rank_acc(pred, y, key, pairs \\ 20_000) do
    n = Nx.axis_size(y, 0)
    {i, key} = Nx.Random.randint(key, 0, n, shape: {pairs})
    {j, _} = Nx.Random.randint(key, 0, n, shape: {pairs})
    dy = Nx.subtract(Nx.take(y, i), Nx.take(y, j))
    dp = Nx.subtract(Nx.take(pred, i), Nx.take(pred, j))
    informative = Nx.greater(Nx.abs(dy), 1.0e-6)
    agree = Nx.equal(Nx.sign(dy), Nx.sign(dp))
    num = Nx.sum(Nx.logical_and(agree, informative))
    Nx.to_number(Nx.divide(num, Nx.max(Nx.sum(informative), 1)))
  end

  # ---- S: bilinear selector --------------------------------------------------

  # phi {n,d}, cands {n,c,13}; score {n,c}
  defn scores(%{w: w, v: v}, phi, cands) do
    # a^T W phi : {n,c,13} x ({d,13} -> {n,13}) -> {n,c}
    proj = Nx.dot(phi, w) |> Nx.new_axis(1)
    Nx.sum(cands * proj, axes: [2]) + Nx.dot(cands, v)
  end

  defn loss(params, phi, cands, target, l2) do
    s = scores(params, phi, cands)
    logp = s - Nx.log(Nx.sum(Nx.exp(s - Nx.reduce_max(s, axes: [1], keep_axes: true)), axes: [1], keep_axes: true)) - Nx.reduce_max(s, axes: [1], keep_axes: true)
    nll = -Nx.mean(Nx.sum(logp * target, axes: [1]))
    nll + l2 * (Nx.sum(params.w * params.w) + Nx.sum(params.v * params.v))
  end

  defn step(params, phi, cands, target, l2, lr) do
    {l, g} = value_and_grad(params, &loss(&1, phi, cands, target, l2))
    {%{w: params.w - lr * g.w, v: params.v - lr * g.v}, l}
  end

  def train_selector(phi, cands, target, opts) do
    steps = Keyword.get(opts, :steps, 2000)
    lr = Keyword.get(opts, :lr, 0.05)
    l2 = Keyword.get(opts, :l2, 1.0e-3)
    batch = Keyword.get(opts, :batch, 4096)
    key = Keyword.get(opts, :key, Nx.Random.key(1))
    d = Nx.axis_size(phi, 1)
    n = Nx.axis_size(phi, 0)

    params = %{w: Nx.broadcast(0.0, {d, 13}), v: Nx.broadcast(0.0, {13})}
    step_fn = Nx.Defn.jit(&__MODULE__.step/6)

    {params, _key, last} =
      Enum.reduce(1..steps, {params, key, nil}, fn i, {p, key, _} ->
        {idx, key} = Nx.Random.randint(key, 0, n, shape: {min(batch, n)})
        {p, l} = step_fn.(p, Nx.take(phi, idx), Nx.take(cands, idx), Nx.take(target, idx), l2, lr)
        if rem(i, 200) == 0, do: Output.puts("  selector step #{i}/#{steps} loss #{Float.round(Nx.to_number(l), 4)}")
        {p, key, l}
      end)

    {params, last}
  end

  # ---- selector metrics -----------------------------------------------------

  # top-1 with master at index 0 of cands
  def top1_with_master(params, phi, cands) do
    s = scores(params, phi, cands)
    Nx.mean(Nx.equal(Nx.argmax(s, axis: 1), 0)) |> Nx.to_number()
  end

  # among samples only: sampling pass@1, selector pass@1, oracle pass@K
  def passk_metrics(params, phi, samples, match) do
    m = Nx.as_type(match, :f32)
    s = scores(params, phi, samples)
    pick = Nx.argmax(s, axis: 1)
    picked = Nx.take_along_axis(m, Nx.new_axis(pick, 1), axis: 1) |> Nx.squeeze(axes: [1])

    %{
      sampling: Nx.to_number(Nx.mean(m)),
      selector: Nx.to_number(Nx.mean(picked)),
      oracle: Nx.to_number(Nx.mean(Nx.reduce_max(m, axes: [1])))
    }
  end
end

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      data: :string, eval_frac: :float, l2: :float, steps: :integer, lr: :float,
      all_rows: :boolean, out: :string, report: :string, seed: :integer
    ]
  )

paths = (opts[:data] || raise("--data required")) |> String.split(",")
key = Nx.Random.key(opts[:seed] || 29)
l2 = opts[:l2] || 1.0e-3

Output.banner("D2 critic — train V(s) and selector S(s,a)")

datas = Enum.map(paths, fn p -> p |> CriticFeatures.load!() |> Nx.backend_transfer(Nx.default_backend()) end)

# merge with disjoint replay indices
{data, _} =
  Enum.reduce(datas, {nil, 0}, fn d, {acc, offset} ->
    d = Map.update!(d, :replay_index, &Nx.add(&1, offset))
    n_rep = Nx.to_number(Nx.reduce_max(d.replay_index)) + 1 - offset
    merged = if acc, do: CriticFeatures.concat([acc, d]) |> Map.put(:replay_index, Nx.concatenate([acc.replay_index, d.replay_index])), else: d
    {merged, offset + n_rep}
  end)

n_rep = Nx.to_number(Nx.reduce_max(data.replay_index)) + 1
eval_frac = opts[:eval_frac] || 0.25
{perm, key} = Nx.Random.shuffle(key, Nx.iota({n_rep}))
n_eval = max(round(n_rep * eval_frac), 1)
eval_reps = perm |> Nx.slice([0], [n_eval]) |> Nx.to_list() |> MapSet.new()

is_eval = data.replay_index |> Nx.to_list() |> Enum.map(&if(MapSet.member?(eval_reps, &1), do: 1, else: 0)) |> Nx.tensor(type: :u8)
use_row = if opts[:all_rows], do: Nx.broadcast(Nx.tensor(1, type: :u8), {Nx.axis_size(data.phi, 0)}), else: data.decision

sel = fn mask -> mask |> Nx.to_list() |> Enum.with_index() |> Enum.filter(fn {v, _} -> v == 1 end) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64) end
tr_idx = sel.(Nx.logical_and(use_row, Nx.logical_not(is_eval)))
ev_idx = sel.(Nx.logical_and(use_row, is_eval))

take = fn t, idx -> Nx.take(t, idx, axis: 0) end
phi_tr = take.(data.phi, tr_idx); phi_ev = take.(data.phi, ev_idx)

Output.config([
  {"Rows train/eval", "#{Nx.axis_size(tr_idx, 0)} / #{Nx.axis_size(ev_idx, 0)}"},
  {"Replays train/eval", "#{n_rep - n_eval} / #{n_eval}"},
  {"phi dims", Nx.axis_size(data.phi, 1)},
  {"Rows", if(opts[:all_rows], do: "ALL", else: "DECISION only")},
  {"L2", l2}
])

# standardize on train
mean = Nx.mean(phi_tr, axes: [0], keep_axes: true)
std = Nx.standard_deviation(phi_tr, axes: [0], keep_axes: true) |> Nx.max(1.0e-6)
stdz = fn x -> Nx.divide(Nx.subtract(x, mean), std) end
xs_tr = stdz.(phi_tr); xs_ev = stdz.(phi_ev)
with_bias = fn x -> Nx.concatenate([x, Nx.broadcast(1.0, {Nx.axis_size(x, 0), 1})], axis: 1) end

# ---- V ----
Output.puts("Fitting V(s) (ridge)...")
y_tr = take.(data.rtg, tr_idx); y_ev = take.(data.rtg, ev_idx)
w_v = CriticTrain.ridge(with_bias.(xs_tr), y_tr, l2)
pred_ev = Nx.dot(with_bias.(xs_ev), Nx.backend_copy(w_v, Nx.default_backend()))
v_r2 = CriticTrain.r2(pred_ev, y_ev)
k2 = Nx.Random.fold_in(key, 2)
v_rank = CriticTrain.pair_rank_acc(pred_ev, y_ev, k2)
{y_shuf, key} = Nx.Random.shuffle(key, y_tr)
w_shuf = CriticTrain.ridge(with_bias.(xs_tr), y_shuf, l2)
pred_shuf = Nx.dot(with_bias.(xs_ev), Nx.backend_copy(w_shuf, Nx.default_backend()))
k3 = Nx.Random.fold_in(key, 3)
v_rank_ctrl = CriticTrain.pair_rank_acc(pred_shuf, y_ev, k3)
Output.puts("  V: held-out R^2 #{Float.round(v_r2, 3)}, pair-rank acc #{Float.round(v_rank, 3)} (shuffled-target control #{Float.round(v_rank_ctrl, 3)})")

# ---- S ----
Output.puts("Training selector S(s,a)...")
cands = fn idx -> Nx.concatenate([Nx.new_axis(take.(data.a_master, idx), 1), take.(data.a_samples, idx)], axis: 1) end
c_tr = cands.(tr_idx); c_ev = cands.(ev_idx)
kk = Nx.axis_size(data.a_samples, 1)
target_tr = Nx.concatenate([Nx.broadcast(1.0, {Nx.axis_size(tr_idx, 0), 1}), Nx.broadcast(0.0, {Nx.axis_size(tr_idx, 0), kk})], axis: 1)
k4 = Nx.Random.fold_in(key, 4)
{s_params, _} = CriticTrain.train_selector(xs_tr, c_tr, target_tr, steps: opts[:steps] || 2000, lr: opts[:lr] || 0.05, l2: l2, key: k4)

top1 = CriticTrain.top1_with_master(s_params, xs_ev, c_ev)
pk = CriticTrain.passk_metrics(s_params, xs_ev, take.(data.a_samples, ev_idx), take.(data.match, ev_idx))

# shuffled-label control: permute which candidate is "the master" per row
k5 = Nx.Random.fold_in(key, 5)
{perm_rows, _} = Nx.Random.shuffle(k5, Nx.iota({Nx.axis_size(tr_idx, 0)}))
c_ctrl = Nx.concatenate([Nx.new_axis(Nx.take(take.(data.a_master, tr_idx), perm_rows), 1), take.(data.a_samples, tr_idx)], axis: 1)
k6 = Nx.Random.fold_in(key, 6)
{ctrl_params, _} = CriticTrain.train_selector(xs_tr, c_ctrl, target_tr, steps: div(opts[:steps] || 2000, 2), lr: opts[:lr] || 0.05, l2: l2, key: k6)
pk_ctrl = CriticTrain.passk_metrics(ctrl_params, xs_ev, take.(data.a_samples, ev_idx), take.(data.match, ev_idx))

fmt = fn v -> :erlang.float_to_binary(v * 100, decimals: 1) end

table = """
| metric | value |
|---|---|
| V held-out R^2 | #{Float.round(v_r2, 3)} |
| V pair-rank acc (chance 0.5) | #{Float.round(v_rank, 3)} |
| V pair-rank acc, shuffled-target control | #{Float.round(v_rank_ctrl, 3)} |
| S top-1 among K+1 incl. master (chance #{fmt.(1 / (kk + 1))}%) | #{fmt.(top1)}% |
| **sampling pass@1** (random sample == master) | #{fmt.(pk.sampling)}% |
| **selector pass@1** (argmax-scored sample == master) | **#{fmt.(pk.selector)}%** |
| oracle pass@#{kk} (any sample == master) | #{fmt.(pk.oracle)}% |
| selector pass@1, shuffled-label control | #{fmt.(pk_ctrl.selector)}% |
"""

IO.puts("\n" <> table)

recovered = if pk.oracle > pk.sampling, do: (pk.selector - pk.sampling) / (pk.oracle - pk.sampling), else: 0.0

verdict =
  cond do
    pk.selector < pk.sampling + 0.02 -> "NULL: the selector does not beat random sampling on held-out replays."
    pk_ctrl.selector > pk.sampling + 0.02 -> "SUSPECT: the shuffled-label control also beats sampling — the gain is not from the (state, action) pairing."
    recovered >= 0.5 -> "STRONG: the selector recovers #{fmt.(recovered)}% of the sampling->oracle gap. Build Best-of-N into the live decode."
    true -> "PARTIAL: the selector recovers #{fmt.(recovered)}% of the gap. Real but linear-limited — try a small MLP head before wiring live."
  end

Output.warning(verdict)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  bin = fn t -> Nx.backend_copy(t, Nx.BinaryBackend) end

  File.write!(out, :erlang.term_to_binary(%{
    kind: :d2_critic_v1,
    mean: bin.(mean), std: bin.(std),
    v: %{w: bin.(w_v)},
    selector: %{w: bin.(s_params.w), v: bin.(s_params.v)},
    phi_size: Nx.axis_size(data.phi, 1),
    metrics: %{v_r2: v_r2, v_rank: v_rank, top1: top1, passk: pk, control: pk_ctrl},
    created: DateTime.utc_now() |> DateTime.to_iso8601()
  }))

  Output.success("saved #{out}")
end

if rep = opts[:report] do
  File.mkdir_p!(Path.dirname(rep))
  File.write!(rep, "# D2 critic training report\n\nData: #{Enum.join(paths, ", ")}\nRows train/eval #{Nx.axis_size(tr_idx, 0)}/#{Nx.axis_size(ev_idx, 0)}, replays #{n_rep - n_eval}/#{n_eval}, #{if(opts[:all_rows], do: "all rows", else: "decision rows")}, K=#{kk}, L2 #{l2}.\n\n#{table}\n\n**Verdict:** #{verdict}\n\nGap recovered: #{fmt.(recovered)}% of (oracle - sampling).\n")
  Output.success("wrote #{rep}")
end
