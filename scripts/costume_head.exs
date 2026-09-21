# COACH_STYLE_PRODUCTS.md S4 — the costume head.
#
# Predict a Fox player's costume (0 Nr, 1 Or, 2 La, 3 Gr) from HOW they play:
# a multinomial logistic regression on the z-scored style fingerprint (the
# same 30+ habits the identity pipeline uses). Then the demo: fingerprint a
# policy's own sim games and let the head pick the costume its play predicts
# — a preference that is a product of training, not a config flag.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/costume_head.exs \
#     --train eval_runs/0917_style_identity/erickfm_fox.jsonl,eval_runs/0917_style_identity/yeti_fox.jsonl \
#     --out eval_runs/0921_costume_head \
#     --pick "ep3=eval_runs/0921_sim_r1/anon_self_n10/sim_fingerprints.jsonl,mix2=eval_runs/0921_step8/fp_mix2/sim_fingerprints.jsonl"

alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [train: :string, out: :string, pick: :string, steps: :integer, lr: :float, seed: :integer])
out = opts[:out] || "eval_runs/0921_costume_head"
File.mkdir_p!(out)
steps = opts[:steps] || 600
lr = opts[:lr] || 0.05
seed = opts[:seed] || 1
names = ["Nr (neutral)", "Or (red)", "La (blue)", "Gr (green)"]

Output.banner("Costume head (S4)")

load = fn path ->
  path
  |> File.stream!()
  |> Stream.map(&Jason.decode!/1)
  |> Enum.filter(fn r -> is_integer(r["costume"]) and r["costume"] in 0..3 and is_map(r["features"]) end)
  |> Enum.map(fn r -> %{path: r["path"], port: r["port"], costume: r["costume"], features: r["features"], tag: r["tag"]} end)
end

rows = (opts[:train] || raise("--train required")) |> String.split(",", trim: true) |> Enum.flat_map(load)
Output.puts("rows: #{length(rows)}; costume histogram: #{inspect(Enum.frequencies_by(rows, & &1.costume) |> Enum.sort())}")

# Feature keys present in >= 99 % of rows, numeric, finite.
counts = rows |> Enum.flat_map(fn r -> for {k, v} <- r.features, is_number(v), do: k end) |> Enum.frequencies()
keys = counts |> Enum.filter(fn {_k, c} -> c >= 0.99 * length(rows) end) |> Enum.map(&elem(&1, 0)) |> Enum.sort()
Output.puts("features: #{length(keys)}")

vec = fn r -> Enum.map(keys, fn k -> v = Map.get(r.features, k); if is_number(v) and v == v and abs(v) < 1.0e9, do: v * 1.0, else: 0.0 end) end

# Game-level split by path hash (80/20), deterministic.
{train, test} = Enum.split_with(rows, fn r -> rem(:erlang.phash2({r.path, seed}), 5) != 0 end)
Output.puts("train #{length(train)}  test #{length(test)}")

xtr = Nx.tensor(Enum.map(train, vec), type: :f32)
mean = Nx.mean(xtr, axes: [0])
std = xtr |> Nx.standard_deviation(axes: [0]) |> Nx.add(1.0e-6)
z = fn x -> x |> Nx.subtract(mean) |> Nx.divide(std) |> Nx.clip(-6.0, 6.0) end
xtr = z.(xtr)
ytr = Nx.tensor(Enum.map(train, & &1.costume), type: :s64)
xte = z.(Nx.tensor(Enum.map(test, vec), type: :f32))
yte = Nx.tensor(Enum.map(test, & &1.costume), type: :s64)

d = length(keys)
key = Nx.Random.key(seed)
{w, key} = Nx.Random.normal(key, 0.0, 0.01, shape: {d, 4}, type: :f32)
b = Nx.broadcast(0.0, {4})
_ = key

loss_fn = fn {w, b}, x, y ->
  logits = x |> Nx.dot(w) |> Nx.add(b)
  logp = logits |> Nx.subtract(Nx.reduce_max(logits, axes: [1], keep_axes: true)) |> then(&Nx.subtract(&1, Nx.log(Nx.sum(Nx.exp(&1), axes: [1], keep_axes: true))))
  onehot = Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, 4}))
  ce = onehot |> Nx.multiply(logp) |> Nx.sum(axes: [1]) |> Nx.mean() |> Nx.negate()
  Nx.add(ce, Nx.multiply(1.0e-3, Nx.sum(Nx.multiply(w, w))))
end

grad_fn = Nx.Defn.jit(fn params, x, y -> Nx.Defn.value_and_grad(params, &loss_fn.(&1, x, y)) end)

{params, _} =
  Enum.reduce(1..steps, {{w, b}, {Nx.broadcast(0.0, {d, 4}), Nx.broadcast(0.0, {4})}}, fn step, {{w, b}, {mw, mb}} ->
    {loss, {gw, gb}} = grad_fn.({w, b}, xtr, ytr)
    # heavy-ball momentum
    mw = Nx.add(Nx.multiply(0.9, mw), gw)
    mb = Nx.add(Nx.multiply(0.9, mb), gb)
    w = Nx.subtract(w, Nx.multiply(lr, mw))
    b = Nx.subtract(b, Nx.multiply(lr, mb))
    if rem(step, 100) == 0, do: Output.puts("  step #{step} loss #{Float.round(Nx.to_number(loss), 4)}")
    {{w, b}, {mw, mb}}
  end)

{w, b} = params
predict = fn x -> x |> Nx.dot(w) |> Nx.add(b) end
probs = fn x -> l = predict.(x); e = Nx.exp(Nx.subtract(l, Nx.reduce_max(l, axes: [1], keep_axes: true))); Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true)) end
acc = fn x, y -> predict.(x) |> Nx.argmax(axis: 1) |> Nx.equal(y) |> Nx.mean() |> Nx.to_number() end

majority = test |> Enum.frequencies_by(& &1.costume) |> Enum.max_by(&elem(&1, 1)) |> elem(1) |> Kernel./(length(test))
Output.puts("train acc #{Float.round(acc.(xtr, ytr), 3)}  TEST acc #{Float.round(acc.(xte, yte), 3)}  (majority-class baseline #{Float.round(majority, 3)}, chance 0.25)")

pred_te = predict.(xte) |> Nx.argmax(axis: 1) |> Nx.to_flat_list()
truth = Nx.to_flat_list(yte)
conf = Enum.zip(truth, pred_te) |> Enum.frequencies()
Output.puts("confusion (rows = true costume, cols = predicted):")
for t <- 0..3, do: Output.puts("  #{String.pad_trailing(Enum.at(names, t), 13)} " <> Enum.map_join(0..3, " ", fn p -> String.pad_leading(to_string(Map.get(conf, {t, p}, 0)), 6) end))

# most informative habits per costume (largest |weight| per class)
wl = Nx.to_batched(Nx.transpose(w), 1) |> Enum.map(&Nx.to_flat_list(Nx.squeeze(&1)))
for {c, ws} <- Enum.with_index(wl) |> Enum.map(fn {ws, c} -> {c, ws} end) do
  top = Enum.zip(keys, ws) |> Enum.sort_by(fn {_, v} -> -abs(v) end) |> Enum.take(4) |> Enum.map(fn {k, v} -> "#{k} #{if v > 0, do: "+", else: "-"}#{Float.round(abs(v), 2)}" end)
  Output.puts("  #{Enum.at(names, c)}: #{Enum.join(top, ", ")}")
end

File.write!(Path.join(out, "head.bin"), :erlang.term_to_binary(%{keys: keys, mean: Nx.to_flat_list(mean), std: Nx.to_flat_list(std), w: Nx.to_flat_list(w), b: Nx.to_flat_list(b), names: names, test_acc: acc.(xte, yte), majority: majority}))
Output.success("wrote #{out}/head.bin")

# Demo: which costume does each policy's own play predict?
if pick = opts[:pick] do
  Output.puts("\nThe model picks its costume (mean class probability over its own sim games; votes = per-game argmax):")

  for spec <- String.split(pick, ",", trim: true) do
    [label, path] = String.split(spec, "=", parts: 2)
    games = path |> load.() |> Enum.filter(&(&1.port == 1))
    games = if games == [], do: path |> File.stream!() |> Stream.map(&Jason.decode!/1) |> Enum.filter(&(&1["port"] == 1)) |> Enum.map(fn r -> %{features: r["features"]} end), else: games

    if games == [] do
      Output.puts("  #{label}: no rows")
    else
      x = z.(Nx.tensor(Enum.map(games, vec), type: :f32))
      p = probs.(x)
      meanp = p |> Nx.mean(axes: [0]) |> Nx.to_flat_list()
      votes = p |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> Enum.frequencies()
      pick_i = meanp |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
      Output.puts("  #{String.pad_trailing(label, 8)} picks #{Enum.at(names, pick_i)}  p=#{Enum.map_join(meanp, " ", &Float.round(&1, 2))}  votes #{inspect(Enum.sort(votes))}  (#{length(games)} games)")
    end
  end
end
