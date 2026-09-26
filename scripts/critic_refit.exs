# Refit the R2 critic from SAVED rollouts (no sim, no policy) and sweep the
# regularization that the first full run lacked.
#
#   devenv shell -- mix run scripts/critic_refit.exs --data eval_runs/0923_r2/v1 --out eval_runs/0923_r2/refit
#
# The 09-23 run reached train EV 0.913 / held-out 0.037 with a peak of 0.242 at
# epoch 1: the fit memorizes correlated frames. This sweeps stride (frames are
# 60 Hz, so neighbours are near-duplicates), width, weight decay and dropout,
# selects the epoch on a VALIDATION round and reports EV on a TEST round that
# selection never saw. Writes OUT/refit.json + OUT/critic_best.bin.

alias ExPhil.Sim.Critic
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [data: :string, out: :string, epochs: :integer, batch: :integer])
dir = opts[:data] || raise("--data required (a dir of data_round*.bin)")
out = opts[:out] || raise("--out required")
epochs = opts[:epochs] || 12
File.mkdir_p!(out)

Output.banner("R2 critic refit (saved rollouts)")

rounds =
  Path.wildcard(Path.join(dir, "data_round*.bin"))
  |> Enum.sort_by(&(&1 |> Path.basename() |> String.replace(~r/\D/, "") |> String.to_integer()))
  |> Enum.map(fn p ->
    m = p |> File.read!() |> Nx.deserialize()
    %{features: m.features, returns: m.returns}
  end)

# Carry the collection's character/policy forward, so a critic can never be
# reused on a different character's features without the mismatch being visible.
provenance =
  case File.read(Path.join(dir, "critic.bin")) do
    {:ok, bin} -> bin |> :erlang.binary_to_term() |> Map.take([:character, :stage, :policy, :gamma])
    _ -> %{}
  end

n_rounds = length(rounds)
if n_rounds < 3, do: raise("need >= 3 rounds for train/val/test, found #{n_rounds}")
d = Nx.axis_size(hd(rounds).features, 2)

# rounds are independent games: train on all but the last two, validate on the
# second-to-last, test on the last. Selection never touches the test round.
flat = fn rs, stride ->
  xs = Enum.map(rs, fn r -> Nx.reshape(r.features, {:auto, d}) end) |> Nx.concatenate()
  ys = Enum.map(rs, fn r -> Nx.reshape(r.returns, {:auto}) end) |> Nx.concatenate()
  if stride > 1 do
    idx = Nx.tensor(Enum.take_every(0..(Nx.axis_size(xs, 0) - 1), stride))
    {Nx.take(xs, idx), Nx.take(ys, idx)}
  else
    {xs, ys}
  end
end

{train_rs, [val_r, test_r]} = Enum.split(rounds, n_rounds - 2)
{val_x, val_y} = flat.([val_r], 1)
{test_x, test_y} = flat.([test_r], 1)

Output.config([
  {"Rounds", "#{n_rounds} (train #{length(train_rs)}, val 1, test 1)"},
  {"Feature dim", d},
  {"Val states", Nx.axis_size(val_x, 0)},
  {"Test states", Nx.axis_size(test_x, 0)},
  {"Return sd (test)", Float.round(Nx.standard_deviation(test_y) |> Nx.to_number(), 4)},
  {"Epochs/config", epochs}
])

grid =
  for stride <- [1, 6],
      hidden <- [64, 256],
      {wd, dropout} <- [{0.0, 0.0}, {1.0e-3, 0.0}, {1.0e-3, 0.2}] do
    %{stride: stride, hidden: hidden, weight_decay: wd, dropout: dropout}
  end

Output.puts("#{length(grid)} configs\n")

results =
  Enum.map(grid, fn c ->
    {tx, ty} = flat.(train_rs, c.stride)
    t0 = System.monotonic_time(:millisecond)

    fit =
      Critic.fit({tx, ty}, {val_x, val_y}, {test_x, test_y},
        epochs: epochs,
        batch: opts[:batch] || 1024,
        hidden: c.hidden,
        weight_decay: c.weight_decay,
        dropout: c.dropout
      )

    ms = System.monotonic_time(:millisecond) - t0

    Output.puts(
      "stride #{c.stride} hidden #{String.pad_leading(Integer.to_string(c.hidden), 3)} wd #{c.weight_decay} drop #{c.dropout} " <>
        "→ TEST EV #{:io_lib.format("~6.3f", [fit.ev])}  (val #{:io_lib.format("~6.3f", [fit.ev_val])} @epoch #{fit.best_epoch}, train #{:io_lib.format("~6.3f", [fit.ev_train])})  " <>
        "#{Nx.axis_size(tx, 0)} states, #{Float.round(ms / 1000, 1)} s"
    )

    Map.merge(c, %{ev: fit.ev, ev_val: fit.ev_val, ev_train: fit.ev_train, best_epoch: fit.best_epoch, states: Nx.axis_size(tx, 0), ms: ms, fit: fit})
  end)

# pick on validation, report its test EV (selecting on test would be the same
# leak the first run had between epochs)
best = Enum.max_by(results, & &1.ev_val)
verdict = if best.ev > 0.3, do: "R2 PASSED (test EV > 0.3)", else: "R2 NOT PASSED (test EV ≤ 0.3)"

Output.puts("")
Output.puts("best by validation: stride #{best.stride}, hidden #{best.hidden}, wd #{best.weight_decay}, dropout #{best.dropout} → test EV #{Float.round(best.ev, 3)}")
Output.puts(verdict)

File.write!(Path.join(out, "critic_best.bin"), :erlang.term_to_binary(%{
  params: ExPhil.Training.PPO.to_binary_backend(best.fit.params),
  d: d, hidden: best.hidden, dropout: best.dropout, stride: best.stride, ev: best.ev, source: dir
} |> Map.merge(provenance)))

File.write!(Path.join(out, "refit.json"), Jason.encode!(%{
  data: dir, feature_dim: d, epochs: epochs, verdict: verdict,
  best: Map.drop(best, [:fit]),
  results: Enum.map(results, &Map.drop(&1, [:fit]))
}, pretty: true))

Output.success("#{verdict} → #{out}/refit.json")
