# W4 addendum 2 (PS): can a linear probe read the CURRENT Pokemon
# Stadium transformation out of the champion's trunk state?
#
# Unlike FoD heights, the transformation moves the whole collision
# landscape (windmill, mountain, tree...), and NONE of it is a network
# input — decodability means the trunk infers the layout from indirect
# cues (own/opponent positions, collision outcomes on transformed
# terrain). Controls as in the FoD probe: INPUT-embedding probe
# (leakage floor) + shuffled-label floor (GOTCHA #79).
# Labels: stream stadium_type (3 fire / 4 grass / 5 normal / 6 rock /
# 9 water; nil before the first event = normal), remapped to 0..4.
#
# Usage:
#   mix run scripts/interp_w4_ps_transform.exs \
#     [--policy checkpoints/ms_g19_ep4.bin] [--delay-id 3] \
#     [--replays "eval_runs/0824_w4_ps_corpus/**/*.slp"]
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, delay_id: :integer]
  )

policy = opts[:policy] || "checkpoints/ms_g19_ep4.bin"
delay_id = opts[:delay_id] || 3
globs = (opts[:replays] || "eval_runs/0824_w4_ps_corpus/**/*.slp") |> String.split(",")
replays = globs |> Enum.flat_map(&Path.wildcard/1) |> Enum.sort()

# Slippi type value -> class index
type_class = %{3 => 1, 4 => 2, 5 => 0, 6 => 3, 9 => 4}
class_names = ["normal", "fire", "grass", "rock", "water"]
num_classes = 5

Output.banner("W4 PS-transform probe")
Output.config([{"Policy", policy}, {"Replays", length(replays)}])
if length(replays) < 2, do: raise("need >=2 PS replays (one held out for eval)")

# stadium_type is set only ON event frames (7-14 per game) — the
# stream announces each layout change (including the type-5 revert to
# normal). FORWARD-FILL the announced type: that's the active-layout
# tracker, the same semantics the RAM transform digit carries.
labels_for = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.map_reduce(5, fn f, current ->
    current = f.game_state.stadium_type || current
    {Map.get(type_class, current, 0), current}
  end)
  |> elem(0)
end

capture_all = fn trunk ->
  Enum.map(replays, fn path ->
    cap = Activations.capture_replay(trunk, path, labels: false, delay_id: delay_id)
    ls = labels_for.(path)
    aligned = Enum.map(0..(cap.n - 1), fn j -> Enum.at(ls, j + cap.frame_offset) end)
    {cap.activations, aligned, path}
  end)
end

run_probe = fn caps, tag ->
  {eval_cap, train_caps} = {List.last(caps), Enum.drop(caps, -1)}
  xs = fn cl -> cl |> Enum.map(&elem(&1, 0)) |> Nx.concatenate(axis: 0) end
  ys = fn cl -> cl |> Enum.flat_map(&elem(&1, 1)) |> Nx.tensor(type: :s64) end

  x_train = xs.(train_caps)
  y_train = ys.(train_caps)
  x_eval = xs.([eval_cap])
  y_eval = ys.([eval_cap])

  res = Probe.fit_eval(x_train, y_train, x_eval, y_eval, num_classes)

  perm =
    Nx.argsort(Nx.Random.uniform(Nx.Random.key(42), shape: Nx.shape(y_train)) |> elem(0))

  floor = Probe.fit_eval(x_train, Nx.take(y_train, perm), x_eval, y_eval, num_classes)

  Output.puts(
    "  #{tag}: bal_acc=#{Float.round(res.balanced_accuracy, 3)} " <>
      "(majority=#{Float.round(res.majority_baseline, 3)}, " <>
      "shuffle_floor=#{Float.round(floor.balanced_accuracy, 3)}) " <>
      "n_train=#{res.n_train} n_eval=#{res.n_eval} " <>
      "per_class=#{inspect(Enum.zip(class_names, Enum.map(res.per_class_recall, fn nil -> :absent; v -> Float.round(v * 1.0, 2) end)))}"
  )

  res
end

# Report label mix per replay (transform coverage sanity)
for path <- replays do
  mix = labels_for.(path) |> Enum.frequencies() |> Enum.sort() |> Enum.map(fn {c, n} -> {Enum.at(class_names, c), n} end)
  Output.puts("  #{Path.basename(path)}: #{inspect(mix)}")
end

Output.puts("")
Output.puts("== TRUNK probe")
trunk = Activations.load_trunk(policy)
t = run_probe.(capture_all.(trunk), "trunk")

Output.puts("")
Output.puts("== INPUT probe (leakage floor)")
window = Map.get(trunk, :window, 60)
in_trunk = Activations.input_trunk(window: window, use_prev_action: Map.get(trunk.config, :use_prev_action, false))
i = run_probe.(capture_all.(in_trunk), "input")

Output.puts("")

Output.puts(
  cond do
    t.balanced_accuracy > i.balanced_accuracy + 0.1 and t.balanced_accuracy > 0.3 ->
      "VERDICT: TRUNK TRACKS the transformation (#{Float.round(t.balanced_accuracy, 3)} vs input #{Float.round(i.balanced_accuracy, 3)})"

    t.balanced_accuracy <= i.balanced_accuracy + 0.05 ->
      "VERDICT: STAGE-BLIND (trunk #{Float.round(t.balanced_accuracy, 3)} ~ input #{Float.round(i.balanced_accuracy, 3)})"

    true ->
      "VERDICT: WEAK/AMBIGUOUS (trunk #{Float.round(t.balanced_accuracy, 3)} vs input #{Float.round(i.balanced_accuracy, 3)})"
  end
)
