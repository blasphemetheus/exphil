# W4 stage-representation audit, rung 1 (INTERP_ROADMAP_V2 — customer
# #33 stage blindness): can a linear probe read the CURRENT FoD
# platform heights out of the champion's trunk state?
#
# Platform heights are NOT network inputs (the embedding carries no
# stage-internal features), so any trunk decodability is ACCUMULATED
# inference from indirect cues (own/opponent y while platform-riding,
# collision outcomes). Controls, per the standing laws:
#   - INPUT probe (Activations.input_trunk): identity over the raw
#     current-frame embedding — heights aren't inputs, so this is the
#     "leakage floor" by construction; trunk >> input = real tracking.
#   - SHUFFLED-label floor (GOTCHA #79) for the trunk probe.
# Labels from replay stream events (peppi fod_platform_left/right;
# nil before the first move = the 20.0/28.0 start heights, the values
# the 0824 RAM probe read verbatim pre-movement).
#
# Usage:
#   mix run scripts/interp_w4_stage_height.exs \
#     [--policy checkpoints/ms_g19_ep4.bin] [--buckets 6] \
#     [--replays "eval_runs/0824_w4_fod_corpus/**/*.slp,eval_runs/0824_stage_merge_smoke/**/*.slp"]
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, buckets: :integer, replays: :string, delay_id: :integer]
  )

policy = opts[:policy] || "checkpoints/ms_g19_ep4.bin"
buckets = opts[:buckets] || 6
delay_id = opts[:delay_id] || 3

globs =
  (opts[:replays] ||
     "eval_runs/0824_w4_fod_corpus/**/*.slp,eval_runs/0824_stage_merge_smoke/**/*.slp")
  |> String.split(",")

replays = globs |> Enum.flat_map(&Path.wildcard/1) |> Enum.sort()

Output.banner("W4 stage-height probe")
Output.config([{"Policy", policy}, {"Replays", length(replays)}, {"Buckets", buckets}])

if length(replays) < 2, do: raise("need >=2 FoD replays (one is held out for eval)")

# Per-replay heights aligned to post-countdown frames
heights_for = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.map(fn f ->
    gs = f.game_state
    {gs.fod_platform_left || 20.0, gs.fod_platform_right || 28.0}
  end)
end

capture_all = fn trunk ->
  Enum.map(replays, fn path ->
    cap = Activations.capture_replay(trunk, path, labels: false, delay_id: delay_id)
    hs = heights_for.(path)

    aligned =
      Enum.map(0..(cap.n - 1), fn j -> Enum.at(hs, j + cap.frame_offset) end)

    {cap.activations, aligned, path}
  end)
end

bucketize = fn pairs_per_replay, side_idx ->
  all = pairs_per_replay |> Enum.flat_map(&elem(&1, 1)) |> Enum.map(&elem(&1, side_idx))
  {mn, mx} = Enum.min_max(all)
  span = max(mx - mn, 1.0e-6)

  to_bucket = fn h -> min(trunc((h - mn) / span * buckets), buckets - 1) end
  {to_bucket, mn, mx}
end

run_side = fn caps, side_idx, side_name ->
  {to_bucket, mn, mx} = bucketize.(caps, side_idx)

  {eval_cap, train_caps} = {List.last(caps), Enum.drop(caps, -1)}

  xs = fn cap_list ->
    cap_list |> Enum.map(&elem(&1, 0)) |> Nx.concatenate(axis: 0)
  end

  ys = fn cap_list ->
    cap_list
    |> Enum.flat_map(&elem(&1, 1))
    |> Enum.map(fn hs -> to_bucket.(elem(hs, side_idx)) end)
    |> Nx.tensor(type: :s64)
  end

  x_train = xs.(train_caps)
  y_train = ys.(train_caps)
  x_eval = xs.([eval_cap])
  y_eval = ys.([eval_cap])

  res = Probe.fit_eval(x_train, y_train, x_eval, y_eval, buckets)

  # Shuffled-label floor (same x, permuted y_train)
  perm = y_train |> Nx.shape() |> elem(0) |> (&Nx.iota({&1})).() |> Nx.take(Nx.argsort(Nx.Random.uniform(Nx.Random.key(42), shape: Nx.shape(y_train)) |> elem(0)))
  y_shuf = Nx.take(y_train, perm)
  floor = Probe.fit_eval(x_train, y_shuf, x_eval, y_eval, buckets)

  Output.puts(
    "  #{side_name}: bal_acc=#{Float.round(res.balanced_accuracy, 3)} " <>
      "(majority=#{Float.round(res.majority_baseline, 3)}, " <>
      "shuffle_floor=#{Float.round(floor.balanced_accuracy, 3)}) " <>
      "range=#{Float.round(mn, 1)}..#{Float.round(mx, 1)} " <>
      "n_train=#{res.n_train} n_eval=#{res.n_eval}"
  )

  res
end

Output.puts("")
Output.puts("== TRUNK probe (accumulated representation)")
trunk = Activations.load_trunk(policy)
trunk_caps = capture_all.(trunk)
trunk_left = run_side.(trunk_caps, 0, "left ")
trunk_right = run_side.(trunk_caps, 1, "right")

Output.puts("")
Output.puts("== INPUT probe (leakage floor: heights are not inputs)")
window = Map.get(trunk, :window, 60)
in_trunk = Activations.input_trunk(window: window, use_prev_action: Map.get(trunk.config, :use_prev_action, false))
in_caps = capture_all.(in_trunk)
in_left = run_side.(in_caps, 0, "left ")
in_right = run_side.(in_caps, 1, "right")

Output.puts("")

verdict = fn t, i, name ->
  cond do
    t.balanced_accuracy > i.balanced_accuracy + 0.1 and t.balanced_accuracy > 1.5 / buckets ->
      "#{name}: TRUNK TRACKS the height (#{Float.round(t.balanced_accuracy, 3)} vs input #{Float.round(i.balanced_accuracy, 3)})"

    t.balanced_accuracy <= i.balanced_accuracy + 0.05 ->
      "#{name}: STAGE-BLIND (trunk #{Float.round(t.balanced_accuracy, 3)} ~ input #{Float.round(i.balanced_accuracy, 3)})"

    true ->
      "#{name}: WEAK/AMBIGUOUS (trunk #{Float.round(t.balanced_accuracy, 3)} vs input #{Float.round(i.balanced_accuracy, 3)})"
  end
end

Output.puts(verdict.(trunk_left, in_left, "LEFT platform"))
Output.puts(verdict.(trunk_right, in_right, "RIGHT platform"))
