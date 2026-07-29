# Does the d1-adapted trunk encode press intent EARLIER than the champion?
# (Mechanistic follow-up to the 2026-07-29 delay-preference inversion.)
#
# A pure representational timing shift is invisible to probe accuracy
# against ground-truth labels (a shifted bijection decodes equally well).
# What CAN differ is the ENCODE HORIZON: balanced accuracy of "an X-press
# edge occurs within <= k frames" probes as a function of lead k. A policy
# that must trigger one frame earlier (the d1 teacher's shifted windows)
# should hold press intent at longer leads.
#
# Design:
#   - Trunks: champion (ms_open_z) vs d1-DAgger (ms_d1_dagger3), plus the
#     RAW EMBEDDING input floor (Activations.input_trunk) as control — if
#     the floor shows the same shift, it lives in the data, not the trunk.
#   - Identical replay set for all three: champion@d0, champion@d1,
#     dagger@d1 games (2 each) — probes are only comparable on shared rows.
#   - Labels: X-press EDGES of port 1 from the recorded controller stream
#     (edge = pressed at t, not at t-1); label(k, t) = edge within (t, t+k].
#   - Split by replay (frames within a game are correlated): eval on one
#     replay per regime.
#
# Pre-registered: dagger trunk's accuracy-vs-k curve dominates the
# champion's at k >= 2 (earlier encoding); input floor shows no such gap.
#
# NO-MIX: one beam; never run beside a live training.
#
#   mix run scripts/interp_d1_timing.exs

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.Output

Output.banner("d1 timing: encode-horizon probes")

replay_set = [
  # {path, regime}
  {"eval_runs/stand_windowed/r1.slp", :champ_d0},
  {"eval_runs/stand_windowed/r2.slp", :champ_d0},
  {"eval_runs/dagger_d1_round0/r1.slp", :champ_d1},
  {"eval_runs/dagger_d1_round0/r2.slp", :champ_d1},
  {"eval_runs/dagger_d1_round2_eval/r1.slp", :dagger_d1},
  {"eval_runs/dagger_d1_round2_eval/r2.slp", :dagger_d1}
]

for {p, _} <- replay_set, not File.exists?(p), do: raise("missing replay: #{p}")

# Eval on one replay per regime (indices in replay_set)
eval_idx = [1, 3, 5]
max_lead = 4

# Frames-until-next-X-edge per frame of one replay (non-negative frames,
# same filter as capture_replay), as a list. :none when no edge follows.
until_next_edge = fn path ->
  {:ok, replay} = Peppi.parse(Path.expand(path))

  presses =
    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(& &1.controller.button_x)

  edges =
    presses
    |> Enum.zip([false | presses])
    |> Enum.map(fn {now, prev} -> now and not prev end)

  # Walk backwards: distance to the next edge strictly after t
  edges
  |> Enum.reverse()
  |> Enum.reduce({[], :none}, fn edge, {acc, dist} ->
    d =
      case dist do
        :none -> :none
        n -> n + 1
      end

    next = if edge, do: 0, else: d
    {[d | acc], next}
  end)
  |> elem(0)
end

trunks = [
  {:champion, fn -> Activations.load_trunk("checkpoints/ms_open_z.bin") end},
  {:dagger_d1, fn -> Activations.load_trunk("checkpoints/ms_d1_dagger3_policy.bin") end},
  # window/prev-action must match the policies' regime (GRU w16, --prev-action)
  {:input_floor, fn -> Activations.input_trunk(window: 16, use_prev_action: true) end}
]

results =
  for {name, load} <- trunks do
    trunk = load.()
    Output.puts("#{name}: capturing #{length(replay_set)} replays...")

    caps =
      Enum.map(replay_set, fn {path, _regime} ->
        cap = Activations.capture_replay(trunk, path, labels: false)
        leads = until_next_edge.(path)
        offset = cap.frame_offset

        # Row i of activations <-> frame i + offset
        aligned = leads |> Enum.drop(offset) |> Enum.take(cap.n)
        {cap.activations, aligned}
      end)

    {eval_caps, train_caps} =
      caps |> Enum.with_index() |> Enum.split_with(fn {_c, i} -> i in eval_idx end)

    cat = fn pairs ->
      x = pairs |> Enum.map(fn {{a, _l}, _i} -> a end) |> Nx.concatenate(axis: 0)
      leads = pairs |> Enum.flat_map(fn {{_a, l}, _i} -> l end)
      {x, leads}
    end

    {x_train, leads_train} = cat.(train_caps)
    {x_eval, leads_eval} = cat.(eval_caps)

    curve =
      for k <- 1..max_lead do
        to_y = fn leads ->
          leads
          |> Enum.map(fn
            :none -> 0
            d when d >= 1 and d <= k -> 1
            _ -> 0
          end)
          |> Nx.tensor(type: :s64)
        end

        r = Probe.fit_eval(x_train, to_y.(leads_train), x_eval, to_y.(leads_eval), 2)
        Output.puts("  k<=#{k}: balanced_acc=#{Float.round(r.balanced_accuracy, 4)} (n=#{r.n_eval})")
        {k, r.balanced_accuracy}
      end

    {name, curve}
  end

Output.puts("")
Output.puts("Encode-horizon curves (balanced accuracy of 'X edge within k frames'):")
header = ["trunk        " | Enum.map(1..max_lead, &"  k<=#{&1}")] |> Enum.join()
Output.puts(header)

for {name, curve} <- results do
  row =
    [String.pad_trailing(to_string(name), 13) | Enum.map(curve, fn {_k, a} -> " #{Float.round(a, 4)}" end)]
    |> Enum.join()

  Output.puts(row)
end

Output.puts("")
Output.puts("Prereg: dagger_d1 > champion at k>=2; input_floor shows no comparable gap.")
