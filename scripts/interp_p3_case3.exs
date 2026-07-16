# P3 case #3 (task #22): opponent-position attribution + compression.
#
#   mix run scripts/interp_p3_case3.exs
#
# The fair-in-place-while-Fox-stands-behind pathology, mechanized:
#   1. ATTRIBUTION — gradient×input saliency of the main-stick head,
#      aggregated into empirically-discovered dim groups. Question: what
#      fraction of the movement decision is driven by OPPONENT position/
#      facing dims, and does it collapse when the opponent is BEHIND?
#   2. COMPRESSION — probe opp_behind from the trunk vs the raw input
#      floor. Trunk << floor = the representation discards opponent side
#      (the P1 compression finding, sharpened to the relevant feature).
# Run on an old-era and a new-era checkpoint for the comparison.

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Attribution, GroundTruth, Probe}
alias ExPhil.Training.{Data, Output}

slippi = Path.expand("~/Slippi")

replays =
  ["Game_20260714T115716.slp", "Game_20260714T234927.slp"]
  |> Enum.map(&Path.join(slippi, &1))
  |> Enum.filter(&File.exists?/1)

policies = [
  {"poolgrow_r1 (old era)", "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin"},
  {"newera_r7 (sighted, sparse)", "checkpoints/mewtwo_combo_newera_r7_policy.bin"},
  {"newera_r8 (sighted + turnaround fixture)", "checkpoints/mewtwo_combo_newera_r8_policy.bin"},
  {"newera_r9 (repaired teacher: turn_toward + scrub)", "checkpoints/mewtwo_combo_newera_r9_policy.bin"},
  {"newera_r10 (DAgger on r9 probes)", "checkpoints/mewtwo_combo_newera_r10_policy.bin"}
]
# Missing checkpoints are skipped (pre-run safety), so this stays runnable
# before r9 lands.

max_rows = 3000
batch = 128

Output.banner("P3 case #3: opponent-position attribution + compression")

dim_groups = Attribution.discover_dims(ExPhil.Embeddings.config(), use_prev_action: true)

Output.puts(
  "dim groups: " <>
    Enum.map_join(dim_groups, " ", fn {k, v} -> "#{k}=#{length(v)}d" end)
)

for {tag, policy} <- policies, File.exists?(policy) do
  Output.puts("")
  Output.puts("=== #{tag} ===")
  heads = Activations.load_heads(policy)
  window = heads.config[:window_size] || 60

  # Collect states batches + aligned opp_behind labels, replay by replay
  {sal_parts, behind_parts} =
    replays
    |> Enum.map(fn path ->
      {:ok, replay} = Peppi.parse(path)

      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      n_rows = length(frames) - window + 1

      labels = frames |> GroundTruth.frame_labels(opponent_port: 2) |> GroundTruth.align_to_windows(window, n_rows)

      dataset =
        frames
        |> Data.from_frames()
        |> Data.precompute_frame_embeddings_cached(
          cache: true,
          replay_files: [path],
          use_prev_action: true,
          prev_action_dropout: 0.0,
          show_progress: false
        )

      rows_wanted = min(n_rows, div(max_rows, length(replays)))

      sals =
        dataset
        |> Data.batched_sequences(
          batch_size: batch,
          window_size: window,
          stride: 1,
          lazy: true,
          shuffle: false,
          drop_last: false
        )
        |> Enum.take(div(rows_wanted, batch))
        |> Enum.map(fn b ->
          Attribution.saliency(heads.predict_fn, heads.params, b.states, :main_x)
          |> Nx.backend_transfer(Nx.BinaryBackend)
        end)

      n_taken = Enum.sum(Enum.map(sals, &Nx.axis_size(&1, 0)))
      behind = labels.opp_behind |> Nx.slice_along_axis(0, n_taken, axis: 0)
      {Nx.concatenate(sals, axis: 0), behind}
    end)
    |> Enum.unzip()

  sal = Nx.concatenate(sal_parts, axis: 0)
  behind = Nx.concatenate(behind_parts, axis: 0) |> Nx.as_type(:f32)
  n = Nx.axis_size(sal, 0)
  n_behind = Nx.sum(behind) |> Nx.to_number() |> trunc()

  shares = Attribution.group_shares(sal, dim_groups)

  cmean = fn t, mask ->
    Nx.sum(Nx.multiply(t, mask)) |> Nx.divide(Nx.max(Nx.sum(mask), 1)) |> Nx.to_number() |> Float.round(4)
  end

  front = Nx.subtract(1.0, behind)
  Output.puts("rows=#{n} behind=#{n_behind} | main-stick saliency shares (front / behind):")

  for g <- [:opp_position, :opp_facing, :opp_action, :own_position, :own_facing, :own_action, :prev_action] do
    s = shares[g]
    Output.puts("  #{g}: #{cmean.(s, front)} / #{cmean.(s, behind)}")
  end

  # Compression: probe opp_behind from trunk vs input floor
  trunk = Activations.load_trunk(policy)
  floor = Activations.input_trunk(window: window, embed_size: heads.config[:embed_size] || 288)

  probe_ba = fn tr ->
    cap = Activations.capture(tr, replays, labels: true)
    split = Probe.split_by_replay(cap, [1])
    y_train = split.labels_train.opp_behind |> Nx.as_type(:s64)
    y_eval = split.labels_eval.opp_behind |> Nx.as_type(:s64)
    r = Probe.fit_eval(split.x_train, y_train, split.x_eval, y_eval, 2)
    r.balanced_accuracy
  end

  ba_trunk = probe_ba.(trunk)
  ba_floor = probe_ba.(floor)

  Output.puts(
    "opp_behind decodability: trunk=#{inspect(ba_trunk)} input-floor=#{inspect(ba_floor)} " <>
      "(trunk << floor => representation discards opponent side)"
  )
end

Output.puts("")
Output.puts("READINGS: opp_position share ~0 => movement head is opponent-blind.")
Output.puts("Share drops when behind => facing-gated attention to the opponent.")
Output.puts("Trunk decodability << floor => compression is upstream of the heads.")
