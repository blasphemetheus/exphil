# Interp Phase 1 v2: leak-proof probe validation.
#
# v1 saturated (targets were input-copyable; even the raw-input floor probed
# 0.98). v2 relabels the cached zoo captures with MEMORY-DEPENDENT targets —
# decodable only if the trunk integrates over its 60-frame window — and adds
# a shuffled-label control plus the lead-time decodability curve.
#
#   mix run scripts/interp_probe_zoo_v2.exs
#
# Reads/updates interp/zoo/*.capture.bin in place (labels only; activations
# untouched). NO-MIX: one beam.

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{GroundTruth, Probe}
alias ExPhil.Training.Output

slippi = Path.expand("~/Slippi")

replays =
  ~w(
    Game_20260713T015257.slp Game_20260713T015819.slp Game_20260713T020600.slp
    Game_20260713T034021.slp Game_20260713T034742.slp Game_20260713T035544.slp
    Game_20260713T044658.slp Game_20260713T045439.slp Game_20260713T050255.slp
    Game_20260713T070100.slp Game_20260713T070616.slp Game_20260713T071203.slp
  )
  |> Enum.map(&Path.join(slippi, &1))

eval_replays = [2, 7, 11]
window = 60

zoo = [
  {"poolgrow_r1", 27.9},
  {"215741_i1", 26.0},
  {"replicate", 25.0},
  {"lr15", 21.1},
  {"tl08", 12.1},
  {"poolgrow_r3", 11.1},
  {"035205_i2", 6.0},
  {"tl15", 0.0},
  {"035205_i1", 0.0},
  {"poolgrow_r2", 0.0},
  {"CTRL_random_init", nil},
  {"CTRL_input_floor", nil}
]

Output.banner("Interp Phase 1 v2: memory-target probes")

# ---------------------------------------------------------------------------
# Relabel once (labels are policy-independent; every capture used window 60)
# ---------------------------------------------------------------------------
Output.puts("Recomputing v2 labels for #{length(replays)} replays...")

labels_per_replay =
  Enum.map(replays, fn path ->
    {:ok, replay} = Peppi.parse(path)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    n = length(frames) - window + 1

    frames
    |> GroundTruth.frame_labels(opponent_port: 2)
    |> GroundTruth.align_to_windows(window, n)
  end)

merged_labels =
  Enum.reduce(labels_per_replay, fn m, acc ->
    Map.new(acc, fn {k, v} -> {k, Nx.concatenate([v, Map.fetch!(m, k)])} end)
  end)

total_rows = merged_labels |> Map.values() |> hd() |> Nx.axis_size(0)
Output.puts("v2 labels: #{total_rows} rows, #{map_size(merged_labels)} features")

# ---------------------------------------------------------------------------
# Probe every cached capture with the v2 suite
# ---------------------------------------------------------------------------
results =
  Enum.map(zoo, fn {tag, conv} ->
    cache = "interp/zoo/#{tag}.capture.bin"
    capture = cache |> File.read!() |> :erlang.binary_to_term()

    n_cap = Nx.axis_size(capture.activations, 0)

    if n_cap != total_rows do
      raise "row mismatch for #{tag}: capture #{n_cap} vs labels #{total_rows}"
    end

    capture = %{capture | labels: merged_labels}
    File.write!(cache, :erlang.term_to_binary(capture))

    split = Probe.split_by_replay(capture, eval_replays)
    suite = Probe.suite_v2(split)

    memory_ba =
      [:time_since_kd_bucket, :frames_until_kd_bucket, :opp_damaged_recent]
      |> Enum.map(&suite[&1].balanced_accuracy)
      |> Enum.reject(&is_nil/1)
      |> then(&(Enum.sum(&1) / max(length(&1), 1)))

    shuffled = Probe.shuffled_control(split, :time_since_kd_bucket, 5)
    curve = Probe.lead_time_curve(split)

    fmt = fn
      nil -> "  -  "
      v -> :io_lib.format("~5.3f", [v]) |> to_string()
    end

    curve_s =
      Enum.map_join(curve, " ", fn {b, ba, ne} -> "b#{b}=#{fmt.(ba)}(#{ne})" end)

    Output.puts(
      "#{String.pad_trailing(tag, 18)} conv=#{String.pad_leading(inspect(conv), 5)} " <>
        "memBA=#{fmt.(memory_ba)} since=#{fmt.(suite[:time_since_kd_bucket].balanced_accuracy)} " <>
        "until=#{fmt.(suite[:frames_until_kd_bucket].balanced_accuracy)} " <>
        "dmg30=#{fmt.(suite[:opp_damaged_recent].balanced_accuracy)} " <>
        "kd(leak-ref)=#{fmt.(suite[:opp_knockdown].balanced_accuracy)} " <>
        "shuf=#{fmt.(shuffled.balanced_accuracy)}"
    )

    Output.puts("  #{String.pad_trailing(tag, 16)} lead-time: #{curve_s}")

    %{tag: tag, conversion: conv, memory_ba: memory_ba, suite_keys: Map.keys(suite), curve: curve}
  end)

File.write!(
  "interp/zoo/probe_v2_results.bin",
  :erlang.term_to_binary(results)
)

scored = Enum.filter(results, & &1.conversion)

rank = fn values ->
  values
  |> Enum.with_index()
  |> Enum.sort_by(&elem(&1, 0))
  |> Enum.with_index()
  |> Map.new(fn {{_v, original_i}, r} -> {original_i, r} end)
end

xs = Enum.map(scored, & &1.memory_ba)
ys = Enum.map(scored, & &1.conversion)
rx = rank.(xs)
ry = rank.(ys)
n = length(scored)
d2 = Enum.sum(for i <- 0..(n - 1), do: :math.pow(rx[i] - ry[i], 2))
rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))

Output.puts("")
Output.puts("Spearman rho (memory-target BA vs conversion %): #{Float.round(rho, 3)} (n=#{n})")
Output.success("Saved: interp/zoo/probe_v2_results.bin")
