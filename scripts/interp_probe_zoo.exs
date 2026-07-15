# Interp Phase 1: THE validation experiment.
#
# Probes every checkpoint in the zoo on an identical fixed replay set and
# asks: does probe accuracy (is the tech situation linearly represented?)
# predict live conversion %? Controls: random-init trunk (same architecture,
# fresh params) and raw last-frame embedding (input floor).
#
#   mix run scripts/interp_probe_zoo.exs
#
# Decision gate (INTERP_ROADMAP P1): Spearman rho >= ~0.7 between mean
# balanced probe accuracy and conversion rank -> adopt probe score as the
# drill's checkpoint selector. Weak correlation -> policies fail on acting,
# not knowing -> redirect Phase 3.
#
# Captures are cached per policy in interp/zoo/ — delete to recompute.
# NO-MIX: one beam; never run beside a live training.

alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.Output

slippi = Path.expand("~/Slippi")

# Fixed replay set: all 12 clean 2026-07-13 probe games (mewtwo P1 vs
# tech_random P2, FD, real-time). Fixed across checkpoints — comparability
# demands identical data.
replays =
  ~w(
    Game_20260713T015257.slp Game_20260713T015819.slp Game_20260713T020600.slp
    Game_20260713T034021.slp Game_20260713T034742.slp Game_20260713T035544.slp
    Game_20260713T044658.slp Game_20260713T045439.slp Game_20260713T050255.slp
    Game_20260713T070100.slp Game_20260713T070616.slp Game_20260713T071203.slp
  )
  |> Enum.map(&Path.join(slippi, &1))

for p <- replays, not File.exists?(p), do: raise("missing replay: #{p}")

# Eval replays (positions in the list above): one from each round family
# with knockdowns. Train on the other 9.
eval_replays = [2, 7, 11]

# {tag, checkpoint path, pooled conversion % (nil = control, excluded from
# the correlation)}
zoo = [
  {"poolgrow_r1", "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin", 27.9},
  {"215741_i1", "checkpoints/mewtwo_combo_daggerloop_20260711_215741_i1_policy.bin", 26.0},
  {"replicate", "checkpoints/mewtwo_combo_replicate215741_policy.bin", 25.0},
  {"lr15", "checkpoints/mewtwo_combo_lr15_policy.bin", 21.1},
  {"tl08", "checkpoints/mewtwo_combo_stopab_tl08_policy.bin", 12.1},
  {"poolgrow_r3", "checkpoints/mewtwo_combo_poolgrow_r3_policy.bin", 11.1},
  {"035205_i2", "checkpoints/mewtwo_combo_daggerloop_20260712_035205_i2_policy.bin", 6.0},
  {"tl15", "checkpoints/mewtwo_combo_stopab_tl15_policy.bin", 0.0},
  {"035205_i1", "checkpoints/mewtwo_combo_daggerloop_20260712_035205_i1_policy.bin", 0.0},
  {"poolgrow_r2", "checkpoints/mewtwo_combo_poolgrow_r2_policy.bin", 0.0},
  {"CTRL_random_init", :random, nil},
  {"CTRL_input_floor", :input, nil}
]

File.mkdir_p!("interp/zoo")

cached_capture = fn tag, kind ->
  cache = "interp/zoo/#{tag}.capture.bin"

  if File.exists?(cache) do
    Output.puts("#{tag}: cached")
    cache |> File.read!() |> :erlang.binary_to_term()
  else
    trunk =
      case kind do
        :random ->
          Activations.load_trunk("checkpoints/mewtwo_combo_poolgrow_r1_policy.bin", init: :random)

        :input ->
          Activations.input_trunk(window: 60, embed_size: 288, use_prev_action: true)

        path ->
          Activations.load_trunk(path)
      end

    Output.puts("#{tag}: capturing #{length(replays)} replays...")
    capture = Activations.capture(trunk, replays)
    File.write!(cache, :erlang.term_to_binary(capture))
    capture
  end
end

Output.banner("Interp Phase 1: probe suite across the checkpoint zoo")

results =
  Enum.map(zoo, fn {tag, kind, conv} ->
    capture = cached_capture.(tag, kind)
    split = Probe.split_by_replay(capture, eval_replays)
    suite = Probe.suite(split)
    mba = Probe.mean_balanced_accuracy(suite)

    ba = fn f -> suite[f].balanced_accuracy end
    fmt = fn
      nil -> "  -  "
      v -> :io_lib.format("~5.3f", [v]) |> to_string()
    end

    Output.puts(
      "#{String.pad_trailing(tag, 18)} conv=#{String.pad_leading(inspect(conv), 5)} " <>
        "meanBA=#{fmt.(mba)} kd=#{fmt.(ba.(:opp_knockdown))} " <>
        "tech=#{fmt.(ba.(:tech_choice))} nextkd=#{fmt.(ba.(:next_kd_choice))} " <>
        "hitstun=#{fmt.(ba.(:opp_hitstun))}"
    )

    %{tag: tag, conversion: conv, mean_ba: mba, suite: suite}
  end)

File.write!(
  "interp/zoo/probe_results.bin",
  :erlang.term_to_binary(
    Enum.map(results, fn r ->
      %{r | suite: Map.new(r.suite, fn {k, v} -> {k, Map.delete(v, :params)} end)}
    end)
  )
)

# Spearman rank correlation between mean balanced accuracy and conversion %
scored = Enum.filter(results, & &1.conversion)

rank = fn values ->
  values
  |> Enum.with_index()
  |> Enum.sort_by(&elem(&1, 0))
  |> Enum.with_index()
  |> Map.new(fn {{_v, original_i}, r} -> {original_i, r} end)
end

xs = Enum.map(scored, & &1.mean_ba)
ys = Enum.map(scored, & &1.conversion)
rx = rank.(xs)
ry = rank.(ys)
n = length(scored)

d2 = Enum.sum(for i <- 0..(n - 1), do: :math.pow(rx[i] - ry[i], 2))
rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))

Output.puts("")
Output.puts("Spearman rho (mean balanced accuracy vs conversion %): #{Float.round(rho, 3)} (n=#{n})")

Output.puts(
  case rho do
    r when r >= 0.7 -> "VERDICT: representations predict behavior — adopt probe-based checkpoint selection."
    r when r >= 0.4 -> "VERDICT: partial signal — probes informative but not sufficient as sole selector."
    _ -> "VERDICT: weak — policies differ in ACTING, not knowing. Redirect to Phase 3 (attribution)."
  end
)

Output.success("Full results: interp/zoo/probe_results.bin")
