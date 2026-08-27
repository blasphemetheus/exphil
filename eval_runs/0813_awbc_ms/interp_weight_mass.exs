# AWBC interp follow-up (RESULTS.md next-step #4): WHERE did B2's weight
# mass land? Recomputes the AdvantageWeighting weights over the arms'
# pool (delay-0 lists — weight structure depends on action states, not
# label alignment; noted caveat) and cross-tabs them against:
#   * ShineChain action family per frame
#   * margin-critical frames (MarginSampling: the aerial-B press edges
#     where chains die — the 08-05 delay-break lever)
#   * time-to-next-shine buckets (the return-to-go gradient made visible)
#
#   mix run eval_runs/0813_awbc_ms/interp_weight_mass.exs [--sample N]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.AdvantageWeighting
alias ExPhil.Training.MarginSampling
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [sample: :integer])
sample = opts[:sample] || 120

fixture = ["test/fixtures/replays/fox_multishine_closed_d1.slp"]

rollouts =
  [
    "eval_runs/dagger_d3_round1_collect/r*.slp",
    "eval_runs/d3_div_b1/r*.slp",
    "eval_runs/d3_div_full2/r*.slp",
    "eval_runs/d3_div_r3/r*.slp",
    "eval_runs/0802_d2pool/r*.slp"
  ]
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.sort()
  |> Enum.take_every(max(1, div(length(Path.wildcard("eval_runs/dagger_d3_round1_collect/r*.slp")) * 5, sample)))
  |> Enum.take(sample)

files = fixture ++ rollouts
Output.banner("AWBC weight-mass interp")
Output.config([{"Files", length(files)}, {"(delay-0 lists — alignment caveat noted)", ""}])

frame_lists =
  files
  |> Task.async_stream(
    fn path ->
      case Peppi.parse(path) do
        {:ok, r} ->
          r
          |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
          |> Enum.reject(&(&1.game_state.frame < 0))

        _ ->
          []
      end
    end,
    max_concurrency: 8,
    timeout: 120_000,
    on_timeout: :kill_task
  )
  |> Enum.flat_map(fn
    {:ok, []} -> []
    {:ok, frames} -> [frames]
    _ -> []
  end)

Output.puts("#{length(frame_lists)} lists, #{Enum.sum(Enum.map(frame_lists, &length/1))} frames")

{weights, stats} = AdvantageWeighting.frame_weights(frame_lists)
Output.puts("beta #{Float.round(stats.beta, 4)}, ratio #{stats.weight_ratio && Float.round(stats.weight_ratio, 2)}, shine entries #{stats.shine_entries}")

{margin_flags, mstats} = MarginSampling.frame_weights(frame_lists, 2.0)
margin_flags = Enum.map(margin_flags, &(&1 != 1.0))
Output.puts("margin-critical frames: #{mstats.upweighted}/#{mstats.frames}")

families =
  Enum.flat_map(frame_lists, fn frames ->
    Enum.map(frames, fn f ->
      p = f.game_state.players[1]
      ShineChain.family(trunc((p && p.action) || 0))
    end)
  end)

# Frames-until-next-shine per frame (within list): the RTG gradient axis
next_shine_dist =
  Enum.flat_map(frame_lists, fn frames ->
    rewards = AdvantageWeighting.rewards(frames)

    rewards
    |> Enum.reverse()
    |> Enum.map_reduce(nil, fn r, dist ->
      cond do
        r > 0.0 -> {0, 0}
        dist == nil -> {nil, nil}
        true -> {dist + 1, dist + 1}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end)

rows = Enum.zip([weights, families, margin_flags, next_shine_dist])

mean = fn ws -> if ws == [], do: nil, else: Float.round(Enum.sum(ws) / length(ws), 3) end

Output.puts("")
Output.puts("== mean AWBC weight by action family")

for fam <- [:ground_reflect, :air_reflect, :jumpsquat, :aerial_jump, :other] do
  ws = for {w, f, _, _} <- rows, f == fam, do: w
  Output.puts("  #{fam}: #{mean.(ws)}  (n #{length(ws)})")
end

Output.puts("")
Output.puts("== margin-critical frames (the aerial-B chain decisions)")
crit = for {w, _, true, _} <- rows, do: w
rest = for {w, _, false, _} <- rows, do: w
Output.puts("  critical: #{mean.(crit)}  (n #{length(crit)})")
Output.puts("  everything else: #{mean.(rest)}")

Output.puts("")
Output.puts("== mean weight by frames-until-next-shine (the RTG gradient)")

for {label, lo, hi} <- [{"at shine", 0, 0}, {"1-30", 1, 30}, {"31-120", 31, 120}, {"121-300", 121, 300}] do
  ws = for {w, _, _, d} <- rows, d != nil, d >= lo, d <= hi, do: w
  Output.puts("  #{label}: #{mean.(ws)}  (n #{length(ws)})")
end

no_shine = for {w, _, _, nil} <- rows, do: w
Output.puts("  no shine ahead: #{mean.(no_shine)}  (n #{length(no_shine)})")

# Where does the TOP-DECILE weight mass sit?
sorted = Enum.sort_by(rows, fn {w, _, _, _} -> -w end)
top = Enum.take(sorted, div(length(rows), 10))
top_near_shine = Enum.count(top, fn {_, _, _, d} -> d != nil and d <= 120 end)
top_critical = Enum.count(top, fn {_, _, c, _} -> c end)

Output.puts("")
Output.puts("== top-decile weight frames (n #{length(top)})")
Output.puts("  within 120f of a shine: #{top_near_shine} (#{Float.round(100 * top_near_shine / max(length(top), 1), 1)}%)")
Output.puts("  margin-critical: #{top_critical}")
