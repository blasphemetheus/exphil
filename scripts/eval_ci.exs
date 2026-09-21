# EVALS_PROGRAM.md item 2 — bootstrap intervals on the existing gates.
#
# Reads the per-start rollouts of drill runs (`rollouts.jsonl`), the
# per-start results of oracle runs (`results.jsonl`) and fingerprint rows
# (`sim_fingerprints.jsonl`, port 1), and prints each rate / mean with a 95 %
# bootstrap interval (resampling starts or games, B = 2000). Paired
# comparisons between two runs on the SAME pool use the per-start
# differences, which is where the iteration decisions actually live.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/eval_ci.exs \
#     --drill "ep3=eval_runs/0921_step8/drill_ep3_self,mix2=eval_runs/0921_step8/drill_mix2_self,mix4=eval_runs/0921_step8/drill_mix4_self" \
#     --oracle "mix2=eval_runs/0921_step8/oracle_mix2_t10" \
#     --fp "dolphin=checkpoints/.../style_probe/bot_fingerprints.jsonl,mix4=eval_runs/0921_step8/fp_mix4/sim_fingerprints.jsonl"

alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [drill: :string, oracle: :string, fp: :string, b: :integer, seed: :integer])
bnum = opts[:b] || 2000
:rand.seed(:exsss, {opts[:seed] || 1, 2, 3})

specs = fn s -> (s || "") |> String.split(",", trim: true) |> Enum.map(fn kv -> [k, v] = String.split(kv, "=", parts: 2); {k, v} end) end
jsonl = fn path -> path |> File.stream!() |> Stream.map(&Jason.decode!/1) |> Enum.to_list() end

boot = fn xs, stat ->
  n = length(xs)
  arr = List.to_tuple(xs)
  samples = for _ <- 1..bnum, do: stat.(for _ <- 1..n, do: elem(arr, :rand.uniform(n) - 1))
  s = Enum.sort(samples)
  {stat.(xs), Enum.at(s, div(bnum * 25, 1000)), Enum.at(s, div(bnum * 975, 1000))}
end
mean = fn xs -> Enum.sum(xs) / max(length(xs), 1) end
fmt = fn {m, lo, hi} -> "#{Float.round(m * 1.0, 3)} [#{Float.round(lo * 1.0, 3)}, #{Float.round(hi * 1.0, 3)}]" end

Output.banner("Bootstrap intervals (B = #{bnum})")

drills = specs.(opts[:drill]) |> Enum.map(fn {k, dir} -> {k, jsonl.(Path.join(dir, "rollouts.jsonl"))} end)

if drills != [] do
  Output.puts("Drill rates per start (95 % CI):")
  for {k, rows} <- drills do
    conv = Enum.map(rows, &(if &1["converted"], do: 1.0, else: 0.0))
    open = Enum.map(rows, &(if &1["contact"], do: 1.0, else: 0.0))
    dmg = Enum.map(rows, & &1["damage"])
    Output.puts("  #{String.pad_trailing(k, 8)} n #{length(rows)}  conversion #{fmt.(boot.(conv, mean))}  opening #{fmt.(boot.(open, mean))}  damage #{fmt.(boot.(dmg, mean))}")
  end

  # paired differences on the same pool (by start id)
  Output.puts("Paired differences in conversion (same starts), 95 % CI — the number iterations were decided on:")
  for {{ka, ra}, {kb, rb}} <- (for a <- drills, b <- drills, a != b, do: {a, b}) |> Enum.filter(fn {{ka, _}, {kb, _}} -> ka < kb end) do
    ma = Map.new(ra, &{&1["id"], &1})
    diffs = for r <- rb, a = ma[r["id"]], do: (if r["converted"], do: 1.0, else: 0.0) - (if a["converted"], do: 1.0, else: 0.0)
    if length(diffs) > 10 do
      {m, lo, hi} = boot.(diffs, mean)
      sig = if lo > 0 or hi < 0, do: "  <- outside zero", else: ""
      Output.puts("  #{kb} − #{ka}: #{fmt.({m, lo, hi})} (n #{length(diffs)})#{sig}")
    end
  end
end

oracles = specs.(opts[:oracle]) |> Enum.map(fn {k, dir} -> {k, jsonl.(Path.join(dir, "results.jsonl"))} end)

if oracles != [] do
  Output.puts("Oracle rates per start (95 % CI):")
  for {k, rows} <- oracles do
    any = Enum.map(rows, &(if &1["best"]["converted"] and &1["best"]["alive"], do: 1.0, else: 0.0))
    per = Enum.map(rows, &(&1["n_converted"] / max(&1["tried"], 1)))
    Output.puts("  #{String.pad_trailing(k, 8)} n #{length(rows)}  any-candidate converted #{fmt.(boot.(any, mean))}  per-candidate #{fmt.(boot.(per, mean))}")
  end
end

fps = specs.(opts[:fp]) |> Enum.map(fn {k, path} -> {k, jsonl.(path) |> Enum.filter(&(&1["port"] == 1 and (not String.contains?(path, "style_probe") or String.contains?(&1["path"] || "", "/anon/"))))} end)

if fps != [] do
  tells = ~w(jump_x_ratio cstick_aerial_frac short_hop_frac aerial_per_min roll_forward_per_min spotdodge_per_min lightshield_frac grab_per_min)
  Output.puts("Fingerprint tells per game (95 % CI; n = games — this is why 10-game verdicts are soft):")
  for {k, rows} <- fps do
    line = Enum.map_join(tells, "  ", fn t -> xs = Enum.map(rows, &((&1["features"] || %{})[t] || 0.0)); "#{t} #{fmt.(boot.(xs, mean))}" end)
    Output.puts("  #{k} (n #{length(rows)}): #{line}")
  end
end
