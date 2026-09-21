# SIM_INTEGRATION.md step 4 (R1) scoring: the prior's fingerprint in the
# sim vs its fingerprint in Dolphin (the S6(c)/D3 probe games), in the same
# NCA style metric + raw habit tells used for the human comparisons.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_r1_compare.exs \
#     --dolphin checkpoints/.../style_probe/bot_fingerprints.jsonl --dolphin-arm anon \
#     --sim eval_runs/0921_sim_r1/anon_self_n10/sim_fingerprints.jsonl

alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Interp.StyleMetric

{opts, _, _} = OptionParser.parse(System.argv(), strict: [dolphin: :string, dolphin_arm: :string, sim: :string])
r_dir = "eval_runs/0917_style_identity"
targets = ["SKWA", "314", "C2", "USSR"]

humans = (Cal.load_jsonl("#{r_dir}/erickfm_fox.jsonl", "erickfm") ++ Cal.load_jsonl("#{r_dir}/yeti_fox.jsonl", "yeti")) |> Cal.tagged(5)
pool = humans |> Map.values() |> List.flatten()
stats = Cal.zscore_stats(pool)
metric = StyleMetric.load!("#{r_dir}/metric_eval_erickfm.bin")
embed = fn rows -> Cal.embed(rows, {:project, stats, &StyleMetric.project(metric, &1)}) end
cent = Map.new(targets, fn t -> {t, Cal.centroid(embed.(humans[t])).vec} end)
dist = fn a, b -> :math.sqrt(Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, s -> s + (x - y) * (x - y) end)) end

arm = opts[:dolphin_arm] || "anon"
dolphin = Cal.load_jsonl(opts[:dolphin], "dolphin") |> Enum.filter(&(&1.port == 1 and String.contains?(&1.path, "/#{arm}/")))
sim_all = Cal.load_jsonl(opts[:sim], "sim")
sim_p1 = Enum.filter(sim_all, &(&1.port == 1))
sim_p2 = Enum.filter(sim_all, &(&1.port == 2))

arms = [{"dolphin:#{arm}", dolphin}, {"sim:p1", sim_p1}, {"sim:p2", sim_p2}]
IO.puts("games per arm: #{inspect(Enum.map(arms, fn {k, v} -> {k, length(v)} end))}")

IO.puts("\nMean distance (learned metric) from each arm's games to each human centroid, and to the OTHER arm's centroid")
d_cent = Cal.centroid(embed.(dolphin)).vec
s_cent = Cal.centroid(embed.(sim_p1)).vec
IO.puts(String.pad_trailing("arm", 14) <> Enum.map_join(targets ++ ["dolphinC", "simC"], "", &String.pad_leading(&1, 10)))
for {name, rows} <- arms, rows != [] do
  e = embed.(rows)
  line = for c <- Enum.map(targets, &cent[&1]) ++ [d_cent, s_cent] do
    d = e |> Enum.map(&dist.(&1.vec, c)) |> then(&(Enum.sum(&1) / length(&1)))
    String.pad_leading(Float.to_string(Float.round(d, 2)), 10)
  end
  IO.puts(String.pad_trailing(name, 14) <> Enum.join(line))
end

# Within-arm spread (mean pairwise distance) vs cross-arm distance: the parity test.
pair_mean = fn a, b ->
  ds = for x <- a, y <- b, x != y, do: dist.(x.vec, y.vec)
  if ds == [], do: 0.0, else: Enum.sum(ds) / length(ds)
end
ed = embed.(dolphin); es = embed.(sim_p1)
IO.puts("\nmean pairwise distance: dolphin-dolphin #{Float.round(pair_mean.(ed, ed), 2)}  sim-sim #{Float.round(pair_mean.(es, es), 2)}  dolphin-sim #{Float.round(pair_mean.(ed, es), 2)}")

IO.puts("\nNearest human per game:")
for {name, rows} <- arms, rows != [] do
  near = embed.(rows) |> Enum.map(fn r -> targets |> Enum.min_by(&dist.(r.vec, cent[&1])) end)
  IO.puts("  #{String.pad_trailing(name, 12)} -> #{inspect(near)}")
end

feats = [:jump_x_ratio, :dashdance_per_min, :wavedash_per_min, :cstick_aerial_frac, :lcancel_press_offset_mean, :short_hop_frac, :lightshield_frac, :roll_forward_per_min, :spotdodge_per_min, :aerial_per_min, :grab_per_min, :airdodge_per_min]
mean = fn rows, k -> rows |> Enum.map(&Map.get(&1.features, k, 0.0)) |> then(&(Enum.sum(&1) / max(length(&1), 1))) |> Float.round(2) end
sd = fn rows, k ->
  vs = Enum.map(rows, &Map.get(&1.features, k, 0.0)); m = Enum.sum(vs) / max(length(vs), 1)
  :math.sqrt(Enum.reduce(vs, 0.0, fn v, s -> s + (v - m) * (v - m) end) / max(length(vs) - 1, 1)) |> Float.round(2)
end
IO.puts("\nRaw habits, mean (sd): rows = arms; cols = " <> Enum.map_join(feats, " ", &to_string/1))
for {name, rows} <- arms, rows != [] do
  IO.puts("  " <> String.pad_trailing(name, 12) <> Enum.map_join(feats, " ", fn k -> String.pad_leading("#{mean.(rows, k)}(#{sd.(rows, k)})", 12) end))
end

# Gate: for each tell, |sim mean - dolphin mean| <= 2 * dolphin sd (n=10 spread).
tells = [:jump_x_ratio, :cstick_aerial_frac, :short_hop_frac, :aerial_per_min, :roll_forward_per_min, :spotdodge_per_min, :lightshield_frac, :grab_per_min]
verdicts = for k <- tells do
  ok = abs(mean.(sim_p1, k) - mean.(dolphin, k)) <= 2 * max(sd.(dolphin, k), 0.01)
  {k, ok}
end
IO.puts("\nR1 tells within 2 sd of the Dolphin arm: #{inspect(verdicts)}")
if Enum.all?(verdicts, &elem(&1, 1)), do: IO.puts("R1 GATE PASSED"), else: IO.puts("R1 GATE: #{Enum.count(verdicts, &(not elem(&1, 1)))} tell(s) outside the spread")
