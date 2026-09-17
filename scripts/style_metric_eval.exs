# STYLE_IDENTITY.md step 2b: does z-scoring / a learned NCA projection beat
# the raw hand-feature distance on players the metric never saw?
#
#   mix run scripts/style_metric_eval.exs --erickfm A.jsonl --yeti B.jsonl \
#     --out eval_runs/0917_style_identity/metric_eval.json [--dim 24 --steps 400]
alias ExPhil.Interp.{StyleCalibration, StyleMetric}
alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [erickfm: :string, yeti: :string, out: :string, dim: :integer, steps: :integer, min_games: :integer, quiet: :boolean])
if opts[:quiet], do: Logger.configure(level: :warning)
out = opts[:out] || raise("--out required")
dim = opts[:dim] || 24
steps = opts[:steps] || 400
min_games = opts[:min_games] || 5

e = Cal.load_jsonl(opts[:erickfm], "erickfm")
y = Cal.load_jsonl(opts[:yeti], "yeti")
te = Cal.tagged(e, min_games)
ty = Cal.tagged(y, min_games)
pool = (Map.values(te) ++ Map.values(ty)) |> List.flatten()
stats = Cal.zscore_stats(pool)
Output.puts("erickfm tags=#{map_size(te)} yeti tags=#{map_size(ty)} pool=#{length(pool)} games")

embed_tagged = fn tagged, how -> Map.new(tagged, fn {t, rs} -> {t, Cal.embed(rs, how)} end) end

score = fn label, q_tagged, g_tagged ->
  within = Cal.retrieval(q_tagged)
  cross = if g_tagged, do: Cal.retrieval(q_tagged, gallery: g_tagged), else: nil
  Output.puts("  #{String.pad_trailing(label, 44)} within top1=#{Float.round(within.top1, 3)} top5=#{Float.round(within.top5, 3)}" <> if(cross, do: "  cross top1=#{Float.round(cross.top1, 3)} top5=#{Float.round(cross.top5, 3)} (#{cross.queries} q, #{cross.tags} shared)", else: ""))
  %{within: Map.delete(within, :per_tag), cross: cross && Map.delete(cross, :per_tag)}
end

fit = fn tagged, label ->
  rows = tagged |> Map.values() |> List.flatten()
  ids = tagged |> Map.keys() |> Enum.with_index() |> Map.new()
  {us, m} = :timer.tc(fn -> StyleMetric.fit(Enum.map(rows, & &1.vec), Enum.map(rows, &ids[&1.tag]), dim: dim, steps: steps, seed: 11) end)
  Output.puts("  fit #{label}: #{length(rows)} games, #{map_size(ids)} players, loss #{Float.round(hd(m.loss), 3)} -> #{Float.round(List.last(m.loss), 3)} (#{div(us, 1000)} ms)")
  m
end

report = %{dim: dim, steps: steps, min_games: min_games, experiments: %{}}

# 1. raw vs z-score
te_raw = embed_tagged.(te, :raw); ty_raw = embed_tagged.(ty, :raw)
te_z = embed_tagged.(te, {:zscore, stats}); ty_z = embed_tagged.(ty, {:zscore, stats})
Output.puts("Baselines (queries erickfm | gallery yeti):")
r1 = score.("raw: erickfm", te_raw, ty_raw)
r2 = score.("zscore: erickfm", te_z, ty_z)
Output.puts("Baselines (queries yeti | gallery erickfm):")
r3 = score.("raw: yeti", ty_raw, te_raw)
r4 = score.("zscore: yeti", ty_z, te_z)

# 2. NCA trained on erickfm players -> Yeti (never seen) and cross
m_e = fit.(te_z, "on erickfm")
proj = fn m -> {:project, stats, &StyleMetric.project(m, &1)} end
r5 = score.("nca(erickfm-trained): yeti queries", embed_tagged.(ty, proj.(m_e)), embed_tagged.(te, proj.(m_e)))
# 3. NCA trained on Yeti -> erickfm (never seen)
m_y = fit.(ty_z, "on yeti")
r6 = score.("nca(yeti-trained): erickfm queries", embed_tagged.(te, proj.(m_y)), embed_tagged.(ty, proj.(m_y)))
# 4. held-out players WITHIN erickfm (80/20 split of tags), the open-set number
{train_tags, test_tags} = te |> Map.keys() |> Enum.sort() |> Enum.split(div(map_size(te) * 4, 5))
m_split = fit.(embed_tagged.(Map.take(te_z, train_tags), :raw) |> then(fn _ -> Map.take(te_z, train_tags) end), "on 80% of erickfm players")
held_raw = score.("held-out 20% erickfm players: zscore", Map.take(te_z, test_tags), nil)
held_nca = score.("held-out 20% erickfm players: nca", embed_tagged.(Map.take(te, test_tags), proj.(m_split)), nil)

StyleMetric.save!(m_e, Path.rootname(out) <> "_erickfm.bin")
StyleMetric.save!(m_y, Path.rootname(out) <> "_yeti.bin")
keys = ExPhil.Interp.StyleFingerprint.keys()
imp = Enum.zip(keys, StyleMetric.importance(m_e)) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(12)
Output.puts("top features (erickfm-trained): " <> Enum.map_join(imp, ", ", fn {k, v} -> "#{k}=#{Float.round(v, 2)}" end))

report = %{report | experiments: %{raw_erickfm: r1, zscore_erickfm: r2, raw_yeti: r3, zscore_yeti: r4, nca_e_to_yeti: r5, nca_y_to_erickfm: r6, heldout_zscore: held_raw, heldout_nca: held_nca, importance: Map.new(imp, fn {k, v} -> {k, v} end), held_out_tags: test_tags}}
File.write!(out, Jason.encode!(report, pretty: true))
Output.success("written #{out}")
