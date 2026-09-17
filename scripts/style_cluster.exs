# STYLE_IDENTITY.md S3b: perceived-player clusters for games the closed-set
# matcher left anonymous.
#
#   mix run scripts/style_cluster.exs --erickfm A.jsonl --yeti B.jsonl \
#     --metric metric_eval_erickfm.bin --tag-map player_tag_map.json \
#     --out cluster.json [--ks 20,40,80 --k 40 --min-size 30 --core 1.0]
#
# Fits k-means on the still-anonymous erickfm rows for each k in --ks and
# reports tagged-game purity (tagged rows are projected into the same
# clusters, not used to fit). Then, at --k, games within --core x the
# cluster's median radius get pseudo-tag ~cNN (clusters smaller than
# --min-size are dropped) and the tag map is extended in place.
alias ExPhil.Interp.{StyleCalibration, StyleClusters, StyleMetric}
alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [erickfm: :string, yeti: :string, metric: :string, tag_map: :string, out: :string, ks: :string, k: :integer, min_size: :integer, core: :float, quiet: :boolean])
if opts[:quiet], do: Logger.configure(level: :warning)
out = opts[:out] || raise("--out required")
ks = (opts[:ks] || "20,40,80") |> String.split(",") |> Enum.map(&String.to_integer/1)
k_final = opts[:k] || 40
min_size = opts[:min_size] || 30
core = opts[:core] || 1.0

e = Cal.load_jsonl(opts[:erickfm], "erickfm")
y = Cal.load_jsonl(opts[:yeti], "yeti")
tagged = (e ++ y) |> Cal.tagged(5)
pool = tagged |> Map.values() |> List.flatten()
stats = Cal.zscore_stats(pool)
metric = StyleMetric.load!(opts[:metric])
embed = fn rows -> Cal.embed(rows, {:project, stats, &StyleMetric.project(metric, &1)}) end

tag_map = File.read!(opts[:tag_map]) |> Jason.decode!()
already = tag_map["entries"] |> Map.keys() |> MapSet.new()
anon = e |> Enum.reject(&(&1.tag || MapSet.member?(already, &1.path))) |> embed.()
tagged_rows = embed.(pool)
Output.puts("clustering #{length(anon)} anonymous erickfm games; purity probe on #{length(tagged_rows)} tagged games")

probe = fn model ->
  pairs = StyleClusters.predict(model, Enum.map(tagged_rows, & &1.vec)) |> Enum.zip(tagged_rows) |> Enum.map(fn {{c, _}, r} -> {r.tag, c} end)
  StyleClusters.purity(pairs)
end

sweep =
  Map.new(ks, fn k ->
    {us, model} = :timer.tc(fn -> StyleClusters.fit(Enum.map(anon, & &1.vec), k, iters: 60) end)
    sizes = model.assignments |> Enum.frequencies() |> Map.values() |> Enum.sort(:desc)
    pur = probe.(model)
    Output.puts("  k=#{k}: inertia #{Float.round(model.inertia, 1)} sizes max #{hd(sizes)} min #{List.last(sizes)} (#{Enum.count(sizes, &(&1 >= min_size))} >= #{min_size}) | tagged purity #{Float.round(pur.purity, 3)} over #{pur.tags} tags in #{pur.clusters_used} clusters (#{div(us, 1000)} ms)")
    {k, %{inertia: model.inertia, sizes: sizes, purity: pur}}
  end)

model = StyleClusters.fit(Enum.map(anon, & &1.vec), k_final, iters: 60)
pred = StyleClusters.predict(model, Enum.map(anon, & &1.vec))
by_cluster = Enum.zip(pred, anon) |> Enum.group_by(fn {{c, _}, _} -> c end)

pseudo =
  by_cluster
  |> Enum.flat_map(fn {c, members} ->
    if length(members) < min_size do
      []
    else
      ds = members |> Enum.map(fn {{_, d}, _} -> d end) |> Enum.sort()
      radius = Enum.at(ds, div(length(ds), 2)) * core
      members |> Enum.filter(fn {{_, d}, _} -> d <= radius end) |> Enum.map(fn {{_, d}, r} -> {r.path, %{tag: "~c#{String.pad_leading(Integer.to_string(c), 2, "0")}", p: d, port: r.port}} end)
    end
  end)
  |> Map.new()

clusters_kept = pseudo |> Map.values() |> Enum.map(& &1.tag) |> Enum.uniq() |> length()
Output.puts("  k=#{k_final}: #{map_size(pseudo)} games -> #{clusters_kept} pseudo-tags (core #{core} x median radius, min size #{min_size})")

entries = Map.merge(tag_map["entries"], pseudo)
File.write!(opts[:tag_map], Jason.encode!(tag_map |> Map.put("entries", entries) |> Map.put("clusters", %{k: k_final, min_size: min_size, core: core, pseudo_tags: clusters_kept}), pretty: true))
File.write!(out, Jason.encode!(%{protocol: "style_cluster_v1", anonymous_games: length(anon), sweep: sweep, chosen_k: k_final, pseudo_games: map_size(pseudo), pseudo_tags: clusters_kept}, pretty: true))
Output.success("tag map now #{map_size(entries)} entries -> #{opts[:tag_map]}; report #{out}")
