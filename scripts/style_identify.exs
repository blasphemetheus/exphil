# STYLE_IDENTITY.md S3a: closed-set identity assignment.
#
#   mix run scripts/style_identify.exs --erickfm A.jsonl --yeti B.jsonl \
#     --metric eval_runs/0917_style_identity/metric_eval_erickfm.bin \
#     --out eval_runs/0917_style_identity/identify.json [--threshold 0.8] [--min-games 5]
#
# Gallery = every tag with >= min-games rows across both corpora, in the
# learned metric's space. Leave-one-out evaluation per corpus (accuracy and
# coverage at several thresholds, with/without costume evidence), then the
# untagged games of each corpus get a pseudo-tag when the posterior clears
# --threshold. Writes <out> and player_tag_map.json next to it.
alias ExPhil.Interp.{StyleCalibration, StyleMatcher, StyleMetric}
alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [erickfm: :string, yeti: :string, metric: :string, out: :string, threshold: :float, min_games: :integer, quiet: :boolean])
if opts[:quiet], do: Logger.configure(level: :warning)
out = opts[:out] || raise("--out required")
threshold = opts[:threshold] || 0.8
min_games = opts[:min_games] || 5

corp = %{"erickfm" => Cal.load_jsonl(opts[:erickfm], "erickfm"), "yeti" => Cal.load_jsonl(opts[:yeti], "yeti")}
tagged_all = corp |> Map.values() |> List.flatten() |> Cal.tagged(min_games)
pool = tagged_all |> Map.values() |> List.flatten()
stats = Cal.zscore_stats(pool)
metric = StyleMetric.load!(opts[:metric])
embed = fn rows -> Cal.embed(rows, {:project, stats, &StyleMetric.project(metric, &1)}) end

gallery_rows = Map.new(tagged_all, fn {t, rs} -> {t, embed.(rs)} end)
gallery = StyleMatcher.gallery(gallery_rows, costume_slots: 4)
Output.puts("gallery: #{length(gallery.entities)} entities from #{length(pool)} tagged games; LR same #{inspect(Map.new(gallery.lr.same, fn {k, v} -> {k, Float.round(v, 3)} end))} diff #{inspect(Map.new(gallery.lr.diff, fn {k, v} -> {k, Float.round(v, 3)} end))}")

# Leave-one-out per corpus (gallery rows of that corpus only as queries)
evals =
  Map.new(corp, fn {name, rows} ->
    mine = rows |> Enum.filter(& &1.tag) |> Enum.group_by(& &1.tag) |> Map.take(Map.keys(gallery_rows)) |> Map.new(fn {t, rs} -> {t, embed.(rs)} end)

    [costume: [], no_costume: [costume: false], flat_prior: [flat_prior: true]]
    |> Enum.map(fn {label, o} ->
      r = StyleMatcher.evaluate(mine, gallery, o)
      line = r.at_threshold |> Enum.sort_by(&elem(&1, 0)) |> Enum.map_join(" ", fn {t, s} -> "τ#{t}: cov #{Float.round(s.coverage, 3)} acc #{s.accuracy && Float.round(s.accuracy, 3)}" end)
      Output.puts("  #{name} [#{label}] n=#{r.queries} top1=#{Float.round(r.top1, 3)} | #{line}")
      {label, r}
    end)
    |> Map.new()
    |> then(&{name, &1})
  end)

# Assign the untagged games
assignments =
  Map.new(corp, fn {name, rows} ->
    untagged = rows |> Enum.reject(& &1.tag) |> embed.()
    assigned = StyleMatcher.assign(untagged, gallery, threshold)
    by_tag = Enum.frequencies_by(assigned, & &1.tag) |> Enum.sort_by(&(-elem(&1, 1)))
    Output.puts("  #{name}: #{length(assigned)} of #{length(untagged)} untagged games assigned at τ#{threshold} (#{Float.round(100 * length(assigned) / max(length(untagged), 1), 1)} %); top: #{inspect(Enum.take(by_tag, 8))}")
    {name, %{untagged: length(untagged), assigned: length(assigned), by_tag: Map.new(by_tag), rows: assigned}}
  end)

tag_map = assignments |> Map.values() |> Enum.flat_map(& &1.rows) |> Map.new(fn a -> {a.path, %{tag: a.tag, p: a.p, port: a.port}} end)
File.write!(Path.join(Path.dirname(out), "player_tag_map.json"), Jason.encode!(%{protocol: "style_identify_v1", threshold: threshold, metric: opts[:metric], entries: tag_map}, pretty: true))

report = %{
  protocol: "style_identify_v1",
  threshold: threshold,
  min_games: min_games,
  gallery: %{entities: length(gallery.entities), tagged_games: length(pool), lr: gallery.lr},
  evaluation: evals,
  assignments: Map.new(assignments, fn {n, a} -> {n, Map.delete(a, :rows)} end)
}

File.write!(out, Jason.encode!(report, pretty: true))
Output.success("written #{out} and player_tag_map.json (#{map_size(tag_map)} entries)")
