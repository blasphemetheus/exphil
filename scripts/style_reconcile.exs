# STYLE_IDENTITY.md S4: reconciliation report for Bradley to adjudicate.
#
#   mix run scripts/style_reconcile.exs --erickfm A.jsonl --yeti B.jsonl \
#     --metric metric_eval_erickfm.bin --tag-map player_tag_map.json \
#     --mains data/identity/braacket_stlmelee_mains.tsv --out reconciliation.md
#
# Sections: collision suspects (one tag, several styles), mislabel suspects
# (a tagged game that confidently looks like another player), cluster
# hypotheses (each ~cNN: size, costumes, nearest known entities), and the
# questions whose answers turn tags into people.
alias ExPhil.Interp.{StyleCalibration, StyleClusters, StyleMatcher, StyleMetric}
alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [erickfm: :string, yeti: :string, metric: :string, tag_map: :string, mains: :string, out: :string, quiet: :boolean])
if opts[:quiet], do: Logger.configure(level: :warning)
out = opts[:out] || raise("--out required")

e = Cal.load_jsonl(opts[:erickfm], "erickfm")
y = Cal.load_jsonl(opts[:yeti], "yeti")
tagged = (e ++ y) |> Cal.tagged(5)
pool = tagged |> Map.values() |> List.flatten()
stats = Cal.zscore_stats(pool)
metric = StyleMetric.load!(opts[:metric])
embed = fn rows -> Cal.embed(rows, {:project, stats, &StyleMetric.project(metric, &1)}) end
gallery_rows = Map.new(tagged, fn {t, rs} -> {t, embed.(rs)} end)
gallery = StyleMatcher.gallery(gallery_rows, costume_slots: 4)
entity = Map.new(gallery.entities, &{&1.tag, &1})
tag_map = File.read!(opts[:tag_map]) |> Jason.decode!()
pct = fn x -> "#{Float.round(x * 100, 1)} %" end
mains =
  case opts[:mains] do
    nil -> []
    p -> p |> File.read!() |> String.split("\n", trim: true) |> Enum.reject(&String.starts_with?(&1, "#")) |> Enum.map(fn l -> [n, m] = String.split(l, "\t"); {n, String.split(m, "|")} end)
  end

# 1. Per-tag leave-one-out: where does each tag's posterior mass go?
per_tag =
  Enum.map(gallery_rows, fn {tag, rows} ->
    posts = Enum.map(rows, &StyleMatcher.posterior(&1, gallery, exclude_self: true))
    tops = Enum.map(posts, &hd/1)
    own = Enum.count(tops, fn {t, _} -> t == tag end) / length(rows)
    stolen = tops |> Enum.reject(fn {t, _} -> t == tag end) |> Enum.frequencies_by(&elem(&1, 0)) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(3)
    costumes = rows |> Enum.map(& &1.costume) |> Enum.filter(&is_integer/1) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1)))
    corpora = rows |> Enum.frequencies_by(& &1.corpus)
    %{tag: tag, n: length(rows), own: own, stolen: stolen, costumes: costumes, corpora: corpora,
      mislabels: Enum.zip(rows, tops) |> Enum.filter(fn {_, {t, p}} -> t != tag and t != "?" and p >= 0.9 end) |> Enum.map(fn {r, {t, p}} -> {Path.basename(r.path), t, p} end)}
  end)
  |> Enum.sort_by(& -&1.n)

collisions = per_tag |> Enum.filter(&(&1.n >= 20 and &1.own < 0.4)) |> Enum.sort_by(& &1.own)
mislabels = per_tag |> Enum.flat_map(fn t -> Enum.map(t.mislabels, &Tuple.insert_at(&1, 0, t.tag)) end) |> Enum.sort_by(&(-elem(&1, 3))) |> Enum.take(40)

# 2. Clusters: members from the tag map, nearest entities by centroid distance
pseudo_entries = tag_map["entries"] |> Enum.filter(fn {_, v} -> String.starts_with?(v["tag"], "~c") end)
by_pseudo = Enum.group_by(pseudo_entries, fn {_, v} -> v["tag"] end, fn {p, _} -> p end)
row_by_path = Map.new(e ++ y, &{Path.expand(&1.path), &1})

clusters =
  by_pseudo
  |> Enum.map(fn {ptag, paths} ->
    rows = paths |> Enum.map(&Map.get(row_by_path, Path.expand(&1))) |> Enum.reject(&is_nil/1) |> embed.()
    cent = rows |> Enum.map(& &1.vec) |> Enum.zip_with(&Enum.sum/1) |> Enum.map(&(&1 / length(rows)))
    near = gallery.entities |> Enum.map(fn en -> {en.tag, :math.sqrt(Enum.zip(cent, en.centroid) |> Enum.reduce(0.0, fn {a, b}, s -> s + (a - b) * (a - b) end))} end) |> Enum.sort_by(&elem(&1, 1)) |> Enum.take(3)
    costumes = rows |> Enum.map(& &1.costume) |> Enum.filter(&is_integer/1) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1)))
    stages = rows |> Enum.map(& &1.stage) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(3)
    %{tag: ptag, n: length(rows), near: near, costumes: costumes, stages: stages}
  end)
  |> Enum.sort_by(& -&1.n)

# 3. Assigned real tags: top assignees and their braacket cross-check
assigned = tag_map["entries"] |> Enum.reject(fn {_, v} -> String.starts_with?(v["tag"], "~c") end) |> Enum.frequencies_by(fn {_, v} -> v["tag"] end) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(25)

md = """
# Style identity reconciliation — #{Date.utc_today()}

Gallery: #{length(gallery.entities)} entities (tags with >= 5 games, erickfm + Yeti, Fox subject), #{length(pool)} tagged games.
LR fit: same mean #{Float.round(gallery.lr.same.mean, 2)} sd #{Float.round(gallery.lr.same.std, 2)}; different mean #{Float.round(gallery.lr.diff.mean, 2)} sd #{Float.round(gallery.lr.diff.std, 2)}.
Tag map: #{map_size(tag_map["entries"])} entries (#{length(pseudo_entries)} pseudo-tagged).

## 1. Collision suspects (one tag, several styles): n >= 20 and < 40 % of games recognised as their own tag

| tag | games | own | mass goes to | costumes | corpora |
| --- | ---: | ---: | --- | --- | --- |
#{Enum.map_join(collisions, "\n", fn t -> "| #{t.tag} | #{t.n} | #{pct.(t.own)} | #{Enum.map_join(t.stolen, ", ", fn {o, c} -> "#{o} (#{c})" end)} | #{Enum.map_join(t.costumes, " ", fn {c, k} -> "#{c}:#{k}" end)} | #{inspect(t.corpora)} |" end)}

Reading: a tag whose games consistently look like ANOTHER tag is either two people sharing a tag, or the same person under two tags (see the "mass goes to" column both ways).

## 2. Mislabel suspects: tagged games that look like another known entity with p >= 0.9 (top 40)

| tagged as | file | looks like | p |
| --- | --- | --- | ---: |
#{Enum.map_join(mislabels, "\n", fn {tag, file, other, p} -> "| #{tag} | #{file} | #{other} | #{Float.round(p, 3)} |" end)}

## 3. Best-recognised tags (own >= 80 %, n >= 20) — the reliable anchors

| tag | games | own | costumes | corpora |
| --- | ---: | ---: | --- | --- |
#{per_tag |> Enum.filter(&(&1.n >= 20 and &1.own >= 0.8)) |> Enum.map_join("\n", fn t -> "| #{t.tag} | #{t.n} | #{pct.(t.own)} | #{Enum.map_join(t.costumes, " ", fn {c, k} -> "#{c}:#{k}" end)} | #{inspect(t.corpora)} |" end)}

## 4. Pseudo-tag clusters (erickfm hashed games): who might they be?

| cluster | games | nearest known entities (distance) | costumes | stages |
| --- | ---: | --- | --- | --- |
#{Enum.map_join(clusters, "\n", fn c -> "| #{c.tag} | #{c.n} | #{Enum.map_join(c.near, ", ", fn {t, d} -> "#{t} (#{Float.round(d, 2)})" end)} | #{Enum.map_join(c.costumes, " ", fn {k, v} -> "#{k}:#{v}" end)} | #{Enum.map_join(c.stages, " ", fn {k, v} -> "#{k}:#{v}" end)} |" end)}

A nearest-entity distance near the same-player mean (#{Float.round(gallery.lr.same.mean, 2)}) suggests the cluster IS that player's hashed games; near the different mean (#{Float.round(gallery.lr.diff.mean, 2)}) it is a style, not a person.

## 5. Real tags assigned to anonymous games (top 25) with braacket mains cross-check

| tag | assigned games | braacket players whose handle contains the tag |
| --- | ---: | --- |
#{Enum.map_join(assigned, "\n", fn {tag, n} -> "| #{tag} | #{n} | #{mains |> Enum.filter(fn {h, _} -> String.contains?(String.downcase(h), String.downcase(tag)) end) |> Enum.map_join(", ", fn {h, m} -> "#{h} (#{Enum.join(m, "/")})" end)} |" end)}

## 6. Questions for Bradley

1. Tag -> handle anchors (YETI_SCENE_PRIORS.md): is **C2** OG Swaglord (green Fox, slot 3)? Is **TJOM** Timtempor (red, slot 1)? Which of **CUMB / FOX / NELL** (blue, slot 2) is Trash Machine? Is **314** Spikenard or messiSTL?
2. Is **FOX** one person? (Section 1/2 shows where its games go.)
3. For each cluster in section 4 whose nearest entity sits near the same-player distance: do you recognise the player? Naming it merges the cluster into that entity.
4. Any tag in section 1 you know to be shared or borrowed setups?

Answers go into `data/identity/entity_aliases.tsv` (`entity<TAB>alias|alias...`) and rerun `style_identify.exs` with merged entities.
"""

File.write!(out, md)
Output.success("wrote #{out} (#{length(collisions)} collision suspects, #{length(mislabels)} mislabel rows, #{length(clusters)} clusters)")
