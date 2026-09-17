# STYLE_IDENTITY.md step 2: calibrate the fingerprint matcher on tagged games.
#
#   mix run scripts/style_calibrate.exs \
#     --rows erickfm=eval_runs/0917_style_identity/erickfm_fox.jsonl \
#     --rows yeti=eval_runs/0917_style_identity/yeti_fox.jsonl \
#     --out eval_runs/0917_style_identity/calibration.json [--min-games 5]
#
# Reports, per corpus and cross-corpus (gallery = one corpus, queries =
# the other, shared tags only): same/different distance distributions,
# thresholds at 1 % / 5 % false-match rate, leave-one-out nearest-centroid
# retrieval top-1/top-5 — for the full metric and the character-invariant
# metric. Pure CPU; no GPU needed.
alias ExPhil.Interp.StyleCalibration, as: Cal
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [rows: :keep, out: :string, min_games: :integer, quiet: :boolean])

if opts[:quiet], do: Logger.configure(level: :warning)
out = opts[:out] || raise("--out PATH required")
min_games = opts[:min_games] || 5

corpora =
  opts
  |> Keyword.get_values(:rows)
  |> Map.new(fn spec ->
    [name, path] = String.split(spec, "=", parts: 2)
    rows = Cal.load_jsonl(path, name)
    Output.puts("#{name}: #{length(rows)} rows, #{Enum.count(rows, & &1.tag)} tagged")
    {name, rows}
  end)

by_tag = Map.new(corpora, fn {name, rows} -> {name, Cal.tagged(rows, min_games)} end)

report_for = fn tagged, label ->
  Map.new([:full, :invariant], fn metric ->
    dists = Cal.pair_distributions(tagged, metric: metric)
    ret = Cal.retrieval(tagged, metric: metric)

    Output.puts(
      "  #{label} [#{metric}] tags=#{map_size(tagged)} games=#{tagged |> Map.values() |> List.flatten() |> length()} " <>
        "same p50=#{Float.round(Cal.summary(dists.same)[:p50] || 0.0, 3)} diff p50=#{Float.round(Cal.summary(dists.different)[:p50] || 0.0, 3)} " <>
        "top1=#{ret.top1 && Float.round(ret.top1, 3)} top5=#{ret.top5 && Float.round(ret.top5, 3)} (gallery #{ret.gallery_size})"
    )

    {metric,
     %{
       same: Cal.summary(dists.same),
       different: Cal.summary(dists.different),
       fmr_1pct: Cal.threshold_at_fmr(dists, 0.01),
       fmr_5pct: Cal.threshold_at_fmr(dists, 0.05),
       retrieval: ret
     }}
  end)
end

within = Map.new(by_tag, fn {name, tagged} -> {name, report_for.(tagged, name)} end)

cross =
  for {qa, ta} <- by_tag, {ga, tg} <- by_tag, qa != ga, into: %{} do
    shared = Map.keys(ta) -- (Map.keys(ta) -- Map.keys(tg))

    res =
      Map.new([:full, :invariant], fn metric ->
        r = Cal.retrieval(ta, gallery: tg, metric: metric)
        Output.puts("  cross queries=#{qa} gallery=#{ga} [#{metric}] shared_tags=#{length(shared)} queries=#{r.queries} top1=#{r.top1 && Float.round(r.top1, 3)} top5=#{r.top5 && Float.round(r.top5, 3)}")
        {metric, r}
      end)

    {"#{qa}->#{ga}", Map.put(res, :shared_tags, shared)}
  end

report = %{
  protocol: "style_calibration_v1",
  min_games: min_games,
  corpora: Map.new(corpora, fn {n, rows} -> {n, %{rows: length(rows), tagged: Enum.count(rows, & &1.tag), tags_kept: map_size(by_tag[n])}} end),
  within: within,
  cross: cross
}

File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(report, pretty: true))
Output.success("calibration written to #{out}")
