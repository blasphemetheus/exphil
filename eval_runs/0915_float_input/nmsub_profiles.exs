# Explicit compatibility declarations for this measured corpus, not a general
# inference from player type, path, or replay format version.
accurate_sources = MapSet.new([
  "eval_runs/local_multishine_20260913_224806/2026-09-Mainline/Game_20260913T224817.slp"
])
manifest = Path.join(__DIR__, "coverage03/mined.json") |> File.read!() |> JSON.decode!()
entries = Enum.map(manifest["entries"], fn entry ->
  Map.put(entry, "accurate_nmsub", MapSet.member?(accurate_sources, entry["slp"]))
end)
File.write!(Path.join(__DIR__, "nmsub_compatible_mined.json"), JSON.encode!(Map.put(manifest, "entries", entries)))
longest = entries |> Enum.group_by(& &1["slp"]) |> Enum.map(fn {_, group} ->
  Enum.max_by(group, & &1["frame"])
end) |> Enum.sort_by(& &1["slp"])
File.write!(Path.join(__DIR__, "nmsub_matched_manifest.json"), JSON.encode!(%{"entries" => longest}))
IO.puts("Declared profiles for #{length(entries)} entries and #{length(longest)} source games")
