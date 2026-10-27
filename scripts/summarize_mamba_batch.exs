# AI-generated; attribution and method: docs/guides/MAMBA_BATCH_PARITY.md.
for path <- System.argv() do
  report = path |> File.read!() |> Jason.decode!()
  rows = Enum.flat_map(report["events"], & &1["rows"])
  isolated = Enum.flat_map(report["events"], &(&1["isolated"] || []))

  stages =
    for key <- hd(isolated)["stages"] |> Map.keys() |> Enum.sort(), into: %{} do
      values = Enum.map(isolated, & &1["stages"][key])

      {key,
       %{
         max_abs: Enum.max(Enum.map(values, & &1["max_abs"])),
         failures: Enum.count(values, &(not &1["close"]))
       }}
    end

  summary = %{
    file: path,
    client: report["client"],
    versions: report["versions"],
    samples: length(rows),
    controller_differences: Enum.count(rows, &(not &1["controller_equal"])),
    embedding_differences: Enum.count(rows, &(&1["embedding"]["max_abs"] != 0)),
    carry:
      for(
        key <- ["h", "conv"],
        into: %{},
        do:
          {key,
           %{
             max_abs: Enum.max(Enum.map(rows, & &1["carry"][key]["max_abs"])),
             failures: Enum.count(rows, &(not &1["carry"][key]["close"]))
           }}
      ),
    isolated_output_max: Enum.max(Enum.map(isolated, & &1["output"]["max_abs"])),
    isolated_stages: stages
  }

  File.write!(Path.rootname(path) <> "_summary.json", Jason.encode!(summary, pretty: true))
  IO.puts(Jason.encode!(summary, pretty: true))
end
