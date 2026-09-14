{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")

entries =
  for category <- ["neutral", "sustain", "recovery"],
      run <-
        File.read!("eval_runs/0913_teacher_ingestion/validated/#{category}_targets.json")
        |> Jason.decode!()
        |> Map.fetch!("runs") do
    unless run["valid"] and run["targets"] == 118, do: raise("unvalidated teacher")
    digest = :crypto.hash(:sha256, File.read!(run["replay"])) |> Base.encode16(case: :lower)
    unless digest == run["replay_sha256"], do: raise("teacher replay changed")

    %{
      slp: run["replay"],
      frame: run["handoff"],
      type: "multishine_reentry",
      note: "Matched executed teacher: #{category}; original checkpoint, cold delay 2",
      teacher_sha256: digest
    }
  end

unless Enum.map(entries, & &1.frame) == [2228, 2566, 900, 4, 75, 146],
  do: raise("unexpected teacher cases")

File.write!(out, Jason.encode!(%{entries: entries}, pretty: true), [:exclusive])
