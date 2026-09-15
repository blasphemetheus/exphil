# Run with devenv shell -- elixir -pa '_build/dev/lib/*/ebin' this_file.exs
# Diagnostic only: the production prefix gate still compares exact bits.
alias ExPhil.Data.Peppi
report = Path.join(__DIR__, "coverage03/teacher.json") |> File.read!() |> JSON.decode!()
runs = Enum.filter(report["runs"], &String.contains?(&1["slp"], "local_multishine"))
{:ok, source} = Peppi.parse(hd(runs)["slp"])
index = fn frames -> Map.new(frames, &{&1.frame_number, &1}) end
source_frames = index.(source.frames)
bits = fn v -> if is_float(v), do: Base.encode16(<<v::float-32>>), else: inspect(v) end
results = for run <- runs do
  [path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
  {:ok, replay} = Peppi.parse(path)
  frames = index.(replay.frames)
  diffs = for n <- -39..(run["frame"] - 1), port <- [1, 2],
      {key, a} <- Map.from_struct(source_frames[n].players[port]), key != :controller,
      b = Map.fetch!(frames[n].players[port], key), bits.(a) != bits.(b) do
    %{frame: n, port: port, field: key, source: a, rerun: b,
      source_bits: bits.(a), rerun_bits: bits.(b), zero_sign_only: a == 0 and b == 0}
  end
  groups = diffs |> Enum.group_by(&{&1.port, &1.field}) |> Enum.map(fn {{port, field}, ds} ->
    %{port: port, field: field, count: length(ds), first: hd(ds), last: List.last(ds),
      only_signed_zero: Enum.all?(ds, & &1.zero_sign_only)}
  end)
  %{handoff: run["frame"], replay: path, source: run["slp"], metadata: Map.from_struct(replay.metadata),
    input_audit_exact: Enum.all?(run["prefix_audit"]["ports"], &is_nil(&1["input"])),
    differing_fields: groups, nonzero_differences: Enum.count(diffs, &(not &1.zero_sign_only)),
    first_nonzero: Enum.find(diffs, &(not &1.zero_sign_only))}
end
plain = fn recur, value ->
  cond do
    is_map(value) -> value |> Map.delete(:__struct__) |> Map.new(fn {k, v} -> {k, recur.(recur, v)} end)
    is_list(value) -> Enum.map(value, &recur.(recur, &1))
    true -> value
  end
end
output = plain.(plain, %{source_metadata: source.metadata, runs: results})
File.write!(Path.join(__DIR__, "human_audit.json"), JSON.encode!(output) <> "\n")
IO.inspect(output, limit: :infinity)
