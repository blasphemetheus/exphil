# Run with plain Elixir and compiled dev beams; no GPU needed.
alias ExPhil.Data.Peppi
root = Path.expand("eval_runs/0915_fox_v3_preflight")
File.mkdir_p!(root)
source = Path.expand("replays/erickfm_ranked/v2_filtered")
stages = [2, 3, 8, 28, 31, 32]
val_counts = %{2 => 3, 3 => 3, 8 => 3, 28 => 3, 31 => 2, 32 => 2}
quotas = Map.new(stages, &{&1, 40 + val_counts[&1]})
hash = fn bytes -> Base.encode16(:crypto.hash(:sha256, bytes), case: :lower) end

files = Path.wildcard(source <> "/**/*.slp")
# Filename screening bounds the inventory cost (the NIF metadata call parses
# full files). Every admitted game must also have actual Fox metadata.
candidates = files |> Enum.filter(&String.contains?(Path.basename(&1), "Fox"))
  |> Enum.sort_by(&hash.("fox-v3-preflight-915:" <> Path.basename(&1)))
IO.puts("Screening #{length(candidates)} named-Fox candidates from #{length(files)} files")

{groups, _seen, scanned} =
  Enum.reduce_while(candidates, {%{}, MapSet.new(), 0}, fn path, {groups, seen, n} ->
    complete = Enum.all?(stages, &(length(Map.get(groups, &1, [])) == quotas[&1]))
    if complete do
      {:halt, {groups, seen, n}}
    else
      result = Peppi.metadata(path)
      admitted = case result do
        {:ok, m} -> m.stage in stages and length(m.players) == 2 and
          Enum.any?(m.players, &(&1.character == 2)) and
          m.duration_frames >= 1800 and m.duration_frames <= 36000 and
          length(Map.get(groups, m.stage, [])) < quotas[m.stage]
        _ -> false
      end

      if admitted do
        {:ok, meta} = result
        sha = hash.(File.read!(path))
        if MapSet.member?(seen, sha) do
          {:cont, {groups, seen, n + 1}}
        else
          row = %{source: path, sha256: sha, stage: meta.stage,
            duration_frames: meta.duration_frames,
            players: Enum.map(meta.players, &Map.from_struct/1)}
          groups = Map.update(groups, meta.stage, [row], &(&1 ++ [row]))
          if rem(Enum.sum(Enum.map(groups, fn {_, rows} -> length(rows) end)), 32) == 0,
            do: IO.inspect(Map.new(groups, fn {k, v} -> {k, length(v)} end))
          {:cont, {groups, MapSet.put(seen, sha), n + 1}}
        end
      else
        {:cont, {groups, seen, n + 1}}
      end
    end
  end)

unless Enum.all?(stages, &(length(Map.get(groups, &1, [])) == quotas[&1])),
  do: raise("Could not fill the declared stage quotas")

rows = Enum.flat_map(stages, fn stage ->
  groups[stage] |> Enum.with_index() |> Enum.map(fn {row, index} ->
    split = if index < 40, do: "00_train", else: "01_validation"
    dest = Path.join([root, "replays", split, Path.basename(row.source)])
    File.mkdir_p!(Path.dirname(dest))
    File.ln_s!(row.source, dest)
    Map.merge(row, %{split: split, path: dest})
  end)
end)
report = %{source_directory: source, total_source_files: length(files),
  named_fox_candidates: length(candidates), metadata_scanned: scanned,
  limitation: "Bounded named-Fox, 30s..10min, six-stage subset; not a uniform full-corpus sample",
  selection: "SHA256 filename order, 40 train per stage, 16 held-out whole games; exact duplicates excluded",
  split_order: "Pipeline preserves sorted paths; final 16 files are 01_validation for both seeds",
  rows: rows}
File.write!(Path.join(root, "corpus.json"), Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(%{files: length(rows), scanned: scanned, splits: Enum.frequencies_by(rows, & &1.split)})

for seed <- [905, 906] do
  out = Path.join(root, "seed_#{seed}")
  File.mkdir_p!(out)
  args = ["run", "scripts/train.exs", "--backbone", "gru", "--temporal", "--stage-internals",
    "--hidden-sizes", "512,512,256", "--batch-size", "128", "--dropout", "0.1", "--precision", "f32",
    "--bptt", "--unroll", "80", "--bptt-overlap", "1", "--bptt-val-files", "16",
    "--learn-player-styles", "--stream-chunk-size", "200", "--replays", Path.join(root,"replays"),
    "--train-character", "fox", "--select-character-port", "--label-delay", "0",
    "--epochs", "2", "--seed", to_string(seed), "--head", "autoregressive", "--save-best",
    "--label-smoothing", "0.0", "--no-focal-loss", "--button-pos-weight", "1,1,1,1,1,1,1,1",
    "--action-oversample", "1.0", "--entropy-weight", "0.0", "--neutral-weight", "1.0",
    "--stick-edge-weight", "1.0", "--no-register", "--checkpoint", Path.join(out,"model.axon")]
  File.write!(Path.join(out,"train_args.json"), Jason.encode!(args, pretty: true), [:exclusive])
end
