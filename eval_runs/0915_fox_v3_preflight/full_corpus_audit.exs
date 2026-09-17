alias ExPhil.Data.{Peppi, SubjectResolver}
base = Path.dirname(__ENV__.file)
root = Path.expand("replays/erickfm_ranked/v2_filtered")
paths = Path.wildcard(Path.join(root, "**/*.slp")) |> Enum.sort()
started = System.monotonic_time(:millisecond)

rows =
  paths
  |> Task.async_stream(
    fn path ->
      case Peppi.metadata(path) do
        {:ok, metadata} ->
          if Enum.any?(metadata.players, &(&1.character == 2)) do
            case SubjectResolver.resolve(metadata.players,
                   subject_character: :fox,
                   ditto_tie_break: :port1
                 ) do
              {:ok, subject} ->
                %{
                  path: path,
                  bytes: File.stat!(path).size,
                  sha256: :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower),
                  stage: metadata.stage,
                  duration_frames: metadata.duration_frames,
                  subject_port: subject.subject_port,
                  opponent_port: subject.opponent_port,
                  provenance: subject.provenance,
                  players: length(metadata.players)
                }

              {:error, error} ->
                %{path: path, error: inspect(error)}
            end
          end

        {:error, error} ->
          %{path: path, error: inspect(error)}
      end
    end,
    max_concurrency: 8,
    timeout: :infinity,
    ordered: true
  )
  |> Enum.map(fn {:ok, row} -> row end)
  |> Enum.reject(&is_nil/1)

{errors, candidates} = Enum.split_with(rows, &Map.has_key?(&1, :error))
{train, validation} = Enum.split(candidates, length(candidates) - 16)
train_hashes = MapSet.new(Enum.map(train, & &1.sha256))
leaks = Enum.filter(validation, &MapSet.member?(train_hashes, &1.sha256))

duplicates =
  candidates |> Enum.group_by(& &1.sha256) |> Enum.filter(fn {_, rs} -> length(rs) > 1 end)

report = %{
  source: root,
  discovered: length(paths),
  selected: length(candidates),
  errors: errors,
  train_files: length(train),
  validation_files: length(validation),
  stage_counts: Enum.frequencies_by(candidates, & &1.stage),
  port_counts: Enum.frequencies_by(candidates, & &1.subject_port),
  ditto_files: Enum.count(candidates, &(&1.provenance == :character_tie_break)),
  non_singles: Enum.filter(candidates, &(&1.players != 2)),
  total_metadata_frames: Enum.sum(Enum.map(candidates, &max(&1.duration_frames || 0, 0))),
  total_bytes: Enum.sum(Enum.map(candidates, & &1.bytes)),
  training_chunks: train |> Enum.chunk_every(200) |> Enum.with_index(1)
    |> Enum.map(fn {rs, index} -> %{index: index, files: length(rs),
      metadata_frames: Enum.sum(Enum.map(rs, &max(&1.duration_frames || 0, 0))),
      bytes: Enum.sum(Enum.map(rs, & &1.bytes))} end),
  longest_replays: candidates |> Enum.sort_by(& &1.duration_frames, :desc) |> Enum.take(10),
  duplicate_groups: duplicates,
  train_validation_hash_overlap: leaks,
  elapsed_ms: System.monotonic_time(:millisecond) - started,
  rows:
    Enum.map(train, &Map.put(&1, :split, :train)) ++
      Enum.map(validation, &Map.put(&1, :split, :validation))
}

File.write!(Path.join(base, "full_corpus.json"), Jason.encode!(report, pretty: true))

IO.puts(
  Jason.encode!(Map.drop(report, [:rows, :duplicate_groups, :non_singles, :errors]), pretty: true)
)

IO.puts(
  "Errors: #{length(errors)}; duplicate groups: #{length(duplicates)}; non-singles: #{length(report.non_singles)}"
)
