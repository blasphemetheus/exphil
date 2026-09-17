alias ExPhil.Data.{Peppi, SubjectResolver}
alias ExPhil.Evaluation.{BPTT, Forward}
alias ExPhil.Training.{Data, PlayerRegistry, Streaming}

[policy] = System.argv()
base = Path.dirname(__ENV__.file)
rows = File.read!(Path.join(base, "corpus.json")) |> Jason.decode!() |> Map.fetch!("rows")
files = Enum.filter(rows, &(&1["split"] == "01_validation"))
unless length(files) == 16, do: raise("expected all 16 held-out files")
artifact = Forward.load!(policy)
unless Nx.Defn.default_options()[:precision] == :highest, do: raise("highest arithmetic required")
embed_config = ExPhil.Embeddings.config(Map.to_list(artifact.config))
registry_path = String.replace_suffix(policy, "_policy.bin", "_players.json")
{:ok, registry} = PlayerRegistry.from_json(registry_path)
started = System.monotonic_time(:millisecond)

results =
  Enum.map(files, fn row ->
    path = row["path"]
    hash = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    unless hash == row["sha256"], do: raise("replay changed: #{path}")
    {:ok, metadata} = Peppi.metadata(path)
    {:ok, subject} = SubjectResolver.resolve(metadata.players, subject_character: :fox)

    {:ok, frames, []} =
      Streaming.parse_chunk([{path, subject.subject_port}],
        subject_character: "Fox",
        label_delay: 0,
        show_progress: false
      )

    if frames == [], do: raise("empty held-out replay: #{path}")

    scores =
      Map.new([anonymous: nil, trained_registry: registry], fn {mode, reg} ->
        dataset =
          frames
          |> Data.from_frames(embed_config: embed_config, player_registry: reg)
          |> Data.precompute_frame_embeddings(show_progress: false)

        result =
          BPTT.evaluate(
            Forward.new(artifact.params, artifact.config),
            BPTT.batches(dataset, artifact.config[:unroll] || 80)
          )

        unless result.frames == length(frames), do: raise("incomplete held-out coverage")
        {mode, result}
      end)

    result = %{
      file: path,
      sha256: hash,
      subject_port: subject.subject_port,
      expected_frames: length(frames),
      scores: scores
    }

    IO.puts(Jason.encode!(result))
    result
  end)

total = Enum.sum(Enum.map(results, & &1.expected_frames))

summary =
  Map.new([:anonymous, :trained_registry], fn mode ->
    loss = Enum.sum(Enum.map(results, &(&1.scores[mode].loss * &1.expected_frames))) / total
    {mode, %{loss: loss, frames: total}}
  end)

report = %{
  policy: Path.expand(policy),
  policy_sha256:
    :crypto.hash(:sha256, File.read!(policy))
    |> Base.encode16(case: :lower),
  arithmetic_precision: :highest,
  protocol: "complete_heldout_teacher_forced_plain_ce_v1",
  files: results,
  total_files: length(results),
  total_frames: total,
  summary: summary,
  elapsed_ms: System.monotonic_time(:millisecond) - started
}

File.write!(Path.join(Path.dirname(policy), "heldout.json"), Jason.encode!(report, pretty: true))
IO.puts(Jason.encode!(Map.drop(report, [:files]), pretty: true))
