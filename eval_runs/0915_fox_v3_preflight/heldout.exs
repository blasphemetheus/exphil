alias ExPhil.Data.{Peppi, SubjectResolver}
alias ExPhil.Evaluation.{BPTT, Forward}
alias ExPhil.Training.{Data, PlayerRegistry, Streaming}

[policy] = System.argv()
base = Path.dirname(__ENV__.file)
# HELDOUT_CORPUS=full uses the full-corpus inventory's validation split (the
# trainer's own 16 hashed val files) instead of the preflight's 16 tagged files.
{corpus_file, split, out_name} =
  case System.get_env("HELDOUT_CORPUS") do
    "full" -> {"full_corpus.json", "validation", "heldout_full.json"}
    _ -> {"corpus.json", "01_validation", "heldout.json"}
  end

rows = File.read!(Path.join(base, corpus_file)) |> Jason.decode!() |> Map.fetch!("rows")
files = Enum.filter(rows, &(&1["split"] == split))
unless length(files) == 16, do: raise("expected all 16 held-out files")
artifact = Forward.load!(policy)

# S5: resolve identities exactly as training did (matched/pseudo tags win)
tag_map =
  case artifact.config[:player_tag_map] do
    nil -> nil
    p -> ExPhil.Training.PlayerTagMap.load!(p)
  end

IO.puts("tag map: " <> if(tag_map, do: "#{tag_map.count} entries", else: "none"))
unless Nx.Defn.default_options()[:precision] == :highest, do: raise("highest arithmetic required")
embed_config = ExPhil.Embeddings.config(Map.to_list(artifact.config))
registry_path = String.replace_suffix(policy, "_policy.bin", "_players.json")
# ONE evaluator for every file and mode: Forward.new builds a fresh predict
# closure and Nx.Defn.jit caches per closure, so per-file construction was
# 32 XLA compiles (~1 min/file, 09-17). Carry resets on each file's first
# chunk (is_resetting = 1), so sharing is exact.
evaluator = Forward.new(artifact.params, artifact.config)
{:ok, registry} = PlayerRegistry.from_json(registry_path)
started = System.monotonic_time(:millisecond)

results =
  Enum.map(files, fn row ->
    path = row["path"]
    hash = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    unless hash == row["sha256"], do: raise("replay changed: #{path}")
    {:ok, metadata} = Peppi.metadata(path)
    {:ok, subject} = SubjectResolver.resolve(metadata.players, subject_character: :fox, ditto_tie_break: :port1)

    {:ok, frames, []} =
      Streaming.parse_chunk([{path, subject.subject_port}],
        subject_character: "Fox",
        label_delay: 0,
        show_progress: false,
        tag_map: tag_map
      )

    if frames == [], do: raise("empty held-out replay: #{path}")

    scores =
      Map.new([anonymous: nil, trained_registry: registry], fn {mode, reg} ->
        dataset =
          frames
          |> Data.from_frames(embed_config: embed_config, player_registry: reg)
          |> Data.precompute_frame_embeddings(show_progress: false)

        result = BPTT.evaluate(evaluator, BPTT.batches(dataset, artifact.config[:unroll] || 80))

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

File.write!(Path.join(Path.dirname(policy), out_name), Jason.encode!(report, pretty: true))
IO.puts(Jason.encode!(Map.drop(report, [:files]), pretty: true))
