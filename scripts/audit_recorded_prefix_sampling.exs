alias ExPhil.Training.{Data, Labels, RecordedFrames, RecordedPrefixSampling}
alias ExPhil.Data.Peppi
Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
{:ok, replay} = Peppi.parse("test/fixtures/replays/fox_multishine_closed_d1.slp")

canonical =
  replay
  |> Peppi.to_training_frames()
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: controller} ->
    controller.main_stick.x < 0.25 and controller.main_stick.y > 0.4 and
      not controller.button_b and not controller.button_x
  end)

teachers =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(fn path -> RecordedPrefixSampling.mark(RecordedFrames.load!(path), path) end)

lists = Enum.map([canonical | teachers], &Labels.at_delay(&1, 2))
{weights, stats} = RecordedPrefixSampling.frame_weights(lists, 64, 18)
dataset = Data.from_frame_lists(lists)

embeddings =
  for {frames, clip} <- Enum.with_index(lists),
      {_frame, index} <- Enum.with_index(frames),
      do: [clip * 1.0, index * 1.0]

dataset = %{dataset | embedded_frames: Nx.tensor(embeddings)}

windows =
  Data.batched_sequences(dataset,
    lazy: true,
    window_size: 16,
    batch_size: 64,
    gpu: false,
    seed: 42,
    sampling_weights: weights
  )
  |> Enum.flat_map(&Nx.to_list(&1.states))

frequencies = Enum.frequencies_by(windows, &List.last/1)

for {frames, clip} <- Enum.with_index(lists), {_frame, index} <- Enum.with_index(frames) do
  expected = if clip > 0 and index < 18, do: 64, else: 1
  unless frequencies[[clip * 1.0, index * 1.0]] == expected, do: raise("wrong replication")
end

crossed =
  Enum.count(windows, fn window ->
    clip = hd(List.last(window))
    Enum.any?(window, &(hd(&1) != clip))
  end)

unless crossed == 0 and stats.targets == 7785 and length(windows) == 14589,
  do: raise("pool/window contract changed")

report =
  Map.merge(stats, %{
    observed_draws: length(windows),
    crossed_clip_windows: crossed,
    retained_unique_targets: map_size(frequencies),
    batches: ceil(length(windows) / 64),
    early_draw_fraction: stats.early_draws / stats.draws,
    cases:
      Enum.map(teachers, fn frames ->
        %{
          handoff: hd(frames).game_state.frame,
          raw_frames: length(frames),
          early_targets: 18,
          early_draws: 1152
        }
      end)
  })

File.write!(out, Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(report)
