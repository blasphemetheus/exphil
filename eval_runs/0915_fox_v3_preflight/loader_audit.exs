alias ExPhil.Training.{Data, TrajectoryCursors, Streaming}
Nx.default_backend(Nx.BinaryBackend)

dataset = fn lengths ->
  frames =
    Enum.flat_map(lengths, fn length ->
      Enum.map(0..(length - 1), fn frame ->
        %{
          game_state: %{frame: frame},
          action: %{
            buttons: Map.new([:a, :b, :x, :y, :z, :l, :r, :d_up], &{&1, false}),
            main_x: 8,
            main_y: 8,
            c_x: 8,
            c_y: 8,
            shoulder: 0
          }
        }
      end)
    end)

  n = length(frames)
  %Data{frames: frames, embedded_frames: Nx.iota({n, 1}, axis: 0) |> Nx.as_type(:f32), size: n}
end

stream = fn ds, batch_size, overlap ->
  TrajectoryCursors.batch_stream(ds,
    batch_size: batch_size,
    unroll: 80,
    overlap: overlap,
    gpu: false,
    neutral_weight: 1.0
  )
  |> Enum.to_list()
end

[first, second | _] = stream.(dataset.([240, 240]), 2, 1)
last_first = first.states[0][79][0] |> Nx.to_number()
first_second = second.states[0][0][0] |> Nx.to_number()

coverage =
  for overlap <- [0, 1] do
    ds = dataset.([80, 8000])
    batches = stream.(ds, 2, overlap)
    seen = Enum.flat_map(batches, &Nx.to_flat_list(&1.states)) |> MapSet.new() |> MapSet.size()

    %{
      overlap: overlap,
      available: ds.size,
      unique_scored: seen,
      dropped: ds.size - seen,
      dropped_fraction: (ds.size - seen) / ds.size,
      documented_max_tail: 2 * 80
    }
  end

tail =
  try do
    stream.(dataset.(List.duplicate(160, 40)), 128, 1)
    "unexpected success"
  rescue
    e in ArgumentError -> Exception.message(e)
  end

report = %{
  verdict: "NO_GO",
  repeated_input: %{
    last_frame_of_first_chunk: last_first,
    first_frame_of_second_chunk: first_second,
    second_chunk_reset: Nx.to_number(second.is_resetting[0]),
    duplicate_with_carried_memory: last_first == first_second
  },
  exhausted_row_coverage: coverage,
  preflight_file_chunk_sizes:
    Streaming.chunk_files(Enum.to_list(1..240), 200) |> Enum.map(&length/1),
  final_chunk_with_one_segment_per_game: tail,
  note:
    "Synthetic counterexamples establish loader contract failures; they do not estimate real-corpus loss."
}

File.write!(
  "eval_runs/0915_fox_v3_preflight/loader_audit.json",
  Jason.encode!(report, pretty: true)
)

IO.inspect(report, pretty: true, limit: :infinity)
