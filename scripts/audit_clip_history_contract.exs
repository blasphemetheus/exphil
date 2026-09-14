alias ExPhil.Training.{Data, Labels, RecordedFrames}

Enum.each(
  [ExPhil.Bridge.GameState, ExPhil.Bridge.Player, ExPhil.Bridge.ControllerState,
   ExPhil.Bridge.Projectile, ExPhil.Data.Peppi],
  &Code.ensure_loaded!/1
)

{opts, _, []} = OptionParser.parse(System.argv(), strict: [frames: :string, out: :string])
window = 16
queue_depth = 3

rows =
  opts
  |> Keyword.fetch!(:frames)
  |> Path.wildcard()
  |> Enum.flat_map(fn path ->
    path
    |> RecordedFrames.load!()
    |> Enum.map(fn frames ->
      shifted = Labels.at_delay(frames, 2)
      dataset = Data.from_frame_lists([shifted])
      targets = Data.sequence_target_indices(dataset, window)
      starts = dataset.metadata.sequence_starts

      padded = Enum.count(targets, fn target -> target - elem(starts, target) < window - 1 end)
      cold_queue = Enum.count(targets, fn target ->
        start = elem(starts, target)
        first_input = max(start, target - window + 1)
        first_input - start < queue_depth
      end)

      %{source: path, handoff: hd(frames).game_state.frame, targets: length(targets),
        padded_windows: padded, windows_containing_reset_queue_embeddings: cold_queue,
        training_first_window_frames: List.duplicate(hd(frames).game_state.frame, window),
        committed_prefix_first_window_frames: Enum.to_list((hd(frames).game_state.frame - window + 1)..hd(frames).game_state.frame)}
    end)
  end)

if rows == [], do: raise("no teacher windows found")
report = %{window: window, queue_depth: queue_depth, reaction_delay: 2,
  comparison: "training cold-start contract versus saturated committed-prefix contract; not a behavioral test",
  clips: rows}
File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(rows, limit: :infinity)
