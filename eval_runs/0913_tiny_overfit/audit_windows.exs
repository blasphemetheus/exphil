alias ExPhil.Training.{Data, Labels, RecordedFrames}
alias ExPhil.Data.Peppi
alias ExPhil.Agents.MultishineExpert

{opts, _, []} = OptionParser.parse(System.argv(), strict: [boundary_safe: :boolean, out: :string])

fixture = "test/fixtures/replays/fox_multishine_closed_d1.slp"
expert = MultishineExpert.from_fixture(fixture)
{:ok, replay} = Peppi.parse(fixture)

canonical =
  replay
  |> Peppi.to_training_frames()
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: controller} ->
    controller.main_stick.x < 0.25 and controller.main_stick.y > 0.4 and
      not controller.button_b and not controller.button_x
  end)

teacher = Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(&RecordedFrames.load!/1)
lists = Enum.map([canonical | teacher], &Labels.at_delay(&1, 2))
frames = List.flatten(lists)
indexed = Enum.with_index(lists)

embeddings =
  for {list, list_id} <- indexed, {frame, local_id} <- Enum.with_index(list),
    do: [list_id * 1.0, local_id * 1.0, frame.game_state.frame * 1.0]

dataset = if opts[:boundary_safe], do: Data.from_frame_lists(lists), else: Data.from_frames(frames)
dataset = %{dataset | embedded_frames: Nx.tensor(embeddings)}

rows =
  dataset
  |> Data.batched_sequences(lazy: true, shuffle: false, window_size: 16, batch_size: 64)
  |> Enum.flat_map(fn batch ->
    batch.states |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_list()
  end)
  |> Enum.map(fn window ->
    [list_id, local_id, frame] = List.last(window)
    list_id = trunc(list_id)
    local_id = trunc(local_id)
    target = lists |> Enum.at(list_id) |> Enum.at(local_id)

    %{list: list_id, frame: trunc(frame),
      crosses_clip: Enum.any?(window, fn [source, _, _] -> trunc(source) != list_id end),
      off_loop: list_id > 0 and not MultishineExpert.on_loop?(expert, target.game_state.players[1])}
  end)

report = %{frames: length(frames), sequences: length(rows),
  crossed_clip_sequences: Enum.count(rows, & &1.crosses_clip),
  off_loop_teacher_targets: Enum.count(rows, & &1.off_loop),
  off_loop_with_wrong_clip_context: Enum.count(rows, &(&1.off_loop and &1.crosses_clip)),
  examples: rows |> Enum.filter(&(&1.off_loop and &1.crosses_clip)) |> Enum.take(8)}
File.write!(opts[:out] || "eval_runs/0913_tiny_overfit/window_audit.json", Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(report)
