alias ExPhil.Agents.MultishineExpert
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Labels, RecordedFrames}

fixture = "test/fixtures/replays/fox_multishine_closed_d1.slp"
expert = MultishineExpert.from_fixture(fixture)
out = "eval_runs/0913_tiny_overfit/held.frames"
if File.exists?(out), do: raise("held comparison already exists")

lists =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(fn path ->
    payload = path |> File.read!() |> :erlang.binary_to_term()
    frames = RecordedFrames.validate!(payload)
    runs = Jason.decode!(payload.teacher_validation_json)["runs"]

    Enum.zip_with(frames, runs, fn list, run ->
      {:ok, replay} = Peppi.parse(run["replay"])
      history = replay |> Peppi.to_training_frames() |> Map.new(&{&1.game_state.frame, &1.controller})

      list
      |> Enum.drop(-2)
      |> Enum.map(fn frame ->
        previous = Map.fetch!(history, frame.game_state.frame - 1)
        {:ok, label} = MultishineExpert.label(expert, frame.game_state.players[1], previous)
        frame |> Map.put(:controller, label) |> Map.put(:prev_controller, previous)
      end)
      |> Labels.tag({:expert, MultishineExpert})
    end)
  end)

payload = %{frame_lists: lists, label_convention: :causal, label_delay: 2,
  experiment: "old projected-label/observed-history recipe; NOT a verified teacher export"}
File.write!(out, :erlang.term_to_binary(payload, [:compressed]), [:exclusive])
IO.puts("Held comparator: #{length(lists)} lists, #{Enum.sum(Enum.map(lists, &length/1))} states")
