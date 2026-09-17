# Export complete scored teacher gameplay, including unsuccessful approaches and
# interruption handling. Finalization inputs are outside the recorded state bounds.
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Labels, RecordedFrames}
[directory, output] = System.argv()
summary = directory |> Path.join("summary.json") |> File.read!() |> Jason.decode!()
unless summary["passed"], do: raise("Teacher gameplay did not pass its opening gate")

{lists, sources} =
  summary["runs"]
  |> Enum.map(fn run ->
    unless run["opening_gate"], do: raise("Unapproved teacher game")
    path = run["replay"]
    state_path = Path.join([directory, "run_#{run["run"]}", "states.bin"])
    states = state_path |> File.read!() |> :erlang.binary_to_term()
    first = hd(states).frame
    last = List.last(states).frame
    {:ok, replay} = Peppi.parse(path)
    players = Map.new(replay.metadata.players, &{&1.port, &1.character_name})

    unless replay.metadata.stage == 32 and players == %{1 => "Mewtwo", 2 => "Fox"},
      do: raise("Expected Mewtwo P1 versus Fox P2 on Final Destination")

    unless Enum.map(states, & &1.frame) == Enum.to_list(first..last),
      do: raise("Teacher state snapshots contain a gap")

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2, remap_ports: true)
      # Last snapshot is followed by a teacher action. Its successor is still
      # before the separate finalization input loop.
      |> Enum.filter(&(&1.game_state.frame >= first - 20 and &1.game_state.frame <= last))
      |> Enum.map(&Map.put(&1, :input_only, &1.game_state.frame < first))
      |> Labels.tag(:recorded)

    unless Enum.count(frames, &(not &1.input_only)) == length(states),
      do: raise("Replay is missing scored teacher frames")

    source = %{
      path: path,
      sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower),
      state_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(state_path)), case: :lower),
      first_target: first,
      last_target: last,
      outcomes: run["metrics"]["neutral"]["outcomes"]
    }

    {frames, source}
  end)
  |> Enum.unzip()

report = %{
  source_type: :scripted_teacher,
  sources: sources,
  frames: Enum.sum(Enum.map(lists, &length/1)),
  targets: Enum.sum(Enum.map(lists, &Enum.count(&1, fn f -> not f.input_only end))),
  selection:
    "Complete scored teacher games, including losses and interruptions; no success-only selection",
  limitation:
    "Teacher passed an opening gate, not an optimality claim; no offstage recovery or combo routing"
}

payload = RecordedFrames.envelope(lists, report)
RecordedFrames.validate!(payload)
File.mkdir_p!(Path.dirname(output))
File.write!(output, :erlang.term_to_binary(payload), [:exclusive])
File.write!(output <> ".json", Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(Map.take(report, [:frames, :targets]))
