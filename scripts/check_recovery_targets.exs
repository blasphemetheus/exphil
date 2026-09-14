alias ExPhil.Bridge.ControllerState
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ScenarioInputTiming
alias ExPhil.Training.Labels

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      scores: :string,
      out: :string,
      delay: :integer,
      export: :string,
      allow_on_loop: :boolean
    ]
  )

scores_path = Keyword.fetch!(opts, :scores)
out_path = Keyword.fetch!(opts, :out)
delay = Keyword.get(opts, :delay, 2)
if delay < 0, do: raise(ArgumentError, "delay must be nonnegative")
if File.exists?(out_path), do: raise("output already exists: #{out_path}")
if opts[:export] && File.exists?(opts[:export]), do: raise("export already exists")
scores = scores_path |> File.read!() |> Jason.decode!()
if scores["driver"] != "teacher", do: raise("expected an executed teacher scoreboard")

buttons = [
  :button_a,
  :button_b,
  :button_x,
  :button_y,
  :button_z,
  :button_l,
  :button_r,
  :button_d_up
]

decode = fn command ->
  [main_x, main_y, c_x, c_y, left, right] = command["analog"]

  struct!(
    ControllerState,
    Enum.zip(buttons, command["buttons"]) ++
      [
        main_stick: %{x: main_x, y: main_y},
        c_stick: %{x: c_x, y: c_y},
        l_shoulder: left,
        r_shoulder: right
      ]
  )
end

results =
  for run <- scores["runs"] do
    if run["error"] || run["diverged"] != false || run["pass"] != true || run["truncated"],
      do: raise("unqualified teacher run at #{run["frame"]}")

    [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    frames = Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)
    recorded = Map.new(frames, &{&1.game_state.frame, &1.controller})

    samples =
      run["teacher_label_audit"]["comparisons"]
      |> Enum.filter(&(&1["shift"] == 0))
      |> Enum.sort_by(& &1["frame"])

    expected_frames = Enum.to_list(run["frame"]..(run["frame"] + run["window"] - 1))

    if Enum.map(samples, & &1["frame"]) != expected_frames,
      do: raise("incomplete teacher issuance trace")

    issued = Map.new(samples, &{&1["frame"], decode.(&1["actual"])})

    trace =
      Enum.map(samples, fn sample ->
        input = ControllerState.to_input(issued[sample["frame"]])
        %{frame: sample["frame"], sent: input, issued: input}
      end)

    timing = ScenarioInputTiming.verify(trace, recorded, 0)
    segment = Enum.filter(frames, &Map.has_key?(issued, &1.game_state.frame))
    targets = segment |> Labels.tag(:recorded) |> Labels.at_delay(delay, require_tagged: true)
    off_loop = Map.new(samples, &{&1["frame"], not &1["on_loop"]})

    comparisons =
      Enum.map(targets, fn target ->
        frame = target.game_state.frame
        input = ControllerState.to_input(target.controller)
        check = [%{frame: frame, sent: input, issued: input}]
        actual = %{frame => Map.fetch!(issued, frame + delay)}

        %{
          frame: frame,
          off_loop: off_loop[frame],
          matches: ScenarioInputTiming.verify(check, actual, 0).valid
        }
      end)

    %{
      handoff: run["frame"],
      replay: replay_path,
      replay_sha256:
        :crypto.hash(:sha256, File.read!(replay_path)) |> Base.encode16(case: :lower),
      timing: timing,
      initial_state: hd(samples)["state"],
      frame_list: Labels.tag(segment, :recorded),
      targets: length(targets),
      off_loop_targets: Enum.count(comparisons, & &1.off_loop),
      mismatches: Enum.reject(comparisons, & &1.matches),
      valid:
        timing.valid and length(targets) == run["window"] - delay and
          (opts[:allow_on_loop] == true or Enum.any?(comparisons, & &1.off_loop)) and
          Enum.all?(comparisons, & &1.matches)
    }
  end

valid = results != [] and Enum.all?(results, & &1.valid)

report = %{
  scores: scores_path,
  reaction_delay: delay,
  valid: valid,
  runs: Enum.map(results, &Map.delete(&1, :frame_list))
}

File.write!(out_path, Jason.encode!(report, pretty: true), [:exclusive])
IO.puts(Jason.encode!(report, pretty: true))
if not valid, do: System.halt(2)

if path = opts[:export] do
  payload = ExPhil.Training.RecordedFrames.envelope(Enum.map(results, & &1.frame_list), report)
  ExPhil.Training.RecordedFrames.validate!(payload)
  File.write!(path, :erlang.term_to_binary(payload, [:compressed]), [:exclusive])
end
