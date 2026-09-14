alias ExPhil.Data.Peppi

{opts, _, []} = OptionParser.parse(System.argv(), strict: [scores: :string, out: :string])
scores = opts |> Keyword.fetch!(:scores) |> File.read!() |> Jason.decode!()
buttons = ~w(a b x y z l r d_up)

runs =
  Enum.map(scores["runs"], fn run ->
    [path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {:ok, replay} = Peppi.parse(path)
    recorded = Map.new(replay.frames, &{&1.frame_number, &1.players[1].controller})

    rows =
      Enum.map(run["policy_inputs"], fn sample ->
        sent = sample["sent"]
        actual = Map.fetch!(recorded, sample["frame"] + 1)

        digital =
          Enum.all?(buttons, fn button ->
            sent["buttons"][button] == Map.fetch!(actual, String.to_existing_atom("button_" <> button))
          end)

        values = [
          {"main_x", sent["main_stick"]["x"], actual.main_stick_x},
          {"main_y", sent["main_stick"]["y"], actual.main_stick_y},
          {"c_x", sent["c_stick"]["x"], actual.c_stick_x},
          {"c_y", sent["c_stick"]["y"], actual.c_stick_y},
          {"left_trigger", sent["shoulder"], actual.l_trigger},
          {"right_trigger", 0.0, actual.r_trigger}
        ]

        differences =
          for {field, expected, observed} <- values, abs(expected - observed) > 0.02,
            do: %{field: field, expected: expected, observed: observed}

        %{frame: sample["frame"], digital_match: digital, differences: differences}
      end)

    grid =
      for offset <- 0..8 do
        matches = Enum.count(run["policy_inputs"], fn sample ->
          case recorded[sample["frame"] + offset] do
            nil -> false
            actual -> Enum.all?(buttons, fn button ->
              sample["sent"]["buttons"][button] == Map.fetch!(actual, String.to_existing_atom("button_" <> button))
            end)
          end
        end)
        %{offset: offset, digital_matches: matches}
      end

    %{frame: run["frame"], run: run["run"], count: length(rows),
      digital_matches: Enum.count(rows, & &1.digital_match),
      analog_mismatch_frames: Enum.count(rows, &(&1.differences != [])),
      digital_offset_grid: grid,
      difference_counts: rows |> Enum.flat_map(& &1.differences) |> Enum.frequencies_by(& &1.field),
      examples: rows |> Enum.filter(&(&1.differences != [] or not &1.digital_match)) |> Enum.take(5)}
  end)

report = %{scores: opts[:scores], expected_send_offset: 1, runs: runs,
  total_commands: Enum.sum(Enum.map(runs, & &1.count)),
  digital_matches: Enum.sum(Enum.map(runs, & &1.digital_matches)),
  analog_mismatch_frames: Enum.sum(Enum.map(runs, & &1.analog_mismatch_frames))}

File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(Map.drop(report, [:runs]))
