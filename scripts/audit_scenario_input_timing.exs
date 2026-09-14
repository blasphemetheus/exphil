alias ExPhil.Data.Peppi

{opts, _, invalid} = OptionParser.parse(System.argv(), strict: [scores: :string, out: :string])
if invalid != [], do: raise(ArgumentError, "invalid options: #{inspect(invalid)}")
scores = opts |> Keyword.fetch!(:scores) |> File.read!() |> Jason.decode!()

buttons = ["a", "b", "x", "y", "z", "l", "r", "d_up"]

matches = fn input, recorded ->
  digital =
    Enum.all?(buttons, fn button ->
      Map.get(input["buttons"], button, false) ==
        Map.fetch!(recorded, String.to_existing_atom("button_" <> button))
    end)

  analog = [
    {input["main_stick"]["x"], recorded.main_stick_x},
    {input["main_stick"]["y"], recorded.main_stick_y},
    {input["c_stick"]["x"], recorded.c_stick_x},
    {input["c_stick"]["y"], recorded.c_stick_y},
    {input["shoulder"] || 0.0, max(recorded.l_trigger, recorded.r_trigger)}
  ]

  digital and Enum.all?(analog, fn {sent, observed} -> abs(sent - observed) <= 0.02 end)
end

results =
  Enum.map(scores["runs"], fn run ->
    paths = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    if length(paths) != 1, do: raise("expected exactly one replay in #{run["replay_dir"]}")
    [path] = paths
    {:ok, replay} = Peppi.parse(path)
    frames = Map.new(replay.frames, &{&1.frame_number, &1.players[1].controller})
    players = Map.new(replay.frames, &{&1.frame_number, &1.players[1]})
    trace = Map.fetch!(run, "policy_inputs")
    sent_frames = MapSet.new(trace, & &1["frame"])

    issued =
      Enum.filter(trace, &MapSet.member?(sent_frames, &1["frame"] + scores["response_delay"]))

    grid =
      for field <- ["issued", "sent"], latency <- 0..8 do
        comparisons =
          for sample <- if(field == "issued", do: issued, else: trace),
              recorded = frames[sample["frame"] + latency],
              recorded != nil,
              do: matches.(sample[field], recorded)

        %{
          field: field,
          latency: latency,
          count: length(comparisons),
          matches: Enum.count(comparisons, & &1)
        }
      end

    alignment =
      for offset <- -2..2 do
        pairs =
          for sample <- trace,
              Map.has_key?(sample, "action"),
              player = players[sample["frame"] + offset],
              player != nil,
              do: {sample, player}

        matching =
          Enum.filter(pairs, fn {sample, player} ->
            sample["action"] == player.action and sample["on_ground"] == player.on_ground
          end)

        deltas =
          matching
          |> Enum.group_by(fn {sample, _player} -> sample["action"] end)
          |> Map.new(fn {action, group} ->
            {action,
             group
             |> Enum.map(fn {sample, player} ->
               sample["action_frame"] - trunc(player.action_frame)
             end)
             |> Enum.uniq()
             |> Enum.sort()}
          end)

        %{
          offset: offset,
          count: length(pairs),
          matching_action_ground: length(matching),
          action_frame_deltas: deltas
        }
      end

    timing_valid =
      Enum.all?(
        [{"issued", scores["response_delay"] + 1}, {"sent", 1}],
        fn {field, latency} ->
          row = Enum.find(grid, &(&1.field == field and &1.latency == latency))
          row != nil and row.count > 0 and row.matches == row.count
        end
      )

    %{
      replay: path,
      frame: run["frame"],
      run: run["run"],
      diverged: run["diverged"],
      unsent_tail_decisions: length(trace) - length(issued),
      grid: grid,
      state_alignment: alignment,
      timing_valid: timing_valid
    }
  end)

report = %{
  scores: opts[:scores],
  expected_issued_latency: scores["response_delay"] + 1,
  expected_sent_latency: 1,
  runs: results
}

File.write!(Keyword.fetch!(opts, :out), Jason.encode!(report, pretty: true), [:exclusive])
IO.inspect(report, limit: :infinity)
