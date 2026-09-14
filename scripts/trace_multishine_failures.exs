alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, Labels, RecordedFrames}
alias ExPhil.Bridge.ControllerState
alias ExPhil.Agents.MultishineExpert

Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [scores: :string, out: :string])
scores_path = Keyword.fetch!(opts, :scores)
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists: #{out}")
scores = scores_path |> File.read!() |> Jason.decode!()

unless scores["prefix_history"] == "cold" and scores["response_delay"] == 2,
  do: raise("expected the cold delay-2 proof")

expert = MultishineExpert.from_fixture("test/fixtures/replays/fox_multishine_closed_d1.slp")

for module <- [
      ExPhil.Bridge.GameState,
      ExPhil.Bridge.Player,
      ControllerState,
      ExPhil.Bridge.Projectile
    ],
    do: Code.ensure_loaded!(module)

teachers =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(&RecordedFrames.load!/1)
  |> Map.new(&{hd(&1).game_state.frame, &1})

decode = fn input ->
  fields =
    Enum.map(input["buttons"], fn {name, value} ->
      {String.to_existing_atom("button_" <> name), value}
    end)

  struct!(
    ControllerState,
    fields ++
      [
        main_stick: %{x: input["main_stick"]["x"], y: input["main_stick"]["y"]},
        c_stick: %{x: input["c_stick"]["x"], y: input["c_stick"]["y"]},
        l_shoulder: input["shoulder"],
        r_shoulder: 0.0
      ]
  )
  |> Data.controller_to_action()
end

state_keys = [
  :action,
  :action_frame,
  :on_ground,
  :x,
  :y,
  :speed_air_x_self,
  :speed_ground_x_self,
  :speed_y_self,
  :jumps_left
]

equal = fn left, right ->
  if is_number(left) and is_number(right), do: abs(left - right) < 1.0e-5, else: left == right
end

results =
  Enum.map(scores["runs"], fn run ->
    unless run["timing_valid"] == true and run["diverged"] == false and
             run["frames_observed"] == 120,
           do: raise("invalid run at #{run["frame"]}")

    [replay_path] = Path.wildcard(Path.join(run["replay_dir"], "*.slp"))
    {:ok, replay} = Peppi.parse(replay_path)
    live = Peppi.to_training_frames(replay) |> Map.new(&{&1.game_state.frame, &1})
    teacher_list = Map.fetch!(teachers, run["frame"])
    teacher = Map.new(teacher_list, &{&1.game_state.frame, &1})

    targets =
      Labels.at_delay(teacher_list, 2)
      |> Map.new(&{&1.game_state.frame, Data.controller_to_action(&1.controller)})

    rows =
      Enum.map(run["policy_inputs"], fn trace ->
        frame = trace["frame"]
        actual = Map.fetch!(live, frame).game_state.players[1]
        expected = Map.fetch!(teacher, frame).game_state.players[1]

        state_differences =
          Enum.filter(state_keys, &(not equal.(Map.fetch!(actual, &1), Map.fetch!(expected, &1))))

        issued = decode.(trace["issued"])

        %{
          frame: frame,
          relative_frame: frame - run["frame"],
          live_state: Map.take(actual, state_keys),
          teacher_state: Map.take(expected, state_keys),
          raw_live_action_frame: trace["action_frame"],
          raw_live_action: trace["action"],
          normalized_live_action_frame:
            ExPhil.Data.ActionFrameConvention.libmelee_to_parsed(
              actual.character,
              trace["action"],
              trace["action_frame"]
            ),
          state_differences: state_differences,
          off_loop: not MultishineExpert.on_loop?(expert, actual),
          issued: issued,
          teacher_delayed_target: targets[frame],
          issued_matches_teacher_target:
            if(targets[frame], do: issued == targets[frame], else: nil),
          sent: decode.(trace["sent"]),
          teacher_sent: Data.controller_to_action(teacher[frame].controller)
        }
      end)

    first_break = rows |> Enum.drop_while(& &1.off_loop) |> Enum.find(& &1.off_loop)

    %{
      handoff: run["frame"],
      run: run["run"],
      max_chain: run["details"]["max_chain"],
      passes_proof: run["details"]["max_chain"] >= 10,
      replay: replay_path,
      first_state_divergence: Enum.find(rows, &(&1.state_differences != [])),
      first_issued_divergence: Enum.find(rows, &(&1.issued_matches_teacher_target == false)),
      pending_inputs: Enum.take(rows, 2),
      pending_mismatches: Enum.take(rows, 2) |> Enum.count(&(&1.sent != &1.teacher_sent)),
      raw_action_frame_mismatches:
        Enum.count(rows, &(&1.raw_live_action_frame != &1.live_state.action_frame)),
      normalized_action_frame_mismatches:
        Enum.count(rows, &(&1.normalized_live_action_frame != &1.live_state.action_frame)),
      action_mismatches: Enum.count(rows, &(&1.raw_live_action != &1.live_state.action)),
      normalization_residuals:
        rows
        |> Enum.filter(&(&1.normalized_live_action_frame != &1.live_state.action_frame))
        |> Enum.group_by(& &1.live_state.action)
        |> Map.new(fn {action, samples} ->
          {action,
           %{
             count: length(samples),
             raw_minus_parsed:
               Enum.uniq(
                 Enum.map(samples, &(&1.raw_live_action_frame - &1.live_state.action_frame))
               )
           }}
        end),
      action_runs:
        rows
        |> Enum.chunk_by(& &1.live_state.action)
        |> Enum.map(fn segment ->
          %{
            action: hd(segment).live_state.action,
            start: hd(segment).relative_frame,
            frames: length(segment)
          }
        end),
      first_loop_exit: first_break,
      loop_exit_context:
        if(first_break,
          do: Enum.filter(rows, &(abs(&1.relative_frame - first_break.relative_frame) <= 5)),
          else: []
        ),
      first24: Enum.take(rows, 24)
    }
  end)

File.write!(
  out,
  Jason.encode!(
    %{
      scores: scores_path,
      comparison:
        "discretized commands and matched replay player-1 states; after divergence teacher targets are reference trajectory only",
      runs: results
    },
    pretty: true
  ),
  [:exclusive]
)

Enum.each(results, &IO.inspect(Map.take(&1, [:handoff, :run, :max_chain, :pending_mismatches])))
