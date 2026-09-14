alias ExPhil.Training.{Data, Labels, RecordedFrames, RecordedContext}
alias ExPhil.Data.Peppi
alias ExPhil.Agents.Agent, as: PolicyAgent
Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out_dir: :string])
out = Keyword.fetch!(opts, :out_dir)
if File.exists?(out), do: raise("output directory exists")
File.mkdir_p!(out)

reports = [
  "eval_runs/0913_teacher_ingestion/validated/neutral_targets.json",
  "eval_runs/0913_teacher_ingestion/validated/sustain_targets.json",
  "eval_runs/0913_teacher_ingestion/validated/recovery_targets.json",
  "eval_runs/0913_context_recovery/teacher/validation_with_on_loop.json"
]

policy = "eval_runs/0913_no_dropout_fit/round21/candidate.bin"

results =
  for source <- reports, run <- (File.read!(source) |> Jason.decode!())["runs"] do
    unless run["valid"], do: raise("unvalidated teacher")
    hash = :crypto.hash(:sha256, File.read!(run["replay"])) |> Base.encode16(case: :lower)
    unless hash == run["replay_sha256"], do: raise("teacher replay hash mismatch")
    {:ok, replay} = Peppi.parse(run["replay"])
    frames = Peppi.to_training_frames(replay)
    response_length = run["targets"] + 2

    for mode <- [:cold, :warm] do
      context = if mode == :warm, do: 18, else: 0
      list = RecordedContext.slice(frames, run["handoff"], response_length, context)
      shifted = Labels.at_delay(list, 2) |> Enum.map(&Map.put(&1, :delay_id, 2))
      dataset = Data.from_frame_lists([shifted])

      dataset =
        %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
        |> Data.precompute_frame_embeddings(
          use_prev_action: true,
          prev_action_dropout: 0.0,
          show_progress: false
        )

      indices = Data.sequence_target_indices(dataset, 16)

      unless length(indices) == run["targets"] and hd(indices) == context,
        do: raise("lost early targets")

      windows =
        Data.batched_sequences(dataset,
          lazy: true,
          gpu: false,
          shuffle: false,
          window_size: 16,
          batch_size: 64
        )
        |> Enum.flat_map(&Nx.to_list(&1.states))
        |> Enum.take(18)

      {:ok, agent} =
        PolicyAgent.start_link(
          policy_path: policy,
          af_convention: :parsed,
          harness: :scenario_suite,
          reaction_delay: 2
        )

      differences =
        Enum.map(Enum.take(shifted, context + 18), fn frame ->
          :ok = PolicyAgent.observe(agent, frame.game_state, frame.controller, player_port: 1)

          if frame.game_state.frame >= run["handoff"] do
            buffer =
              :sys.get_state(agent).frame_buffer
              |> :queue.to_list()
              |> Enum.map(&Nx.to_flat_list/1)

            padded = List.duplicate(hd(buffer), max(16 - length(buffer), 0)) ++ buffer
            index = frame.game_state.frame - run["handoff"]
            expected = Enum.at(windows, index) |> List.flatten()
            actual = List.flatten(padded)
            unless length(actual) == length(expected), do: raise("window shape mismatch")

            difference =
              Enum.zip(actual, expected)
              |> Enum.map(fn {left, right} -> abs(left - right) end)
              |> Enum.max()

            if difference > 1.0e-6 do
              IO.inspect(%{maximum_absolute_difference: difference})

              IO.inspect(
                Enum.zip(actual, expected)
                |> Enum.with_index()
                |> Enum.filter(fn {{left, right}, _} -> abs(left - right) > 1.0e-6 end)
                |> Enum.take(10)
              )

              raise("Agent/training window mismatch #{run["handoff"]} #{mode} index#{index}")
            end

            difference
          end
        end)

      GenServer.stop(agent)

      envelope =
        RecordedFrames.envelope([list], %{
          source_validation: source,
          source_sha256: hash,
          history: mode
        })

      RecordedFrames.validate!(envelope)
      path = Path.join(out, "#{run["handoff"]}_#{mode}.frames")
      File.write!(path, :erlang.term_to_binary(envelope, [:compressed]), [:exclusive])

      %{
        handoff: run["handoff"],
        history: mode,
        context_frames: context,
        targets: length(indices),
        first_target_frame: Enum.at(shifted, hd(indices)).game_state.frame,
        agent_windows_within_tolerance: 18,
        absolute_tolerance: 1.0e-6,
        maximum_absolute_difference: differences |> Enum.reject(&is_nil/1) |> Enum.max(),
        source_replay: run["replay"],
        source_sha256: hash,
        export: path
      }
    end
  end
  |> List.flatten()

File.write!(
  Path.join(out, "report.json"),
  Jason.encode!(
    %{window: 16, queue_depth: 3, delay: 2, augmentation_dropout: 0.0, results: results},
    pretty: true
  ),
  [:exclusive]
)
