alias ExPhil.Training.{Data, Labels, RecordedFrames, RecordedContext}
alias ExPhil.Data.Peppi
alias ExPhil.Agents.Agent, as: PolicyAgent
Nx.global_default_backend(Nx.BinaryBackend)
# Export validated teacher windows as cold/warm recorded clips and audit the
# first 18 supervised windows against the actual Agent's frame buffer.
#
#   --out-dir DIR         exclusive-create
#   --delay k             reaction delay the clips are shifted/audited at (2)
#   --queue-depth q       committed-action queue depth (default k + 1)
#   --context c           warm input-only prefix length (default window 16 - 1 + q)
#   --reports a,b,c       check_recovery_targets reports (default: the four
#                         delay-2 reports of the 09-13 proof); each report's
#                         own `reaction_delay` recovers the raw response length
#   --tag-source          name clips HANDOFF_<sha6>_MODE.frames (handoff frames
#                         collide across replays; default names keep clips_v6
#                         reproducible)
#   --policy PATH|none    Agent used for the window-parity audit; `none` skips
#                         the audit (report says so) — use it to export clips
#                         for a delay no trained policy exists at yet, then
#                         re-run with the trained candidate and cmp the files
{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      out_dir: :string,
      delay: :integer,
      queue_depth: :integer,
      context: :integer,
      reports: :string,
      policy: :string,
      tag_source: :boolean
    ]
  )

out = Keyword.fetch!(opts, :out_dir)
if File.exists?(out), do: raise("output directory exists")
File.mkdir_p!(out)
window = 16
delay = opts[:delay] || 2
queue_depth = opts[:queue_depth] || delay + 1
warm_context = opts[:context] || window - 1 + queue_depth
if delay < 0 or queue_depth < 1 or warm_context < 1, do: raise("invalid delay/queue/context")

reports =
  case opts[:reports] do
    nil ->
      [
        "eval_runs/0913_teacher_ingestion/validated/neutral_targets.json",
        "eval_runs/0913_teacher_ingestion/validated/sustain_targets.json",
        "eval_runs/0913_teacher_ingestion/validated/recovery_targets.json",
        "eval_runs/0913_context_recovery/teacher/validation_with_on_loop.json"
      ]

    list ->
      String.split(list, ",", trim: true)
  end

policy =
  case opts[:policy] || "eval_runs/0913_no_dropout_fit/round21/candidate.bin" do
    "none" -> nil
    path -> path
  end

results =
  for source <- reports,
      report = File.read!(source) |> Jason.decode!(),
      source_delay = report["reaction_delay"] || 2,
      run <- report["runs"] do
    unless run["valid"], do: raise("unvalidated teacher")
    hash = :crypto.hash(:sha256, File.read!(run["replay"])) |> Base.encode16(case: :lower)
    unless hash == run["replay_sha256"], do: raise("teacher replay hash mismatch")
    {:ok, replay} = Peppi.parse(run["replay"])
    frames = Peppi.to_training_frames(replay)
    # the report counts targets at ITS delay; the raw response is delay-free
    response_length = run["targets"] + source_delay
    targets = response_length - delay

    for mode <- [:cold, :warm] do
      context = if mode == :warm, do: warm_context, else: 0
      list = RecordedContext.slice(frames, run["handoff"], response_length, context)
      shifted = Labels.at_delay(list, delay) |> Enum.map(&Map.put(&1, :delay_id, delay))
      dataset = Data.from_frame_lists([shifted])

      dataset =
        %{dataset | embed_config: %{dataset.embed_config | queue_depth: queue_depth, with_delay_id: true}}
        |> Data.precompute_frame_embeddings(
          use_prev_action: true,
          prev_action_dropout: 0.0,
          show_progress: false
        )

      indices = Data.sequence_target_indices(dataset, 16)

      unless length(indices) == targets and hd(indices) == context,
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

      agent =
        if policy do
          {:ok, agent} =
            PolicyAgent.start_link(
              policy_path: policy,
              af_convention: :parsed,
              harness: :scenario_suite,
              reaction_delay: delay
            )

          agent
        end

      differences =
        if agent == nil, do: [], else:
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

      if agent, do: GenServer.stop(agent)

      envelope =
        RecordedFrames.envelope([list], %{
          source_validation: source,
          source_sha256: hash,
          history: mode
        })

      RecordedFrames.validate!(envelope)
      tag = if opts[:tag_source], do: "_" <> String.slice(hash, 0, 6), else: ""
      path = Path.join(out, "#{run["handoff"]}#{tag}_#{mode}.frames")
      File.write!(path, :erlang.term_to_binary(envelope, [:compressed]), [:exclusive])

      %{
        handoff: run["handoff"],
        history: mode,
        context_frames: context,
        targets: length(indices),
        first_target_frame: Enum.at(shifted, hd(indices)).game_state.frame,
        agent_parity: if(policy, do: :audited, else: :skipped),
        agent_windows_within_tolerance: if(policy, do: 18, else: 0),
        absolute_tolerance: 1.0e-6,
        maximum_absolute_difference: differences |> Enum.reject(&is_nil/1) |> Enum.max(fn -> nil end),
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
    %{window: window, queue_depth: queue_depth, delay: delay, warm_context: warm_context,
      augmentation_dropout: 0.0, parity_policy: policy, reports: reports, results: results},
    pretty: true
  ),
  [:exclusive]
)
