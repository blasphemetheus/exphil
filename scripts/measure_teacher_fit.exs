alias ExPhil.Training.{Checkpoint, Data, Labels, RecordedFrames, Utils}
alias ExPhil.Training.Imitation.Loss
alias ExPhil.Networks.Policy
alias ExPhil.Eval.TeacherFit
alias ExPhil.Data.Peppi
alias ExPhil.Agents.MultishineExpert

# Frozen teacher-forced fit of the fixed delay-2 cold/warm GRU proof recipe.
#
#   --policy PATH             checkpoint to measure
#   --out PATH                report (exclusive-create)
#   --recorded-frames GLOB    recorded teacher clips (default: the six cold
#                             0913_teacher_ingestion/validated windows)
#   --expected-targets N      frozen pool size in SUPERVISED targets, canonical
#                             included (default 7,785 = canonical 7,077 + 6x118;
#                             the 18 cold/warm clips_v6 pool is 10,641)
#   --include-canonical-rows  keep the canonical fixture's per-row table
#
# Clips may carry an input-only prefix (`input_only: true`, RecordedContext):
# those frames feed history and the previous-action queue but are never
# targets. Every row index below is the SUPERVISED index (0 = the first
# response target), never a position in the raw clip; `frame_index` keeps
# the raw position. A clip with a prefix reports `history: :warm` and its
# `context_frames`; the first-18 summary and the gate index supervised targets.
{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      out: :string,
      include_canonical_rows: :boolean,
      recorded_frames: :string,
      expected_targets: :integer,
      delay: :integer,
      queue_depth: :integer
    ]
  )

path = Keyword.fetch!(opts, :policy)
out = Keyword.fetch!(opts, :out)
recorded_glob = opts[:recorded_frames] || "eval_runs/0913_teacher_ingestion/validated/*.frames"
expected_targets = opts[:expected_targets] || 7785
delay = opts[:delay] || 2
queue_depth = opts[:queue_depth] || delay + 1
if File.exists?(out), do: raise("output exists: #{out}")
{:ok, export} = Checkpoint.load_policy(path)
config = export.config
execution = ExPhil.Networks.Policy.ExecutionContract.load(config)
config = Map.put(config, :recurrent_state, execution.recurrent_state)

unless config.head == :autoregressive and config.backbone == :gru and config.train_delays == [delay] and
         config.window_size == 16 and config.queue_depth == queue_depth and config.use_prev_action and
         config.with_delay_id and config.action_frame_buckets == 0 and config.axis_buckets == 16 and
         config.shoulder_buckets == 4 and config.action_mode == :learned and
         config.character_mode == :learned and
         config.nana_mode == :compact and config.with_projectiles and not config.with_items and
         not config.stage_internals,
       do: raise("this audit implements the fixed windowed-GRU proof recipe only (delay #{delay}, queue #{queue_depth})")

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
  |> Labels.tag(:recorded)
  |> Labels.at_delay(delay)

recorded_paths = Path.wildcard(recorded_glob)
if recorded_paths == [], do: raise("--recorded-frames matched no files: #{recorded_glob}")

supervised? = fn frame -> frame[:input_only] != true end

teachers =
  recorded_paths
  |> Enum.flat_map(fn source ->
    RecordedFrames.load!(source)
    |> Enum.map(fn frames ->
      # Name by the first SUPERVISED frame (the handoff), not the first
      # context frame, so cold and warm exports of one handoff align.
      first = frames |> Enum.find(supervised?) |> then(& &1.game_state.frame)
      {"#{Path.basename(source, ".frames")}:#{first}", Labels.at_delay(frames, delay)}
    end)
  end)

cases = [{"canonical", canonical} | teachers]

pool_targets =
  cases |> Enum.map(fn {_, frames} -> Enum.count(frames, supervised?) end) |> Enum.sum()

unless pool_targets == expected_targets,
  do: raise("pool has #{pool_targets} supervised targets; --expected-targets is #{expected_targets}")

{_, predict} =
  Policy.build_temporal(Map.to_list(config)) |> Utils.build_compiled(mode: :inference)

params = Utils.ensure_model_state(export.params)
heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

results =
  Enum.map(cases, fn {name, frames} ->
    frames = Enum.map(frames, &Map.put(&1, :delay_id, delay))
    dataset = Data.from_frame_lists([frames])

    dataset =
      %{dataset | embed_config: %{dataset.embed_config | queue_depth: queue_depth, with_delay_id: true}}
      |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

    unless elem(Nx.shape(dataset.embedded_frames), 1) == config.embed_size,
      do: raise("embedding width mismatch")

    metrics =
      dataset
      |> Data.batched_sequences(lazy: true, shuffle: false, window_size: 16, batch_size: 64)
      |> Enum.flat_map(fn batch ->
        logits =
          predict.(
            params,
            Loss.policy_forward_inputs(:autoregressive, true, batch.states, batch.actions)
          )

        outputs =
          Tuple.to_list(logits)
          |> Enum.map(&(Nx.backend_copy(&1, Nx.BinaryBackend) |> Nx.to_list()))

        targets =
          Enum.map(
            heads,
            &(Nx.backend_copy(batch.actions[&1], Nx.BinaryBackend) |> Nx.to_list())
          )

        Enum.zip(Enum.zip(outputs), Enum.zip(targets))
        |> Enum.map(fn {prediction, target} ->
          TeacherFit.row(
            Map.new(Enum.zip(heads, Tuple.to_list(prediction))),
            Map.new(Enum.zip(heads, Tuple.to_list(target)))
          )
        end)
      end)

    # Raw positions of every SUPERVISED target (the boundary-aware layout
    # skips input-only context); the supervised index is its rank.
    indices = Data.sequence_target_indices(dataset, 16)
    context_frames = Enum.count(frames, &(not supervised?.(&1)))
    unless length(metrics) == length(indices), do: raise("metric/target count mismatch")

    rows =
      Enum.zip(metrics, indices)
      |> Enum.with_index()
      |> Enum.map(fn {{metric, frame_index}, index} ->
        frame = Enum.at(frames, frame_index)
        if not supervised?.(frame), do: raise("input-only frame #{frame_index} reached the loss")

        Map.merge(metric, %{
          index: index,
          frame_index: frame_index,
          frame: frame.game_state.frame,
          action: frame.game_state.players[1].action,
          action_frame: frame.game_state.players[1].action_frame,
          off_loop: not MultishineExpert.on_loop?(expert, frame.game_state.players[1])
        })
      end)

    unless length(rows) == length(frames) - context_frames, do: raise("lost targets")

    result = %{
      case: name,
      history: if(context_frames > 0, do: :warm, else: :cold),
      context_frames: context_frames,
      targets: length(rows),
      all: TeacherFit.summarize(rows),
      first18: TeacherFit.summarize(Enum.filter(rows, &(&1.index < 18))),
      off_loop: TeacherFit.summarize(Enum.filter(rows, & &1.off_loop)),
      rows: if(name == "canonical" and opts[:include_canonical_rows] != true, do: [], else: rows)
    }

    IO.inspect(Map.drop(result, [:rows]), label: name)
    result
  end)

report = %{
  execution_contract: execution,
  policy: path,
  policy_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower),
  recorded_frames: recorded_glob,
  recorded_sources:
    Map.new(recorded_paths, &{&1, Base.encode16(:crypto.hash(:sha256, File.read!(&1)), case: :lower)}),
  expected_targets: expected_targets,
  delay: delay,
  queue_depth: queue_depth,
  # Every recorded case is gated (cold AND warm); the canonical fixture is not.
  gate_cases: Enum.map(teachers, &elem(&1, 0)),
  index_convention: "row.index is the supervised-target rank (0 = handoff); input-only context is never a row",
  metric:
    "unweighted joint action NLL; AR teacher-forced conditional argmax, not free-running accuracy",
  cases: results
}

File.write!(out, Jason.encode!(report, pretty: true), [:exclusive])
