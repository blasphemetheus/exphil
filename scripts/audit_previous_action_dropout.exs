alias ExPhil.Training.{Checkpoint, Data, Labels, RecordedFrames, Utils}
alias ExPhil.Training.Imitation.Loss
alias ExPhil.Networks.Policy
alias ExPhil.Eval.TeacherFit
alias ExPhil.Data.Peppi

Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [report: :string, out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
report = opts |> Keyword.fetch!(:report) |> File.read!() |> Jason.decode!()
path = report["policy"]
digest = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
unless digest == report["policy_sha256"], do: raise("checkpoint hash mismatch")
{:ok, export} = Checkpoint.load_policy(path)
Policy.ExecutionContract.verify_report!(export.config, report)
{:ok, replay} = Peppi.parse("test/fixtures/replays/fox_multishine_closed_d1.slp")

canonical =
  replay
  |> Peppi.to_training_frames()
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: controller} ->
    controller.main_stick.x < 0.25 and controller.main_stick.y > 0.4 and
      not controller.button_b and not controller.button_x
  end)
  |> Labels.tag(:recorded)

teachers =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(fn source ->
    RecordedFrames.load!(source)
    |> Enum.map(&{"#{Path.basename(source, ".frames")}:#{hd(&1).game_state.frame}", &1})
  end)

cases = Map.new([{"canonical", canonical} | teachers])

{_, predict} =
  Policy.build_temporal(Map.to_list(export.config)) |> Utils.build_compiled(mode: :inference)

params = Utils.ensure_model_state(export.params)
heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

results =
  for entry <- report["cases"], row <- entry["rows"], row["tf_argmax_correct"] == false do
    index = row["index"]
    unless index < 18, do: raise("audit is restricted to early prefix failures")

    frames =
      cases
      |> Map.fetch!(entry["case"])
      |> Labels.at_delay(2)
      |> Enum.map(&Map.put(&1, :delay_id, 2))
      |> Enum.take(18)

    variants =
      [{"no_dropout", 0.0, 1}, {"no_dropout_other_seed", 0.0, 99}, {"all_slots_absent", 1.0, 1}] ++
        Enum.map(1..32, &{"dropout_seed_#{&1}", 0.1, &1})

    inputs =
      Enum.map(variants, fn {name, dropout, seed} ->
        :rand.seed(:exsss, {seed, seed + 1, seed + 2})
        dataset = Data.from_frame_lists([frames])

        dataset =
          %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
          |> Data.precompute_frame_embeddings(
            use_prev_action: true,
            prev_action_dropout: dropout,
            show_progress: false
          )

        batch =
          Data.batched_sequences(dataset,
            lazy: true,
            gpu: false,
            shuffle: false,
            window_size: 16,
            batch_size: 64
          )
          |> Enum.at(0)

        %{
          name: name,
          dropout: dropout,
          states: Nx.slice_along_axis(batch.states, index, 1, axis: 0),
          actions:
            Map.new(batch.actions, fn {head, value} ->
              {head, Nx.slice_along_axis(value, index, 1, axis: 0)}
            end)
        }
      end)

    baseline = hd(inputs)

    unless Nx.to_binary(baseline.states) == Nx.to_binary(Enum.at(inputs, 1).states),
      do: raise("dropout zero depends on RNG")

    for input <- inputs, head <- heads do
      unless Nx.to_binary(input.actions[head]) == Nx.to_binary(baseline.actions[head]),
        do: raise("dropout changed targets")
    end

    states = inputs |> Enum.map(& &1.states) |> Nx.concatenate()

    actions =
      Map.new(heads, fn head ->
        {head, inputs |> Enum.map(& &1.actions[head]) |> Nx.concatenate()}
      end)

    logits =
      predict.(params, Loss.policy_forward_inputs(:autoregressive, true, states, actions))
      |> Tuple.to_list()
      |> Enum.map(&(Nx.backend_transfer(&1, Nx.BinaryBackend) |> Nx.to_list()))

    measurements =
      for {input, position} <- Enum.with_index(inputs) do
        prediction = Map.new(Enum.zip(heads, Enum.map(logits, &Enum.at(&1, position))))
        targets = Map.new(heads, &{&1, input.actions[&1] |> Nx.to_list() |> hd()})
        metric = TeacherFit.row(prediction, targets)

        %{
          name: input.name,
          dropout: input.dropout,
          correct: metric.tf_argmax_correct,
          target_probability: metric.target_probability,
          x_probability: Enum.at(metric.button_probabilities, 2),
          changed_input_coordinates:
            Nx.not_equal(input.states, baseline.states) |> Nx.sum() |> Nx.to_number(),
          window_sha256:
            :crypto.hash(:sha256, Nx.to_binary(input.states)) |> Base.encode16(case: :lower)
        }
      end

    %{
      case: entry["case"],
      index: index,
      frame: row["frame"],
      action: row["action"],
      action_frame: row["action_frame"],
      target: row["target"],
      original_target_probability: row["target_probability"],
      variants: measurements
    }
  end

pool_cases =
  Enum.map(cases, fn {name, frames} ->
    {name, Labels.at_delay(frames, 2) |> Enum.map(&Map.put(&1, :delay_id, 2))}
  end)

metadata =
  for {name, frames} <- pool_cases,
      {frame, index} <- Enum.with_index(frames),
      do: %{case: name, index: index, frame: frame.game_state.frame}

dataset = Data.from_frame_lists(Enum.map(pool_cases, &elem(&1, 1)))

dataset =
  %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
  |> Data.precompute_frame_embeddings(
    use_prev_action: true,
    prev_action_dropout: 0.0,
    show_progress: false
  )

hashes = MapSet.new(results, &hd(&1.variants).window_sha256)

{matches, count} =
  Data.batched_sequences(dataset,
    lazy: true,
    gpu: false,
    shuffle: false,
    window_size: 16,
    batch_size: 64
  )
  |> Enum.reduce({[], 0}, fn batch, {matches, offset} ->
    found =
      batch.states
      |> Nx.to_batched(1)
      |> Enum.with_index()
      |> Enum.flat_map(fn {window, index} ->
        hash = :crypto.hash(:sha256, Nx.to_binary(window)) |> Base.encode16(case: :lower)

        if MapSet.member?(hashes, hash) do
          target =
            Map.new(heads, fn head ->
              value = batch.actions[head][index]
              {head, if(Nx.rank(value) == 0, do: Nx.to_number(value), else: Nx.to_list(value))}
            end)
            |> Jason.encode!()
            |> Jason.decode!()

          [Map.merge(Enum.at(metadata, offset + index), %{window_sha256: hash, target: target})]
        else
          []
        end
      end)

    {matches ++ found, offset + Nx.axis_size(batch.states, 0)}
  end)

unless count == 7785, do: raise("incorrect pool size")

results =
  Enum.map(results, fn result ->
    exact =
      Enum.filter(matches, &(&1.window_sha256 == hd(result.variants).window_sha256))
      |> Enum.map(&Map.put(&1, :same_target, &1.target == result.target))

    Map.put(result, :exact_unaugmented_pool_matches, exact)
  end)

File.write!(
  out,
  Jason.encode!(
    %{
      policy: path,
      policy_sha256: digest,
      historical_dropout_mask_available: false,
      interpretation:
        "Seeded counterfactual masks, not reconstruction of the historical training mask",
      optimization_steps: 0,
      results: results
    },
    pretty: true
  ),
  [:exclusive]
)
