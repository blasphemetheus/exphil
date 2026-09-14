alias ExPhil.Training.{Checkpoint, Data, Labels, RecordedFrames, Utils}
alias ExPhil.Training.Imitation.Loss
alias ExPhil.Networks.Policy
alias ExPhil.Eval.TeacherFit
alias ExPhil.Data.Peppi

Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
path = "eval_runs/0913_early_prefix_fit/round21/candidate.bin"
{:ok, export} = Checkpoint.load_policy(path)
{:ok, replay} = Peppi.parse("test/fixtures/replays/fox_multishine_closed_d1.slp")

canonical =
  replay
  |> Peppi.to_training_frames()
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: controller} ->
    controller.main_stick.x < 0.25 and controller.main_stick.y > 0.4 and
      not controller.button_b and not controller.button_x
  end)

teachers =
  Path.wildcard("eval_runs/0913_teacher_ingestion/validated/*.frames")
  |> Enum.flat_map(fn source ->
    RecordedFrames.load!(source)
    |> Enum.map(&{"#{Path.basename(source, ".frames")}:#{hd(&1).game_state.frame}", &1})
  end)

cases =
  Enum.map([{"canonical", canonical} | teachers], fn {name, frames} ->
    {name, Labels.at_delay(frames, 2) |> Enum.map(&Map.put(&1, :delay_id, 2))}
  end)

embed = fn lists ->
  dataset = Data.from_frame_lists(lists)

  %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
  |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)
end

dataset = embed.(Enum.map(cases, &elem(&1, 1)))
unless dataset.size == 7785, do: raise("wrong pool")

metadata =
  for {name, frames} <- cases,
      {frame, index} <- Enum.with_index(frames),
      do: %{
        case: name,
        index: index,
        frame: frame.game_state.frame,
        action: frame.game_state.players[1].action,
        action_frame: frame.game_state.players[1].action_frame
      }

batches =
  Data.batched_sequences(dataset,
    lazy: true,
    gpu: false,
    shuffle: false,
    window_size: 16,
    batch_size: 64
  )
  |> Enum.to_list()

states = Enum.map(batches, & &1.states) |> Nx.concatenate()
heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

actions =
  Map.new(heads, fn head -> {head, Enum.map(batches, & &1.actions[head]) |> Nx.concatenate()} end)

action_rows =
  Enum.map(heads, &Nx.to_list(actions[&1]))
  |> Enum.zip()
  |> Enum.map(&(Enum.zip(heads, Tuple.to_list(&1)) |> Map.new()))

all_windows = states |> Nx.to_batched(1) |> Enum.map(&Nx.to_binary/1)
hashes = Enum.map(all_windows, &:crypto.hash(:sha256, &1))
model = Policy.build_temporal(Map.to_list(export.config))
policy = Axon.MixedPrecision.create_policy(params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32})

bf16_model =
  Axon.MixedPrecision.apply_policy(model, policy, except: [:batch_norm, :layer_norm, :group_norm])

{_, f32} = Utils.build_compiled(model, mode: :inference)
{_, bf16} = Utils.build_compiled(bf16_model, mode: :inference)
params = Utils.ensure_model_state(export.params)

measure = fn predict, batch_states, batch_actions, row, precision ->
  inputs =
    Loss.policy_forward_inputs(
      :autoregressive,
      true,
      Nx.as_type(batch_states, precision),
      batch_actions
    )

  logits = predict.(params, inputs) |> Tuple.to_list()

  values =
    Enum.map(logits, &(Nx.backend_copy(&1, Nx.BinaryBackend) |> Nx.to_list() |> Enum.at(row)))

  targets = Enum.map(heads, &(Nx.to_list(batch_actions[&1]) |> Enum.at(row)))
  TeacherFit.row(Map.new(Enum.zip(heads, values)), Map.new(Enum.zip(heads, targets)))
end

targets = [{"recovery:4", 0}, {"sustain:900", 1}, {"sustain:900", 3}]

results =
  Enum.map(targets, fn {name, index} ->
    global = Enum.find_index(metadata, &(&1.case == name and &1.index == index))
    target = Nx.slice_along_axis(states, global, 1, axis: 0)

    target_actions =
      Map.new(actions, fn {head, tensor} ->
        {head, Nx.slice_along_axis(tensor, global, 1, axis: 0)}
      end)

    target_action = Enum.at(action_rows, global)
    target_hash = Enum.at(hashes, global)

    exact =
      Enum.with_index(hashes)
      |> Enum.filter(&(elem(&1, 0) == target_hash))
      |> Enum.map(fn {_, candidate} ->
        Map.merge(Enum.at(metadata, candidate), %{
          same_label: Enum.at(action_rows, candidate) == target_action
        })
      end)

    distances =
      Nx.subtract(states, target) |> Nx.pow(2) |> Nx.mean(axes: [1, 2]) |> Nx.to_flat_list()

    neighbors =
      distances
      |> Enum.with_index()
      |> Enum.reject(&(elem(&1, 1) == global))
      |> Enum.sort()
      |> Enum.take(5)
      |> Enum.map(fn {distance, candidate} ->
        Map.merge(Enum.at(metadata, candidate), %{
          mean_squared_distance: distance,
          target: Enum.at(action_rows, candidate)
        })
      end)

    frames = cases |> Enum.find(&(elem(&1, 0) == name)) |> elem(1)

    isolated =
      embed.([frames])
      |> Data.batched_sequences(
        lazy: true,
        gpu: false,
        shuffle: false,
        window_size: 16,
        batch_size: 64
      )
      |> Enum.at(0)

    isolated_target = Nx.slice_along_axis(isolated.states, index, 1, axis: 0)

    difference =
      Nx.subtract(target, isolated_target) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    pooled = Enum.at(batches, div(global, 64))
    repeated_states = Nx.broadcast(target, {64, 16, export.config.embed_size})

    repeated_actions =
      Map.new(target_actions, fn {head, tensor} ->
        shape = put_elem(Nx.shape(tensor), 0, 64)
        {head, Nx.broadcast(tensor, shape)}
      end)

    variants =
      Map.new(
        [
          {"isolated_f32", f32, isolated.states, isolated.actions, index, :f32},
          {"pool_f32", f32, pooled.states, pooled.actions, rem(global, 64), :f32},
          {"single_f32", f32, target, target_actions, 0, :f32},
          {"repeated_f32", f32, repeated_states, repeated_actions, 0, :f32},
          {"bf16_input_only", f32, isolated.states, isolated.actions, index, :bf16},
          {"training_precision", bf16, isolated.states, isolated.actions, index, :bf16}
        ],
        fn {variant, predict, input, labels, row, precision} ->
          {variant, measure.(predict, input, labels, row, precision)}
        end
      )

    result = %{
      case: name,
      index: index,
      global_index: global,
      exact_matches: exact,
      nearest_windows: neighbors,
      pool_vs_isolated_max_difference: difference,
      variants: variants
    }

    IO.inspect(result, label: name, limit: :infinity)
    result
  end)

teacher_results =
  Enum.map(Enum.drop(cases, 1), fn {name, frames} ->
    rows =
      embed.([frames])
      |> Data.batched_sequences(
        lazy: true,
        gpu: false,
        shuffle: false,
        window_size: 16,
        batch_size: 64
      )
      |> Enum.flat_map(fn batch ->
        logits =
          bf16.(
            params,
            Loss.policy_forward_inputs(
              :autoregressive,
              true,
              Nx.as_type(batch.states, :bf16),
              batch.actions
            )
          )

        values =
          logits
          |> Tuple.to_list()
          |> Enum.map(&(Nx.backend_copy(&1, Nx.BinaryBackend) |> Nx.to_list()))
          |> Enum.zip()

        targets = Enum.map(heads, &Nx.to_list(batch.actions[&1])) |> Enum.zip()

        Enum.zip(values, targets)
        |> Enum.map(fn {prediction, labels} ->
          TeacherFit.row(
            Map.new(Enum.zip(heads, Tuple.to_list(prediction))),
            Map.new(Enum.zip(heads, Tuple.to_list(labels)))
          )
        end)
      end)

    %{
      case: name,
      all: TeacherFit.summarize(rows),
      first18: TeacherFit.summarize(Enum.take(rows, 18)),
      early_minimum_probability:
        Enum.take(rows, 18) |> Enum.map(& &1.target_probability) |> Enum.min()
    }
  end)

report = %{
  policy: path,
  checkpoint_has_precision: Map.has_key?(export.config, :precision),
  note:
    "Same fixed params, no gradients. Training uses inference mode with BF16 states and Axon mixed-precision policy; deployed reconstruction is F32.",
  targets: results,
  training_precision_teacher_fit: teacher_results
}

File.write!(out, Jason.encode!(report, pretty: true), [:exclusive])
