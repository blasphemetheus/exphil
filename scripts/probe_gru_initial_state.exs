alias ExPhil.Training.{Checkpoint, Data, Labels, RecordedFrames, Utils}
alias ExPhil.Networks.Policy
alias ExPhil.Training.Imitation.Loss
Nx.global_default_backend(Nx.BinaryBackend)
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
path = "eval_runs/0913_early_prefix_fit/round21/candidate.bin"
{:ok, export} = Checkpoint.load_policy(path)

for module <- [
      ExPhil.Bridge.GameState,
      ExPhil.Bridge.Player,
      ExPhil.Bridge.ControllerState,
      ExPhil.Bridge.Projectile
    ],
    do: Code.ensure_loaded!(module)

model = Policy.build_temporal(Map.to_list(export.config))
nodes = Enum.filter(Map.values(model.nodes), &(&1.op_name == :recurrent_state))
unless length(nodes) == 2, do: raise("expected two GRU initial-state nodes")
hidden_size = export.config.hidden_size

zero_model =
  Axon.map_nodes(model, fn node ->
    if node.op_name == :recurrent_state do
      %{
        node
        | op: fn input, _key, _opts ->
            Nx.broadcast(0.0, {Nx.axis_size(input, 0), hidden_size})
          end
      }
    else
      node
    end
  end)

{_, legacy} = Utils.build_compiled(model, mode: :inference)
{_, zero} = Utils.build_compiled(zero_model, mode: :inference)
params = Utils.ensure_model_state(export.params)

results =
  Enum.map([{"recovery", 4, 0}, {"sustain", 900, 1}, {"sustain", 900, 3}], fn {category, frame,
                                                                               index} ->
    frames =
      RecordedFrames.load!("eval_runs/0913_teacher_ingestion/validated/#{category}.frames")
      |> Enum.find(&(hd(&1).game_state.frame == frame))
      |> Labels.at_delay(2)
      |> Enum.map(&Map.put(&1, :delay_id, 2))

    dataset = Data.from_frame_lists([frames])

    dataset =
      %{dataset | embed_config: %{dataset.embed_config | queue_depth: 3, with_delay_id: true}}
      |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

    batch =
      Data.batched_sequences(dataset,
        lazy: true,
        shuffle: false,
        gpu: false,
        window_size: 16,
        batch_size: 64
      )
      |> Enum.at(0)

    target = Nx.slice_along_axis(batch.states, index, 1, axis: 0)

    variants =
      Map.new([{"legacy", legacy}, {"zero_counterfactual", zero}], fn {name, predict} ->
        sizes =
          Map.new([1, 64], fn size ->
            states = Nx.broadcast(target, {size, 16, export.config.embed_size})

            actions =
              Map.new(batch.actions, fn {head, tensor} ->
                sliced = Nx.slice_along_axis(tensor, index, 1, axis: 0)
                {head, Nx.broadcast(sliced, put_elem(Nx.shape(sliced), 0, size))}
              end)

            logits =
              predict.(params, Loss.policy_forward_inputs(:autoregressive, true, states, actions))
              |> elem(0)

            x_logits =
              Nx.slice_along_axis(logits, 2, 1, axis: 1)
              |> Nx.backend_copy(Nx.BinaryBackend)
              |> Nx.to_flat_list()

            probabilities =
              Enum.map(x_logits, fn logit ->
                if logit >= 0,
                  do: 1 / (1 + :math.exp(-logit)),
                  else: :math.exp(logit) / (1 + :math.exp(logit))
              end)

            {size,
             %{
               min_x_probability: Enum.min(probabilities),
               max_x_probability: Enum.max(probabilities),
               x_argmax_pressed_rows: Enum.count(probabilities, &(&1 >= 0.5)),
               probabilities: probabilities
             }}
          end)

        {name, sizes}
      end)

    stable = variants["zero_counterfactual"]

    result = %{
      case: "#{category}:#{frame}",
      index: index,
      variants: variants,
      zero_within_batch_spread: stable[64].max_x_probability - stable[64].min_x_probability,
      zero_between_batch_difference:
        abs(stable[1].min_x_probability - stable[64].min_x_probability)
    }

    IO.inspect(result, limit: :infinity)
    result
  end)

File.write!(
  out,
  Jason.encode!(
    %{
      policy: path,
      replaced_initial_state_nodes: length(nodes),
      note:
        "Diagnostic graph-only zero-state counterfactual. No parameters or exported policies changed; not a new qualified model.",
      cases: results
    },
    pretty: true
  ),
  [:exclusive]
)
