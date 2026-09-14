alias ExPhil.Training.{Imitation, Checkpoint, Utils}
alias ExPhil.Networks.Policy
alias ExPhil.Networks.Policy.ExecutionContract
alias ExPhil.Agents.Agent, as: PolicyAgent
{opts, [], []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists")
path = out <> ".policy.bin"
if File.exists?(path), do: raise("temporary checkpoint exists")

trainer =
  Imitation.new(
    temporal: true,
    backbone: :gru,
    hidden_size: 16,
    num_layers: 2,
    window_size: 16,
    head: :autoregressive,
    precision: :f32,
    recurrent_state: :zeros,
    dropout: 0.0
  )

:ok = Imitation.export_policy(trainer, path)
{:ok, export} = Checkpoint.load_policy(path)
contract = ExecutionContract.load(export.config)
unless contract.execution_contract == :windowed_gru_f32_v1, do: raise("missing contract")
model = Policy.build_temporal(Map.to_list(export.config))

refute_random = fn model ->
  if Enum.any?(Map.values(model.nodes), &(&1.op_name == :recurrent_state)),
    do: raise("random initial state remains")
end

refute_random.(model)
{_, reloaded} = Utils.build_compiled(model, mode: :inference)
{_, train_mode} = Utils.build_compiled(model, mode: :train)
params = Utils.ensure_model_state(export.params)
width = export.config.embed_size
single = Nx.iota({1, 16, width}, type: :f32) |> Nx.remainder(11) |> Nx.divide(11)

actions = %{
  buttons: Nx.broadcast(0, {1, 8}),
  main_x: Nx.tensor([8]),
  main_y: Nx.tensor([8]),
  c_x: Nx.tensor([8]),
  c_y: Nx.tensor([8]),
  shoulder: Nx.tensor([0])
}

inputs = Imitation.Loss.policy_forward_inputs(:autoregressive, true, single, actions)
expected = trainer.predict_fn.(trainer.policy_params, inputs)
actual = reloaded.(params, inputs)
trained = train_mode.(params, inputs).prediction

max_difference = fn left, right ->
  Enum.zip(Tuple.to_list(left), Tuple.to_list(right))
  |> Enum.map(fn {before, after_value} ->
    Nx.subtract(before, after_value) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end)
  |> Enum.max()
end

reload_difference = max_difference.(expected, actual)
mode_difference = max_difference.(actual, trained)
copies = Nx.broadcast(single, {64, 16, width})

copied_actions =
  Map.new(actions, fn {head, value} ->
    {head, Nx.broadcast(value, put_elem(Nx.shape(value), 0, 64))}
  end)

batch =
  reloaded.(
    params,
    Imitation.Loss.policy_forward_inputs(:autoregressive, true, copies, copied_actions)
  )

expanded =
  actual
  |> Tuple.to_list()
  |> Enum.map(&Nx.broadcast(&1, put_elem(Nx.shape(&1), 0, 64)))
  |> List.to_tuple()

batch_difference = max_difference.(expanded, batch)

unless reload_difference < 1.0e-5 and mode_difference < 1.0e-5 and batch_difference < 5.0e-3,
  do:
    raise(
      "forward parity failed: #{inspect({reload_difference, mode_difference, batch_difference})}"
    )

{:ok, agent} = PolicyAgent.start_link(policy_path: path)
state = :sys.get_state(agent)
features = state.predict_fn.(state.policy_params, single)

sample =
  Policy.sample_autoregressive_from_features(state.policy_params, features, deterministic: true)

sample_actions = Map.take(sample, [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder])

sample_logits =
  reloaded.(
    params,
    Imitation.Loss.policy_forward_inputs(:autoregressive, true, single, sample_actions)
  )

heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]
sequential_logits = heads |> Enum.map(&Map.fetch!(sample.logits, &1)) |> List.to_tuple()
sequential_difference = max_difference.(sample_logits, sequential_logits)

unless sequential_difference < 1.0e-3,
  do: raise("sequential logits differ: #{sequential_difference}")

for {head, logits} <- Enum.zip(heads, Tuple.to_list(sample_logits)) do
  predicted =
    if head == :buttons, do: Nx.greater(logits, 0), else: Nx.argmax(logits, axis: -1)

  unless Nx.all(Nx.equal(predicted, sample_actions[head])) |> Nx.to_number() == 1,
    do: raise("sequential sampler disagrees at #{head}")
end

GenServer.stop(agent)

File.write!(
  out,
  Jason.encode!(
    %{
      contract: contract,
      reload_max_difference: reload_difference,
      train_mode_max_difference: mode_difference,
      batch1_vs64_max_difference: batch_difference,
      batch_size_absolute_tolerance: 5.0e-3,
      sequential_head_argmax_matches: true,
      sequential_logits_max_difference: sequential_difference,
      optimization_steps: 0
    },
    pretty: true
  ),
  [:exclusive]
)

File.rm!(path)
