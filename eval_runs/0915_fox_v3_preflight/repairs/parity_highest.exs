Nx.Defn.default_options(compiler: EXLA, precision: :highest)
alias ExPhil.Evaluation.Forward
alias ExPhil.Networks.Policy
alias ExPhil.Training.Utils

[path] = System.argv()
artifact = Forward.load!(path)
checkpoint_path = String.replace_suffix(path, "_policy.bin", ".axon")
checkpoint = Forward.load!(checkpoint_path)
export_matches = Nx.serialize(Utils.ensure_model_state(checkpoint.params).data) ==
  Nx.serialize(Utils.ensure_model_state(artifact.params).data)
unless export_matches, do: raise("Exported parameters differ from the matching training checkpoint")
config = artifact.config
params = Utils.ensure_model_state(artifact.params)
params = %{params | data: Nx.backend_copy(params.data, Nx.default_backend())}
evaluator = Forward.new(params, config)
steps = 83
{frames, _} = Nx.Random.normal(Nx.Random.key(915), shape: {1, steps, config.embed_size})

initial =
  Edifice.Recurrent.init_state(params.data,
    batch_size: 1,
    hidden_size: config.hidden_size,
    num_layers: config.num_layers,
    cell_type: :gru
  )

step = Edifice.Stateful.jit_step(Edifice.Recurrent, EXLA)

{samples, state} =
  Enum.map_reduce(0..(steps - 1), initial, fn t, state ->
    frame = frames |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
    {features, state} = step.(params.data, state, frame)

    sample =
      Policy.sample_autoregressive_from_features(params, features,
        deterministic: false,
        temperature: 1.0,
        key: Nx.Random.key(1000 + t)
      )

    {sample, state}
  end)

heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

actions =
  Map.new(heads, fn head ->
    {head, Nx.stack(Enum.map(samples, & &1[head]), axis: 1)}
  end)

batch = %{states: frames, actions: actions, is_resetting: Nx.tensor([1], type: :u8)}
{logits, _, final} = Forward.batch(evaluator, batch)
delta = fn a, b -> Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() end

deltas =
  Enum.zip(heads, Tuple.to_list(logits))
  |> Map.new(fn {head, logits} ->
    sequential = Nx.concatenate(Enum.map(samples, & &1.logits[head]), axis: 0)
    {head, delta.(logits, sequential)}
  end)

deltas = Map.put(deltas, :carry, delta.(final.carry, state.h))
# Cross an ordinary unroll boundary without resetting, including a short tail.
{chunks, carried} =
  Enum.map_reduce([{0, 80}, {80, 3}], evaluator, fn {start, length}, ev ->
    chunk = %{
      states: Nx.slice_along_axis(frames, start, length, axis: 1),
      actions:
        Map.new(actions, fn {k, v} -> {k, Nx.slice_along_axis(v, start, length, axis: 1)} end),
      is_resetting: Nx.tensor([if(start == 0, do: 1, else: 0)], type: :u8)
    }

    {out, _, ev} = Forward.batch(ev, chunk)
    {out, ev}
  end)

chunk_delta =
  Enum.with_index(Tuple.to_list(logits))
  |> Enum.map(fn {full, i} ->
    split = Nx.concatenate(Enum.map(chunks, &elem(&1, i)), axis: 0)
    delta.(full, split)
  end)
  |> Enum.max()

deltas =
  deltas
  |> Map.put(:chunk_logits, chunk_delta)
  |> Map.put(:chunk_carry, delta.(carried.carry, final.carry))

passed = Enum.all?(deltas, fn {_, d} -> is_number(d) and d < 1.0e-4 end)

report = %{
  export_matches_checkpoint: export_matches,
  checkpoint: checkpoint_path,
  policy_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower),
  policy: path,
  steps: steps,
  tolerance: 1.0e-4,
  passed: passed,
  maximum_absolute_deltas: deltas,
  config: Map.take(config, [:hidden_size, :num_layers, :head, :bptt, :execution_contract])
}

File.write!(Path.join(Path.dirname(path), "parity_highest.json"), Jason.encode!(report, pretty: true))
IO.inspect(report, label: "TRAINED STATEFUL / SAMPLED AR / CHUNK PARITY")
unless passed, do: raise("Trained inference parity failed")
