# Replay a captured fatal BPTT step (CLAUDE.md "capture the crime scene").
#
#   mix run scripts/replay_fatal_batch.exs <run_dir>/fatal_batch_<step>.bin <run_dir>/model_config.json
#
# Rebuilds the trainer from the run's config, loads the PRE-update params and
# optimizer state from the capture, and reports:
#   1. input hygiene: non-finite counts and max |value| per batch tensor;
#   2. forward-only (eval) loss on the batch: NaN here = data/loss pathology;
#   3. per-row loss: which of the 128 rows blow up;
#   4. loss + gradient with the training function: NaN only here = gradient
#      pathology; max |grad| per parameter, and the clipped update norm.
# Run under EXPHIL_EXLA_PRECISION=highest (as trained) and again without it
# to see whether arithmetic matters.
alias ExPhil.Training.{Imitation, Utils}
alias ExPhil.Training.Output

[capture_path, config_path] = System.argv()
payload = capture_path |> File.read!() |> :erlang.binary_to_term()
config = config_path |> File.read!() |> Jason.decode!()
Output.puts("capture: step #{payload.step} batch_idx #{payload.batch_idx} epoch #{payload.epoch} loss #{inspect(payload.loss)}")

finite? = fn t -> Nx.logical_not(Nx.logical_or(Nx.is_nan(t), Nx.is_infinity(t))) end

tensors = fn
  %Nx.Tensor{} = t, acc -> [t | acc]
  m, acc when is_map(m) and not is_struct(m) -> Enum.reduce(m, acc, fn {_, v}, a -> if(is_struct(v, Nx.Tensor), do: [v | a], else: if(is_map(v) and not is_struct(v), do: Enum.reduce(v, a, fn {_, w}, b -> if(is_struct(w, Nx.Tensor), do: [w | b], else: b) end), else: a)) end)
  _, acc -> acc
end

leaves = fn m -> tensors.(m, []) end

des = fn
  %Nx.Tensor{} = t -> t
  v when is_list(v) or is_binary(v) -> Nx.deserialize(v)
  m when is_map(m) -> Nx.Container.traverse(m, nil, fn x, acc -> {Nx.deserialize(x), acc} end) |> elem(0)
  v -> v
end

batch = Map.new(payload.batch, fn {k, v} -> {k, des.(v)} end)
carry = des.(payload.carry)
params = des.(payload.policy_params)
opt_state = des.(payload.optimizer_state)

# 1. input hygiene
for {k, v} <- batch, is_struct(v, Nx.Tensor) do
  nf = Nx.to_number(Nx.sum(Nx.as_type(Nx.logical_or(Nx.is_nan(v), Nx.is_infinity(v)), :s64)))
  mx = if Nx.type(v) |> elem(0) == :f, do: Nx.to_number(Nx.reduce_max(Nx.abs(Nx.select(finite?.(v), v, 0.0)))), else: Nx.to_number(Nx.reduce_max(v))
  Output.puts("  #{k}: shape #{inspect(Nx.shape(v))} non-finite #{nf} max|v| #{mx}")
end
for {k, v} <- batch.actions do
  Output.puts("  actions.#{k}: shape #{inspect(Nx.shape(v))} min #{Nx.to_number(Nx.reduce_min(v))} max #{Nx.to_number(Nx.reduce_max(v))}")
end
nf_params = params |> leaves.() |> Enum.reduce(0, fn t, acc -> acc + Nx.to_number(Nx.sum(Nx.as_type(Nx.logical_or(Nx.is_nan(t), Nx.is_infinity(t)), :s64))) end)
Output.puts("  pre-update params non-finite: #{nf_params}")

# rebuild the trainer with the run's config
cfg = Map.new(config, fn {k, v} -> {String.to_atom(k), v} end)
trainer =
  Imitation.new(
    embed_size: cfg[:embed_size], temporal: true, bptt: true, backbone: :gru, head: String.to_atom(cfg[:head] || "autoregressive"),
    unroll: cfg[:unroll] || 80, hidden_size: cfg[:hidden_size] || hd(cfg[:hidden_sizes]), num_layers: cfg[:num_layers], precision: :f32,
    batch_size: Nx.axis_size(batch.states, 0), dropout: cfg[:dropout] || 0.1, learning_rate: cfg[:learning_rate] || 1.0e-4,
    max_grad_norm: cfg[:max_grad_norm] || 1.0, hidden_sizes: cfg[:hidden_sizes]
  )

params_state = %{Utils.ensure_model_state(trainer.policy_params) | data: params}
trainer = %{trainer | policy_params: params_state, optimizer_state: opt_state}

# 2/3. forward-only loss, whole batch and per row
{loss, _carry} = trainer.eval_loss_fn.(trainer.policy_params, batch.states, batch.actions, batch.frame_weights, carry)
Output.puts("forward-only loss (eval fn): #{inspect(Nx.to_number(loss))}")

rows = Nx.axis_size(batch.states, 0)
bad_rows =
  for r <- 0..(rows - 1), reduce: [] do
    acc ->
      sl = fn t -> Nx.slice_along_axis(t, r, 1, axis: 0) end
      {l, _} = trainer.eval_loss_fn.(trainer.policy_params, sl.(batch.states), Map.new(batch.actions, fn {k, v} -> {k, sl.(v)} end), sl.(batch.frame_weights), sl.(carry))
      if is_atom(Nx.to_number(l)), do: [r | acc], else: acc
  end
Output.puts("rows with non-finite forward loss: #{inspect(Enum.reverse(bad_rows))} (of #{rows})")

# 4. training loss + gradient
{{tloss, _}, grads} = trainer.loss_and_grad_fn.(trainer.policy_params, batch.states, batch.actions, batch.frame_weights, carry)
Output.puts("training loss (grad fn): #{inspect(Nx.to_number(tloss))}")
gdata = Utils.ensure_model_state(grads).data
worst =
  gdata |> leaves.() |> Enum.map(fn t -> Nx.to_number(Nx.reduce_max(Nx.abs(Nx.select(finite?.(t), t, 0.0)))) end)
nf_grads = gdata |> leaves.() |> Enum.reduce(0, fn t, acc -> acc + Nx.to_number(Nx.sum(Nx.as_type(Nx.logical_or(Nx.is_nan(t), Nx.is_infinity(t)), :s64))) end)
Output.puts("gradient: non-finite elements #{nf_grads}, max |grad| #{Enum.max(worst, fn -> 0 end)}")

report = %{capture: capture_path, step: payload.step, forward_loss: inspect(Nx.to_number(loss)), bad_rows: Enum.reverse(bad_rows), training_loss: inspect(Nx.to_number(tloss)), grad_nonfinite: nf_grads, grad_max: Enum.max(worst, fn -> 0 end), params_nonfinite: nf_params, precision: Nx.Defn.default_options()[:precision] || :default}
File.write!(Path.rootname(capture_path) <> "_replay.json", Jason.encode!(report, pretty: true))
Output.success("replay written to #{Path.rootname(capture_path)}_replay.json")
