defmodule ExPhil.Evaluation.Forward do
  @moduledoc """
  Teacher-forced policy evaluation with explicit recurrent carry.

  BPTT batches must arrive in order and supply `is_resetting` for each row.
  Returned logits and targets are flattened for frame-level metrics. They
  measure teacher-forced likelihood, not free-running controller behavior.
  """

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.{Checkpoint, Utils}
  alias ExPhil.Training.Imitation.Loss

  defstruct [:params, :config, :predict_fn, :carry]

  def normalize(config) do
    config = Map.new(config)

    Enum.reduce(config, %{}, fn {source_key, value}, normalized ->
      key = known_key(source_key)
      value = if is_binary(source_key) and Map.has_key?(config, key), do: config[key], else: value

      value =
        cond do
          value == "true" ->
            true

          value == "false" ->
            false

          key in [
            :backbone,
            :head,
            :precision,
            :action_mode,
            :character_mode,
            :stage_mode,
            :nana_mode,
            :label_convention
          ] and is_binary(value) ->
            String.to_existing_atom(value)

          true ->
            value
        end

      if key, do: Map.put(normalized, key, value), else: normalized
    end)
  end

  defp known_key(key) when is_atom(key), do: key

  defp known_key(key) when is_binary(key) do
    String.to_existing_atom(key)
  rescue
    ArgumentError -> nil
  end

  def load!(path, opts \\ []) do
    loaded =
      if Path.extname(path) == ".axon",
        do: Checkpoint.load(path),
        else: Checkpoint.load_policy(path)

    artifact =
      case loaded do
        {:ok, artifact} -> artifact
        {:error, reason} -> raise ArgumentError, "Cannot load #{path}: #{inspect(reason)}"
      end

    sidecar =
      [String.replace(path, ~r/(_best_policy|_policy|_best)?\.(axon|bin)$/, "_config.json")]
      |> Enum.find_value(%{}, fn candidate ->
        if candidate != path and File.regular?(candidate),
          do: candidate |> File.read!() |> Jason.decode!() |> normalize()
      end)

    config = Map.merge(sidecar, normalize(artifact.config))

    config =
      if Keyword.get(opts, :bptt) == true, do: Map.put_new(config, :bptt, true), else: config

    %{path: path, params: artifact[:params] || artifact[:policy_params], config: config}
  end

  def new(params, config, opts \\ []) do
    config = normalize(config)
    build_opts = Map.to_list(config)

    model =
      cond do
        config[:bptt] ->
          Policy.build_temporal_bptt(Keyword.put(build_opts, :window_size, nil))

        config[:temporal] ->
          Policy.build_temporal(build_opts)

        true ->
          Policy.build(build_opts)
      end

    {_init, predict} = Utils.build_compiled(model, opts)
    %__MODULE__{params: Utils.ensure_model_state(params), config: config, predict_fn: predict}
  end

  def from_trainer(trainer) do
    %__MODULE__{
      params: Utils.ensure_model_state(trainer.policy_params),
      config: trainer.config,
      predict_fn: trainer.predict_fn
    }
  end

  def batch(evaluator, batch) do
    config = evaluator.config
    head = config[:head] || if(config[:bptt], do: :autoregressive, else: :independent)
    inputs = Loss.policy_forward_inputs(head, config[:temporal], batch.states, batch.actions)

    if config[:bptt] do
      batch_size = Nx.axis_size(batch.states, 0)
      shape = {batch_size, config[:num_layers] || 2, config[:hidden_size] || 256}
      carry = evaluator.carry || Nx.broadcast(0.0, shape)

      if Nx.shape(carry) != shape,
        do: raise(ArgumentError, "BPTT row count changed without resetting the evaluator")

      resetting = Map.fetch!(batch, :is_resetting)
      keep = resetting |> Nx.equal(0) |> Nx.reshape({batch_size, 1, 1}) |> Nx.broadcast(shape)
      carry = Nx.select(keep, carry, 0.0)

      inputs =
        if head == :independent, do: %{"state_sequence" => batch.states}, else: inputs

      {logits, carry} =
        evaluator.predict_fn.(evaluator.params, Map.put(inputs, "initial_hidden", carry))

      logits = logits |> Tuple.to_list() |> Enum.map(&flatten/1) |> List.to_tuple()
      actions = Map.new(batch.actions, fn {key, tensor} -> {key, flatten(tensor)} end)
      {logits, actions, %{evaluator | carry: carry}}
    else
      {evaluator.predict_fn.(evaluator.params, inputs), batch.actions, evaluator}
    end
  end

  def stream(evaluator, batches) do
    Stream.transform(batches, evaluator, fn batch, state ->
      {logits, actions, state} = batch(state, batch)
      {[{logits, actions}], state}
    end)
  end

  defp flatten(tensor) do
    shape = Tuple.to_list(Nx.shape(tensor))
    [batch_size, timesteps | rest] = shape
    Nx.reshape(tensor, List.to_tuple([batch_size * timesteps | rest]))
  end
end
