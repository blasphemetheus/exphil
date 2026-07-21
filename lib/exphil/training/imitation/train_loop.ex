defmodule ExPhil.Training.Imitation.TrainLoop do
  @moduledoc """
  Training loop and gradient handling for imitation learning.

  This module implements the core training loop including:
  - Epoch iteration over datasets
  - Gradient computation and accumulation
  - Parameter updates with optimizer
  - Mixed precision training support
  - Periodic logging and garbage collection

  ## Gradient Accumulation

  For large effective batch sizes that don't fit in GPU memory, gradient
  accumulation allows training with `batch_size * accumulation_steps`
  effective samples per update.

  ## Mixed Precision Training

  When enabled, forward/backward passes use BF16 for speed while maintaining
  FP32 master weights for numerical stability. This provides:
  - ~2x speedup on tensor cores
  - Preserved small gradient updates via FP32 accumulation

  ## See Also

  - `ExPhil.Training.Imitation` - Main imitation learning module
  - `ExPhil.Training.Imitation.Loss` - Loss function builders
  - `ExPhil.Training.MixedPrecision` - Mixed precision state management
  """

  alias ExPhil.Training.MixedPrecision
  alias ExPhil.Training.Imitation.Loss

  require Logger

  # ============================================================================
  # Main Training Functions
  # ============================================================================

  @doc """
  Train on a dataset for multiple epochs.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `dataset` - Enumerable of batch maps with `:states` and `:actions` keys
  - `opts` - Options:
    - `:epochs` - Number of epochs (default: 1)
    - `:callback` - Function called after each step with metrics

  ## Returns

  - `{:ok, updated_trainer}` on success
  - `{:error, exception}` on failure
  """
  @spec train(struct(), Enumerable.t(), keyword()) :: {:ok, struct()} | {:error, term()}
  def train(trainer, dataset, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 1)
    callback = Keyword.get(opts, :callback, fn _metrics -> :ok end)

    try do
      final_trainer =
        Enum.reduce(1..epochs, trainer, fn epoch, acc ->
          Logger.info("Starting epoch #{epoch}/#{epochs}")
          train_epoch(acc, dataset, epoch, callback)
        end)

      {:ok, final_trainer}
    rescue
      e -> {:error, e}
    end
  end

  @doc """
  Train for a single epoch.

  Supports gradient accumulation via `config.accumulation_steps`.
  With `accumulation_steps=N`, gradients are averaged over N mini-batches
  before applying updates, effectively training with `batch_size*N`.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `dataset` - Enumerable of batch maps
  - `epoch` - Current epoch number (for logging)
  - `callback` - Function called after each optimizer step
  """
  @spec train_epoch(struct(), Enumerable.t(), non_neg_integer(), function()) :: struct()
  def train_epoch(trainer, dataset, epoch, callback) do
    accumulation_steps = trainer.config[:accumulation_steps] || 1

    if accumulation_steps == 1 do
      # Fast path: no accumulation, original behavior
      train_epoch_no_accumulation(trainer, dataset, epoch, callback)
    else
      # Gradient accumulation path
      train_epoch_with_accumulation(trainer, dataset, epoch, callback, accumulation_steps)
    end
  end

  @doc """
  Perform a single training step.

  Uses cached `loss_and_grad_fn` built in `Imitation.new/1` to avoid per-batch overhead.
  No more `deep_backend_copy` or closure creation every batch.

  When mixed precision is enabled:
  - Forward/backward pass uses BF16 compute params for speed
  - Gradients are cast to FP32 before accumulation (preserves small updates)
  - FP32 master weights maintain full precision across many steps

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `batch` - Map with `:states` and `:actions` keys
  - `_loss_fn` - Unused (kept for API compatibility)

  ## Returns

  `{updated_trainer, metrics_map}` where metrics includes `:loss` and `:step`.
  """
  @spec train_step(struct(), map(), function()) :: {struct(), map()}
  def train_step(%{mixed_precision_state: nil} = trainer, batch, _loss_fn) do
    # Standard training path (no mixed precision)
    train_step_standard(trainer, batch)
  end

  def train_step(%{mixed_precision_state: mp_state} = trainer, batch, _loss_fn) do
    # Mixed precision training path
    train_step_mixed_precision(trainer, batch, mp_state)
  end

  # ============================================================================
  # Private - Epoch Training Without Accumulation
  # ============================================================================

  defp train_epoch_no_accumulation(trainer, dataset, epoch, callback) do
    {_predict_fn, loss_fn} = Loss.build_loss_fn(trainer.policy_model)
    gc_every = trainer.config[:gc_every] || 0

    Enum.reduce(dataset, trainer, fn batch, acc ->
      {new_trainer, metrics} = train_step(acc, batch, loss_fn)

      # Call callback
      full_metrics = Map.merge(metrics, %{epoch: epoch, step: new_trainer.step})
      callback.(full_metrics)

      # Log periodically
      if rem(new_trainer.step, acc.config.log_interval) == 0 do
        Logger.info("Step #{new_trainer.step}: loss=#{Float.round(metrics.loss, 4)}")
      end

      # Periodic garbage collection to prevent memory buildup
      if gc_every > 0 and rem(new_trainer.step, gc_every) == 0 do
        :erlang.garbage_collect()
      end

      new_trainer
    end)
  end

  # ============================================================================
  # Private - Epoch Training With Gradient Accumulation
  # ============================================================================

  defp train_epoch_with_accumulation(trainer, dataset, epoch, callback, accumulation_steps) do
    gc_every = trainer.config[:gc_every] || 0

    # Track accumulated gradients and losses
    init_accum = %{
      trainer: trainer,
      grads: nil,
      losses: [],
      count: 0
    }

    final_accum =
      Enum.reduce(dataset, init_accum, fn batch, accum ->
        # Compute gradients without applying updates
        {grads, loss} = compute_gradients(accum.trainer, batch)

        # Accumulate gradients (sum them)
        new_grads =
          if accum.grads == nil do
            grads
          else
            add_gradients(accum.grads, grads)
          end

        new_accum = %{
          accum
          | grads: new_grads,
            losses: [loss | accum.losses],
            count: accum.count + 1
        }

        # Check if we should apply update
        if new_accum.count >= accumulation_steps do
          # Average gradients and apply update
          avg_grads = scale_gradients(new_accum.grads, 1.0 / accumulation_steps)
          avg_loss = Enum.sum(new_accum.losses) / accumulation_steps

          new_trainer = apply_gradients(new_accum.trainer, avg_grads)

          # Call callback
          metrics = %{loss: avg_loss, step: new_trainer.step}
          full_metrics = Map.merge(metrics, %{epoch: epoch, step: new_trainer.step})
          callback.(full_metrics)

          # Log periodically
          if rem(new_trainer.step, new_trainer.config.log_interval) == 0 do
            Logger.info(
              "Step #{new_trainer.step}: loss=#{Float.round(avg_loss, 4)} (accum=#{accumulation_steps})"
            )
          end

          # Periodic garbage collection to prevent memory buildup
          if gc_every > 0 and rem(new_trainer.step, gc_every) == 0 do
            :erlang.garbage_collect()
          end

          # Reset accumulation state
          %{trainer: new_trainer, grads: nil, losses: [], count: 0}
        else
          new_accum
        end
      end)

    # Handle remaining batches if dataset size isn't divisible by accumulation_steps
    if final_accum.count > 0 and final_accum.grads != nil do
      avg_grads = scale_gradients(final_accum.grads, 1.0 / final_accum.count)
      avg_loss = Enum.sum(final_accum.losses) / final_accum.count

      new_trainer = apply_gradients(final_accum.trainer, avg_grads)

      # Log final partial accumulation
      Logger.info(
        "Step #{new_trainer.step}: loss=#{Float.round(avg_loss, 4)} (partial accum=#{final_accum.count})"
      )

      new_trainer
    else
      final_accum.trainer
    end
  end

  # ============================================================================
  # Private - Train Step Implementations
  # ============================================================================

  # Standard training without mixed precision
  defp train_step_standard(trainer, batch) do
    %{states: states, actions: actions} = batch
    frame_weights = Map.get(batch, :frame_weights) || default_frame_weights(states)
    policy_type = trainer.config[:policy_type] || :autoregressive

    # Compute loss and gradients based on policy type
    {loss, grads} = compute_policy_loss_and_grad(
      policy_type, trainer, states, actions, frame_weights
    )

    # Extract data for optimizer (grads has same structure as ModelState)
    grads_data = get_params_data(grads)
    params_data = get_params_data(trainer.policy_params)

    # NaN forensics (config :debug_grads_after, integer step): cheap fused
    # global grad finiteness check per step; on the first non-finite step,
    # dump per-layer grad norms to name the operator that detonates.
    # Three 2026-07-13 forensics runs showed single-step NaN births that
    # per-100-step sampling cannot localize — this catches the fatal step.
    debug_after = trainer.config[:debug_grads_after]

    if debug_after && trainer.step >= debug_after do
      check_grad_detonation(loss, grads_data, trainer, batch)
    end

    # Update parameters using the optimizer
    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads_data,
        trainer.optimizer_state,
        params_data
      )

    # Use cached apply_updates_fn (built once in new/1, reused every step)
    new_params_data = trainer.apply_updates_fn.(params_data, updates)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    new_trainer = %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        step: trainer.step + 1
    }

    {new_trainer, %{loss: loss, step: new_trainer.step}}
  end

  # One fused kernel (jit-cached on the grads container shape) taking the
  # global max |grad|. Max never overflows (unlike sum-of-squares norms,
  # which hit inf at |g| ~ 1e19 in ANY 32-bit float — that ambiguity made
  # the 2026-07-14 dumps unable to distinguish "inf grads" from "large
  # finite grads"; note clip_by_global_norm has the SAME overflow inside
  # it, making the clipper itself a NaN amplifier for large finite bursts).
  # Trip threshold 1.0e15: far above healthy grads (~1e-2..1), far below
  # the norm-overflow zone — catches the burst while it is still finite.
  defp check_grad_detonation(loss, grads_data, trainer, batch) do
    step = trainer.step
    tensors = flatten_grad_tensors(grads_data) |> List.to_tuple()

    global_max =
      Nx.Defn.jit_apply(
        fn tup ->
          tup
          |> Tuple.to_list()
          |> Enum.map(&Nx.reduce_max(Nx.abs(&1)))
          |> Enum.reduce(&Nx.max/2)
        end,
        [tensors],
        compiler: EXLA
      )
      |> Nx.to_number()

    if not is_number(global_max) or global_max > 1.0e15 do
      loss_num = Nx.to_number(loss)

      per_layer =
        Enum.map(grads_data, fn {layer, params} ->
          maxes =
            params
            |> flatten_grad_tensors()
            |> Enum.map(fn t -> Nx.reduce_max(Nx.abs(t)) |> Nx.to_number() end)

          {layer,
           if(Enum.all?(maxes, &is_number/1), do: Enum.max([0.0 | maxes]), else: :NONFINITE)}
        end)

      Logger.error(
        "GRAD_DETONATION step=#{step} loss=#{inspect(loss_num)} " <>
          "global_max=#{inspect(global_max)} per_layer_max=#{inspect(per_layer, limit: :infinity)}"
      )

      # Crime-scene capture ("build the time-savers first", 2026-07-14):
      # serialize everything needed to replay THIS exact backward pass
      # offline — the fatal batch, the pre-step params, the optimizer
      # state. One saved scene = unlimited second-long re-examinations
      # (precision sweeps, op bisection) instead of 40-90 min approach
      # runs per hypothesis. See scripts/replay_crime_scene.exs.
      save_crime_scene(trainer, batch, step, loss_num)
    end
  end

  defp save_crime_scene(trainer, batch, step, loss_num) do
    dir = "interp/crime_scenes"
    File.mkdir_p!(dir)
    path = Path.join(dir, "scene_step#{step}.bin")

    transfer = fn term ->
      try do
        Nx.backend_transfer(term, Nx.BinaryBackend)
      rescue
        _ -> term
      end
    end

    scene = %{
      step: step,
      loss: loss_num,
      batch: %{
        states: transfer.(batch.states),
        actions: transfer.(batch.actions),
        frame_weights: transfer.(Map.get(batch, :frame_weights))
      },
      params: transfer.(trainer.policy_params),
      optimizer_state: transfer.(trainer.optimizer_state),
      config: Map.drop(trainer.config, [:kmeans_centers])
    }

    File.write!(path, :erlang.term_to_binary(scene))
    Logger.error("CRIME_SCENE saved: #{path}")
  rescue
    e -> Logger.error("crime-scene save failed: #{inspect(e)}")
  end

  defp flatten_grad_tensors(%Nx.Tensor{} = t), do: [t]

  defp flatten_grad_tensors(map) when is_map(map) and not is_struct(map) do
    map |> Map.values() |> Enum.flat_map(&flatten_grad_tensors/1)
  end

  defp flatten_grad_tensors(_), do: []

  # Mixed precision training with FP32 master weights
  defp train_step_mixed_precision(trainer, batch, mp_state) do
    %{states: states, actions: actions} = batch
    policy_type = trainer.config[:policy_type] || :autoregressive

    # Get BF16 compute params for forward/backward pass (fast on tensor cores)
    compute_params = MixedPrecision.get_compute_params(mp_state)
    compute_model_state = put_params_data(trainer.policy_params, compute_params)

    frame_weights = Map.get(batch, :frame_weights) || default_frame_weights(states)

    # Forward + backward in BF16 (compute precision)
    {loss, grads} = compute_policy_loss_and_grad(
      policy_type, %{trainer | policy_params: compute_model_state}, states, actions, frame_weights
    )
    grads_data = get_params_data(grads)

    # Cast gradients to FP32 (preserves small gradient updates)
    grads_f32 = MixedPrecision.cast_grads_to_f32(mp_state, grads_data)

    # Get FP32 master weights for optimizer
    master_params = MixedPrecision.get_master_params(mp_state)

    # Apply optimizer to FP32 master weights (same call pattern as non-mixed precision)
    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads_f32,
        trainer.optimizer_state,
        master_params
      )

    # Apply updates to FP32 master weights
    new_params_data = trainer.apply_updates_fn.(master_params, updates)

    # Update mixed precision state with new master params
    new_mp_state = MixedPrecision.set_master_params(mp_state, new_params_data)

    # Also update policy_params (for checkpointing)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    new_trainer = %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        mixed_precision_state: new_mp_state,
        step: trainer.step + 1
    }

    {new_trainer, %{loss: loss, step: new_trainer.step}}
  end

  # ============================================================================
  # Private - Gradient Accumulation Helpers
  # ============================================================================

  # Compute gradients for a batch without applying updates
  @doc "Compute gradients for a batch without applying updates. Returns {grads_data, loss_number}."
  def compute_gradients(trainer, batch) do
    %{states: states, actions: actions} = batch
    frame_weights = Map.get(batch, :frame_weights) || default_frame_weights(states)
    policy_type = trainer.config[:policy_type] || :autoregressive

    {loss, grads} = compute_policy_loss_and_grad(policy_type, trainer, states, actions, frame_weights)
    grads_data = get_params_data(grads)
    loss_val = Nx.to_number(loss)

    {grads_data, loss_val}
  end

  @doc "Add two gradient maps element-wise."
  def add_gradients(grads1, grads2) do
    deep_map2(grads1, grads2, fn t1, t2 -> Nx.add(t1, t2) end)
  end

  @doc "Scale gradients by a factor."
  def scale_gradients(grads, factor) do
    factor_t = Nx.tensor(factor, type: :f32)
    deep_map(grads, fn t -> Nx.multiply(t, factor_t) end)
  end

  @doc "Apply averaged gradients to update parameters."
  def apply_gradients(trainer, grads) do
    params_data = get_params_data(trainer.policy_params)

    {updates, new_optimizer_state} =
      trainer.optimizer.(
        grads,
        trainer.optimizer_state,
        params_data
      )

    new_params_data = trainer.apply_updates_fn.(params_data, updates)
    new_params = put_params_data(trainer.policy_params, new_params_data)

    %{
      trainer
      | policy_params: new_params,
        optimizer_state: new_optimizer_state,
        step: trainer.step + 1
    }
  end

  # ============================================================================
  # Private - Deep Map Helpers for Nested Gradient Structures
  # ============================================================================

  # Deep map over nested gradient structures
  defp deep_map(%Nx.Tensor{} = t, fun), do: fun.(t)

  defp deep_map(map, fun) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_map(v, fun)} end)
  end

  defp deep_map(other, _fun), do: other

  # Deep map2 over two nested gradient structures
  defp deep_map2(%Nx.Tensor{} = t1, %Nx.Tensor{} = t2, fun), do: fun.(t1, t2)

  defp deep_map2(map1, map2, fun) when is_map(map1) and is_map(map2) and not is_struct(map1) do
    Map.new(map1, fn {k, v1} ->
      v2 = Map.fetch!(map2, k)
      {k, deep_map2(v1, v2, fun)}
    end)
  end

  defp deep_map2(other, _other2, _fun), do: other

  # ============================================================================
  # Gradient Norm Diagnostics
  # ============================================================================

  @doc """
  Compute per-layer gradient L2 norms for a single batch.

  Returns a list of `{layer_name, norm}` tuples sorted by norm descending,
  useful for diagnosing which layers have exploding or vanishing gradients.

  ## Parameters

  - `trainer` - The imitation trainer struct
  - `batch` - Map with `:states` and `:actions` keys

  ## Returns

  List of `{layer_name, l2_norm}` sorted by norm descending.
  """
  @spec compute_grad_norms(struct(), map()) :: [{String.t(), float()}]
  def compute_grad_norms(trainer, batch) do
    %{states: states, actions: actions} = batch
    frame_weights = Map.get(batch, :frame_weights) || default_frame_weights(states)
    policy_type = trainer.config[:policy_type] || :autoregressive

    {_loss, grads} = compute_policy_loss_and_grad(policy_type, trainer, states, actions, frame_weights)
    grads_data = get_params_data(grads)

    flatten_grad_norms(grads_data, "")
    |> Enum.sort_by(fn {_name, norm} -> norm end, :desc)
  end

  # Recursively flatten nested gradient maps into {path, l2_norm} pairs
  defp flatten_grad_norms(grads, prefix) when is_map(grads) and not is_struct(grads) do
    Enum.flat_map(grads, fn {key, value} ->
      path = if prefix == "", do: key, else: "#{prefix}.#{key}"
      flatten_grad_norms(value, path)
    end)
  end

  defp flatten_grad_norms(tensor, path) do
    norm =
      tensor
      |> Nx.flatten()
      |> Nx.LinAlg.norm()
      |> Nx.to_number()

    [{path, norm}]
  end

  # ============================================================================
  # Private - Axon.ModelState Helpers
  # ============================================================================

  @doc "Extract raw parameter data from ModelState or raw map."
  def get_params_data(%Axon.ModelState{data: data}), do: data
  def get_params_data(params) when is_map(params), do: params

  @doc "Wrap parameter data back into the original container format."
  def put_params_data(%Axon.ModelState{} = state, data), do: %{state | data: data}
  def put_params_data(_original, data), do: data

  # Default frame weights: uniform 1.0 for all frames in the batch
  # Used when batch doesn't include pre-computed frame_weights
  defp default_frame_weights(states) do
    batch_size = elem(Nx.shape(states), 0)
    Nx.broadcast(Nx.tensor(1.0, type: :f32), {batch_size})
  end

  # ============================================================================
  # Private - Policy-Type-Aware Loss and Gradient Computation
  # ============================================================================

  # Dispatch loss computation based on policy type
  # For autoregressive and ACT: uses cached loss_and_grad_fn with 4 args (includes frame_weights)
  # For diffusion and flow_matching: samples noise/timestep and uses 5 args
  defp compute_policy_loss_and_grad(:autoregressive, trainer, states, actions, frame_weights) do
    # Probe-as-regularizer (r15): the loss fn was built with a 5th argument
    # (the online-refit probe direction) whenever probe_direction is set
    if trainer.probe_direction do
      trainer.loss_and_grad_fn.(
        trainer.policy_params,
        states,
        actions,
        frame_weights,
        trainer.probe_direction
      )
    else
      trainer.loss_and_grad_fn.(trainer.policy_params, states, actions, frame_weights)
    end
  end

  defp compute_policy_loss_and_grad(:act, trainer, states, actions, frame_weights) do
    trainer.loss_and_grad_fn.(trainer.policy_params, states, actions, frame_weights)
  end

  defp compute_policy_loss_and_grad(:diffusion, trainer, states, actions, _frame_weights) do
    # Sample random noise and timesteps for diffusion training
    {noise, timestep} = sample_noise_and_timestep(trainer, actions)
    trainer.loss_and_grad_fn.(trainer.policy_params, states, actions, noise, timestep)
  end

  defp compute_policy_loss_and_grad(:flow_matching, trainer, states, actions, _frame_weights) do
    # Sample random noise and timesteps for flow matching training
    {noise, timestep} = sample_noise_and_timestep(trainer, actions)
    trainer.loss_and_grad_fn.(trainer.policy_params, states, actions, noise, timestep)
  end

  # Sample noise and timesteps for generative policy training
  # Returns {noise, timestep} tensors matching action shape
  defp sample_noise_and_timestep(trainer, actions) do
    # Get action shape from first action tensor (for diffusion/flow, actions should be continuous)
    action_shape = get_action_shape(actions)
    batch_size = elem(action_shape, 0)

    # Get number of diffusion timesteps from config
    num_timesteps = trainer.config[:num_inference_steps] || 20

    # Sample random noise with same shape as actions
    noise_key = Nx.Random.key(:erlang.unique_integer())
    {noise, _new_key} = Nx.Random.normal(noise_key, shape: action_shape, type: :f32)

    # Sample random timesteps in [0, num_timesteps-1]
    timestep_key = Nx.Random.key(:erlang.unique_integer())
    {timestep_f, _new_key} = Nx.Random.uniform(timestep_key, shape: {batch_size}, type: :f32)
    timestep = Nx.floor(Nx.multiply(timestep_f, num_timesteps)) |> Nx.as_type(:s32)

    {noise, timestep}
  end

  # Get action shape - handles both map format and tensor format
  defp get_action_shape(actions) when is_map(actions) do
    # For autoregressive-style actions, we need to convert to continuous
    # This shouldn't happen for diffusion/flow, but handle gracefully
    first_tensor = actions |> Map.values() |> List.first()
    Nx.shape(first_tensor)
  end

  defp get_action_shape(actions) when is_struct(actions, Nx.Tensor) do
    Nx.shape(actions)
  end
end
