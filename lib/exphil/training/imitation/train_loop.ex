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

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.{MixedPrecision, Utils}
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

    # Use cached loss+grad function (built once in new/1)
    {loss, grads} = trainer.loss_and_grad_fn.(trainer.policy_params, states, actions)

    # Extract data for optimizer (grads has same structure as ModelState)
    grads_data = get_params_data(grads)
    params_data = get_params_data(trainer.policy_params)

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

  # Mixed precision training with FP32 master weights
  defp train_step_mixed_precision(trainer, batch, mp_state) do
    %{states: states, actions: actions} = batch

    # Get BF16 compute params for forward/backward pass (fast on tensor cores)
    compute_params = MixedPrecision.get_compute_params(mp_state)
    compute_model_state = put_params_data(trainer.policy_params, compute_params)

    # Forward + backward in BF16 (compute precision)
    {loss, grads} = trainer.loss_and_grad_fn.(compute_model_state, states, actions)
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
  defp compute_gradients(trainer, batch) do
    %{states: states, actions: actions} = batch

    states = Nx.backend_copy(states)
    actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

    # Convert states to training precision (bf16 for ~2x speedup)
    states = Nx.as_type(states, trainer.config.precision)

    predict_fn = trainer.predict_fn
    model_state = deep_backend_copy(trainer.policy_params)
    label_smoothing = trainer.config[:label_smoothing] || 0.0
    focal_loss = trainer.config[:focal_loss] || false
    focal_gamma = trainer.config[:focal_gamma] || 2.0
    button_weight = trainer.config[:button_weight] || 1.0
    stick_edge_weight = trainer.config[:stick_edge_weight]

    loss_fn = fn params ->
      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      Policy.imitation_loss(logits, actions,
        label_smoothing: label_smoothing,
        focal_loss: focal_loss,
        focal_gamma: focal_gamma,
        button_weight: button_weight,
        stick_edge_weight: stick_edge_weight
      )
    end

    {loss, grads} = Nx.Defn.value_and_grad(loss_fn).(model_state)
    grads_data = get_params_data(grads)
    loss_val = Nx.to_number(loss)

    {grads_data, loss_val}
  end

  # Add two gradient maps element-wise
  defp add_gradients(grads1, grads2) do
    deep_map2(grads1, grads2, fn t1, t2 -> Nx.add(t1, t2) end)
  end

  # Scale gradients by a factor
  defp scale_gradients(grads, factor) do
    factor_t = Nx.tensor(factor, type: :f32)
    deep_map(grads, fn t -> Nx.multiply(t, factor_t) end)
  end

  # Apply averaged gradients to update parameters
  defp apply_gradients(trainer, grads) do
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

  # Deep copy all tensors in a nested map/ModelState to avoid EXLA/Expr mismatch
  # NOTE: Clause order matters! Nx.Tensor and Axon.ModelState are structs (i.e. maps),
  # so they must be pattern-matched BEFORE the generic is_map guard clause.
  defp deep_backend_copy(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor)

  defp deep_backend_copy(%Axon.ModelState{data: data} = state) do
    %{state | data: deep_backend_copy(data)}
  end

  defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
  end

  defp deep_backend_copy(other), do: other

  # ============================================================================
  # Private - Axon.ModelState Helpers
  # ============================================================================

  defp get_params_data(%Axon.ModelState{data: data}), do: data
  defp get_params_data(params) when is_map(params), do: params

  defp put_params_data(%Axon.ModelState{} = state, data), do: %{state | data: data}
  defp put_params_data(_original, data), do: data
end
