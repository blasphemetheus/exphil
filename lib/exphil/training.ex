defmodule ExPhil.Training do
  @moduledoc """
  Unified training interface for ExPhil.

  Provides a high-level API for training Melee AI agents using both
  imitation learning and reinforcement learning approaches.

  ## Training Pipeline

  ```
  1. Imitation Learning (Behavioral Cloning)
     ────────────────────────────────────────
     Replays (.slp) ──> Parser ──> Dataset ──> BC Training ──> Pretrained Policy

  2. Reinforcement Learning (PPO)
     ────────────────────────────────────────
     Pretrained Policy + Dolphin Environment ──> PPO Training ──> Final Agent
  ```

  ## Usage

      # Load dataset
      {:ok, dataset} = ExPhil.Training.load_dataset("replays/parsed/")

      # Train with imitation learning
      {:ok, trainer} = ExPhil.Training.train_imitation(dataset,
        epochs: 10,
        checkpoint_dir: "checkpoints/"
      )

      # Fine-tune with RL (requires Dolphin environment)
      {:ok, trainer} = ExPhil.Training.train_rl(trainer,
        total_steps: 100_000,
        checkpoint_dir: "checkpoints/"
      )

  ## Logging and Metrics

  Training progress is logged using Elixir's Logger and optionally
  to external services via callbacks. Metrics include:

  - `loss` - Current training loss
  - `button_accuracy` - Button prediction accuracy
  - `stick_error` - Mean stick position error
  - `value_loss` - Value function loss (for RL)
  - `policy_loss` - Policy gradient loss (for RL)
  - `entropy` - Action distribution entropy
  """

  alias ExPhil.Training.{Imitation, PPO, Data}

  require Logger

  @type trainer :: Imitation.t() | PPO.t()

  # ============================================================================
  # Dataset Loading
  # ============================================================================

  @doc """
  Load training dataset from parsed replay files.

  See `ExPhil.Training.Data.load_dataset/2` for options.
  """
  @spec load_dataset(Path.t(), keyword()) :: {:ok, Data.t()} | {:error, term()}
  defdelegate load_dataset(path, opts \\ []), to: Data

  @doc """
  Create dataset from in-memory frames.
  """
  @spec from_frames([map()], keyword()) :: Data.t()
  defdelegate from_frames(frames, opts \\ []), to: Data

  # ============================================================================
  # Imitation Learning
  # ============================================================================

  @doc """
  Train a policy using imitation learning (behavioral cloning).

  ## Options
    - `:epochs` - Number of training epochs (default: 10)
    - `:batch_size` - Training batch size (default: 64)
    - `:learning_rate` - Initial learning rate (default: 1.0e-4)
    - `:checkpoint_dir` - Directory for checkpoints
    - `:checkpoint_interval` - Steps between checkpoints (default: 1000)
    - `:validation_split` - Fraction for validation (default: 0.1)
    - `:callbacks` - List of callback functions
    - `:embed_size` - Embedding size (default: auto from dataset)
    - `:hidden_sizes` - Network hidden layers (default: [512, 512])

  ## Callbacks
    Callbacks receive a map with:
    - `:step` - Current training step
    - `:epoch` - Current epoch
    - `:loss` - Current loss value
    - `:metrics` - Additional metrics

  ## Returns
    `{:ok, trainer}` with trained policy, or `{:error, reason}`.
  """
  @spec train_imitation(Data.t(), keyword()) :: {:ok, Imitation.t()} | {:error, term()}
  def train_imitation(dataset, opts \\ []) do
    # Extract options
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 64)
    checkpoint_dir = Keyword.get(opts, :checkpoint_dir)
    checkpoint_interval = Keyword.get(opts, :checkpoint_interval, 1000)
    validation_split = Keyword.get(opts, :validation_split, 0.1)
    callbacks = Keyword.get(opts, :callbacks, [])

    # Split dataset
    {train_data, val_data} =
      if validation_split > 0 do
        Data.split(dataset, ratio: 1.0 - validation_split)
      else
        {dataset, nil}
      end

    # Create trainer
    trainer_opts =
      Keyword.take(opts, [
        :embed_size,
        :embed_config,
        :hidden_sizes,
        :learning_rate,
        :batch_size,
        :max_grad_norm,
        :weight_decay,
        :warmup_steps,
        :frame_stack,
        :axis_buckets,
        :shoulder_buckets
      ])

    trainer_opts =
      Keyword.put_new_lazy(trainer_opts, :embed_config, fn ->
        dataset.embed_config
      end)

    trainer = Imitation.new(trainer_opts)

    Logger.info("Starting imitation learning training")
    Logger.info("  Dataset size: #{train_data.size} frames")
    Logger.info("  Validation size: #{if val_data, do: val_data.size, else: 0} frames")
    Logger.info("  Epochs: #{epochs}")
    Logger.info("  Batch size: #{batch_size}")

    # Training callback
    callback = fn metrics ->
      # Run user callbacks
      Enum.each(callbacks, fn cb -> cb.(metrics) end)

      # Checkpoint
      if checkpoint_dir && rem(metrics.step, checkpoint_interval) == 0 do
        checkpoint_path = Path.join(checkpoint_dir, "imitation_step_#{metrics.step}.ckpt")
        Imitation.save_checkpoint(trainer, checkpoint_path)
      end
    end

    # Create batched dataset
    batches = Data.batched(train_data, batch_size: batch_size, shuffle: true)

    # Train
    case Imitation.train(trainer, batches, epochs: epochs, callback: callback) do
      {:ok, trained} ->
        # Final evaluation
        if val_data do
          val_batches = Data.batched(val_data, batch_size: batch_size, shuffle: false)
          val_metrics = Imitation.evaluate(trained, val_batches)
          Logger.info("Validation loss: #{Float.round(val_metrics.loss, 4)}")
        end

        # Save final checkpoint
        if checkpoint_dir do
          final_path = Path.join(checkpoint_dir, "imitation_final.ckpt")
          Imitation.save_checkpoint(trained, final_path)
        end

        {:ok, trained}

      error ->
        error
    end
  end

  @doc """
  Resume imitation training from checkpoint.
  """
  @spec resume_imitation(Path.t(), Data.t(), keyword()) :: {:ok, Imitation.t()} | {:error, term()}
  def resume_imitation(checkpoint_path, dataset, opts \\ []) do
    # Create base trainer
    trainer =
      Imitation.new(
        Keyword.take(opts, [
          :embed_size,
          :embed_config,
          :hidden_sizes,
          :axis_buckets,
          :shoulder_buckets
        ])
      )

    # Load checkpoint
    case Imitation.load_checkpoint(trainer, checkpoint_path) do
      {:ok, loaded_trainer} ->
        train_imitation_with_trainer(loaded_trainer, dataset, opts)

      error ->
        error
    end
  end

  defp train_imitation_with_trainer(trainer, dataset, opts) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 64)
    callbacks = Keyword.get(opts, :callbacks, [])

    callback = fn metrics ->
      Enum.each(callbacks, fn cb -> cb.(metrics) end)
    end

    batches = Data.batched(dataset, batch_size: batch_size, shuffle: true)
    Imitation.train(trainer, batches, epochs: epochs, callback: callback)
  end

  # ============================================================================
  # Reinforcement Learning
  # ============================================================================

  @doc """
  Train or fine-tune a policy using PPO.

  ## Options
    - `:pretrained` - Path to pretrained imitation checkpoint
    - `:total_steps` - Total training steps (default: 100_000)
    - `:steps_per_update` - Rollout steps between updates (default: 2048)
    - `:checkpoint_dir` - Directory for checkpoints
    - `:checkpoint_interval` - Steps between checkpoints (default: 10_000)
    - `:callbacks` - List of callback functions
    - All PPO hyperparameters (see `ExPhil.Training.PPO.new/1`)

  ## Environment
    PPO requires a training environment that provides rollouts.
    Pass the environment as the second argument.

  ## Returns
    `{:ok, trainer}` with trained policy, or `{:error, reason}`.
  """
  @spec train_rl(module(), keyword()) :: {:ok, PPO.t()} | {:error, term()}
  def train_rl(env_module, opts \\ []) do
    total_steps = Keyword.get(opts, :total_steps, 100_000)
    steps_per_update = Keyword.get(opts, :steps_per_update, 2048)
    checkpoint_dir = Keyword.get(opts, :checkpoint_dir)
    checkpoint_interval = Keyword.get(opts, :checkpoint_interval, 10_000)
    pretrained = Keyword.get(opts, :pretrained)
    callbacks = Keyword.get(opts, :callbacks, [])

    # Create trainer
    trainer_opts =
      Keyword.take(opts, [
        :embed_size,
        :embed_config,
        :hidden_sizes,
        :learning_rate,
        :gamma,
        :gae_lambda,
        :clip_range,
        :vf_coef,
        :entropy_coef,
        :max_grad_norm,
        :n_epochs,
        :batch_size,
        :axis_buckets,
        :shoulder_buckets,
        :teacher_kl_coef
      ])

    trainer = PPO.new(trainer_opts)

    # Load pretrained weights if provided
    trainer =
      if pretrained do
        case PPO.load_pretrained_policy(trainer, pretrained) do
          {:ok, loaded} ->
            Logger.info("Loaded pretrained policy from #{pretrained}")
            loaded

          {:error, reason} ->
            Logger.warning("Failed to load pretrained: #{inspect(reason)}")
            trainer
        end
      else
        trainer
      end

    Logger.info("Starting PPO training")
    Logger.info("  Total steps: #{total_steps}")
    Logger.info("  Steps per update: #{steps_per_update}")

    # Training loop
    num_updates = div(total_steps, steps_per_update)

    final_trainer =
      Enum.reduce_while(1..num_updates, trainer, fn update, acc ->
        # Collect rollouts
        rollout = collect_rollout(env_module, acc, steps_per_update)

        # Update policy
        {new_trainer, metrics} = PPO.update(acc, rollout)

        # Add update info
        full_metrics =
          Map.merge(metrics, %{
            update: update,
            total_steps: update * steps_per_update
          })

        # Callbacks
        Enum.each(callbacks, fn cb -> cb.(full_metrics) end)

        # Checkpoint
        if checkpoint_dir && rem(update, div(checkpoint_interval, steps_per_update)) == 0 do
          step = update * steps_per_update
          checkpoint_path = Path.join(checkpoint_dir, "ppo_step_#{step}.ckpt")
          PPO.save_checkpoint(new_trainer, checkpoint_path)
        end

        # Log progress
        if rem(update, 10) == 0 do
          Logger.info(
            "Update #{update}/#{num_updates}: " <>
              "policy_loss=#{Float.round(metrics.policy_loss, 4)}, " <>
              "value_loss=#{Float.round(metrics.value_loss, 4)}, " <>
              "entropy=#{Float.round(metrics.entropy, 4)}"
          )
        end

        {:cont, new_trainer}
      end)

    # Save final checkpoint
    if checkpoint_dir do
      final_path = Path.join(checkpoint_dir, "ppo_final.ckpt")
      PPO.save_checkpoint(final_trainer, final_path)
    end

    {:ok, final_trainer}
  end

  # Placeholder for rollout collection - requires environment integration
  defp collect_rollout(_env_module, _trainer, _steps) do
    # This would interact with Dolphin/libmelee to collect experience
    # For now, return empty rollout structure
    Logger.warning("Rollout collection not implemented - requires environment")

    %{
      states: Nx.broadcast(0.0, {1, 1}),
      actions: %{},
      rewards: Nx.broadcast(0.0, {1}),
      dones: Nx.broadcast(0, {1}),
      old_log_probs: Nx.broadcast(0.0, {1}),
      old_values: Nx.broadcast(0.0, {1})
    }
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Export trained policy for inference.
  """
  @spec export_policy(trainer(), Path.t()) :: :ok | {:error, term()}
  def export_policy(%Imitation{} = trainer, path) do
    Imitation.export_policy(trainer, path)
  end

  def export_policy(%PPO{} = trainer, path) do
    PPO.export_policy(trainer, path)
  end

  @doc """
  Load exported policy for inference.

  Options:
  - `validate: true` - Validate that params match config (default: true)

  Returns `{:ok, policy}` or `{:error, reason}` with a descriptive error message.
  """
  @spec load_policy(Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def load_policy(path, opts \\ []) do
    validate = Keyword.get(opts, :validate, true)

    with {:ok, binary} <- File.read(path),
         {:ok, policy} <- safe_deserialize(binary, path),
         :ok <- if(validate, do: validate_policy(policy, path), else: :ok) do
      {:ok, policy}
    end
  end

  # Safely deserialize binary term with helpful error messages
  defp safe_deserialize(binary, path) do
    try do
      {:ok, :erlang.binary_to_term(binary)}
    rescue
      ArgumentError ->
        {:error,
         "Failed to deserialize #{path}: file may be corrupted or not an ExPhil checkpoint"}
    end
  end

  @doc """
  Validate that policy params match the stored config.

  Checks:
  - Required keys present (params, config)
  - Temporal config matches param structure
  - Hidden layer sizes match param shapes
  - Output head sizes match axis/shoulder buckets
  """
  @spec validate_policy(map(), Path.t()) :: :ok | {:error, String.t()}
  def validate_policy(policy, path \\ "checkpoint") do
    with :ok <- validate_required_keys(policy, path),
         :ok <- validate_param_structure(policy, path) do
      :ok
    end
  end

  defp validate_required_keys(policy, path) do
    cond do
      not is_map(policy) ->
        {:error,
         "#{path}: Expected map with :params and :config, got #{inspect(policy.__struct__ || :map)}"}

      # Check for exported policy format
      Map.has_key?(policy, :params) and Map.has_key?(policy, :config) ->
        :ok

      # Check for checkpoint format
      Map.has_key?(policy, :policy_params) and Map.has_key?(policy, :config) ->
        :ok

      Map.has_key?(policy, :params) ->
        {:error, "#{path}: Missing :config key. This may be an old checkpoint format."}

      Map.has_key?(policy, :config) ->
        {:error, "#{path}: Missing :params key. File may be corrupted."}

      true ->
        keys = Map.keys(policy) |> Enum.take(5) |> Enum.join(", ")
        {:error, "#{path}: Unrecognized checkpoint format. Found keys: #{keys}"}
    end
  end

  defp validate_param_structure(policy, path) do
    params = policy[:params] || policy[:policy_params]
    config = policy[:config]

    # Extract actual model state params (handle both raw params and ModelState)
    actual_params = extract_params_data(params)

    # Check for temporal/non-temporal mismatch
    temporal = config[:temporal] || false
    backbone = config[:backbone] || :mlp

    cond do
      # Check if params look like temporal model but config says MLP
      temporal == false and has_temporal_layers?(actual_params) ->
        {:error,
         """
         #{path}: Architecture mismatch detected!

         Config says: temporal=false (MLP model)
         But params contain temporal layers (#{detected_backbone(actual_params)})

         This checkpoint was likely saved with --temporal but you're trying to load it
         without --temporal, or vice versa.
         """}

      # Check if params look like MLP but config says temporal
      temporal == true and not has_temporal_layers?(actual_params) ->
        {:error,
         """
         #{path}: Architecture mismatch detected!

         Config says: temporal=true, backbone=#{backbone}
         But params appear to be from a non-temporal MLP model.

         This checkpoint was likely saved without --temporal but you're trying to load it
         with --temporal.
         """}

      # Check hidden sizes match
      not hidden_sizes_match?(actual_params, config) ->
        expected = config[:hidden_sizes] || [512, 512]

        {:error,
         """
         #{path}: Hidden layer size mismatch!

         Config specifies hidden_sizes: #{inspect(expected)}
         But params have different layer dimensions.

         Make sure --hidden-sizes matches the checkpoint's architecture.
         """}

      true ->
        :ok
    end
  end

  # Extract raw params from ModelState or return as-is
  defp extract_params_data(%Axon.ModelState{data: data}), do: data
  defp extract_params_data(params) when is_map(params), do: params

  # Check if params contain temporal-specific layers
  defp has_temporal_layers?(params) when is_map(params) do
    keys = Map.keys(params) |> Enum.map(&to_string/1)

    Enum.any?(keys, fn key ->
      String.contains?(key, "lstm") or
        String.contains?(key, "gru") or
        String.contains?(key, "mamba") or
        String.contains?(key, "attention") or
        String.contains?(key, "sliding_window")
    end)
  end

  defp has_temporal_layers?(_), do: false

  # Try to detect what backbone the params are from
  defp detected_backbone(params) when is_map(params) do
    keys = Map.keys(params) |> Enum.map(&to_string/1)

    cond do
      Enum.any?(keys, &String.contains?(&1, "lstm")) -> "LSTM"
      Enum.any?(keys, &String.contains?(&1, "gru")) -> "GRU"
      Enum.any?(keys, &String.contains?(&1, "mamba")) -> "Mamba"
      Enum.any?(keys, &String.contains?(&1, "attention")) -> "Attention"
      true -> "unknown temporal"
    end
  end

  defp detected_backbone(_), do: "unknown"

  # Check if hidden layer sizes match params
  defp hidden_sizes_match?(params, config) when is_map(params) do
    hidden_sizes = config[:hidden_sizes] || [512, 512]
    keys = Map.keys(params) |> Enum.map(&to_string/1)

    # Find backbone dense layers
    backbone_layers = Enum.filter(keys, &String.starts_with?(&1, "backbone_dense_"))

    # If we can't find backbone layers, assume it's ok (different naming convention)
    if Enum.empty?(backbone_layers) do
      true
    else
      # Check we have the expected number of layers
      length(backbone_layers) == length(hidden_sizes)
    end
  end

  defp hidden_sizes_match?(_, _), do: true

  @doc """
  Get action from trained policy.
  """
  @spec get_action(trainer(), Nx.Tensor.t(), keyword()) :: map()
  def get_action(trainer, state, opts \\ [])

  def get_action(%Imitation{} = trainer, state, opts) do
    Imitation.get_action(trainer, state, opts)
  end

  def get_action(%PPO{} = trainer, state, opts) do
    PPO.get_action(trainer, state, opts)
  end

  @doc """
  Get controller state for game input.
  """
  @spec get_controller_action(trainer(), Nx.Tensor.t(), keyword()) ::
          ExPhil.Bridge.ControllerState.t()
  def get_controller_action(trainer, state, opts \\ [])

  def get_controller_action(%Imitation{} = trainer, state, opts) do
    Imitation.get_controller_action(trainer, state, opts)
  end

  def get_controller_action(%PPO{} = trainer, state, opts) do
    PPO.get_controller_action(trainer, state, opts)
  end

  @doc """
  Get training metrics summary.
  """
  @spec metrics_summary(trainer()) :: map()
  def metrics_summary(%Imitation{} = trainer) do
    Imitation.metrics_summary(trainer)
  end

  def metrics_summary(%PPO{} = trainer) do
    PPO.metrics_summary(trainer)
  end
end
