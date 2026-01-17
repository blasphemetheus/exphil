defmodule ExPhil.Training.PPO do
  @moduledoc """
  Proximal Policy Optimization (PPO) trainer for Melee AI.

  PPO is an on-policy reinforcement learning algorithm that improves
  upon vanilla policy gradient by using a clipped surrogate objective
  to prevent large policy updates.

  ## Training Loop

  ```
  ┌──────────────────────────────────────────────────────────────┐
  │                     PPO Training Loop                         │
  │                                                               │
  │  1. Collect rollouts (play games, record transitions)        │
  │  2. Compute returns and advantages (GAE)                     │
  │  3. For K epochs:                                            │
  │     - Sample minibatches from rollout                        │
  │     - Compute policy ratio and clipped loss                  │
  │     - Compute value loss (clipped)                           │
  │     - Compute entropy bonus                                   │
  │     - Update parameters                                       │
  │  4. Repeat                                                    │
  └──────────────────────────────────────────────────────────────┘
  ```

  ## Usage

      # Create PPO trainer (optionally from imitation checkpoint)
      trainer = PPO.new(
        embed_size: 1024,
        pretrained_path: "checkpoints/imitation.axon"
      )

      # Training requires an environment runner
      {:ok, trained} = PPO.train(trainer, env_runner,
        total_timesteps: 1_000_000,
        rollout_length: 2048
      )

  ## Key Hyperparameters

  - `clip_range`: How much the policy ratio can change (default: 0.2)
  - `value_coef`: Weight of value loss (default: 0.5)
  - `entropy_coef`: Weight of entropy bonus (default: 0.01)
  - `gae_lambda`: GAE lambda for advantage estimation (default: 0.95)
  - `gamma`: Discount factor (default: 0.99)
  """

  alias ExPhil.Networks.{Policy, Value, ActorCritic}
  alias ExPhil.Embeddings

  require Logger

  defstruct [
    :model,                # Combined actor-critic model
    :params,               # Model parameters
    :old_params,           # Parameters from previous iteration (for ratio)
    :optimizer,            # Optimizer
    :optimizer_state,      # Optimizer state
    :embed_config,         # Embedding configuration
    :config,               # Training configuration
    :step,                 # Training step counter
    :timesteps,            # Total timesteps collected
    :metrics               # Training metrics
  ]

  @type t :: %__MODULE__{}

  # Default PPO configuration
  @default_config %{
    # PPO hyperparameters
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    value_coef: 0.5,
    entropy_coef: 0.01,
    max_grad_norm: 0.5,

    # Training settings
    learning_rate: 3.0e-4,
    batch_size: 64,
    num_epochs: 10,
    num_minibatches: 4,
    rollout_length: 2048,

    # Regularization
    teacher_kl_coef: 0.0,    # KL penalty to stay close to teacher policy
    value_clip: true,

    # Discretization
    axis_buckets: 16,
    shoulder_buckets: 4,

    # Logging
    log_interval: 10,
    checkpoint_interval: 100
  }

  @doc """
  Create a new PPO trainer.

  ## Options
    - `:embed_size` - Size of state embedding (required unless embed_config provided)
    - `:embed_config` - Embedding configuration
    - `:pretrained_path` - Path to pretrained imitation policy
    - `:hidden_sizes` - Network hidden layer sizes
    - All config options from @default_config
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    # Get embedding configuration
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config(Keyword.take(opts, [:with_speeds, :with_nana, :with_projectiles]))
    end)

    embed_size = Keyword.get(opts, :embed_size, Embeddings.embedding_size(embed_config))

    # Build config
    config = @default_config
    |> Map.merge(Map.new(Keyword.take(opts, Map.keys(@default_config))))

    # Build combined actor-critic model
    model = ActorCritic.build_combined(
      embed_size: embed_size,
      hidden_sizes: Keyword.get(opts, :hidden_sizes, [512, 512]),
      axis_buckets: config.axis_buckets,
      shoulder_buckets: config.shoulder_buckets
    )

    # Initialize or load parameters using newer Axon API
    {init_fn, _predict_fn} = Axon.build(model)
    initial_params = init_fn.(Nx.template({1, embed_size}, :f32), Axon.ModelState.empty())

    params = case Keyword.get(opts, :pretrained_path) do
      nil -> initial_params
      path -> load_pretrained(initial_params, path)
    end

    # Create optimizer
    {optimizer_init, optimizer_update} = create_optimizer(config)
    optimizer_state = optimizer_init.(params)

    %__MODULE__{
      model: model,
      params: params,
      old_params: deep_copy_params(params),
      optimizer: optimizer_update,
      optimizer_state: optimizer_state,
      embed_config: embed_config,
      config: config,
      step: 0,
      timesteps: 0,
      metrics: init_metrics()
    }
  end

  defp load_pretrained(initial_params, path) do
    case File.read(path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)
        # Merge pretrained policy params with initial params
        # This handles cases where the architecture is slightly different
        merge_params(initial_params, checkpoint.policy_params || checkpoint.params)
      {:error, reason} ->
        Logger.warning("Could not load pretrained weights from #{path}: #{reason}")
        initial_params
    end
  end

  defp merge_params(target, source) do
    # Handle Axon.ModelState structs
    target_data = get_params_data(target)
    source_data = get_params_data(source)

    merged = do_merge_params(target_data, source_data)

    # Return in the same format as target
    case target do
      %Axon.ModelState{} = state -> %{state | data: merged}
      _ -> merged
    end
  end

  defp do_merge_params(target, source) when is_map(target) and is_map(source) do
    Map.merge(target, source, fn _k, t, s ->
      if is_map(t) and is_map(s) do
        do_merge_params(t, s)
      else
        s
      end
    end)
  end

  defp do_merge_params(_target, source), do: source

  defp deep_copy_params(params) do
    # Handle Axon.ModelState structs
    case params do
      %Axon.ModelState{} = state ->
        %{state | data: do_deep_copy(state.data)}
      data when is_map(data) ->
        do_deep_copy(data)
    end
  end

  defp do_deep_copy(data) when is_map(data) do
    Map.new(data, fn {k, v} ->
      cond do
        is_struct(v, Nx.Tensor) -> {k, Nx.backend_copy(v)}
        is_map(v) -> {k, do_deep_copy(v)}
        true -> {k, v}
      end
    end)
  end

  defp get_params_data(%Axon.ModelState{data: data}), do: data
  defp get_params_data(params) when is_map(params), do: params

  @doc """
  Create optimizer with learning rate schedule.

  Returns `{init_fn, update_fn}` tuple.
  """
  @spec create_optimizer(map()) :: {function(), function()}
  def create_optimizer(config) do
    Polaris.Optimizers.adam(learning_rate: config.learning_rate)
  end

  @doc """
  Perform a PPO update from collected rollout data.

  ## Parameters
    - `trainer` - The PPO trainer
    - `rollout` - Rollout data containing:
      - `:states` - [T, embed_size] tensor of states
      - `:actions` - Map of action tensors
      - `:rewards` - [T] tensor of rewards
      - `:dones` - [T] tensor of episode terminations
      - `:values` - [T] tensor of value estimates
      - `:log_probs` - [T] tensor of action log probabilities

  ## Returns
    Updated trainer and metrics.
  """
  @spec update(t(), map()) :: {t(), map()}
  def update(trainer, rollout) do
    config = trainer.config

    # Compute advantages and returns
    {advantages, returns} = compute_advantages(
      rollout.rewards,
      rollout.values,
      rollout.dones,
      config.gamma,
      config.gae_lambda
    )

    # Normalize advantages
    advantages = Value.normalize_advantages(advantages)

    # Store old parameters for ratio computation
    old_params = deep_copy_params(trainer.params)

    # Get old log probs and values
    old_log_probs = rollout.log_probs
    old_values = rollout.values

    # Run multiple epochs of minibatch updates
    {final_trainer, epoch_metrics} = run_ppo_epochs(
      trainer,
      rollout.states,
      rollout.actions,
      advantages,
      returns,
      old_log_probs,
      old_values,
      config.num_epochs,
      config.num_minibatches
    )

    # Update old params for next iteration
    final_trainer = %{final_trainer | old_params: old_params}

    # Aggregate metrics
    metrics = aggregate_epoch_metrics(epoch_metrics)
    metrics = Map.put(metrics, :timesteps, Nx.axis_size(rollout.states, 0))

    {final_trainer, metrics}
  end

  defp run_ppo_epochs(trainer, states, actions, advantages, returns, old_log_probs, old_values, num_epochs, num_minibatches) do
    batch_size = Nx.axis_size(states, 0)
    minibatch_size = div(batch_size, num_minibatches)

    {final_trainer, all_metrics} =
      Enum.reduce(1..num_epochs, {trainer, []}, fn _epoch, {acc_trainer, acc_metrics} ->
        # Shuffle indices
        indices = Nx.Random.shuffle(Nx.Random.key(System.system_time()), Nx.iota({batch_size}))
        |> elem(0)

        # Run minibatch updates
        {epoch_trainer, epoch_metrics} =
          Enum.reduce(0..(num_minibatches - 1), {acc_trainer, []}, fn mb_idx, {mb_trainer, mb_metrics} ->
            start_idx = mb_idx * minibatch_size
            mb_indices = Nx.slice(indices, [start_idx], [minibatch_size])

            # Gather minibatch data
            mb_states = gather_batch(states, mb_indices)
            mb_actions = gather_actions(actions, mb_indices)
            mb_advantages = Nx.take(advantages, mb_indices)
            mb_returns = Nx.take(returns, mb_indices)
            mb_old_log_probs = Nx.take(old_log_probs, mb_indices)
            mb_old_values = Nx.take(old_values, mb_indices)

            # Single minibatch update
            {new_trainer, metrics} = minibatch_update(
              mb_trainer,
              mb_states,
              mb_actions,
              mb_advantages,
              mb_returns,
              mb_old_log_probs,
              mb_old_values
            )

            {new_trainer, [metrics | mb_metrics]}
          end)

        {epoch_trainer, epoch_metrics ++ acc_metrics}
      end)

    {final_trainer, all_metrics}
  end

  defp gather_batch(tensor, indices) do
    Nx.take(tensor, indices)
  end

  defp gather_actions(actions, indices) do
    Map.new(actions, fn {k, v} ->
      {k, Nx.take(v, indices)}
    end)
  end

  defp minibatch_update(trainer, states, actions, advantages, returns, old_log_probs, old_values) do
    {_init_fn, predict_fn} = Axon.build(trainer.model)
    config = trainer.config

    # Loss function that returns only the scalar loss (for gradient computation)
    loss_fn = fn params ->
      %{policy: policy_logits, value: values} = predict_fn.(params, states)

      # Compute new log probs
      new_log_probs = ActorCritic.compute_log_probs(policy_logits, actions)

      # Policy loss (clipped PPO objective)
      ratio = Nx.exp(Nx.subtract(new_log_probs, old_log_probs))
      clipped_ratio = Nx.clip(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)

      surr1 = Nx.multiply(ratio, advantages)
      surr2 = Nx.multiply(clipped_ratio, advantages)
      policy_loss = Nx.negate(Nx.mean(Nx.min(surr1, surr2)))

      # Value loss (optionally clipped)
      value_loss = if config.value_clip do
        Value.clipped_value_loss(values, old_values, returns, config.clip_range)
      else
        Value.value_loss(values, returns)
      end

      # Entropy bonus
      entropy = ActorCritic.compute_entropy(policy_logits)

      # Total loss (this is what we differentiate)
      policy_loss
      |> Nx.add(Nx.multiply(config.value_coef, value_loss))
      |> Nx.subtract(Nx.multiply(config.entropy_coef, entropy))
    end

    # Compute gradients
    {loss, grads} = Nx.Defn.value_and_grad(loss_fn).(trainer.params)

    # Compute metrics in a separate forward pass (no gradients needed)
    %{policy: policy_logits, value: values} = predict_fn.(trainer.params, states)
    new_log_probs = ActorCritic.compute_log_probs(policy_logits, actions)
    ratio = Nx.exp(Nx.subtract(new_log_probs, old_log_probs))

    policy_loss = compute_policy_loss(ratio, advantages, config.clip_range)
    value_loss = Value.value_loss(values, returns)
    entropy = ActorCritic.compute_entropy(policy_logits)

    # Update parameters using Polaris
    {updates, new_optimizer_state} = trainer.optimizer.(
      grads,
      trainer.optimizer_state,
      trainer.params
    )

    new_params = Polaris.Updates.apply_updates(trainer.params, updates)

    new_trainer = %{trainer |
      params: new_params,
      optimizer_state: new_optimizer_state,
      step: trainer.step + 1
    }

    metrics = %{
      loss: Nx.to_number(loss),
      policy_loss: Nx.to_number(policy_loss),
      value_loss: Nx.to_number(value_loss),
      entropy: Nx.to_number(entropy),
      approx_kl: compute_approx_kl(ratio),
      clip_fraction: compute_clip_fraction(ratio, config.clip_range)
    }

    {new_trainer, metrics}
  end

  defp compute_policy_loss(ratio, advantages, clip_range) do
    clipped_ratio = Nx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr1 = Nx.multiply(ratio, advantages)
    surr2 = Nx.multiply(clipped_ratio, advantages)
    Nx.negate(Nx.mean(Nx.min(surr1, surr2)))
  end

  defp compute_approx_kl(ratio) do
    # Approximate KL divergence: E[(ratio - 1) - log(ratio)]
    kl = Nx.subtract(Nx.subtract(ratio, 1), Nx.log(ratio))
    Nx.to_number(Nx.mean(kl))
  end

  defp compute_clip_fraction(ratio, clip_range) do
    clipped = Nx.logical_or(
      Nx.less(ratio, 1.0 - clip_range),
      Nx.greater(ratio, 1.0 + clip_range)
    )
    Nx.to_number(Nx.mean(Nx.as_type(clipped, :f32)))
  end

  defp compute_advantages(rewards, values, dones, gamma, lambda) do
    Value.compute_gae(rewards, values, dones, gamma, lambda)
  end

  defp aggregate_epoch_metrics(metrics_list) do
    keys = [:loss, :policy_loss, :value_loss, :entropy, :approx_kl, :clip_fraction]

    Map.new(keys, fn key ->
      values = Enum.map(metrics_list, &Map.get(&1, key, 0.0))
      {key, Enum.sum(values) / max(length(values), 1)}
    end)
  end

  # ============================================================================
  # Rollout Collection
  # ============================================================================

  @doc """
  Collect a rollout by running the policy in an environment.

  This is a simplified version - real implementation would interface
  with the game environment through the bridge.
  """
  @spec collect_rollout(t(), function(), non_neg_integer()) :: map()
  def collect_rollout(trainer, step_fn, num_steps) do
    {_init_fn, predict_fn} = Axon.build(trainer.model)

    {states, actions, rewards, dones, values, log_probs, _final_state} =
      Enum.reduce(1..num_steps, {[], [], [], [], [], [], nil}, fn _step, acc ->
        {states_acc, actions_acc, rewards_acc, dones_acc, values_acc, log_probs_acc, state} = acc

        # Get current state from environment (or use provided step_fn)
        current_state = step_fn.(:get_state, state)

        # Get action from policy
        %{policy: policy_logits, value: value} = predict_fn.(trainer.params, current_state)

        # Sample action
        action = sample_action(policy_logits, trainer.config)
        log_prob = ActorCritic.compute_log_probs(policy_logits, action)

        # Step environment
        {next_state, reward, done} = step_fn.(:step, {state, action})

        # Accumulate
        {
          [current_state | states_acc],
          [action | actions_acc],
          [reward | rewards_acc],
          [done | dones_acc],
          [Nx.to_number(value) | values_acc],
          [Nx.to_number(log_prob) | log_probs_acc],
          next_state
        }
      end)

    # Stack into tensors
    %{
      states: Nx.stack(Enum.reverse(states)),
      actions: stack_actions(Enum.reverse(actions)),
      rewards: Nx.tensor(Enum.reverse(rewards), type: :f32),
      dones: Nx.tensor(Enum.reverse(dones), type: :f32),
      values: Nx.tensor(Enum.reverse(values) ++ [0.0], type: :f32),  # Bootstrap value
      log_probs: Nx.tensor(Enum.reverse(log_probs), type: :f32)
    }
  end

  defp sample_action(policy_logits, _config) do
    {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

    %{
      buttons: Policy.sample_buttons(buttons, false),
      main_x: Policy.sample_categorical(main_x, 1.0, false),
      main_y: Policy.sample_categorical(main_y, 1.0, false),
      c_x: Policy.sample_categorical(c_x, 1.0, false),
      c_y: Policy.sample_categorical(c_y, 1.0, false),
      shoulder: Policy.sample_categorical(shoulder, 1.0, false)
    }
  end

  defp stack_actions(actions_list) do
    # Stack each action component
    keys = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

    Map.new(keys, fn key ->
      tensors = Enum.map(actions_list, &Map.get(&1, key))
      {key, Nx.stack(tensors)}
    end)
  end

  # ============================================================================
  # Checkpointing
  # ============================================================================

  @doc """
  Save PPO checkpoint.
  """
  @spec save_checkpoint(t(), Path.t()) :: :ok | {:error, term()}
  def save_checkpoint(trainer, path) do
    checkpoint = %{
      params: trainer.params,
      optimizer_state: trainer.optimizer_state,
      config: trainer.config,
      step: trainer.step,
      timesteps: trainer.timesteps,
      metrics: trainer.metrics
    }

    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    case File.write(path, :erlang.term_to_binary(checkpoint)) do
      :ok ->
        Logger.info("Saved PPO checkpoint to #{path}")
        :ok
      error ->
        error
    end
  end

  @doc """
  Load PPO checkpoint.
  """
  @spec load_checkpoint(t(), Path.t()) :: {:ok, t()} | {:error, term()}
  def load_checkpoint(trainer, path) do
    case File.read(path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)

        new_trainer = %{trainer |
          params: checkpoint.params,
          old_params: deep_copy_params(checkpoint.params),
          optimizer_state: checkpoint.optimizer_state,
          step: checkpoint.step,
          timesteps: checkpoint.timesteps,
          metrics: checkpoint.metrics
        }

        Logger.info("Loaded PPO checkpoint from #{path}")
        {:ok, new_trainer}

      error ->
        error
    end
  end

  @doc """
  Export just the policy parameters for inference.
  """
  @spec export_policy(t(), Path.t()) :: :ok | {:error, term()}
  def export_policy(trainer, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    export = %{
      params: trainer.params,
      config: %{
        axis_buckets: trainer.config.axis_buckets,
        shoulder_buckets: trainer.config.shoulder_buckets
      }
    }

    File.write(path, :erlang.term_to_binary(export))
  end

  @doc """
  Load pretrained policy weights from imitation learning checkpoint.
  """
  @spec load_pretrained_policy(t(), Path.t()) :: {:ok, t()} | {:error, term()}
  def load_pretrained_policy(trainer, path) do
    case File.read(path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)
        pretrained_params = checkpoint.policy_params || checkpoint.params

        # Merge pretrained params into current params
        new_params = merge_params(trainer.params, pretrained_params)

        new_trainer = %{trainer |
          params: new_params,
          old_params: deep_copy_params(new_params)
        }

        {:ok, new_trainer}

      error ->
        error
    end
  end

  # ============================================================================
  # Inference
  # ============================================================================

  @doc """
  Get action from the trained policy.
  """
  @spec get_action(t(), Nx.Tensor.t(), keyword()) :: map()
  def get_action(trainer, state, opts \\ []) do
    {_init_fn, predict_fn} = Axon.build(trainer.model)
    deterministic = Keyword.get(opts, :deterministic, false)

    %{policy: policy_logits, value: value} = predict_fn.(trainer.params, state)

    {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

    %{
      buttons: Policy.sample_buttons(buttons, deterministic),
      main_x: Policy.sample_categorical(main_x, 1.0, deterministic),
      main_y: Policy.sample_categorical(main_y, 1.0, deterministic),
      c_x: Policy.sample_categorical(c_x, 1.0, deterministic),
      c_y: Policy.sample_categorical(c_y, 1.0, deterministic),
      shoulder: Policy.sample_categorical(shoulder, 1.0, deterministic),
      value: value
    }
  end

  @doc """
  Get action as ControllerState.
  """
  @spec get_controller_action(t(), Nx.Tensor.t(), keyword()) :: ExPhil.Bridge.ControllerState.t()
  def get_controller_action(trainer, state, opts \\ []) do
    samples = get_action(trainer, state, opts)

    Policy.to_controller_state(samples,
      axis_buckets: trainer.config.axis_buckets,
      shoulder_buckets: trainer.config.shoulder_buckets
    )
  end

  @doc """
  Get value estimate for a state.
  """
  @spec get_value(t(), Nx.Tensor.t()) :: float()
  def get_value(trainer, state) do
    {_init_fn, predict_fn} = Axon.build(trainer.model)
    %{value: value} = predict_fn.(trainer.params, state)

    value
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  # ============================================================================
  # Metrics
  # ============================================================================

  defp init_metrics do
    %{
      losses: [],
      policy_losses: [],
      value_losses: [],
      entropies: [],
      kls: [],
      clip_fractions: []
    }
  end

  @doc """
  Get training metrics summary.
  """
  @spec metrics_summary(t()) :: map()
  def metrics_summary(trainer) do
    %{
      step: trainer.step,
      timesteps: trainer.timesteps,
      recent_loss: safe_mean(Enum.take(trainer.metrics.losses, 10)),
      recent_entropy: safe_mean(Enum.take(trainer.metrics.entropies, 10)),
      recent_kl: safe_mean(Enum.take(trainer.metrics.kls, 10))
    }
  end

  defp safe_mean([]), do: 0.0
  defp safe_mean(list), do: Enum.sum(list) / length(list)
end
