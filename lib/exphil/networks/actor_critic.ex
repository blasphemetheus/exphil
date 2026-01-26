defmodule ExPhil.Networks.ActorCritic do
  @moduledoc """
  Combined Actor-Critic network for PPO training.

  This module provides a unified interface for training both the policy
  (actor) and value function (critic) networks, optionally with shared
  backbone parameters.

  ## Architecture Options

  ### Separate Networks
  ```
  State ──> Policy Backbone ──> Policy Head ──> Actions
  State ──> Value Backbone  ──> Value Head  ──> Value
  ```

  ### Shared Backbone
  ```
  State ──> Shared Backbone ──┬──> Policy Head ──> Actions
                              └──> Value Head  ──> Value
  ```

  Shared backbones can improve sample efficiency but may create
  interference between policy and value gradients.

  ## Usage

      # Build actor-critic model
      {policy, value} = ActorCritic.build(embed_size: 1024, shared_backbone: true)

      # Or use the combined model
      model = ActorCritic.build_combined(embed_size: 1024)

      # Training step
      {loss, grads} = ActorCritic.train_step(params, batch, opts)

  """

  alias ExPhil.Networks.{Policy, Value}

  # Default training hyperparameters
  @default_gamma 0.99
  @default_gae_lambda 0.95
  @default_clip_range 0.2
  @default_value_coef 0.5
  @default_entropy_coef 0.01

  @doc """
  Build separate policy and value networks.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - Hidden layer sizes (default: [512, 512])
    - `:shared_backbone` - Share backbone between networks (default: false)
    - Other options passed to Policy and Value builders

  ## Returns
    Tuple of {policy_model, value_model}
  """
  @spec build(keyword()) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    policy = Policy.build(opts)
    value = Value.build(opts)
    {policy, value}
  end

  @doc """
  Build a combined model that outputs both policy logits and value.

  This is more efficient for training as it shares the forward pass.
  """
  @spec build_combined(keyword()) :: Axon.t()
  def build_combined(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, [512, 512])
    activation = Keyword.get(opts, :activation, :relu)
    dropout = Keyword.get(opts, :dropout, 0.1)
    axis_buckets = Keyword.get(opts, :axis_buckets, 16)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, 4)

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # Input
    input = Axon.input("state", shape: {nil, embed_size})

    # Shared backbone
    backbone = Policy.build_backbone(input, hidden_sizes, activation, dropout)

    # Policy heads
    buttons =
      backbone
      |> Axon.dense(64, name: "buttons_hidden")
      |> Axon.relu()
      |> Axon.dense(8, name: "buttons_logits")

    main_x =
      backbone
      |> Axon.dense(64, name: "main_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_x_logits")

    main_y =
      backbone
      |> Axon.dense(64, name: "main_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_y_logits")

    c_x =
      backbone
      |> Axon.dense(64, name: "c_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_x_logits")

    c_y =
      backbone
      |> Axon.dense(64, name: "c_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_y_logits")

    shoulder =
      backbone
      |> Axon.dense(32, name: "shoulder_hidden")
      |> Axon.relu()
      |> Axon.dense(shoulder_size, name: "shoulder_logits")

    # Value head
    value =
      backbone
      |> Axon.dense(128, name: "value_hidden")
      |> Axon.relu()
      |> Axon.dense(1, name: "value_output")
      |> Axon.nx(fn x -> Nx.squeeze(x, axes: [-1]) end, name: "value_squeeze")

    # Combined output
    policy_logits = Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
    Axon.container(%{policy: policy_logits, value: value})
  end

  # ============================================================================
  # PPO Loss Functions
  # ============================================================================

  @doc """
  Compute the full PPO loss.

  ## Parameters
    - `new_logits` - Policy logits from current parameters
    - `old_logits` - Policy logits from old parameters (detached)
    - `actions` - Actions taken
    - `advantages` - Advantage estimates
    - `new_values` - Value estimates from current parameters
    - `old_values` - Value estimates from old parameters
    - `returns` - Target returns
    - `opts` - Training options

  ## Returns
    Map with total loss and component losses.
  """
  @spec ppo_loss(
          map(),
          map(),
          map(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: map()
  def ppo_loss(
        new_logits,
        old_logits,
        actions,
        advantages,
        new_values,
        old_values,
        returns,
        opts \\ []
      ) do
    clip_range = Keyword.get(opts, :clip_range, @default_clip_range)
    value_coef = Keyword.get(opts, :value_coef, @default_value_coef)
    entropy_coef = Keyword.get(opts, :entropy_coef, @default_entropy_coef)

    # Policy loss
    {policy_loss, approx_kl, clip_fraction} =
      policy_loss(new_logits, old_logits, actions, advantages, clip_range)

    # Value loss
    value_loss = Value.clipped_value_loss(new_values, old_values, returns, clip_range)

    # Entropy bonus (encourages exploration)
    entropy = compute_entropy(new_logits)

    # Total loss
    total_loss =
      policy_loss
      |> Nx.add(Nx.multiply(value_coef, value_loss))
      |> Nx.subtract(Nx.multiply(entropy_coef, entropy))

    %{
      total: total_loss,
      policy: policy_loss,
      value: value_loss,
      entropy: entropy,
      approx_kl: approx_kl,
      clip_fraction: clip_fraction
    }
  end

  @doc """
  Compute clipped policy loss for PPO.
  """
  @spec policy_loss(map(), map(), map(), Nx.Tensor.t(), float()) ::
          {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def policy_loss(new_logits, old_logits, actions, advantages, clip_range) do
    # Compute log probabilities
    new_log_probs = compute_log_probs(new_logits, actions)
    old_log_probs = compute_log_probs(old_logits, actions)

    # Probability ratio
    ratio = Nx.exp(Nx.subtract(new_log_probs, old_log_probs))

    # Clipped ratio
    clipped_ratio = Nx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Surrogate objectives
    surr1 = Nx.multiply(ratio, advantages)
    surr2 = Nx.multiply(clipped_ratio, advantages)

    # Take the minimum (pessimistic bound)
    policy_loss = Nx.negate(Nx.mean(Nx.min(surr1, surr2)))

    # Diagnostics
    approx_kl = Nx.mean(Nx.subtract(old_log_probs, new_log_probs))

    clip_fraction =
      Nx.mean(
        Nx.as_type(
          Nx.not_equal(ratio, clipped_ratio),
          :f32
        )
      )

    {policy_loss, approx_kl, clip_fraction}
  end

  @doc """
  Compute log probability of actions under the policy.
  """
  @spec compute_log_probs(map(), map()) :: Nx.Tensor.t()
  def compute_log_probs(logits, actions) do
    {buttons, main_x, main_y, c_x, c_y, shoulder} = logits

    # Button log probs (Bernoulli)
    button_log_probs = compute_bernoulli_log_probs(buttons, actions.buttons)

    # Categorical log probs for sticks/shoulder
    main_x_log_probs = compute_categorical_log_prob(main_x, actions.main_x)
    main_y_log_probs = compute_categorical_log_prob(main_y, actions.main_y)
    c_x_log_probs = compute_categorical_log_prob(c_x, actions.c_x)
    c_y_log_probs = compute_categorical_log_prob(c_y, actions.c_y)
    shoulder_log_probs = compute_categorical_log_prob(shoulder, actions.shoulder)

    # Sum all log probs
    button_log_probs
    |> Nx.add(main_x_log_probs)
    |> Nx.add(main_y_log_probs)
    |> Nx.add(c_x_log_probs)
    |> Nx.add(c_y_log_probs)
    |> Nx.add(shoulder_log_probs)
  end

  defp compute_bernoulli_log_probs(logits, actions) do
    # actions: [batch, 8] binary
    # logits: [batch, 8]
    # log_sigmoid(x) = -log(1 + exp(-x)) = x - log(1 + exp(x)) for numerical stability
    log_probs = log_sigmoid(logits)
    log_one_minus_probs = log_sigmoid(Nx.negate(logits))

    # log p(action) = action * log(p) + (1 - action) * log(1 - p)
    per_button =
      Nx.add(
        Nx.multiply(actions, log_probs),
        Nx.multiply(Nx.subtract(1, actions), log_one_minus_probs)
      )

    # Sum over buttons
    Nx.sum(per_button, axes: [1])
  end

  # Numerically stable log sigmoid: log(sigmoid(x)) = -softplus(-x)
  defp log_sigmoid(x) do
    Nx.negate(softplus(Nx.negate(x)))
  end

  # softplus(x) = log(1 + exp(x)), numerically stable version
  defp softplus(x) do
    # For large x, softplus(x) ≈ x
    # For small x, softplus(x) ≈ log(1 + exp(x))
    Nx.select(
      Nx.greater(x, 20),
      x,
      Nx.log(Nx.add(1, Nx.exp(x)))
    )
  end

  defp compute_categorical_log_prob(logits, actions) do
    # Compute log softmax
    log_probs = log_softmax(logits)

    # Gather log prob of selected action
    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)

    # One-hot encode actions
    actions_one_hot =
      Nx.equal(
        Nx.iota({batch_size, num_classes}, axis: 1),
        Nx.reshape(actions, {batch_size, 1})
      )

    # Extract selected log probs
    Nx.sum(Nx.multiply(log_probs, actions_one_hot), axes: [1])
  end

  defp log_softmax(logits) do
    max_val = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_val)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end

  @doc """
  Compute entropy of the policy distribution.
  """
  @spec compute_entropy(map()) :: Nx.Tensor.t()
  def compute_entropy(logits) do
    {buttons, main_x, main_y, c_x, c_y, shoulder} = logits

    # Bernoulli entropy for buttons
    button_probs = Nx.sigmoid(buttons)

    button_entropy =
      Nx.negate(
        Nx.sum(
          Nx.add(
            Nx.multiply(button_probs, Nx.log(Nx.add(button_probs, 1.0e-10))),
            Nx.multiply(
              Nx.subtract(1, button_probs),
              Nx.log(Nx.add(Nx.subtract(1, button_probs), 1.0e-10))
            )
          ),
          axes: [1]
        )
      )

    # Categorical entropy for sticks/shoulder
    main_x_entropy = categorical_entropy(main_x)
    main_y_entropy = categorical_entropy(main_y)
    c_x_entropy = categorical_entropy(c_x)
    c_y_entropy = categorical_entropy(c_y)
    shoulder_entropy = categorical_entropy(shoulder)

    # Average total entropy
    Nx.mean(
      button_entropy
      |> Nx.add(main_x_entropy)
      |> Nx.add(main_y_entropy)
      |> Nx.add(c_x_entropy)
      |> Nx.add(c_y_entropy)
      |> Nx.add(shoulder_entropy)
    )
  end

  defp categorical_entropy(logits) do
    probs = Nx.exp(log_softmax(logits))
    log_probs = log_softmax(logits)
    Nx.negate(Nx.sum(Nx.multiply(probs, log_probs), axes: [1]))
  end

  # ============================================================================
  # Training Utilities
  # ============================================================================

  @doc """
  Get default training configuration.
  """
  @spec default_config() :: map()
  def default_config do
    %{
      gamma: @default_gamma,
      gae_lambda: @default_gae_lambda,
      clip_range: @default_clip_range,
      value_coef: @default_value_coef,
      entropy_coef: @default_entropy_coef,
      max_grad_norm: 0.5,
      learning_rate: 3.0e-4,
      batch_size: 64,
      num_epochs: 10,
      num_minibatches: 4
    }
  end

  @doc """
  Create an optimizer for training.
  """
  @spec create_optimizer(keyword()) :: Polaris.Updates.t()
  def create_optimizer(opts \\ []) do
    learning_rate = Keyword.get(opts, :learning_rate, 3.0e-4)
    max_grad_norm = Keyword.get(opts, :max_grad_norm, 0.5)

    Polaris.Optimizers.adam(learning_rate: learning_rate)
    |> Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm)
  end
end
