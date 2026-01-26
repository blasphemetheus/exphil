defmodule ExPhil.Networks.Value do
  @moduledoc """
  Value network (critic) for estimating state values.

  The value network estimates the expected discounted return from a given
  state, used as a baseline for variance reduction in policy gradient
  methods (PPO, A2C, etc.).

  ## Architecture

  ```
  Embedded State [batch, embed_size]
        │
        ▼
  ┌─────────────┐
  │   Backbone  │  (shared or separate from policy)
  │  [hidden]   │
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  Value Head │
  │  [scalar]   │
  └─────────────┘
  ```

  ## Usage

      # Build standalone value network
      model = ExPhil.Networks.Value.build(embed_size: 1024)

      # Or build with shared backbone
      model = ExPhil.Networks.Value.build_from_backbone(backbone)

      # Forward pass
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
      values = predict_fn.(params, embedded_state)

  """

  require Axon

  # Default architecture hyperparameters
  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  @doc """
  Build the value network.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    An Axon model that outputs a scalar value estimate.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Input layer
    input = Axon.input("state", shape: {nil, embed_size})

    # Build backbone
    backbone = build_backbone(input, hidden_sizes, activation, dropout)

    # Value head
    build_value_head(backbone)
  end

  @doc """
  Build value network using an existing backbone.

  Useful for sharing parameters between policy and value networks.
  """
  @spec build_from_backbone(Axon.t()) :: Axon.t()
  def build_from_backbone(backbone) do
    build_value_head(backbone)
  end

  @doc """
  Build the backbone network.
  """
  @spec build_backbone(Axon.t(), list(), atom(), float()) :: Axon.t()
  def build_backbone(input, hidden_sizes, activation, dropout) do
    Enum.reduce(hidden_sizes, input, fn size, acc ->
      acc
      |> Axon.dense(size, name: "value_backbone_dense_#{size}")
      |> Axon.activation(activation)
      |> Axon.dropout(rate: dropout)
    end)
  end

  @doc """
  Build the value head that outputs a scalar.
  """
  @spec build_value_head(Axon.t()) :: Axon.t()
  def build_value_head(backbone) do
    backbone
    |> Axon.dense(128, name: "value_hidden")
    |> Axon.relu()
    |> Axon.dense(1, name: "value_output")
    # [batch, 1] -> [batch]
    |> Axon.nx(fn x -> Nx.squeeze(x, axes: [-1]) end, name: "value_squeeze")
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute value loss (mean squared error).

  ## Parameters
    - `predicted` - Predicted values from the network [batch]
    - `targets` - Target values (returns or TD targets) [batch]

  ## Returns
    MSE loss scalar.
  """
  @spec value_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def value_loss(predicted, targets) do
    diff = Nx.subtract(predicted, targets)
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Compute Huber loss (smooth L1) for more robust value estimation.

  Less sensitive to outliers than MSE.

  ## Parameters
    - `predicted` - Predicted values [batch]
    - `targets` - Target values [batch]
    - `delta` - Huber threshold (default: 1.0)
  """
  @spec huber_loss(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def huber_loss(predicted, targets, delta \\ 1.0) do
    diff = Nx.subtract(predicted, targets)
    abs_diff = Nx.abs(diff)

    # Quadratic for |diff| < delta, linear otherwise
    quadratic = Nx.multiply(0.5, Nx.multiply(diff, diff))
    linear = Nx.subtract(Nx.multiply(delta, abs_diff), Nx.multiply(0.5, delta * delta))

    loss = Nx.select(Nx.less(abs_diff, delta), quadratic, linear)
    Nx.mean(loss)
  end

  @doc """
  Compute clipped value loss for PPO.

  Prevents value function from changing too much between updates.

  ## Parameters
    - `new_values` - Values from updated network
    - `old_values` - Values from previous network
    - `targets` - Target returns
    - `clip_range` - PPO clip range (default: 0.2)
  """
  @spec clipped_value_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def clipped_value_loss(new_values, old_values, targets, clip_range \\ 0.2) do
    # Clipped value estimate
    clipped_values =
      Nx.add(
        old_values,
        Nx.clip(
          Nx.subtract(new_values, old_values),
          -clip_range,
          clip_range
        )
      )

    # Value losses
    loss_unclipped =
      Nx.multiply(
        Nx.subtract(new_values, targets),
        Nx.subtract(new_values, targets)
      )

    loss_clipped =
      Nx.multiply(
        Nx.subtract(clipped_values, targets),
        Nx.subtract(clipped_values, targets)
      )

    # Take the maximum (pessimistic)
    loss = Nx.max(loss_unclipped, loss_clipped)
    Nx.multiply(0.5, Nx.mean(loss))
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Compute Generalized Advantage Estimation (GAE).

  GAE provides a trade-off between bias and variance in advantage estimation.

  ## Parameters
    - `rewards` - Rewards [time_steps]
    - `values` - Value estimates [time_steps + 1] (includes bootstrap)
    - `dones` - Episode termination flags [time_steps]
    - `gamma` - Discount factor (default: 0.99)
    - `lambda` - GAE lambda (default: 0.95)

  ## Returns
    Advantages [time_steps] and returns [time_steps]
  """
  @spec compute_gae(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def compute_gae(rewards, values, dones, gamma \\ 0.99, lambda \\ 0.95) do
    # values has shape [time_steps + 1]
    # rewards, dones have shape [time_steps]
    time_steps = Nx.axis_size(rewards, 0)

    # Compute TD residuals
    # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    next_values = Nx.slice(values, [1], [time_steps])
    current_values = Nx.slice(values, [0], [time_steps])
    not_dones = Nx.subtract(1, dones)

    deltas =
      Nx.add(
        rewards,
        Nx.subtract(
          Nx.multiply(gamma, Nx.multiply(next_values, not_dones)),
          current_values
        )
      )

    # Compute GAE using reverse accumulation
    # This is done outside defn for now (would need Nx.while for proper implementation)
    advantages = compute_gae_loop(deltas, not_dones, gamma, lambda, time_steps)

    # Returns = advantages + values
    returns = Nx.add(advantages, current_values)

    {advantages, returns}
  end

  defp compute_gae_loop(deltas, not_dones, gamma, lambda, _time_steps) do
    # Convert to lists for reverse iteration
    delta_list = Nx.to_flat_list(deltas)
    not_done_list = Nx.to_flat_list(not_dones)

    # Compute advantages in reverse
    {advantages, _} =
      delta_list
      |> Enum.zip(not_done_list)
      |> Enum.reverse()
      |> Enum.reduce({[], 0.0}, fn {delta, not_done}, {acc, gae} ->
        gae = delta + gamma * lambda * not_done * gae
        {[gae | acc], gae}
      end)

    Nx.tensor(advantages, type: :f32)
  end

  @doc """
  Normalize advantages for more stable training.
  """
  @spec normalize_advantages(Nx.Tensor.t()) :: Nx.Tensor.t()
  def normalize_advantages(advantages) do
    mean = Nx.mean(advantages)
    std = Nx.standard_deviation(advantages)
    # Add small epsilon for numerical stability
    Nx.divide(Nx.subtract(advantages, mean), Nx.add(std, 1.0e-8))
  end
end
