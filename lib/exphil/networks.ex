defmodule ExPhil.Networks do
  @moduledoc """
  Neural network architectures for ExPhil.

  This module provides the main API for building and using neural networks
  for Melee AI. It includes policy networks (actor), value networks (critic),
  and combined actor-critic models.

  ## Architecture Overview

  ```
  Game State ──> Embeddings ──> Networks ──> Actions
                                   │
                                   ├── Policy (Actor)
                                   │     └── Autoregressive Controller Head
                                   │
                                   └── Value (Critic)
                                         └── Scalar Value Estimate
  ```

  ## Submodules

  - `ExPhil.Networks.Policy` - Policy network with autoregressive action head
  - `ExPhil.Networks.Value` - Value function network
  - `ExPhil.Networks.ActorCritic` - Combined actor-critic for PPO

  ## Usage

      # Get embedding size from config
      embed_config = ExPhil.Embeddings.config()
      embed_size = ExPhil.Embeddings.embedding_size(embed_config)

      # Build networks
      {policy, value} = ExPhil.Networks.build(embed_size: embed_size)

      # Or use combined model
      model = ExPhil.Networks.build_combined(embed_size: embed_size)

      # Initialize and run
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, embedded_state)

  """

  alias ExPhil.Networks.{Policy, Value, ActorCritic}

  @doc """
  Build separate policy and value networks.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Stick discretization (default: 16)
    - `:shoulder_buckets` - Trigger discretization (default: 4)

  ## Returns
    Tuple of {policy_model, value_model}
  """
  @spec build(keyword()) :: {Axon.t(), Axon.t()}
  defdelegate build(opts), to: ActorCritic

  @doc """
  Build a combined actor-critic model.

  More efficient for training as it shares the forward pass through
  the backbone.
  """
  @spec build_combined(keyword()) :: Axon.t()
  defdelegate build_combined(opts), to: ActorCritic

  @doc """
  Build just the policy network.
  """
  @spec build_policy(keyword()) :: Axon.t()
  defdelegate build_policy(opts), to: Policy, as: :build

  @doc """
  Build just the value network.
  """
  @spec build_value(keyword()) :: Axon.t()
  defdelegate build_value(opts), to: Value, as: :build

  @doc """
  Sample actions from a policy.

  ## Options
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:deterministic` - Use argmax instead of sampling (default: false)
  """
  @spec sample(map(), function(), Nx.Tensor.t(), keyword()) :: map()
  defdelegate sample(params, predict_fn, state, opts \\ []), to: Policy

  @doc """
  Convert sampled action indices to a ControllerState.
  """
  @spec to_controller_state(map(), keyword()) :: ExPhil.Bridge.ControllerState.t()
  defdelegate to_controller_state(samples, opts \\ []), to: Policy

  @doc """
  Compute PPO loss for training.
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
  defdelegate ppo_loss(
                new_logits,
                old_logits,
                actions,
                advantages,
                new_values,
                old_values,
                returns,
                opts \\ []
              ),
              to: ActorCritic

  @doc """
  Compute imitation learning loss.
  """
  @spec imitation_loss(map(), map()) :: Nx.Tensor.t()
  defdelegate imitation_loss(logits, targets), to: Policy

  @doc """
  Compute Generalized Advantage Estimation.
  """
  @spec compute_gae(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  defdelegate compute_gae(rewards, values, dones, gamma \\ 0.99, lambda \\ 0.95), to: Value

  @doc """
  Get default training configuration.
  """
  @spec default_config() :: map()
  defdelegate default_config(), to: ActorCritic

  @doc """
  Create an optimizer for training.
  """
  @spec create_optimizer(keyword()) :: Polaris.Updates.t()
  defdelegate create_optimizer(opts \\ []), to: ActorCritic

  @doc """
  Get the total action dimensions for the policy.
  """
  @spec action_dims(keyword()) :: non_neg_integer()
  defdelegate action_dims(opts \\ []), to: Policy, as: :total_action_dims

  @doc """
  Get output sizes for each controller component.
  """
  @spec output_sizes(keyword()) :: map()
  defdelegate output_sizes(opts \\ []), to: Policy
end
