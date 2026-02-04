defmodule ExPhil.Networks.WorldModel do
  @moduledoc """
  World Model for Melee game state prediction.

  Predicts the next game state given current state and action, enabling:
  - Model-based RL (planning ahead)
  - Imagined rollouts for training
  - Counterfactual reasoning ("what if I had done X?")

  ## Architecture

  ```
  Current State [batch, state_dim]
       │
       └──┬── Action [batch, action_dim]
          ▼
  ┌───────────────────────────────────────┐
  │           State-Action Encoder         │
  │  concat(state, action) → hidden        │
  └───────────────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────────┐
  │           Dynamics Network             │
  │  hidden → next_state_delta             │
  └───────────────────────────────────────┘
          │
          ▼
  ┌───────────────────────────────────────┐
  │           Reward Predictor             │
  │  hidden → reward, done                 │
  └───────────────────────────────────────┘
          │
          ▼
  Predicted: (next_state, reward, done)
  ```

  ## Training

  The world model is trained to minimize:
  - State prediction MSE: ||s_pred - s_actual||²
  - Reward prediction MSE: ||r_pred - r_actual||²
  - Done prediction BCE: BCE(done_pred, done_actual)

  ## Usage

      # Build world model
      model = WorldModel.build(
        state_dim: 287,
        action_dim: 13,
        hidden_size: 512
      )

      # Training
      {next_state, reward, done} = predict_fn.(params, %{
        "state" => current_state,
        "action" => action
      })

      # Planning (imagined rollouts)
      trajectory = WorldModel.rollout(params, predict_fn, initial_state, policy_fn, horizon: 60)

  ## For Melee

  Key predictions:
  - Player positions (x, y) - continuous
  - Action states - discrete (one-hot or embedding)
  - Damage percentages - continuous
  - Stock counts - discrete
  - Hitstun/shieldstun frames - discrete
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 512
  @default_num_layers 3
  @default_dropout 0.1
  @default_activation :gelu

  @doc """
  Build the world model.

  ## Options

  **Architecture:**
    - `:state_dim` - Dimension of state embedding (required)
    - `:action_dim` - Dimension of action embedding (required)
    - `:hidden_size` - Hidden layer size (default: 512)
    - `:num_layers` - Number of hidden layers (default: 3)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:activation` - Activation function (default: :gelu)

  **Outputs:**
    - `:predict_reward` - Whether to predict reward (default: true)
    - `:predict_done` - Whether to predict episode termination (default: true)
    - `:residual_prediction` - Predict state delta instead of full state (default: true)

  ## Returns

  An Axon model that outputs a map with keys:
  - `:next_state` - Predicted next state [batch, state_dim]
  - `:reward` - Predicted reward [batch, 1] (if enabled)
  - `:done` - Predicted done probability [batch, 1] (if enabled)
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    state_dim = Keyword.fetch!(opts, :state_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)
    predict_reward = Keyword.get(opts, :predict_reward, true)
    predict_done = Keyword.get(opts, :predict_done, true)
    residual = Keyword.get(opts, :residual_prediction, true)

    # Inputs
    state_input = Axon.input("state", shape: {nil, state_dim})
    action_input = Axon.input("action", shape: {nil, action_dim})

    # Concatenate state and action
    combined = Axon.concatenate([state_input, action_input], name: "state_action_concat")

    # Dynamics encoder
    hidden = build_encoder(combined, hidden_size, num_layers, activation, dropout)

    # State prediction head
    state_delta = Axon.dense(hidden, state_dim, name: "state_delta_head")

    next_state =
      if residual do
        # Residual prediction: predict change, add to current state
        Axon.add(state_input, state_delta, name: "next_state_residual")
      else
        # Direct prediction
        state_delta
      end

    # Build output container
    outputs = %{next_state: next_state}

    # Optional reward head
    outputs =
      if predict_reward do
        reward =
          hidden
          |> Axon.dense(64, activation: activation, name: "reward_hidden")
          |> Axon.dense(1, name: "reward_head")

        Map.put(outputs, :reward, reward)
      else
        outputs
      end

    # Optional done head
    outputs =
      if predict_done do
        done =
          hidden
          |> Axon.dense(64, activation: activation, name: "done_hidden")
          |> Axon.dense(1, activation: :sigmoid, name: "done_head")

        Map.put(outputs, :done, done)
      else
        outputs
      end

    # Return as container
    Axon.container(outputs)
  end

  @doc """
  Build an ensemble world model for uncertainty estimation.

  Uses multiple independent models and aggregates predictions.
  Useful for model-based planning with uncertainty penalties.

  ## Options
    - `:num_models` - Number of ensemble members (default: 5)
    - All options from `build/1`

  ## Returns

  A map with `:models` (list of Axon models) and `:aggregate_fn`.
  """
  @spec build_ensemble(keyword()) :: map()
  def build_ensemble(opts \\ []) do
    num_models = Keyword.get(opts, :num_models, 5)

    models =
      for i <- 1..num_models do
        build(Keyword.put(opts, :name_prefix, "ensemble_#{i}"))
      end

    %{
      models: models,
      num_models: num_models,
      aggregate_fn: &aggregate_ensemble_predictions/2
    }
  end

  @doc """
  Build a recurrent world model for multi-step prediction.

  Uses LSTM/GRU to maintain state across predictions, useful for
  capturing longer-range dynamics.

  ## Options
    - `:recurrent_type` - :lstm or :gru (default: :lstm)
    - All options from `build/1`
  """
  @spec build_recurrent(keyword()) :: Axon.t()
  def build_recurrent(opts \\ []) do
    state_dim = Keyword.fetch!(opts, :state_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    recurrent_type = Keyword.get(opts, :recurrent_type, :lstm)
    predict_reward = Keyword.get(opts, :predict_reward, true)
    predict_done = Keyword.get(opts, :predict_done, true)

    # Inputs: sequences for recurrent processing
    state_seq = Axon.input("state_sequence", shape: {nil, nil, state_dim})
    action_seq = Axon.input("action_sequence", shape: {nil, nil, action_dim})

    # Concatenate along feature dimension
    combined = Axon.concatenate([state_seq, action_seq], axis: 2, name: "seq_concat")

    # Recurrent processing
    recurrent_out =
      case recurrent_type do
        :lstm ->
          combined
          |> Axon.lstm(hidden_size, name: "dynamics_lstm", return_sequences: true)
          |> elem(0)

        :gru ->
          combined
          |> Axon.gru(hidden_size, name: "dynamics_gru", return_sequences: true)
          |> elem(0)
      end

    # Predict state deltas for each timestep
    state_delta = Axon.dense(recurrent_out, state_dim, name: "state_delta_seq")

    # Add residual: next_state = current_state + delta
    next_states = Axon.add(state_seq, state_delta, name: "next_states_residual")

    outputs = %{next_states: next_states}

    outputs =
      if predict_reward do
        rewards = Axon.dense(recurrent_out, 1, name: "reward_seq_head")
        Map.put(outputs, :rewards, rewards)
      else
        outputs
      end

    outputs =
      if predict_done do
        dones = Axon.dense(recurrent_out, 1, activation: :sigmoid, name: "done_seq_head")
        Map.put(outputs, :dones, dones)
      else
        outputs
      end

    Axon.container(outputs)
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute world model loss.

  ## Parameters
    - `predictions` - Map with :next_state, :reward, :done
    - `targets` - Map with :next_state, :reward, :done
    - `opts` - Options:
      - `:state_weight` - Weight for state loss (default: 1.0)
      - `:reward_weight` - Weight for reward loss (default: 0.5)
      - `:done_weight` - Weight for done loss (default: 0.1)

  ## Returns
    Total weighted loss.
  """
  @spec compute_loss(map(), map(), keyword()) :: Nx.Tensor.t()
  def compute_loss(predictions, targets, opts \\ []) do
    state_weight = Keyword.get(opts, :state_weight, 1.0)
    reward_weight = Keyword.get(opts, :reward_weight, 0.5)
    done_weight = Keyword.get(opts, :done_weight, 0.1)

    # State prediction loss (MSE)
    state_loss =
      predictions.next_state
      |> Nx.subtract(targets.next_state)
      |> Nx.pow(2)
      |> Nx.mean()

    total = Nx.multiply(state_weight, state_loss)

    # Reward prediction loss (MSE)
    total =
      if Map.has_key?(predictions, :reward) and Map.has_key?(targets, :reward) do
        reward_loss =
          predictions.reward
          |> Nx.subtract(targets.reward)
          |> Nx.pow(2)
          |> Nx.mean()

        Nx.add(total, Nx.multiply(reward_weight, reward_loss))
      else
        total
      end

    # Done prediction loss (BCE)
    if Map.has_key?(predictions, :done) and Map.has_key?(targets, :done) do
      done_loss = binary_cross_entropy(predictions.done, targets.done)
      Nx.add(total, Nx.multiply(done_weight, done_loss))
    else
      total
    end
  end

  # ============================================================================
  # Planning / Rollout Functions
  # ============================================================================

  @doc """
  Perform imagined rollout using the world model.

  ## Parameters
    - `params` - World model parameters
    - `predict_fn` - World model prediction function
    - `initial_state` - Starting state [batch, state_dim]
    - `policy_fn` - Function: state -> action
    - `opts` - Options:
      - `:horizon` - Number of steps to roll out (default: 60)
      - `:discount` - Reward discount factor (default: 0.99)

  ## Returns
    Map with:
    - `:states` - Predicted state trajectory [horizon, batch, state_dim]
    - `:actions` - Actions taken [horizon, batch, action_dim]
    - `:rewards` - Predicted rewards [horizon, batch, 1]
    - `:total_reward` - Discounted sum of rewards [batch]
  """
  @spec rollout(map(), function(), Nx.Tensor.t(), function(), keyword()) :: map()
  def rollout(params, predict_fn, initial_state, policy_fn, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, 60)
    discount = Keyword.get(opts, :discount, 0.99)

    {states, actions, rewards} =
      Enum.reduce(0..(horizon - 1), {[initial_state], [], []}, fn _step, {states, actions, rewards} ->
        current_state = List.last(states)

        # Get action from policy
        action = policy_fn.(current_state)

        # Predict next state
        prediction = predict_fn.(params, %{
          "state" => current_state,
          "action" => action
        })

        next_state = prediction.next_state
        reward = Map.get(prediction, :reward, Nx.broadcast(0.0, {Nx.axis_size(current_state, 0), 1}))

        {states ++ [next_state], actions ++ [action], rewards ++ [reward]}
      end)

    # Stack tensors
    states_tensor = Nx.stack(states)
    actions_tensor = Nx.stack(actions)
    rewards_tensor = Nx.stack(rewards)

    # Compute discounted total reward
    discounts = Nx.pow(discount, Nx.iota({horizon})) |> Nx.reshape({horizon, 1, 1})
    total_reward = Nx.sum(Nx.multiply(rewards_tensor, discounts), axes: [0, 2])

    %{
      states: states_tensor,
      actions: actions_tensor,
      rewards: rewards_tensor,
      total_reward: total_reward
    }
  end

  @doc """
  Model Predictive Control (MPC) using the world model.

  Optimizes action sequence by simulating multiple rollouts and
  selecting the best action.

  ## Parameters
    - `params` - World model parameters
    - `predict_fn` - World model prediction function
    - `current_state` - Current state [batch, state_dim]
    - `candidate_actions` - Candidate action sequences [num_candidates, horizon, action_dim]
    - `opts` - Options:
      - `:horizon` - Planning horizon (default: 10)
      - `:num_samples` - Number of action samples for CEM (default: 64)

  ## Returns
    Best first action [batch, action_dim]
  """
  @spec mpc(map(), function(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def mpc(params, predict_fn, current_state, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, 10)
    num_samples = Keyword.get(opts, :num_samples, 64)
    action_dim = Keyword.get(opts, :action_dim, 13)

    batch_size = Nx.axis_size(current_state, 0)

    # Sample random action sequences
    key = Nx.Random.key(:erlang.unique_integer())
    {action_sequences, _} = Nx.Random.uniform(key, shape: {num_samples, horizon, action_dim})

    # Evaluate each action sequence
    rewards =
      for sample_idx <- 0..(num_samples - 1) do
        actions = Nx.slice_along_axis(action_sequences, sample_idx, 1, axis: 0) |> Nx.squeeze(axes: [0])

        # Simulate rollout with this action sequence
        {_final_state, total_reward} =
          Enum.reduce(0..(horizon - 1), {current_state, Nx.broadcast(0.0, {batch_size})}, fn step, {state, acc_reward} ->
            action = Nx.slice_along_axis(actions, step, 1, axis: 0) |> Nx.squeeze(axes: [0])
            # Broadcast action to batch
            action = Nx.broadcast(action, {batch_size, action_dim})

            prediction = predict_fn.(params, %{
              "state" => state,
              "action" => action
            })

            reward = Map.get(prediction, :reward, Nx.broadcast(0.0, {batch_size, 1})) |> Nx.squeeze(axes: [1])
            {prediction.next_state, Nx.add(acc_reward, reward)}
          end)

        total_reward
      end

    # Stack rewards and find best
    rewards_tensor = Nx.stack(rewards)  # [num_samples, batch]
    _best_idx_per_batch = Nx.argmax(rewards_tensor, axis: 0)  # [batch]

    # Return first action of best sequence
    # For simplicity, just return the globally best action
    global_best = Nx.argmax(Nx.mean(rewards_tensor, axes: [1])) |> Nx.to_number()
    Nx.slice_along_axis(action_sequences, global_best, 1, axis: 0)
    |> Nx.squeeze(axes: [0])
    |> Nx.slice_along_axis(0, 1, axis: 0)
    |> Nx.squeeze(axes: [0])
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp build_encoder(input, hidden_size, num_layers, activation, dropout) do
    Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
      acc
      |> Axon.dense(hidden_size, activation: activation, name: "encoder_#{layer_idx}")
      |> then(fn x ->
        if dropout > 0 do
          Axon.dropout(x, rate: dropout, name: "encoder_dropout_#{layer_idx}")
        else
          x
        end
      end)
    end)
    |> Axon.layer_norm(name: "encoder_final_norm")
  end

  defp aggregate_ensemble_predictions(predictions_list, _opts) do
    # Average predictions across ensemble members
    num_models = length(predictions_list)

    next_states = Enum.map(predictions_list, & &1.next_state)
    mean_state = Nx.stack(next_states) |> Nx.mean(axes: [0])

    # Variance for uncertainty estimation
    variance =
      next_states
      |> Nx.stack()
      |> Nx.subtract(mean_state)
      |> Nx.pow(2)
      |> Nx.mean(axes: [0])

    %{
      next_state: mean_state,
      uncertainty: variance,
      num_models: num_models
    }
  end

  defp binary_cross_entropy(predictions, targets) do
    # Clip to avoid log(0)
    eps = 1.0e-7
    preds = Nx.clip(predictions, eps, 1.0 - eps)

    Nx.negate(
      Nx.add(
        Nx.multiply(targets, Nx.log(preds)),
        Nx.multiply(Nx.subtract(1.0, targets), Nx.log(Nx.subtract(1.0, preds)))
      )
    )
    |> Nx.mean()
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Get recommended defaults for Melee world model.
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      state_dim: 287,
      action_dim: 13,
      hidden_size: 512,
      num_layers: 3,
      dropout: 0.1,
      activation: :gelu,
      predict_reward: true,
      predict_done: true,
      residual_prediction: true
    ]
  end

  @doc """
  Calculate approximate parameter count.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    state_dim = Keyword.get(opts, :state_dim, 287)
    action_dim = Keyword.get(opts, :action_dim, 13)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    input_size = state_dim + action_dim

    # First layer
    first = input_size * hidden_size + hidden_size

    # Hidden layers
    hidden = (num_layers - 1) * (hidden_size * hidden_size + hidden_size)

    # Output heads
    state_head = hidden_size * state_dim + state_dim
    reward_head = hidden_size * 64 + 64 + 64 * 1 + 1
    done_head = hidden_size * 64 + 64 + 64 * 1 + 1

    first + hidden + state_head + reward_head + done_head
  end
end
