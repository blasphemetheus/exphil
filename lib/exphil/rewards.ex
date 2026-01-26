defmodule ExPhil.Rewards do
  @moduledoc """
  Reward computation for Melee RL training.

  Rewards are the training signal for reinforcement learning. This module
  provides both standard game-outcome rewards and shaped rewards that
  encourage specific behaviors.

  ## Reward Types

  ### Standard Rewards
  Basic rewards based on game outcomes:
  - Stock differential: +1 for taking stock, -1 for losing stock
  - Damage differential: reward proportional to damage dealt vs taken
  - Win/loss: large bonus/penalty for game outcome

  ### Shaped Rewards
  Additional rewards to speed up learning:
  - Approach: reward for moving toward opponent
  - Combo: reward for consecutive hits
  - Edge guard: reward for attacking recovering opponent
  - Recovery: penalty for being off-stage

  ## Usage

      # Compute reward for a frame transition
      reward = ExPhil.Rewards.compute(prev_state, curr_state, player_port: 1)

      # Use specific reward function
      reward = ExPhil.Rewards.Standard.compute(prev_state, curr_state, player_port: 1)

      # Combine multiple reward sources
      config = ExPhil.Rewards.default_config()
      reward = ExPhil.Rewards.compute_weighted(prev_state, curr_state, config)

  """

  alias ExPhil.Rewards.{Standard, Shaped}
  alias ExPhil.Bridge.GameState

  @doc """
  Default reward configuration with weights for each component.
  """
  @spec default_config() :: map()
  def default_config do
    %{
      # Standard rewards
      stock_weight: 1.0,
      damage_weight: 0.01,
      win_bonus: 5.0,

      # Shaped rewards
      approach_weight: 0.001,
      combo_weight: 0.05,
      edge_guard_weight: 0.1,
      recovery_penalty: 0.02,

      # Discount and normalization
      gamma: 0.99,
      normalize: true
    }
  end

  @doc """
  Compute the total reward for a state transition.

  ## Parameters
    - `prev_state` - Previous game state
    - `curr_state` - Current game state
    - `opts` - Options including `:player_port` (required)

  ## Returns
    Total reward as a float.
  """
  @spec compute(GameState.t(), GameState.t(), keyword()) :: float()
  def compute(prev_state, curr_state, opts \\ []) do
    config = Keyword.get(opts, :config, default_config())
    compute_weighted(prev_state, curr_state, config, opts)
  end

  @doc """
  Compute weighted combination of all reward components.
  """
  @spec compute_weighted(GameState.t(), GameState.t(), map(), keyword()) :: float()
  def compute_weighted(prev_state, curr_state, config, opts \\ []) do
    player_port = Keyword.fetch!(opts, :player_port)

    # Standard rewards
    standard = Standard.compute(prev_state, curr_state, player_port: player_port)

    # Shaped rewards
    shaped = Shaped.compute(prev_state, curr_state, player_port: player_port)

    # Weighted sum
    reward =
      standard.stock * config.stock_weight +
        standard.damage * config.damage_weight +
        standard.win * config.win_bonus +
        shaped.approach * config.approach_weight +
        shaped.combo * config.combo_weight +
        shaped.edge_guard * config.edge_guard_weight -
        shaped.recovery_risk * config.recovery_penalty

    if config.normalize do
      # Clip to reasonable range
      max(-10.0, min(10.0, reward))
    else
      reward
    end
  end

  @doc """
  Compute rewards for a sequence of states (trajectory).

  Returns a list of rewards, one per transition.
  """
  @spec compute_trajectory([GameState.t()], keyword()) :: [float()]
  def compute_trajectory(states, opts \\ []) when length(states) > 1 do
    states
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [prev, curr] ->
      compute(prev, curr, opts)
    end)
  end

  @doc """
  Compute discounted returns from rewards.

  Given rewards [r0, r1, r2, ...], computes returns:
  G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
  """
  @spec compute_returns([float()], float()) :: [float()]
  def compute_returns(rewards, gamma \\ 0.99) do
    rewards
    |> Enum.reverse()
    |> Enum.reduce({[], 0.0}, fn reward, {returns, running} ->
      g = reward + gamma * running
      {[g | returns], g}
    end)
    |> elem(0)
  end

  @doc """
  Get a breakdown of reward components for debugging/logging.
  """
  @spec breakdown(GameState.t(), GameState.t(), keyword()) :: map()
  def breakdown(prev_state, curr_state, opts \\ []) do
    player_port = Keyword.fetch!(opts, :player_port)

    %{
      standard: Standard.compute(prev_state, curr_state, player_port: player_port),
      shaped: Shaped.compute(prev_state, curr_state, player_port: player_port),
      total: compute(prev_state, curr_state, opts)
    }
  end
end
