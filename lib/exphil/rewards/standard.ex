defmodule ExPhil.Rewards.Standard do
  @moduledoc """
  Standard reward functions based on game outcomes.

  These are the fundamental rewards derived directly from the game state:
  stock changes, damage dealt/taken, and win/loss conditions.

  ## Components

  - **Stock**: +1 for taking opponent's stock, -1 for losing own stock
  - **Damage**: Proportional to damage dealt minus damage taken
  - **Win**: Large bonus/penalty for game end

  ## Usage

      rewards = Standard.compute(prev_state, curr_state, player_port: 1)
      # => %{stock: 1.0, damage: 0.15, win: 0.0}

  """

  alias ExPhil.Bridge.{GameState, Player}

  @doc """
  Compute standard rewards for a state transition.

  ## Returns
    Map with `:stock`, `:damage`, and `:win` components.
  """
  @spec compute(GameState.t(), GameState.t(), keyword()) :: map()
  def compute(prev_state, curr_state, opts \\ []) do
    player_port = Keyword.fetch!(opts, :player_port)
    opponent_port = get_opponent_port(player_port)

    prev_player = get_player(prev_state, player_port)
    curr_player = get_player(curr_state, player_port)
    prev_opponent = get_player(prev_state, opponent_port)
    curr_opponent = get_player(curr_state, opponent_port)

    %{
      stock: compute_stock_reward(prev_player, curr_player, prev_opponent, curr_opponent),
      damage: compute_damage_reward(prev_player, curr_player, prev_opponent, curr_opponent),
      win: compute_win_reward(curr_player, curr_opponent)
    }
  end

  @doc """
  Compute stock differential reward.

  +1 for taking opponent's stock, -1 for losing own stock.
  """
  @spec compute_stock_reward(Player.t(), Player.t(), Player.t(), Player.t()) :: float()
  def compute_stock_reward(prev_player, curr_player, prev_opponent, curr_opponent) do
    player_lost = stock_lost?(prev_player, curr_player)
    opponent_lost = stock_lost?(prev_opponent, curr_opponent)

    cond do
      opponent_lost and not player_lost -> 1.0
      player_lost and not opponent_lost -> -1.0
      opponent_lost and player_lost -> 0.0  # Both died (rare, could be explosion)
      true -> 0.0
    end
  end

  @doc """
  Compute damage differential reward.

  Reward is (damage dealt to opponent - damage taken) normalized.
  """
  @spec compute_damage_reward(Player.t(), Player.t(), Player.t(), Player.t()) :: float()
  def compute_damage_reward(prev_player, curr_player, prev_opponent, curr_opponent) do
    # Handle stock resets (percent goes back to 0)
    damage_taken = if stock_lost?(prev_player, curr_player) do
      0.0  # Don't penalize damage on death frame
    else
      max(0.0, curr_player.percent - prev_player.percent)
    end

    damage_dealt = if stock_lost?(prev_opponent, curr_opponent) do
      # Credit remaining damage needed to kill
      prev_opponent.percent
    else
      max(0.0, curr_opponent.percent - prev_opponent.percent)
    end

    # Return differential (positive = good)
    damage_dealt - damage_taken
  end

  @doc """
  Compute win/loss reward.

  Large bonus for winning, large penalty for losing.
  """
  @spec compute_win_reward(Player.t(), Player.t()) :: float()
  def compute_win_reward(player, opponent) do
    cond do
      opponent.stock <= 0 and player.stock > 0 -> 1.0   # Player won
      player.stock <= 0 and opponent.stock > 0 -> -1.0  # Player lost
      true -> 0.0  # Game ongoing or tie
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp get_player(%GameState{players: players}, port) do
    Map.get(players, port, %Player{stock: 0, percent: 0.0})
  end

  defp get_opponent_port(1), do: 2
  defp get_opponent_port(2), do: 1
  defp get_opponent_port(3), do: 4
  defp get_opponent_port(4), do: 3

  defp stock_lost?(prev_player, curr_player) do
    curr_player.stock < prev_player.stock
  end
end
