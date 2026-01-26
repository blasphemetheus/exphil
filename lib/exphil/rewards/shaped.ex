defmodule ExPhil.Rewards.Shaped do
  @moduledoc """
  Shaped reward functions to accelerate learning.

  Shaped rewards provide denser training signal by rewarding intermediate
  behaviors that lead to good outcomes. These help the agent learn faster
  but must be carefully designed to avoid reward hacking.

  ## Components

  - **Approach**: Reward for moving toward opponent (encourages aggression)
  - **Combo**: Reward for consecutive hits (encourages follow-ups)
  - **Edge Guard**: Reward for attacking recovering opponent
  - **Recovery Risk**: Penalty for being off-stage (encourages safe play)

  ## Character-Specific Considerations

  Some shaped rewards may need adjustment per character:
  - Mewtwo: Teleport recovery is very safe, reduce recovery_risk penalty
  - Link: Projectile zoning is valid, don't over-penalize distance
  - Ganondorf: Slow movement, be careful with approach reward

  """

  alias ExPhil.Bridge.{GameState, Player}

  # Stage boundaries (approximate for Final Destination)
  @stage_left -85.0
  @stage_right 85.0
  @stage_floor 0.0
  @offstage_threshold -20.0

  @doc """
  Compute shaped rewards for a state transition.

  ## Returns
    Map with `:approach`, `:combo`, `:edge_guard`, and `:recovery_risk` components.
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
      approach: compute_approach_reward(prev_state, curr_state, player_port),
      combo: compute_combo_reward(prev_opponent, curr_opponent),
      edge_guard:
        compute_edge_guard_reward(prev_player, curr_player, prev_opponent, curr_opponent),
      recovery_risk: compute_recovery_risk(curr_player)
    }
  end

  @doc """
  Reward for approaching the opponent.

  Positive reward when distance decreases, negative when it increases.
  Scaled to be small so it doesn't dominate other rewards.
  """
  @spec compute_approach_reward(GameState.t(), GameState.t(), integer()) :: float()
  def compute_approach_reward(prev_state, curr_state, player_port) do
    prev_distance = get_distance(prev_state, player_port)
    curr_distance = get_distance(curr_state, player_port)

    # Reward for closing distance
    distance_delta = prev_distance - curr_distance

    # Normalize by typical approach speed (~2-3 units per frame)
    distance_delta / 3.0
  end

  @doc """
  Reward for combo continuation.

  Detects when opponent is in hitstun and rewards keeping them there.
  Higher reward for longer combos (more hitstun frames).
  """
  @spec compute_combo_reward(Player.t(), Player.t()) :: float()
  def compute_combo_reward(prev_opponent, curr_opponent) do
    prev_hitstun = prev_opponent.hitstun_frames_left || 0
    curr_hitstun = curr_opponent.hitstun_frames_left || 0

    cond do
      # New hit landed (hitstun increased)
      curr_hitstun > prev_hitstun ->
        # Bonus scaled by hitstun (stronger hits = more reward)
        min(1.0, curr_hitstun / 30.0)

      # Maintaining combo (opponent still in hitstun)
      curr_hitstun > 0 and prev_hitstun > 0 ->
        0.1

      # Combo ended or no combo
      true ->
        0.0
    end
  end

  @doc """
  Reward for edge guarding.

  Bonus for attacking an opponent who is off-stage and trying to recover.
  """
  @spec compute_edge_guard_reward(Player.t(), Player.t(), Player.t(), Player.t()) :: float()
  def compute_edge_guard_reward(_prev_player, curr_player, prev_opponent, curr_opponent) do
    opponent_offstage = offstage?(curr_opponent)
    opponent_hit = hit_landed?(prev_opponent, curr_opponent)
    player_onstage = not offstage?(curr_player)

    if opponent_offstage and opponent_hit and player_onstage do
      # Big bonus for edge guard hits
      1.0
    else
      0.0
    end
  end

  @doc """
  Penalty for being in a risky recovery situation.

  Penalizes being off-stage, with higher penalty for being lower/further out.
  """
  @spec compute_recovery_risk(Player.t()) :: float()
  def compute_recovery_risk(player) do
    if offstage?(player) do
      # Penalty based on how far off-stage and how low
      horizontal_risk = horizontal_danger(player)
      vertical_risk = vertical_danger(player)

      # Combined risk (0-1 scale each, so max ~2)
      horizontal_risk + vertical_risk
    else
      0.0
    end
  end

  # ============================================================================
  # Position Analysis
  # ============================================================================

  @doc """
  Check if a player is off-stage (past ledge).
  """
  @spec offstage?(Player.t()) :: boolean()
  def offstage?(player) do
    player.x < @stage_left or player.x > @stage_right or player.y < @offstage_threshold
  end

  @doc """
  Check if player is in a dangerous horizontal position.
  """
  @spec horizontal_danger(Player.t()) :: float()
  def horizontal_danger(player) do
    cond do
      player.x < @stage_left ->
        min(1.0, (@stage_left - player.x) / 50.0)

      player.x > @stage_right ->
        min(1.0, (player.x - @stage_right) / 50.0)

      true ->
        0.0
    end
  end

  @doc """
  Check if player is in a dangerous vertical position.
  """
  @spec vertical_danger(Player.t()) :: float()
  def vertical_danger(player) do
    if player.y < @stage_floor do
      min(1.0, (@stage_floor - player.y) / 100.0)
    else
      0.0
    end
  end

  @doc """
  Check if player is near the ledge.
  """
  @spec near_ledge?(Player.t()) :: boolean()
  def near_ledge?(player) do
    at_left_edge = player.x < @stage_left + 15 and player.x > @stage_left - 10
    at_right_edge = player.x > @stage_right - 15 and player.x < @stage_right + 10
    near_floor = player.y < @stage_floor + 20 and player.y > @stage_floor - 30

    (at_left_edge or at_right_edge) and near_floor
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp get_player(%GameState{players: players}, port) do
    Map.get(players, port, %Player{
      x: 0.0,
      y: 0.0,
      stock: 0,
      percent: 0.0,
      hitstun_frames_left: 0
    })
  end

  defp get_distance(%GameState{players: players}, player_port) do
    opponent_port = get_opponent_port(player_port)
    player = Map.get(players, player_port, %Player{x: 0.0, y: 0.0})
    opponent = Map.get(players, opponent_port, %Player{x: 50.0, y: 0.0})

    dx = player.x - opponent.x
    dy = player.y - opponent.y
    :math.sqrt(dx * dx + dy * dy)
  end

  defp get_opponent_port(1), do: 2
  defp get_opponent_port(2), do: 1
  defp get_opponent_port(3), do: 4
  defp get_opponent_port(4), do: 3

  defp hit_landed?(prev_opponent, curr_opponent) do
    prev_hitstun = prev_opponent.hitstun_frames_left || 0
    curr_hitstun = curr_opponent.hitstun_frames_left || 0
    curr_hitstun > prev_hitstun
  end
end
