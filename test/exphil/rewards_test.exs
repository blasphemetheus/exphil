defmodule ExPhil.RewardsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Rewards
  alias ExPhil.Rewards.{Standard, Shaped}
  alias ExPhil.Bridge.{GameState, Player}

  # Helper to create mock game state
  defp mock_state(opts \\ []) do
    player = %Player{
      character: 10,
      x: Keyword.get(opts, :player_x, 0.0),
      y: Keyword.get(opts, :player_y, 0.0),
      percent: Keyword.get(opts, :player_percent, 0.0),
      stock: Keyword.get(opts, :player_stock, 4),
      facing: 1,
      action: 14,
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: Keyword.get(opts, :player_hitstun, 0),
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }

    opponent = %Player{
      character: 2,
      x: Keyword.get(opts, :opponent_x, 50.0),
      y: Keyword.get(opts, :opponent_y, 0.0),
      percent: Keyword.get(opts, :opponent_percent, 0.0),
      stock: Keyword.get(opts, :opponent_stock, 4),
      facing: -1,
      action: 14,
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: Keyword.get(opts, :opponent_hitstun, 0),
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }

    %GameState{
      frame: Keyword.get(opts, :frame, 0),
      stage: 32,
      menu_state: 2,
      players: %{1 => player, 2 => opponent},
      projectiles: [],
      distance: abs(player.x - opponent.x)
    }
  end

  describe "Rewards.default_config/0" do
    test "returns config with all required keys" do
      config = Rewards.default_config()

      assert Map.has_key?(config, :stock_weight)
      assert Map.has_key?(config, :damage_weight)
      assert Map.has_key?(config, :approach_weight)
      assert Map.has_key?(config, :gamma)
    end
  end

  describe "Rewards.compute/3" do
    test "returns float reward" do
      prev = mock_state()
      curr = mock_state()

      reward = Rewards.compute(prev, curr, player_port: 1)

      assert is_float(reward)
    end

    test "clips reward to reasonable range" do
      # Create extreme scenario
      prev = mock_state(player_stock: 4, opponent_stock: 4)
      # Took all stocks
      curr = mock_state(player_stock: 4, opponent_stock: 0)

      reward = Rewards.compute(prev, curr, player_port: 1)

      assert reward <= 10.0
      assert reward >= -10.0
    end
  end

  describe "Rewards.compute_trajectory/2" do
    test "returns list of rewards" do
      states = [
        mock_state(frame: 0),
        mock_state(frame: 1, opponent_percent: 10.0),
        mock_state(frame: 2, opponent_percent: 20.0)
      ]

      rewards = Rewards.compute_trajectory(states, player_port: 1)

      assert length(rewards) == 2
      assert Enum.all?(rewards, &is_float/1)
    end
  end

  describe "Rewards.compute_returns/2" do
    test "computes discounted returns" do
      rewards = [1.0, 1.0, 1.0, 1.0]

      returns = Rewards.compute_returns(rewards, 0.99)

      assert length(returns) == 4
      # First return should be highest (includes all future rewards)
      assert hd(returns) > List.last(returns)
    end

    test "last return equals last reward" do
      rewards = [1.0, 2.0, 3.0]

      returns = Rewards.compute_returns(rewards, 0.99)

      assert_in_delta List.last(returns), 3.0, 0.001
    end
  end

  describe "Rewards.breakdown/3" do
    test "returns detailed breakdown" do
      prev = mock_state()
      curr = mock_state(opponent_percent: 15.0)

      breakdown = Rewards.breakdown(prev, curr, player_port: 1)

      assert Map.has_key?(breakdown, :standard)
      assert Map.has_key?(breakdown, :shaped)
      assert Map.has_key?(breakdown, :total)
    end
  end

  # ============================================================================
  # Standard Rewards
  # ============================================================================

  describe "Standard.compute_stock_reward/4" do
    test "returns +1 when opponent loses stock" do
      prev_player = %Player{stock: 4, percent: 0.0}
      curr_player = %Player{stock: 4, percent: 0.0}
      prev_opponent = %Player{stock: 4, percent: 150.0}
      curr_opponent = %Player{stock: 3, percent: 0.0}

      reward =
        Standard.compute_stock_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      assert reward == 1.0
    end

    test "returns -1 when player loses stock" do
      prev_player = %Player{stock: 4, percent: 150.0}
      curr_player = %Player{stock: 3, percent: 0.0}
      prev_opponent = %Player{stock: 4, percent: 0.0}
      curr_opponent = %Player{stock: 4, percent: 0.0}

      reward =
        Standard.compute_stock_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      assert reward == -1.0
    end

    test "returns 0 when no stock change" do
      prev_player = %Player{stock: 4, percent: 50.0}
      curr_player = %Player{stock: 4, percent: 60.0}
      prev_opponent = %Player{stock: 4, percent: 30.0}
      curr_opponent = %Player{stock: 4, percent: 30.0}

      reward =
        Standard.compute_stock_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      assert reward == 0.0
    end
  end

  describe "Standard.compute_damage_reward/4" do
    test "positive reward for dealing damage" do
      prev_player = %Player{stock: 4, percent: 0.0}
      curr_player = %Player{stock: 4, percent: 0.0}
      prev_opponent = %Player{stock: 4, percent: 0.0}
      curr_opponent = %Player{stock: 4, percent: 20.0}

      reward =
        Standard.compute_damage_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      assert reward == 20.0
    end

    test "negative reward for taking damage" do
      prev_player = %Player{stock: 4, percent: 0.0}
      curr_player = %Player{stock: 4, percent: 15.0}
      prev_opponent = %Player{stock: 4, percent: 0.0}
      curr_opponent = %Player{stock: 4, percent: 0.0}

      reward =
        Standard.compute_damage_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      assert reward == -15.0
    end

    test "net reward for damage trade" do
      prev_player = %Player{stock: 4, percent: 0.0}
      curr_player = %Player{stock: 4, percent: 10.0}
      prev_opponent = %Player{stock: 4, percent: 0.0}
      curr_opponent = %Player{stock: 4, percent: 25.0}

      reward =
        Standard.compute_damage_reward(prev_player, curr_player, prev_opponent, curr_opponent)

      # Dealt 25, took 10, net +15
      assert reward == 15.0
    end
  end

  describe "Standard.compute_win_reward/2" do
    test "returns +1 when player wins" do
      player = %Player{stock: 1, percent: 100.0}
      opponent = %Player{stock: 0, percent: 0.0}

      reward = Standard.compute_win_reward(player, opponent)

      assert reward == 1.0
    end

    test "returns -1 when player loses" do
      player = %Player{stock: 0, percent: 0.0}
      opponent = %Player{stock: 2, percent: 50.0}

      reward = Standard.compute_win_reward(player, opponent)

      assert reward == -1.0
    end

    test "returns 0 during game" do
      player = %Player{stock: 3, percent: 80.0}
      opponent = %Player{stock: 2, percent: 120.0}

      reward = Standard.compute_win_reward(player, opponent)

      assert reward == 0.0
    end
  end

  # ============================================================================
  # Shaped Rewards
  # ============================================================================

  describe "Shaped.compute_approach_reward/3" do
    test "positive reward for closing distance" do
      prev = mock_state(player_x: 0.0, opponent_x: 50.0)
      curr = mock_state(player_x: 10.0, opponent_x: 50.0)

      reward = Shaped.compute_approach_reward(prev, curr, 1)

      assert reward > 0
    end

    test "negative reward for moving away" do
      prev = mock_state(player_x: 0.0, opponent_x: 50.0)
      curr = mock_state(player_x: -10.0, opponent_x: 50.0)

      reward = Shaped.compute_approach_reward(prev, curr, 1)

      assert reward < 0
    end
  end

  describe "Shaped.compute_combo_reward/2" do
    test "positive reward for landing hit" do
      prev_opponent = %Player{hitstun_frames_left: 0, stock: 4, percent: 0.0}
      curr_opponent = %Player{hitstun_frames_left: 20, stock: 4, percent: 15.0}

      reward = Shaped.compute_combo_reward(prev_opponent, curr_opponent)

      assert reward > 0
    end

    test "small reward for maintaining combo" do
      prev_opponent = %Player{hitstun_frames_left: 15, stock: 4, percent: 30.0}
      curr_opponent = %Player{hitstun_frames_left: 10, stock: 4, percent: 30.0}

      reward = Shaped.compute_combo_reward(prev_opponent, curr_opponent)

      assert reward == 0.1
    end

    test "zero reward when no combo" do
      prev_opponent = %Player{hitstun_frames_left: 0, stock: 4, percent: 50.0}
      curr_opponent = %Player{hitstun_frames_left: 0, stock: 4, percent: 50.0}

      reward = Shaped.compute_combo_reward(prev_opponent, curr_opponent)

      assert reward == 0.0
    end
  end

  describe "Shaped.offstage?/1" do
    test "returns true when past left edge" do
      player = %Player{x: -100.0, y: 0.0, stock: 4, percent: 0.0}

      assert Shaped.offstage?(player)
    end

    test "returns true when past right edge" do
      player = %Player{x: 100.0, y: 0.0, stock: 4, percent: 0.0}

      assert Shaped.offstage?(player)
    end

    test "returns true when below stage" do
      player = %Player{x: 0.0, y: -50.0, stock: 4, percent: 0.0}

      assert Shaped.offstage?(player)
    end

    test "returns false when on stage" do
      player = %Player{x: 0.0, y: 0.0, stock: 4, percent: 0.0}

      refute Shaped.offstage?(player)
    end
  end

  describe "Shaped.compute_recovery_risk/1" do
    test "returns 0 when on stage" do
      player = %Player{x: 0.0, y: 10.0, stock: 4, percent: 0.0}

      risk = Shaped.compute_recovery_risk(player)

      assert risk == 0.0
    end

    test "returns positive penalty when off stage" do
      player = %Player{x: -100.0, y: -30.0, stock: 4, percent: 0.0}

      risk = Shaped.compute_recovery_risk(player)

      assert risk > 0
    end

    test "higher penalty for worse position" do
      near = %Player{x: -90.0, y: -10.0, stock: 4, percent: 0.0}
      far = %Player{x: -120.0, y: -80.0, stock: 4, percent: 0.0}

      near_risk = Shaped.compute_recovery_risk(near)
      far_risk = Shaped.compute_recovery_risk(far)

      assert far_risk > near_risk
    end
  end
end
