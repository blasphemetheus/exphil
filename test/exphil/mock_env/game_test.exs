defmodule ExPhil.MockEnv.GameTest do
  use ExUnit.Case, async: true

  alias ExPhil.MockEnv.Game

  describe "new/1" do
    test "creates game with two players at starting positions" do
      game = Game.new()

      assert game.p1.x == -40.0
      assert game.p2.x == 40.0
      assert game.p1.stock == 4
      assert game.p2.stock == 4
      assert game.frame == 0
      assert game.done == false
    end
  end

  describe "physics" do
    test "gravity pulls airborne player down" do
      game = Game.new()
      # Put P1 in air
      game = %{game | p1: %{game.p1 | y: 50.0, on_ground: false, action: 26}}

      # Step with no input
      game = Game.step(game, nil, nil)

      # Should have fallen (vel_y decreased, y decreased)
      assert game.p1.vel_y < 0
      assert game.p1.y < 50.0
    end

    test "player lands on ground" do
      game = Game.new()
      # Put P1 just above ground, falling
      game = %{game | p1: %{game.p1 | y: 0.5, vel_y: -1.0, on_ground: false, action: 26}}

      # Step
      game = Game.step(game, nil, nil)

      # Should land
      assert game.p1.on_ground == true
      assert game.p1.y == 0.0
    end

    test "player can jump" do
      game = Game.new()
      action = %{stick_x: 0.0, stick_y: 0.0, button_x: true, button_y: false,
                 button_a: false, button_b: false, button_l: false, button_r: false}

      # Press jump - enters jump squat
      game = Game.step(game, action, nil)
      assert game.p1.action == 24  # Jump squat

      # Continue holding through jump squat (3 frames: 0, 1, 2)
      game = Game.step(game, action, nil)  # frame 1
      game = Game.step(game, action, nil)  # frame 2
      game = Game.step(game, action, nil)  # frame 3 - jump executes
      game = Game.step(game, action, nil)  # frame 4 - now airborne

      # Should now be airborne
      assert game.p1.on_ground == false
      assert game.p1.vel_y > 0
    end

    test "player can move horizontally" do
      game = Game.new()
      action = %{stick_x: 1.0, stick_y: 0.0, button_x: false, button_y: false,
                 button_a: false, button_b: false, button_l: false, button_r: false}

      initial_x = game.p1.x
      game = Game.step(game, action, nil)

      assert game.p1.x > initial_x
      assert game.p1.facing == 1
    end

    test "player can fast fall" do
      game = Game.new()
      # Put P1 in air, already falling
      game = %{game | p1: %{game.p1 | y: 50.0, vel_y: -0.5, on_ground: false, action: 26}}

      # Tap down
      action = %{stick_x: 0.0, stick_y: -1.0, button_x: false, button_y: false,
                 button_a: false, button_b: false, button_l: false, button_r: false}
      game = Game.step(game, action, nil)

      # Should fast fall
      assert game.p1.fastfalling == true
      assert game.p1.vel_y <= -3.0
    end
  end

  describe "blast zones" do
    test "player dies when hitting bottom blast zone" do
      game = Game.new()
      # Put P1 below blast zone
      game = %{game | p1: %{game.p1 | y: -150.0, on_ground: false}}

      game = Game.step(game, nil, nil)

      # Should lose a stock and respawn
      assert game.p1.stock == 3
      assert game.p1.y > 0  # Respawned above stage
    end

    test "player dies when hitting side blast zone" do
      game = Game.new()
      # Put P1 past side blast zone
      game = %{game | p1: %{game.p1 | x: 230.0, on_ground: false}}

      game = Game.step(game, nil, nil)

      assert game.p1.stock == 3
    end

    test "game ends when player loses all stocks" do
      game = Game.new()
      # Set P1 to 1 stock and put in blast zone
      game = %{game | p1: %{game.p1 | stock: 1, y: -150.0, on_ground: false}}

      game = Game.step(game, nil, nil)

      assert game.done == true
      assert game.winner == 2
    end
  end

  describe "to_game_state/1" do
    test "converts to GameState for embedding" do
      game = Game.new()
      game_state = Game.to_game_state(game)

      assert game_state.frame == 0
      assert game_state.stage == 2
      assert game_state.menu_state == 2
      assert Map.has_key?(game_state.players, 1)
      assert Map.has_key?(game_state.players, 2)
      assert game_state.players[1].x == -40.0
    end
  end

  describe "double jump" do
    test "player can double jump in air" do
      game = Game.new()
      # Put P1 in air after using first jump
      game = %{game | p1: %{game.p1 |
        y: 30.0,
        vel_y: -0.5,
        on_ground: false,
        jumps_left: 1,
        action: 26
      }}

      action = %{stick_x: 0.0, stick_y: 0.0, button_x: true, button_y: false,
                 button_a: false, button_b: false, button_l: false, button_r: false}

      game = Game.step(game, action, nil)

      # Should double jump
      assert game.p1.vel_y > 0
      assert game.p1.jumps_left == 0
    end

    test "player cannot triple jump" do
      game = Game.new()
      # P1 in air with no jumps left
      game = %{game | p1: %{game.p1 |
        y: 30.0,
        vel_y: -0.5,
        on_ground: false,
        jumps_left: 0,
        action: 26
      }}

      initial_vel_y = game.p1.vel_y
      action = %{stick_x: 0.0, stick_y: 0.0, button_x: true, button_y: false,
                 button_a: false, button_b: false, button_l: false, button_r: false}

      game = Game.step(game, action, nil)

      # Should not jump (gravity still applies)
      assert game.p1.vel_y < initial_vel_y
    end
  end
end
