defmodule ExPhil.BridgeTest do
  use ExUnit.Case, async: false

  alias ExPhil.Bridge.{GameState, Player, ControllerState, ControllerInput, Projectile}

  describe "GameState" do
    test "in_game?/1 returns true for in-game states" do
      gs = %GameState{menu_state: 2}  # IN_GAME
      assert GameState.in_game?(gs)

      gs = %GameState{menu_state: 3}  # SUDDEN_DEATH
      assert GameState.in_game?(gs)
    end

    test "in_game?/1 returns false for menu states" do
      gs = %GameState{menu_state: 0}  # Some menu state
      refute GameState.in_game?(gs)
    end

    test "get_player/2 retrieves player by port" do
      player1 = %Player{x: 0.0, y: 0.0, percent: 0.0}
      player2 = %Player{x: 10.0, y: 0.0, percent: 50.0}

      gs = %GameState{
        players: %{1 => player1, 2 => player2}
      }

      assert GameState.get_player(gs, 1) == player1
      assert GameState.get_player(gs, 2) == player2
      assert GameState.get_player(gs, 3) == nil
    end
  end

  describe "Player" do
    test "dying?/1 detects death states" do
      # Action states 0-10 are death animations
      assert Player.dying?(%Player{action: 0})
      assert Player.dying?(%Player{action: 5})
      assert Player.dying?(%Player{action: 10})
      refute Player.dying?(%Player{action: 11})
      refute Player.dying?(%Player{action: 100})
    end

    test "offstage?/1 detects offstage positions" do
      assert Player.offstage?(%Player{x: 100.0, y: 0.0})
      assert Player.offstage?(%Player{x: -100.0, y: 0.0})
      assert Player.offstage?(%Player{x: 0.0, y: -10.0})
      refute Player.offstage?(%Player{x: 0.0, y: 0.0})
      refute Player.offstage?(%Player{x: 50.0, y: 10.0})
    end

    test "in_hitstun?/1 detects hitstun" do
      assert Player.in_hitstun?(%Player{hitstun_frames_left: 10})
      assert Player.in_hitstun?(%Player{hitstun_frames_left: 1})
      refute Player.in_hitstun?(%Player{hitstun_frames_left: 0})
    end

    test "shielding?/1 detects shield states" do
      assert Player.shielding?(%Player{action: 178})  # SHIELD_START
      assert Player.shielding?(%Player{action: 179})  # SHIELD
      assert Player.shielding?(%Player{action: 180})  # SHIELD_RELEASE
      refute Player.shielding?(%Player{action: 14})   # WAIT
    end
  end

  describe "ControllerState" do
    test "neutral/0 creates centered neutral state" do
      cs = ControllerState.neutral()

      assert cs.main_stick == %{x: 0.5, y: 0.5}
      assert cs.c_stick == %{x: 0.5, y: 0.5}
      assert cs.l_shoulder == 0.0
      refute cs.button_a
      refute cs.button_b
    end

    test "to_input/1 converts to input format" do
      cs = %ControllerState{
        main_stick: %{x: 1.0, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.5,
        r_shoulder: 0.0,
        button_a: true,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      input = ControllerState.to_input(cs)

      assert input.main_stick == %{x: 1.0, y: 0.5}
      assert input.shoulder == 0.5
      assert input.buttons.a == true
      assert input.buttons.b == false
    end
  end

  describe "ControllerInput" do
    test "neutral/0 creates neutral input" do
      input = ControllerInput.neutral()

      assert input.main_stick == %{x: 0.5, y: 0.5}
      assert input.c_stick == %{x: 0.5, y: 0.5}
      assert input.shoulder == 0.0
      assert input.buttons == %{}
    end

    test "button/1 creates button press input" do
      input = ControllerInput.button(:a)
      assert input.buttons == %{a: true}

      input = ControllerInput.button(:b)
      assert input.buttons == %{b: true}
    end

    test "main_stick/2 creates stick input" do
      input = ControllerInput.main_stick(1.0, 0.5)
      assert input.main_stick == %{x: 1.0, y: 0.5}
    end

    test "main_stick/2 clamps values to [0, 1]" do
      input = ControllerInput.main_stick(-0.5, 1.5)
      assert input.main_stick == %{x: 0.0, y: 1.0}
    end

    test "combine/1 merges multiple inputs" do
      inputs = [
        ControllerInput.main_stick(1.0, 0.5),
        ControllerInput.button(:a),
        ControllerInput.button(:b)
      ]

      combined = ControllerInput.combine(inputs)

      assert combined.main_stick == %{x: 1.0, y: 0.5}
      assert combined.buttons == %{a: true, b: true}
    end

    test "helper functions create correct inputs" do
      assert ControllerInput.jump().buttons == %{x: true}
      assert ControllerInput.a().buttons == %{a: true}
      assert ControllerInput.b().buttons == %{b: true}
      assert ControllerInput.grab().buttons == %{z: true}

      assert ControllerInput.left().main_stick == %{x: 0.0, y: 0.5}
      assert ControllerInput.right().main_stick == %{x: 1.0, y: 0.5}
      assert ControllerInput.crouch().main_stick == %{x: 0.5, y: 0.0}

      assert ControllerInput.shield().shoulder == 1.0
    end
  end

  describe "Projectile" do
    test "struct holds projectile data" do
      proj = %Projectile{
        owner: 1,
        x: 50.0,
        y: 10.0,
        type: 10,
        subtype: 0,
        speed_x: 5.0,
        speed_y: 0.0
      }

      assert proj.owner == 1
      assert proj.x == 50.0
      assert proj.speed_x == 5.0
    end
  end
end
