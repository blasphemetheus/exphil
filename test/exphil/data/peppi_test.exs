defmodule ExPhil.Data.PeppiTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.Peppi

  describe "parse/2" do
    test "returns error for non-existent file" do
      assert {:error, message} = Peppi.parse("nonexistent.slp")
      assert message =~ "Failed to open file"
    end

    @tag :external
    test "parses a valid .slp file" do
      # Skip if no replay file is available
      replay_path = System.get_env("TEST_REPLAY_PATH")

      if replay_path && File.exists?(replay_path) do
        assert {:ok, replay} = Peppi.parse(replay_path)

        # Check structure
        assert %Peppi.ParsedReplay{} = replay
        assert is_list(replay.frames)
        assert %Peppi.ReplayMeta{} = replay.metadata
        assert is_integer(replay.metadata.stage)
        assert is_integer(replay.metadata.duration_frames)
        assert is_list(replay.metadata.players)

        # Check player metadata
        for player <- replay.metadata.players do
          assert %Peppi.PlayerMeta{} = player
          assert is_integer(player.port)
          assert is_integer(player.character)
          assert is_binary(player.character_name)
        end

        # Check frame data
        if replay.frames != [] do
          [frame | _] = replay.frames
          assert %Peppi.GameFrame{} = frame
          assert is_integer(frame.frame_number)
          assert is_map(frame.players)

          for {port, player_frame} <- frame.players do
            assert is_integer(port)
            assert %Peppi.PlayerFrame{} = player_frame
            assert is_float(player_frame.x)
            assert is_float(player_frame.y)
            assert is_float(player_frame.percent)
            assert is_integer(player_frame.stock)

            # Check controller
            assert %Peppi.Controller{} = player_frame.controller
            assert is_float(player_frame.controller.main_stick_x)
            assert is_float(player_frame.controller.main_stick_y)
            assert is_boolean(player_frame.controller.button_a)
          end
        end
      end
    end
  end

  describe "metadata/1" do
    test "returns error for non-existent file" do
      assert {:error, message} = Peppi.metadata("nonexistent.slp")
      assert message =~ "Failed to open file"
    end
  end

  describe "parse_many/2" do
    test "returns empty list for empty input" do
      assert [] = Peppi.parse_many([])
    end

    test "handles non-existent files gracefully" do
      results = Peppi.parse_many(["nonexistent1.slp", "nonexistent2.slp"])
      assert length(results) == 2
      assert Enum.all?(results, &match?({:error, _}, &1))
    end
  end

  describe "to_training_frames/2" do
    test "converts parsed replay to training format" do
      # Create a mock parsed replay structure
      controller = %Peppi.Controller{
        main_stick_x: 0.5,
        main_stick_y: 0.5,
        c_stick_x: 0.5,
        c_stick_y: 0.5,
        l_trigger: 0.0,
        r_trigger: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_start: false,
        button_d_up: false,
        button_d_down: false,
        button_d_left: false,
        button_d_right: false
      }

      player_frame = %Peppi.PlayerFrame{
        # Mewtwo
        character: 10,
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        facing: 1,
        action: 14,
        action_frame: 0.0,
        invulnerable: false,
        jumps_left: 2,
        on_ground: true,
        shield_strength: 60.0,
        hitstun_frames_left: 0.0,
        speed_air_x_self: 0.0,
        speed_ground_x_self: 0.0,
        speed_y_self: 0.0,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0,
        controller: controller
      }

      game_frame = %Peppi.GameFrame{
        frame_number: 0,
        players: %{1 => player_frame, 2 => player_frame}
      }

      metadata = %Peppi.ReplayMeta{
        path: "test.slp",
        # Final Destination
        stage: 32,
        duration_frames: 1,
        players: [
          %Peppi.PlayerMeta{port: 1, character: 10, character_name: "Mewtwo", tag: nil},
          %Peppi.PlayerMeta{port: 2, character: 2, character_name: "Fox", tag: nil}
        ]
      }

      replay = %Peppi.ParsedReplay{
        frames: [game_frame],
        metadata: metadata
      }

      training_frames = Peppi.to_training_frames(replay)

      assert length(training_frames) == 1
      [frame] = training_frames

      assert %{game_state: game_state, controller: controller_state} = frame
      assert %ExPhil.Bridge.GameState{} = game_state
      assert game_state.frame == 0
      assert game_state.stage == 32

      assert %ExPhil.Bridge.ControllerState{} = controller_state
    end

    test "includes player_tag from metadata" do
      # Create a mock parsed replay with player tags
      controller = %Peppi.Controller{
        main_stick_x: 0.5,
        main_stick_y: 0.5,
        c_stick_x: 0.5,
        c_stick_y: 0.5,
        l_trigger: 0.0,
        r_trigger: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_start: false,
        button_d_up: false,
        button_d_down: false,
        button_d_left: false,
        button_d_right: false
      }

      player_frame = %Peppi.PlayerFrame{
        character: 10,
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        facing: 1,
        action: 14,
        action_frame: 0.0,
        invulnerable: false,
        jumps_left: 2,
        on_ground: true,
        shield_strength: 60.0,
        hitstun_frames_left: 0.0,
        speed_air_x_self: 0.0,
        speed_ground_x_self: 0.0,
        speed_y_self: 0.0,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0,
        controller: controller
      }

      game_frame = %Peppi.GameFrame{
        frame_number: 0,
        players: %{1 => player_frame, 2 => player_frame}
      }

      metadata = %Peppi.ReplayMeta{
        path: "test.slp",
        stage: 32,
        duration_frames: 1,
        players: [
          %Peppi.PlayerMeta{port: 1, character: 10, character_name: "Mewtwo", tag: "Plup"},
          %Peppi.PlayerMeta{port: 2, character: 2, character_name: "Fox", tag: "Jmook"}
        ]
      }

      replay = %Peppi.ParsedReplay{
        frames: [game_frame],
        metadata: metadata
      }

      # Test player 1
      training_frames_p1 = Peppi.to_training_frames(replay, player_port: 1)
      [frame_p1] = training_frames_p1
      assert frame_p1[:player_tag] == "Plup"

      # Test player 2
      training_frames_p2 = Peppi.to_training_frames(replay, player_port: 2)
      [frame_p2] = training_frames_p2
      assert frame_p2[:player_tag] == "Jmook"
    end

    test "player_tag is nil when tag not set" do
      controller = %Peppi.Controller{
        main_stick_x: 0.5,
        main_stick_y: 0.5,
        c_stick_x: 0.5,
        c_stick_y: 0.5,
        l_trigger: 0.0,
        r_trigger: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_start: false,
        button_d_up: false,
        button_d_down: false,
        button_d_left: false,
        button_d_right: false
      }

      player_frame = %Peppi.PlayerFrame{
        character: 10,
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        facing: 1,
        action: 14,
        action_frame: 0.0,
        invulnerable: false,
        jumps_left: 2,
        on_ground: true,
        shield_strength: 60.0,
        hitstun_frames_left: 0.0,
        speed_air_x_self: 0.0,
        speed_ground_x_self: 0.0,
        speed_y_self: 0.0,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0,
        controller: controller
      }

      game_frame = %Peppi.GameFrame{
        frame_number: 0,
        players: %{1 => player_frame, 2 => player_frame}
      }

      metadata = %Peppi.ReplayMeta{
        path: "test.slp",
        stage: 32,
        duration_frames: 1,
        players: [
          %Peppi.PlayerMeta{port: 1, character: 10, character_name: "Mewtwo", tag: nil},
          %Peppi.PlayerMeta{port: 2, character: 2, character_name: "Fox", tag: ""}
        ]
      }

      replay = %Peppi.ParsedReplay{frames: [game_frame], metadata: metadata}

      [frame_p1] = Peppi.to_training_frames(replay, player_port: 1)
      assert frame_p1[:player_tag] == nil

      [frame_p2] = Peppi.to_training_frames(replay, player_port: 2)
      # Empty string is treated as nil
      assert frame_p2[:player_tag] == nil
    end
  end
end
