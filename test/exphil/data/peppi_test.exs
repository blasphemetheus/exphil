defmodule ExPhil.Data.PeppiTest do
  use ExUnit.Case, async: true
  @moduletag :data

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
        frames: [game_frame, %{game_frame | frame_number: 1}],
        metadata: metadata
      }

      training_frames = Peppi.to_training_frames(replay)

      # Two source frames -> ONE training frame: the causal pairing gives
      # frame 0 the input issued from it (frame 1's controller) and drops
      # the last frame, which has no successor (INVARIANTS.md item 1).
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
        frames: [game_frame, %{game_frame | frame_number: 1}],
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

    # V3 preflight, 2026-09-17: cartridge tags are stored FULL-WIDTH
    # ("ＦＯＸ") while registries are built from ASCII filename tags
    # ("[FOX]"). Without normalization at the source every tagged subject
    # trained as name_id 0 (0/61 tagged files resolved in the 256-file
    # subset) and the held-out gate scored anonymous == registry exactly.
    test "player_tag is normalized from full-width cartridge tags" do
      controller = %Peppi.Controller{
        main_stick_x: 0.5, main_stick_y: 0.5, c_stick_x: 0.5, c_stick_y: 0.5,
        l_trigger: 0.0, r_trigger: 0.0, button_a: false, button_b: false,
        button_x: false, button_y: false, button_z: false, button_l: false,
        button_r: false, button_start: false, button_d_up: false,
        button_d_down: false, button_d_left: false, button_d_right: false
      }

      player_frame = %Peppi.PlayerFrame{
        character: 2, x: 0.0, y: 0.0, percent: 0.0, stock: 4, facing: 1,
        action: 14, action_frame: 0.0, invulnerable: false, jumps_left: 2,
        on_ground: true, shield_strength: 60.0, hitstun_frames_left: 0.0,
        speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0,
        speed_x_attack: 0.0, speed_y_attack: 0.0, controller: controller
      }

      game_frame = %Peppi.GameFrame{frame_number: 0, players: %{1 => player_frame, 2 => player_frame}}

      metadata = %Peppi.ReplayMeta{
        path: "19_19_52 Falco + [FOX] Fox (YS).slp",
        stage: 8,
        duration_frames: 1,
        players: [
          %Peppi.PlayerMeta{port: 1, character: 20, character_name: "Falco", tag: ""},
          %Peppi.PlayerMeta{port: 2, character: 2, character_name: "Fox", tag: "ＦＯＸ"}
        ]
      }

      replay = %Peppi.ParsedReplay{frames: [game_frame, %{game_frame | frame_number: 1}], metadata: metadata}
      [frame] = Peppi.to_training_frames(replay, player_port: 2)

      assert frame[:player_tag] == "FOX"
      assert frame[:player_tag] == ExPhil.Data.FilenameTags.subject_tag(metadata.path, "Fox")

      registry = ExPhil.Training.PlayerRegistry.from_tags(["FOX"], first_id: 1)
      dataset = ExPhil.Training.Data.from_frames([frame], player_registry: registry)
      assert hd(dataset.frames)[:name_id] == ExPhil.Training.PlayerRegistry.get_id(registry, "FOX")
      assert hd(dataset.frames)[:name_id] > 0
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

      replay = %Peppi.ParsedReplay{frames: [game_frame, %{game_frame | frame_number: 1}], metadata: metadata}

      [frame_p1] = Peppi.to_training_frames(replay, player_port: 1)
      assert frame_p1[:player_tag] == nil

      [frame_p2] = Peppi.to_training_frames(replay, player_port: 2)
      # Empty string is treated as nil
      assert frame_p2[:player_tag] == nil
    end
  end
end

defmodule ExPhil.Data.PeppiIdentityMetadataTest do
  use ExUnit.Case, async: true
  @moduletag :nif
  alias ExPhil.Data.Peppi

  # 2026-09-17: identity evidence beyond names (STYLE_IDENTITY.md,
  # YETI_SCENE_PRIORS.md). Pins that the NIF surfaces costume / player
  # type / cpu level / team per player and startAt / random seed per game.
  test "metadata carries costume, player type and session context" do
    {:ok, meta} = Peppi.metadata("test/fixtures/replays/fox_multishine.slp")

    for p <- meta.players do
      assert is_integer(p.costume) and p.costume >= 0
      assert p.player_type in ["human", "cpu", "demo"]
      assert is_nil(p.cpu_level) or (is_integer(p.cpu_level) and p.cpu_level in 1..9)
      assert is_nil(p.team) or p.team in 0..2
      if p.player_type == "cpu", do: assert(is_integer(p.cpu_level))
    end

    assert is_integer(meta.random_seed)
    assert is_nil(meta.started_at) or String.match?(meta.started_at, ~r/^\d{4}-\d{2}-\d{2}T/)
  end
end

defmodule ExPhil.Data.PeppiNonFiniteTest do
  use ExUnit.Case, async: true
  @moduletag :nif
  alias ExPhil.Data.Peppi

  # erickfm master-master-0a30db85… carries 78 NaN/Inf floats. Erlang cannot
  # represent them, enif_make_double refuses, and rustler raised a bare
  # `badarg` that took the whole fingerprint pass down (2026-09-17). The NIF
  # now sanitizes to 0.0 and reports the count.
  test "non-finite floats are sanitized and counted instead of raising badarg" do
    {:ok, replay} = Peppi.parse("test/fixtures/replays/corrupt_nonfinite_floats.slp", player_port: 1)
    assert replay.metadata.nonfinite_values == 78
    assert length(replay.frames) > 10_000

    {:ok, clean} = Peppi.parse("test/fixtures/replays/fox_multishine.slp", player_port: 1)
    assert clean.metadata.nonfinite_values == 0
  end
end
