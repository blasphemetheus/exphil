defmodule ExPhil.EmbeddingsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings
  alias ExPhil.Embeddings.Primitives
  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Embeddings.Game, as: GameEmbed
  alias ExPhil.Bridge.{GameState, ControllerState, Nana}
  alias ExPhil.Bridge.Player, as: PlayerState

  # ============================================================================
  # Primitives Tests
  # ============================================================================

  describe "Primitives.one_hot/2" do
    test "creates correct one-hot encoding" do
      result = Primitives.one_hot(3, size: 5)

      assert Nx.shape(result) == {5}
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 1.0, 0.0]
    end

    test "clamps values to valid range" do
      result = Primitives.one_hot(10, size: 5)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]

      result = Primitives.one_hot(-5, size: 5)
      assert Nx.to_flat_list(result) == [1.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "handles zero index" do
      result = Primitives.one_hot(0, size: 5)
      assert Nx.to_flat_list(result) == [1.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "accepts Nx tensor as input" do
      result = Primitives.one_hot(Nx.tensor(2), size: 4)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 1.0, 0.0]
    end
  end

  describe "Primitives.one_hot_with_unknown/2" do
    test "creates one-hot with unknown dimension for valid values" do
      result = Primitives.one_hot_with_unknown(2, size: 4)

      # 4 normal dims + 1 unknown dim = 5 total
      assert Nx.shape(result) == {5}
      assert Nx.to_flat_list(result) == [0.0, 0.0, 1.0, 0.0, 0.0]
    end

    test "maps out-of-range to unknown dimension" do
      result = Primitives.one_hot_with_unknown(10, size: 4)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]

      result = Primitives.one_hot_with_unknown(-1, size: 4)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  describe "Primitives.float_embed/2" do
    test "scales values correctly" do
      result = Primitives.float_embed(100.0, scale: 0.01)
      assert Nx.shape(result) == {1}
      assert_in_delta Nx.to_number(Nx.squeeze(result)), 1.0, 0.001
    end

    test "applies bias before scaling" do
      result = Primitives.float_embed(50.0, scale: 0.1, bias: -50.0)
      assert_in_delta Nx.to_number(Nx.squeeze(result)), 0.0, 0.001
    end

    test "clamps to lower and upper bounds" do
      result = Primitives.float_embed(1000.0, scale: 1.0, upper: 5.0)
      assert Nx.to_number(Nx.squeeze(result)) == 5.0

      result = Primitives.float_embed(-1000.0, scale: 1.0, lower: -5.0)
      assert Nx.to_number(Nx.squeeze(result)) == -5.0
    end
  end

  describe "Primitives.bool_embed/2" do
    test "embeds true as 1.0 by default" do
      result = Primitives.bool_embed(true)
      assert Nx.to_number(Nx.squeeze(result)) == 1.0
    end

    test "embeds false as 0.0 by default" do
      result = Primitives.bool_embed(false)
      assert Nx.to_number(Nx.squeeze(result)) == 0.0
    end

    test "supports custom on/off values" do
      result = Primitives.bool_embed(true, on: 5.0, off: -5.0)
      assert Nx.to_number(Nx.squeeze(result)) == 5.0

      result = Primitives.bool_embed(false, on: 5.0, off: -5.0)
      assert Nx.to_number(Nx.squeeze(result)) == -5.0
    end
  end

  describe "Primitives.facing_embed/1" do
    test "embeds facing right as +1.0" do
      result = Primitives.facing_embed(true)
      assert Nx.to_number(Nx.squeeze(result)) == 1.0
    end

    test "embeds facing left as -1.0" do
      result = Primitives.facing_embed(false)
      assert Nx.to_number(Nx.squeeze(result)) == -1.0
    end
  end

  describe "Primitives.embedding_size/1" do
    test "returns correct sizes for known types" do
      assert Primitives.embedding_size(:action) == 0x18F
      assert Primitives.embedding_size(:character) == 0x21
      assert Primitives.embedding_size(:stage) == 64
      assert Primitives.embedding_size(:jumps_left) == 7
      assert Primitives.embedding_size(:float) == 1
      assert Primitives.embedding_size(:bool) == 1
    end
  end

  describe "specialized embeddings" do
    test "action_embed creates correct size" do
      result = Primitives.action_embed(10)
      assert Nx.shape(result) == {0x18F}
    end

    test "character_embed creates correct size" do
      result = Primitives.character_embed(5)
      assert Nx.shape(result) == {0x21}
    end

    test "stage_embed creates correct size" do
      result = Primitives.stage_embed(2)
      assert Nx.shape(result) == {64}
    end

    test "jumps_left_embed creates correct size" do
      result = Primitives.jumps_left_embed(2)
      assert Nx.shape(result) == {7}
    end

    test "percent_embed scales correctly" do
      result = Primitives.percent_embed(100.0)
      assert_in_delta Nx.to_number(Nx.squeeze(result)), 1.0, 0.001
    end
  end

  # ============================================================================
  # Controller Tests
  # ============================================================================

  describe "ControllerEmbed.embedding_size/1" do
    test "calculates correct size for default config" do
      config = ControllerEmbed.default_config()
      size = ControllerEmbed.embedding_size(config)

      # 8 buttons + 2*(16+1) main + 2*(16+1) c + (4+1) shoulder
      expected = 8 + 2 * 17 + 2 * 17 + 5
      assert size == expected
    end
  end

  describe "ControllerEmbed.continuous_embedding_size/0" do
    test "returns correct size" do
      # 8 buttons + 2 main + 2 c + 1 shoulder = 13
      assert ControllerEmbed.continuous_embedding_size() == 13
    end
  end

  describe "ControllerEmbed.embed_continuous/1" do
    test "embeds nil controller as zeros" do
      result = ControllerEmbed.embed_continuous(nil)

      assert Nx.shape(result) == {13}
      assert Nx.to_flat_list(result) == List.duplicate(0.0, 13)
    end

    test "embeds controller state correctly" do
      controller = %ControllerState{
        button_a: true,
        button_b: false,
        button_x: true,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false,
        main_stick: %{x: 0.5, y: 0.75},
        c_stick: %{x: 0.25, y: 0.5},
        l_shoulder: 0.5
      }

      result = ControllerEmbed.embed_continuous(controller)
      values = Nx.to_flat_list(result)

      # Buttons: A=1, B=0, X=1, rest 0
      assert Enum.at(values, 0) == 1.0
      assert Enum.at(values, 1) == 0.0
      assert Enum.at(values, 2) == 1.0

      # Main stick: centered at 0.5, scaled to [-1, 1]
      # x: (0.5 - 0.5) * 2 = 0.0
      # y: (0.75 - 0.5) * 2 = 0.5
      assert_in_delta Enum.at(values, 8), 0.0, 0.001
      assert_in_delta Enum.at(values, 9), 0.5, 0.001
    end
  end

  describe "ControllerEmbed.discretize_axis/2" do
    test "maps 0.0 to bucket 0" do
      assert ControllerEmbed.discretize_axis(0.0, 16) == 0
    end

    test "maps 1.0 to max bucket" do
      assert ControllerEmbed.discretize_axis(1.0, 16) == 16
    end

    test "maps 0.5 to middle bucket" do
      assert ControllerEmbed.discretize_axis(0.5, 16) == 8
    end

    test "clamps out of range values" do
      assert ControllerEmbed.discretize_axis(-0.5, 16) == 0
      assert ControllerEmbed.discretize_axis(1.5, 16) == 16
    end
  end

  describe "ControllerEmbed.undiscretize_axis/2" do
    test "inverts discretization" do
      bucket = ControllerEmbed.discretize_axis(0.5, 16)
      value = ControllerEmbed.undiscretize_axis(bucket, 16)
      assert value == 0.5
    end

    test "maps bucket 0 to 0.0" do
      assert ControllerEmbed.undiscretize_axis(0, 16) == 0.0
    end

    test "maps max bucket to 1.0" do
      assert ControllerEmbed.undiscretize_axis(16, 16) == 1.0
    end
  end

  describe "ControllerEmbed.component_indices/1" do
    test "returns correct indices for default config" do
      indices = ControllerEmbed.component_indices()

      assert indices.buttons == {0, 8}
      assert indices.main_x == {8, 25}
      assert indices.main_y == {25, 42}
      assert indices.c_x == {42, 59}
      assert indices.c_y == {59, 76}
      assert indices.shoulder == {76, 81}
    end
  end

  # ============================================================================
  # Player Tests
  # ============================================================================

  describe "PlayerEmbed.embedding_size/1" do
    test "calculates base size correctly" do
      # Disable all optional features to test pure base size
      config = %ExPhil.Embeddings.Player{
        with_speeds: false,
        with_nana: false,
        with_frame_info: false,
        with_stock: false,
        with_ledge_distance: false,
        jumps_normalized: false  # Use classic 7-dim jumps
      }
      size = PlayerEmbed.embedding_size(config)

      # 1 (percent) + 1 (facing) + 1 (x) + 1 (y) + 399 (action) + 33 (char) +
      # 1 (invuln) + 7 (jumps) + 1 (shield) + 1 (ground) = 446
      expected =
        1 + 1 + 1 + 1 +
        Primitives.embedding_size(:action) +
        Primitives.embedding_size(:character) +
        1 + Primitives.embedding_size(:jumps_left) + 1 + 1

      assert size == expected
    end

    test "calculates base size with normalized jumps" do
      # Disable optional features except jumps_normalized
      config = %ExPhil.Embeddings.Player{
        with_speeds: false,
        with_nana: false,
        with_frame_info: false,
        with_stock: false,
        with_ledge_distance: false,
        jumps_normalized: true  # 1-dim instead of 7-dim
      }
      size = PlayerEmbed.embedding_size(config)

      # 1 (percent) + 1 (facing) + 1 (x) + 1 (y) + 399 (action) + 33 (char) +
      # 1 (invuln) + 1 (jumps normalized) + 1 (shield) + 1 (ground) = 440
      expected =
        1 + 1 + 1 + 1 +
        Primitives.embedding_size(:action) +
        Primitives.embedding_size(:character) +
        1 + 1 + 1 + 1  # jumps is now 1 dim

      assert size == expected
    end

    test "adds speed dimensions when enabled" do
      base_config = %ExPhil.Embeddings.Player{with_speeds: false, with_nana: false}
      speed_config = %ExPhil.Embeddings.Player{with_speeds: true, with_nana: false}

      base_size = PlayerEmbed.embedding_size(base_config)
      speed_size = PlayerEmbed.embedding_size(speed_config)

      assert speed_size == base_size + 5
    end

    test "adds nana dimensions when enabled (compact mode)" do
      # Compact nana mode is the default - adds 39 dims (preserves IC tech)
      base_config = %ExPhil.Embeddings.Player{with_speeds: false, with_nana: false}
      nana_config = %ExPhil.Embeddings.Player{with_speeds: false, with_nana: true, nana_mode: :compact}

      base_size = PlayerEmbed.embedding_size(base_config)
      nana_size = PlayerEmbed.embedding_size(nana_config)

      # Compact nana adds 39 dims (not full base + 1)
      assert nana_size == base_size + 39
    end

    test "adds nana dimensions when enabled (full mode)" do
      # Full nana mode adds complete player embedding
      base_config = %ExPhil.Embeddings.Player{
        with_speeds: false,
        with_nana: false,
        with_frame_info: false,
        with_stock: false,
        with_ledge_distance: false
      }
      nana_config = %ExPhil.Embeddings.Player{
        with_speeds: false,
        with_nana: true,
        nana_mode: :full,
        with_frame_info: false,
        with_stock: false,
        with_ledge_distance: false
      }

      base_size = PlayerEmbed.embedding_size(base_config)
      nana_size = PlayerEmbed.embedding_size(nana_config)

      # Full nana adds base + 1 (exists flag)
      assert nana_size == 2 * base_size + 1
    end
  end

  describe "PlayerEmbed.embed/2" do
    test "embeds nil player as zeros" do
      config = PlayerEmbed.default_config()
      result = PlayerEmbed.embed(nil, config)

      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}
      # All zeros
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == 0.0))
    end

    test "embeds player state" do
      player = %PlayerState{
        percent: 50.0,
        facing: true,
        x: 0.0,
        y: 0.0,
        action: 14,  # Wait action
        character: 10,  # Mewtwo
        invulnerable: false,
        jumps_left: 2,
        shield_strength: 60.0,
        on_ground: true,
        nana: nil,
        speed_air_x_self: 0.0,
        speed_ground_x_self: 0.0,
        speed_y_self: 0.0,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0
      }

      config = %ExPhil.Embeddings.Player{with_speeds: false, with_nana: false}
      result = PlayerEmbed.embed(player, config)

      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}

      # Check first value is scaled percent (50 * 0.01 = 0.5)
      first_val = result |> Nx.slice([0], [1]) |> Nx.squeeze() |> Nx.to_number()
      assert_in_delta first_val, 0.5, 0.001
    end
  end

  describe "PlayerEmbed.embed_nana/2" do
    test "embeds nil nana with exists=false" do
      config = PlayerEmbed.default_config()
      result = PlayerEmbed.embed_nana(nil, config)

      values = Nx.to_flat_list(result)
      # All zeros including the exists flag
      assert Enum.all?(values, &(&1 == 0.0))
    end

    test "embeds nana with exists=true (compact mode)" do
      nana = %Nana{
        percent: 25.0,
        facing: true,
        x: 10.0,
        y: 5.0,
        action: 14,
        stock: 3
      }

      config = PlayerEmbed.default_config()  # Uses compact mode by default
      result = PlayerEmbed.embed_nana(nana, config)

      values = Nx.to_flat_list(result)
      # In compact mode, first value should be 1.0 (exists)
      assert List.first(values) == 1.0
    end
  end

  # ============================================================================
  # Game Tests
  # ============================================================================

  describe "GameEmbed.embedding_size/1" do
    test "calculates total size correctly" do
      config = GameEmbed.default_config()
      size = GameEmbed.embedding_size(config)

      player_size = PlayerEmbed.embedding_size(config.player)
      stage_size = Primitives.embedding_size(:stage)
      prev_action_size = ControllerEmbed.continuous_embedding_size()
      name_size = config.num_player_names

      # Projectiles (5 slots × 7 dims each by default)
      projectile_size = if config.with_projectiles, do: config.max_projectiles * 7, else: 0

      # Spatial features
      distance_size = if config.with_distance, do: 1, else: 0
      relative_pos_size = if config.with_relative_pos, do: 2, else: 0
      frame_count_size = if config.with_frame_count, do: 1, else: 0

      expected = 2 * player_size + stage_size + prev_action_size + name_size +
                 projectile_size + distance_size + relative_pos_size + frame_count_size

      assert size == expected
    end
  end

  describe "GameEmbed.embed/4" do
    test "embeds complete game state" do
      game_state = make_game_state()
      config = GameEmbed.default_config()

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "handles nil previous action" do
      game_state = make_game_state()
      config = GameEmbed.default_config()

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert is_struct(result, Nx.Tensor)
    end

    test "uses ego-centric ordering" do
      game_state = make_game_state()
      config = GameEmbed.default_config()

      # When we're on port 1, player 1 should be first
      result_p1 = GameEmbed.embed(game_state, nil, 1, config: config)

      # When we're on port 2, player 2 should be first
      result_p2 = GameEmbed.embed(game_state, nil, 2, config: config)

      # The embeddings should be different due to ego-centric ordering
      refute Nx.to_flat_list(result_p1) == Nx.to_flat_list(result_p2)
    end
  end

  describe "GameEmbed.dummy/1" do
    test "creates zero tensor of correct size" do
      config = GameEmbed.default_config()
      result = GameEmbed.dummy(config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == 0.0))
    end
  end

  describe "GameEmbed.dummy_batch/2" do
    test "creates batched zero tensor" do
      config = GameEmbed.default_config()
      batch_size = 32
      result = GameEmbed.dummy_batch(batch_size, config: config)

      assert Nx.shape(result) == {batch_size, GameEmbed.embedding_size(config)}
    end
  end

  # ============================================================================
  # Main API Tests
  # ============================================================================

  describe "Embeddings.config/1" do
    test "creates default config" do
      config = Embeddings.config()

      assert config.with_projectiles == false
      assert config.player.with_speeds == false
      assert config.controller.axis_buckets == 16
    end

    test "accepts custom options" do
      config = Embeddings.config(
        with_speeds: true,
        axis_buckets: 8,
        with_projectiles: true
      )

      assert config.player.with_speeds == true
      assert config.controller.axis_buckets == 8
      assert config.with_projectiles == true
    end
  end

  describe "Embeddings.embed/3" do
    test "embeds game state with default config" do
      game_state = make_game_state()

      result = Embeddings.embed(game_state)

      assert is_struct(result, Nx.Tensor)
      assert Nx.shape(result) == {Embeddings.embedding_size()}
    end

    test "accepts custom port" do
      game_state = make_game_state()

      result = Embeddings.embed(game_state, nil, own_port: 2)

      assert is_struct(result, Nx.Tensor)
    end
  end

  describe "Embeddings.embed_for_character/4" do
    test "adds mewtwo-specific features" do
      game_state = make_game_state()
      base_size = Embeddings.embedding_size()

      result = Embeddings.embed_for_character(game_state, nil, :mewtwo)

      # Mewtwo adds 3 features
      assert Nx.shape(result) == {base_size + 3}
    end

    test "adds gnw-specific features" do
      game_state = make_game_state()
      base_size = Embeddings.embedding_size()

      result = Embeddings.embed_for_character(game_state, nil, :game_and_watch)

      # G&W adds 2 features
      assert Nx.shape(result) == {base_size + 2}
    end

    test "adds link-specific features" do
      game_state = make_game_state()
      base_size = Embeddings.embedding_size()

      result = Embeddings.embed_for_character(game_state, nil, :link)

      # Link adds 2 features
      assert Nx.shape(result) == {base_size + 2}
    end

    test "returns base embedding for ganondorf" do
      game_state = make_game_state()
      base_size = Embeddings.embedding_size()

      result = Embeddings.embed_for_character(game_state, nil, :ganondorf)

      # Ganondorf uses standard embedding
      assert Nx.shape(result) == {base_size}
    end

    test "returns base embedding for unknown character" do
      game_state = make_game_state()
      base_size = Embeddings.embedding_size()

      result = Embeddings.embed_for_character(game_state, nil, :unknown)

      assert Nx.shape(result) == {base_size}
    end
  end

  describe "Embeddings.embed_batch/2" do
    test "embeds multiple states" do
      game_state = make_game_state()
      states = [
        {game_state, nil, 1},
        {game_state, nil, 1},
        {game_state, nil, 2}
      ]

      result = Embeddings.embed_batch(states)

      assert Nx.shape(result) == {3, Embeddings.embedding_size()}
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp make_game_state do
    player = %PlayerState{
      percent: 50.0,
      facing: true,
      x: 0.0,
      y: 0.0,
      action: 14,
      character: 10,
      invulnerable: false,
      jumps_left: 2,
      shield_strength: 60.0,
      on_ground: true,
      nana: nil,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      controller_state: nil,
      stock: 4,
      action_frame: 0,
      hitstun_frames_left: 0
    }

    opponent = %PlayerState{
      percent: 75.0,
      facing: false,
      x: 20.0,
      y: 0.0,
      action: 14,
      character: 2,
      invulnerable: false,
      jumps_left: 2,
      shield_strength: 60.0,
      on_ground: true,
      nana: nil,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      controller_state: nil,
      stock: 4,
      action_frame: 0,
      hitstun_frames_left: 0
    }

    %GameState{
      frame: 1000,
      stage: 2,
      menu_state: 2,  # IN_GAME
      players: %{1 => player, 2 => opponent},
      projectiles: [],
      distance: 20.0
    }
  end
end
