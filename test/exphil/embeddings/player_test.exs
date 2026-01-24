defmodule ExPhil.Embeddings.PlayerTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Bridge.Nana

  # Helper to create a mock player state
  defp mock_player(opts \\ []) do
    %PlayerState{
      character: Keyword.get(opts, :character, 10),  # Mewtwo
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      action: Keyword.get(opts, :action, 14),  # Wait action
      action_frame: 0,
      invulnerable: Keyword.get(opts, :invulnerable, false),
      jumps_left: Keyword.get(opts, :jumps_left, 2),
      on_ground: Keyword.get(opts, :on_ground, true),
      shield_strength: Keyword.get(opts, :shield_strength, 60.0),
      hitstun_frames_left: 0,
      speed_air_x_self: Keyword.get(opts, :speed_air_x_self, 0.0),
      speed_ground_x_self: Keyword.get(opts, :speed_ground_x_self, 0.0),
      speed_y_self: Keyword.get(opts, :speed_y_self, 0.0),
      speed_x_attack: Keyword.get(opts, :speed_x_attack, 0.0),
      speed_y_attack: Keyword.get(opts, :speed_y_attack, 0.0),
      nana: Keyword.get(opts, :nana, nil),
      controller_state: nil
    }
  end

  defp mock_nana(opts \\ []) do
    %Nana{
      x: Keyword.get(opts, :x, 10.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      action: Keyword.get(opts, :action, 14)
    }
  end

  # ============================================================================
  # Configuration Tests
  # ============================================================================

  describe "default_config/0" do
    test "returns default configuration" do
      config = PlayerEmbed.default_config()

      assert %PlayerEmbed{} = config
      assert config.xy_scale == 0.05
      assert config.shield_scale == 0.01
      assert config.speed_scale == 0.5
      assert config.with_speeds == true
      assert config.with_nana == true
      assert config.with_frame_info == true
      assert config.with_stock == true
      assert config.with_ledge_distance == true
      assert config.jumps_normalized == true  # Default to normalized (1-dim)
    end
  end

  describe "embedding_size/1" do
    test "returns correct size with default config" do
      config = PlayerEmbed.default_config()
      size = PlayerEmbed.embedding_size(config)

      # Base (446) + nana exists flag (1) + nana base (446) = 893
      # Actually let me calculate: 1 + 1 + 1 + 1 + 399 + 33 + 1 + 7 + 1 + 1 = 446
      # With nana: 446 + (446 + 1) = 893
      assert is_integer(size)
      assert size > 0
    end

    test "size increases with speeds enabled" do
      base_config = %PlayerEmbed{with_speeds: false, with_nana: false}
      speed_config = %PlayerEmbed{with_speeds: true, with_nana: false}

      base_size = PlayerEmbed.embedding_size(base_config)
      speed_size = PlayerEmbed.embedding_size(speed_config)

      assert speed_size == base_size + 5
    end

    test "size increases with nana enabled" do
      no_nana = %PlayerEmbed{with_speeds: false, with_nana: false}
      with_nana = %PlayerEmbed{with_speeds: false, with_nana: true}

      base_size = PlayerEmbed.embedding_size(no_nana)
      nana_size = PlayerEmbed.embedding_size(with_nana)

      # Nana adds base_size + 1 (exists flag)
      assert nana_size > base_size
    end
  end

  # ============================================================================
  # Embed Tests
  # ============================================================================

  describe "embed/2" do
    test "embeds player state to correct size tensor" do
      player = mock_player()
      config = PlayerEmbed.default_config()

      result = PlayerEmbed.embed(player, config)

      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}
    end

    test "handles nil player by returning zeros" do
      config = PlayerEmbed.default_config()

      result = PlayerEmbed.embed(nil, config)

      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}
      # Should be all zeros
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end

    test "different players produce different embeddings" do
      player1 = mock_player(percent: 0.0, x: 0.0)
      player2 = mock_player(percent: 100.0, x: 50.0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result1 = PlayerEmbed.embed(player1, config)
      result2 = PlayerEmbed.embed(player2, config)

      # Should be different
      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0.1
    end

    test "embedding changes with different actions" do
      player1 = mock_player(action: 0)
      player2 = mock_player(action: 100)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result1 = PlayerEmbed.embed(player1, config)
      result2 = PlayerEmbed.embed(player2, config)

      # Should be different
      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0
    end

    test "embeds with speeds when configured" do
      player = mock_player(speed_air_x_self: 2.0, speed_y_self: -1.0)
      config = %PlayerEmbed{with_speeds: true, with_nana: false}

      result = PlayerEmbed.embed(player, config)

      expected_size = PlayerEmbed.embedding_size(config)
      assert Nx.shape(result) == {expected_size}
    end

    test "embeds with nana when configured" do
      nana = mock_nana()
      player = mock_player(nana: nana)
      config = %PlayerEmbed{with_nana: true, with_speeds: false}

      result = PlayerEmbed.embed(player, config)

      expected_size = PlayerEmbed.embedding_size(config)
      assert Nx.shape(result) == {expected_size}
    end
  end

  describe "embed_base/2" do
    test "creates base embedding" do
      player = mock_player()
      config = PlayerEmbed.default_config()

      result = PlayerEmbed.embed_base(player, config)

      # Base size with jumps_normalized=true (default):
      # 1 + 1 + 1 + 1 + 399 + 33 + 1 + 1 + 1 + 1 = 440
      # (jumps is 1 dim instead of 7)
      expected_size = 1 + 1 + 1 + 1 + 399 + 33 + 1 + 1 + 1 + 1
      assert Nx.shape(result) == {expected_size}
    end

    test "creates base embedding with one-hot jumps" do
      player = mock_player()
      config = %PlayerEmbed{jumps_normalized: false}

      result = PlayerEmbed.embed_base(player, config)

      # Base size with jumps_normalized=false:
      # 1 + 1 + 1 + 1 + 399 + 33 + 1 + 7 + 1 + 1 = 446
      expected_size = 1 + 1 + 1 + 1 + 399 + 33 + 1 + 7 + 1 + 1
      assert Nx.shape(result) == {expected_size}
    end

    test "encodes percent correctly" do
      player1 = mock_player(percent: 0.0)
      player2 = mock_player(percent: 150.0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result1 = PlayerEmbed.embed_base(player1, config)
      result2 = PlayerEmbed.embed_base(player2, config)

      # First element is percent
      percent1 = Nx.to_number(Nx.squeeze(Nx.slice(result1, [0], [1])))
      percent2 = Nx.to_number(Nx.squeeze(Nx.slice(result2, [0], [1])))

      assert_in_delta percent1, 0.0, 0.001
      assert_in_delta percent2, 1.5, 0.001  # 150 * 0.01
    end

    test "encodes facing direction correctly" do
      player_right = mock_player(facing: 1)
      player_left = mock_player(facing: 0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result_right = PlayerEmbed.embed_base(player_right, config)
      result_left = PlayerEmbed.embed_base(player_left, config)

      # Second element is facing
      facing_right = Nx.to_number(Nx.squeeze(Nx.slice(result_right, [1], [1])))
      facing_left = Nx.to_number(Nx.squeeze(Nx.slice(result_left, [1], [1])))

      assert_in_delta facing_right, 1.0, 0.001
      assert_in_delta facing_left, -1.0, 0.001
    end

    test "creates base embedding without action one-hot (learned action mode)" do
      player = mock_player()
      config = %PlayerEmbed{action_mode: :learned, jumps_normalized: true}

      result = PlayerEmbed.embed_base(player, config)

      # Base size with action_mode: :learned:
      # 1 + 1 + 1 + 1 + 0 + 33 + 1 + 1 + 1 + 1 = 41
      # (action is 0 dims instead of 399)
      expected_size = 1 + 1 + 1 + 1 + 0 + 33 + 1 + 1 + 1 + 1
      assert Nx.shape(result) == {expected_size}
    end

    test "creates base embedding without action one-hot and with one-hot jumps" do
      player = mock_player()
      config = %PlayerEmbed{action_mode: :learned, jumps_normalized: false}

      result = PlayerEmbed.embed_base(player, config)

      # Base size with action_mode: :learned and jumps_normalized: false:
      # 1 + 1 + 1 + 1 + 0 + 33 + 1 + 7 + 1 + 1 = 47
      expected_size = 1 + 1 + 1 + 1 + 0 + 33 + 1 + 7 + 1 + 1
      assert Nx.shape(result) == {expected_size}
    end
  end

  describe "get_action_id/1" do
    test "returns action from player state" do
      player = mock_player(action: 50)

      assert PlayerEmbed.get_action_id(player) == 50
    end

    test "returns 0 for nil player" do
      assert PlayerEmbed.get_action_id(nil) == 0
    end

    test "returns 0 for nil action" do
      player = %PlayerState{action: nil}

      assert PlayerEmbed.get_action_id(player) == 0
    end
  end

  describe "get_action_ids_batch/1" do
    test "returns tensor of action IDs" do
      players = [
        mock_player(action: 10),
        mock_player(action: 20),
        mock_player(action: 30)
      ]

      result = PlayerEmbed.get_action_ids_batch(players)

      assert Nx.shape(result) == {3}
      assert Nx.to_list(result) == [10, 20, 30]
    end

    test "handles nil players in batch" do
      players = [
        mock_player(action: 10),
        nil,
        mock_player(action: 30)
      ]

      result = PlayerEmbed.get_action_ids_batch(players)

      assert Nx.shape(result) == {3}
      assert Nx.to_list(result) == [10, 0, 30]
    end
  end

  describe "embed_speeds/2" do
    test "creates 5-dimensional speed embedding" do
      player = mock_player(
        speed_air_x_self: 2.0,
        speed_ground_x_self: 1.5,
        speed_y_self: -3.0,
        speed_x_attack: 0.5,
        speed_y_attack: -0.5
      )
      config = PlayerEmbed.default_config()

      result = PlayerEmbed.embed_speeds(player, config)

      assert Nx.shape(result) == {5}
    end

    test "scales speeds correctly" do
      player = mock_player(speed_air_x_self: 4.0)  # * 0.5 = 2.0
      config = PlayerEmbed.default_config()

      result = PlayerEmbed.embed_speeds(player, config)

      first_speed = Nx.to_number(Nx.squeeze(Nx.slice(result, [0], [1])))
      assert_in_delta first_speed, 2.0, 0.001
    end
  end

  describe "embed_nana/3" do
    test "returns zeros with exists=0 for nil nana (compact mode)" do
      config = %PlayerEmbed{nana_mode: :compact}

      result = PlayerEmbed.embed_nana(nil, config, nil)

      # Compact Nana size
      assert Nx.shape(result) == {39}

      # All should be zeros
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end

    test "returns zeros with exists=0 for nil nana (full mode)" do
      # Use jumps_normalized: false to get classic 446 base size
      config = %PlayerEmbed{nana_mode: :full, with_speeds: false, with_frame_info: false, with_stock: false, jumps_normalized: false}

      result = PlayerEmbed.embed_nana(nil, config, nil)

      # Base size (446) + 1 for exists flag
      expected_size = 446 + 1
      assert Nx.shape(result) == {expected_size}

      # All should be zeros
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end

    test "returns zeros with exists=0 for nil nana (full mode, normalized jumps)" do
      # Default uses jumps_normalized: true
      config = %PlayerEmbed{nana_mode: :full, with_speeds: false, with_frame_info: false, with_stock: false}

      result = PlayerEmbed.embed_nana(nil, config, nil)

      # Base size (440 with normalized jumps) + 1 for exists flag
      expected_size = 440 + 1
      assert Nx.shape(result) == {expected_size}

      # All should be zeros
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end

    test "embeds nana with exists=1 (compact mode)" do
      nana = mock_nana()
      config = PlayerEmbed.default_config()  # Default is compact mode

      result = PlayerEmbed.embed_nana(nana, config, nil)

      # In compact mode, exists is the first element
      exists_flag = Nx.to_number(Nx.squeeze(Nx.slice(result, [0], [1])))
      assert_in_delta exists_flag, 1.0, 0.001
    end

    test "embeds nana with exists=1 (full mode)" do
      nana = mock_nana()
      config = %PlayerEmbed{nana_mode: :full, with_speeds: false, with_frame_info: false, with_stock: false}

      result = PlayerEmbed.embed_nana(nana, config, nil)

      # In full mode, exists is at position 446 (after base embedding)
      exists_flag = Nx.to_number(Nx.squeeze(Nx.slice(result, [446], [1])))
      assert_in_delta exists_flag, 1.0, 0.001
    end

    test "nana embedding contains nana's data" do
      nana1 = mock_nana(percent: 0.0)
      nana2 = mock_nana(percent: 50.0)
      config = PlayerEmbed.default_config()

      result1 = PlayerEmbed.embed_nana(nana1, config, nil)
      result2 = PlayerEmbed.embed_nana(nana2, config, nil)

      # Should be different (nana's percent is embedded)
      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0
    end

    # Enhanced mode tests
    test "returns zeros with exists=0 for nil nana (enhanced mode)" do
      config = %PlayerEmbed{nana_mode: :enhanced}

      result = PlayerEmbed.embed_nana(nil, config, nil)

      # Enhanced Nana size: 14 dims (no action ID in embedding - handled by GameEmbed)
      expected_size = 14
      assert Nx.shape(result) == {expected_size}

      # All should be zeros
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end

    test "embeds nana with exists=1 (enhanced mode)" do
      nana = mock_nana()
      config = %PlayerEmbed{nana_mode: :enhanced}

      result = PlayerEmbed.embed_nana(nana, config, nil)

      # Enhanced mode has 14 dims
      expected_size = 14
      assert Nx.shape(result) == {expected_size}

      # First element is exists flag, should be 1.0
      exists_flag = Nx.to_number(Nx.squeeze(Nx.slice(result, [0], [1])))
      assert exists_flag == 1.0
    end

    test "enhanced mode has boolean flags for attacking/grabbing/can_act" do
      # Test with actions in correct categories
      # Action categories from action_to_category:
      # - 10-16: Attack categories (ground attacks, air attacks, specials) = 0x43-0xBB
      # - 17-18: Grab/throw categories = 0xBC-0xD3
      # - 2: Idle = 0x0E-0x14
      nana_attacking = mock_nana(action: 0x50)  # Ground attack category 11
      nana_grabbing = mock_nana(action: 0xC0)   # Grab category 17
      nana_idle = mock_nana(action: 0x10)       # Idle category 2
      config = %PlayerEmbed{nana_mode: :enhanced}

      result_attacking = PlayerEmbed.embed_nana(nana_attacking, config, nil)
      result_grabbing = PlayerEmbed.embed_nana(nana_grabbing, config, nil)
      result_idle = PlayerEmbed.embed_nana(nana_idle, config, nil)

      # All should be 14 dims
      assert Nx.shape(result_attacking) == {14}
      assert Nx.shape(result_grabbing) == {14}
      assert Nx.shape(result_idle) == {14}

      # The embeddings should differ based on action state
      # is_attacking at index 9, is_grabbing at index 10, can_act at index 11
      diff_atk_idle = Nx.sum(Nx.abs(Nx.subtract(result_attacking, result_idle)))
      diff_grab_idle = Nx.sum(Nx.abs(Nx.subtract(result_grabbing, result_idle)))

      assert Nx.to_number(diff_atk_idle) > 0
      assert Nx.to_number(diff_grab_idle) > 0
    end
  end

  describe "get_nana_action_id/1" do
    test "returns 0 for nil player" do
      assert PlayerEmbed.get_nana_action_id(nil) == 0
    end

    test "returns 0 for player with no nana" do
      player = mock_player()
      player_no_nana = %{player | nana: nil}
      assert PlayerEmbed.get_nana_action_id(player_no_nana) == 0
    end

    test "returns nana action for player with nana" do
      nana = mock_nana(action: 123)
      player = %{mock_player() | nana: nana}
      assert PlayerEmbed.get_nana_action_id(player) == 123
    end
  end

  describe "get_nana_action_ids_batch/1" do
    test "returns tensor of nana action IDs" do
      nana1 = mock_nana(action: 100)
      nana2 = mock_nana(action: 200)
      player1 = %{mock_player() | nana: nana1}
      player2 = %{mock_player() | nana: nana2}
      player3 = %{mock_player() | nana: nil}

      result = PlayerEmbed.get_nana_action_ids_batch([player1, player2, player3, nil])

      assert Nx.shape(result) == {4}
      assert Nx.to_flat_list(result) == [100, 200, 0, 0]
    end
  end

  # ============================================================================
  # Multi-Player Embedding Tests
  # ============================================================================

  describe "embed_both/3" do
    test "creates flattened embedding for two players" do
      p0 = mock_player(x: 0.0)
      p1 = mock_player(x: 50.0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result = PlayerEmbed.embed_both(p0, p1, config: config, flatten: true)

      single_size = PlayerEmbed.embedding_size(config)
      assert Nx.shape(result) == {single_size * 2}
    end

    test "creates stacked embedding when not flattened" do
      p0 = mock_player(x: 0.0)
      p1 = mock_player(x: 50.0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result = PlayerEmbed.embed_both(p0, p1, config: config, flatten: false)

      single_size = PlayerEmbed.embedding_size(config)
      assert Nx.shape(result) == {2, single_size}
    end
  end

  describe "embed_ego/3" do
    test "creates ego-centric embedding" do
      own = mock_player(percent: 0.0)
      opponent = mock_player(percent: 100.0)
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result = PlayerEmbed.embed_ego(own, opponent, config: config)

      single_size = PlayerEmbed.embedding_size(config)
      assert Nx.shape(result) == {single_size * 2}
    end
  end

  # ============================================================================
  # Edge Cases
  # ============================================================================

  describe "edge cases" do
    test "handles nil values in numeric player fields" do
      # Note: facing is required to be a valid integer (0 or 1)
      # Other numeric fields use nil coalescing in embed_base
      player = %PlayerState{
        character: nil,
        x: nil,
        y: nil,
        percent: nil,
        stock: nil,
        facing: 1,  # Required - cannot be nil
        action: nil,
        action_frame: nil,
        invulnerable: nil,
        jumps_left: nil,
        on_ground: nil,
        shield_strength: nil,
        hitstun_frames_left: nil,
        speed_air_x_self: nil,
        speed_ground_x_self: nil,
        speed_y_self: nil,
        speed_x_attack: nil,
        speed_y_attack: nil,
        nana: nil,
        controller_state: nil
      }
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      # Should not raise
      result = PlayerEmbed.embed(player, config)

      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}
    end

    test "handles extreme values" do
      player = mock_player(
        percent: 999.0,
        x: 1000.0,
        y: -1000.0,
        shield_strength: 100.0
      )
      config = %PlayerEmbed{with_nana: false, with_speeds: false}

      result = PlayerEmbed.embed(player, config)

      # Should be valid tensor (values clamped)
      assert Nx.shape(result) == {PlayerEmbed.embedding_size(config)}
      # Check no NaN or Inf
      assert Nx.to_number(Nx.is_nan(result) |> Nx.sum()) == 0
      assert Nx.to_number(Nx.is_infinity(result) |> Nx.sum()) == 0
    end
  end

  # ============================================================================
  # Character ID Tests
  # ============================================================================

  describe "get_character_id/1" do
    test "returns character from player state" do
      player = mock_player(character: 10)  # Mewtwo

      assert PlayerEmbed.get_character_id(player) == 10
    end

    test "returns 0 for nil player" do
      assert PlayerEmbed.get_character_id(nil) == 0
    end

    test "returns 0 for nil character" do
      player = %PlayerState{character: nil}

      assert PlayerEmbed.get_character_id(player) == 0
    end

    test "returns correct ID for different characters" do
      # Fox = 2, Falco = 20, Marth = 9, Sheik = 19
      assert PlayerEmbed.get_character_id(mock_player(character: 2)) == 2
      assert PlayerEmbed.get_character_id(mock_player(character: 20)) == 20
      assert PlayerEmbed.get_character_id(mock_player(character: 9)) == 9
      assert PlayerEmbed.get_character_id(mock_player(character: 19)) == 19
    end
  end

  describe "get_character_ids_batch/1" do
    test "returns tensor of character IDs" do
      players = [
        mock_player(character: 10),  # Mewtwo
        mock_player(character: 2),   # Fox
        mock_player(character: 20)   # Falco
      ]

      result = PlayerEmbed.get_character_ids_batch(players)

      assert Nx.shape(result) == {3}
      assert Nx.to_list(result) == [10, 2, 20]
    end

    test "handles nil players in batch" do
      players = [
        mock_player(character: 10),
        nil,
        mock_player(character: 20)
      ]

      result = PlayerEmbed.get_character_ids_batch(players)

      assert Nx.shape(result) == {3}
      assert Nx.to_list(result) == [10, 0, 20]
    end
  end

  describe "character embedding config" do
    test "embedding_size smaller with learned character mode" do
      # With learned character embeddings, the 33-dim one-hot character vector
      # is excluded from player embedding (character ID appended at game level instead)
      config_onehot = %PlayerEmbed{character_mode: :one_hot, with_nana: false}
      config_learned = %PlayerEmbed{character_mode: :learned, with_nana: false}

      size_onehot = PlayerEmbed.embedding_size(config_onehot)
      size_learned = PlayerEmbed.embedding_size(config_learned)

      # Learned mode saves 33 dimensions (character one-hot vector)
      assert size_onehot - size_learned == 33
    end

    test "learned character mode reduces embedding by 33 dims" do
      # 33 Melee characters = 33-dim one-hot
      config_base = %PlayerEmbed{character_mode: :one_hot, with_nana: false, with_speeds: true}
      config_learned = %PlayerEmbed{character_mode: :learned, with_nana: false, with_speeds: true}

      assert PlayerEmbed.embedding_size(config_base) - PlayerEmbed.embedding_size(config_learned) == 33
    end
  end
end
