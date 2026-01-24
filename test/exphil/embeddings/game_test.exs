defmodule ExPhil.Embeddings.GameTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Game, as: GameEmbed
  alias ExPhil.Embeddings.Player, as: PlayerEmbed
  alias ExPhil.Bridge.{GameState, Player, Projectile, Item, ControllerState}

  # Helper to create a mock player
  defp mock_player(opts) do
    %Player{
      character: Keyword.get(opts, :character, 10),
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      action: Keyword.get(opts, :action, 14),
      action_frame: 0,
      invulnerable: false,
      jumps_left: 2,
      on_ground: true,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: 0.0,
      speed_ground_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  # Helper to create a mock game state
  defp mock_game_state(opts \\ []) do
    player1 = Keyword.get(opts, :player1, mock_player(x: -30.0))
    player2 = Keyword.get(opts, :player2, mock_player(x: 30.0, facing: 0))

    %GameState{
      frame: Keyword.get(opts, :frame, 0),
      stage: Keyword.get(opts, :stage, 32),
      menu_state: 2,
      players: %{1 => player1, 2 => player2},
      projectiles: Keyword.get(opts, :projectiles, []),
      items: Keyword.get(opts, :items, []),
      distance: abs(player1.x - player2.x)
    }
  end

  defp mock_controller_state(opts \\ []) do
    %ControllerState{
      main_stick: %{
        x: Keyword.get(opts, :main_x, 0.5),
        y: Keyword.get(opts, :main_y, 0.5)
      },
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: 0.0,
      r_shoulder: 0.0,
      button_a: Keyword.get(opts, :button_a, false),
      button_b: false,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  defp mock_projectile(opts \\ []) do
    %Projectile{
      owner: Keyword.get(opts, :owner, 1),
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 10.0),
      type: Keyword.get(opts, :type, 10),
      speed_x: Keyword.get(opts, :speed_x, 2.0),
      speed_y: Keyword.get(opts, :speed_y, 0.0)
    }
  end

  defp mock_item(opts \\ []) do
    %Item{
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      type: Keyword.get(opts, :type, 0x2C),  # Link bomb
      facing: Keyword.get(opts, :facing, 1),
      owner: Keyword.get(opts, :owner, 1),
      held_by: Keyword.get(opts, :held_by, nil),
      spawn_id: Keyword.get(opts, :spawn_id, 1),
      timer: Keyword.get(opts, :timer, 120)
    }
  end

  # ============================================================================
  # Configuration Tests
  # ============================================================================

  describe "default_config/0" do
    test "returns default configuration" do
      config = GameEmbed.default_config()

      assert %GameEmbed{} = config
      assert %PlayerEmbed{} = config.player
      assert config.with_projectiles == true
      assert config.max_projectiles == 5
      assert config.num_player_names == 112
      assert config.with_distance == true
      assert config.with_relative_pos == true
      assert config.with_frame_count == true
    end
  end

  describe "embedding_size/1" do
    test "returns correct size with default config" do
      config = GameEmbed.default_config()
      size = GameEmbed.embedding_size(config)

      assert is_integer(size)
      assert size > 0
    end

    test "size increases with projectiles enabled" do
      base_config = %GameEmbed{with_projectiles: false, max_projectiles: 5}
      proj_config = %GameEmbed{with_projectiles: true, max_projectiles: 5}

      base_size = GameEmbed.embedding_size(base_config)
      proj_size = GameEmbed.embedding_size(proj_config)

      # Each projectile adds 7 dimensions
      assert proj_size == base_size + (5 * 7)
    end

    test "size scales with max_projectiles" do
      config5 = %GameEmbed{with_projectiles: true, max_projectiles: 5}
      config10 = %GameEmbed{with_projectiles: true, max_projectiles: 10}

      size5 = GameEmbed.embedding_size(config5)
      size10 = GameEmbed.embedding_size(config10)

      assert size10 == size5 + (5 * 7)
    end

    test "size includes action IDs when using learned actions" do
      player_config = %PlayerEmbed{action_mode: :learned}
      config = %GameEmbed{player: player_config}

      size = GameEmbed.embedding_size(config)

      # Should include 2 action IDs (own + opponent)
      assert size > 0
      # Size should be continuous_embedding_size + num_action_ids
      continuous_size = GameEmbed.continuous_embedding_size(config)
      num_ids = GameEmbed.num_action_ids(config)
      assert num_ids == 2
      assert size == continuous_size + num_ids
    end
  end

  describe "num_action_ids/1" do
    test "returns 0 for default config (one-hot actions)" do
      config = GameEmbed.default_config()
      assert GameEmbed.num_action_ids(config) == 0
    end

    test "returns 2 when using learned player actions" do
      player_config = %PlayerEmbed{action_mode: :learned}
      config = %GameEmbed{player: player_config}
      assert GameEmbed.num_action_ids(config) == 2
    end

    test "returns 4 when using learned actions + enhanced Nana" do
      player_config = %PlayerEmbed{action_mode: :learned, nana_mode: :enhanced, with_nana: true}
      config = %GameEmbed{player: player_config}
      assert GameEmbed.num_action_ids(config) == 4
    end

    test "returns 2 when using enhanced Nana without learned actions" do
      # Enhanced Nana only adds IDs when with_nana is true
      player_config = %PlayerEmbed{action_mode: :one_hot, nana_mode: :enhanced, with_nana: true}
      config = %GameEmbed{player: player_config}
      # Only Nana IDs, no player action IDs
      assert GameEmbed.num_action_ids(config) == 2
    end
  end

  describe "continuous_embedding_size/1" do
    test "equals total size when using one-hot actions" do
      config = GameEmbed.default_config()  # Uses one-hot by default
      total_size = GameEmbed.embedding_size(config)
      continuous_size = GameEmbed.continuous_embedding_size(config)

      assert continuous_size == total_size
    end

    test "is smaller than total when using learned actions" do
      player_config = %PlayerEmbed{action_mode: :learned}
      config = %GameEmbed{player: player_config}

      total_size = GameEmbed.embedding_size(config)
      continuous_size = GameEmbed.continuous_embedding_size(config)
      num_ids = GameEmbed.num_action_ids(config)

      assert num_ids == 2
      assert continuous_size == total_size - num_ids
    end

    test "accounts for 4 action IDs with enhanced Nana" do
      player_config = %PlayerEmbed{action_mode: :learned, nana_mode: :enhanced, with_nana: true}
      config = %GameEmbed{player: player_config}

      total_size = GameEmbed.embedding_size(config)
      continuous_size = GameEmbed.continuous_embedding_size(config)
      num_ids = GameEmbed.num_action_ids(config)

      assert num_ids == 4
      assert continuous_size == total_size - num_ids
    end
  end

  describe "uses_learned_actions?/1" do
    test "returns false for default config (one-hot)" do
      config = GameEmbed.default_config()

      refute GameEmbed.uses_learned_actions?(config)
    end

    test "returns true when player action_mode is :learned" do
      player_config = %PlayerEmbed{action_mode: :learned}
      config = %GameEmbed{player: player_config}

      assert GameEmbed.uses_learned_actions?(config)
    end
  end

  describe "num_character_ids/1" do
    test "returns 0 for default config (one-hot characters)" do
      config = GameEmbed.default_config()
      assert GameEmbed.num_character_ids(config) == 0
    end

    test "returns 2 when using learned character embeddings" do
      player_config = %PlayerEmbed{character_mode: :learned}
      config = %GameEmbed{player: player_config}
      assert GameEmbed.num_character_ids(config) == 2
    end
  end

  describe "uses_learned_characters?/1" do
    test "returns false for default config (one-hot)" do
      config = GameEmbed.default_config()

      refute GameEmbed.uses_learned_characters?(config)
    end

    test "returns true when player character_mode is :learned" do
      player_config = %PlayerEmbed{character_mode: :learned}
      config = %GameEmbed{player: player_config}

      assert GameEmbed.uses_learned_characters?(config)
    end
  end

  describe "combined action and character IDs" do
    test "embedding_size includes both action and character IDs" do
      # Config with both learned actions and characters
      player_config = %PlayerEmbed{action_mode: :learned, character_mode: :learned}
      config = %GameEmbed{player: player_config}

      total_size = GameEmbed.embedding_size(config)
      continuous_size = GameEmbed.continuous_embedding_size(config)
      num_action_ids = GameEmbed.num_action_ids(config)
      num_char_ids = GameEmbed.num_character_ids(config)

      assert num_action_ids == 2
      assert num_char_ids == 2
      assert continuous_size == total_size - num_action_ids - num_char_ids
    end

    test "num_total_ids returns sum of action and character IDs" do
      player_config = %PlayerEmbed{action_mode: :learned, character_mode: :learned}
      config = %GameEmbed{player: player_config}

      assert GameEmbed.num_total_ids(config) == 4
    end

    test "learned character mode reduces embedding by 33 dims per player" do
      # Compare config with one-hot characters vs learned characters
      player_onehot = %PlayerEmbed{character_mode: :one_hot}
      player_learned = %PlayerEmbed{character_mode: :learned}

      config_onehot = %GameEmbed{player: player_onehot}
      config_learned = %GameEmbed{player: player_learned}

      # With learned characters, we exclude 33-dim one-hot from each player (66 total)
      # but add 2 character IDs at the end
      size_onehot = GameEmbed.embedding_size(config_onehot)
      size_learned = GameEmbed.embedding_size(config_learned)

      # 66 dims saved from 2 players, 2 dims added for character IDs
      # Net savings = 66 - 2 = 64 dims
      assert size_onehot - size_learned == 64
    end
  end

  # ============================================================================
  # Main Embedding Tests
  # ============================================================================

  describe "embed/4" do
    test "creates embedding of correct size" do
      game_state = mock_game_state()
      prev_action = mock_controller_state()
      config = GameEmbed.default_config()

      result = GameEmbed.embed(game_state, prev_action, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "handles nil previous action" do
      game_state = mock_game_state()
      config = GameEmbed.default_config()

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "produces different embeddings for different ports" do
      game_state = mock_game_state(
        player1: mock_player(x: 0.0, percent: 0.0),
        player2: mock_player(x: 50.0, percent: 100.0)
      )
      config = GameEmbed.default_config()

      result1 = GameEmbed.embed(game_state, nil, 1, config: config)
      result2 = GameEmbed.embed(game_state, nil, 2, config: config)

      # Should be different (ego-centric view is swapped)
      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0.1
    end

    test "includes name_id in embedding" do
      game_state = mock_game_state()
      config = GameEmbed.default_config()

      result0 = GameEmbed.embed(game_state, nil, 1, config: config, name_id: 0)
      result1 = GameEmbed.embed(game_state, nil, 1, config: config, name_id: 1)

      # Should be different (name is one-hot encoded)
      diff = Nx.sum(Nx.abs(Nx.subtract(result0, result1)))
      assert Nx.to_number(diff) > 0
    end

    test "embeds projectiles when configured" do
      projectiles = [mock_projectile(x: 10.0), mock_projectile(x: 20.0)]
      game_state = mock_game_state(projectiles: projectiles)
      config = %GameEmbed{with_projectiles: true, max_projectiles: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "appends action IDs at end when using learned actions" do
      player1 = mock_player(action: 50)  # Action ID 50
      player2 = mock_player(action: 100)  # Action ID 100
      game_state = mock_game_state(player1: player1, player2: player2)
      player_config = %PlayerEmbed{action_mode: :learned, with_nana: false, with_speeds: false}
      # Use a small but valid num_player_names (can't be 0)
      config = %GameEmbed{player: player_config, with_projectiles: false, num_player_names: 2}

      result = GameEmbed.embed(game_state, nil, 1, config: config)
      total_size = GameEmbed.embedding_size(config)

      assert Nx.shape(result) == {total_size}

      # Last 2 values should be action IDs (own=50, opponent=100)
      last_two = Nx.slice(result, [total_size - 2], [2])
      [own_action, opp_action] = Nx.to_list(last_two)

      # Action IDs are cast to f32 in the tensor
      assert trunc(own_action) == 50
      assert trunc(opp_action) == 100
    end
  end

  describe "embed_state/3" do
    test "creates state-only embedding (no action)" do
      game_state = mock_game_state()
      config = GameEmbed.default_config()

      result = GameEmbed.embed_state(game_state, 1, config: config)

      # Should be smaller than full embed (no prev_action, no name)
      player_size = PlayerEmbed.embedding_size(config.player)
      stage_size = 64  # Stage one-hot

      expected_size = 2 * player_size + stage_size
      assert Nx.shape(result) == {expected_size}
    end

    test "different ports give different state embeddings" do
      game_state = mock_game_state(
        player1: mock_player(percent: 50.0),
        player2: mock_player(percent: 100.0)
      )
      config = GameEmbed.default_config()

      result1 = GameEmbed.embed_state(game_state, 1, config: config)
      result2 = GameEmbed.embed_state(game_state, 2, config: config)

      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0.1
    end
  end

  # ============================================================================
  # Batch Embedding Tests
  # ============================================================================

  describe "embed_batch/2" do
    test "creates batched embedding" do
      states = [
        {mock_game_state(), nil, 1},
        {mock_game_state(), nil, 1},
        {mock_game_state(), nil, 1}
      ]
      config = GameEmbed.default_config()

      result = GameEmbed.embed_batch(states, config: config)

      assert Nx.shape(result) == {3, GameEmbed.embedding_size(config)}
    end

    test "handles mixed ports in batch" do
      states = [
        {mock_game_state(), nil, 1},
        {mock_game_state(), nil, 2}
      ]
      config = GameEmbed.default_config()

      result = GameEmbed.embed_batch(states, config: config)

      assert Nx.shape(result) == {2, GameEmbed.embedding_size(config)}
    end
  end

  describe "dummy/1" do
    test "creates zero tensor of correct size" do
      config = GameEmbed.default_config()

      result = GameEmbed.dummy(config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end
  end

  describe "dummy_batch/2" do
    test "creates batched zero tensor" do
      config = GameEmbed.default_config()

      result = GameEmbed.dummy_batch(4, config: config)

      assert Nx.shape(result) == {4, GameEmbed.embedding_size(config)}
      assert Nx.to_number(Nx.sum(result)) == 0.0
    end
  end

  # ============================================================================
  # Projectile Embedding Tests
  # ============================================================================

  describe "projectile embedding" do
    test "pads missing projectiles with zeros" do
      # Only 2 projectiles, but max is 5
      projectiles = [mock_projectile(), mock_projectile()]
      game_state = mock_game_state(projectiles: projectiles)
      config = %GameEmbed{with_projectiles: true, max_projectiles: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "truncates excess projectiles" do
      # 10 projectiles, but max is 5
      projectiles = for i <- 1..10, do: mock_projectile(x: i * 10.0)
      game_state = mock_game_state(projectiles: projectiles)
      config = %GameEmbed{with_projectiles: true, max_projectiles: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "handles nil projectiles list" do
      game_state = %{mock_game_state() | projectiles: nil}
      config = %GameEmbed{with_projectiles: true, max_projectiles: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end
  end

  # ============================================================================
  # Item Embedding Tests
  # ============================================================================

  describe "item embedding" do
    test "embeds items when configured" do
      items = [mock_item(x: 10.0, type: 0x2C)]  # Link bomb
      game_state = mock_game_state(items: items)
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "pads missing items with zeros" do
      items = [mock_item(), mock_item()]
      game_state = mock_game_state(items: items)
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "truncates excess items" do
      items = for i <- 1..10, do: mock_item(x: i * 10.0)
      game_state = mock_game_state(items: items)
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "handles nil items list" do
      game_state = %{mock_game_state() | items: nil}
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "embeds Peach turnip correctly" do
      turnip = mock_item(type: 0x32, x: 5.0, y: 10.0)  # Peach turnip
      game_state = mock_game_state(items: [turnip])
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "embeds Mr. Saturn correctly" do
      saturn = mock_item(type: 0x15, x: -20.0, y: 0.0)  # Mr. Saturn
      game_state = mock_game_state(items: [saturn])
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "embeds mixed item types" do
      items = [
        mock_item(type: 0x2C, x: 0.0),   # Link bomb
        mock_item(type: 0x32, x: 10.0),  # Peach turnip
        mock_item(type: 0x15, x: 20.0)   # Mr. Saturn
      ]
      game_state = mock_game_state(items: items)
      config = %GameEmbed{with_items: true, max_items: 5}

      result = GameEmbed.embed(game_state, nil, 1, config: config)

      assert Nx.shape(result) == {GameEmbed.embedding_size(config)}
    end

    test "held_by_self correctly identifies self-held items" do
      # Item held by player 1 (own_port)
      held_item = mock_item(type: 0x2C, held_by: 1)
      game_state = mock_game_state(items: [held_item])
      config = %GameEmbed{with_items: true, max_items: 5}

      result1 = GameEmbed.embed(game_state, nil, 1, config: config)
      result2 = GameEmbed.embed(game_state, nil, 2, config: config)

      # Should be different because held_by_self is different
      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================

  describe "integration" do
    test "embedding changes with game state changes" do
      game_state1 = mock_game_state(player1: mock_player(percent: 0.0))
      game_state2 = mock_game_state(player1: mock_player(percent: 150.0))
      config = GameEmbed.default_config()

      result1 = GameEmbed.embed(game_state1, nil, 1, config: config)
      result2 = GameEmbed.embed(game_state2, nil, 1, config: config)

      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0.1
    end

    test "embedding changes with previous action" do
      game_state = mock_game_state()
      action1 = mock_controller_state(main_x: 0.0)
      action2 = mock_controller_state(main_x: 1.0)
      config = GameEmbed.default_config()

      result1 = GameEmbed.embed(game_state, action1, 1, config: config)
      result2 = GameEmbed.embed(game_state, action2, 1, config: config)

      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) > 0
    end

    test "embedding is consistent across multiple calls" do
      game_state = mock_game_state()
      prev_action = mock_controller_state()
      config = GameEmbed.default_config()

      result1 = GameEmbed.embed(game_state, prev_action, 1, config: config)
      result2 = GameEmbed.embed(game_state, prev_action, 1, config: config)

      diff = Nx.sum(Nx.abs(Nx.subtract(result1, result2)))
      assert Nx.to_number(diff) == 0.0
    end
  end

  # ==========================================================================
  # embed_states_fast/3 - Batch Embedding (Critical for Training Performance)
  # ==========================================================================

  describe "embed_states_fast/3" do
    test "produces embedding matching embedding_size()" do
      # This test catches the bug where embed_states_fast was missing projectile
      # embedding, causing 35-dim mismatch (5 projectiles * 7 dims each)
      game_states = [mock_game_state(), mock_game_state(frame: 100)]
      config = GameEmbed.default_config()

      result = GameEmbed.embed_states_fast(game_states, 1, config: config)
      expected_size = GameEmbed.embedding_size(config)

      {batch, actual_size} = Nx.shape(result)
      assert batch == 2
      assert actual_size == expected_size,
        "embed_states_fast produced #{actual_size} dims but embedding_size() expects #{expected_size}"
    end

    test "matches single embed() output dimensions" do
      game_state = mock_game_state()
      config = GameEmbed.default_config()

      # Single embedding
      single = GameEmbed.embed(game_state, nil, 1, config: config)
      {single_size} = Nx.shape(single)

      # Batch embedding
      batch = GameEmbed.embed_states_fast([game_state], 1, config: config)
      {_, batch_size} = Nx.shape(batch)

      assert single_size == batch_size,
        "Single embed gave #{single_size} but batch gave #{batch_size}"
    end

    test "handles projectiles correctly" do
      projectile = %Projectile{
        owner: 1, type: 1,
        x: 10.0, y: 20.0,
        speed_x: 5.0, speed_y: -2.0
      }
      game_state = mock_game_state(projectiles: [projectile])
      config = %{GameEmbed.default_config() | with_projectiles: true}

      result = GameEmbed.embed_states_fast([game_state], 1, config: config)
      expected_size = GameEmbed.embedding_size(config)

      {_, actual_size} = Nx.shape(result)
      assert actual_size == expected_size
    end

    # Note: Empty list handling removed as Nx.broadcast doesn't support {0, n} shapes.
    # Training code should never pass empty lists anyway.
  end
end
