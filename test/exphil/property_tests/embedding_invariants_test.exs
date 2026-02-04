defmodule ExPhil.PropertyTests.EmbeddingInvariantsTest do
  @moduledoc """
  Property-based tests for embedding invariants.

  These tests verify mathematical and logical properties that should hold
  for all valid inputs, not just specific test cases.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import ExPhil.Test.Generators

  alias ExPhil.Embeddings.{Player, Game, Primitives}
  alias ExPhil.Embeddings.Nana, as: NanaEmbed

  @moduletag :property

  describe "batch size invariance" do
    property "player embedding dimension is constant across batch sizes" do
      check all(
              batch_size <- StreamData.integer(1..32),
              players <- StreamData.list_of(player_gen(), length: batch_size),
              max_runs: 30
            ) do
        config = Player.default_config()
        expected_size = Player.embedding_size(config)

        embedded = Player.embed_batch(players, config)
        {actual_batch, actual_dim} = Nx.shape(embedded)

        assert actual_batch == batch_size
        assert actual_dim == expected_size
      end
    end

    property "game embedding dimension is constant across batch sizes" do
      check all(
              batch_size <- StreamData.integer(1..16),
              game_states <- StreamData.list_of(game_state_gen(), length: batch_size),
              max_runs: 20
            ) do
        config = Game.default_config()
        expected_size = Game.embedding_size(config)

        # Game.embed_batch expects list of {game_state, prev_action, own_port} tuples
        states_with_context = Enum.map(game_states, fn gs -> {gs, nil, 1} end)
        embedded = Game.embed_batch(states_with_context, config: config)
        {actual_batch, actual_dim} = Nx.shape(embedded)

        assert actual_batch == batch_size
        assert actual_dim == expected_size
      end
    end
  end

  describe "single vs batch equivalence" do
    property "single player embed equals batch embed at index 0" do
      check all(
              player <- player_gen(),
              max_runs: 50
            ) do
        config = Player.default_config()

        single = Player.embed(player, config)
        batch = Player.embed_batch([player], config)
        batch_first = Nx.squeeze(batch, axes: [0])

        # Should be very close (floating point)
        diff = Nx.subtract(single, batch_first) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        assert diff < 1.0e-5
      end
    end

    property "single game embed equals batch embed at index 0" do
      check all(
              game_state <- game_state_gen(),
              max_runs: 30
            ) do
        config = Game.default_config()

        # Game.embed takes (game_state, prev_action, own_port, opts)
        single = Game.embed(game_state, nil, 1, config: config)
        # Game.embed_batch expects list of {game_state, prev_action, own_port} tuples
        batch = Game.embed_batch([{game_state, nil, 1}], config: config)
        batch_first = Nx.squeeze(batch, axes: [0])

        diff = Nx.subtract(single, batch_first) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        assert diff < 1.0e-5
      end
    end
  end

  describe "primitives invariants" do
    property "one_hot has exactly one 1.0 for valid indices" do
      # Use bind for dependent generators (index depends on size)
      size_and_index_gen =
        StreamData.bind(StreamData.integer(10..100), fn size ->
          StreamData.bind(StreamData.integer(0..(size - 1)), fn index ->
            StreamData.constant({size, index})
          end)
        end)

      check all(
              {size, index} <- size_and_index_gen,
              max_runs: 100
            ) do
        one_hot = Primitives.one_hot(index, size: size)
        values = Nx.to_flat_list(one_hot)

        # Exactly one 1.0
        ones_count = Enum.count(values, fn v -> v == 1.0 end)
        assert ones_count == 1

        # The 1.0 is at the correct position
        assert Enum.at(values, index) == 1.0

        # All other values are 0.0
        zeros_count = Enum.count(values, fn v -> v == 0.0 end)
        assert zeros_count == size - 1
      end
    end

    property "batch_one_hot preserves one-hot property for each row" do
      check all(
              batch_size <- StreamData.integer(1..32),
              indices <- StreamData.list_of(StreamData.integer(0..24), length: batch_size),
              max_runs: 50
            ) do
        indices_tensor = Nx.tensor(indices, type: :s32)
        one_hot = Primitives.batch_one_hot(indices_tensor, size: 25)

        {rows, cols} = Nx.shape(one_hot)
        assert rows == batch_size
        assert cols == 25

        # Each row should have exactly one 1.0
        for row_idx <- 0..(batch_size - 1) do
          row = Nx.slice(one_hot, [row_idx, 0], [1, 25]) |> Nx.squeeze()
          values = Nx.to_flat_list(row)

          ones_count = Enum.count(values, fn v -> v == 1.0 end)
          assert ones_count == 1

          # The 1.0 should be at the correct index
          expected_idx = Enum.at(indices, row_idx)
          assert Enum.at(values, expected_idx) == 1.0
        end
      end
    end

    property "float_embed is bounded by lower and upper" do
      check all(
              value <- StreamData.float(min: -1000.0, max: 1000.0),
              scale <- StreamData.float(min: 0.001, max: 1.0),
              lower <- StreamData.float(min: -10.0, max: 0.0),
              upper <- StreamData.float(min: 0.0, max: 10.0),
              max_runs: 100
            ) do
        embedded = Primitives.float_embed(value, scale: scale, lower: lower, upper: upper)
        # float_embed returns a tensor, squeeze to scalar then convert
        val = embedded |> Nx.squeeze() |> Nx.to_number()

        # Allow small epsilon for f32 precision issues
        epsilon = 1.0e-5
        assert val >= lower - epsilon, "val #{val} should be >= lower #{lower}"
        assert val <= upper + epsilon, "val #{val} should be <= upper #{upper}"
      end
    end
  end

  describe "action category mapping" do
    property "all action IDs map to valid category (0-24)" do
      check all(
              action <- StreamData.integer(0..400),
              max_runs: 200
            ) do
        category = Player.action_to_category(action)

        assert is_integer(category)
        assert category >= 0
        assert category <= 24
      end
    end

    property "action_to_category is deterministic" do
      check all(
              action <- StreamData.integer(0..400),
              max_runs: 100
            ) do
        cat1 = Player.action_to_category(action)
        cat2 = Player.action_to_category(action)

        assert cat1 == cat2
      end
    end
  end

  describe "Nana helpers" do
    property "extract_batch_values returns correct number of elements" do
      check all(
              batch_size <- StreamData.integer(1..32),
              players <- StreamData.list_of(player_gen(), length: batch_size),
              max_runs: 30
            ) do
        values = NanaEmbed.extract_batch_values(players)

        assert length(values.exists) == batch_size
        assert length(values.xs) == batch_size
        assert length(values.ys) == batch_size
        assert length(values.facings) == batch_size
        assert length(values.percents) == batch_size
        assert length(values.stocks) == batch_size
        assert length(values.actions) == batch_size
      end
    end

    property "compute_batch_flags returns correct number of elements" do
      check all(
              batch_size <- StreamData.integer(1..32),
              players <- StreamData.list_of(player_gen(), length: batch_size),
              popo_actions <- StreamData.list_of(StreamData.integer(0..400), length: batch_size),
              max_runs: 30
            ) do
        values = NanaEmbed.extract_batch_values(players)
        flags = NanaEmbed.compute_batch_flags(values, popo_actions)

        assert length(flags.is_attacking) == batch_size
        assert length(flags.is_grabbing) == batch_size
        assert length(flags.can_act) == batch_size
        assert length(flags.is_synced) == batch_size
        assert length(flags.on_ground) == batch_size
        assert length(flags.categories) == batch_size
      end
    end

    property "IC flags are boolean" do
      check all(
              nana_action <- StreamData.integer(0..400),
              popo_action <- StreamData.integer(0..400),
              nana_y <- StreamData.float(min: -100.0, max: 100.0),
              max_runs: 100
            ) do
        flags = NanaEmbed.compute_flags(nana_action, popo_action, nana_y)

        assert is_boolean(flags.is_attacking)
        assert is_boolean(flags.is_grabbing)
        assert is_boolean(flags.can_act)
        assert is_boolean(flags.is_synced)
        assert is_boolean(flags.on_ground)
        assert is_integer(flags.category)
      end
    end

    property "sync_mode :exact is stricter than :category" do
      check all(
              nana_action <- StreamData.integer(0..400),
              popo_action <- StreamData.integer(0..400),
              nana_y <- StreamData.float(min: -100.0, max: 100.0),
              max_runs: 100
            ) do
        exact_flags = NanaEmbed.compute_flags(nana_action, popo_action, nana_y, sync_mode: :exact)
        cat_flags = NanaEmbed.compute_flags(nana_action, popo_action, nana_y, sync_mode: :category)

        # If exact sync is true, category sync must also be true
        # (same action implies same category)
        if exact_flags.is_synced do
          assert cat_flags.is_synced
        end

        # Category sync can be true when exact sync is false
        # (different actions in same category)
      end
    end
  end
end
