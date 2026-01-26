defmodule ExPhil.Embeddings.PropertyTest do
  @moduledoc """
  Property-based tests for embedding modules.

  These tests use StreamData generators to verify that embeddings
  work correctly for a wide range of inputs.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import ExPhil.Test.Generators

  alias ExPhil.Embeddings.{Player, Game, Controller}

  @moduletag :slow

  describe "Player.embed/2" do
    property "produces finite tensor for any valid player" do
      check all(
              player <- player_gen(),
              max_runs: 50
            ) do
        # Player.embed(player, config) - use default config
        embedded = Player.embed(player)

        # Should be a tensor
        assert is_struct(embedded, Nx.Tensor)

        # Should have correct shape
        {size} = Nx.shape(embedded)
        assert size > 0

        # Should be finite (no NaN or Inf)
        assert Nx.all(Nx.is_nan(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    property "embedding size is consistent regardless of input" do
      check all(
              player1 <- player_gen(),
              player2 <- player_gen(),
              max_runs: 20
            ) do
        # Use same config for both
        embedded1 = Player.embed(player1)
        embedded2 = Player.embed(player2)

        assert Nx.shape(embedded1) == Nx.shape(embedded2)
      end
    end
  end

  describe "Controller.embed_continuous/1" do
    property "produces finite tensor for any valid controller state" do
      check all(
              controller <- controller_state_gen(),
              max_runs: 50
            ) do
        embedded = Controller.embed_continuous(controller)

        assert is_struct(embedded, Nx.Tensor)

        # Should be finite
        assert Nx.all(Nx.is_nan(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    property "button embeddings are binary" do
      check all(
              controller <- controller_state_gen(),
              max_runs: 50
            ) do
        embedded = Controller.embed_continuous(controller)

        # First 8 values are button embeddings (0 or 1)
        buttons = Nx.slice(embedded, [0], [8])
        values = Nx.to_flat_list(buttons)

        for val <- values do
          assert val == 0.0 or val == 1.0
        end
      end
    end
  end

  describe "Game.embed/4" do
    property "produces finite tensor for any valid game state" do
      check all(
              game_state <- game_state_gen(),
              max_runs: 30
            ) do
        # Game.embed(game_state, prev_action, own_port, opts)
        # Use nil for prev_action (no previous controller state)
        embedded = Game.embed(game_state, nil, 1)

        assert is_struct(embedded, Nx.Tensor)

        # Should be finite
        assert Nx.all(Nx.is_nan(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    property "embedding respects player perspective" do
      check all(
              game_state <- game_state_gen(),
              max_runs: 20
            ) do
        # Embedding from player 1 vs player 2 perspective should differ
        embed1 = Game.embed(game_state, nil, 1)
        embed2 = Game.embed(game_state, nil, 2)

        # Shapes should be the same
        assert Nx.shape(embed1) == Nx.shape(embed2)

        # But values should differ (unless both players are identical)
        p1 = game_state.players[1]
        p2 = game_state.players[2]

        if p1.x != p2.x or p1.y != p2.y do
          refute Nx.equal(embed1, embed2) |> Nx.all() |> Nx.to_number() == 1
        end
      end
    end
  end
end
