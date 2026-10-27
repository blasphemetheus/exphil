defmodule ExPhil.Embeddings.PlayerFrameParityTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias ExPhil.Embeddings.Player
  alias ExPhil.Bridge.Player, as: PlayerState

  test "single-game distance uses the same f32 arithmetic as batches" do
    alias ExPhil.Embeddings.Game.Spatial

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      inputs =
        for i <- 0..64 do
          {%ExPhil.Bridge.GameState{distance: i * 1.234567}, nil, nil}
        end

      singles =
        inputs
        |> Enum.map(fn {game, own, opp} -> Spatial.embed_distance(game, own, opp) end)
        |> Nx.stack()

      assert_equal(singles, Spatial.embed_distances_batch(inputs))
    end)
  end

  test "single-player timing features use the same f32 arithmetic as training batches" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      config = Player.default_config()

      players =
        for frame <- [-5, 0, 1, 7, 29, 61, 119, 121, 250] do
          %PlayerState{
            character: 2,
            action: 14,
            action_frame: frame,
            hitstun_frames_left: frame,
            x: 12.3,
            y: -4.7,
            percent: 37.0,
            shield_strength: 56.0,
            jumps_left: 2,
            stock: 4,
            facing: true,
            on_ground: true,
            invulnerable: false
          }
        end

      singles = players |> Enum.map(&Player.embed(&1, config)) |> Nx.stack()
      batch = Player.embed_batch(players, config)
      # These host inputs must be identical before the model is compiled.
      assert_equal(singles, batch)
    end)
  end
end
