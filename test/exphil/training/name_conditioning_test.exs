defmodule ExPhil.Training.NameConditioningTest do
  use ExUnit.Case, async: false

  alias ExPhil.Training.{Data, EmbeddingCache, PlayerRegistry}
  alias ExPhil.Bridge.{GameState, Player}

  # Minimal real game state (mirrors test/exphil/embeddings/game_test.exs
  # fixtures) — identical for every frame, so the ONLY thing that can vary
  # between embedded rows is the 112-dim name one-hot slot.
  defp mock_player(x) do
    %Player{
      character: 2, x: x, y: 0.0, percent: 0.0, stock: 4, facing: 1,
      action: 14, action_frame: 0, invulnerable: false, jumps_left: 2,
      on_ground: true, shield_strength: 60.0, hitstun_frames_left: 0,
      speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0,
      speed_x_attack: 0.0, speed_y_attack: 0.0, nana: nil, controller_state: nil
    }
  end

  defp game_state do
    %GameState{
      frame: 0,
      stage: 32,
      menu_state: 2,
      players: %{1 => mock_player(-30.0), 2 => mock_player(30.0)},
      projectiles: [],
      items: [],
      distance: 60.0
    }
  end

  defp frame(tag) do
    %{
      game_state: game_state(),
      player_tag: tag,
      action: %{
        buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
        main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0
      }
    }
  end

  test "registry ids flow through from_frames into distinct embedded bytes" do
    registry = PlayerRegistry.from_tags(["MANG", "ZAIN"])

    ds =
      Data.from_frames([frame("MANG"), frame("ZAIN"), frame("MANG")],
        player_registry: registry
      )

    ids = Enum.map(ds.frames, & &1[:name_id])
    assert length(Enum.uniq(Enum.take(ids, 2))) == 2
    assert Enum.at(ids, 0) == Enum.at(ids, 2)

    embedded = Data.precompute_frame_embeddings(ds, show_progress: false)
    e = embedded.embedded_frames

    # Same tag -> identical embedding rows; different tag -> different rows
    # (identical game_state, so ONLY the name one-hot slot can differ).
    assert Nx.all_close(e[0], e[2]) |> Nx.to_number() == 1
    refute Nx.all_close(e[0], e[1]) |> Nx.to_number() == 1
  end

  test "without a registry the name slot is inert (all frames id 0)" do
    ds = Data.from_frames([frame("MANG"), frame("ZAIN")])
    embedded = Data.precompute_frame_embeddings(ds, show_progress: false)
    e = embedded.embedded_frames
    assert Nx.all_close(e[0], e[1]) |> Nx.to_number() == 1
  end

  test "cache key changes with the registry mapping (stale-hit guard)" do
    reg_a = PlayerRegistry.from_tags(["MANG", "ZAIN"])
    reg_b = PlayerRegistry.from_tags(["MANG", "COSMOS"])
    config = ExPhil.Embeddings.config([])
    files = ["a.slp", "b.slp"]

    k_none = EmbeddingCache.cache_key(config, files, [])
    k_a = EmbeddingCache.cache_key(config, files, player_registry: reg_a)
    k_b = EmbeddingCache.cache_key(config, files, player_registry: reg_b)

    assert k_none != k_a
    assert k_a != k_b
    assert k_a == EmbeddingCache.cache_key(config, files, player_registry: reg_a)
  end
end
