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

defmodule ExPhil.Training.NameConditioningLiveParityTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Data, PlayerRegistry}
  alias ExPhil.Bridge.{GameState, Player}

  # STYLE_IDENTITY.md S6(b): the live Agent embeds a frame with
  # `name_id: style_id` through Embeddings.Game.embed/4 (agent.ex ~2095);
  # training embeds through Data.precompute_frame_embeddings/2 with the
  # per-frame :name_id the registry assigned. Same frame + same id must
  # produce the same row, and --style-tag must resolve to the trainer's id.
  defp game_state do
    p = %Player{x: 1.0, y: 0.0, percent: 12.0, stock: 4, facing: 1, action: 14, action_frame: 3, character: 2,
                jumps_left: 2, on_ground: true, shield_strength: 60.0, invulnerable: false,
                speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0, speed_x_attack: 0.0, speed_y_attack: 0.0,
                hitstun_frames_left: 0, controller_state: nil, nana: nil}
    %GameState{frame: 100, stage: 32, players: %{1 => p, 2 => %{p | x: -1.0, facing: -1, character: 9}}, projectiles: [], items: [], distance: 2.0}
  end

  @tag :tmp_dir
  test "live name_id embedding equals the training row for the same id, and --style-tag resolves to it", %{tmp_dir: dir} do
    registry = PlayerRegistry.from_tags(["TITP", "SKWA", "~c07"], first_id: 1)
    path = Path.join(dir, "players.json")
    PlayerRegistry.to_json(registry, path)

    for tag <- ["TITP", "~c07"] do
      id = ExPhil.Agents.Decode.resolve_style_id(style_tag: tag, player_registry: path)
      assert id == PlayerRegistry.get_id(registry, tag)
      assert id > 0

      action = %{buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}, main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0}
      ds = Data.from_frames([%{game_state: game_state(), player_tag: tag, action: action}], player_registry: registry)
      assert hd(ds.frames)[:name_id] == id
      train_row = Data.precompute_frame_embeddings(ds, show_progress: false).embedded_frames[0]
      live_row = ExPhil.Embeddings.Game.embed(game_state(), nil, 1, name_id: id)

      assert Nx.shape(train_row) == Nx.shape(live_row)
      assert Nx.to_number(Nx.all_close(train_row, live_row, atol: 1.0e-6)) == 1
    end

    # and a different id changes the row (the slot is live)
    a = ExPhil.Embeddings.Game.embed(game_state(), nil, 1, name_id: 1)
    b = ExPhil.Embeddings.Game.embed(game_state(), nil, 1, name_id: 2)
    refute Nx.to_number(Nx.all_close(a, b)) == 1
  end
end
