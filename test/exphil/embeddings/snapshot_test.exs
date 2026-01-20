defmodule ExPhil.Embeddings.SnapshotTest do
  @moduledoc """
  Snapshot tests for embedding outputs.

  These tests verify that embedding functions produce consistent outputs
  by comparing against saved "snapshots". If the embedding logic changes
  unintentionally, these tests will catch it.

  ## How It Works

  1. Each test computes embeddings on a fixed input
  2. The output is compared against a saved snapshot (JSON file)
  3. If no snapshot exists, one is created (run with SNAPSHOT_UPDATE=1)
  4. Small floating-point differences are tolerated (atol=1e-5)

  ## Updating Snapshots

      # Update all snapshots when changes are intentional
      SNAPSHOT_UPDATE=1 mix test --only snapshot

  ## When Tests Fail

  If a snapshot test fails, it means the embedding output changed.
  Either:
  1. The change is a bug - fix it
  2. The change is intentional - run with SNAPSHOT_UPDATE=1
  """

  use ExUnit.Case, async: true
  import ExPhil.Test.Helpers
  import ExPhil.Test.Factories
  import ExPhil.Test.ReplayFixtures

  alias ExPhil.Embeddings.{Player, Game, Controller}

  @moduletag :snapshot
  @snapshot_dir "test/fixtures/embedding_snapshots"

  setup_all do
    File.mkdir_p!(@snapshot_dir)
    :ok
  end

  describe "Player.embed/1 snapshots" do
    test "neutral standing player" do
      player = build_player(
        character: 10,  # Mewtwo
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        facing: 1,
        action: 14,  # Wait
        on_ground: true
      )

      embedding = Player.embed(player)
      assert_snapshot("player_neutral_standing", embedding)
    end

    test "player in hitstun airborne" do
      player = build_player(
        character: 2,  # Fox
        x: 45.0,
        y: 30.0,
        percent: 85.0,
        stock: 3,
        facing: -1,
        action: 75,  # Damage fly
        action_frame: 8,
        on_ground: false,
        jumps_left: 1,
        hitstun_frames_left: 15,
        speed_air_x_self: 2.5,
        speed_y_self: -1.2,
        speed_x_attack: 1.8,
        speed_y_attack: 2.0
      )

      embedding = Player.embed(player)
      assert_snapshot("player_hitstun_airborne", embedding)
    end

    test "player shielding low health" do
      player = build_player(
        character: 9,  # Marth
        x: -20.0,
        y: 0.0,
        percent: 120.0,
        stock: 1,
        facing: 1,
        action: 178,  # Shield
        shield_strength: 35.0,
        on_ground: true
      )

      embedding = Player.embed(player)
      assert_snapshot("player_shielding_low", embedding)
    end
  end

  describe "Game.embed/3 snapshots" do
    test "neutral game state" do
      game_state = neutral_game_fixture(:mewtwo_vs_fox)

      embedding = Game.embed(game_state, nil, 1)
      assert_snapshot("game_neutral_mewtwo_fox", embedding)
    end

    test "edge guard scenario" do
      game_state = edge_guard_fixture(:fox_recovering_low)

      embedding = Game.embed(game_state, nil, 1)
      assert_snapshot("game_edge_guard", embedding)
    end

    test "different player perspective" do
      game_state = neutral_game_fixture(:mewtwo_vs_fox)

      # Same game, different perspective
      embed_p1 = Game.embed(game_state, nil, 1)
      embed_p2 = Game.embed(game_state, nil, 2)

      # Save both perspectives
      assert_snapshot("game_perspective_p1", embed_p1)
      assert_snapshot("game_perspective_p2", embed_p2)

      # Verify they're different (players swapped)
      refute Nx.all(Nx.equal(embed_p1, embed_p2)) |> Nx.to_number() == 1
    end
  end

  describe "Controller.embed_continuous/1 snapshots" do
    test "neutral controller" do
      controller = build_controller_state()

      embedding = Controller.embed_continuous(controller)
      assert_snapshot("controller_neutral", embedding)
    end

    test "attack input" do
      controller = build_controller_state(
        button_a: true,
        main_stick: %{x: 0.8, y: 0.5}  # Forward tilt
      )

      embedding = Controller.embed_continuous(controller)
      assert_snapshot("controller_attack", embedding)
    end

    test "shield input" do
      controller = build_controller_state(
        button_r: true,
        l_shoulder: 0.9
      )

      embedding = Controller.embed_continuous(controller)
      assert_snapshot("controller_shield", embedding)
    end
  end

  # ============================================================================
  # Snapshot Assertion Helper
  # ============================================================================

  defp assert_snapshot(name, tensor) do
    snapshot_path = Path.join(@snapshot_dir, "#{name}.json")

    if System.get_env("SNAPSHOT_UPDATE") do
      save_snapshot(snapshot_path, tensor)
      IO.puts("[Snapshot] Updated: #{name}")
    else
      case load_snapshot(snapshot_path) do
        {:ok, expected} ->
          assert_tensors_close(tensor, expected, atol: 1.0e-5)

        {:error, :not_found} ->
          # First run - create the snapshot
          save_snapshot(snapshot_path, tensor)
          IO.puts("[Snapshot] Created: #{name}")
      end
    end
  end

  defp save_snapshot(path, tensor) do
    data = %{
      "shape" => Tuple.to_list(Nx.shape(tensor)),
      "type" => Atom.to_string(Nx.type(tensor) |> elem(0)),
      "values" => Nx.to_flat_list(tensor),
      "updated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    File.write!(path, Jason.encode!(data, pretty: true))
  end

  defp load_snapshot(path) do
    case File.read(path) do
      {:ok, content} ->
        data = Jason.decode!(content)
        shape = List.to_tuple(data["shape"])
        type = String.to_atom(data["type"])

        tensor = data["values"]
        |> Nx.tensor(type: {type, 32})
        |> Nx.reshape(shape)

        {:ok, tensor}

      {:error, :enoent} ->
        {:error, :not_found}
    end
  end
end
