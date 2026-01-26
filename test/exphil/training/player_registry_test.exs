defmodule ExPhil.Training.PlayerRegistryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.PlayerRegistry

  describe "new/1" do
    test "creates empty registry with defaults" do
      registry = PlayerRegistry.new()

      assert registry.max_players == 112
      assert registry.unknown_strategy == nil
      assert PlayerRegistry.size(registry) == 0
    end

    test "accepts custom max_players" do
      registry = PlayerRegistry.new(max_players: 50)
      assert registry.max_players == 50
    end

    test "accepts unknown_strategy option" do
      registry = PlayerRegistry.new(unknown_strategy: :hash)
      assert registry.unknown_strategy == :hash
    end
  end

  describe "from_tags/2" do
    test "creates registry from list of tags" do
      tags = ["Plup", "Jmook", "Mango"]
      registry = PlayerRegistry.from_tags(tags)

      assert PlayerRegistry.size(registry) == 3
      assert PlayerRegistry.get_id(registry, "Plup") == 0
      assert PlayerRegistry.get_id(registry, "Jmook") == 1
      assert PlayerRegistry.get_id(registry, "Mango") == 2
    end

    test "limits to max_players" do
      tags = ["A", "B", "C", "D", "E"]
      registry = PlayerRegistry.from_tags(tags, max_players: 3)

      assert PlayerRegistry.size(registry) == 3
      assert PlayerRegistry.has_tag?(registry, "A")
      assert PlayerRegistry.has_tag?(registry, "B")
      assert PlayerRegistry.has_tag?(registry, "C")
      refute PlayerRegistry.has_tag?(registry, "D")
    end
  end

  describe "get_id/2" do
    test "returns id for known tag" do
      registry = PlayerRegistry.from_tags(["Plup", "Jmook"])

      assert PlayerRegistry.get_id(registry, "Plup") == 0
      assert PlayerRegistry.get_id(registry, "Jmook") == 1
    end

    test "returns nil for unknown tag with :nil strategy" do
      registry = PlayerRegistry.from_tags(["Plup"], unknown_strategy: nil)

      assert PlayerRegistry.get_id(registry, "Unknown") == nil
    end

    test "returns 0 for unknown tag with :default strategy" do
      registry = PlayerRegistry.from_tags(["Plup"], unknown_strategy: :default)

      assert PlayerRegistry.get_id(registry, "Unknown") == 0
    end

    test "returns hash for unknown tag with :hash strategy" do
      registry = PlayerRegistry.from_tags(["Plup"], max_players: 100, unknown_strategy: :hash)

      id = PlayerRegistry.get_id(registry, "SomeRandomPlayer")
      assert is_integer(id)
      assert id >= 0 and id < 100

      # Same tag should give same hash
      assert PlayerRegistry.get_id(registry, "SomeRandomPlayer") == id
    end

    test "handles nil tag" do
      registry = PlayerRegistry.from_tags(["Plup"])
      assert PlayerRegistry.get_id(registry, nil) == nil
    end

    test "handles empty string tag" do
      registry = PlayerRegistry.from_tags(["Plup"])
      assert PlayerRegistry.get_id(registry, "") == nil
    end
  end

  describe "get_tag/2" do
    test "returns tag for known id" do
      registry = PlayerRegistry.from_tags(["Plup", "Jmook"])

      assert PlayerRegistry.get_tag(registry, 0) == "Plup"
      assert PlayerRegistry.get_tag(registry, 1) == "Jmook"
    end

    test "returns nil for unknown id" do
      registry = PlayerRegistry.from_tags(["Plup"])

      assert PlayerRegistry.get_tag(registry, 99) == nil
    end
  end

  describe "add_tag/2" do
    test "adds new tag and returns id" do
      registry = PlayerRegistry.new()
      {registry, id} = PlayerRegistry.add_tag(registry, "Plup")

      assert id == 0
      assert PlayerRegistry.has_tag?(registry, "Plup")
      assert PlayerRegistry.size(registry) == 1
    end

    test "returns existing id for duplicate tag" do
      registry = PlayerRegistry.from_tags(["Plup"])
      {registry2, id} = PlayerRegistry.add_tag(registry, "Plup")

      assert id == 0
      assert PlayerRegistry.size(registry2) == 1
    end

    test "handles overflow with unknown_strategy" do
      registry = PlayerRegistry.from_tags(["A", "B"], max_players: 2, unknown_strategy: :hash)
      {_registry, id} = PlayerRegistry.add_tag(registry, "C")

      # Should use hash since registry is full
      assert is_integer(id)
      assert id >= 0 and id < 2
    end
  end

  describe "list_tags/1" do
    test "returns tags in order by id" do
      registry = PlayerRegistry.from_tags(["Plup", "Jmook", "Mango"])

      assert PlayerRegistry.list_tags(registry) == ["Plup", "Jmook", "Mango"]
    end
  end

  describe "has_tag?/2" do
    test "returns true for known tags" do
      registry = PlayerRegistry.from_tags(["Plup"])

      assert PlayerRegistry.has_tag?(registry, "Plup")
    end

    test "returns false for unknown tags" do
      registry = PlayerRegistry.from_tags(["Plup"])

      refute PlayerRegistry.has_tag?(registry, "Unknown")
    end
  end

  describe "JSON serialization" do
    @tag :tmp_dir
    test "round-trips through JSON", %{tmp_dir: tmp_dir} do
      registry =
        PlayerRegistry.from_tags(["Plup", "Jmook", "Mango"],
          max_players: 50,
          unknown_strategy: :hash
        )

      json_path = Path.join(tmp_dir, "players.json")

      assert :ok = PlayerRegistry.to_json(registry, json_path)
      assert {:ok, loaded} = PlayerRegistry.from_json(json_path)

      assert PlayerRegistry.size(loaded) == 3
      assert PlayerRegistry.get_id(loaded, "Plup") == 0
      assert PlayerRegistry.get_id(loaded, "Jmook") == 1
      assert PlayerRegistry.get_id(loaded, "Mango") == 2
      assert loaded.max_players == 50
      assert loaded.unknown_strategy == :hash
    end
  end
end
