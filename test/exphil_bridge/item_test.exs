defmodule ExPhil.Bridge.ItemTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.Item

  defp mock_item(opts) do
    %Item{
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      type: Keyword.get(opts, :type, 0x2C),
      facing: Keyword.get(opts, :facing, 1),
      owner: Keyword.get(opts, :owner, 1),
      held_by: Keyword.get(opts, :held_by, nil),
      spawn_id: Keyword.get(opts, :spawn_id, 1),
      timer: Keyword.get(opts, :timer, 120)
    }
  end

  describe "bomb?/1" do
    test "returns true for Link bomb" do
      assert Item.bomb?(mock_item(type: 0x2C))
    end

    test "returns true for Young Link bomb" do
      assert Item.bomb?(mock_item(type: 0x2D))
    end

    test "returns true for Bob-omb" do
      assert Item.bomb?(mock_item(type: 0x13))
    end

    test "returns false for non-bomb items" do
      refute Item.bomb?(mock_item(type: 0x32))  # Turnip
      refute Item.bomb?(mock_item(type: 0x04))  # Beam Sword
    end

    test "returns false for nil" do
      refute Item.bomb?(nil)
    end
  end

  describe "turnip?/1" do
    test "returns true for Peach turnip" do
      assert Item.turnip?(mock_item(type: 0x32))
    end

    test "returns false for non-turnip items" do
      refute Item.turnip?(mock_item(type: 0x2C))  # Link bomb
      refute Item.turnip?(mock_item(type: 0x15))  # Mr. Saturn
    end

    test "returns false for nil" do
      refute Item.turnip?(nil)
    end
  end

  describe "mr_saturn?/1" do
    test "returns true for Mr. Saturn" do
      assert Item.mr_saturn?(mock_item(type: 0x15))
    end

    test "returns false for non-Mr. Saturn items" do
      refute Item.mr_saturn?(mock_item(type: 0x2C))  # Link bomb
      refute Item.mr_saturn?(mock_item(type: 0x32))  # Turnip
    end

    test "returns false for nil" do
      refute Item.mr_saturn?(nil)
    end
  end

  describe "held?/1" do
    test "returns true when item is held" do
      assert Item.held?(mock_item(held_by: 1))
      assert Item.held?(mock_item(held_by: 2))
    end

    test "returns false when item is not held" do
      refute Item.held?(mock_item(held_by: nil))
      refute Item.held?(mock_item(held_by: 0))
    end

    test "returns false for nil" do
      refute Item.held?(nil)
    end
  end

  describe "owned_by?/2" do
    test "returns true when owner matches" do
      assert Item.owned_by?(mock_item(owner: 1), 1)
      assert Item.owned_by?(mock_item(owner: 2), 2)
    end

    test "returns false when owner doesn't match" do
      refute Item.owned_by?(mock_item(owner: 1), 2)
      refute Item.owned_by?(mock_item(owner: 2), 1)
    end
  end

  describe "item_category/1" do
    test "returns 1 for bombs" do
      assert Item.item_category(mock_item(type: 0x2C)) == 1  # Link bomb
      assert Item.item_category(mock_item(type: 0x2D)) == 1  # Young Link bomb
      assert Item.item_category(mock_item(type: 0x13)) == 1  # Bob-omb
    end

    test "returns 2 for melee weapons" do
      assert Item.item_category(mock_item(type: 0x04)) == 2  # Beam Sword
      assert Item.item_category(mock_item(type: 0x05)) == 2  # Home Run Bat
      assert Item.item_category(mock_item(type: 0x06)) == 2  # Fan
    end

    test "returns 3 for ranged weapons" do
      assert Item.item_category(mock_item(type: 0x08)) == 3  # Ray Gun
      assert Item.item_category(mock_item(type: 0x09)) == 3  # Super Scope
    end

    test "returns 4 for containers" do
      assert Item.item_category(mock_item(type: 0x00)) == 4  # Capsule
      assert Item.item_category(mock_item(type: 0x01)) == 4  # Crate
      assert Item.item_category(mock_item(type: 0x02)) == 4  # Barrel
    end

    test "returns 5 for thrown/character-specific items" do
      assert Item.item_category(mock_item(type: 0x32)) == 5  # Peach turnip
      assert Item.item_category(mock_item(type: 0x15)) == 5  # Mr. Saturn
    end

    test "returns 5 for unknown items" do
      assert Item.item_category(mock_item(type: 0xFF)) == 5
      assert Item.item_category(mock_item(type: 0x99)) == 5
    end

    test "returns 0 for nil" do
      assert Item.item_category(nil) == 0
    end
  end
end
