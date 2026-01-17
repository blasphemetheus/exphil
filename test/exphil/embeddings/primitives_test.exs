defmodule ExPhil.Embeddings.PrimitivesTest do
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings.Primitives

  # Helper to convert tensor to number, handling {1} shape
  defp tensor_to_number(tensor) do
    case Nx.shape(tensor) do
      {} -> Nx.to_number(tensor)
      {1} -> tensor |> Nx.squeeze() |> Nx.to_number()
      _ -> raise "Expected scalar or {1} tensor"
    end
  end

  # ============================================================================
  # One-Hot Encoding Tests
  # ============================================================================

  describe "one_hot/2" do
    test "creates correct one-hot vector" do
      result = Primitives.one_hot(3, size: 5)

      assert Nx.shape(result) == {5}
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 1.0, 0.0]
    end

    test "handles zero index" do
      result = Primitives.one_hot(0, size: 5)

      assert Nx.to_flat_list(result) == [1.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "handles last index" do
      result = Primitives.one_hot(4, size: 5)

      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]
    end

    test "clamps values above max by default" do
      result = Primitives.one_hot(10, size: 5)

      # Should clamp to index 4
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]
    end

    test "clamps negative values by default" do
      result = Primitives.one_hot(-5, size: 5)

      # Should clamp to index 0
      assert Nx.to_flat_list(result) == [1.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "accepts tensor input" do
      result = Primitives.one_hot(Nx.tensor(2), size: 5)

      assert Nx.to_flat_list(result) == [0.0, 0.0, 1.0, 0.0, 0.0]
    end

    test "disabling clamp allows out-of-range values" do
      # Without clamp, values outside range produce incorrect results
      # This is intentional for advanced use cases
      result = Primitives.one_hot(3, size: 5, clamp: false)

      assert Nx.shape(result) == {5}
    end
  end

  describe "one_hot_with_unknown/2" do
    test "creates one-hot vector with unknown dimension" do
      result = Primitives.one_hot_with_unknown(2, size: 4)

      # Size 4 + 1 unknown = 5 dimensions
      assert Nx.shape(result) == {5}
      # Index 2 should be hot, unknown should be 0
      assert Nx.to_flat_list(result) == [0.0, 0.0, 1.0, 0.0, 0.0]
    end

    test "maps negative values to unknown" do
      result = Primitives.one_hot_with_unknown(-1, size: 4)

      assert Nx.shape(result) == {5}
      # All regular dimensions 0, unknown = 1
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]
    end

    test "maps values >= size to unknown" do
      result = Primitives.one_hot_with_unknown(5, size: 4)

      # Last dimension (unknown) should be 1
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  describe "batch_one_hot/2" do
    test "creates batch of one-hot vectors" do
      values = Nx.tensor([0, 2, 4])
      result = Primitives.batch_one_hot(values, size: 5)

      assert Nx.shape(result) == {3, 5}

      [row0, row1, row2] = Nx.to_list(result)
      assert row0 == [1.0, 0.0, 0.0, 0.0, 0.0]
      assert row1 == [0.0, 0.0, 1.0, 0.0, 0.0]
      assert row2 == [0.0, 0.0, 0.0, 0.0, 1.0]
    end

    test "clamps out-of-range values in batch" do
      values = Nx.tensor([-1, 10])
      result = Primitives.batch_one_hot(values, size: 5)

      assert Nx.shape(result) == {2, 5}

      [row0, row1] = Nx.to_list(result)
      # -1 clamped to 0
      assert row0 == [1.0, 0.0, 0.0, 0.0, 0.0]
      # 10 clamped to 4
      assert row1 == [0.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  # ============================================================================
  # Float Embedding Tests
  # ============================================================================

  describe "float_embed/2" do
    test "embeds float with default options" do
      result = Primitives.float_embed(5.0)

      assert Nx.shape(result) == {1}
      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end

    test "applies scale" do
      result = Primitives.float_embed(50.0, scale: 0.1)

      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end

    test "applies bias before scale" do
      result = Primitives.float_embed(10.0, bias: 5.0, scale: 0.1)

      # (10 + 5) * 0.1 = 1.5
      assert_in_delta tensor_to_number(result), 1.5, 0.001
    end

    test "clamps to upper bound" do
      result = Primitives.float_embed(100.0, upper: 5.0)

      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end

    test "clamps to lower bound" do
      result = Primitives.float_embed(-100.0, lower: -3.0)

      assert_in_delta tensor_to_number(result), -3.0, 0.001
    end

    test "accepts integer input" do
      result = Primitives.float_embed(5)

      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end

    test "accepts tensor input" do
      result = Primitives.float_embed(Nx.tensor(5.0))

      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end
  end

  describe "xy_embed/2" do
    test "scales position by default 0.05" do
      result = Primitives.xy_embed(100.0)

      # 100 * 0.05 = 5.0
      assert_in_delta tensor_to_number(result), 5.0, 0.001
    end

    test "accepts custom scale" do
      result = Primitives.xy_embed(100.0, scale: 0.1)

      # 100 * 0.1 = 10.0, clamped to 10.0
      assert_in_delta tensor_to_number(result), 10.0, 0.001
    end

    test "handles negative positions" do
      result = Primitives.xy_embed(-80.0)

      # -80 * 0.05 = -4.0
      assert_in_delta tensor_to_number(result), -4.0, 0.001
    end
  end

  describe "percent_embed/1" do
    test "scales percent by 0.01" do
      result = Primitives.percent_embed(150.0)

      # 150 * 0.01 = 1.5
      assert_in_delta tensor_to_number(result), 1.5, 0.001
    end

    test "clamps to 0-5 range" do
      low_result = Primitives.percent_embed(-50.0)
      high_result = Primitives.percent_embed(600.0)

      assert_in_delta tensor_to_number(low_result), 0.0, 0.001
      assert_in_delta tensor_to_number(high_result), 5.0, 0.001
    end

    test "handles zero percent" do
      result = Primitives.percent_embed(0.0)

      assert_in_delta tensor_to_number(result), 0.0, 0.001
    end
  end

  describe "shield_embed/1" do
    test "scales shield by 0.01" do
      result = Primitives.shield_embed(60.0)

      # 60 * 0.01 = 0.6
      assert_in_delta tensor_to_number(result), 0.6, 0.001
    end

    test "clamps to 0-1 range" do
      high_result = Primitives.shield_embed(200.0)

      assert_in_delta tensor_to_number(high_result), 1.0, 0.001
    end
  end

  describe "speed_embed/1" do
    test "scales speed by 0.5" do
      result = Primitives.speed_embed(4.0)

      # 4.0 * 0.5 = 2.0
      assert_in_delta tensor_to_number(result), 2.0, 0.001
    end
  end

  # ============================================================================
  # Bool Embedding Tests
  # ============================================================================

  describe "bool_embed/2" do
    test "embeds true as 1.0" do
      result = Primitives.bool_embed(true)

      assert Nx.shape(result) == {1}
      assert_in_delta tensor_to_number(result), 1.0, 0.001
    end

    test "embeds false as 0.0" do
      result = Primitives.bool_embed(false)

      assert_in_delta tensor_to_number(result), 0.0, 0.001
    end

    test "accepts custom on/off values" do
      result = Primitives.bool_embed(true, on: 2.0, off: -2.0)

      assert_in_delta tensor_to_number(result), 2.0, 0.001
    end

    test "treats non-zero integers as true" do
      result = Primitives.bool_embed(1)

      assert_in_delta tensor_to_number(result), 1.0, 0.001
    end

    test "treats zero as false" do
      result = Primitives.bool_embed(0)

      assert_in_delta tensor_to_number(result), 0.0, 0.001
    end
  end

  describe "facing_embed/1" do
    test "embeds right-facing as 1.0" do
      result = Primitives.facing_embed(true)

      assert_in_delta tensor_to_number(result), 1.0, 0.001
    end

    test "embeds left-facing as -1.0" do
      result = Primitives.facing_embed(false)

      assert_in_delta tensor_to_number(result), -1.0, 0.001
    end

    test "accepts integer input" do
      right = Primitives.facing_embed(1)
      left = Primitives.facing_embed(0)

      assert_in_delta tensor_to_number(right), 1.0, 0.001
      assert_in_delta tensor_to_number(left), -1.0, 0.001
    end
  end

  # ============================================================================
  # Specialized Embedding Tests
  # ============================================================================

  describe "action_embed/1" do
    test "creates 399-dimensional one-hot vector" do
      result = Primitives.action_embed(14)  # Common wait action

      assert Nx.shape(result) == {399}
      # Sum should be 1.0
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 0.001
    end

    test "clamps action to valid range" do
      # Action beyond max should clamp
      result = Primitives.action_embed(1000)

      assert Nx.shape(result) == {399}
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 0.001
    end
  end

  describe "character_embed/1" do
    test "creates 33-dimensional one-hot vector" do
      result = Primitives.character_embed(10)  # Mewtwo

      assert Nx.shape(result) == {33}
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 0.001
    end
  end

  describe "stage_embed/1" do
    test "creates 64-dimensional one-hot vector" do
      result = Primitives.stage_embed(32)  # Final Destination

      assert Nx.shape(result) == {64}
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 0.001
    end
  end

  describe "jumps_left_embed/1" do
    test "creates 7-dimensional one-hot vector" do
      result = Primitives.jumps_left_embed(2)

      assert Nx.shape(result) == {7}
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 0.001
    end

    test "handles max jumps (6 for Puff/Kirby)" do
      result = Primitives.jumps_left_embed(6)

      assert Nx.shape(result) == {7}
      # Index 6 should be hot
      assert tensor_to_number(Nx.slice(result, [6], [1])) == 1.0
    end
  end

  describe "item_type_embed/1" do
    test "creates embedding with unknown dimension" do
      result = Primitives.item_type_embed(10)

      # 237 known types + 1 unknown
      assert Nx.shape(result) == {238}
    end

    test "maps invalid item types to unknown" do
      result = Primitives.item_type_embed(-1)

      # Last dimension should be 1 (unknown)
      unknown_value = tensor_to_number(Nx.slice(result, [237], [1]))
      assert_in_delta unknown_value, 1.0, 0.001
    end
  end

  describe "item_state_embed/1" do
    test "creates embedding with unknown dimension" do
      result = Primitives.item_state_embed(5)

      # 12 known states + 1 unknown
      assert Nx.shape(result) == {13}
    end
  end

  # ============================================================================
  # Embedding Size Tests
  # ============================================================================

  describe "embedding_size/1" do
    test "returns correct sizes for all types" do
      assert Primitives.embedding_size(:action) == 399
      assert Primitives.embedding_size(:character) == 33
      assert Primitives.embedding_size(:stage) == 64
      assert Primitives.embedding_size(:jumps_left) == 7
      assert Primitives.embedding_size(:item_type) == 238
      assert Primitives.embedding_size(:item_state) == 13
      assert Primitives.embedding_size(:float) == 1
      assert Primitives.embedding_size(:bool) == 1
      assert Primitives.embedding_size(:xy) == 1
      assert Primitives.embedding_size(:percent) == 1
      assert Primitives.embedding_size(:shield) == 1
      assert Primitives.embedding_size(:facing) == 1
    end
  end
end
