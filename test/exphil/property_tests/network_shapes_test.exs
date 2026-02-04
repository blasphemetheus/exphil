defmodule ExPhil.PropertyTests.NetworkShapesTest do
  @moduledoc """
  Property-based tests for neural network shape invariants.

  These tests verify that network outputs have expected shapes
  regardless of batch size or sequence length variations.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExPhil.Networks.{Policy, Mamba}
  alias ExPhil.Networks.Mamba.Common, as: MambaCommon

  @moduletag :property
  @moduletag :slow

  describe "Policy network shapes" do
    property "policy output shape matches expected heads" do
      check all(
              batch_size <- StreamData.integer(1..8),
              embed_size <- StreamData.member_of([128, 256]),
              max_runs: 5
            ) do
        # Build a simple policy network
        opts = [
          embed_size: embed_size,
          hidden_sizes: [128],
          temporal: false
        ]

        model = Policy.build(opts)
        {init_fn, predict_fn} = Axon.build(model)

        # Initialize with dummy input
        input = Nx.broadcast(0.0, {batch_size, embed_size})
        params = init_fn.(input, %{})

        # Run forward pass
        output = predict_fn.(params, input)

        # Output is a tuple: {buttons, main_x, main_y, c_x, c_y, shoulder}
        assert is_tuple(output)
        assert tuple_size(output) == 6

        {buttons, main_x, _main_y, _c_x, _c_y, shoulder} = output

        # Each head should have correct batch dimension
        {buttons_batch, buttons_dim} = Nx.shape(buttons)
        assert buttons_batch == batch_size
        assert buttons_dim == 8  # 8 buttons

        {main_x_batch, main_x_dim} = Nx.shape(main_x)
        assert main_x_batch == batch_size
        assert main_x_dim == 17  # 16 + 1 axis buckets

        {shoulder_batch, shoulder_dim} = Nx.shape(shoulder)
        assert shoulder_batch == batch_size
        assert shoulder_dim == 5  # 4 + 1 shoulder buckets
      end
    end
  end

  describe "Mamba network shapes" do
    @tag :slow
    property "Mamba output shape is [batch, hidden_size]" do
      check all(
              batch_size <- StreamData.integer(1..4),
              seq_len <- StreamData.member_of([30, 60]),
              hidden_size <- StreamData.member_of([64, 128]),
              max_runs: 4
            ) do
        embed_size = 64

        opts = [
          embed_size: embed_size,
          hidden_size: hidden_size,
          num_layers: 1,
          seq_len: seq_len
        ]

        model = Mamba.build(opts)
        {init_fn, predict_fn} = Axon.build(model)

        # Initialize with dummy input
        input = Nx.broadcast(0.0, {batch_size, seq_len, embed_size})
        params = init_fn.(input, %{})

        # Run forward pass
        output = predict_fn.(params, input)

        # Should be [batch, hidden_size]
        assert Nx.shape(output) == {batch_size, hidden_size}
      end
    end

    @tag :slow
    property "Mamba output is finite for random inputs" do
      check all(
              batch_size <- StreamData.integer(1..2),
              seq_len <- StreamData.member_of([30]),
              max_runs: 3
            ) do
        embed_size = 64
        hidden_size = 64

        opts = [
          embed_size: embed_size,
          hidden_size: hidden_size,
          num_layers: 1,
          seq_len: seq_len
        ]

        model = Mamba.build(opts)
        {init_fn, predict_fn} = Axon.build(model)

        # Random input in reasonable range
        key = Nx.Random.key(42)
        {input, _key} = Nx.Random.normal(key, shape: {batch_size, seq_len, embed_size})
        input = Nx.multiply(input, 0.1)  # Scale down to avoid instability

        params = init_fn.(input, %{})
        output = predict_fn.(params, input)

        # Should be finite (no NaN or Inf)
        assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end
  end

  describe "Mamba Common utilities" do
    property "param_count scales with num_layers" do
      check all(
              num_layers <- StreamData.integer(1..4),
              max_runs: 10
            ) do
        opts1 = [embed_size: 128, hidden_size: 128, num_layers: num_layers]
        opts2 = [embed_size: 128, hidden_size: 128, num_layers: num_layers + 1]

        count1 = MambaCommon.param_count(opts1)
        count2 = MambaCommon.param_count(opts2)

        # More layers = more parameters
        assert count2 > count1
      end
    end

    property "param_count scales with hidden_size" do
      check all(
              hidden_size <- StreamData.member_of([64, 128, 256]),
              max_runs: 10
            ) do
        opts1 = [embed_size: 128, hidden_size: hidden_size, num_layers: 1]
        opts2 = [embed_size: 128, hidden_size: hidden_size * 2, num_layers: 1]

        count1 = MambaCommon.param_count(opts1)
        count2 = MambaCommon.param_count(opts2)

        # Larger hidden size = more parameters
        assert count2 > count1
      end
    end

    property "output_size equals hidden_size" do
      check all(
              hidden_size <- StreamData.integer(32..512),
              max_runs: 20
            ) do
        opts = [hidden_size: hidden_size]
        assert MambaCommon.output_size(opts) == hidden_size
      end
    end

    property "melee_defaults returns valid config" do
      # This is deterministic, just verify it once
      defaults = MambaCommon.melee_defaults()

      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :window_size)

      assert defaults[:hidden_size] > 0
      assert defaults[:state_size] > 0
      assert defaults[:num_layers] > 0
      assert defaults[:window_size] > 0
    end
  end

  describe "sequential vs parallel scan equivalence" do
    property "sequential_scan and blelloch_scan produce same result for short sequences" do
      check all(
              batch_size <- StreamData.integer(1..4),
              seq_len <- StreamData.integer(4..16),
              hidden_size <- StreamData.member_of([4, 8]),
              state_size <- StreamData.member_of([2, 4]),
              max_runs: 20
            ) do
        # Create random a and b tensors
        key = Nx.Random.key(System.unique_integer([:positive]))
        shape = {batch_size, seq_len, hidden_size, state_size}

        # a should be in (0, 1) for stability (decay factors)
        {a_raw, key} = Nx.Random.uniform(key, shape: shape)
        a = Nx.multiply(a_raw, 0.9) |> Nx.add(0.05)

        # b can be any reasonable values
        {b, _key} = Nx.Random.normal(key, shape: shape)
        b = Nx.multiply(b, 0.1)

        # Run both scans
        seq_result = MambaCommon.sequential_scan(a, b)
        blelloch_result = MambaCommon.blelloch_scan(a, b)

        # Should be very close (floating point tolerance)
        diff = Nx.subtract(seq_result, blelloch_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        # Allow some tolerance for floating point differences
        assert diff < 1.0e-4, "Max diff was #{diff}"
      end
    end
  end
end
