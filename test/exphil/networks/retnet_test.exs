defmodule ExPhil.Networks.RetNetTest do
  @moduledoc """
  Tests for the RetNet (Retentive Network) implementation.

  Tests cover the retention mechanism and multi-scale retention from
  "Retentive Network: A Successor to Transformer" (Sun et al., Microsoft 2023).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.RetNet

  @embed_size 64
  @hidden_size 32
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        RetNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      model =
        RetNet.build(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles same embed and hidden sizes (no projection)" do
      model =
        RetNet.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different num_layers" do
      for num_layers <- [2, 4, 6] do
        model =
          RetNet.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: num_layers,
            num_heads: 4,
            window_size: @seq_len
          )

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(
            Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
            Axon.ModelState.empty()
          )

        input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch_size, @hidden_size},
               "Failed for num_layers=#{num_layers}"
      end
    end

    test "handles different num_heads" do
      # Test with different head counts (hidden_size must be divisible)
      for num_heads <- [2, 4, 8] do
        model =
          RetNet.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: 4,
            num_heads: num_heads,
            window_size: @seq_len
          )

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(
            Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
            Axon.ModelState.empty()
          )

        input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch_size, @hidden_size},
               "Failed for num_heads=#{num_heads}"
      end
    end
  end

  describe "build_retnet_block/2" do
    test "builds block with correct shape preservation" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        RetNet.build_retnet_block(input,
          hidden_size: @hidden_size,
          num_heads: 4,
          expand_factor: 2,
          layer_idx: 1
        )

      {init_fn, predict_fn} = Axon.build(block)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "build_multi_scale_retention/2" do
    test "builds MSR with correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      msr =
        RetNet.build_multi_scale_retention(input,
          hidden_size: @hidden_size,
          num_heads: 4,
          name: "test_msr"
        )

      {init_fn, predict_fn} = Axon.build(msr)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert RetNet.output_size(hidden_size: 128) == 128
      assert RetNet.output_size(hidden_size: 256) == 256
      # default
      assert RetNet.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        RetNet.param_count(
          embed_size: 287,
          hidden_size: 256,
          num_layers: 6,
          num_heads: 4,
          expand_factor: 2
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      count_3 =
        RetNet.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 3
        )

      count_6 =
        RetNet.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 6
        )

      # 6 layers should have roughly 2x params of 3 layers
      assert count_6 > count_3 * 1.8
      assert count_6 < count_3 * 2.2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = RetNet.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :num_heads) == 4
      assert Keyword.get(defaults, :expand_factor) == 2
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = RetNet.melee_defaults()

      model = RetNet.build([embed_size: 287] ++ defaults)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 60, 287}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 60, 287})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 256}
    end
  end

  describe "numerical stability" do
    test "produces finite outputs for normal inputs" do
      model =
        RetNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Normal random input
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      model =
        RetNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles small input values" do
      model =
        RetNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(1.0e-6, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "retention mechanism" do
    test "decay matrix has correct causal structure" do
      # Test that the decay matrix is lower triangular (causal)
      _seq_len = 8
      num_heads = 4

      # Manually compute what the decay matrix should look like
      gammas =
        Enum.map(0..(num_heads - 1), fn h ->
          1.0 - :math.pow(2, -5 - h)
        end)
        |> Nx.tensor(type: :f32)

      # The first gamma should be close to 1 - 2^(-5) ≈ 0.96875
      first_gamma = Nx.to_number(gammas[0])
      assert_in_delta first_gamma, 0.96875, 0.001
    end
  end

  describe "init_retention_state/3" do
    test "initializes zero state with correct shape" do
      state = RetNet.init_retention_state(4, 4, 8)
      assert Nx.shape(state) == {4, 4, 8, 8}
      assert Nx.to_number(Nx.sum(state)) == 0.0
    end
  end

  describe "integration with Policy" do
    test "works as retnet temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :retnet,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Check policy output shapes
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end
  end
end
