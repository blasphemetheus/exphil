defmodule ExPhil.Networks.KANTest do
  @moduledoc """
  Tests for the KAN (Kolmogorov-Arnold Networks) implementation.

  Tests cover the learnable activation functions and multi-layer KAN
  from "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.KAN

  @embed_size 64
  @hidden_size 32
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        KAN.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          grid_size: 8
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
        KAN.build(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 2,
          grid_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles same embed and hidden sizes (no projection)" do
      model =
        KAN.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          grid_size: 4
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
          KAN.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: num_layers,
            grid_size: 4
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

    test "handles different grid sizes" do
      for grid_size <- [4, 8, 16] do
        model =
          KAN.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: 2,
            grid_size: grid_size
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
               "Failed for grid_size=#{grid_size}"
      end
    end
  end

  describe "build_kan_block/2" do
    test "builds block with correct shape preservation" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        KAN.build_kan_block(input,
          hidden_size: @hidden_size,
          grid_size: 4,
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

  describe "basis functions" do
    test "sine_basis produces valid output" do
      x = Nx.broadcast(0.5, {2, 4, 8})
      frequencies = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      phases = Nx.tensor([0.0, 0.5, 1.0, 1.5])

      result = KAN.sine_basis(x, frequencies, phases)

      # Result should be sin(freq * x + phase) for each frequency
      # Shape: [batch, seq, in_size, grid_size]
      assert Nx.shape(result) == {2, 4, 8, 4}

      # Values should be in [-1, 1]
      assert Nx.to_number(Nx.reduce_min(result)) >= -1.0
      assert Nx.to_number(Nx.reduce_max(result)) <= 1.0
    end

    test "chebyshev_basis produces valid output" do
      x = Nx.broadcast(0.5, {2, 4, 8})

      result = KAN.chebyshev_basis(x, 5)

      # Result should have 5 basis functions
      assert Nx.shape(result) == {2, 4, 8, 5}

      # T_0(x) = 1 for all x
      t0 = Nx.slice_along_axis(result, 0, 1, axis: 3)
      assert Nx.to_number(Nx.mean(t0)) == 1.0

      # T_1(x) = x
      t1 = Nx.slice_along_axis(result, 1, 1, axis: 3)
      assert_in_delta Nx.to_number(Nx.mean(t1)), 0.5, 0.001
    end

    test "rbf_basis produces valid output" do
      x = Nx.broadcast(0.5, {2, 4, 8})
      centers = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
      sigma = 0.5

      result = KAN.rbf_basis(x, centers, sigma)

      # Result should have 5 basis functions (one per center)
      assert Nx.shape(result) == {2, 4, 8, 5}

      # All values should be positive (RBF is always positive)
      assert Nx.to_number(Nx.reduce_min(result)) >= 0.0

      # Values should be <= 1 (max at center)
      assert Nx.to_number(Nx.reduce_max(result)) <= 1.0 + 1.0e-6
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert KAN.output_size(hidden_size: 128) == 128
      assert KAN.output_size(hidden_size: 256) == 256
      # default
      assert KAN.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        KAN.param_count(
          embed_size: 287,
          hidden_size: 256,
          num_layers: 4,
          grid_size: 8
        )

      # Should have significant params for this config
      assert count > 500_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with grid_size" do
      count_4 =
        KAN.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2,
          grid_size: 4
        )

      count_8 =
        KAN.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2,
          grid_size: 8
        )

      # Larger grid should have more params
      assert count_8 > count_4
    end

    test "scales with num_layers" do
      count_2 =
        KAN.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2,
          grid_size: 4
        )

      count_4 =
        KAN.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4,
          grid_size: 4
        )

      # 4 layers should have ~2x params of 2 layers
      assert count_4 > count_2 * 1.8
      assert count_4 < count_2 * 2.2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = KAN.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :grid_size) == 8
      assert Keyword.get(defaults, :basis) == :sine
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = KAN.melee_defaults()

      model = KAN.build([embed_size: 287] ++ defaults)

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
        KAN.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          grid_size: 4
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
        KAN.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          grid_size: 4
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
        KAN.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          grid_size: 4
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

  describe "integration with Policy" do
    test "works as kan temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :kan,
          hidden_size: @hidden_size,
          num_layers: 2,
          grid_size: 4,
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
