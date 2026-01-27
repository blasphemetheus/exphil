defmodule ExPhil.Networks.MambaTest do
  @moduledoc """
  Tests for the true Mamba implementation with parallel associative scan.

  This tests the Mamba architecture from "Mamba: Linear-Time Sequence Modeling
  with Selective State Spaces" (Gu & Dao, 2023).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Mamba

  @embed_size 64
  @hidden_size 32
  @state_size 8
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
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
        Mamba.build(
          embed_size: 128,
          hidden_size: 64,
          state_size: @state_size,
          num_layers: 2,
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
        Mamba.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
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

    test "respects expand_factor option" do
      # Should still work with different expand factors
      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          expand_factor: 4,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "respects conv_size option" do
      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          conv_size: 8,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different num_layers" do
      for num_layers <- [1, 3, 4] do
        model =
          Mamba.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            state_size: @state_size,
            num_layers: num_layers,
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

    test "handles different state_sizes" do
      for state_size <- [4, 16, 32] do
        model =
          Mamba.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            state_size: state_size,
            num_layers: 2,
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
               "Failed for state_size=#{state_size}"
      end
    end
  end

  describe "build_mamba_block/2" do
    test "builds a single block with correct shape preservation" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        Mamba.build_mamba_block(input,
          hidden_size: @hidden_size,
          state_size: @state_size,
          expand_factor: 2,
          conv_size: 4,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(block)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block output matches input shape (for residual connections)
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Mamba.output_size(hidden_size: 128) == 128
      assert Mamba.output_size(hidden_size: 256) == 256
      # default
      assert Mamba.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        Mamba.param_count(
          embed_size: 287,
          hidden_size: 256,
          state_size: 16,
          expand_factor: 2,
          num_layers: 2
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 10_000_000
    end

    test "scales with num_layers" do
      count_2 =
        Mamba.param_count(
          embed_size: 64,
          hidden_size: 32,
          state_size: 8,
          num_layers: 2
        )

      count_4 =
        Mamba.param_count(
          embed_size: 64,
          hidden_size: 32,
          state_size: 8,
          num_layers: 4
        )

      # 4 layers should have roughly 2x params of 2 layers (plus small fixed overhead)
      assert count_4 > count_2 * 1.8
      assert count_4 < count_2 * 2.2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = Mamba.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :state_size) == 16
      assert Keyword.get(defaults, :expand_factor) == 2
      assert Keyword.get(defaults, :conv_size) == 4
      assert Keyword.get(defaults, :num_layers) == 2
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = Mamba.melee_defaults()

      model = Mamba.build([embed_size: 287] ++ defaults)

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
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
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
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Should still produce valid output
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles small input values" do
      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(1.0e-6, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Should produce finite output
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "parallel scan behavior" do
    test "longer sequences use blelloch scan path" do
      # Sequence > 32 should use parallel scan
      long_seq = 64

      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: long_seq
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, long_seq, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, long_seq, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "short sequences use sequential scan path" do
      # Sequence <= 32 should use sequential scan
      short_seq = 16

      model =
        Mamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: short_seq
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, short_seq, @embed_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, short_seq, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "integration with Policy" do
    test "works as mamba temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :mamba,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
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
