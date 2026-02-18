defmodule ExPhil.Networks.XLSTMTest do
  @moduledoc """
  Tests for the xLSTM (Extended LSTM) implementation.

  Tests cover sLSTM (scalar), mLSTM (matrix), and mixed variants
  from "xLSTM: Extended Long Short-Term Memory" (Beck et al., NeurIPS 2024).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.XLSTM

  @embed_size 64
  @hidden_size 32
  @seq_len 12
  @batch_size 4

  describe "build/1 with :mixed variant (default)" do
    test "builds model with correct output shape" do
      model =
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :mixed,
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
        XLSTM.build(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 4,
          variant: :mixed,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles different num_layers" do
      for num_layers <- [2, 4, 6] do
        model =
          XLSTM.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: num_layers,
            variant: :mixed,
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
  end

  describe "build/1 with :slstm variant" do
    test "builds sLSTM-only model" do
      model =
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :slstm,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "sLSTM handles exponential gating without overflow" do
      model =
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          variant: :slstm,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Use larger input values to stress exponential gating
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, -2.0, 2.0, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with :mlstm variant" do
    test "builds mLSTM-only model" do
      model =
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :mlstm,
          num_heads: 4,
          head_dim: 8,  # 4 * 8 = 32 = hidden_size
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "mLSTM handles different head configurations" do
      # Test different num_heads * head_dim combinations
      configs = [
        {2, 16},  # 32
        {4, 8},   # 32
        {8, 4}    # 32
      ]

      for {num_heads, head_dim} <- configs do
        model =
          XLSTM.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: 2,
            variant: :mlstm,
            num_heads: num_heads,
            head_dim: head_dim,
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
               "Failed for num_heads=#{num_heads}, head_dim=#{head_dim}"
      end
    end
  end

  describe "build_xlstm_block/2" do
    test "builds sLSTM block with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        XLSTM.build_xlstm_block(input,
          hidden_size: @hidden_size,
          expand_factor: 2,
          block_type: :slstm,
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

    test "builds mLSTM block with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        XLSTM.build_xlstm_block(input,
          hidden_size: @hidden_size,
          expand_factor: 2,
          block_type: :mlstm,
          num_heads: 4,
          head_dim: 8,
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

  describe "output_size/1" do
    test "returns hidden_size" do
      assert XLSTM.output_size(hidden_size: 128) == 128
      assert XLSTM.output_size(hidden_size: 256) == 256
      # default
      assert XLSTM.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        XLSTM.param_count(
          embed_size: 287,
          hidden_size: 256,
          num_layers: 4,
          num_heads: 4,
          head_dim: 64,
          expand_factor: 2,
          variant: :mixed
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "slstm has fewer params than mlstm" do
      slstm_count =
        XLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4,
          variant: :slstm
        )

      mlstm_count =
        XLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4,
          num_heads: 4,
          head_dim: 8,
          variant: :mlstm
        )

      # mLSTM has extra K, V, Q projections
      assert mlstm_count > slstm_count
    end

    test "scales with num_layers" do
      count_2 =
        XLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2,
          variant: :slstm
        )

      count_4 =
        XLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4,
          variant: :slstm
        )

      # 4 layers should have roughly 2x params of 2 layers
      assert count_4 > count_2 * 1.8
      assert count_4 < count_2 * 2.2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = XLSTM.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :num_heads) == 4
      assert Keyword.get(defaults, :head_dim) == 64
      assert Keyword.get(defaults, :expand_factor) == 2
      assert Keyword.get(defaults, :variant) == :mixed
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = XLSTM.melee_defaults()

      model = XLSTM.build([embed_size: 287] ++ defaults)

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
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :mixed,
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
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :mixed,
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
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 4,
          variant: :mixed,
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

  describe "integration with Policy" do
    test "works as xlstm temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :xlstm,
          hidden_size: @hidden_size,
          num_layers: 4,
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

    test "works as xlstm_slstm temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :xlstm_slstm,
          hidden_size: @hidden_size,
          num_layers: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, _main_x, _main_y, _c_x, _c_y, _shoulder} = predict_fn.(params, input)

      assert Nx.shape(buttons) == {@batch_size, 8}
    end

    test "works as xlstm_mlstm temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :xlstm_mlstm,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: 4,
          head_dim: 8,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, _main_x, _main_y, _c_x, _c_y, _shoulder} = predict_fn.(params, input)

      assert Nx.shape(buttons) == {@batch_size, 8}
    end
  end
end
