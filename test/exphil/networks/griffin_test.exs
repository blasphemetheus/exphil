defmodule ExPhil.Networks.GriffinTest do
  @moduledoc """
  Tests for the Griffin/Hawk implementation with RG-LRU + Local Attention.

  This tests the Griffin architecture from "Griffin: Mixing Gated Linear Recurrences
  with Local Attention for Efficient Language Models" (De et al., 2024).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Griffin

  @embed_size 64
  @hidden_size 32
  @seq_len 12
  @batch_size 4

  describe "build/1 (Griffin - RG-LRU + Local Attention)" do
    test "builds model with correct output shape" do
      model =
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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
        Griffin.build(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 6,
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
        Griffin.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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
      for num_layers <- [3, 6, 9] do
        model =
          Griffin.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
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

    test "respects expand_factor option" do
      model =
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          expand_factor: 4,
          num_layers: 6,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "respects local_attn_window option" do
      model =
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          local_attn_window: 8,
          num_layers: 6,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_hawk/1 (Pure RG-LRU)" do
    test "builds model with correct output shape" do
      model =
        Griffin.build_hawk(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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

    test "works with different layer counts" do
      for num_layers <- [2, 4, 6] do
        model =
          Griffin.build_hawk(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
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
  end

  describe "build_griffin_block/2" do
    test "builds RG-LRU block with correct shape preservation" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        Griffin.build_griffin_block(input,
          hidden_size: @hidden_size,
          expand_factor: 3,
          block_type: :rg_lru,
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

      # Block output matches input shape (for residual connections)
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end

    test "builds local attention block with correct shape preservation" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block =
        Griffin.build_griffin_block(input,
          hidden_size: @hidden_size,
          expand_factor: 3,
          block_type: :local_attention,
          num_heads: 4,
          local_attn_window: 8,
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

      # Block output matches input shape
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Griffin.output_size(hidden_size: 128) == 128
      assert Griffin.output_size(hidden_size: 256) == 256
      # default
      assert Griffin.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        Griffin.param_count(
          embed_size: 287,
          hidden_size: 256,
          expand_factor: 3,
          num_layers: 6
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      count_3 =
        Griffin.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 3
        )

      count_6 =
        Griffin.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 6
        )

      # 6 layers should have roughly 2x params of 3 layers (plus small fixed overhead)
      assert count_6 > count_3 * 1.8
      assert count_6 < count_3 * 2.2
    end

    test "hawk has fewer params than griffin (no attention layers)" do
      griffin_count =
        Griffin.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 6,
          use_local_attention: true
        )

      hawk_count =
        Griffin.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 6,
          use_local_attention: false
        )

      # Hawk should have fewer params (no attention layers)
      assert hawk_count < griffin_count
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = Griffin.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :expand_factor) == 3
      assert Keyword.get(defaults, :local_attn_window) == 32
      assert Keyword.get(defaults, :num_heads) == 4
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = Griffin.melee_defaults()

      model = Griffin.build([embed_size: 287] ++ defaults)

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
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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
        Griffin.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 6,
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

  describe "RG-LRU recurrence behavior" do
    test "RG-LRU scan produces correct output shape" do
      model =
        Griffin.build_hawk(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "init_lambda/1" do
    test "produces values in expected range" do
      lambda = Griffin.init_lambda({32})

      # Lambda values should be finite
      assert Nx.all(Nx.is_nan(lambda) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(lambda) |> Nx.logical_not()) |> Nx.to_number() == 1

      # When converted to a = sigmoid(lambda), a^c should be in [0.9, 0.999]
      a = Nx.sigmoid(lambda)
      c = Griffin.rg_lru_c()
      a_c = Nx.pow(a, c)

      # Check bounds (with some tolerance for edge cases)
      min_val = Nx.reduce_min(a_c) |> Nx.to_number()
      max_val = Nx.reduce_max(a_c) |> Nx.to_number()

      assert min_val >= 0.85, "min a^c should be >= 0.85, got #{min_val}"
      assert max_val <= 1.0, "max a^c should be <= 1.0, got #{max_val}"
    end
  end

  describe "integration with Policy" do
    test "works as griffin temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :griffin,
          hidden_size: @hidden_size,
          num_layers: 6,
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

    test "works as hawk temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :hawk,
          hidden_size: @hidden_size,
          num_layers: 6,
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
