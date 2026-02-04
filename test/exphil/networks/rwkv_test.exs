defmodule ExPhil.Networks.RWKVTest do
  @moduledoc """
  Tests for RWKV-7 "Goose" backbone.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.RWKV

  @embed_size 64
  @hidden_size 32
  @head_size 16
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        RWKV.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          head_size: @head_size,
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
        RWKV.build(
          embed_size: 128,
          hidden_size: 64,
          head_size: 32,
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
        RWKV.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          head_size: @head_size,
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

    test "handles different num_layers" do
      for num_layers <- [1, 2, 4] do
        model =
          RWKV.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            head_size: @head_size,
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

    test "handles different head_size values" do
      for head_size <- [8, 16, 32] do
        # hidden_size must be divisible by head_size
        hidden_size = 64

        model =
          RWKV.build(
            embed_size: @embed_size,
            hidden_size: hidden_size,
            head_size: head_size,
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

        assert Nx.shape(output) == {@batch_size, hidden_size},
               "Failed for head_size=#{head_size}"
      end
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert RWKV.output_size(hidden_size: 128) == 128
      assert RWKV.output_size(hidden_size: 256) == 256
      # default
      assert RWKV.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        RWKV.param_count(
          embed_size: 287,
          hidden_size: 256,
          num_layers: 6
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      count_2 =
        RWKV.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2
        )

      count_4 =
        RWKV.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4
        )

      # 4 layers should have roughly 2x params of 2 layers
      assert count_4 > count_2 * 1.8
      assert count_4 < count_2 * 2.2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = RWKV.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :head_size) == 64
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = RWKV.melee_defaults()

      model = RWKV.build([embed_size: 287] ++ defaults)

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
        RWKV.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          head_size: @head_size,
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
        RWKV.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          head_size: @head_size,
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
  end

  describe "init_cache/1" do
    test "initializes cache with correct shapes" do
      cache = RWKV.init_cache(
        batch_size: 4,
        hidden_size: 64,
        head_size: 16,
        num_layers: 2
      )

      assert cache.step == 0
      assert cache.config.hidden_size == 64
      assert cache.config.num_heads == 4  # 64 / 16
      assert cache.config.num_layers == 2

      # Check layer caches
      layer_1 = cache.layers["layer_1"]
      assert Nx.shape(layer_1.wkv_numerator) == {4, 4, 16}
      assert Nx.shape(layer_1.wkv_denominator) == {4, 4, 16}
      assert Nx.shape(layer_1.last_token_time) == {4, 64}
      assert Nx.shape(layer_1.last_token_channel) == {4, 64}
    end
  end

  describe "integration with Policy" do
    test "works as rwkv temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :rwkv,
          hidden_size: @hidden_size,
          head_size: @head_size,
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
