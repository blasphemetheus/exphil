defmodule ExPhil.Networks.MambaTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Mamba

  @embed_size 64
  @hidden_size 32
  @state_size 8
  @seq_len 10
  @batch_size 4

  describe "build/1" do
    test "builds Mamba model with correct output shape" do
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "builds stacked Mamba layers" do
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      # When embed_size != hidden_size, an input projection is added
      model = Mamba.build(
        embed_size: 128,
        hidden_size: 64,
        state_size: @state_size,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "respects expand_factor option" do
      # expand_factor controls the inner dimension of the Mamba block
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        expand_factor: 4,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "respects conv_size option" do
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        conv_size: 8,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_mamba_block/2" do
    test "builds a single Mamba block with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      block = Mamba.build_mamba_block(input,
        hidden_size: @hidden_size,
        state_size: @state_size,
        expand_factor: 2,
        conv_size: 4,
        name: "test_block"
      )

      {init_fn, predict_fn} = Axon.build(block)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block preserves shape (for residual connections)
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "build_causal_conv1d/4" do
    test "maintains sequence length after causal convolution" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      conv = Mamba.build_causal_conv1d(input, @hidden_size, 4, "test_conv")

      {init_fn, predict_fn} = Axon.build(conv)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Causal conv should preserve sequence length
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end

    test "works with different kernel sizes" do
      for kernel_size <- [2, 4, 8] do
        input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

        conv = Mamba.build_causal_conv1d(input, @hidden_size, kernel_size, "conv_k#{kernel_size}")

        {init_fn, predict_fn} = Axon.build(conv)
        params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

        input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
        output = predict_fn.(params, input_data)

        assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
      end
    end
  end

  describe "build_selective_ssm/2" do
    test "produces correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      ssm = Mamba.build_selective_ssm(input,
        hidden_size: @hidden_size,
        state_size: @state_size,
        name: "test_ssm"
      )

      {init_fn, predict_fn} = Axon.build(ssm)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # SSM preserves sequence shape
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Mamba.output_size(hidden_size: 128) == 128
      assert Mamba.output_size(hidden_size: 256) == 256
      assert Mamba.output_size() == 256  # default
    end
  end

  describe "param_count/1" do
    test "returns approximate parameter count" do
      count = Mamba.param_count(
        embed_size: 1991,
        hidden_size: 256,
        state_size: 16,
        expand_factor: 2,
        num_layers: 2
      )

      # Should be a reasonable number (millions of params for production model)
      assert count > 1_000_000
      assert count < 100_000_000
    end

    test "scales with num_layers" do
      count_1 = Mamba.param_count(
        embed_size: 64,
        hidden_size: 32,
        state_size: 8,
        expand_factor: 2,
        num_layers: 1
      )

      count_2 = Mamba.param_count(
        embed_size: 64,
        hidden_size: 32,
        state_size: 8,
        expand_factor: 2,
        num_layers: 2
      )

      # 2 layers should have roughly 2x params (plus input projection)
      assert count_2 > count_1
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

    test "builds a valid model with defaults" do
      defaults = Mamba.melee_defaults()

      model = Mamba.build([embed_size: 1991] ++ defaults)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 60, 1991}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 60, 1991})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 256}
    end
  end

  describe "concrete seq_len for JIT optimization" do
    test "uses window_size as concrete seq_len by default" do
      window_size = 20

      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        window_size: window_size
        # NOTE: Not passing seq_len - should default to window_size
      )

      # Model should build and compile (would be slow with dynamic shapes)
      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        Nx.template({@batch_size, window_size, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, window_size, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "numerical behavior" do
    test "produces finite outputs for normal inputs" do
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 2,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Normal input range
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      model = Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Should still produce valid output
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "integration with Policy" do
    test "works as temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model = Policy.build_temporal(
        embed_size: @embed_size,
        backbone: :mamba,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

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
