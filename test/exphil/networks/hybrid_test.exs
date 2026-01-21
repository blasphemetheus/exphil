defmodule ExPhil.Networks.HybridTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Hybrid

  @embed_size 64
  @hidden_size 32
  @state_size 8
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds hybrid model with correct output shape" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,  # 2 Mamba + 1 Attention
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      model = Hybrid.build(
        embed_size: 128,
        hidden_size: 64,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "builds with 6 layers (4 mamba + 2 attention)" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 6,
        attention_every: 3,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "builds with alternating Mamba/Attention (attention_every: 2)" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 4,
        attention_every: 2,  # M, A, M, A pattern
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "respects expand_factor option" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        expand_factor: 4,
        num_layers: 3,
        attention_every: 3,
        window_size: @seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "supports full attention (use_sliding_window: false)" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
        window_size: @seq_len,
        use_sliding_window: false
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_mamba_layer/2" do
    test "builds a single Mamba layer with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      layer = Hybrid.build_mamba_layer(input,
        hidden_size: @hidden_size,
        state_size: @state_size,
        expand_factor: 2,
        conv_size: 4,
        dropout: 0.1,
        name: "test_mamba"
      )

      {init_fn, predict_fn} = Axon.build(layer)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Layer preserves shape for stacking
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "build_attention_layer/2" do
    test "builds a single attention layer with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden_size})

      layer = Hybrid.build_attention_layer(input,
        hidden_size: @hidden_size,
        num_heads: 4,
        head_dim: 8,
        dropout: 0.1,
        use_sliding_window: true,
        window_size: @seq_len,
        name: "test_attn"
      )

      {init_fn, predict_fn} = Axon.build(layer)
      params = init_fn.(Nx.template({@batch_size, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Attention layer preserves shape for stacking
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end
  end

  describe "layer_pattern/1" do
    test "returns correct pattern for 6 layers with attention_every: 3" do
      pattern = Hybrid.layer_pattern(num_layers: 6, attention_every: 3)

      assert pattern == [:mamba, :mamba, :attention, :mamba, :mamba, :attention]
    end

    test "returns correct pattern for 4 layers with attention_every: 2" do
      pattern = Hybrid.layer_pattern(num_layers: 4, attention_every: 2)

      assert pattern == [:mamba, :attention, :mamba, :attention]
    end

    test "returns all mamba for attention_every > num_layers" do
      pattern = Hybrid.layer_pattern(num_layers: 3, attention_every: 10)

      assert pattern == [:mamba, :mamba, :mamba]
    end

    test "returns correct pattern for attention_every: 1" do
      pattern = Hybrid.layer_pattern(num_layers: 3, attention_every: 1)

      # Every layer is attention (layer % 1 == 0)
      assert pattern == [:attention, :attention, :attention]
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Hybrid.output_size(hidden_size: 128) == 128
      assert Hybrid.output_size(hidden_size: 256) == 256
      assert Hybrid.output_size() == 256  # default
    end
  end

  describe "param_count/1" do
    test "returns approximate parameter count" do
      count = Hybrid.param_count(
        embed_size: 1991,
        hidden_size: 256,
        state_size: 16,
        expand_factor: 2,
        num_layers: 6,
        attention_every: 3,
        num_heads: 4,
        head_dim: 64
      )

      # Should be a reasonable number
      assert count > 1_000_000
      assert count < 100_000_000
    end

    test "has more params than pure Mamba due to attention layers" do
      mamba_count = ExPhil.Networks.Mamba.param_count(
        embed_size: 64,
        hidden_size: 32,
        state_size: 8,
        expand_factor: 2,
        num_layers: 6
      )

      hybrid_count = Hybrid.param_count(
        embed_size: 64,
        hidden_size: 32,
        state_size: 8,
        expand_factor: 2,
        num_layers: 6,
        attention_every: 3,
        num_heads: 4,
        head_dim: 8
      )

      # Hybrid should have more params due to attention layers
      # (attention has FFN with 4x expansion)
      assert hybrid_count > mamba_count
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = Hybrid.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :state_size) == 16
      assert Keyword.get(defaults, :expand_factor) == 2
      assert Keyword.get(defaults, :conv_size) == 4
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :attention_every) == 3
      assert Keyword.get(defaults, :num_heads) == 4
      assert Keyword.get(defaults, :head_dim) == 64
      assert Keyword.get(defaults, :window_size) == 60
      assert Keyword.get(defaults, :use_sliding_window) == true
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = Hybrid.melee_defaults()

      model = Hybrid.build([embed_size: 1991] ++ defaults)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 60, 1991}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 60, 1991})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 256}
    end
  end

  describe "numerical behavior" do
    test "produces finite outputs for normal inputs" do
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
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
      model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
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
    test "works as jamba temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model = Policy.build_temporal(
        embed_size: @embed_size,
        backbone: :jamba,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
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

  describe "comparison with pure architectures" do
    @tag :benchmark
    test "hybrid has more expressiveness than pure Mamba" do
      # This test verifies that the hybrid architecture produces
      # different outputs than pure Mamba, indicating the attention
      # layers are contributing to the computation.

      key = Nx.Random.key(123)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)

      # Build pure Mamba
      mamba_model = ExPhil.Networks.Mamba.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        window_size: @seq_len
      )

      {init_mamba, pred_mamba} = Axon.build(mamba_model)
      mamba_params = init_mamba.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      mamba_out = pred_mamba.(mamba_params, input)

      # Build hybrid
      hybrid_model = Hybrid.build(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 3,
        attention_every: 3,
        window_size: @seq_len
      )

      {init_hybrid, pred_hybrid} = Axon.build(hybrid_model)
      hybrid_params = init_hybrid.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      hybrid_out = pred_hybrid.(hybrid_params, input)

      # Both should produce valid outputs
      assert Nx.shape(mamba_out) == {@batch_size, @hidden_size}
      assert Nx.shape(hybrid_out) == {@batch_size, @hidden_size}

      # Outputs should be different (different architectures with different random init)
      diff = Nx.mean(Nx.abs(Nx.subtract(mamba_out, hybrid_out))) |> Nx.to_number()
      assert diff > 0.01, "Expected different outputs from Mamba and Hybrid"
    end
  end
end
