defmodule ExPhil.Networks.AttentionTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Attention

  @embed_size 64
  @seq_len 10
  @batch_size 2
  @num_heads 2
  @head_dim 16

  describe "scaled_dot_product_attention/4" do
    test "computes attention with correct output shape" do
      query = Nx.broadcast(0.5, {@batch_size, @seq_len, 32})
      key = Nx.broadcast(0.5, {@batch_size, @seq_len, 32})
      value = Nx.broadcast(0.5, {@batch_size, @seq_len, 32})

      output = Attention.scaled_dot_product_attention(query, key, value)

      assert Nx.shape(output) == {@batch_size, @seq_len, 32}
    end

    test "works with causal mask" do
      query = Nx.broadcast(0.5, {@batch_size, @seq_len, 32})
      key = Nx.broadcast(0.5, {@batch_size, @seq_len, 32})
      value = Nx.iota({@batch_size, @seq_len, 32}, axis: 1) |> Nx.as_type(:f32)

      mask = Attention.causal_mask(@seq_len)
      output = Attention.scaled_dot_product_attention(query, key, value, mask: mask)

      assert Nx.shape(output) == {@batch_size, @seq_len, 32}
    end
  end

  describe "causal_mask/1" do
    test "creates lower triangular mask" do
      mask = Attention.causal_mask(4)

      # Should be lower triangular
      expected = Nx.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
      ], type: :u8)

      assert Nx.to_list(mask) == Nx.to_list(expected)
    end
  end

  describe "window_mask/2" do
    test "creates sliding window mask" do
      mask = Attention.window_mask(5, 2)

      # Window of 2 means each position sees itself and 1 previous
      # Position 0: sees [0]
      # Position 1: sees [0, 1]
      # Position 2: sees [1, 2]
      # etc.
      assert Nx.shape(mask) == {5, 5}

      # First position can only see itself
      row0 = Nx.slice(mask, [0, 0], [1, 5]) |> Nx.squeeze()
      assert Nx.to_list(row0) == [1, 0, 0, 0, 0]

      # Last position sees last 2
      row4 = Nx.slice(mask, [4, 0], [1, 5]) |> Nx.squeeze()
      assert Nx.to_list(row4) == [0, 0, 0, 1, 1]
    end
  end

  describe "build_sliding_window/1" do
    test "builds model with correct output shape" do
      model = Attention.build_sliding_window(
        embed_size: @embed_size,
        window_size: 5,
        seq_len: @seq_len,  # Use test's seq_len (differs from window_size)
        num_heads: @num_heads,
        head_dim: @head_dim,
        num_layers: 1
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(
        Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, num_heads * head_dim]
      expected_dim = @num_heads * @head_dim
      assert Nx.shape(output) == {@batch_size, expected_dim}
    end

    test "works with multiple layers" do
      model = Attention.build_sliding_window(
        embed_size: @embed_size,
        window_size: 5,
        seq_len: @seq_len,  # Use test's seq_len (differs from window_size)
        num_heads: @num_heads,
        head_dim: @head_dim,
        num_layers: 3
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(
        Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @num_heads * @head_dim}
    end
  end

  describe "build_hybrid/1" do
    test "builds LSTM + attention hybrid model" do
      model = Attention.build_hybrid(
        embed_size: @embed_size,
        lstm_hidden: 32,
        lstm_layers: 1,
        num_heads: @num_heads,
        head_dim: @head_dim,
        seq_len: @seq_len  # Use test's seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(
        Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      expected_dim = @num_heads * @head_dim
      assert Nx.shape(output) == {@batch_size, expected_dim}
    end
  end

  describe "build_hybrid_mlp/1" do
    test "adds MLP layers to hybrid model" do
      model = Attention.build_hybrid_mlp(
        embed_size: @embed_size,
        lstm_hidden: 32,
        num_heads: @num_heads,
        head_dim: @head_dim,
        mlp_sizes: [64, 32],
        seq_len: @seq_len  # Use test's seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(
        Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be last MLP size
      assert Nx.shape(output) == {@batch_size, 32}
    end
  end

  describe "add_positional_encoding/2" do
    test "adds encoding without changing shape" do
      input = Axon.input("test", shape: {nil, nil, @embed_size})
      encoded = Attention.add_positional_encoding(input)

      {init_fn, predict_fn} = Axon.build(encoded)
      params = init_fn.(
        Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input_data = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch_size, @seq_len, @embed_size}

      # Output should not be all zeros (positional encoding added)
      assert Nx.to_number(Nx.sum(Nx.abs(output))) > 0
    end
  end

  describe "output_size/1" do
    test "returns correct dimension" do
      assert Attention.output_size(num_heads: 4, head_dim: 64) == 256
      assert Attention.output_size(num_heads: 8, head_dim: 32) == 256
      assert Attention.output_size() == 256  # defaults
    end
  end

  describe "melee_defaults/0" do
    test "returns sensible defaults for 60fps gameplay" do
      defaults = Attention.melee_defaults()

      assert defaults[:window_size] == 60  # 1 second
      assert defaults[:num_heads] == 4
      assert defaults[:head_dim] == 64
    end
  end

  describe "concrete seq_len regression test" do
    # Regression test for: dynamic sequence length causes infinite JIT compilation
    # When seq_len is nil (dynamic), XLA tries to compile kernels for arbitrary
    # sequence lengths which can take 30+ minutes or never complete.
    # The fix: default seq_len to window_size for efficient compilation.

    test "build_sliding_window defaults seq_len to window_size (not nil)" do
      # This test verifies the fix is in place:
      # - Model should compile quickly with concrete seq_len
      # - Input shape should match window_size, not be dynamic

      window_size = 5
      model = Attention.build_sliding_window(
        embed_size: @embed_size,
        window_size: window_size,
        num_heads: @num_heads,
        head_dim: @head_dim
        # NOTE: Not passing seq_len - should default to window_size
      )

      # Model should build and compile within reasonable time (would hang with dynamic)
      {init_fn, predict_fn} = Axon.build(model)

      # Template must use window_size as seq_len (the default)
      params = init_fn.(
        Nx.template({@batch_size, window_size, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      # Should execute without issue
      input = Nx.broadcast(0.5, {@batch_size, window_size, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @num_heads * @head_dim}
    end

    test "build_sliding_window with explicit seq_len works" do
      # When seq_len differs from window_size (e.g., for variable-length inference),
      # it should still work but requires explicit seq_len parameter

      window_size = 5
      seq_len = 10  # Different from window_size

      model = Attention.build_sliding_window(
        embed_size: @embed_size,
        window_size: window_size,
        seq_len: seq_len,  # Explicit seq_len
        num_heads: @num_heads,
        head_dim: @head_dim
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(
        Nx.template({@batch_size, seq_len, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

      input = Nx.broadcast(0.5, {@batch_size, seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @num_heads * @head_dim}
    end
  end
end
