defmodule ExPhil.Networks.Attention do
  @moduledoc """
  Temporal attention mechanisms for Melee game state processing.

  Provides two main architectures:

  ## Option C: Sliding Window Attention

  Efficient attention that only looks at the last K frames:

  ```
  Frame 55  Frame 56  Frame 57  Frame 58  Frame 59  Frame 60
     │         │         │         │         │         │
     └─────────┴─────────┴─────────┴─────────┴─────────┘
                         Attend to window
                               │
                               ▼
                        Current Output
  ```

  O(K²) complexity instead of O(N²) - practical for real-time.

  ## Option B: Hybrid LSTM + Attention

  LSTM compresses temporal information, then attention finds long-range patterns:

  ```
  Frames → LSTM → [h1, h2, ..., hN] → Self-Attention → Output
  ```

  Best of both worlds:
  - LSTM captures local sequential patterns
  - Attention finds sparse long-range dependencies

  ## Why Attention Helps Melee

  1. **Direct frame access**: "What happened exactly 12 frames ago?"
  2. **Learned relevance**: Model decides which past frames matter
  3. **Parallel training**: Unlike LSTM, attention can process all frames simultaneously
  4. **Interpretable**: Attention weights show what the model focuses on

  ## Usage

      # Sliding window model
      model = Attention.build_sliding_window(
        embed_size: 1024,
        window_size: 60,  # 1 second at 60fps
        num_heads: 4,
        head_dim: 64
      )

      # Hybrid LSTM + Attention
      model = Attention.build_hybrid(
        embed_size: 1024,
        lstm_hidden: 256,
        num_heads: 4,
        head_dim: 64
      )
  """

  require Axon

  alias ExPhil.Networks.FusedOps
  alias ExPhil.Networks.Recurrent

  # Default hyperparameters
  @default_num_heads 4
  @default_head_dim 64
  # 1 second at 60fps
  @default_window_size 60
  @default_dropout 0.1

  # ============================================================================
  # Core Attention Mechanisms
  # ============================================================================

  @doc """
  Scaled dot-product attention.

  Computes: softmax(QK^T / sqrt(d_k)) * V

  ## Parameters
    - `query` - Query tensor [batch, seq_q, dim]
    - `key` - Key tensor [batch, seq_k, dim]
    - `value` - Value tensor [batch, seq_k, dim]
    - `opts` - Options including :mask for causal/window masking

  ## Returns
    Attention output [batch, seq_q, dim]
  """
  @spec scaled_dot_product_attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def scaled_dot_product_attention(query, key, value, opts \\ []) do
    mask = opts[:mask]

    d_k = Nx.axis_size(key, -1)
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    # Batched dot product: QK^T
    scores = Nx.dot(query, [2], [0], key, [2], [0])
    scores = Nx.divide(scores, scale)

    # Apply mask if provided
    scores =
      if mask do
        # Broadcast mask to match scores shape if needed
        mask =
          if tuple_size(Nx.shape(mask)) == 2 do
            # Mask is [seq, seq], need to broadcast to [batch, seq, seq]
            Nx.broadcast(Nx.new_axis(mask, 0), Nx.shape(scores))
          else
            mask
          end

        Nx.select(
          mask,
          scores,
          Nx.broadcast(-1.0e9, Nx.shape(scores))
        )
      else
        scores
      end

    # Numerically stable softmax (fused for better performance)
    weights = FusedOps.fused_softmax(scores)

    # Batched: weights @ value
    Nx.dot(weights, [2], [0], value, [1], [0])
  end

  @doc """
  Create a causal (autoregressive) attention mask.

  Each position can only attend to itself and previous positions.
  """
  @spec causal_mask(non_neg_integer()) :: Nx.Tensor.t()
  def causal_mask(seq_len) do
    # Lower triangular matrix of 1s
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    Nx.greater_equal(rows, cols)
  end

  @doc """
  Create a sliding window attention mask.

  Each position can only attend to positions within the window.
  """
  @spec window_mask(non_neg_integer(), non_neg_integer()) :: Nx.Tensor.t()
  def window_mask(seq_len, window_size) do
    # Each position attends to [max(0, i - window + 1), i]
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    # Valid if: col <= row (causal) AND col >= row - window + 1
    causal = Nx.greater_equal(rows, cols)
    in_window = Nx.greater_equal(cols, Nx.subtract(rows, window_size - 1))

    Nx.logical_and(causal, in_window)
  end

  # ============================================================================
  # Multi-Head Attention Layer
  # ============================================================================

  @doc """
  Build a simplified self-attention Axon layer.

  Uses single-head attention for simplicity and Axon compatibility.

  ## Options
    - `:hidden_dim` - Hidden dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:causal` - Use causal masking (default: true)
    - `:qk_layernorm` - Normalize Q and K before attention (stabilizes training, default: false)
    - `:name` - Layer name prefix
  """
  @spec self_attention(Axon.t(), keyword()) :: Axon.t()
  def self_attention(input, opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, 256)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    causal = Keyword.get(opts, :causal, true)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, false)
    name = Keyword.get(opts, :name, "self_attn")

    # Project to Q, K, V and concatenate for single layer call
    qkv = Axon.dense(input, hidden_dim * 3, name: "#{name}_qkv")

    # Apply attention in a single Axon.nx call
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {_batch, seq_len, _} = Nx.shape(qkv_tensor)

          # Split into Q, K, V
          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

          # QK LayerNorm: normalize Q and K to prevent attention explosion
          {query, key} =
            if qk_layernorm do
              {qk_layer_norm(query), qk_layer_norm(key)}
            else
              {query, key}
            end

          mask =
            if causal do
              causal_mask(seq_len)
            else
              nil
            end

          scaled_dot_product_attention(query, key, value, mask: mask)
        end,
        name: "#{name}_compute"
      )

    # Output projection
    attended
    |> Axon.dense(hidden_dim, name: "#{name}_output")
    |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
  end

  # Keep multi_head_attention as alias for compatibility
  @doc """
  Alias for self_attention with configurable dimension.

  Passes through all options including `:qk_layernorm`.
  """
  @spec multi_head_attention(Axon.t(), keyword()) :: Axon.t()
  def multi_head_attention(input, opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    hidden_dim = num_heads * head_dim

    # Pass all options (including qk_layernorm) to self_attention
    self_attention(input, Keyword.put(opts, :hidden_dim, hidden_dim))
  end

  @doc """
  Build a sliding window attention layer.

  More efficient than full attention - O(K²) per position instead of O(N²).

  ## Options
    - `:window_size` - Attention window size (default: 60)
    - `:hidden_dim` - Hidden dimension (default: 256)
    - `:mask` - Pre-computed attention mask (recommended for efficient compilation)
    - `:qk_layernorm` - Normalize Q and K before attention (stabilizes training, default: false)
    - `:name` - Layer name prefix
  """
  @spec sliding_window_attention(Axon.t(), keyword()) :: Axon.t()
  def sliding_window_attention(input, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    precomputed_mask = Keyword.get(opts, :mask, nil)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, false)
    name = Keyword.get(opts, :name, "window_attn")

    hidden_dim = num_heads * head_dim

    # Project to Q, K, V in single dense layer
    qkv = Axon.dense(input, hidden_dim * 3, name: "#{name}_qkv")

    # Apply windowed attention with pre-computed mask (captured from outer scope)
    # This avoids dynamic mask creation inside Axon.nx which causes XLA issues
    Axon.nx(
      qkv,
      fn qkv_tensor ->
        {_batch, seq_len, _} = Nx.shape(qkv_tensor)

        # Split into Q, K, V
        query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
        key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
        value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

        # QK LayerNorm: normalize Q and K to prevent attention explosion
        # This is used in modern transformers like ViT-22B and PaLM-2
        {query, key} =
          if qk_layernorm do
            {qk_layer_norm(query), qk_layer_norm(key)}
          else
            {query, key}
          end

        # Use pre-computed mask if available, otherwise compute (slow path)
        mask =
          if precomputed_mask != nil do
            precomputed_mask
          else
            window_mask(seq_len, window_size)
          end

        scaled_dot_product_attention(query, key, value, mask: mask)
      end,
      name: "#{name}_compute"
    )
  end

  @doc """
  Apply layer normalization to Q or K tensors.

  QK LayerNorm normalizes across the feature dimension to prevent
  attention score explosion in deep networks.
  """
  @spec qk_layer_norm(Nx.Tensor.t()) :: Nx.Tensor.t()
  def qk_layer_norm(tensor) do
    # Normalize across the last axis (feature dimension)
    # Mean and variance over last axis
    mean = Nx.mean(tensor, axes: [-1], keep_axes: true)
    variance = Nx.variance(tensor, axes: [-1], keep_axes: true)
    eps = 1.0e-6

    # Normalize
    Nx.divide(Nx.subtract(tensor, mean), Nx.sqrt(Nx.add(variance, eps)))
  end

  # ============================================================================
  # Positional Encoding
  # ============================================================================

  @doc """
  Add sinusoidal positional encoding to input.

  Uses sin for position encoding - compatible with Axon's JIT compilation.
  Each position gets a unique encoding based on sine waves at different
  frequencies across the embedding dimensions.
  """
  @spec add_positional_encoding(Axon.t(), keyword()) :: Axon.t()
  def add_positional_encoding(input, opts \\ []) do
    name = Keyword.get(opts, :name, "pos_enc")
    scale = Keyword.get(opts, :scale, 0.01)

    Axon.nx(
      input,
      fn tensor ->
        # Get actual shape from tensor
        shape = Nx.shape(tensor)
        seq_len = elem(shape, 1)
        embed_dim = elem(shape, 2)

        # Position indices [1, seq_len, 1] - will broadcast to [batch, seq_len, embed_dim]
        pos = Nx.iota({1, seq_len, 1}, axis: 1) |> Nx.as_type(:f32)

        # Dimension indices [1, 1, embed_dim] - different frequency per dimension
        dim = Nx.iota({1, 1, embed_dim}, axis: 2) |> Nx.as_type(:f32)

        # Compute angles: combine position and dimension
        # Higher dimensions get lower frequencies (standard positional encoding behavior)
        # angle = pos * exp(-dim * log(10000) / embed_dim)
        # Simplified: angle = pos / (1 + dim * scale_factor)
        freq = Nx.divide(1.0, Nx.add(1.0, Nx.multiply(dim, scale)))

        # angles shape: [1, seq_len, embed_dim] via broadcasting
        angles = Nx.multiply(pos, freq)

        # Apply sin to create positional encoding
        pe = Nx.sin(angles)

        # Add to input (broadcasts over batch dimension)
        Nx.add(tensor, pe)
      end,
      name: name
    )
  end

  # ============================================================================
  # Option C: Sliding Window Model
  # ============================================================================

  @doc """
  Build a complete sliding window attention model.

  Efficient for real-time inference - only attends to recent frames.

  ## Options
    - `:embed_size` - Input embedding size (required)
    - `:window_size` - Attention window (default: 60 = 1 second)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:num_layers` - Number of attention layers (default: 2)
    - `:ffn_dim` - Feed-forward dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_dim] from last position.
  """
  @spec build_sliding_window(keyword()) :: Axon.t()
  def build_sliding_window(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_layers = Keyword.get(opts, :num_layers, 2)
    ffn_dim = Keyword.get(opts, :ffn_dim, 256)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    hidden_dim = num_heads * head_dim

    # Sequence length configuration:
    # - :seq_len option: Explicit sequence length for the input
    # - Defaults to window_size for training efficiency (concrete shape = fast JIT)
    # - Set to nil for dynamic sequence length (slower JIT, more flexible)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Pre-compute attention mask if seq_len is concrete
    # This avoids creating masks inside Axon.nx which causes XLA compilation issues
    # IMPORTANT: Convert mask to BinaryBackend to avoid EXLA/Defn.Expr mismatch
    # when the mask is captured in Axon.nx closures during JIT compilation
    {precomputed_mask, input_seq_dim} =
      if seq_len do
        mask = window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
        {mask, seq_len}
      else
        # Dynamic - mask computed at runtime (slow path)
        {nil, nil}
      end

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project to hidden dimension
    x = Axon.dense(input, hidden_dim, name: "input_proj")

    # Add positional encoding
    x = add_positional_encoding(x, name: "pos_encoding")

    # Stack of attention + FFN layers
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Sliding window attention with pre-computed mask for efficient compilation
        attended =
          sliding_window_attention(acc,
            window_size: window_size,
            num_heads: num_heads,
            head_dim: head_dim,
            mask: precomputed_mask,
            name: "layer_#{layer_idx}_attn"
          )

        # Residual + LayerNorm
        acc = Axon.add(acc, attended, name: "layer_#{layer_idx}_residual1")
        acc = Axon.layer_norm(acc, name: "layer_#{layer_idx}_norm1")

        # Feed-forward network
        ffn =
          acc
          |> Axon.dense(ffn_dim, name: "layer_#{layer_idx}_ffn1")
          |> Axon.gelu()
          |> Axon.dense(hidden_dim, name: "layer_#{layer_idx}_ffn2")
          |> Axon.dropout(rate: dropout)

        # Residual + LayerNorm
        acc = Axon.add(acc, ffn, name: "layer_#{layer_idx}_residual2")
        Axon.layer_norm(acc, name: "layer_#{layer_idx}_norm2")
      end)

    # Take last position output: [batch, seq, hidden] -> [batch, hidden]
    # Use concrete seq_len if available for efficient compilation
    Axon.nx(
      x,
      fn tensor ->
        last_idx =
          if seq_len do
            seq_len - 1
          else
            Nx.axis_size(tensor, 1) - 1
          end

        Nx.slice_along_axis(tensor, last_idx, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_position"
    )
  end

  # ============================================================================
  # Option B: Hybrid LSTM + Attention
  # ============================================================================

  @doc """
  Build a hybrid LSTM + Attention model.

  LSTM captures local sequential patterns, attention finds long-range dependencies.

  ## Architecture
  ```
  Frames → LSTM → Hidden States → Self-Attention → Output
  ```

  ## Options
    - `:embed_size` - Input embedding size (required)
    - `:lstm_hidden` - LSTM hidden size (default: 256)
    - `:lstm_layers` - Number of LSTM layers (default: 1)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_dim] combining LSTM and attention.
  """
  @spec build_hybrid(keyword()) :: Axon.t()
  def build_hybrid(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    lstm_hidden = Keyword.get(opts, :lstm_hidden, 256)
    lstm_layers = Keyword.get(opts, :lstm_layers, 1)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    hidden_dim = num_heads * head_dim

    # Sequence length configuration (same as build_sliding_window)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # LSTM backbone (returns all timesteps)
    lstm_output =
      Recurrent.build_backbone(input,
        hidden_size: lstm_hidden,
        num_layers: lstm_layers,
        cell_type: :lstm,
        dropout: dropout,
        # Need all timesteps for attention
        return_sequences: true
      )

    # Project LSTM output to attention dimension
    x = Axon.dense(lstm_output, hidden_dim, name: "lstm_to_attn_proj")

    # Self-attention over LSTM hidden states
    attended =
      multi_head_attention(x,
        num_heads: num_heads,
        head_dim: head_dim,
        dropout: dropout,
        causal: true,
        name: "hybrid_attn"
      )

    # Residual connection
    x = Axon.add(x, attended, name: "attn_residual")
    x = Axon.layer_norm(x, name: "attn_norm")

    # Take last position - use concrete seq_len if available
    Axon.nx(
      x,
      fn tensor ->
        last_idx =
          if seq_len do
            seq_len - 1
          else
            Nx.axis_size(tensor, 1) - 1
          end

        Nx.slice_along_axis(tensor, last_idx, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "hybrid_last"
    )
  end

  @doc """
  Build hybrid model with additional MLP layers on top.

  Good for policy/value heads that need more non-linearity.
  """
  @spec build_hybrid_mlp(keyword()) :: Axon.t()
  def build_hybrid_mlp(opts \\ []) do
    mlp_sizes = Keyword.get(opts, :mlp_sizes, [256, 256])
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Build base hybrid model
    hybrid = build_hybrid(opts)

    # Add MLP layers
    mlp_sizes
    |> Enum.with_index()
    |> Enum.reduce(hybrid, fn {size, idx}, acc ->
      acc
      |> Axon.dense(size, name: "hybrid_mlp_#{idx}")
      |> Axon.relu()
      |> Axon.dropout(rate: dropout)
    end)
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_heads * head_dim
  end

  @doc """
  Default configuration for Melee (60fps, 1-second window).
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      # 1 second at 60fps
      window_size: 60,
      num_heads: 4,
      # 256 total dim
      head_dim: 64,
      num_layers: 2,
      dropout: 0.1
    ]
  end
end
