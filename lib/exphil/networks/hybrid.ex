defmodule ExPhil.Networks.Hybrid do
  @moduledoc """
  Jamba: Hybrid Mamba+Attention architecture for efficient sequence modeling.

  Named after AI21's Jamba architecture, this module combines Mamba's O(L)
  efficiency for local context with attention's ability to capture long-range
  dependencies.

  Use via CLI: `--backbone jamba`

  ## Architecture Pattern

  ```
  Input [batch, seq_len, embed_size]
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Mamba Block 1                       │
  ├─────────────────────────────────────┤
  │  Mamba Block 2                       │
  ├─────────────────────────────────────┤
  │  Mamba Block 3                       │
  ├─────────────────────────────────────┤
  │  Attention Block (every N layers)   │  ← Long-range dependencies
  ├─────────────────────────────────────┤
  │  Mamba Block 4                       │
  ├─────────────────────────────────────┤
  │  ...                                 │
  └─────────────────────────────────────┘
        │
        ▼
  [batch, hidden_size]
  ```

  ## Key Advantages

  1. **Efficiency**: Most layers are O(L) Mamba blocks
  2. **Long-range**: Periodic attention captures distant dependencies
  3. **Melee-specific**: Attention at reaction-time intervals (12-18 frames)
  4. **Memory**: Far less than pure attention, slightly more than pure Mamba

  ## Usage

      # Default hybrid (3 Mamba : 1 Attention ratio)
      model = Hybrid.build(
        embed_size: 1991,
        hidden_size: 256,
        num_layers: 8,
        attention_every: 4
      )

      # More attention for complex dependencies
      model = Hybrid.build(
        embed_size: 1991,
        num_layers: 6,
        attention_every: 2  # Alternating Mamba/Attention
      )
  """

  require Axon

  alias ExPhil.Networks.Mamba
  alias ExPhil.Networks.Attention

  # Default hyperparameters
  @default_hidden_size 256
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4
  @default_num_layers 6
  @default_attention_every 3  # Attention every 3rd layer
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1

  @doc """
  Build a hybrid Mamba+Attention model.

  ## Options

  **Architecture:**
    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Total number of layers (default: 6)
    - `:attention_every` - Insert attention every N layers (default: 3)

  **Mamba-specific:**
    - `:state_size` - SSM state dimension (default: 16)
    - `:expand_factor` - Mamba expansion factor (default: 2)
    - `:conv_size` - Causal conv kernel size (default: 4)

  **Attention-specific:**
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:window_size` - Attention window size (default: 60)
    - `:use_sliding_window` - Use sliding window vs full attention (default: true)

  **General:**
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Fixed sequence length for JIT optimization (default: window_size)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.

  ## Example

      # For Melee: 60fps, ~1 second window, efficient inference
      model = Hybrid.build(
        embed_size: 1991,
        hidden_size: 256,
        num_layers: 6,
        attention_every: 3,  # 4 Mamba + 2 Attention layers
        window_size: 60
      )
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Mamba options
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)

    # Attention options
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    use_sliding_window = Keyword.get(opts, :use_sliding_window, true)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Pre-compute attention mask for sliding window
    precomputed_mask = if use_sliding_window and seq_len do
      Attention.window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
    else
      nil
    end

    attn_hidden_dim = num_heads * head_dim

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x = if embed_size != hidden_size do
      Axon.dense(input, hidden_size, name: "input_projection")
    else
      input
    end

    # Add positional encoding (important for attention layers to know positions)
    x = Attention.add_positional_encoding(x, name: "pos_encoding")

    # Build interleaved layers
    x = Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
      is_attention_layer = rem(layer_idx, attention_every) == 0

      if is_attention_layer do
        # Attention layer
        build_attention_layer(
          acc,
          hidden_size: hidden_size,
          attn_hidden_dim: attn_hidden_dim,
          num_heads: num_heads,
          head_dim: head_dim,
          dropout: dropout,
          use_sliding_window: use_sliding_window,
          window_size: window_size,
          precomputed_mask: precomputed_mask,
          name: "layer_#{layer_idx}_attn"
        )
      else
        # Mamba layer
        build_mamba_layer(
          acc,
          hidden_size: hidden_size,
          state_size: state_size,
          expand_factor: expand_factor,
          conv_size: conv_size,
          dropout: dropout,
          name: "layer_#{layer_idx}_mamba"
        )
      end
    end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(x, fn tensor ->
      seq_len_actual = Nx.axis_size(tensor, 1)
      Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "last_timestep")
  end

  @doc """
  Build a Mamba layer with residual connection.
  """
  @spec build_mamba_layer(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "mamba_layer")

    # Mamba block (includes internal normalization)
    block = Mamba.build_mamba_block(
      input,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      name: name
    )

    # Residual connection + dropout
    residual = Axon.add(input, block, name: "#{name}_residual")

    if dropout > 0 do
      Axon.dropout(residual, rate: dropout, name: "#{name}_dropout")
    else
      residual
    end
  end

  @doc """
  Build an attention layer with residual connection and FFN.
  """
  @spec build_attention_layer(Axon.t(), keyword()) :: Axon.t()
  def build_attention_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    attn_hidden_dim = Keyword.get(opts, :attn_hidden_dim, hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_sliding_window = Keyword.get(opts, :use_sliding_window, true)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    precomputed_mask = Keyword.get(opts, :precomputed_mask, nil)
    name = Keyword.get(opts, :name, "attn_layer")

    # Pre-attention normalization
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Project to attention dimension if needed
    x = if hidden_size != attn_hidden_dim do
      Axon.dense(normalized, attn_hidden_dim, name: "#{name}_attn_proj_in")
    else
      normalized
    end

    # Attention
    attended = if use_sliding_window do
      Attention.sliding_window_attention(x,
        window_size: window_size,
        num_heads: num_heads,
        head_dim: head_dim,
        mask: precomputed_mask,
        name: name
      )
    else
      Attention.multi_head_attention(x,
        num_heads: num_heads,
        head_dim: head_dim,
        dropout: dropout,
        causal: true,
        name: name
      )
    end

    # Project back to hidden_size if needed
    attended = if hidden_size != attn_hidden_dim do
      Axon.dense(attended, hidden_size, name: "#{name}_attn_proj_out")
    else
      attended
    end

    # Residual + post-norm
    x = Axon.add(input, attended, name: "#{name}_residual1")
    x = Axon.layer_norm(x, name: "#{name}_post_norm1")

    # Feed-forward network
    ffn_dim = hidden_size * 4  # Standard 4x expansion
    ffn = x
    |> Axon.dense(ffn_dim, name: "#{name}_ffn1")
    |> Axon.gelu()
    |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    ffn = if dropout > 0 do
      Axon.dropout(ffn, rate: dropout, name: "#{name}_ffn_dropout")
    else
      ffn
    end

    # Final residual + norm
    x = Axon.add(x, ffn, name: "#{name}_residual2")
    Axon.layer_norm(x, name: "#{name}_post_norm2")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a hybrid model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a hybrid model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 1991)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    inner_size = hidden_size * expand_factor
    attn_hidden_dim = num_heads * head_dim
    dt_rank = div(hidden_size, state_size)

    # Per Mamba layer
    mamba_per_layer = hidden_size * (2 * inner_size) +  # in_proj
                      inner_size * 4 +                    # conv
                      inner_size * (2 * state_size) +     # BC
                      inner_size * dt_rank + dt_rank * inner_size +  # dt
                      inner_size * hidden_size            # out_proj

    # Per Attention layer
    ffn_dim = hidden_size * 4
    attn_per_layer = attn_hidden_dim * 3 * attn_hidden_dim +  # QKV
                     attn_hidden_dim * attn_hidden_dim +       # output proj
                     hidden_size * ffn_dim +                   # FFN1
                     ffn_dim * hidden_size                     # FFN2

    # Count layer types
    num_attn_layers = div(num_layers, attention_every)
    num_mamba_layers = num_layers - num_attn_layers

    # Input projection
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + (mamba_per_layer * num_mamba_layers) + (attn_per_layer * num_attn_layers)
  end

  @doc """
  Get recommended defaults for Melee gameplay (60fps).

  Optimized for:
  - Real-time inference (~10ms budget)
  - 1-second context window
  - Balance between local patterns (Mamba) and reaction timing (Attention)
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      num_layers: 6,
      attention_every: 3,  # 2 Mamba : 1 Attention ratio (4 Mamba, 2 Attn layers)
      num_heads: 4,
      head_dim: 64,
      window_size: 60,
      use_sliding_window: true,
      dropout: 0.1
    ]
  end

  @doc """
  Get the layer pattern for a given configuration.

  Returns a list describing each layer type for debugging/visualization.

  ## Example

      iex> Hybrid.layer_pattern(num_layers: 6, attention_every: 3)
      [:mamba, :mamba, :attention, :mamba, :mamba, :attention]
  """
  @spec layer_pattern(keyword()) :: [atom()]
  def layer_pattern(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)

    Enum.map(1..num_layers, fn idx ->
      if rem(idx, attention_every) == 0, do: :attention, else: :mamba
    end)
  end
end
