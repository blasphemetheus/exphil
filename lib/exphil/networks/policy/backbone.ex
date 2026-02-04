defmodule ExPhil.Networks.Policy.Backbone do
  @moduledoc """
  Backbone network builders for policy networks.

  Provides builders for various temporal and non-temporal backbones:
  - MLP (feedforward)
  - LSTM / GRU (recurrent)
  - Attention (sliding window)
  - Mamba variants (SSM)
  - Jamba (Mamba + Attention hybrid)

  ## Backbone Types

  | Type | Description | Best For |
  |------|-------------|----------|
  | `:mlp` | Simple feedforward | Single-frame, fast inference |
  | `:lstm` | Long Short-Term Memory | Sequential patterns |
  | `:gru` | Gated Recurrent Unit | Lighter than LSTM |
  | `:sliding_window` | Windowed attention | Fixed-length sequences |
  | `:mamba` | State space model | Long sequences, fast training |
  | `:jamba` | Mamba + Attention | Best quality, moderate speed |
  | `:griffin` | RG-LRU + Local Attention | Simpler recurrence, good quality |
  | `:hawk` | Pure RG-LRU | Fastest recurrent, simpler than Mamba |
  | `:xlstm` | Extended LSTM (mixed) | Exponential gating, matrix memory |
  | `:retnet` | Retentive Network | O(1) inference, decay-based attention |

  ## Usage

      # Build temporal backbone
      backbone = Backbone.build_temporal_backbone(embed_size, :mamba, opts)

      # Get output size
      output_size = Backbone.temporal_backbone_output_size(:mamba, opts)

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Networks.Mamba` - Mamba SSM implementation
  - `ExPhil.Networks.Attention` - Attention layers
  """

  require Axon

  alias ExPhil.Networks.Attention
  alias ExPhil.Networks.GatedSSM
  alias ExPhil.Networks.Griffin
  alias ExPhil.Networks.Hybrid
  alias ExPhil.Networks.Mamba
  alias ExPhil.Networks.MambaCumsum
  alias ExPhil.Networks.MambaNIF
  alias ExPhil.Networks.Recurrent

  # Default architecture hyperparameters
  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  @type backbone_type ::
          :mlp
          | :sliding_window
          | :attention
          | :lstm_hybrid
          | :lstm
          | :gru
          | :mamba
          | :mamba_nif
          | :mamba_cumsum
          | :mamba_hillis_steele
          | :mamba_ssd
          | :gated_ssm
          | :jamba
          | :zamba
          | :griffin
          | :hawk
          | :xlstm
          | :xlstm_slstm
          | :xlstm_mlstm
          | :retnet
          | :rwkv
          | :gla
          | :hgrn
          | :s5
          | :decision_transformer
          | :liquid

  @doc """
  Build a temporal backbone that processes frame sequences.

  Returns an Axon layer that outputs [batch, hidden_dim] from sequence input.

  ## Parameters
    - `embed_size` - Size of input embeddings per frame
    - `backbone_type` - Type of backbone to build
    - `opts` - Backbone-specific options

  ## Common Options
    - `:window_size` - Sequence length (default: 60)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of layers (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:num_heads` - Attention heads for attention-based (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
  """
  @spec build_temporal_backbone(non_neg_integer(), backbone_type(), keyword()) :: Axon.t()
  def build_temporal_backbone(embed_size, backbone_type, opts \\ []) do
    case backbone_type do
      :sliding_window ->
        build_sliding_window_backbone(embed_size, opts)

      :attention ->
        # Pure attention (alias for sliding_window)
        build_sliding_window_backbone(embed_size, opts)

      :lstm_hybrid ->
        # LSTM + Attention hybrid
        build_lstm_attention_backbone(embed_size, opts)

      :jamba ->
        # Recommended: Mamba + Attention hybrid
        build_jamba_backbone(embed_size, opts)

      :zamba ->
        # Mamba + Single Shared Attention (more efficient than Jamba)
        build_zamba_backbone(embed_size, opts)

      :griffin ->
        # Griffin: RG-LRU + Local Attention hybrid (2:1 pattern)
        build_griffin_backbone(embed_size, opts)

      :hawk ->
        # Hawk: Pure RG-LRU (no local attention, simpler/faster than Griffin)
        build_hawk_backbone(embed_size, opts)

      :xlstm ->
        # xLSTM: Mixed sLSTM/mLSTM (alternating)
        build_xlstm_backbone(embed_size, Keyword.put(opts, :variant, :mixed))

      :xlstm_slstm ->
        # xLSTM: Pure sLSTM (scalar memory, state tracking)
        build_xlstm_backbone(embed_size, Keyword.put(opts, :variant, :slstm))

      :xlstm_mlstm ->
        # xLSTM: Pure mLSTM (matrix memory, memorization)
        build_xlstm_backbone(embed_size, Keyword.put(opts, :variant, :mlstm))

      :retnet ->
        # RetNet: Retentive Network with multi-scale retention
        build_retnet_backbone(embed_size, opts)

      :rwkv ->
        # RWKV-7 "Goose" - O(1) space complexity linear attention
        build_rwkv_backbone(embed_size, opts)

      :gla ->
        # GLA (Gated Linear Attention) - O(L) with data-dependent gating
        build_gla_backbone(embed_size, opts)

      :hgrn ->
        # HGRN-2 - Hierarchically Gated Linear RNN with state expansion
        build_hgrn_backbone(embed_size, opts)

      :s5 ->
        # S5 - Simplified State Space (MIMO SSM)
        build_s5_backbone(embed_size, opts)

      :decision_transformer ->
        # Decision Transformer - Return-conditioned sequence modeling
        build_decision_transformer_backbone(embed_size, opts)

      :liquid ->
        # Liquid Neural Networks - Continuous-time adaptive dynamics
        build_liquid_backbone(embed_size, opts)

      :lstm ->
        build_lstm_backbone(embed_size, opts)

      :gru ->
        build_gru_backbone(embed_size, opts)

      :gated_ssm ->
        build_gated_ssm_backbone(embed_size, opts)

      :mamba ->
        build_mamba_backbone(embed_size, opts)

      :mamba_nif ->
        build_mamba_nif_backbone(embed_size, opts)

      :mamba_cumsum ->
        build_mamba_cumsum_backbone(embed_size, opts)

      :mamba_hillis_steele ->
        build_mamba_hillis_steele_backbone(embed_size, opts)

      :mamba_ssd ->
        build_mamba_ssd_backbone(embed_size, opts)

      :mlp ->
        # For MLP, expect single frame input, add sequence handling
        build_mlp_temporal_backbone(embed_size, opts)
    end
  end

  @doc """
  Build just the backbone network (useful for sharing with value network).

  ## Options
    - `:layer_norm` - If true, applies layer normalization after each dense layer (default: false)
    - `:residual` - If true, adds residual (skip) connections between layers (default: false)

  ## Residual Connections

  When `:residual` is enabled, each layer adds its input to its output:

      output = dropout(activation(layer_norm(dense(x)))) + project(x)

  If the input and output dimensions differ, a projection layer is added.
  This enables training deeper networks (+5-15% accuracy improvement).
  """
  @spec build_backbone(Axon.t(), list(), atom(), float(), keyword()) :: Axon.t()
  def build_backbone(input, hidden_sizes, activation, dropout, opts \\ []) do
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)

    # Get input dimension for residual connections
    # We track the previous layer's size to know when we need projection
    {final_layer, _} =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce({input, nil}, fn {size, idx}, {acc, prev_size} ->
        # Build the main transformation path
        layer =
          acc
          |> Axon.dense(size, name: "backbone_dense_#{idx}")

        # Optional layer normalization (post-dense, pre-activation)
        layer =
          if layer_norm do
            Axon.layer_norm(layer, name: "backbone_ln_#{idx}")
          else
            layer
          end

        layer =
          layer
          |> Axon.activation(activation)
          |> Axon.dropout(rate: dropout)

        # Add residual connection if enabled
        layer =
          if residual do
            add_residual_connection(acc, layer, prev_size, size, idx)
          else
            layer
          end

        {layer, size}
      end)

    final_layer
  end

  @doc """
  Get the output size of a temporal backbone.

  Useful for connecting to other networks (value function, etc).
  """
  @spec temporal_backbone_output_size(backbone_type(), keyword()) :: non_neg_integer()
  def temporal_backbone_output_size(backbone_type, opts \\ []) do
    case backbone_type do
      type when type in [:sliding_window, :attention] ->
        num_heads = Keyword.get(opts, :num_heads, 4)
        head_dim = Keyword.get(opts, :head_dim, 64)
        num_heads * head_dim

      :lstm_hybrid ->
        # LSTM + Attention
        num_heads = Keyword.get(opts, :num_heads, 4)
        head_dim = Keyword.get(opts, :head_dim, 64)
        num_heads * head_dim

      :jamba ->
        # New Mamba + Attention hybrid
        Keyword.get(opts, :hidden_size, 256)

      :zamba ->
        # Mamba + Single Shared Attention
        Keyword.get(opts, :hidden_size, 256)

      :griffin ->
        # Griffin: RG-LRU + Local Attention
        Keyword.get(opts, :hidden_size, 256)

      :hawk ->
        # Hawk: Pure RG-LRU
        Keyword.get(opts, :hidden_size, 256)

      type when type in [:xlstm, :xlstm_slstm, :xlstm_mlstm] ->
        # xLSTM variants
        Keyword.get(opts, :hidden_size, 256)

      :retnet ->
        # RetNet: Retentive Network
        Keyword.get(opts, :hidden_size, 256)

      :rwkv ->
        # RWKV-7 "Goose"
        Keyword.get(opts, :hidden_size, 256)

      :gla ->
        # Gated Linear Attention
        Keyword.get(opts, :hidden_size, 256)

      :hgrn ->
        # HGRN-2 Hierarchically Gated RNN
        Keyword.get(opts, :hidden_size, 256)

      :s5 ->
        # S5 Simplified State Space
        Keyword.get(opts, :hidden_size, 256)

      :decision_transformer ->
        # Decision Transformer
        Keyword.get(opts, :hidden_size, 256)

      :liquid ->
        # Liquid Neural Networks
        Keyword.get(opts, :hidden_size, 256)

      :lstm ->
        Keyword.get(opts, :hidden_size, 256)

      :gru ->
        Keyword.get(opts, :hidden_size, 256)

      :gated_ssm ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba_nif ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba_cumsum ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba_hillis_steele ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba_ssd ->
        Keyword.get(opts, :hidden_size, 256)

      :mlp ->
        hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
        List.last(hidden_sizes)
    end
  end

  @doc """
  Get recommended defaults for Melee temporal policy.

  60fps gameplay with 1-second attention window.
  """
  @spec melee_temporal_defaults() :: keyword()
  def melee_temporal_defaults do
    [
      backbone: :sliding_window,
      window_size: 60,
      num_heads: 4,
      head_dim: 64,
      num_layers: 2,
      dropout: 0.1
    ]
  end

  # ============================================================================
  # Private Backbone Builders
  # ============================================================================

  # Add a residual connection between input and output
  # If dimensions differ, add a projection layer
  defp add_residual_connection(input, output, prev_size, current_size, idx) do
    if prev_size == current_size do
      # Dimensions match, simple addition
      Axon.add(input, output, name: "backbone_residual_#{idx}")
    else
      # Dimensions differ, need projection
      projected = Axon.dense(input, current_size, name: "backbone_proj_#{idx}")
      Axon.add(projected, output, name: "backbone_residual_#{idx}")
    end
  end

  defp build_sliding_window_backbone(embed_size, opts) do
    window_size = Keyword.get(opts, :window_size, 60)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    Attention.build_sliding_window(
      embed_size: embed_size,
      window_size: window_size,
      num_heads: num_heads,
      head_dim: head_dim,
      num_layers: num_layers,
      dropout: dropout
    )
  end

  # Legacy LSTM + Attention hybrid (kept for backwards compatibility)
  defp build_lstm_attention_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    lstm_layers = Keyword.get(opts, :num_layers, 1)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Attention.build_hybrid(
      embed_size: embed_size,
      lstm_hidden: hidden_size,
      lstm_layers: lstm_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      # For concrete seq_len (efficient JIT)
      window_size: window_size
    )
  end

  # New Mamba + Attention hybrid (recommended for efficiency + long-range)
  defp build_jamba_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 6)
    attention_every = Keyword.get(opts, :attention_every, 3)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    use_sliding_window = Keyword.get(opts, :use_sliding_window, true)
    # seq_len must be passed through for concrete mask shapes (avoids shape mismatch)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    # Stability options (Pre-LN + QK LayerNorm prevent NaN in Jamba)
    pre_norm = Keyword.get(opts, :pre_norm, true)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, true)

    Hybrid.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      attention_every: attention_every,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len,
      use_sliding_window: use_sliding_window,
      pre_norm: pre_norm,
      qk_layernorm: qk_layernorm
    )
  end

  # Zamba: Mamba + Single Shared Attention (more efficient than Jamba)
  defp build_zamba_backbone(embed_size, opts) do
    alias ExPhil.Networks.Zamba

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 6)
    attention_every = Keyword.get(opts, :attention_every, 3)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Zamba.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      attention_every: attention_every,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Griffin: RG-LRU + Local Attention hybrid (2:1 pattern)
  defp build_griffin_backbone(embed_size, opts) do
    alias ExPhil.Networks.Griffin

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    expand_factor = Keyword.get(opts, :expand_factor, 3)
    local_attn_window = Keyword.get(opts, :local_attn_window, 32)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Griffin.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      expand_factor: expand_factor,
      local_attn_window: local_attn_window,
      num_heads: num_heads,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len,
      use_local_attention: true
    )
  end

  # Hawk: Pure RG-LRU (no local attention, simpler/faster than Griffin)
  defp build_hawk_backbone(embed_size, opts) do
    alias ExPhil.Networks.Griffin

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    expand_factor = Keyword.get(opts, :expand_factor, 3)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Griffin.build_hawk(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      expand_factor: expand_factor,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # xLSTM: Extended LSTM with exponential gating (Hochreiter et al., 2024)
  defp build_xlstm_backbone(embed_size, opts) do
    alias ExPhil.Networks.XLSTM

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    variant = Keyword.get(opts, :variant, :mixed)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    XLSTM.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      expand_factor: expand_factor,
      variant: variant,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # RetNet: Retentive Network with multi-scale retention (Microsoft, 2023)
  defp build_retnet_backbone(embed_size, opts) do
    alias ExPhil.Networks.RetNet

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, 4)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RetNet.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      expand_factor: expand_factor,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # RWKV-7 "Goose": O(1) space complexity linear attention
  defp build_rwkv_backbone(embed_size, opts) do
    alias ExPhil.Networks.RWKV

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    head_size = Keyword.get(opts, :head_size, 64)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RWKV.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      head_size: head_size,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # GLA: Gated Linear Attention with data-dependent gating
  defp build_gla_backbone(embed_size, opts) do
    alias ExPhil.Networks.GLA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GLA.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      expand_factor: expand_factor,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # HGRN-2: Hierarchically Gated Linear RNN with state expansion
  defp build_hgrn_backbone(embed_size, opts) do
    alias ExPhil.Networks.HGRN

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    state_expansion = Keyword.get(opts, :state_expansion, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    HGRN.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      state_expansion: state_expansion,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # S5: Simplified State Space (MIMO SSM)
  defp build_s5_backbone(embed_size, opts) do
    alias ExPhil.Networks.S5

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    S5.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Decision Transformer: Return-conditioned sequence modeling
  defp build_decision_transformer_backbone(embed_size, opts) do
    alias ExPhil.Networks.DecisionTransformer

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, 8)
    head_dim = Keyword.get(opts, :head_dim, 32)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use the simplified build for backbone integration
    DecisionTransformer.build_simple(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Liquid Neural Networks: Continuous-time adaptive dynamics
  defp build_liquid_backbone(embed_size, opts) do
    alias ExPhil.Networks.Liquid

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    integration_steps = Keyword.get(opts, :integration_steps, 1)
    solver = Keyword.get(opts, :solver, :euler)

    Liquid.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len,
      integration_steps: integration_steps,
      solver: solver
    )
  end

  defp build_lstm_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    truncate_bptt = Keyword.get(opts, :truncate_bptt, nil)

    Recurrent.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: :lstm,
      dropout: dropout,
      return_sequences: false,
      # For concrete seq_len (efficient JIT)
      window_size: window_size,
      # Optional: limit gradient flow for faster training
      truncate_bptt: truncate_bptt
    )
  end

  defp build_gru_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    truncate_bptt = Keyword.get(opts, :truncate_bptt, nil)

    Recurrent.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: :gru,
      dropout: dropout,
      return_sequences: false,
      window_size: window_size,
      truncate_bptt: truncate_bptt
    )
  end

  # GatedSSM backbone (simplified gated temporal model, NOT true Mamba)
  # Use :gated_ssm for this. :mamba will use true Mamba once implemented.
  defp build_gated_ssm_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    gradient_checkpoint = Keyword.get(opts, :gradient_checkpoint, false)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 1)

    gated_ssm_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      checkpoint_every: checkpoint_every
    ]

    # Use checkpointed version for memory-efficient training
    if gradient_checkpoint do
      GatedSSM.build_checkpointed(gated_ssm_opts)
    else
      GatedSSM.build(gated_ssm_opts)
    end
  end

  # True Mamba backbone with parallel associative scan
  defp build_mamba_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    mamba_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    ]

    Mamba.build(mamba_opts)
  end

  # Mamba with CUDA-accelerated selective scan via Rust NIF (5x faster!)
  defp build_mamba_nif_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    mamba_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    ]

    MambaNIF.build(mamba_opts)
  end

  # Mamba with cumsum-based scan (optimized for training speed)
  defp build_mamba_cumsum_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    mamba_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    ]

    MambaCumsum.build(mamba_opts)
  end

  defp build_mamba_hillis_steele_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    mamba_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    ]

    ExPhil.Networks.MambaHillisSteele.build(mamba_opts)
  end

  defp build_mamba_ssd_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    chunk_size = Keyword.get(opts, :chunk_size, 16)

    mamba_opts = [
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      chunk_size: chunk_size
    ]

    ExPhil.Networks.MambaSSD.build(mamba_opts)
  end

  defp build_mlp_temporal_backbone(embed_size, opts) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)

    # Sequence length configuration (same as attention models)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: sequence [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Take last frame: [batch, embed_size]
    # Use concrete index when available for efficient compilation
    last_frame =
      Axon.nx(
        input,
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
        name: "last_frame"
      )

    # Apply MLP backbone
    build_backbone(last_frame, hidden_sizes, activation, dropout,
      layer_norm: layer_norm,
      residual: residual
    )
  end

  @doc """
  Build a backbone on already-processed input for temporal models with learned embeddings.

  Uses simple Axon layers directly for flexibility when action/character embeddings
  have already been preprocessed.
  """
  @spec build_on_processed_input(Axon.t(), backbone_type(), keyword()) :: Axon.t()
  def build_on_processed_input(processed_input, backbone_type, opts) do
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    num_layers = Keyword.get(opts, :num_layers, 2)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    case backbone_type do
      type when type in [:sliding_window, :attention] ->
        # Simple multi-head attention stack
        output_dim = num_heads * head_dim

        # Project to attention dimension
        projected = Axon.dense(processed_input, output_dim, name: "action_emb_project")

        # Apply attention layers
        attended =
          Enum.reduce(1..num_layers, projected, fn i, acc ->
            # Self-attention (simplified - just dense layers for now)
            acc
            |> Axon.dense(output_dim, name: "action_emb_attn_#{i}")
            |> Axon.relu()
            |> Axon.dropout(rate: dropout)
          end)

        # Take last frame output
        Axon.nx(
          attended,
          fn x ->
            seq_len = Nx.axis_size(x, 1)
            Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "action_emb_last_frame"
        )

      :gated_ssm ->
        # Project and apply simple recurrent-like processing
        projected = Axon.dense(processed_input, hidden_size, name: "action_emb_gated_ssm_project")

        processed =
          Enum.reduce(1..num_layers, projected, fn i, acc ->
            acc
            |> Axon.dense(hidden_size, name: "action_emb_gated_ssm_#{i}")
            |> Axon.silu()
            |> Axon.dropout(rate: dropout)
          end)

        # Take last frame
        Axon.nx(
          processed,
          fn x ->
            seq_len = Nx.axis_size(x, 1)
            Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "action_emb_gated_ssm_last_frame"
        )

      :mamba ->
        # TODO: Use true Mamba when implemented
        # For now, same as gated_ssm
        projected = Axon.dense(processed_input, hidden_size, name: "action_emb_mamba_project")

        processed =
          Enum.reduce(1..num_layers, projected, fn i, acc ->
            acc
            |> Axon.dense(hidden_size, name: "action_emb_mamba_#{i}")
            |> Axon.silu()
            |> Axon.dropout(rate: dropout)
          end)

        # Take last frame
        Axon.nx(
          processed,
          fn x ->
            seq_len = Nx.axis_size(x, 1)
            Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "action_emb_mamba_last_frame"
        )

      :jamba ->
        # Similar to mamba but with attention every few layers
        projected = Axon.dense(processed_input, hidden_size, name: "action_emb_jamba_project")

        processed =
          Enum.reduce(1..num_layers, projected, fn i, acc ->
            acc
            |> Axon.dense(hidden_size, name: "action_emb_jamba_#{i}")
            |> Axon.silu()
            |> Axon.dropout(rate: dropout)
          end)

        Axon.nx(
          processed,
          fn x ->
            seq_len = Nx.axis_size(x, 1)
            Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "action_emb_jamba_last_frame"
        )

      :mlp ->
        # Take last frame and apply MLP
        last_frame =
          Axon.nx(
            processed_input,
            fn x ->
              seq_len = Nx.axis_size(x, 1)
              Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
            end,
            name: "action_emb_mlp_last_frame"
          )

        Enum.reduce(1..num_layers, last_frame, fn i, acc ->
          acc
          |> Axon.dense(hidden_size, name: "action_emb_mlp_#{i}")
          |> Axon.relu()
          |> Axon.dropout(rate: dropout)
        end)

      other ->
        # For less common backbones, raise helpful error
        raise ArgumentError, """
        Backbone #{inspect(other)} with action_embed_size is not yet supported.
        Supported backbones with learned action embeddings:
        - :sliding_window
        - :mamba
        - :jamba
        - :mlp

        Use action_mode: :one_hot in your embedding config for #{inspect(other)}.
        """
    end
  end
end
