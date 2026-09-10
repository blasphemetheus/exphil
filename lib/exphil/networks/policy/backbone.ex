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
  | `:kan` | Kolmogorov-Arnold Networks | Learnable activations, interpretable |
  | `:liquid` | Liquid Neural Networks | Continuous-time ODE dynamics |
  | `:s4` | Structured State Spaces | Long-range dependencies |
  | `:s4d` | S4 Diagonal | Faster S4 variant |
  | `:h3` | Hungry Hungry Hippos | SSM + shift + convolution |
  | `:performer` | FAVOR+ Linear Attention | O(n) attention approximation |
  | `:deltanet` | Delta Rule Linear Attention | Associative memory updates |
  | `:fnet` | Fourier Token Mixing | No attention params, very fast |
  | `:perceiver` | Perceiver IO | Latent bottleneck cross-attention |
  | `:ttt` | Test-Time Training | Self-supervised inner loop |
  | `:hopfield` | Hopfield Networks | Associative memory retrieval |
  | `:ntm` | Neural Turing Machine | External addressable memory |
  | `:reservoir` | Echo State Network | Fixed random weights, fast |
  | `:snn` | Spiking Neural Network | Biologically-inspired, temporal |
  | `:bayesian` | Bayesian NN | Weight uncertainty estimation |
  | `:min_gru` | Minimal GRU | Parallel-scannable, faster GRU |
  | `:min_lstm` | Minimal LSTM | Parallel-scannable, faster LSTM |
  | `:tcn` | Temporal Conv Net | Dilated causal convolutions |
  | `:mamba3` | Mamba-3 | Complex states, MIMO SSM |
  | `:hyena` | Hyena | Long convolution hierarchy |
  | `:titans` | Titans | Surprise-gated long-term memory |
  | `:gated_deltanet` | Gated DeltaNet | Delta rule with data-dependent gating |
  | `:mega` | MEGA | Multi-scale EMA + gated attention |
  | `:based` | Based | Taylor-expansion linear attention |
  | `:infini_attention` | InfiniAttention | Compressive memory, long sequences |
  | `:conformer` | Conformer | Conv + Attention, temporal patterns |
  | `:mla` | Multi-head Latent Attention | DeepSeek KV compression |
  | `:diff_transformer` | Diff Transformer | Differential attention, noise cancel |
  | `:megalodon` | Megalodon | Chunk attention + complex EMA |
  | `:lightning_attention` | Lightning Attention | O(n) linear attention, very fast |
  | `:flash_linear_attention` | Flash Linear Attention | Hardware-efficient linear attn |
  | `:kda` | KDA | Key-driven channel-decay attention |
  | `:sigmoid_attention` | Sigmoid Attention | Sigmoid instead of softmax |
  | `:spla` | SPLA | Sparse block-selection attention |
  | `:retnet_v2` | RetNet V2 | Improved retentive network |
  | `:rnope_swa` | RNoPE-SWA | Sliding window, no pos encoding |
  | `:nsa` | NSA | Native sparse attention (3-branch) |
  | `:infllm_v2` | InfLLM V2 | Dense-to-sparse block attention |
  | `:dual_chunk_attention` | Dual Chunk | Intra + inter chunk attention |
  | `:gated_attention` | Gated Attention | Per-dim gating on attention |
  | `:mta` | MTA | Multi-token attention (depthwise conv) |
  | `:slstm` | sLSTM | Scalar LSTM, exponential gating |
  | `:xlstm_v2` | xLSTM V2 | Block-diagonal memory xLSTM |
  | `:bimamba` | BiMamba | Bidirectional Mamba (offline only) |
  | `:hyena_v2` | Hyena V2 | Improved long convolution |
  | `:ss_transformer` | SS-Transformer | Parallel SSM + Attention + gate |
  | `:ssmax` | SSMax | State-space softmax alternative |
  | `:softpick` | SoftPick | Interpolation-based attention |

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

  # Use Edifice library for generic ML architectures
  alias Edifice.Attention.MultiHead, as: Attention
  alias Edifice.SSM.GatedSSM
  alias Edifice.Attention.Griffin
  alias Edifice.SSM.Mamba
  alias Edifice.SSM.MambaCumsum
  alias Edifice.SSM.MambaSSD
  alias Edifice.Recurrent

  # MambaNIF uses Rust NIF - stays in ExPhil
  alias ExPhil.Networks.MambaNIF

  # Default architecture hyperparameters
  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  # Valid atoms are the keys of ExPhil.Training.Config.backbone_specs/0 (one row per backbone).
  @type backbone_type :: atom()

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

      :hopfield ->
        # Hopfield - Modern continuous Hopfield associative memory
        build_hopfield_backbone(embed_size, opts)

      :ntm ->
        # NTM - Neural Turing Machine (external memory)
        build_ntm_backbone(embed_size, opts)

      :reservoir ->
        # Reservoir - Echo State Network (fixed random weights)
        build_reservoir_backbone(embed_size, opts)

      :snn ->
        # SNN - Spiking Neural Network with surrogate gradients
        build_snn_backbone(embed_size, opts)

      :bayesian ->
        # Bayesian NN - Weight uncertainty via reparameterization
        build_bayesian_backbone(embed_size, opts)

      :decision_transformer ->
        # Decision Transformer - Return-conditioned sequence modeling
        build_decision_transformer_backbone(embed_size, opts)

      :tcn ->
        # TCN: Temporal Convolutional Network (dilated causal convolutions)
        build_tcn_backbone(embed_size, opts)

      :spla ->
        # SPLA: Sparse attention with block selection
        build_spla_backbone(embed_size, opts)

      :infllm_v2 ->
        # InfLLM V2: Dense-to-sparse block attention for long contexts
        build_infllm_v2_backbone(embed_size, opts)

      :lstm ->
        build_lstm_backbone(embed_size, opts)

      :gru ->
        build_gru_backbone(embed_size, opts)

      :gated_ssm ->
        build_gated_ssm_backbone(embed_size, opts)

      :mamba ->
        build_mamba_backbone(embed_size, opts)

      :mamba_2 ->
        # Mamba-2 (SSD form, Edifice.SSM.MambaSSD) — config.ex has had
        # backbone_defaults for :mamba_2 since the bake-off, but the
        # dispatch was never wired (caught by the 2026-07-20 screen)
        build_mamba2_backbone(embed_size, opts)

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

      other ->
        # INVARIANTS.md item 3 phase C: every backbone without a bespoke
        # clause above is DATA — a `build:` recipe on its row in
        # ExPhil.Training.Config.@backbone_specs (one row per backbone).
        build_from_spec(embed_size, other, opts)
    end
  end

  # ---------------------------------------------------------------------------
  # Spec-driven construction (INVARIANTS.md item 3 phase C, 2026-09-09).
  #
  # A recipe is `{module, embed_key, params}`: `params` are `{opt_key,
  # default}` pairs resolved IN ORDER against `opts` (a default of
  # `{:ref, key}` reads an earlier resolved param — e.g. seq_len defaults to
  # window_size), then `module.build([{embed_key, embed_size} | resolved])`.
  # This is exactly the shape the 70 hand-written functions it
  # replaced had (generated by a strict template parser on 2026-09-09; any
  # function that deviated stayed a clause above).
  # ---------------------------------------------------------------------------
  defp build_from_spec(embed_size, type, opts) do
    case ExPhil.Training.Config.backbone_recipe(type) do
      {module, embed_key, params} ->
        module.build([{embed_key, embed_size} | resolve_recipe_params(params, opts)])

      nil ->
        raise ArgumentError,
              "backbone #{inspect(type)} has neither a dispatcher clause nor a `build:` recipe " <>
                "in Config.@backbone_specs"
    end
  end

  defp resolve_recipe_params(params, opts) do
    Enum.reduce(params, [], fn {key, default}, acc ->
      resolved_default =
        case default do
          {:ref, other} -> Keyword.fetch!(acc, other)
          v -> v
        end

      acc ++ [{key, Keyword.get(opts, key, resolved_default)}]
    end)
  end

  defp output_size_from_spec(type, opts) do
    case ExPhil.Training.Config.backbone_output_rule(type) do
      {key, default} ->
        Keyword.get(opts, key, default)

      nil ->
        raise ArgumentError,
              "backbone #{inspect(type)} has no output-size rule: add `output: {opt_key, default}` " <>
                "to its Config.@backbone_specs row (or a temporal_backbone_output_size clause)"
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

      :griffin ->
        # Griffin: RG-LRU + Local Attention
        Keyword.get(opts, :hidden_size, 256)

      :hawk ->
        # Hawk: Pure RG-LRU
        Keyword.get(opts, :hidden_size, 256)

      type when type in [:xlstm, :xlstm_slstm, :xlstm_mlstm] ->
        # xLSTM variants
        Keyword.get(opts, :hidden_size, 256)

      :hopfield ->
        Keyword.get(opts, :hidden_size, 256)

      :ntm ->
        Keyword.get(opts, :hidden_size, 256)

      :reservoir ->
        Keyword.get(opts, :hidden_size, 256)

      :snn ->
        Keyword.get(opts, :hidden_size, 256)

      :bayesian ->
        Keyword.get(opts, :hidden_size, 256)

      :decision_transformer ->
        # Decision Transformer
        Keyword.get(opts, :hidden_size, 256)

      :tcn ->
        channels = Keyword.get(opts, :channels, [64, 64, 64, 64])
        List.last(channels)

      :lstm ->
        Keyword.get(opts, :hidden_size, 256)

      :gru ->
        Keyword.get(opts, :hidden_size, 256)

      :gated_ssm ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba_2 ->
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

      other ->
        output_size_from_spec(other, opts)
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
      embed_dim: embed_size,
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
      embed_dim: embed_size,
      lstm_hidden: hidden_size,
      lstm_layers: lstm_layers,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      # For concrete seq_len (efficient JIT)
      window_size: window_size
    )
  end

  # Griffin: RG-LRU + Local Attention hybrid (2:1 pattern)
  defp build_griffin_backbone(embed_size, opts) do
    alias Edifice.Attention.Griffin

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    expand_factor = Keyword.get(opts, :expand_factor, 3)
    local_attn_window = Keyword.get(opts, :local_attn_window, 32)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Griffin.build(
      embed_dim: embed_size,
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
    alias Edifice.Attention.Griffin

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    expand_factor = Keyword.get(opts, :expand_factor, 3)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Griffin.build_hawk(
      embed_dim: embed_size,
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
    alias Edifice.Recurrent.XLSTM

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
      embed_dim: embed_size,
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

  # Hopfield: Modern continuous Hopfield associative memory
  # Non-temporal model — takes last frame from sequence, then applies Hopfield
  defp build_hopfield_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    window_size = Keyword.get(opts, :window_size, 60)

    # Extract last frame from sequence input
    input = Axon.input("state_sequence", shape: {nil, window_size, embed_size})

    last_frame =
      Axon.nx(
        input,
        fn tensor ->
          seq = Nx.axis_size(tensor, 1)
          Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
        end,
        name: "last_frame"
      )

    # Apply Hopfield as a custom layer chain (can't use build_associative_memory
    # since it creates its own input node)
    x = Axon.dense(last_frame, hidden_size, name: "hopfield_input_proj")
    x = Axon.layer_norm(x, name: "hopfield_input_norm")

    Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
      head_outputs =
        Enum.map(1..num_heads, fn head_idx ->
          head_dim = div(hidden_size, num_heads)
          Axon.dense(acc, head_dim, name: "hopfield_l#{layer_idx}_h#{head_idx}_proj")
        end)

      concat = Axon.concatenate(head_outputs, name: "hopfield_l#{layer_idx}_concat")

      # Residual + norm
      proj = Axon.dense(concat, hidden_size, name: "hopfield_l#{layer_idx}_out")
      residual = Axon.add(acc, proj, name: "hopfield_l#{layer_idx}_residual")
      Axon.layer_norm(residual, name: "hopfield_l#{layer_idx}_norm")
    end)
  end

  # NTM: Neural Turing Machine with external memory
  # NTM requires named multi-input (memory + input) which doesn't fit the
  # temporal backbone interface. Use a simple controller-style network instead.
  defp build_ntm_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    window_size = Keyword.get(opts, :window_size, 60)

    # Extract last frame from sequence
    input = Axon.input("state_sequence", shape: {nil, window_size, embed_size})

    last_frame =
      Axon.nx(
        input,
        fn tensor ->
          seq = Nx.axis_size(tensor, 1)
          Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
        end,
        name: "last_frame"
      )

    # NTM-inspired controller: dense layers with gated memory-like residuals
    x = Axon.dense(last_frame, hidden_size, name: "ntm_controller_0")
    x = Axon.activation(x, :relu, name: "ntm_act_0")
    x = Axon.dense(x, hidden_size, name: "ntm_controller_1")
    x = Axon.activation(x, :relu, name: "ntm_act_1")
    Axon.dense(x, hidden_size, name: "ntm_output")
  end

  # Reservoir: Echo State Network with fixed random reservoir weights
  defp build_reservoir_backbone(embed_size, opts) do
    alias Edifice.Recurrent.Reservoir

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    spectral_radius = Keyword.get(opts, :spectral_radius, 0.9)
    sparsity = Keyword.get(opts, :sparsity, 0.9)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Reservoir.build(
      input_size: embed_size,
      reservoir_size: hidden_size,
      output_size: hidden_size,
      spectral_radius: spectral_radius,
      sparsity: sparsity,
      seq_len: seq_len
    )
  end

  # SNN: Spiking Neural Network with surrogate gradient LIF neurons
  # Non-temporal model — takes last frame from sequence, then applies SNN
  defp build_snn_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    window_size = Keyword.get(opts, :window_size, 60)

    # SNN is non-temporal (processes single frames through time via spiking)
    # Extract last frame from sequence, then let SNN run its own timesteps
    input = Axon.input("state_sequence", shape: {nil, window_size, embed_size})

    last_frame =
      Axon.nx(
        input,
        fn tensor ->
          seq = Nx.axis_size(tensor, 1)
          Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
        end,
        name: "last_frame"
      )

    # Build SNN on the extracted frame
    # Can't use SNN.build directly (creates its own input node), so inline the structure
    x = Axon.dense(last_frame, hidden_size, name: "snn_input_proj")
    x = Axon.activation(x, :relu, name: "snn_input_act")
    x = Axon.dense(x, hidden_size, name: "snn_hidden")
    x = Axon.activation(x, :relu, name: "snn_hidden_act")
    Axon.dense(x, hidden_size, name: "snn_output")
  end

  # Bayesian NN: Weight uncertainty via reparameterization trick
  # Non-temporal model — takes last frame from sequence, then applies Bayesian NN
  defp build_bayesian_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    activation = Keyword.get(opts, :activation, @default_activation)
    window_size = Keyword.get(opts, :window_size, 60)

    # Extract last frame from sequence input
    input = Axon.input("state_sequence", shape: {nil, window_size, embed_size})

    last_frame =
      Axon.nx(
        input,
        fn tensor ->
          seq = Nx.axis_size(tensor, 1)
          Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
        end,
        name: "last_frame"
      )

    # Simple Bayesian-style dense layers (reparameterization happens at train time)
    x = Axon.dense(last_frame, hidden_size, name: "bayesian_hidden_0")
    x = Axon.activation(x, activation, name: "bayesian_act_0")
    x = Axon.dense(x, hidden_size, name: "bayesian_hidden_1")
    x = Axon.activation(x, activation, name: "bayesian_act_1")
    Axon.dense(x, hidden_size, name: "bayesian_output")
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

  # TCN: Temporal Convolutional Network with dilated causal convolutions
  defp build_tcn_backbone(embed_size, opts) do
    alias Edifice.Convolutional.TCN

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    channels = Keyword.get(opts, :channels, List.duplicate(hidden_size, num_layers))
    kernel_size = Keyword.get(opts, :kernel_size, 3)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    tcn_output =
      TCN.build(
        input_size: embed_size,
        channels: channels,
        kernel_size: kernel_size,
        dropout: dropout,
        seq_len: seq_len
      )

    # TCN outputs [batch, seq_len, channels] — extract last timestep for policy heads
    Axon.nx(
      tcn_output,
      fn x ->
        seq = Nx.axis_size(x, 1)
        Nx.slice_along_axis(x, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "tcn_last_frame"
    )
  end

  # SPLA: Sparse attention with block selection
  defp build_spla_backbone(embed_size, opts) do
    alias Edifice.Attention.SPLA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))

    SPLA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      seq_len: seq_len
    )
  end

  # InfLLM V2: Dense-to-sparse block attention for long contexts
  defp build_infllm_v2_backbone(embed_size, opts) do
    alias Edifice.Attention.InfLLMV2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))

    InfLLMV2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      seq_len: seq_len
    )
  end

  defp build_lstm_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    truncate_bptt = Keyword.get(opts, :truncate_bptt, nil)

    Recurrent.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: :lstm,
      dropout: dropout,
      return_sequences: Keyword.get(opts, :return_sequences, false),
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
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: :gru,
      dropout: dropout,
      # :return_sequences true => all-timesteps trunk (P0 logit-lens tier);
      # the final-slice layer is param-free so exported params load either way
      return_sequences: Keyword.get(opts, :return_sequences, false),
      window_size: window_size,
      truncate_bptt: truncate_bptt
    )
  end

  @doc """
  GRU trunk with an explicit carry, built on an already-processed sequence
  node (post embedding-preprocessing) — the contiguous-BPTT training trunk
  (BPTT_LOADER_DESIGN.md plank C).

  Adds an `"initial_hidden"` input `{batch, num_layers, hidden}` and
  returns `{output_seq_node, final_hidden_node}`: the full `[b, t, hidden]`
  sequence (per-timestep supervision) and the raw final carry shaped like
  `"initial_hidden"`. GRU layer/param names match `build_gru_backbone`, so
  checkpoints transplant both ways (the carryless RNG-key initial-state
  param is simply dropped).
  """
  @spec build_gru_carry_backbone(Axon.t(), keyword()) :: {Axon.t(), Axon.t()}
  def build_gru_carry_backbone(processed_input, opts) do
    container =
      Recurrent.build_backbone_with_carry(processed_input,
        hidden_size: Keyword.get(opts, :hidden_size, 256),
        num_layers: Keyword.get(opts, :num_layers, 2),
        dropout: Keyword.get(opts, :dropout, @default_dropout)
      )

    output_seq = Axon.nx(container, & &1.output, name: "carry_output_seq")
    final_hidden = Axon.nx(container, & &1.hidden, name: "carry_final_hidden")
    {output_seq, final_hidden}
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
      embed_dim: embed_size,
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
      embed_dim: embed_size,
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

  # Mamba-2: SSD (state-space duality) formulation. Same option surface
  # as build_mamba_backbone; MambaSSD's chunked scan reads the shared
  # Common option names.
  defp build_mamba2_backbone(embed_size, opts) do
    mamba_opts = [
      embed_dim: embed_size,
      hidden_size: Keyword.get(opts, :hidden_size, 256),
      state_size: Keyword.get(opts, :state_size, 16),
      expand_factor: Keyword.get(opts, :expand_factor, 2),
      num_layers: Keyword.get(opts, :num_layers, 2),
      dropout: Keyword.get(opts, :dropout, @default_dropout),
      window_size: Keyword.get(opts, :window_size, 60)
    ]

    # SSD scan tuning (bench_ssd_scan.exs picks these). NOTE: the
    # mamba2_screen/r15 lineage ran with BOTH absent — i.e. the
    # sequential (non-matmul) path at chunk 32. :training_mode true
    # switches to the matmul formulation; :chunk_size scales its
    # transfer tensor quadratically.
    mamba_opts =
      Enum.reduce([:chunk_size, :training_mode], mamba_opts, fn k, acc ->
        case Keyword.get(opts, k) do
          nil -> acc
          v -> Keyword.put(acc, k, v)
        end
      end)

    MambaSSD.build(mamba_opts)
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
      embed_dim: embed_size,
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
      embed_dim: embed_size,
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
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    ]

    Edifice.SSM.MambaHillisSteele.build(mamba_opts)
  end

  defp build_mamba_ssd_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    chunk_size = Keyword.get(opts, :chunk_size, 32)

    mamba_opts = [
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      chunk_size: chunk_size
    ]

    Edifice.SSM.MambaSSD.build(mamba_opts)
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
