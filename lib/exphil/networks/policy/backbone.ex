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
  alias Edifice.SSM.Hybrid
  alias Edifice.SSM.Mamba
  alias Edifice.SSM.MambaCumsum
  alias Edifice.Recurrent

  # MambaNIF uses Rust NIF - stays in ExPhil
  alias ExPhil.Networks.MambaNIF

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
          | :s4
          | :s4d
          | :h3
          | :performer
          | :deltanet
          | :fnet
          | :perceiver
          | :ttt
          | :hopfield
          | :ntm
          | :reservoir
          | :snn
          | :bayesian
          | :decision_transformer
          | :liquid
          | :kan
          | :transformer_like
          | :deep_res_lstm
          | :min_gru
          | :min_lstm
          | :tcn
          | :mamba3
          | :hyena
          | :titans
          | :gated_deltanet
          | :native_recurrence
          | :longhorn
          | :samba
          | :hymba
          | :gss
          | :delta_product
          | :gla_v2
          | :hgrn_v2
          | :ttt_e2e
          | :gsa
          | :rla
          | :nha
          | :fox
          | :log_linear
          | :laser
          | :moba
          | :tnn
          | :miras
          | :mixture_of_mamba
          | :huginn
          | :coconut
          | :mega
          | :based
          | :infini_attention
          | :conformer
          | :mla
          | :diff_transformer
          | :megalodon
          | :lightning_attention
          | :flash_linear_attention
          | :kda
          | :sigmoid_attention
          | :spla
          | :retnet_v2
          | :rnope_swa
          | :nsa
          | :infllm_v2
          | :dual_chunk_attention
          | :gated_attention
          | :mta
          | :slstm
          | :xlstm_v2
          | :bimamba
          | :hyena_v2
          | :ss_transformer
          | :ssmax
          | :softpick

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

      :s4 ->
        # S4 - Structured State Spaces (Blelloch parallel scan)
        build_s4_backbone(embed_size, opts)

      :s4d ->
        # S4D - S4 with diagonal state matrix (faster, simpler)
        build_s4d_backbone(embed_size, opts)

      :h3 ->
        # H3 - Hungry Hungry Hippos (SSM + shift + conv)
        build_h3_backbone(embed_size, opts)

      :performer ->
        # Performer - FAVOR+ linear attention approximation
        build_performer_backbone(embed_size, opts)

      :deltanet ->
        # DeltaNet - Linear attention with delta rule updates
        build_deltanet_backbone(embed_size, opts)

      :fnet ->
        # FNet - Fourier transform token mixing (no attention)
        build_fnet_backbone(embed_size, opts)

      :perceiver ->
        # Perceiver IO - Latent bottleneck cross-attention
        build_perceiver_backbone(embed_size, opts)

      :ttt ->
        # TTT - Test-Time Training (self-supervised inner loop)
        build_ttt_backbone(embed_size, opts)

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

      :liquid ->
        # Liquid Neural Networks - Continuous-time adaptive dynamics
        build_liquid_backbone(embed_size, opts)

      :kan ->
        # KAN: Kolmogorov-Arnold Networks with learnable activations
        build_kan_backbone(embed_size, opts)

      :transformer_like ->
        # TransformerLike: Alternating LSTM/GRU + FFN residual blocks (slippi-ai's tx_like)
        build_transformer_like_backbone(embed_size, opts)

      :deep_res_lstm ->
        # DeepResLSTM: Stacked residual LSTM blocks (slippi-ai's res_lstm)
        build_deep_res_lstm_backbone(embed_size, opts)

      :min_gru ->
        # MinGRU: Minimal GRU, parallel-scannable (faster than standard GRU)
        build_min_gru_backbone(embed_size, opts)

      :min_lstm ->
        # MinLSTM: Minimal LSTM, parallel-scannable (faster than standard LSTM)
        build_min_lstm_backbone(embed_size, opts)

      :tcn ->
        # TCN: Temporal Convolutional Network (dilated causal convolutions)
        build_tcn_backbone(embed_size, opts)

      :mamba3 ->
        # Mamba-3: Complex states, trapezoidal discretization, MIMO
        build_mamba3_backbone(embed_size, opts)

      :hyena ->
        # Hyena: Long convolution hierarchy with implicit filters
        build_hyena_backbone(embed_size, opts)

      :titans ->
        # Titans: Neural long-term memory with surprise gating
        build_titans_backbone(embed_size, opts)

      :gated_deltanet ->
        # Gated DeltaNet: Linear attention with data-dependent gating
        build_gated_deltanet_backbone(embed_size, opts)

      :native_recurrence ->
        # NativeRecurrence: Efficient GRU variants (elu_gru, real_gru, diag_linear)
        build_native_recurrence_backbone(embed_size, opts)

      :longhorn ->
        # Longhorn: Drop-in Mamba replacement, no forget gate
        build_longhorn_backbone(embed_size, opts)

      :samba ->
        # Samba: Hybrid Mamba + Sliding Window Attention + MLP
        build_samba_backbone(embed_size, opts)

      :hymba ->
        # Hymba: Parallel Mamba + Attention with meta tokens
        build_hymba_backbone(embed_size, opts)

      :gss ->
        # GSS: Gated State Space model
        build_gss_backbone(embed_size, opts)

      :delta_product ->
        # DeltaProduct: Multi-step DeltaNet via Householder products
        build_delta_product_backbone(embed_size, opts)

      :gla_v2 ->
        # GLAv2: Gated Linear Attention v2 (improved GLA)
        build_gla_v2_backbone(embed_size, opts)

      :hgrn_v2 ->
        # HGRNv2: HGRN v2 with outer product state expansion
        build_hgrn_v2_backbone(embed_size, opts)

      :ttt_e2e ->
        # TTT-E2E: End-to-end Test-Time Training
        build_ttt_e2e_backbone(embed_size, opts)

      :gsa ->
        # GSA: Gated Slot Attention (linear time, fixed slots)
        build_gsa_backbone(embed_size, opts)

      :rla ->
        # RLA: Residual Linear Attention
        build_rla_backbone(embed_size, opts)

      :nha ->
        # NHA: Native Hybrid Attention (per-layer linear vs full)
        build_nha_backbone(embed_size, opts)

      :fox ->
        # FoX: Forgetting Transformer with learnable forget on softmax
        build_fox_backbone(embed_size, opts)

      :log_linear ->
        # LogLinear: O(log T) space hierarchical attention
        build_log_linear_backbone(embed_size, opts)

      :laser ->
        # LASER: Log-exponential attention for larger gradients
        build_laser_backbone(embed_size, opts)

      :moba ->
        # MoBA: Mixture of Block Attention (production-proven)
        build_moba_backbone(embed_size, opts)

      :tnn ->
        # TNN: Toeplitz Neural Network, O(n log n) with good extrapolation
        build_tnn_backbone(embed_size, opts)

      :miras ->
        # MIRAS: Memory as Iterative Reasoning (Moneta, Yaad, Memora)
        build_miras_backbone(embed_size, opts)

      :mixture_of_mamba ->
        # MixtureOfMamba: Modality-aware Mamba with sparse routing
        build_mixture_of_mamba_backbone(embed_size, opts)

      :huginn ->
        # Huginn: Depth-recurrent transformer with adaptive iteration
        build_huginn_backbone(embed_size, opts)

      :coconut ->
        # Coconut: Continuous chain of thought (latent reasoning)
        build_coconut_backbone(embed_size, opts)

      :mega ->
        # MEGA: Multi-scale EMA + gated attention
        build_mega_backbone(embed_size, opts)

      :based ->
        # Based: Taylor-expansion linear attention
        build_based_backbone(embed_size, opts)

      :infini_attention ->
        # InfiniAttention: Compressive memory for long sequences
        build_infini_attention_backbone(embed_size, opts)

      :conformer ->
        # Conformer: Conv + Attention (audio/temporal patterns)
        build_conformer_backbone(embed_size, opts)

      :mla ->
        # MLA: Multi-head Latent Attention (DeepSeek-style KV compression)
        build_mla_backbone(embed_size, opts)

      :diff_transformer ->
        # DiffTransformer: Differential attention (noise cancellation)
        build_diff_transformer_backbone(embed_size, opts)

      :megalodon ->
        # Megalodon: Chunk-wise attention with complex EMA
        build_megalodon_backbone(embed_size, opts)

      :lightning_attention ->
        # LightningAttention: O(n) linear attention, very fast
        build_lightning_attention_backbone(embed_size, opts)

      :flash_linear_attention ->
        # FlashLinearAttention: Hardware-efficient chunked linear attention
        build_flash_linear_attention_backbone(embed_size, opts)

      :kda ->
        # KDA: Key-Driven channel-decay Attention
        build_kda_backbone(embed_size, opts)

      :sigmoid_attention ->
        # SigmoidAttention: Sigmoid instead of softmax (no normalization)
        build_sigmoid_attention_backbone(embed_size, opts)

      :spla ->
        # SPLA: Sparse attention with block selection
        build_spla_backbone(embed_size, opts)

      :retnet_v2 ->
        # RetNet V2: Improved retentive network
        build_retnet_v2_backbone(embed_size, opts)

      :rnope_swa ->
        # RNoPE-SWA: Sliding window attention without positional encoding
        build_rnope_swa_backbone(embed_size, opts)

      :nsa ->
        # NSA: Native Sparse Attention (compression + selection + sliding)
        build_nsa_backbone(embed_size, opts)

      :infllm_v2 ->
        # InfLLM V2: Dense-to-sparse block attention for long contexts
        build_infllm_v2_backbone(embed_size, opts)

      :dual_chunk_attention ->
        # DualChunkAttention: Intra-chunk + inter-chunk attention
        build_dual_chunk_attention_backbone(embed_size, opts)

      :gated_attention ->
        # GatedAttention: Per-dimension gating on attention output
        build_gated_attention_backbone(embed_size, opts)

      :mta ->
        # MTA: Multi-Token Attention with depthwise convolutions
        build_mta_backbone(embed_size, opts)

      :slstm ->
        # sLSTM: Scalar LSTM with exponential gating (standalone)
        build_slstm_backbone(embed_size, opts)

      :xlstm_v2 ->
        # xLSTM V2: Updated xLSTM with block-diagonal memory
        build_xlstm_v2_backbone(embed_size, opts)

      :bimamba ->
        # BiMamba: Bidirectional Mamba (offline analysis only)
        build_bimamba_backbone(embed_size, opts)

      :hyena_v2 ->
        # Hyena V2: Improved long convolution hierarchy
        build_hyena_v2_backbone(embed_size, opts)

      :ss_transformer ->
        # SSTransformer: Parallel SSM + Attention with gating
        build_ss_transformer_backbone(embed_size, opts)

      :ssmax ->
        # SSMax: State-space inspired softmax alternative
        build_ssmax_backbone(embed_size, opts)

      :softpick ->
        # SoftPick: Interpolation-based attention (between softmax and hardmax)
        build_softpick_backbone(embed_size, opts)

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

      :s4 ->
        Keyword.get(opts, :hidden_size, 256)

      :s4d ->
        Keyword.get(opts, :hidden_size, 256)

      :h3 ->
        Keyword.get(opts, :hidden_size, 256)

      :performer ->
        Keyword.get(opts, :hidden_size, 256)

      :deltanet ->
        Keyword.get(opts, :hidden_size, 256)

      :fnet ->
        Keyword.get(opts, :hidden_size, 256)

      :perceiver ->
        Keyword.get(opts, :latent_dim, 256)

      :ttt ->
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

      :liquid ->
        # Liquid Neural Networks
        Keyword.get(opts, :hidden_size, 256)

      :kan ->
        # KAN: Kolmogorov-Arnold Networks
        Keyword.get(opts, :hidden_size, 256)

      :transformer_like ->
        # TransformerLike (slippi-ai's tx_like): defaults to 512 to match slippi-ai
        Keyword.get(opts, :hidden_size, 512)

      :deep_res_lstm ->
        # DeepResLSTM (slippi-ai's res_lstm): defaults to 512 to match slippi-ai
        Keyword.get(opts, :hidden_size, 512)

      :min_gru ->
        Keyword.get(opts, :hidden_size, 256)

      :min_lstm ->
        Keyword.get(opts, :hidden_size, 256)

      :tcn ->
        channels = Keyword.get(opts, :channels, [64, 64, 64, 64])
        List.last(channels)

      :mamba3 ->
        Keyword.get(opts, :hidden_size, 256)

      :hyena ->
        Keyword.get(opts, :hidden_size, 256)

      :titans ->
        Keyword.get(opts, :hidden_size, 256)

      :gated_deltanet ->
        Keyword.get(opts, :hidden_size, 256)

      type
      when type in [
             :native_recurrence,
             :longhorn,
             :samba,
             :hymba,
             :gss,
             :delta_product,
             :gla_v2,
             :hgrn_v2,
             :ttt_e2e,
             :gsa,
             :rla,
             :nha,
             :fox,
             :log_linear,
             :laser,
             :moba,
             :tnn,
             :miras,
             :mixture_of_mamba,
             :huginn,
             :coconut,
             :mega,
             :based,
             :infini_attention,
             :conformer,
             :mla,
             :diff_transformer,
             :megalodon,
             :lightning_attention,
             :flash_linear_attention,
             :kda,
             :sigmoid_attention,
             :spla,
             :retnet_v2,
             :rnope_swa,
             :nsa,
             :infllm_v2,
             :dual_chunk_attention,
             :gated_attention,
             :mta,
             :slstm,
             :xlstm_v2,
             :bimamba,
             :hyena_v2,
             :ss_transformer,
             :ssmax,
             :softpick
           ] ->
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
      embed_dim: embed_size,
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
    alias Edifice.SSM.Zamba

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
      embed_dim: embed_size,
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

  # RetNet: Retentive Network with multi-scale retention (Microsoft, 2023)
  defp build_retnet_backbone(embed_size, opts) do
    alias Edifice.Attention.RetNet

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, 4)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RetNet.build(
      embed_dim: embed_size,
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
    alias Edifice.Attention.RWKV

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    head_size = Keyword.get(opts, :head_size, 64)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RWKV.build(
      embed_dim: embed_size,
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
    alias Edifice.Attention.GLA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GLA.build(
      embed_dim: embed_size,
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
    alias Edifice.Attention.HGRN

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    state_expansion = Keyword.get(opts, :state_expansion, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    HGRN.build(
      embed_dim: embed_size,
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
    alias Edifice.SSM.S5

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    S5.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # S4: Structured State Spaces for Sequences (Blelloch parallel scan)
  defp build_s4_backbone(embed_size, opts) do
    alias Edifice.SSM.S4

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    S4.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # S4D: S4 with diagonal state matrix (faster, simpler)
  defp build_s4d_backbone(embed_size, opts) do
    alias Edifice.SSM.S4D

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    S4D.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # H3: Hungry Hungry Hippos (SSM + shift + convolution)
  defp build_h3_backbone(embed_size, opts) do
    alias Edifice.SSM.H3

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 64)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    H3.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Performer: FAVOR+ linear attention approximation
  defp build_performer_backbone(embed_size, opts) do
    alias Edifice.Attention.Performer

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_features = Keyword.get(opts, :num_features, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Performer.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_features: num_features,
      num_layers: num_layers,
      num_heads: num_heads,
      dropout: dropout,
      window_size: window_size
    )
  end

  # DeltaNet: Linear attention with delta rule updates
  defp build_deltanet_backbone(embed_size, opts) do
    alias Edifice.Recurrent.DeltaNet

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    DeltaNet.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # FNet: Fourier transform token mixing (no attention parameters)
  defp build_fnet_backbone(embed_size, opts) do
    alias Edifice.Attention.FNet

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    FNet.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # Perceiver IO: Latent bottleneck with cross-attention
  defp build_perceiver_backbone(embed_size, opts) do
    alias Edifice.Attention.Perceiver

    latent_dim = Keyword.get(opts, :latent_dim, 256)
    num_latents = Keyword.get(opts, :num_latents, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    num_cross_layers = Keyword.get(opts, :num_cross_layers, 1)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    Perceiver.build(
      input_dim: embed_size,
      latent_dim: latent_dim,
      num_latents: num_latents,
      num_layers: num_layers,
      num_cross_layers: num_cross_layers,
      num_heads: num_heads,
      dropout: dropout
    )
  end

  # TTT: Test-Time Training with self-supervised inner loop
  defp build_ttt_backbone(embed_size, opts) do
    alias Edifice.Recurrent.TTT

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    inner_size = Keyword.get(opts, :inner_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    TTT.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      inner_size: inner_size,
      num_layers: num_layers,
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
      Axon.nx(input, fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end, name: "last_frame")

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
      Axon.nx(input, fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end, name: "last_frame")

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
      Axon.nx(input, fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end, name: "last_frame")

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
      Axon.nx(input, fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end, name: "last_frame")

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

  # Liquid Neural Networks: Continuous-time adaptive dynamics
  defp build_liquid_backbone(embed_size, opts) do
    alias Edifice.Liquid

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    integration_steps = Keyword.get(opts, :integration_steps, 1)
    solver = Keyword.get(opts, :solver, :exact)

    Liquid.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len,
      integration_steps: integration_steps,
      solver: solver
    )
  end

  defp build_kan_backbone(embed_size, opts) do
    alias Edifice.Feedforward.KAN

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    grid_size = Keyword.get(opts, :grid_size, 8)
    basis = Keyword.get(opts, :basis, :sine)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    KAN.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      grid_size: grid_size,
      basis: basis,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # TransformerLike: Alternating LSTM/GRU + FFN residual blocks (slippi-ai's tx_like)
  defp build_transformer_like_backbone(embed_size, opts) do
    alias Edifice.Recurrent.TransformerLike

    hidden_size = Keyword.get(opts, :hidden_size, 512)
    num_layers = Keyword.get(opts, :num_layers, 3)
    cell_type = Keyword.get(opts, :cell_type, :lstm)
    ffn_multiplier = Keyword.get(opts, :ffn_multiplier, 2)
    activation = Keyword.get(opts, :activation, :gelu)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, :layer_norm)
    recurrent_norm = Keyword.get(opts, :recurrent_norm, false)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    TransformerLike.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: cell_type,
      ffn_multiplier: ffn_multiplier,
      activation: activation,
      dropout: dropout,
      norm: norm,
      recurrent_norm: recurrent_norm,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # DeepResLSTM: Stacked residual LSTM blocks (slippi-ai's res_lstm)
  defp build_deep_res_lstm_backbone(embed_size, opts) do
    alias Edifice.Recurrent.DeepResLSTM

    hidden_size = Keyword.get(opts, :hidden_size, 512)
    num_layers = Keyword.get(opts, :num_layers, 3)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, :layer_norm)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    DeepResLSTM.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      norm: norm,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MinGRU: Minimal GRU with parallel scan (Feng et al., 2024)
  defp build_min_gru_backbone(embed_size, opts) do
    alias Edifice.Recurrent.MinGRU

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MinGRU.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MinLSTM: Minimal LSTM with parallel scan (Feng et al., 2024)
  defp build_min_lstm_backbone(embed_size, opts) do
    alias Edifice.Recurrent.MinLSTM

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MinLSTM.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
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
    Axon.nx(tcn_output, fn x ->
      seq = Nx.axis_size(x, 1)
      Nx.slice_along_axis(x, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
    end, name: "tcn_last_frame")
  end

  # Mamba-3: Complex states, trapezoidal discretization, MIMO
  defp build_mamba3_backbone(embed_size, opts) do
    alias Edifice.SSM.Mamba3

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Mamba3.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # Hyena: Long convolution hierarchy with implicit filters
  defp build_hyena_backbone(embed_size, opts) do
    alias Edifice.SSM.Hyena

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    order = Keyword.get(opts, :order, 2)
    filter_size = Keyword.get(opts, :filter_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Hyena.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      order: order,
      filter_size: filter_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Titans: Neural long-term memory with surprise-gated updates
  defp build_titans_backbone(embed_size, opts) do
    alias Edifice.Recurrent.Titans

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    memory_size = Keyword.get(opts, :memory_size, 64)
    num_layers = Keyword.get(opts, :num_layers, 4)
    momentum = Keyword.get(opts, :momentum, 0.9)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Titans.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      memory_size: memory_size,
      num_layers: num_layers,
      momentum: momentum,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Gated DeltaNet: Linear attention with data-dependent gating
  defp build_gated_deltanet_backbone(embed_size, opts) do
    alias Edifice.Recurrent.GatedDeltaNet

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    conv_size = Keyword.get(opts, :conv_size, 4)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GatedDeltaNet.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      conv_size: conv_size,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # NativeRecurrence: Efficient GRU variants from NativeRecurrence paper
  defp build_native_recurrence_backbone(embed_size, opts) do
    alias Edifice.Recurrent.NativeRecurrence

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    NativeRecurrence.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Longhorn: Drop-in Mamba replacement, closed-form online recall
  defp build_longhorn_backbone(embed_size, opts) do
    alias Edifice.SSM.Longhorn

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Longhorn.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Samba: Hybrid Mamba + Sliding Window Attention + MLP
  defp build_samba_backbone(embed_size, opts) do
    alias Edifice.SSM.Samba

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Samba.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Hymba: Parallel Mamba + Attention with meta tokens
  defp build_hymba_backbone(embed_size, opts) do
    alias Edifice.SSM.Hymba

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Hymba.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # GSS: Gated State Space model
  defp build_gss_backbone(embed_size, opts) do
    alias Edifice.SSM.GSS

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GSS.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # DeltaProduct: Multi-step DeltaNet via Householder products
  defp build_delta_product_backbone(embed_size, opts) do
    alias Edifice.Recurrent.DeltaProduct

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    DeltaProduct.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # GLAv2: Gated Linear Attention v2
  defp build_gla_v2_backbone(embed_size, opts) do
    alias Edifice.Attention.GLAv2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GLAv2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # HGRNv2: Hierarchically Gated RNN v2 with outer product state
  defp build_hgrn_v2_backbone(embed_size, opts) do
    alias Edifice.Attention.HGRNv2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    HGRNv2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # TTT-E2E: End-to-end Test-Time Training
  defp build_ttt_e2e_backbone(embed_size, opts) do
    alias Edifice.Recurrent.TTTE2E

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    TTTE2E.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # GSA: Gated Slot Attention (linear time, fixed slots)
  defp build_gsa_backbone(embed_size, opts) do
    alias Edifice.Attention.GSA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    GSA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # RLA: Residual Linear Attention
  defp build_rla_backbone(embed_size, opts) do
    alias Edifice.Attention.RLA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RLA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # NHA: Native Hybrid Attention (per-layer linear vs full selection)
  defp build_nha_backbone(embed_size, opts) do
    alias Edifice.Attention.NHA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    NHA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # FoX: Forgetting Transformer with learnable forget on softmax
  defp build_fox_backbone(embed_size, opts) do
    alias Edifice.Attention.FoX

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    FoX.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # LogLinear: O(log T) space hierarchical attention
  defp build_log_linear_backbone(embed_size, opts) do
    alias Edifice.Attention.LogLinear

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    LogLinear.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # LASER: Log-exponential attention for larger gradients
  defp build_laser_backbone(embed_size, opts) do
    alias Edifice.Attention.LASER

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    LASER.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MoBA: Mixture of Block Attention (production-proven, Kimi)
  defp build_moba_backbone(embed_size, opts) do
    alias Edifice.Attention.MoBA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MoBA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # TNN: Toeplitz Neural Network, O(n log n) with good extrapolation
  defp build_tnn_backbone(embed_size, opts) do
    alias Edifice.Attention.TNN

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    TNN.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MIRAS: Memory as Iterative Reasoning (Moneta, Yaad, Memora variants)
  defp build_miras_backbone(embed_size, opts) do
    alias Edifice.Recurrent.MIRAS

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MIRAS.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MixtureOfMamba: Modality-aware Mamba with sparse routing
  defp build_mixture_of_mamba_backbone(embed_size, opts) do
    alias Edifice.SSM.MixtureOfMamba

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MixtureOfMamba.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Huginn: Depth-recurrent transformer with adaptive iteration
  defp build_huginn_backbone(embed_size, opts) do
    alias Edifice.Recurrent.Huginn

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Huginn.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Coconut: Continuous chain of thought (latent reasoning)
  defp build_coconut_backbone(embed_size, opts) do
    alias Edifice.Meta.Coconut

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Coconut.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # MEGA: Multi-scale EMA + gated attention
  defp build_mega_backbone(embed_size, opts) do
    alias Edifice.Attention.Mega

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Mega.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Based: Taylor-expansion linear attention
  defp build_based_backbone(embed_size, opts) do
    alias Edifice.Attention.Based

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Based.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # InfiniAttention: Compressive memory for long sequences
  defp build_infini_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.InfiniAttention

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    segment_size = Keyword.get(opts, :segment_size, 32)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    InfiniAttention.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      segment_size: segment_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # Conformer: Conv + Attention (good for temporal patterns)
  defp build_conformer_backbone(embed_size, opts) do
    alias Edifice.Attention.Conformer

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, 31)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Conformer.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      conv_kernel_size: conv_kernel_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # MLA: Multi-head Latent Attention (DeepSeek-style KV compression)
  defp build_mla_backbone(embed_size, opts) do
    alias Edifice.Attention.MLA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    MLA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # DiffTransformer: Differential attention (noise cancellation)
  defp build_diff_transformer_backbone(embed_size, opts) do
    alias Edifice.Attention.DiffTransformer

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    DiffTransformer.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # Megalodon: Chunk-wise attention with complex EMA
  defp build_megalodon_backbone(embed_size, opts) do
    alias Edifice.Attention.Megalodon

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    Megalodon.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # LightningAttention: O(n) linear attention, very fast
  defp build_lightning_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.LightningAttention

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    LightningAttention.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # FlashLinearAttention: Hardware-efficient chunked linear attention
  defp build_flash_linear_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.FlashLinearAttention

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    FlashLinearAttention.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # KDA: Key-Driven channel-decay Attention
  defp build_kda_backbone(embed_size, opts) do
    alias Edifice.Attention.KDA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    KDA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # SigmoidAttention: Sigmoid instead of softmax (no normalization)
  defp build_sigmoid_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.SigmoidAttention

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    SigmoidAttention.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
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

  # RetNet V2: Improved retentive network
  defp build_retnet_v2_backbone(embed_size, opts) do
    alias Edifice.Attention.RetNetV2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    RetNetV2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # RNoPE-SWA: Sliding window attention without positional encoding
  defp build_rnope_swa_backbone(embed_size, opts) do
    alias Edifice.Attention.RNoPESWA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    RNoPESWA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # NSA: Native Sparse Attention (compression + selection + sliding)
  defp build_nsa_backbone(embed_size, opts) do
    alias Edifice.Attention.NSA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    NSA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
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

  # DualChunkAttention: Intra-chunk + inter-chunk attention
  defp build_dual_chunk_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.DualChunk

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    DualChunk.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout
    )
  end

  # GatedAttention: Per-dimension gating on attention output
  defp build_gated_attention_backbone(embed_size, opts) do
    alias Edifice.Attention.GatedAttention

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    GatedAttention.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # MTA: Multi-Token Attention with depthwise convolutions on Q/K/heads
  defp build_mta_backbone(embed_size, opts) do
    alias Edifice.Attention.MTA

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    MTA.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # sLSTM: Scalar LSTM with exponential gating (standalone)
  defp build_slstm_backbone(embed_size, opts) do
    alias Edifice.Recurrent.SLSTM

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    SLSTM.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # xLSTM V2: Updated xLSTM with block-diagonal memory
  defp build_xlstm_v2_backbone(embed_size, opts) do
    alias Edifice.Recurrent.XLSTMv2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    XLSTMv2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # BiMamba: Bidirectional Mamba (offline analysis / replay processing only)
  defp build_bimamba_backbone(embed_size, opts) do
    alias Edifice.SSM.BiMamba

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    BiMamba.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # Hyena V2: Improved long convolution hierarchy
  defp build_hyena_v2_backbone(embed_size, opts) do
    alias Edifice.SSM.HyenaV2

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    HyenaV2.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # SSTransformer: Parallel SSM + Attention with gating
  defp build_ss_transformer_backbone(embed_size, opts) do
    alias Edifice.SSM.SSTransformer

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    SSTransformer.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      seq_len: seq_len
    )
  end

  # SSMax: State-space inspired softmax alternative
  defp build_ssmax_backbone(embed_size, opts) do
    alias Edifice.Blocks.SSMax

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    SSMax.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  # SoftPick: Interpolation-based attention (between softmax and hardmax)
  defp build_softpick_backbone(embed_size, opts) do
    alias Edifice.Blocks.Softpick

    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Softpick.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_heads: num_heads,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
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
      embed_dim: embed_size,
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
