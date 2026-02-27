defmodule ExPhil.Networks.TransformerLike do
  @moduledoc """
  TransformerLike: Hybrid Recurrent + FFN Architecture from slippi-ai.

  Implements the `tx_like` architecture from vladfi1/slippi-ai's `networks.py`.
  This is the production backbone used by slippi-ai's competitive Melee bots.

  ## Key Design

  Alternates LSTM (or GRU) layers with feed-forward residual blocks,
  combining recurrent state tracking with transformer-style depth:

  ```
  Input [batch, seq_len, embed_size]
        │
        ▼
  Dense projection → hidden_size
        │
  Per layer:
    1. x = x + LSTM/GRU(x)     # Residual recurrence
    2. x = x + FFN(LayerNorm(x)) # Pre-norm FFN residual
        │ (repeat num_layers)
        ▼
  Final LayerNorm → last timestep
        │
  [batch, hidden_size]
  ```

  ## Why This Architecture?

  Unlike pure attention (quadratic cost) or pure LSTM (limited depth),
  TransformerLike gets the best of both worlds:
  - **Recurrent state**: Maintains hidden state across frames (critical for 60 FPS)
  - **Depth**: FFN residual blocks add representational capacity like a transformer
  - **Efficiency**: LSTM/GRU layers are O(n) in sequence length

  This is the architecture that produced slippi-ai's strongest bots,
  trained with behavioral cloning + PPO self-play.

  ## Cell Types

  | Cell | Description | Use Case |
  |------|-------------|----------|
  | `:lstm` | Long Short-Term Memory (default) | Best for state tracking |
  | `:gru` | Gated Recurrent Unit | Lighter, fewer parameters |

  ## Compared to Other Backbones

  | Aspect | TransformerLike | Mamba | Attention |
  |--------|-----------------|-------|-----------|
  | State tracking | LSTM hidden state | SSM state | None (window) |
  | Depth mechanism | FFN residual blocks | MLP in Mamba block | MHA + FFN |
  | Inference | Sequential (recurrent) | Parallel (scan) | Parallel |
  | Parameters | Moderate | Higher | Highest |
  | Proven in Melee | Yes (slippi-ai) | Benchmarked | Benchmarked |

  ## Usage

      # Build TransformerLike backbone
      model = TransformerLike.build(
        embed_size: 287,
        hidden_size: 512,
        num_layers: 3,
        cell_type: :lstm,
        ffn_multiplier: 2,
        window_size: 60
      )

      # Use via --backbone transformer_like

  ## References

  - vladfi1/slippi-ai `networks.py` — `TransformerLike` / `tx_like`
  - Architecture concept similar to pre-norm Transformer with recurrence replacing attention
  """

  require Axon

  alias Edifice.Recurrent.TransformerLike, as: EdificeTransformerLike

  # ============================================================================
  # Default Hyperparameters (matching slippi-ai defaults where sensible)
  # ============================================================================

  @default_hidden_size 512
  @default_num_layers 3
  @default_ffn_multiplier 2
  @default_activation :gelu
  @default_dropout 0.0
  @default_cell_type :lstm
  @default_window_size 60

  @doc "Default hidden dimension"
  def default_hidden_size, do: @default_hidden_size

  @doc "Default number of layers (recurrent + FFN pairs)"
  def default_num_layers, do: @default_num_layers

  @doc "Default FFN expansion multiplier"
  def default_ffn_multiplier, do: @default_ffn_multiplier

  @doc "Default activation function"
  def default_activation, do: @default_activation

  @doc "Default dropout rate"
  def default_dropout, do: @default_dropout

  @doc "Default recurrent cell type"
  def default_cell_type, do: @default_cell_type

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a TransformerLike model for sequence processing.

  Delegates to `Edifice.Recurrent.TransformerLike.build/1` with ExPhil-style
  option names.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Residual stream width (default: 512)
    - `:num_layers` - Number of recurrent+FFN pairs (default: 3)
    - `:cell_type` - `:lstm` or `:gru` (default: `:lstm`)
    - `:ffn_multiplier` - FFN expansion factor (default: 2)
    - `:activation` - FFN activation (default: `:gelu`)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:norm` - `:layer_norm` or `:rms_norm` (default: `:layer_norm`)
    - `:recurrent_norm` - Apply norm before recurrence (default: false)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    ffn_multiplier = Keyword.get(opts, :ffn_multiplier, @default_ffn_multiplier)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, :layer_norm)
    recurrent_norm = Keyword.get(opts, :recurrent_norm, false)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Delegate to Edifice with its option naming convention
    EdificeTransformerLike.build(
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

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Get the output size of the model.

  Always equals `:hidden_size`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended defaults for Melee gameplay.

  These match slippi-ai's production configuration adapted for
  ExPhil's 288-dim learned embeddings.
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      cell_type: @default_cell_type,
      ffn_multiplier: @default_ffn_multiplier,
      activation: @default_activation,
      dropout: 0.1,
      window_size: 60
    ]
  end

  @doc """
  Estimate parameter count for a given configuration.

  ## Options

    Same as `build/1`.
  """
  @spec param_count(keyword()) :: pos_integer()
  def param_count(opts \\ []) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    ffn_multiplier = Keyword.get(opts, :ffn_multiplier, @default_ffn_multiplier)

    # Input projection
    proj_params =
      if embed_size != hidden_size do
        embed_size * hidden_size + hidden_size
      else
        0
      end

    # Per layer: recurrent + FFN
    recurrent_params_per_layer =
      case cell_type do
        :lstm ->
          # LSTM: 4 gates × (input_size + hidden_size) × hidden_size + 4 × hidden_size bias
          4 * (hidden_size + hidden_size) * hidden_size + 4 * hidden_size

        :gru ->
          # GRU: 3 gates × (input_size + hidden_size) × hidden_size + 3 × hidden_size bias
          3 * (hidden_size + hidden_size) * hidden_size + 3 * hidden_size
      end

    ffn_inner = hidden_size * ffn_multiplier

    # FFN: up projection + down projection + biases + layer norm
    ffn_params_per_layer =
      hidden_size * ffn_inner + ffn_inner +
        ffn_inner * hidden_size + hidden_size +
        2 * hidden_size

    layer_params = recurrent_params_per_layer + ffn_params_per_layer

    # Final layer norm: 2 × hidden_size
    final_norm = 2 * hidden_size

    proj_params + num_layers * layer_params + final_norm
  end
end
