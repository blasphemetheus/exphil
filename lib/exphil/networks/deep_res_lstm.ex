defmodule ExPhil.Networks.DeepResLSTM do
  @moduledoc """
  DeepResLSTM: Stacked Residual LSTM Architecture from slippi-ai.

  Implements the `res_lstm` architecture from vladfi1/slippi-ai's `networks.py`.
  Stacks residual LSTM blocks — each block applies LayerNorm -> LSTM ->
  zero-initialized decoder -> residual add.

  ## Key Design

  Simpler than TransformerLike (no FFN blocks) but deeper than plain LSTM
  (residual connections enable stacking more layers without gradient degradation):

  ```
  Input [batch, seq_len, embed_size]
        |
  Dense encoder -> hidden_size
        |
  Per layer:
    residual --------------------------------+
        |                                    |
    LayerNorm                                |
        |                                    |
    LSTM(hidden_size)                        |
        |                                    |
    Dense(hidden_size, zero_init)            |
        |                                    |
    + ---------------------------------------+
        | (repeat num_layers)
        |
  Final LayerNorm -> last timestep
        |
  [batch, hidden_size]
  ```

  ## Why This Architecture?

  The zero-initialized decoder means the network starts as identity — each
  residual block initially just passes through the encoder output. As training
  progresses, the LSTM layers gradually learn to contribute useful temporal
  features. This enables stacking many layers without the vanishing gradient
  problems of deep plain LSTMs.

  ## Compared to Other Backbones

  | Aspect | DeepResLSTM | TransformerLike | Plain LSTM |
  |--------|-------------|-----------------|------------|
  | Depth mechanism | Residual + zero-init | FFN residual blocks | None |
  | Parameters | Lower | Moderate | Lowest |
  | Complexity | Simple | Moderate | Simple |
  | Deep stacking | Yes (residual) | Yes (residual) | Difficult |

  ## Usage

      # Build DeepResLSTM backbone
      model = DeepResLSTM.build(
        embed_size: 287,
        hidden_size: 512,
        num_layers: 3,
        window_size: 60
      )

      # Use via --backbone deep_res_lstm

  ## References

  - vladfi1/slippi-ai `networks.py` — `DeepResLSTM` / `res_lstm`
  """

  require Axon

  alias Edifice.Recurrent.DeepResLSTM, as: EdificeDeepResLSTM

  # ============================================================================
  # Default Hyperparameters (matching slippi-ai defaults where sensible)
  # ============================================================================

  @default_hidden_size 512
  @default_num_layers 3
  @default_dropout 0.0
  @default_window_size 60

  @doc "Default hidden dimension"
  def default_hidden_size, do: @default_hidden_size

  @doc "Default number of residual LSTM blocks"
  def default_num_layers, do: @default_num_layers

  @doc "Default dropout rate"
  def default_dropout, do: @default_dropout

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a DeepResLSTM model for sequence processing.

  Delegates to `Edifice.Recurrent.DeepResLSTM.build/1` with ExPhil-style
  option names.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Residual stream width (default: 512)
    - `:num_layers` - Number of residual LSTM blocks (default: 3)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:norm` - `:layer_norm` or `:rms_norm` (default: `:layer_norm`)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, :layer_norm)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Delegate to Edifice with its option naming convention
    EdificeDeepResLSTM.build(
      embed_dim: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      norm: norm,
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

    # Encoder: embed_size -> hidden_size
    encoder_params = embed_size * hidden_size + hidden_size

    # Per layer: LayerNorm + LSTM + zero-init decoder
    # LayerNorm: 2 * hidden_size (scale + bias)
    norm_params = 2 * hidden_size

    # LSTM: 4 gates * (input_size + hidden_size) * hidden_size + 4 * hidden_size bias
    lstm_params = 4 * (hidden_size + hidden_size) * hidden_size + 4 * hidden_size

    # Decoder: hidden_size -> hidden_size (zero-init dense)
    decoder_params = hidden_size * hidden_size + hidden_size

    layer_params = norm_params + lstm_params + decoder_params

    # Final layer norm: 2 * hidden_size
    final_norm = 2 * hidden_size

    encoder_params + num_layers * layer_params + final_norm
  end
end
