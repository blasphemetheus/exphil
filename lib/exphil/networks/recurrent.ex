defmodule ExPhil.Networks.Recurrent do
  @moduledoc """
  Recurrent neural network layers for temporal game state processing.

  Provides LSTM and GRU architectures for learning temporal dependencies
  in Melee gameplay - essential for understanding:
  - Multi-frame actions (charged moves, combos)
  - Opponent patterns and habits
  - Recovery and edgeguard sequences
  - Tech chase reactions

  ## Architecture

  The recurrent backbone processes sequences of embedded game states:

  ```
  Frame Sequence [batch, seq_len, embed_size]
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │ ←─ hidden state (h, c for LSTM)
  │  Layer 1    │
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │  (optional stacked layers)
  │  Layer 2    │
  └─────────────┘
        │
        ▼
  Hidden Output [batch, hidden_size]
  ```

  ## Hidden State Management

  For real-time inference, hidden states must be carried between frames:

      # Initialize hidden state
      hidden = Recurrent.initial_hidden(model, batch_size)

      # Process frame, get new hidden
      {output, new_hidden} = Recurrent.forward_with_state(model, params, frame, hidden)

      # Use new_hidden for next frame
      ...

  ## Usage

      # Build recurrent backbone
      model = Recurrent.build(
        embed_size: 1024,
        hidden_size: 256,
        num_layers: 2,
        cell_type: :lstm,
        dropout: 0.1
      )

      # Use as backbone in policy network
      input = Axon.input("state_sequence", shape: {nil, nil, 1024})
      backbone = Recurrent.build_backbone(input, hidden_size: 256, cell_type: :gru)
      policy_head = build_policy_head(backbone)

  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 1
  @default_cell_type :lstm
  @default_dropout 0.0
  # @default_bidirectional false  # Future: bidirectional support

  @type cell_type :: :lstm | :gru
  @type hidden_state :: Nx.Tensor.t() | {Nx.Tensor.t(), Nx.Tensor.t()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a recurrent model for sequence processing.

  ## Options
    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Size of recurrent hidden state (default: 256)
    - `:num_layers` - Number of stacked recurrent layers (default: 1)
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:dropout` - Dropout rate between layers (default: 0.0)
    - `:bidirectional` - Use bidirectional processing (default: false)
    - `:return_sequences` - Return all timesteps or just last (default: false)

  ## Returns
    An Axon model that processes sequences and outputs hidden representations.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    return_sequences = Keyword.get(opts, :return_sequences, false)

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, nil, embed_size})

    build_backbone(input,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: cell_type,
      dropout: dropout,
      return_sequences: return_sequences
    )
  end

  @doc """
  Build the recurrent backbone from an existing input layer.

  Useful for integrating into larger networks (policy, value).

  ## Options
    - `:hidden_size` - Size of recurrent hidden state (default: 256)
    - `:num_layers` - Number of stacked recurrent layers (default: 1)
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:dropout` - Dropout rate between layers (default: 0.0)
    - `:return_sequences` - Return all timesteps or just last (default: false)
  """
  @spec build_backbone(Axon.t(), keyword()) :: Axon.t()
  def build_backbone(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    return_sequences = Keyword.get(opts, :return_sequences, false)

    # Build stacked recurrent layers
    output =
      Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
        is_last_layer = layer_idx == num_layers

        # Only return sequences for intermediate layers, or if explicitly requested
        layer_return_seq = not is_last_layer or return_sequences

        layer = build_recurrent_layer(acc, hidden_size, cell_type,
          name: "#{cell_type}_#{layer_idx}",
          return_sequences: layer_return_seq
        )

        # Add dropout between layers (not after last)
        if dropout > 0 and not is_last_layer do
          Axon.dropout(layer, rate: dropout, name: "recurrent_dropout_#{layer_idx}")
        else
          layer
        end
      end)

    output
  end

  @doc """
  Build a single recurrent layer (LSTM or GRU).
  """
  @spec build_recurrent_layer(Axon.t(), non_neg_integer(), cell_type(), keyword()) :: Axon.t()
  def build_recurrent_layer(input, hidden_size, cell_type, opts \\ []) do
    name = Keyword.get(opts, :name, "recurrent")
    return_sequences = Keyword.get(opts, :return_sequences, true)

    recurrent_opts = [
      name: name,
      recurrent_initializer: :glorot_uniform,
      use_bias: true
    ]

    # Axon.lstm/gru returns {output_sequence, hidden_state_tuple}
    # We need to extract just the output sequence using Axon.elem
    {output_seq, _hidden} = case cell_type do
      :lstm -> Axon.lstm(input, hidden_size, recurrent_opts)
      :gru -> Axon.gru(input, hidden_size, recurrent_opts)
    end

    if return_sequences do
      output_seq
    else
      # Take the last timestep: [batch, seq_len, hidden] -> [batch, hidden]
      Axon.nx(output_seq, fn tensor ->
        seq_len = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_len - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end, name: "#{name}_last")
    end
  end

  # ============================================================================
  # Stateful Inference (for real-time play)
  # ============================================================================

  @doc """
  Build a stateful recurrent model that explicitly manages hidden state.

  This is essential for real-time inference where we process one frame at a time
  and need to carry hidden state between frames.

  Returns a simple model that processes single frames. Hidden state management
  is handled externally using `initial_hidden/2`.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_size` - Size of hidden state (default: 256)
    - `:cell_type` - :lstm or :gru (default: :lstm)

  ## Returns
    An Axon model that takes single frames and outputs hidden representations.
  """
  @spec build_stateful(keyword()) :: Axon.t()
  def build_stateful(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)

    # Single frame input reshaped as sequence of 1: [batch, 1, embed_size]
    frame_input = Axon.input("frame", shape: {nil, 1, embed_size})

    # Build single recurrent layer
    # Axon.lstm/gru returns {output_sequence, hidden_state}
    {output_seq, _hidden} = case cell_type do
      :lstm ->
        Axon.lstm(frame_input, hidden_size,
          name: "lstm_stateful",
          recurrent_initializer: :glorot_uniform
        )

      :gru ->
        Axon.gru(frame_input, hidden_size,
          name: "gru_stateful",
          recurrent_initializer: :glorot_uniform
        )
    end

    # Squeeze the sequence dimension (seq_len=1)
    Axon.nx(output_seq, fn tensor ->
      Nx.squeeze(tensor, axes: [1])
    end, name: "stateful_output")
  end

  @doc """
  Create initial hidden state for a given batch size.

  ## Options
    - `:hidden_size` - Size of hidden state (default: 256)
    - `:cell_type` - :lstm or :gru (default: :lstm)

  ## Returns
    For LSTM: `{h, c}` tuple of zero tensors
    For GRU: single zero tensor
  """
  @spec initial_hidden(non_neg_integer(), keyword()) :: hidden_state()
  def initial_hidden(batch_size, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)

    case cell_type do
      :lstm ->
        h = Nx.broadcast(0.0, {batch_size, hidden_size})
        c = Nx.broadcast(0.0, {batch_size, hidden_size})
        {h, c}

      :gru ->
        Nx.broadcast(0.0, {batch_size, hidden_size})
    end
  end

  # ============================================================================
  # Hybrid Architectures (Recurrent + MLP)
  # ============================================================================

  @doc """
  Build a hybrid recurrent-MLP backbone.

  Combines recurrent layers for temporal processing with MLP layers
  for non-linear transformation. This often works better than pure RNN.

  ```
  Sequence [batch, seq_len, embed_size]
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │
  │  Layers     │
  └─────────────┘
        │
        ▼
  [batch, hidden_size]
        │
        ▼
  ┌─────────────┐
  │    MLP      │
  │  Layers     │
  └─────────────┘
        │
        ▼
  [batch, output_size]
  ```

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:recurrent_size` - Size of recurrent hidden (default: 256)
    - `:mlp_sizes` - List of MLP layer sizes (default: [256])
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:num_recurrent_layers` - Number of RNN layers (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:activation` - MLP activation (default: :relu)
  """
  @spec build_hybrid(keyword()) :: Axon.t()
  def build_hybrid(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    recurrent_size = Keyword.get(opts, :recurrent_size, @default_hidden_size)
    mlp_sizes = Keyword.get(opts, :mlp_sizes, [256])
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    num_recurrent_layers = Keyword.get(opts, :num_recurrent_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, 0.1)
    activation = Keyword.get(opts, :activation, :relu)

    # Input sequence
    input = Axon.input("state_sequence", shape: {nil, nil, embed_size})

    # Recurrent backbone (outputs last timestep)
    recurrent_output = build_backbone(input,
      hidden_size: recurrent_size,
      num_layers: num_recurrent_layers,
      cell_type: cell_type,
      dropout: dropout,
      return_sequences: false
    )

    # MLP layers on top
    mlp_sizes
    |> Enum.with_index()
    |> Enum.reduce(recurrent_output, fn {size, idx}, acc ->
      acc
      |> Axon.dense(size, name: "hybrid_mlp_#{idx}")
      |> Axon.activation(activation)
      |> Axon.dropout(rate: dropout)
    end)
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Calculate the output size of a recurrent backbone.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Create a sequence from individual frames for batch processing.

  Takes a list of embedded frames and stacks them into a sequence tensor.
  """
  @spec frames_to_sequence([Nx.Tensor.t()]) :: Nx.Tensor.t()
  def frames_to_sequence(frames) when is_list(frames) do
    # frames: list of [embed_size] or [batch, embed_size] tensors
    # output: [batch, seq_len, embed_size]
    frames
    |> Enum.map(fn frame ->
      case Nx.shape(frame) do
        {_embed} -> Nx.new_axis(frame, 0)  # Add batch dim
        {_batch, _embed} -> frame
      end
    end)
    |> Nx.stack(axis: 1)
  end

  @doc """
  Pad or truncate sequence to fixed length.

  Useful for batch processing sequences of different lengths.
  """
  @spec pad_sequence(Nx.Tensor.t(), non_neg_integer(), keyword()) :: Nx.Tensor.t()
  def pad_sequence(sequence, target_length, opts \\ []) do
    pad_value = Keyword.get(opts, :pad_value, 0.0)

    current_length = Nx.axis_size(sequence, 1)

    cond do
      current_length == target_length ->
        sequence

      current_length > target_length ->
        # Truncate (keep last target_length frames)
        start = current_length - target_length
        Nx.slice_along_axis(sequence, start, target_length, axis: 1)

      true ->
        # Pad at the beginning
        batch_size = Nx.axis_size(sequence, 0)
        embed_size = Nx.axis_size(sequence, 2)
        padding_length = target_length - current_length

        padding = Nx.broadcast(pad_value, {batch_size, padding_length, embed_size})
        Nx.concatenate([padding, sequence], axis: 1)
    end
  end

  @doc """
  Get supported cell types.
  """
  @spec cell_types() :: [cell_type()]
  def cell_types, do: [:lstm, :gru]
end
