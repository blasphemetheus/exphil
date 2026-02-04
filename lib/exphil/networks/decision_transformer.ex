defmodule ExPhil.Networks.DecisionTransformer do
  @moduledoc """
  Decision Transformer: Return-Conditioned Sequence Modeling for RL.

  Decision Transformer frames reinforcement learning as a sequence modeling
  problem. Instead of learning a value function, it directly predicts actions
  conditioned on a desired return-to-go (future cumulative reward).

  Use via CLI: `--backbone decision_transformer`

  ## Key Innovation: Return Conditioning

  The model takes as input:
  - Return-to-go (R): Desired future cumulative reward
  - State (s): Current observation
  - Action (a): Previous action

  And predicts the next action to achieve the desired return.

  ## Input Format

  ```
  (R₁, s₁, a₁, R₂, s₂, a₂, ..., Rₜ, sₜ) → aₜ
  ```

  The model sees a sequence of (return, state, action) triplets and
  predicts what action to take to achieve the given return.

  ## Architecture

  ```
  ┌───────────────────────────────────────────────────────────┐
  │                    Token Embeddings                        │
  │                                                            │
  │   R₁    s₁    a₁    R₂    s₂    a₂    ...    Rₜ    sₜ     │
  │    │     │     │     │     │     │            │     │      │
  │    ▼     ▼     ▼     ▼     ▼     ▼            ▼     ▼      │
  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐      ┌───┐ ┌───┐    │
  │  │ R │ │ S │ │ A │ │ R │ │ S │ │ A │ ...  │ R │ │ S │    │
  │  │emb│ │emb│ │emb│ │emb│ │emb│ │emb│      │emb│ │emb│    │
  │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘      └───┘ └───┘    │
  │    │     │     │     │     │     │            │     │      │
  │    └─────┴─────┴─────┴─────┴─────┴────────────┴─────┘      │
  │                         │                                   │
  │                         ▼                                   │
  │            ┌────────────────────────┐                      │
  │            │   Causal Transformer   │                      │
  │            │   (GPT-style blocks)   │                      │
  │            └────────────────────────┘                      │
  │                         │                                   │
  │                         ▼                                   │
  │            Action prediction head                          │
  │            (from state token positions)                    │
  └───────────────────────────────────────────────────────────┘
  ```

  ## Return-to-Go for Melee

  In Melee, return-to-go can be:
  - Stock lead (e.g., +2 = "I want to be 2 stocks ahead")
  - Damage differential
  - Binary game outcome (win = 1, loss = 0)

  ## Usage

      # Build Decision Transformer
      model = DecisionTransformer.build(
        state_size: 287,
        action_size: 64,
        hidden_size: 256,
        num_layers: 6,
        context_length: 20  # Number of (R, s, a) triplets
      )

      # Training uses separate pipeline (see training/decision_transformer.ex)

  ## When to Use Decision Transformer

  From "When Should We Prefer Decision Transformers?" (2024):
  1. DT requires more data than CQL but is more robust
  2. DT is substantially better in sparse-reward and low-quality data settings
  3. DT and BC are preferable as task horizon increases

  ## Reference

  - Paper: "Decision Transformer" (arXiv:2106.01345)
  - Multi-Game DT: Single model plays 46 Atari games at near-human level
  """

  require Axon

  alias ExPhil.Networks.Attention

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 8
  @default_head_dim 32
  @default_context_length 20
  @default_dropout 0.1

  @doc """
  Build a Decision Transformer for sequence modeling.

  ## Options

    - `:state_size` - Size of state embedding (required)
    - `:action_size` - Size of action embedding (required, default: 64)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:head_dim` - Dimension per head (default: 32)
    - `:context_length` - Number of (R, s, a) triplets in context (default: 20)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Input Shape

  The model expects three separate inputs:
  - `returns`: [batch, context_length] - Return-to-go values
  - `states`: [batch, context_length, state_size] - State embeddings
  - `actions`: [batch, context_length, action_size] - Action embeddings

  ## Output Shape

  - Action predictions: [batch, context_length, action_size]
  - (Only the last position is typically used for inference)
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    state_size = Keyword.fetch!(opts, :state_size)
    action_size = Keyword.get(opts, :action_size, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    context_length = Keyword.get(opts, :context_length, @default_context_length)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Sequence length = 3 * context_length (R, s, a for each timestep)
    seq_len = 3 * context_length

    # Inputs
    returns_input = Axon.input("returns", shape: {nil, context_length})
    states_input = Axon.input("states", shape: {nil, context_length, state_size})
    actions_input = Axon.input("actions", shape: {nil, context_length, action_size})

    # Embed returns (scalar -> hidden_size)
    return_embed = embed_returns(returns_input, hidden_size, name: "return_embed")

    # Embed states (state_size -> hidden_size)
    state_embed = Axon.dense(states_input, hidden_size, name: "state_embed")

    # Embed actions (action_size -> hidden_size)
    action_embed = Axon.dense(actions_input, hidden_size, name: "action_embed")

    # Interleave tokens: [R₁, s₁, a₁, R₂, s₂, a₂, ...]
    # This creates a sequence of length 3 * context_length
    tokens = Axon.layer(
      &interleave_tokens/4,
      [return_embed, state_embed, action_embed],
      name: "interleave",
      context_length: context_length,
      hidden_size: hidden_size,
      op_name: :interleave_tokens
    )

    # Add positional encoding
    tokens = add_timestep_encoding(tokens, seq_len, hidden_size, name: "timestep_encoding")

    # Transformer blocks (causal)
    tokens =
      Enum.reduce(1..num_layers, tokens, fn layer_idx, acc ->
        build_transformer_block(
          acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          dropout: dropout,
          seq_len: seq_len,
          name: "transformer_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    tokens = Axon.layer_norm(tokens, name: "final_norm")

    # Extract action predictions from state token positions
    # State tokens are at positions 1, 4, 7, ... (every 3rd starting from 1)
    action_preds = Axon.layer(
      &extract_action_predictions/2,
      [tokens],
      name: "action_predictions",
      context_length: context_length,
      action_size: action_size,
      hidden_size: hidden_size,
      op_name: :extract_action_predictions
    )

    # Project to action size
    Axon.dense(action_preds, action_size, name: "action_head")
  end

  @doc """
  Build a simplified Decision Transformer that uses the existing backbone pattern.

  This version takes pre-embedded sequences (return + state concatenated) and
  predicts actions. It's simpler to integrate with existing training code.

  ## Input

  - `state_sequence`: [batch, seq_len, embed_size] where each frame includes
    the return-to-go concatenated with the state embedding.

  ## Output

  - [batch, hidden_size] from the last position (same as other backbones)
  """
  @spec build_simple(keyword()) :: Axon.t()
  def build_simple(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    # embed_size should include return-to-go embedding concatenated with state
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project to hidden size
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Add positional encoding
    x = Attention.add_positional_encoding(x, name: "pos_encoding")

    # Stack transformer blocks (causal self-attention)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_transformer_block(
          acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          dropout: dropout,
          seq_len: seq_len,
          name: "dt_block_#{layer_idx}"
        )
      end)

    # Final norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # Internal Components
  # ============================================================================

  # Embed scalar returns to hidden dimension
  defp embed_returns(returns, hidden_size, opts) do
    name = Keyword.get(opts, :name, "return_embed")

    # returns: [batch, context_length]
    # Output: [batch, context_length, hidden_size]
    Axon.nx(
      returns,
      fn r ->
        # Expand to [batch, context_length, 1]
        r_expanded = Nx.new_axis(r, 2)
        # Broadcast to hidden_size (will be transformed by dense)
        r_expanded
      end,
      name: "#{name}_expand"
    )
    |> Axon.dense(hidden_size, name: name)
  end

  # Interleave R, s, a tokens into a single sequence
  defp interleave_tokens(return_embed, state_embed, action_embed, opts) do
    context_length = opts[:context_length]
    _hidden_size = opts[:hidden_size]

    # return_embed: [batch, context_length, hidden_size]
    # state_embed: [batch, context_length, hidden_size]
    # action_embed: [batch, context_length, hidden_size]

    batch = Nx.axis_size(return_embed, 0)
    hidden = Nx.axis_size(return_embed, 2)

    # Stack along a new dimension: [batch, context_length, 3, hidden_size]
    stacked = Nx.stack([return_embed, state_embed, action_embed], axis: 2)

    # Reshape to interleaved: [batch, context_length * 3, hidden_size]
    Nx.reshape(stacked, {batch, context_length * 3, hidden})
  end

  # Add timestep (not position) encoding
  # Each triplet (R, s, a) at time t gets the same timestep embedding
  defp add_timestep_encoding(tokens, seq_len, hidden_size, opts) do
    name = Keyword.get(opts, :name, "timestep_encoding")

    Axon.nx(
      tokens,
      fn t ->
        batch = Nx.axis_size(t, 0)

        # Create timestep indices: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
        timesteps = Nx.iota({seq_len})
        timesteps = Nx.divide(timesteps, 3) |> Nx.floor() |> Nx.as_type(:s32)

        # Create sinusoidal encoding
        pos = Nx.new_axis(timesteps, 1)  # [seq_len, 1]
        dim = Nx.iota({hidden_size})  # [hidden_size]

        # Compute angles: [seq_len, hidden_size]
        angles = Nx.divide(pos, Nx.pow(10000.0, Nx.divide(dim, hidden_size)))

        # Apply sin to even indices, cos to odd
        sin_part = Nx.sin(angles)
        cos_part = Nx.cos(angles)

        # Create mask for even/odd indices: [1, hidden_size] then broadcast to [seq_len, hidden_size]
        is_even = Nx.equal(Nx.remainder(dim, 2), 0)
        is_even = Nx.broadcast(is_even, {seq_len, hidden_size})

        # Interleave sin and cos: [seq_len, hidden_size]
        encoding = Nx.select(is_even, sin_part, cos_part)

        # Broadcast to batch: [batch, seq_len, hidden_size]
        encoding = Nx.broadcast(encoding, {batch, seq_len, hidden_size})

        Nx.add(t, encoding)
      end,
      name: name
    )
  end

  # Extract predictions from state token positions
  defp extract_action_predictions(tokens, opts) do
    context_length = opts[:context_length]

    # tokens: [batch, 3 * context_length, hidden_size]
    # State tokens are at positions 1, 4, 7, ... (1 + 3*i for i = 0..context_length-1)

    # Extract state token positions
    # Using gather would be ideal, but we can use slicing
    indices = Enum.map(0..(context_length - 1), fn i -> 1 + 3 * i end)

    state_tokens =
      indices
      |> Enum.map(fn idx ->
        Nx.slice_along_axis(tokens, idx, 1, axis: 1)
      end)
      |> Nx.concatenate(axis: 1)

    # state_tokens: [batch, context_length, hidden_size]
    state_tokens
  end

  @doc """
  Build a causal transformer block.
  """
  @spec build_transformer_block(Axon.t(), keyword()) :: Axon.t()
  def build_transformer_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, nil)
    name = Keyword.get(opts, :name, "transformer_block")

    attn_dim = num_heads * head_dim

    # Self-attention sub-block
    x = Axon.layer_norm(input, name: "#{name}_attn_norm")

    # Project to attention dimension if needed
    x =
      if hidden_size != attn_dim do
        Axon.dense(x, attn_dim, name: "#{name}_attn_proj_in")
      else
        x
      end

    # Causal self-attention
    x = Attention.multi_head_attention(x,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      causal: true,
      seq_len: seq_len,
      name: "#{name}_attn"
    )

    # Project back if needed
    x =
      if hidden_size != attn_dim do
        Axon.dense(x, hidden_size, name: "#{name}_attn_proj_out")
      else
        x
      end

    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_attn_dropout")
      else
        x
      end

    x = Axon.add(input, x, name: "#{name}_attn_residual")

    # FFN sub-block
    ffn_input = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_dim = hidden_size * 4

    ffn =
      ffn_input
      |> Axon.dense(ffn_dim, name: "#{name}_ffn1")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    ffn =
      if dropout > 0 do
        Axon.dropout(ffn, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn
      end

    Axon.add(x, ffn, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Decision Transformer.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    state_size = Keyword.get(opts, :state_size, 287)
    action_size = Keyword.get(opts, :action_size, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    attn_dim = num_heads * head_dim
    ffn_dim = hidden_size * 4

    # Embedding layers
    # return embedding
    # state embedding
    # action embedding
    embed_params =
      1 * hidden_size +
      state_size * hidden_size +
      action_size * hidden_size

    # Per transformer layer
    # QKV + output
    # FFN
    per_layer =
      attn_dim * 3 * attn_dim + attn_dim * attn_dim +
      hidden_size * ffn_dim + ffn_dim * hidden_size

    # Action head
    action_head = hidden_size * action_size

    embed_params + per_layer * num_layers + action_head
  end

  @doc """
  Get recommended defaults for Melee gameplay.
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 8,
      head_dim: 32,
      context_length: 20,
      dropout: 0.1
    ]
  end

  @doc """
  Compute return-to-go from game state.

  For Melee, we can use:
  - Stock difference (current stocks - opponent stocks)
  - Damage differential
  - Game outcome (if known)
  """
  @spec compute_return_to_go(map(), keyword()) :: float()
  def compute_return_to_go(game_state, opts \\ []) do
    mode = Keyword.get(opts, :mode, :stock_lead)

    case mode do
      :stock_lead ->
        # +1 for each stock ahead, -1 for each behind
        player_stocks = game_state[:player_stocks] || 4
        opponent_stocks = game_state[:opponent_stocks] || 4
        player_stocks - opponent_stocks

      :damage_lead ->
        # Positive if opponent has more damage
        player_damage = game_state[:player_damage] || 0
        opponent_damage = game_state[:opponent_damage] || 0
        (opponent_damage - player_damage) / 100.0

      :game_outcome ->
        # 1 if won, 0 if lost, 0.5 if ongoing
        case game_state[:game_result] do
          :win -> 1.0
          :loss -> 0.0
          _ -> 0.5
        end

      :combined ->
        # Weighted combination
        stock_weight = Keyword.get(opts, :stock_weight, 0.7)
        damage_weight = Keyword.get(opts, :damage_weight, 0.3)

        stock_lead = compute_return_to_go(game_state, mode: :stock_lead)
        damage_lead = compute_return_to_go(game_state, mode: :damage_lead)

        stock_weight * stock_lead + damage_weight * damage_lead
    end
  end
end
