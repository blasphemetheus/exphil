defmodule ExPhil.Networks.ActionChunking do
  @moduledoc """
  ACT: Action Chunking with Transformers.

  Implements ACT from "Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware" (Zhao et al., RSS 2023). Predicts action chunks
  (sequences) using a CVAE with Transformer encoder-decoder.

  ## Key Innovation: Action Chunks

  Instead of predicting single actions, ACT predicts k actions at once:
  - Reduces effective task horizon by factor k
  - Mitigates compounding errors in imitation learning
  - Captures temporal correlations in action sequences

  ```
  Standard BC:  o_t → a_t           (horizon = T)
  ACT:          o_t → [a_t, ..., a_{t+k-1}]  (horizon = T/k)
  ```

  ## CVAE Architecture

  ```
  Training:
  ┌─────────────────────────────────────────────────┐
  │  Encoder (only during training)                  │
  │  [CLS] + obs + action_sequence → z ~ N(μ, σ²)   │
  └─────────────────────────────────────────────────┘
                      │ z
                      ▼
  ┌─────────────────────────────────────────────────┐
  │  Decoder                                         │
  │  obs + z → action_chunk [k actions]             │
  └─────────────────────────────────────────────────┘

  Inference:
  - Skip encoder, set z = 0 (prior mean)
  - Decoder: obs → action_chunk
  ```

  ## Temporal Ensembling

  At inference, query policy every step and average overlapping predictions:
  ```
  a_t = Σ w_i * a_t^(i) / Σ w_i
  where w_i = exp(-m * i), i = query age
  ```

  ## Training Loss

  ```
  L = L1(actions, predicted_actions) + β * KL(q(z|o,a) || p(z))
  ```

  Where β=10 balances reconstruction vs regularization.

  ## Usage

      # Build ACT policy
      model = ActionChunking.build(
        obs_size: 287,
        action_dim: 64,
        chunk_size: 16
      )

      # Training: encoder + decoder
      {loss, predicted_actions} = ActionChunking.training_step(
        model, params, observations, action_chunks, key
      )

      # Inference: decoder only with z=0
      action_chunk = ActionChunking.inference(model, params, observations)

  ## Melee Application

  For Melee at 60fps:
  - `chunk_size`: 8-16 frames (133-266ms lookahead)
  - `action_dim`: 64 (discretized controller)
  - Query every frame, use temporal ensembling

  ## References
  - Paper: https://arxiv.org/abs/2304.13705
  - Project: https://tonyzhaozh.github.io/aloha/
  - HuggingFace: https://huggingface.co/docs/lerobot/en/act
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default action chunk size"
  def default_chunk_size, do: 16

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default number of transformer layers"
  def default_num_layers, do: 4

  @doc "Default number of attention heads"
  def default_num_heads, do: 8

  @doc "Default latent dimension"
  def default_latent_dim, do: 32

  @doc "Default KL weight (β)"
  def default_kl_weight, do: 10.0

  @doc "Default temporal ensembling decay"
  def default_ensemble_decay, do: 0.01

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an ACT (Action Chunking with Transformers) model.

  Returns both encoder and decoder for training.

  ## Options
    - `:obs_size` - Size of observation embedding (required)
    - `:action_dim` - Dimension of action space (required)
    - `:chunk_size` - Number of actions per chunk (default: 16)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:latent_dim` - Dimension of latent z (default: 32)

  ## Returns
    A map with :encoder and :decoder Axon models.
  """
  @spec build(keyword()) :: %{encoder: Axon.t(), decoder: Axon.t()}
  def build(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, default_chunk_size())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    latent_dim = Keyword.get(opts, :latent_dim, default_latent_dim())

    encoder = build_encoder(opts)
    decoder = build_decoder(opts)

    %{
      encoder: encoder,
      decoder: decoder,
      config: %{
        obs_size: obs_size,
        action_dim: action_dim,
        chunk_size: chunk_size,
        hidden_size: hidden_size,
        latent_dim: latent_dim
      }
    }
  end

  @doc """
  Build just the decoder for inference (z=0).

  This is more efficient for deployment since we don't need the encoder.
  """
  @spec build_inference(keyword()) :: Axon.t()
  def build_inference(opts \\ []) do
    build_decoder(opts)
  end

  # ============================================================================
  # CVAE Encoder
  # ============================================================================

  @doc """
  Build the CVAE encoder.

  Takes observations and action sequence, outputs latent distribution (μ, σ).

  Architecture:
  - [CLS] token + obs embedding + action embeddings
  - Transformer encoder
  - [CLS] output → μ, log_σ
  """
  @spec build_encoder(keyword()) :: Axon.t()
  def build_encoder(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, default_chunk_size())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    latent_dim = Keyword.get(opts, :latent_dim, default_latent_dim())

    # Inputs
    observations = Axon.input("observations", shape: {nil, obs_size})
    action_sequence = Axon.input("action_sequence", shape: {nil, chunk_size, action_dim})

    # Project observations to hidden size
    obs_embed = Axon.dense(observations, hidden_size, name: "enc_obs_proj")
    obs_embed = Axon.nx(obs_embed, fn x ->
      Nx.new_axis(x, 1)  # [batch, 1, hidden]
    end, name: "enc_obs_expand")

    # Project actions to hidden size
    action_embed = Axon.dense(action_sequence, hidden_size, name: "enc_action_proj")

    # Learnable [CLS] token (will be added as a constant in the layer)
    # For simplicity, we'll use a projected zero vector that gets learned
    cls_token = Axon.nx(observations, fn x ->
      batch = Nx.axis_size(x, 0)
      Nx.broadcast(0.0, {batch, 1, hidden_size})
    end, name: "enc_cls_init")
    cls_token = Axon.dense(cls_token, hidden_size, name: "enc_cls_proj")

    # Concatenate: [CLS, obs, action_1, ..., action_k]
    # Shape: [batch, 1 + 1 + chunk_size, hidden]
    sequence = Axon.concatenate([cls_token, obs_embed, action_embed], axis: 1, name: "enc_sequence")

    # Add positional embeddings
    seq_len = 2 + chunk_size
    sequence = add_positional_embedding(sequence, seq_len, hidden_size, "enc_pos")

    # Transformer encoder layers
    encoded = Enum.reduce(1..num_layers, sequence, fn layer_idx, acc ->
      build_transformer_block(acc, hidden_size, num_heads, "enc_layer_#{layer_idx}")
    end)

    # Extract [CLS] token output
    cls_output = Axon.nx(encoded, fn x ->
      Nx.slice_along_axis(x, 0, 1, axis: 1) |> Nx.squeeze(axes: [1])
    end, name: "enc_cls_extract")

    # Project to latent distribution parameters
    mu = Axon.dense(cls_output, latent_dim, name: "enc_mu")
    log_var = Axon.dense(cls_output, latent_dim, name: "enc_log_var")

    # Return both μ and log_σ²
    Axon.container({mu, log_var})
  end

  # ============================================================================
  # CVAE Decoder
  # ============================================================================

  @doc """
  Build the CVAE decoder.

  Takes observations and latent z, outputs action chunk.

  Architecture:
  - Concatenate obs embedding + z
  - Transformer decoder with learned action queries
  - Output: chunk_size actions
  """
  @spec build_decoder(keyword()) :: Axon.t()
  def build_decoder(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, default_chunk_size())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    latent_dim = Keyword.get(opts, :latent_dim, default_latent_dim())

    # Inputs
    observations = Axon.input("observations", shape: {nil, obs_size})
    latent_z = Axon.input("latent_z", shape: {nil, latent_dim})

    # Project observations
    obs_embed = Axon.dense(observations, hidden_size, name: "dec_obs_proj")

    # Project latent
    z_embed = Axon.dense(latent_z, hidden_size, name: "dec_z_proj")

    # Combine obs and z
    combined = Axon.add(obs_embed, z_embed, name: "dec_combine")
    combined = Axon.nx(combined, fn x ->
      Nx.new_axis(x, 1)  # [batch, 1, hidden]
    end, name: "dec_expand")

    # Learnable action queries (one per action in chunk)
    action_queries = Axon.nx(observations, fn x ->
      batch = Nx.axis_size(x, 0)
      Nx.broadcast(0.0, {batch, chunk_size, hidden_size})
    end, name: "dec_queries_init")
    action_queries = Axon.dense(action_queries, hidden_size, name: "dec_queries_proj")

    # Add positional embeddings to queries
    action_queries = add_positional_embedding(action_queries, chunk_size, hidden_size, "dec_pos")

    # Transformer decoder layers with cross-attention to combined
    decoded = Enum.reduce(1..num_layers, action_queries, fn layer_idx, acc ->
      build_decoder_block(acc, combined, hidden_size, num_heads, "dec_layer_#{layer_idx}")
    end)

    # Project to action space
    Axon.dense(decoded, action_dim, name: "dec_action_out")
  end

  # ============================================================================
  # Transformer Components
  # ============================================================================

  # Add sinusoidal positional embeddings
  defp add_positional_embedding(input, seq_len, hidden_size, name) do
    Axon.layer(
      &positional_embedding_impl/2,
      [input],
      name: name,
      seq_len: seq_len,
      hidden_size: hidden_size,
      op_name: :pos_embed
    )
  end

  defp positional_embedding_impl(x, opts) do
    seq_len = opts[:seq_len]
    hidden_size = opts[:hidden_size]

    # Create position indices [seq_len, 1]
    positions = Nx.iota({seq_len, 1}, type: :f32)

    # Create dimension indices [1, hidden_size]
    dims = Nx.iota({1, hidden_size}, type: :f32)

    # Compute angles: pos / 10000^(2i/d) -> [seq_len, hidden_size]
    exponent = Nx.divide(Nx.multiply(-2.0, Nx.floor(Nx.divide(dims, 2))), hidden_size)
    angles = Nx.multiply(positions, Nx.pow(10000.0, exponent))

    # Compute sin and cos for all positions
    sin_vals = Nx.sin(angles)
    cos_vals = Nx.cos(angles)

    # Create even/odd mask and broadcast to full shape [seq_len, hidden_size]
    dims_int = Nx.as_type(dims, :s32)
    even_mask_1d = Nx.equal(Nx.remainder(dims_int, 2), 0)
    even_mask = Nx.broadcast(even_mask_1d, {seq_len, hidden_size})

    # Apply sin to even dimensions, cos to odd
    pos_embed = Nx.select(even_mask, sin_vals, cos_vals)

    # Broadcast and add to input [batch, seq_len, hidden_size]
    Nx.add(x, pos_embed)
  end

  # Transformer encoder block (self-attention + FFN)
  defp build_transformer_block(input, hidden_size, num_heads, name) do
    # Self-attention with residual
    attn_out = build_self_attention(input, hidden_size, num_heads, "#{name}_attn")
    x = Axon.add(input, attn_out, name: "#{name}_attn_res")
    x = Axon.layer_norm(x, name: "#{name}_attn_norm")

    # FFN with residual
    ffn_out = build_ffn(x, hidden_size, "#{name}_ffn")
    x = Axon.add(x, ffn_out, name: "#{name}_ffn_res")
    Axon.layer_norm(x, name: "#{name}_ffn_norm")
  end

  # Transformer decoder block (self-attention + cross-attention + FFN)
  defp build_decoder_block(input, context, hidden_size, num_heads, name) do
    # Self-attention
    self_attn = build_self_attention(input, hidden_size, num_heads, "#{name}_self_attn")
    x = Axon.add(input, self_attn, name: "#{name}_self_res")
    x = Axon.layer_norm(x, name: "#{name}_self_norm")

    # Cross-attention to context
    cross_attn = build_cross_attention(x, context, hidden_size, num_heads, "#{name}_cross_attn")
    x = Axon.add(x, cross_attn, name: "#{name}_cross_res")
    x = Axon.layer_norm(x, name: "#{name}_cross_norm")

    # FFN
    ffn_out = build_ffn(x, hidden_size, "#{name}_ffn")
    x = Axon.add(x, ffn_out, name: "#{name}_ffn_res")
    Axon.layer_norm(x, name: "#{name}_ffn_norm")
  end

  # Self-attention
  defp build_self_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    Axon.layer(
      &multi_head_attention/4,
      [q, k, v],
      name: name,
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :mha
    )
  end

  # Cross-attention
  defp build_cross_attention(query_input, kv_input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(query_input, hidden_size, name: "#{name}_q")
    k = Axon.dense(kv_input, hidden_size, name: "#{name}_k")
    v = Axon.dense(kv_input, hidden_size, name: "#{name}_v")

    Axon.layer(
      &multi_head_attention/4,
      [q, k, v],
      name: name,
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :mha
    )
  end

  # Multi-head attention implementation
  defp multi_head_attention(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_q = Nx.axis_size(q, 1)
    seq_kv = Nx.axis_size(k, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_q, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_kv, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_kv, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Softmax (numerically stable)
    max_scores = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    sum_exp = Nx.add(Nx.sum(exp_scores, axes: [3], keep_axes: true), 1.0e-8)
    weights = Nx.divide(exp_scores, sum_exp)

    # Apply to values
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_q, num_heads * head_dim})
  end

  # Feed-forward network
  defp build_ffn(input, hidden_size, name) do
    inner_size = hidden_size * 4

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Training Functions
  # ============================================================================

  @doc """
  Reparameterization trick for sampling z.

  z = μ + σ * ε, where ε ~ N(0, I)
  """
  @spec reparameterize(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn reparameterize(mu, log_var, epsilon) do
    # σ = exp(log_var / 2)
    std = Nx.exp(Nx.multiply(0.5, log_var))
    Nx.add(mu, Nx.multiply(std, epsilon))
  end

  @doc """
  Compute KL divergence between q(z|x) and p(z) = N(0, I).

  KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))
  """
  @spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn kl_divergence(mu, log_var) do
    kl_per_dim = Nx.subtract(
      Nx.add(1.0, log_var),
      Nx.add(Nx.multiply(mu, mu), Nx.exp(log_var))
    )
    Nx.mean(Nx.multiply(-0.5, Nx.sum(kl_per_dim, axes: [1])))
  end

  @doc """
  Compute L1 reconstruction loss.
  """
  @spec reconstruction_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn reconstruction_loss(target, predicted) do
    Nx.mean(Nx.abs(Nx.subtract(target, predicted)))
  end

  @doc """
  Compute total CVAE loss.

  L = L1(actions, predicted) + β * KL(q(z|o,a) || p(z))
  """
  @spec cvae_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn cvae_loss(target_actions, predicted_actions, mu, log_var, kl_weight) do
    recon = reconstruction_loss(target_actions, predicted_actions)
    kl = kl_divergence(mu, log_var)
    Nx.add(recon, Nx.multiply(kl_weight, kl))
  end

  # ============================================================================
  # Temporal Ensembling
  # ============================================================================

  @doc """
  Compute temporal ensemble weights.

  Weights decay exponentially with query age:
  w_i = exp(-m * i)

  ## Parameters
    - `num_queries` - Number of overlapping predictions
    - `decay` - Decay rate m (default: 0.01)

  ## Returns
    Normalized weight tensor.
  """
  @spec ensemble_weights(pos_integer(), float()) :: Nx.Tensor.t()
  def ensemble_weights(num_queries, decay \\ 0.01) do
    indices = Nx.iota({num_queries}, type: :f32)
    weights = Nx.exp(Nx.multiply(-decay, indices))
    # Normalize
    Nx.divide(weights, Nx.sum(weights))
  end

  @doc """
  Apply temporal ensembling to overlapping action chunks.

  ## Parameters
    - `action_chunks` - List of action chunks, oldest first
    - `weights` - Ensemble weights (from `ensemble_weights/2`)

  ## Returns
    Single action (weighted average of predictions for current timestep).
  """
  @spec apply_ensemble([Nx.Tensor.t()], Nx.Tensor.t()) :: Nx.Tensor.t()
  def apply_ensemble(action_chunks, weights) do
    # Each chunk predicts the current action at a different index
    # Chunk 0 (oldest): action is at index (k-1)
    # Chunk 1: action is at index (k-2)
    # ...
    # Chunk k-1 (newest): action is at index 0

    num_chunks = length(action_chunks)

    predictions =
      action_chunks
      |> Enum.with_index()
      |> Enum.map(fn {chunk, i} ->
        # Extract the action for current timestep
        # For chunk i, the current action is at position (num_chunks - 1 - i)
        action_idx = num_chunks - 1 - i
        Nx.slice_along_axis(chunk, action_idx, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end)

    # Weighted average
    weighted_sum =
      predictions
      |> Enum.with_index()
      |> Enum.reduce(Nx.broadcast(0.0, Nx.shape(hd(predictions))), fn {pred, i}, acc ->
        weight = Nx.to_number(weights[i])
        Nx.add(acc, Nx.multiply(weight, pred))
      end)

    weighted_sum
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an ACT model.

  Returns chunk_size * action_dim.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, 64)
    chunk_size = Keyword.get(opts, :chunk_size, default_chunk_size())
    chunk_size * action_dim
  end

  @doc """
  Calculate approximate parameter count for an ACT model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    obs_size = Keyword.get(opts, :obs_size, 287)
    action_dim = Keyword.get(opts, :action_dim, 64)
    # chunk_size doesn't affect weight matrix sizes (dense layers are per-element on last dim)
    _chunk_size = Keyword.get(opts, :chunk_size, default_chunk_size())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    latent_dim = Keyword.get(opts, :latent_dim, default_latent_dim())

    # Encoder projections
    enc_proj = obs_size * hidden_size + action_dim * hidden_size + hidden_size * hidden_size
    # Encoder transformer layers (self-attn: 3 * h*h, ffn: h*4h + 4h*h)
    enc_layer = 3 * hidden_size * hidden_size + 2 * hidden_size * 4 * hidden_size
    # Encoder latent output
    enc_latent = hidden_size * latent_dim * 2

    # Decoder projections
    dec_proj = obs_size * hidden_size + latent_dim * hidden_size + hidden_size * hidden_size
    # Decoder transformer layers (self + cross attn: 6 * h*h, ffn: h*4h + 4h*h)
    dec_layer = 6 * hidden_size * hidden_size + 2 * hidden_size * 4 * hidden_size
    # Decoder action output
    dec_out = hidden_size * action_dim

    enc_proj + enc_latent + num_layers * enc_layer +
    dec_proj + dec_out + num_layers * dec_layer
  end

  @doc """
  Get recommended defaults for Melee gameplay (60fps).
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      action_dim: 64,
      chunk_size: 8,
      hidden_size: 256,
      num_layers: 4,
      num_heads: 8,
      latent_dim: 32,
      kl_weight: 10.0,
      ensemble_decay: 0.01
    ]
  end

  @doc """
  Lightweight configuration for fast inference.
  """
  @spec fast_defaults() :: keyword()
  def fast_defaults do
    [
      action_dim: 64,
      chunk_size: 4,
      hidden_size: 128,
      num_layers: 2,
      num_heads: 4,
      latent_dim: 16,
      kl_weight: 10.0,
      ensemble_decay: 0.01
    ]
  end
end
