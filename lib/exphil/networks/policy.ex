defmodule ExPhil.Networks.Policy do
  @moduledoc """
  Policy network for action selection in Melee.

  The policy network takes embedded game state and outputs a probability
  distribution over controller actions. It uses an autoregressive structure
  where each controller component is conditioned on previous samples.

  ## Architecture

  ```
  Embedded State [batch, embed_size]
        │
        ▼
  ┌─────────────┐
  │   Backbone  │  (MLP or Transformer)
  │  [hidden]   │
  └─────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │         Autoregressive Controller Head       │
  │                                              │
  │  buttons → main_x → main_y → c_x → c_y → L  │
  │     ↓         ↓        ↓       ↓      ↓     │
  │  [8 Bern]  [17 Cat] [17 Cat] [17] [17] [5]  │
  └─────────────────────────────────────────────┘
  ```

  ## Autoregressive Sampling

  During inference, we sample each component sequentially:
  1. Sample 8 buttons (independent Bernoulli)
  2. Embed sampled buttons, concatenate with hidden state
  3. Sample main stick X (17-way categorical)
  4. Embed main_x, concatenate, sample main_y
  5. Continue for c_stick and shoulder

  During training, we use teacher forcing (ground truth previous actions).

  ## Module Organization

  This module delegates to focused submodules:
  - `Policy.Backbone` - Backbone network builders (MLP, LSTM, Mamba, etc.)
  - `Policy.Heads` - Controller output heads
  - `Policy.Embeddings` - Action/character embedding preprocessing
  - `Policy.Sampling` - Action sampling functions
  - `Policy.Loss` - Loss computation (BCE, CE, focal)

  ## Usage

      # Build the model
      model = ExPhil.Networks.Policy.build(embed_size: 1024)

      # Initialize parameters
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

      # Forward pass (training)
      logits = predict_fn.(params, embedded_state)

      # Sampling (inference)
      actions = ExPhil.Networks.Policy.sample(model, params, embedded_state)

  ## See Also

  - `ExPhil.Networks.Mamba` - Mamba SSM backbone for temporal modeling
  - `ExPhil.Networks.Attention` - Attention layers for hybrid architectures
  - `ExPhil.Networks.Recurrent` - LSTM/GRU backbones
  - `ExPhil.Embeddings.Controller` - Controller action embedding
  """

  require Axon

  # Submodule aliases
  alias ExPhil.Networks.Policy.Backbone
  alias ExPhil.Networks.Policy.Embeddings
  alias ExPhil.Networks.Policy.Heads
  alias ExPhil.Networks.Policy.Loss
  alias ExPhil.Networks.Policy.Sampling

  # External aliases
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed

  # Default architecture hyperparameters
  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  # Backbone types
  @type backbone_type :: Backbone.backbone_type()

  # Controller output sizes
  @axis_buckets 16
  @shoulder_buckets 4

  # Default embedding IDs
  @default_num_action_ids Embeddings.default_num_action_ids()
  @default_num_character_ids Embeddings.default_num_character_ids()

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build the policy network model.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Discretization for stick axes (default: 16)
    - `:shoulder_buckets` - Discretization for triggers (default: 4)
    - `:layer_norm` - Apply layer normalization (default: false)
    - `:residual` - Use residual connections (default: false)
    - `:action_embed_size` - Size of learned action embedding (default: nil = one-hot)
    - `:num_action_ids` - Number of action IDs in input (default: 2)
    - `:character_embed_size` - Size of learned character embedding (default: nil = one-hot)
    - `:num_character_ids` - Number of character IDs in input (default: 2)

  ## Returns
    An Axon model that outputs a map of logits for each controller component.
  """
  @spec build(keyword()) :: Axon.t() | map()
  def build(opts \\ []) do
    policy_type = Keyword.get(opts, :policy_type, :autoregressive)

    case policy_type do
      :autoregressive ->
        build_autoregressive(opts)

      :diffusion ->
        alias ExPhil.Networks.DiffusionPolicy
        DiffusionPolicy.build(opts)

      :act ->
        alias ExPhil.Networks.ActionChunking
        ActionChunking.build(opts)

      :flow_matching ->
        alias ExPhil.Networks.FlowMatching
        FlowMatching.build(opts)

      other ->
        raise ArgumentError, "Unknown policy type: #{inspect(other)}. " <>
          "Valid types: :autoregressive, :diffusion, :act, :flow_matching"
    end
  end

  @doc """
  Build an autoregressive policy (the default policy type).

  This is the original policy architecture with 6-head autoregressive
  controller prediction.
  """
  @spec build_autoregressive(keyword()) :: Axon.t()
  def build_autoregressive(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)
    action_embed_size = Keyword.get(opts, :action_embed_size, nil)
    num_action_ids = Keyword.get(opts, :num_action_ids, @default_num_action_ids)
    character_embed_size = Keyword.get(opts, :character_embed_size, nil)
    num_character_ids = Keyword.get(opts, :num_character_ids, @default_num_character_ids)

    # Input layer
    input = Axon.input("state", shape: {nil, embed_size})

    # Process input: extract IDs and embed if using learned embeddings
    processed_input =
      Embeddings.build_embedding_preprocessing(
        input,
        embed_size,
        action_embed_size,
        num_action_ids,
        character_embed_size,
        num_character_ids
      )

    # Build backbone
    backbone =
      Backbone.build_backbone(processed_input, hidden_sizes, activation, dropout,
        layer_norm: layer_norm,
        residual: residual
      )

    # Build autoregressive controller head
    Heads.build_controller_head(backbone, axis_buckets, shoulder_buckets)
  end

  @doc """
  Build a temporal policy network that processes frame sequences.

  This version uses attention or recurrent backbones to capture temporal
  patterns in gameplay - essential for understanding combos, reactions,
  and opponent habits.

  ## Options
    - `:embed_size` - Size of per-frame embedding (required)
    - `:backbone` - Backbone type: :sliding_window, :jamba, :lstm_hybrid, :lstm (default: :sliding_window)
    - `:window_size` - Attention window size for sliding_window (default: 60)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:hidden_size` - Hidden size for LSTM/hybrid (default: 256)
    - `:num_layers` - Number of attention/recurrent layers (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Stick discretization (default: 16)
    - `:shoulder_buckets` - Trigger discretization (default: 4)
    - `:action_embed_size` - Size of learned action embedding (default: nil = one-hot)
    - `:character_embed_size` - Size of learned character embedding (default: nil = one-hot)

  ## Input Shape
    `[batch, seq_len, embed_size]` - Sequence of embedded game states

  ## Output
    Map of logits for each controller component (same as `build/1`)
  """
  @spec build_temporal(keyword()) :: Axon.t() | map()
  def build_temporal(opts \\ []) do
    policy_type = Keyword.get(opts, :policy_type, :autoregressive)

    case policy_type do
      :autoregressive ->
        build_temporal_autoregressive(opts)

      :diffusion ->
        # DiffusionPolicy is inherently temporal, just pass temporal options
        alias ExPhil.Networks.DiffusionPolicy
        DiffusionPolicy.build(opts)

      :act ->
        # ActionChunking is inherently temporal with transformer
        alias ExPhil.Networks.ActionChunking
        ActionChunking.build(opts)

      :flow_matching ->
        # FlowMatching can work temporally
        alias ExPhil.Networks.FlowMatching
        FlowMatching.build(opts)

      other ->
        raise ArgumentError, "Unknown policy type: #{inspect(other)}. " <>
          "Valid types: :autoregressive, :diffusion, :act, :flow_matching"
    end
  end

  @doc """
  Build a temporal autoregressive policy network.

  This is the original temporal policy with attention/recurrent backbones
  and 6-head autoregressive controller prediction.
  """
  @spec build_temporal_autoregressive(keyword()) :: Axon.t()
  def build_temporal_autoregressive(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    backbone_type = Keyword.get(opts, :backbone, :sliding_window)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)
    action_embed_size = Keyword.get(opts, :action_embed_size, nil)
    num_action_ids = Keyword.get(opts, :num_action_ids, @default_num_action_ids)
    character_embed_size = Keyword.get(opts, :character_embed_size, nil)
    num_character_ids = Keyword.get(opts, :num_character_ids, @default_num_character_ids)

    # Calculate how many ID slots are in the input tensor
    num_action_slots = if action_embed_size, do: num_action_ids, else: 0
    num_char_slots = if character_embed_size, do: num_character_ids, else: 0

    # Calculate effective embed size after embedding expansion
    continuous_size = embed_size - num_action_slots - num_char_slots

    effective_embed_size =
      continuous_size +
        if(action_embed_size, do: num_action_ids * action_embed_size, else: 0) +
        if character_embed_size, do: num_character_ids * character_embed_size, else: 0

    # Build temporal backbone based on type
    backbone_opts =
      Keyword.merge(opts,
        action_embed_size: action_embed_size,
        character_embed_size: character_embed_size,
        original_embed_size: embed_size,
        effective_embed_size: effective_embed_size,
        num_action_ids: num_action_ids,
        num_character_ids: num_character_ids
      )

    backbone =
      if action_embed_size || character_embed_size do
        build_temporal_with_learned_embeddings(
          effective_embed_size,
          embed_size,
          backbone_type,
          backbone_opts
        )
      else
        Backbone.build_temporal_backbone(effective_embed_size, backbone_type, backbone_opts)
      end

    # Build controller head on top
    Heads.build_controller_head(backbone, axis_buckets, shoulder_buckets)
  end

  # Wrap a temporal backbone with learned embedding preprocessing
  defp build_temporal_with_learned_embeddings(
         _effective_embed_size,
         original_embed_size,
         backbone_type,
         opts
       ) do
    window_size = Keyword.get(opts, :window_size, 60)
    action_embed_size = Keyword.get(opts, :action_embed_size, nil)
    character_embed_size = Keyword.get(opts, :character_embed_size, nil)
    num_action_ids = Keyword.get(opts, :num_action_ids, @default_num_action_ids)
    num_character_ids = Keyword.get(opts, :num_character_ids, @default_num_character_ids)

    # Build temporal embedding preprocessing
    {_input, combined, _effective_embed_size} =
      Embeddings.build_temporal_embedding_preprocessing(
        original_embed_size,
        window_size,
        action_embed_size,
        num_action_ids,
        character_embed_size,
        num_character_ids
      )

    # Build backbone on processed input
    Backbone.build_on_processed_input(combined, backbone_type, opts)
  end

  # ============================================================================
  # Delegated Functions - Embedding Preprocessing
  # ============================================================================

  @doc """
  Build embedding preprocessing that handles both action and character IDs.
  See `ExPhil.Networks.Policy.Embeddings.build_embedding_preprocessing/6`.
  """
  defdelegate build_embedding_preprocessing(
                input,
                embed_size,
                action_embed_size,
                num_action_ids,
                character_embed_size,
                num_character_ids
              ),
              to: Embeddings

  @doc """
  Build combined action and character embedding preprocessing.
  See `ExPhil.Networks.Policy.Embeddings.build_combined_embedding_layer/6`.
  """
  defdelegate build_combined_embedding_layer(
                input,
                continuous_size,
                num_action_ids,
                action_embed_size,
                num_character_ids,
                character_embed_size
              ),
              to: Embeddings

  @doc """
  Build character-only embedding layer.
  See `ExPhil.Networks.Policy.Embeddings.build_character_embedding_layer/4`.
  """
  defdelegate build_character_embedding_layer(
                input,
                total_embed_size,
                character_embed_size,
                num_character_ids
              ),
              to: Embeddings

  @doc """
  Build the action embedding layer.
  See `ExPhil.Networks.Policy.Embeddings.build_action_embedding_layer/4`.
  """
  defdelegate build_action_embedding_layer(
                input,
                total_embed_size,
                action_embed_size,
                num_action_ids
              ),
              to: Embeddings

  # ============================================================================
  # Delegated Functions - Backbone
  # ============================================================================

  @doc """
  Build a temporal backbone that processes frame sequences.
  See `ExPhil.Networks.Policy.Backbone.build_temporal_backbone/3`.
  """
  defdelegate build_temporal_backbone(embed_size, backbone_type, opts \\ []), to: Backbone

  @doc """
  Build just the backbone network (useful for sharing with value network).
  See `ExPhil.Networks.Policy.Backbone.build_backbone/5`.
  """
  defdelegate build_backbone(input, hidden_sizes, activation, dropout, opts \\ []), to: Backbone

  @doc """
  Get the output size of a temporal backbone.
  See `ExPhil.Networks.Policy.Backbone.temporal_backbone_output_size/2`.
  """
  defdelegate temporal_backbone_output_size(backbone_type, opts \\ []), to: Backbone

  @doc """
  Get recommended defaults for Melee temporal policy.
  See `ExPhil.Networks.Policy.Backbone.melee_temporal_defaults/0`.
  """
  defdelegate melee_temporal_defaults(), to: Backbone

  # ============================================================================
  # Delegated Functions - Controller Heads
  # ============================================================================

  @doc """
  Build the autoregressive controller head.
  See `ExPhil.Networks.Policy.Heads.build_controller_head/3`.
  """
  defdelegate build_controller_head(backbone, axis_buckets \\ 16, shoulder_buckets \\ 4),
    to: Heads

  # NOTE: build_autoregressive is now defined locally above (not delegated)
  # since it was extended to support policy_type dispatch.

  @doc """
  Get the output sizes for each controller component.
  See `ExPhil.Networks.Policy.Heads.output_sizes/1`.
  """
  defdelegate output_sizes(opts \\ []), to: Heads

  @doc """
  Calculate the total number of action dimensions.
  See `ExPhil.Networks.Policy.Heads.total_action_dims/1`.
  """
  defdelegate total_action_dims(opts \\ []), to: Heads

  # ============================================================================
  # Delegated Functions - Sampling
  # ============================================================================

  @doc """
  Sample actions from the policy.
  See `ExPhil.Networks.Policy.Sampling.sample/4`.
  """
  defdelegate sample(params, predict_fn, state, opts \\ []), to: Sampling

  @doc """
  Sample buttons from logits (independent Bernoulli).
  See `ExPhil.Networks.Policy.Sampling.sample_buttons/2`.
  """
  defdelegate sample_buttons(logits, deterministic \\ false), to: Sampling

  @doc """
  Sample from categorical distribution with temperature.
  See `ExPhil.Networks.Policy.Sampling.sample_categorical/3`.
  """
  defdelegate sample_categorical(logits, temperature \\ 1.0, deterministic \\ false), to: Sampling

  @doc """
  Compute confidence scores from action logits.
  See `ExPhil.Networks.Policy.Sampling.compute_confidence/1`.
  """
  defdelegate compute_confidence(logits_or_samples), to: Sampling

  # ============================================================================
  # Delegated Functions - Loss
  # ============================================================================

  @doc """
  Compute policy loss (cross-entropy) for imitation learning.
  See `ExPhil.Networks.Policy.Loss.imitation_loss/2`.
  """
  def imitation_loss(logits, targets), do: Loss.imitation_loss(logits, targets)

  @doc """
  Compute imitation loss with options.
  See `ExPhil.Networks.Policy.Loss.imitation_loss/3`.
  """
  def imitation_loss(logits, targets, opts), do: Loss.imitation_loss(logits, targets, opts)

  @doc """
  Binary cross-entropy loss for buttons.
  See `ExPhil.Networks.Policy.Loss.binary_cross_entropy/2`.
  """
  def binary_cross_entropy(logits, targets), do: Loss.binary_cross_entropy(logits, targets)

  @doc """
  Binary cross-entropy loss with label smoothing.
  See `ExPhil.Networks.Policy.Loss.binary_cross_entropy/3`.
  """
  def binary_cross_entropy(logits, targets, label_smoothing),
    do: Loss.binary_cross_entropy(logits, targets, label_smoothing)

  @doc """
  Categorical cross-entropy loss for sticks/shoulder.
  See `ExPhil.Networks.Policy.Loss.categorical_cross_entropy/2`.
  """
  def categorical_cross_entropy(logits, targets),
    do: Loss.categorical_cross_entropy(logits, targets)

  @doc """
  Categorical cross-entropy with label smoothing.
  See `ExPhil.Networks.Policy.Loss.categorical_cross_entropy/3`.
  """
  def categorical_cross_entropy(logits, targets, label_smoothing),
    do: Loss.categorical_cross_entropy(logits, targets, label_smoothing)

  @doc """
  Categorical cross-entropy with per-bucket weighting.
  See `ExPhil.Networks.Policy.Loss.weighted_categorical_cross_entropy/4`.
  """
  defdelegate weighted_categorical_cross_entropy(logits, targets, label_smoothing, edge_weight),
    to: Loss

  @doc """
  Focal binary cross-entropy loss.
  See `ExPhil.Networks.Policy.Loss.focal_binary_cross_entropy/4`.
  """
  defdelegate focal_binary_cross_entropy(logits, targets, label_smoothing, gamma), to: Loss

  @doc """
  Focal categorical cross-entropy loss.
  See `ExPhil.Networks.Policy.Loss.focal_categorical_cross_entropy/4`.
  """
  defdelegate focal_categorical_cross_entropy(logits, targets, label_smoothing, gamma), to: Loss

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Convert sampled action indices to ControllerState.

  ## Options
    - `:axis_buckets` - Number of buckets for uniform discretization (default: 16)
    - `:shoulder_buckets` - Number of buckets for shoulder (default: 4)
    - `:kmeans_centers` - Tensor of K-means cluster centers (overrides axis_buckets)
  """
  @spec to_controller_state(map(), keyword()) :: ExPhil.Bridge.ControllerState.t()
  def to_controller_state(samples, opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)
    kmeans_centers = Keyword.get(opts, :kmeans_centers)

    # Extract first element if batched
    buttons = squeeze_if_batched(samples.buttons)
    main_x = squeeze_if_batched(samples.main_x)
    main_y = squeeze_if_batched(samples.main_y)
    c_x = squeeze_if_batched(samples.c_x)
    c_y = squeeze_if_batched(samples.c_y)
    shoulder = squeeze_if_batched(samples.shoulder)

    # Convert to Elixir values
    button_list = Nx.to_flat_list(buttons)
    [a, b, x, y, z, l, r, d_up] = Enum.map(button_list, &(&1 == 1))

    # Use K-means undiscretization if centers provided
    undiscretize_axis =
      if kmeans_centers do
        fn index -> ExPhil.Embeddings.KMeans.undiscretize(Nx.to_number(index), kmeans_centers) end
      else
        fn index -> ControllerEmbed.undiscretize_axis(Nx.to_number(index), axis_buckets) end
      end

    %ExPhil.Bridge.ControllerState{
      main_stick: %{
        x: undiscretize_axis.(main_x),
        y: undiscretize_axis.(main_y)
      },
      c_stick: %{
        x: undiscretize_axis.(c_x),
        y: undiscretize_axis.(c_y)
      },
      l_shoulder: ControllerEmbed.undiscretize_axis(Nx.to_number(shoulder), shoulder_buckets),
      r_shoulder: 0.0,
      button_a: a,
      button_b: b,
      button_x: x,
      button_y: y,
      button_z: z,
      button_l: l,
      button_r: r,
      button_d_up: d_up
    }
  end

  defp squeeze_if_batched(tensor) do
    case Nx.shape(tensor) do
      # Already scalar
      {} -> tensor
      # Shape {1} -> scalar
      {1} -> Nx.squeeze(tensor)
      {1, _} -> Nx.squeeze(tensor, axes: [0])
      {_, _} -> Nx.slice(tensor, [0, 0], [1, Nx.axis_size(tensor, 1)]) |> Nx.squeeze(axes: [0])
      _ -> tensor
    end
  end
end
