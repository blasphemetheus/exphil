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

  """

  require Axon

  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Networks.Attention
  alias ExPhil.Networks.Mamba
  alias ExPhil.Networks.Recurrent

  # Default architecture hyperparameters
  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  # Backbone types
  @type backbone_type :: :mlp | :sliding_window | :hybrid | :lstm | :gru | :mamba

  # Controller output sizes
  @num_buttons 8
  @axis_buckets 16  # 0-16 = 17 values
  @shoulder_buckets 4  # 0-4 = 5 values

  @doc """
  Build the policy network model.

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Discretization for stick axes (default: 16)
    - `:shoulder_buckets` - Discretization for triggers (default: 4)

  ## Returns
    An Axon model that outputs a map of logits for each controller component.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

    # Input layer
    input = Axon.input("state", shape: {nil, embed_size})

    # Build backbone
    backbone = build_backbone(input, hidden_sizes, activation, dropout)

    # Build autoregressive controller head
    build_controller_head(backbone, axis_buckets, shoulder_buckets)
  end

  @doc """
  Build a temporal policy network that processes frame sequences.

  This version uses attention or recurrent backbones to capture temporal
  patterns in gameplay - essential for understanding combos, reactions,
  and opponent habits.

  ## Options
    - `:embed_size` - Size of per-frame embedding (required)
    - `:backbone` - Backbone type: :sliding_window, :hybrid, :lstm (default: :sliding_window)
    - `:window_size` - Attention window size for sliding_window (default: 60)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:hidden_size` - Hidden size for LSTM/hybrid (default: 256)
    - `:num_layers` - Number of attention/recurrent layers (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Stick discretization (default: 16)
    - `:shoulder_buckets` - Trigger discretization (default: 4)

  ## Input Shape
    `[batch, seq_len, embed_size]` - Sequence of embedded game states

  ## Output
    Map of logits for each controller component (same as `build/1`)
  """
  @spec build_temporal(keyword()) :: Axon.t()
  def build_temporal(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    backbone_type = Keyword.get(opts, :backbone, :sliding_window)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

    # Build temporal backbone based on type
    backbone = build_temporal_backbone(embed_size, backbone_type, opts)

    # Build controller head on top
    build_controller_head(backbone, axis_buckets, shoulder_buckets)
  end

  @doc """
  Build a temporal backbone that processes frame sequences.

  Returns an Axon layer that outputs [batch, hidden_dim] from sequence input.
  """
  @spec build_temporal_backbone(non_neg_integer(), backbone_type(), keyword()) :: Axon.t()
  def build_temporal_backbone(embed_size, backbone_type, opts \\ []) do
    case backbone_type do
      :sliding_window ->
        build_sliding_window_backbone(embed_size, opts)

      :hybrid ->
        build_hybrid_backbone(embed_size, opts)

      :lstm ->
        build_lstm_backbone(embed_size, opts)

      :gru ->
        build_gru_backbone(embed_size, opts)

      :mamba ->
        build_mamba_backbone(embed_size, opts)

      :mlp ->
        # For MLP, expect single frame input, add sequence handling
        build_mlp_temporal_backbone(embed_size, opts)
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

  defp build_hybrid_backbone(embed_size, opts) do
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
      window_size: window_size  # For concrete seq_len (efficient JIT)
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
      window_size: window_size,  # For concrete seq_len (efficient JIT)
      truncate_bptt: truncate_bptt  # Optional: limit gradient flow for faster training
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

  defp build_mamba_backbone(embed_size, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    Mamba.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size
    )
  end

  defp build_mlp_temporal_backbone(embed_size, opts) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    # Sequence length configuration (same as attention models)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: sequence [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Take last frame: [batch, embed_size]
    # Use concrete index when available for efficient compilation
    last_frame = Axon.nx(input, fn tensor ->
      last_idx = if seq_len do
        seq_len - 1
      else
        Nx.axis_size(tensor, 1) - 1
      end
      Nx.slice_along_axis(tensor, last_idx, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "last_frame")

    # Apply MLP backbone
    build_backbone(last_frame, hidden_sizes, activation, dropout)
  end

  @doc """
  Build just the backbone network (useful for sharing with value network).
  """
  @spec build_backbone(Axon.t(), list(), atom(), float()) :: Axon.t()
  def build_backbone(input, hidden_sizes, activation, dropout) do
    hidden_sizes
    |> Enum.with_index()
    |> Enum.reduce(input, fn {size, idx}, acc ->
      acc
      |> Axon.dense(size, name: "backbone_dense_#{idx}")
      |> Axon.activation(activation)
      |> Axon.dropout(rate: dropout)
    end)
  end

  @doc """
  Build the autoregressive controller head.

  For training, this outputs logits for each component that can be used
  with cross-entropy loss. The autoregressive conditioning is handled
  by the training loop using teacher forcing.
  """
  @spec build_controller_head(Axon.t(), non_neg_integer(), non_neg_integer()) :: Axon.t()
  def build_controller_head(backbone, axis_buckets, shoulder_buckets) do
    # Each head takes the backbone output and produces logits
    # During training, we compute all heads in parallel (teacher forcing)
    # During inference, we sample sequentially

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # Button logits (8 independent Bernoulli distributions)
    buttons = backbone
    |> Axon.dense(64, name: "buttons_hidden")
    |> Axon.relu()
    |> Axon.dense(@num_buttons, name: "buttons_logits")

    # Main stick X
    main_x = backbone
    |> Axon.dense(64, name: "main_x_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "main_x_logits")

    # Main stick Y
    main_y = backbone
    |> Axon.dense(64, name: "main_y_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "main_y_logits")

    # C-stick X
    c_x = backbone
    |> Axon.dense(64, name: "c_x_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "c_x_logits")

    # C-stick Y
    c_y = backbone
    |> Axon.dense(64, name: "c_y_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "c_y_logits")

    # Shoulder/trigger
    shoulder = backbone
    |> Axon.dense(32, name: "shoulder_hidden")
    |> Axon.relu()
    |> Axon.dense(shoulder_size, name: "shoulder_logits")

    # Combine into a container output
    Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
  end

  @doc """
  Build the full autoregressive policy with conditioning.

  This version includes embedding layers for conditioning each component
  on previously sampled components. Used during inference.
  """
  @spec build_autoregressive(keyword()) :: Axon.t()
  def build_autoregressive(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # State input
    state_input = Axon.input("state", shape: {nil, embed_size})

    # Previous action inputs (for conditioning)
    prev_buttons = Axon.input("prev_buttons", shape: {nil, @num_buttons})
    prev_main_x = Axon.input("prev_main_x", shape: {nil, axis_size})
    prev_main_y = Axon.input("prev_main_y", shape: {nil, axis_size})
    prev_c_x = Axon.input("prev_c_x", shape: {nil, axis_size})
    prev_c_y = Axon.input("prev_c_y", shape: {nil, axis_size})

    # Backbone
    backbone = build_backbone(state_input, hidden_sizes, activation, dropout)

    # Buttons head (no conditioning, first in sequence)
    buttons = backbone
    |> Axon.dense(64, name: "buttons_hidden")
    |> Axon.relu()
    |> Axon.dense(@num_buttons, name: "buttons_logits")

    # Main X head (conditioned on buttons)
    main_x_input = Axon.concatenate([backbone, prev_buttons], name: "main_x_concat")
    main_x = main_x_input
    |> Axon.dense(64, name: "main_x_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "main_x_logits")

    # Main Y head (conditioned on buttons + main_x)
    main_y_input = Axon.concatenate([backbone, prev_buttons, prev_main_x], name: "main_y_concat")
    main_y = main_y_input
    |> Axon.dense(64, name: "main_y_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "main_y_logits")

    # C-stick X (conditioned on buttons + main stick)
    c_x_input = Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y], name: "c_x_concat")
    c_x = c_x_input
    |> Axon.dense(64, name: "c_x_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "c_x_logits")

    # C-stick Y
    c_y_input = Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y, prev_c_x], name: "c_y_concat")
    c_y = c_y_input
    |> Axon.dense(64, name: "c_y_hidden")
    |> Axon.relu()
    |> Axon.dense(axis_size, name: "c_y_logits")

    # Shoulder (conditioned on everything)
    shoulder_input = Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y, prev_c_x, prev_c_y], name: "shoulder_concat")
    shoulder = shoulder_input
    |> Axon.dense(32, name: "shoulder_hidden")
    |> Axon.relu()
    |> Axon.dense(shoulder_size, name: "shoulder_logits")

    Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
  end

  # ============================================================================
  # Sampling Functions
  # ============================================================================

  @doc """
  Sample actions from the policy.

  This performs autoregressive sampling, where each controller component
  is sampled conditioned on previously sampled components.

  ## Options
    - `:temperature` - Softmax temperature for exploration (default: 1.0)
    - `:deterministic` - If true, use argmax instead of sampling (default: false)
  """
  @spec sample(map(), Nx.Tensor.t(), keyword()) :: map()
  def sample(params, predict_fn, state, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

    _axis_size = axis_buckets + 1
    _shoulder_size = shoulder_buckets + 1

    # Get batch size
    _batch_size = Nx.axis_size(state, 0)

    # Forward pass to get all logits
    {buttons_logits, main_x_logits, main_y_logits, c_x_logits, c_y_logits, shoulder_logits} =
      predict_fn.(params, state)

    # Sample buttons (independent Bernoulli)
    buttons = sample_buttons(buttons_logits, deterministic)

    # Sample stick and shoulder (categorical)
    main_x = sample_categorical(main_x_logits, temperature, deterministic)
    main_y = sample_categorical(main_y_logits, temperature, deterministic)
    c_x = sample_categorical(c_x_logits, temperature, deterministic)
    c_y = sample_categorical(c_y_logits, temperature, deterministic)
    shoulder = sample_categorical(shoulder_logits, temperature, deterministic)

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder,
      # Also include logits for loss computation
      logits: %{
        buttons: buttons_logits,
        main_x: main_x_logits,
        main_y: main_y_logits,
        c_x: c_x_logits,
        c_y: c_y_logits,
        shoulder: shoulder_logits
      }
    }
  end

  @doc """
  Sample buttons from logits (independent Bernoulli).
  """
  @spec sample_buttons(Nx.Tensor.t(), boolean()) :: Nx.Tensor.t()
  def sample_buttons(logits, deterministic \\ false) do
    probs = Nx.sigmoid(logits)

    if deterministic do
      Nx.greater(probs, 0.5)
    else
      # Sample from Bernoulli using Nx.Random
      key = Nx.Random.key(System.system_time())
      {random, _new_key} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      Nx.less(random, probs)
    end
  end

  @doc """
  Sample from categorical distribution with temperature.
  """
  @spec sample_categorical(Nx.Tensor.t(), float(), boolean()) :: Nx.Tensor.t()
  def sample_categorical(logits, temperature \\ 1.0, deterministic \\ false) do
    if deterministic do
      Nx.argmax(logits, axis: -1)
    else
      # Apply temperature
      scaled_logits = Nx.divide(logits, temperature)

      # Gumbel-max trick for sampling
      key = Nx.Random.key(System.system_time())
      {gumbel_noise, _new_key} = Nx.Random.uniform(key, shape: Nx.shape(scaled_logits))
      gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(gumbel_noise, 1.0e-10)))))

      Nx.argmax(Nx.add(scaled_logits, gumbel), axis: -1)
    end
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute policy loss (cross-entropy) for imitation learning.

  ## Parameters
    - `logits` - Map of logits from forward pass
    - `targets` - Map of target actions (ground truth from replay)

  ## Returns
    Total loss (sum of button BCE + categorical CE for sticks/shoulder)
  """
  @spec imitation_loss(map(), map()) :: Nx.Tensor.t()
  def imitation_loss(logits, targets) do
    imitation_loss(logits, targets, label_smoothing: 0.0)
  end

  @doc """
  Compute imitation loss with optional label smoothing.

  Label smoothing prevents overconfidence by replacing hard targets with soft targets:
  - For categorical targets: target class gets (1-ε), others get ε/(num_classes-1)
  - For binary targets: target gets (1-ε), non-target gets ε

  ## Options
    - `:label_smoothing` - Smoothing factor ε (default: 0.0, typical: 0.1)
  """
  @spec imitation_loss(map(), map(), keyword()) :: Nx.Tensor.t()
  def imitation_loss(logits, targets, opts) do
    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)

    # Button loss (binary cross-entropy with optional label smoothing)
    button_loss = binary_cross_entropy(logits.buttons, targets.buttons, label_smoothing)

    # Stick losses (categorical cross-entropy with optional label smoothing)
    main_x_loss = categorical_cross_entropy(logits.main_x, targets.main_x, label_smoothing)
    main_y_loss = categorical_cross_entropy(logits.main_y, targets.main_y, label_smoothing)
    c_x_loss = categorical_cross_entropy(logits.c_x, targets.c_x, label_smoothing)
    c_y_loss = categorical_cross_entropy(logits.c_y, targets.c_y, label_smoothing)

    # Shoulder loss
    shoulder_loss = categorical_cross_entropy(logits.shoulder, targets.shoulder, label_smoothing)

    # Combine losses
    Nx.add(button_loss,
      Nx.add(main_x_loss,
        Nx.add(main_y_loss,
          Nx.add(c_x_loss,
            Nx.add(c_y_loss, shoulder_loss)))))
  end

  @doc """
  Binary cross-entropy loss for buttons with optional label smoothing.

  With label smoothing ε > 0:
  - Target 1 becomes (1-ε)
  - Target 0 becomes ε

  This prevents the model from becoming overconfident on button predictions.
  """
  @spec binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def binary_cross_entropy(logits, targets) do
    binary_cross_entropy(logits, targets, 0.0)
  end

  @spec binary_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def binary_cross_entropy(logits, targets, label_smoothing) do
    # Apply label smoothing to targets
    # Smoothed targets: t_smooth = t * (1 - ε) + (1 - t) * ε = t * (1 - 2ε) + ε
    smoothed_targets = if label_smoothing > 0.0 do
      Nx.add(
        Nx.multiply(targets, 1.0 - 2.0 * label_smoothing),
        label_smoothing
      )
    else
      targets
    end

    # Numerically stable BCE
    # loss = max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    max_val = Nx.max(logits, 0)
    abs_logits = Nx.abs(logits)

    loss = Nx.subtract(max_val, Nx.multiply(logits, smoothed_targets))
    loss = Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_logits)))))

    Nx.mean(loss)
  end

  @doc """
  Categorical cross-entropy loss for discretized sticks/shoulder with optional label smoothing.

  With label smoothing ε > 0:
  - Target class gets probability (1-ε)
  - Other classes share ε equally: ε/(num_classes-1)

  This encourages the model to be less overconfident on stick/shoulder predictions.
  """
  @spec categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def categorical_cross_entropy(logits, targets) do
    categorical_cross_entropy(logits, targets, 0.0)
  end

  @spec categorical_cross_entropy(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def categorical_cross_entropy(logits, targets, label_smoothing) do
    # targets are indices, logits are [batch, num_classes]
    # Use log_softmax for numerical stability
    log_probs = log_softmax(logits)

    # Gather the log probability of the target class
    # targets: [batch], log_probs: [batch, num_classes]
    batch_size = Nx.axis_size(logits, 0)
    num_classes = Nx.axis_size(logits, 1)

    # Create one-hot targets
    targets_one_hot = Nx.equal(
      Nx.iota({batch_size, num_classes}, axis: 1),
      Nx.reshape(targets, {batch_size, 1})
    )

    # Apply label smoothing if enabled
    # Smoothed one-hot: (1-ε) for target class, ε/(n-1) for others
    smoothed_targets = if label_smoothing > 0.0 do
      # off_value = ε / (n-1)
      # on_value = 1 - ε
      off_value = label_smoothing / (num_classes - 1)
      on_value = 1.0 - label_smoothing

      # Start with uniform ε/(n-1), then add (1-ε - ε/(n-1)) to target class
      # = off_value + targets_one_hot * (on_value - off_value)
      Nx.add(
        off_value,
        Nx.multiply(targets_one_hot, on_value - off_value)
      )
    else
      targets_one_hot
    end

    # Cross-entropy with soft targets: -sum(p * log_q)
    nll = Nx.negate(Nx.sum(Nx.multiply(log_probs, smoothed_targets), axes: [1]))
    Nx.mean(nll)
  end

  defp log_softmax(logits) do
    max_val = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_val)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Convert sampled action indices to ControllerState.
  """
  @spec to_controller_state(map(), keyword()) :: ExPhil.Bridge.ControllerState.t()
  def to_controller_state(samples, opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

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

    %ExPhil.Bridge.ControllerState{
      main_stick: %{
        x: ControllerEmbed.undiscretize_axis(Nx.to_number(main_x), axis_buckets),
        y: ControllerEmbed.undiscretize_axis(Nx.to_number(main_y), axis_buckets)
      },
      c_stick: %{
        x: ControllerEmbed.undiscretize_axis(Nx.to_number(c_x), axis_buckets),
        y: ControllerEmbed.undiscretize_axis(Nx.to_number(c_y), axis_buckets)
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
      {} -> tensor  # Already scalar
      {1} -> Nx.squeeze(tensor)  # Shape {1} -> scalar
      {1, _} -> Nx.squeeze(tensor, axes: [0])
      {_, _} -> Nx.slice(tensor, [0, 0], [1, Nx.axis_size(tensor, 1)]) |> Nx.squeeze(axes: [0])
      _ -> tensor
    end
  end

  @doc """
  Get the output sizes for each controller component.
  """
  @spec output_sizes(keyword()) :: map()
  def output_sizes(opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, @axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @shoulder_buckets)

    %{
      buttons: @num_buttons,
      main_x: axis_buckets + 1,
      main_y: axis_buckets + 1,
      c_x: axis_buckets + 1,
      c_y: axis_buckets + 1,
      shoulder: shoulder_buckets + 1
    }
  end

  @doc """
  Calculate the total number of action dimensions.
  """
  @spec total_action_dims(keyword()) :: non_neg_integer()
  def total_action_dims(opts \\ []) do
    sizes = output_sizes(opts)
    sizes.buttons + sizes.main_x + sizes.main_y + sizes.c_x + sizes.c_y + sizes.shoulder
  end

  @doc """
  Get the output size of a temporal backbone.

  Useful for connecting to other networks (value function, etc).
  """
  @spec temporal_backbone_output_size(backbone_type(), keyword()) :: non_neg_integer()
  def temporal_backbone_output_size(backbone_type, opts \\ []) do
    case backbone_type do
      :sliding_window ->
        num_heads = Keyword.get(opts, :num_heads, 4)
        head_dim = Keyword.get(opts, :head_dim, 64)
        num_heads * head_dim

      :hybrid ->
        num_heads = Keyword.get(opts, :num_heads, 4)
        head_dim = Keyword.get(opts, :head_dim, 64)
        num_heads * head_dim

      :lstm ->
        Keyword.get(opts, :hidden_size, 256)

      :gru ->
        Keyword.get(opts, :hidden_size, 256)

      :mamba ->
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
end
