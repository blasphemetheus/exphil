defmodule ExPhil.Networks.Policy.Heads do
  @moduledoc """
  Controller output heads for policy networks.

  Builds the autoregressive controller head that outputs probability
  distributions over Melee controller actions.

  ## Architecture

  ```
  Backbone Output [batch, hidden]
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

  During training: Teacher forcing (ground truth previous actions)
  During inference: Sample sequentially, each conditioned on previous

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Networks.Policy.Sampling` - Action sampling
  """

  require Axon

  # Controller output sizes
  @num_buttons 8
  @default_axis_buckets 16
  @default_shoulder_buckets 4

  @doc """
  Build the autoregressive controller head.

  For training, this outputs logits for each component that can be used
  with cross-entropy loss. The autoregressive conditioning is handled
  by the training loop using teacher forcing.

  ## Parameters
    - `backbone` - Axon layer providing hidden state
    - `axis_buckets` - Number of buckets for stick discretization (default: 16)
    - `shoulder_buckets` - Number of buckets for shoulder (default: 4)

  ## Returns
    Axon container with `{buttons, main_x, main_y, c_x, c_y, shoulder}` logits
  """
  @spec build_controller_head(Axon.t(), non_neg_integer(), non_neg_integer()) :: Axon.t()
  def build_controller_head(backbone, axis_buckets \\ @default_axis_buckets, shoulder_buckets \\ @default_shoulder_buckets) do
    # Each head takes the backbone output and produces logits
    # During training, we compute all heads in parallel (teacher forcing)
    # During inference, we sample sequentially

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # Button logits (8 independent Bernoulli distributions)
    buttons =
      backbone
      |> Axon.dense(64, name: "buttons_hidden")
      |> Axon.relu()
      |> Axon.dense(@num_buttons, name: "buttons_logits")

    # Main stick X
    main_x =
      backbone
      |> Axon.dense(64, name: "main_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_x_logits")

    # Main stick Y
    main_y =
      backbone
      |> Axon.dense(64, name: "main_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_y_logits")

    # C-stick X
    c_x =
      backbone
      |> Axon.dense(64, name: "c_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_x_logits")

    # C-stick Y
    c_y =
      backbone
      |> Axon.dense(64, name: "c_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_y_logits")

    # Shoulder/trigger
    shoulder =
      backbone
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

  ## Options
    - `:embed_size` - Size of input embedding (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:axis_buckets` - Discretization for stick axes (default: 16)
    - `:shoulder_buckets` - Discretization for triggers (default: 4)
    - `:layer_norm` - Apply layer normalization (default: false)
    - `:residual` - Use residual connections (default: false)
  """
  @spec build_autoregressive(keyword()) :: Axon.t()
  def build_autoregressive(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, [512, 512])
    activation = Keyword.get(opts, :activation, :relu)
    dropout = Keyword.get(opts, :dropout, 0.1)
    axis_buckets = Keyword.get(opts, :axis_buckets, @default_axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @default_shoulder_buckets)
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)

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
    backbone =
      build_mlp_backbone(state_input, hidden_sizes, activation, dropout,
        layer_norm: layer_norm,
        residual: residual
      )

    # Buttons head (no conditioning, first in sequence)
    buttons =
      backbone
      |> Axon.dense(64, name: "buttons_hidden")
      |> Axon.relu()
      |> Axon.dense(@num_buttons, name: "buttons_logits")

    # Main X head (conditioned on buttons)
    main_x_input = Axon.concatenate([backbone, prev_buttons], name: "main_x_concat")

    main_x =
      main_x_input
      |> Axon.dense(64, name: "main_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_x_logits")

    # Main Y head (conditioned on buttons + main_x)
    main_y_input = Axon.concatenate([backbone, prev_buttons, prev_main_x], name: "main_y_concat")

    main_y =
      main_y_input
      |> Axon.dense(64, name: "main_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "main_y_logits")

    # C-stick X (conditioned on buttons + main stick)
    c_x_input =
      Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y], name: "c_x_concat")

    c_x =
      c_x_input
      |> Axon.dense(64, name: "c_x_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_x_logits")

    # C-stick Y
    c_y_input =
      Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y, prev_c_x],
        name: "c_y_concat"
      )

    c_y =
      c_y_input
      |> Axon.dense(64, name: "c_y_hidden")
      |> Axon.relu()
      |> Axon.dense(axis_size, name: "c_y_logits")

    # Shoulder (conditioned on everything)
    shoulder_input =
      Axon.concatenate([backbone, prev_buttons, prev_main_x, prev_main_y, prev_c_x, prev_c_y],
        name: "shoulder_concat"
      )

    shoulder =
      shoulder_input
      |> Axon.dense(32, name: "shoulder_hidden")
      |> Axon.relu()
      |> Axon.dense(shoulder_size, name: "shoulder_logits")

    Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
  end

  @doc """
  Get the output sizes for each controller component.

  ## Options
    - `:axis_buckets` - Stick discretization (default: 16)
    - `:shoulder_buckets` - Shoulder discretization (default: 4)
  """
  @spec output_sizes(keyword()) :: map()
  def output_sizes(opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, @default_axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @default_shoulder_buckets)

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

  # Simple MLP backbone for autoregressive model
  defp build_mlp_backbone(input, hidden_sizes, activation, dropout, opts) do
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)

    {final_layer, _} =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce({input, nil}, fn {size, idx}, {acc, prev_size} ->
        layer =
          acc
          |> Axon.dense(size, name: "backbone_dense_#{idx}")

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

        layer =
          if residual do
            add_residual(acc, layer, prev_size, size, idx)
          else
            layer
          end

        {layer, size}
      end)

    final_layer
  end

  defp add_residual(input, output, prev_size, current_size, idx) do
    if prev_size == current_size do
      Axon.add(input, output, name: "backbone_residual_#{idx}")
    else
      projected = Axon.dense(input, current_size, name: "backbone_proj_#{idx}")
      Axon.add(projected, output, name: "backbone_residual_#{idx}")
    end
  end
end
