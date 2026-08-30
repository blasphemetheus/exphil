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

  # Default residual stream width for the autoregressive head
  # (slippi-ai AutoRegressive: residual_size 128, component_depth 0)
  @default_residual_size 128
  # Hidden width of each per-component MLP (matches the independent heads)
  @default_component_hidden 64

  @doc """
  Build the TRUE autoregressive controller head (AUTOREGRESSIVE_HEAD_PLAN §3).

  Within a frame the six components are predicted sequentially, each
  conditioned on the components before it via a residual stream:

      r0 = W · trunk                     (dense -> residual_size)
      logits_k = MLP_k(r_{k-1})          (64-hidden, same size as today's heads)
      r_k = r_{k-1} + E_k(a_k)           (zeros-init embed of the component)

  Order: buttons -> main_x -> main_y -> c_x -> c_y -> shoulder
  (coarse-to-fine; the control-flow-bearing component first).

  **Training / offline scoring (this graph): teacher forcing.** The same
  frame's TARGET components arrive as extra inputs, so all six logits are
  computed in ONE forward pass — no sequential loop:

    - `"tf_buttons"`  {batch, 8}  float multi-hot (the target buttons)
    - `"tf_main_x"` / `"tf_main_y"` / `"tf_c_x"` / `"tf_c_y"`  {batch} int
      bucket indices (the target sticks)

  The shoulder needs no `tf_` input — nothing is conditioned on it.
  Because every `E_k` is zero-initialised (slippi-ai's decoder trick),
  the head starts out exactly equivalent to independent heads and learns
  the conditioning from zero — gradients flow into `E_k` from the
  components after k.

  **Inference** does NOT run this graph directly — see
  `ExPhil.Networks.Policy.Sampling.sample_autoregressive/4`, which
  replays the same math sequentially from the exported params (layer
  names below are its contract).

  ## Parameters
    - `trunk` - Axon node producing `[batch, hidden]` features (a temporal
      trunk, or an `Axon.input("trunk", ...)` for head-only fits)

  ## Options
    - `:axis_buckets` (default 16), `:shoulder_buckets` (default 4)
    - `:residual_size` - residual stream width (default 128)
    - `:component_hidden` - per-component MLP hidden width (default 64)

  ## Returns
    Axon container `{buttons, main_x, main_y, c_x, c_y, shoulder}` logits —
    same shape contract as `build_controller_head/3`.
  """
  @spec build_autoregressive_head(Axon.t(), keyword()) :: Axon.t()
  def build_autoregressive_head(trunk, opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, @default_axis_buckets)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, @default_shoulder_buckets)
    residual_size = Keyword.get(opts, :residual_size, @default_residual_size)
    component_hidden = Keyword.get(opts, :component_hidden, @default_component_hidden)

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # Teacher-forced component inputs (targets of the SAME frame)
    tf_buttons = Axon.input("tf_buttons", shape: {nil, @num_buttons})
    tf_main_x = Axon.input("tf_main_x", shape: {nil})
    tf_main_y = Axon.input("tf_main_y", shape: {nil})
    tf_c_x = Axon.input("tf_c_x", shape: {nil})
    tf_c_y = Axon.input("tf_c_y", shape: {nil})

    # r0: project trunk features into the residual stream
    r0 = Axon.dense(trunk, residual_size, name: "ar_residual_proj")

    # Component embeddings E_k — ZERO-initialised so the head starts as
    # an exact independent factorization (conditioning is learned, and the
    # teacher-forced graph equals the sequential replay at init).
    embed_buttons =
      Axon.dense(tf_buttons, residual_size,
        name: "ar_buttons_embed",
        use_bias: false,
        kernel_initializer: :zeros
      )

    embed_cat = fn input, vocab, name ->
      Axon.embedding(input, vocab, residual_size,
        name: name,
        kernel_initializer: :zeros
      )
    end

    component = fn r, out_size, prefix ->
      r
      |> Axon.dense(component_hidden, name: "#{prefix}_hidden")
      |> Axon.relu()
      |> Axon.dense(out_size, name: "#{prefix}_logits")
    end

    # buttons <- r0
    buttons = component.(r0, @num_buttons, "ar_buttons")
    r1 = Axon.add(r0, embed_buttons, name: "ar_r1")

    # main_x <- r1 (conditioned on buttons)
    main_x = component.(r1, axis_size, "ar_main_x")
    r2 = Axon.add(r1, embed_cat.(tf_main_x, axis_size, "ar_main_x_embed"), name: "ar_r2")

    # main_y <- r2 (conditioned on buttons + main_x)
    main_y = component.(r2, axis_size, "ar_main_y")
    r3 = Axon.add(r2, embed_cat.(tf_main_y, axis_size, "ar_main_y_embed"), name: "ar_r3")

    # c_x <- r3
    c_x = component.(r3, axis_size, "ar_c_x")
    r4 = Axon.add(r3, embed_cat.(tf_c_x, axis_size, "ar_c_x_embed"), name: "ar_r4")

    # c_y <- r4
    c_y = component.(r4, axis_size, "ar_c_y")
    r5 = Axon.add(r4, embed_cat.(tf_c_y, axis_size, "ar_c_y_embed"), name: "ar_r5")

    # shoulder <- r5 (conditioned on everything; nothing conditions on it)
    shoulder = component.(r5, shoulder_size, "ar_shoulder")

    Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
  end

  @doc """
  Template map for the teacher-forced inputs of the autoregressive head.

  Merge with the state input template when initialising or warming a model
  built with `build_autoregressive_head/2`:

      Map.merge(%{"state_sequence" => state_template}, Heads.tf_templates(1))
  """
  @spec tf_templates(pos_integer()) :: %{String.t() => Nx.Tensor.t()}
  def tf_templates(batch \\ 1) do
    %{
      "tf_buttons" => Nx.template({batch, @num_buttons}, :f32),
      "tf_main_x" => Nx.template({batch}, :s64),
      "tf_main_y" => Nx.template({batch}, :s64),
      "tf_c_x" => Nx.template({batch}, :s64),
      "tf_c_y" => Nx.template({batch}, :s64)
    }
  end

  @doc """
  Build the teacher-forced input map for the autoregressive head from a
  targets/actions map (`%{buttons: {b,8}, main_x: {b}, ...}` as produced
  by `ExPhil.Training.Data.actions_to_tensors/1`). Buttons are cast to f32
  (they arrive as s64 multi-hot).
  """
  @spec tf_inputs(map()) :: %{String.t() => Nx.Tensor.t()}
  def tf_inputs(actions) do
    %{
      "tf_buttons" => Nx.as_type(actions.buttons, :f32),
      "tf_main_x" => actions.main_x,
      "tf_main_y" => actions.main_y,
      "tf_c_x" => actions.c_x,
      "tf_c_y" => actions.c_y
    }
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

end
