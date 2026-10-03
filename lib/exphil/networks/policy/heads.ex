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
  Chunk-target auxiliary heads (2026-10-02): `horizon` independent
  six-component heads on the same trunk features, head j predicting the
  controller at t+j. Training only — the live model is built without them
  and `Imitation.export_policy/2` drops their `future*` params.

  Why: with the prev-action channel, frame t's controller is explained
  almost entirely by frame t-1's (the copy shortcut), so the trunk's state
  pathway gets little gradient and the live policy holds. The controller
  at t+j is NOT in the channel; the trunk must read the game state to
  score it, so the shortcut loses its monopoly on the gradient. Nothing
  conditions on these heads at inference.

  Returns an Axon container `{head_1, ..., head_horizon}`, each a
  `{buttons, main_x, main_y, c_x, c_y, shoulder}` logits container.
  """
  @spec build_future_heads(Axon.t(), pos_integer(), non_neg_integer(), non_neg_integer()) :: Axon.t()
  def build_future_heads(backbone, horizon, axis_buckets \\ @default_axis_buckets, shoulder_buckets \\ @default_shoulder_buckets) do
    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    mlp = fn hidden, out, name ->
      backbone
      |> Axon.dense(hidden, name: "#{name}_hidden")
      |> Axon.relu()
      |> Axon.dense(out, name: "#{name}_logits")
    end

    for j <- 1..horizon do
      p = "future#{j}_"

      Axon.container(
        {mlp.(64, @num_buttons, p <> "buttons"), mlp.(64, axis_size, p <> "main_x"),
         mlp.(64, axis_size, p <> "main_y"), mlp.(64, axis_size, p <> "c_x"),
         mlp.(64, axis_size, p <> "c_y"), mlp.(32, shoulder_size, p <> "shoulder")}
      )
    end
    |> List.to_tuple()
    |> Axon.container()
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
    - `:per_timestep` - trunk is a full sequence `[batch, time, hidden]`
      and the tf inputs carry a time dim (`{b, t, 8}` / `{b, t}`); every
      dense/embedding below broadcasts over the extra axis, so logits come
      out `[batch, time, k]` with the SAME param names/shapes (BPTT
      training, default false)

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
    per_timestep = Keyword.get(opts, :per_timestep, false)
    # Press/release event buttons (2026-10-02): an Axon node {batch, 8} with
    # the PREVIOUS frame's button states, or nil for the plain state head.
    button_events_prev = Keyword.get(opts, :button_events_prev)

    # Hold-or-change stick heads (2026-10-02): an Axon node {batch, 4} (s64)
    # with the PREVIOUS frame's bucket for main_x, main_y, c_x, c_y, or nil.
    stick_events_prev = Keyword.get(opts, :stick_events_prev)

    if (button_events_prev || stick_events_prev) && per_timestep do
      raise ArgumentError, "button/stick events are not supported with per_timestep (BPTT) heads"
    end

    axis_size = axis_buckets + 1
    shoulder_size = shoulder_buckets + 1

    # Teacher-forced component inputs (targets of the SAME frame).
    # per_timestep adds a time dim; params are rank-agnostic (dense acts on
    # the last axis, embedding adds one), so both variants share weights.
    {buttons_shape, cat_shape} =
      if per_timestep,
        do: {{nil, nil, @num_buttons}, {nil, nil}},
        else: {{nil, @num_buttons}, {nil}}

    tf_buttons = Axon.input("tf_buttons", shape: buttons_shape)
    tf_main_x = Axon.input("tf_main_x", shape: cat_shape)
    tf_main_y = Axon.input("tf_main_y", shape: cat_shape)
    tf_c_x = Axon.input("tf_c_x", shape: cat_shape)
    tf_c_y = Axon.input("tf_c_y", shape: cat_shape)

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

    # A stick axis with hold-or-change: K change logits + 1 hold logit,
    # collapsed to K log-probabilities given the previous bucket (column j).
    stick_component = fn r, out_size, prefix, j ->
      case stick_events_prev do
        nil ->
          component.(r, out_size, prefix)

        prev ->
          raw = component.(r, out_size + 1, prefix)

          Axon.layer(
            fn raw, prev, _opts ->
              col = prev |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1])
              collapse_hold_change(raw, col)
            end,
            [raw, prev],
            name: "#{prefix}_hold_change",
            op_name: :hold_change
          )
      end
    end

    # buttons <- r0
    buttons =
      case button_events_prev do
        nil ->
          component.(r0, @num_buttons, "ar_buttons")

        prev ->
          raw = component.(r0, 2 * @num_buttons, "ar_buttons")

          Axon.layer(fn raw, prev, _opts -> collapse_button_events(raw, prev) end, [raw, prev],
            name: "ar_button_events",
            op_name: :button_events
          )
      end

    r1 = Axon.add(r0, embed_buttons, name: "ar_r1")

    # main_x <- r1 (conditioned on buttons)
    main_x = stick_component.(r1, axis_size, "ar_main_x", 0)
    r2 = Axon.add(r1, embed_cat.(tf_main_x, axis_size, "ar_main_x_embed"), name: "ar_r2")

    # main_y <- r2 (conditioned on buttons + main_x)
    main_y = stick_component.(r2, axis_size, "ar_main_y", 1)
    r3 = Axon.add(r2, embed_cat.(tf_main_y, axis_size, "ar_main_y_embed"), name: "ar_r3")

    # c_x <- r3
    c_x = stick_component.(r3, axis_size, "ar_c_x", 2)
    r4 = Axon.add(r3, embed_cat.(tf_c_x, axis_size, "ar_c_x_embed"), name: "ar_r4")

    # c_y <- r4
    c_y = stick_component.(r4, axis_size, "ar_c_y", 3)
    r5 = Axon.add(r4, embed_cat.(tf_c_y, axis_size, "ar_c_y_embed"), name: "ar_r5")

    # shoulder <- r5 (conditioned on everything; nothing conditions on it)
    shoulder = component.(r5, shoulder_size, "ar_shoulder")

    Axon.container({buttons, main_x, main_y, c_x, c_y, shoulder})
  end

  @doc """
  Collapse press/release event logits into ordinary "button is down" logits.

  `raw` is `{batch, 16}`: columns 0..7 are PRESS logits (probability the
  button goes down given it was up), columns 8..15 are RELEASE logits
  (probability it comes up given it was down). `prev` is `{batch, 8}` (or
  `{1, 8}`, broadcast) with the previous frame's button states. The result
  is `{batch, 8}` logits of "down this frame":

      up last frame    ->  press logit
      down last frame  -> -release logit

  so the usual BCE against the button STATE target trains the press head on
  frames that start up and the release head on frames that start down, and
  the usual per-frame Bernoulli sampler draws hazards instead of states.
  The previous state selects the head; it is never an input to the trunk.
  """
  @spec collapse_button_events(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def collapse_button_events(raw, prev) do
    press = Nx.slice_along_axis(raw, 0, @num_buttons, axis: -1)
    release = Nx.slice_along_axis(raw, @num_buttons, @num_buttons, axis: -1)
    # arithmetic blend, not Nx.select: select takes its shape from the
    # predicate, so a {1, 8} prev would not broadcast over tiled rows
    held = Nx.as_type(Nx.greater(prev, 0.5), Nx.type(raw))
    Nx.subtract(Nx.multiply(Nx.subtract(1, held), press), Nx.multiply(held, release))
  end

  @doc """
  Collapse hold-or-change stick logits into ordinary bucket log-probabilities.

  `raw` is `{batch, K + 1}`: K "change" logits over the buckets and one
  "hold" logit. `prev` is `{batch}` (or `{1}`, broadcast): the bucket held
  on the previous frame. With h = sigmoid(hold):

      p(bucket) = h * [bucket == prev] + (1 - h) * softmax(change)[bucket]

  returned as log-probabilities `{batch, K}`, so the usual cross-entropy and
  the usual categorical sampler apply unchanged. Staying on the previous
  bucket is one explicit decision instead of a coincidence of two
  independent per-frame draws; the previous bucket only selects where the
  hold mass lands and is never an input to the trunk.
  """
  @spec collapse_hold_change(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def collapse_hold_change(raw, prev) do
    k = Nx.axis_size(raw, -1) - 1
    change = Nx.slice_along_axis(raw, 0, k, axis: -1)
    hold = Nx.slice_along_axis(raw, k, 1, axis: -1)
    log_sig = fn x -> Nx.subtract(Nx.min(x, 0), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(x))))) end

    log_h = log_sig.(hold)
    a = Nx.add(log_sig.(Nx.negate(hold)), Nx.subtract(change, Nx.logsumexp(change, axes: [-1], keep_axes: true)))
    both = Nx.add(Nx.max(a, log_h), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(Nx.subtract(a, log_h))))))

    at_prev = Nx.equal(Nx.iota({1, k}), Nx.new_axis(prev, -1)) |> Nx.as_type(Nx.type(raw))
    Nx.add(Nx.multiply(at_prev, both), Nx.multiply(Nx.subtract(1, at_prev), a))
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
