defmodule ExPhil.Training.Imitation.Loss do
  @moduledoc """
  Loss function builders for imitation learning.

  This module provides functions to build loss and gradient computation functions
  that are JIT-compiled once and reused for all training/validation batches.

  ## Supported Policy Types

  - `:autoregressive` - 6-head cross-entropy (buttons, sticks, shoulder)
  - `:diffusion` - MSE noise prediction loss
  - `:act` - CVAE reconstruction + KL divergence loss
  - `:flow_matching` - MSE velocity field loss

  ## Key Functions

  - `build_loss_fn/2` - Build basic loss function for training
  - `build_loss_and_grad_fn/2` - Build compiled loss+gradient function (training)
  - `build_eval_loss_fn/2` - Build compiled loss function (validation, no gradients)

  ## Why Build Functions Once?

  JIT compilation is expensive (seconds to minutes). By building these functions
  once in `Imitation.new/1` and storing them in the trainer struct, we avoid:

  1. Repeated JIT compilation overhead every batch
  2. `deep_backend_copy` calls to handle tensor backend mismatches
  3. Closure creation that captures tensors incorrectly

  ## See Also

  - `ExPhil.Training.Imitation` - Main imitation learning module
  - `ExPhil.Networks.Policy` - Loss computation implementation
  """

  alias ExPhil.Networks.Policy
  alias ExPhil.Training.Imitation.LossConfig
  alias ExPhil.Networks.DiffusionPolicy
  alias ExPhil.Networks.ActionChunking
  alias ExPhil.Networks.FlowMatching
  alias ExPhil.Training.ProbeRegularizer
  alias ExPhil.Training.Utils

  @doc """
  Build the loss function for training.

  Returns a tuple of `{predict_fn, loss_fn}` where:
  - `predict_fn` - Forward pass function
  - `loss_fn` - Function taking (params, states, actions) and returning loss

  ## Options
    - `:label_smoothing` - Label smoothing factor (default: 0.0)
    - `:focal_loss` - Enable focal loss for hard examples (default: false)
    - `:focal_gamma` - Focal loss gamma parameter (default: 2.0)
    - `:button_weight` - Weight for button loss component (default: 1.0)
    - `:button_pos_weight` - Per-button positive class weights [8] tensor (default: nil)
    - `:stick_edge_weight` - Extra weight for stick edge values (default: nil)
  """
  @spec build_loss_fn(Axon.t(), keyword()) :: {function(), function()}
  def build_loss_fn(policy_model, opts \\ []) do
    # ONE typed loss config (INVARIANTS.md item 8). Absent keys fall back
    # to Config.defaults/0, so this builder can no longer drift from the
    # gradient builders' defaults.
    loss_opts = opts |> LossConfig.from_config() |> LossConfig.to_loss_opts()
    {_init_fn, predict_fn} = Utils.build_compiled(policy_model)

    loss_fn = fn params, states, actions ->
      # Forward pass
      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      # NOTE: the f32-loss NaN fix lives in Policy.imitation_loss itself
      # (the loss math entry point), so every builder in this file is
      # covered — including build_autoregressive_loss_and_grad_fn, which is
      # what train_step ACTUALLY uses (it ignores the loss_fn argument).
      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Compute loss with optional label smoothing and focal loss
      Policy.imitation_loss(logits, actions, loss_opts)
    end

    {predict_fn, loss_fn}
  end

  @doc """
  Build a compiled loss+gradient function for efficient training.

  This function is built ONCE in `Imitation.new/1` and reused for all training steps.
  It avoids the need for `deep_backend_copy` every batch by:

  1. Taking all inputs (params, states, actions) as explicit arguments
  2. Using JIT compilation to cache the computation graph
  3. Not capturing any tensors in closures

  ## Parameters

  - `predict_fn` - The compiled forward pass function from `Axon.build/2`
  - `config` - Training configuration map with loss options

  ## Returns

  A JIT-compiled function that takes `(params, states, actions, ...)` and returns `{loss, grads}`.
  For :diffusion and :flow_matching, additional inputs (noise, timestep) are required.

  ## Technical Notes

  The strategy here is to JIT compile a function that takes (params, states, actions) as
  explicit arguments. By using `Nx.Defn.jit` on the outer function, all tensors flow through
  as arguments and get properly traced together.

  The inner `value_and_grad` closure is fine because when the outer function is JIT compiled,
  states/actions become `Defn.Expr` during tracing (not EXLA tensors).
  """
  @spec build_loss_and_grad_fn(function(), map()) :: function()
  def build_loss_and_grad_fn(predict_fn, config) do
    policy_type = config[:policy_type] || :autoregressive

    case policy_type do
      :autoregressive -> build_autoregressive_loss_and_grad_fn(predict_fn, config)
      :diffusion -> build_diffusion_loss_and_grad_fn(predict_fn, config)
      :act -> build_act_loss_and_grad_fn(predict_fn, config)
      :flow_matching -> build_flow_matching_loss_and_grad_fn(predict_fn, config)
    end
  end

  # Autoregressive: cross-entropy loss on 6 controller heads
  defp build_autoregressive_loss_and_grad_fn(predict_fn, config) do
    # ONE typed loss config (INVARIANTS.md item 8) — captured once when
    # building the function, not every batch.
    lc = LossConfig.from_config(config)
    precision = lc.precision

    # Probe-as-regularizer (r15): when enabled, the loss takes the CURRENT
    # probe direction as a 5th ARGUMENT (never a closure capture — the
    # direction refits mid-training and captured tensors break value_and_grad,
    # GOTCHAS #3) and adds weight * mean((trunk(p, states) . v)^2). The trunk
    # predict fn shares param names with the full policy, so gradients flow
    # into the same trunk weights; a zero direction makes the term exactly 0.
    probe_reg_weight = config[:probe_reg_weight] || 0.0
    probe_trunk_fn = config[:probe_trunk_fn]

    # True autoregressive head (AUTOREGRESSIVE_HEAD_PLAN §3): the forward
    # takes the same frame's TARGET components as teacher-forced inputs, so
    # the predict call needs an input map built from states + actions.
    head = lc.head
    temporal = lc.temporal

    if head == :autoregressive and
         ((config[:distill_weight] || 0.0) > 0 or probe_reg_weight > 0) do
      # scheduled_sampling IS supported since 2026-10-02: it only rewrites the
      # input states before the loss (ScheduledSampling.build_autoregressive/2)
      raise ArgumentError,
            "head: :autoregressive is not yet supported together with distill_weight " <>
              "or probe_reg_weight"
    end

    # chunk_weight scales the future heads' loss (popped off again in
    # autoregressive_bc_loss before Policy.imitation_loss sees the opts)
    loss_opts = LossConfig.to_loss_opts(lc) ++ [chunk_weight: config[:chunk_weight] || 1.0]

    # KL-distillation anchor (F3 Route A): when distill_weight > 0 the
    # loss takes teacher logits + a distill mask as 5th/6th ARGUMENTS
    # (per-batch tensors, precomputed at pool build — the teacher never
    # runs in-graph, GOTCHA #3).
    distill_weight = config[:distill_weight] || 0.0
    distill_tau = config[:distill_tau] || 1.0

    if distill_weight > 0 and probe_reg_weight > 0 do
      raise ArgumentError,
            "distill_weight and probe_reg_weight are separate loss variants — enable one at a time"
    end

    # Build the loss+grad function using JIT compilation
    # predict_fn is captured here (once), not in train_step (every batch)
    inner_fn =
      cond do
        distill_weight > 0 ->
          fn params, states, actions, frame_weights, teacher_logits, distill_mask ->
            states = Nx.as_type(states, precision)

            loss_fn = fn p ->
              {buttons, main_x, main_y, c_x, c_y, shoulder} =
                predict_fn.(Utils.ensure_model_state(p), states)

              logits = %{
                buttons: buttons,
                main_x: main_x,
                main_y: main_y,
                c_x: c_x,
                c_y: c_y,
                shoulder: shoulder
              }

              bc =
                Policy.imitation_loss(
                  logits,
                  actions,
                  loss_opts ++ [frame_weights: frame_weights]
                )

              kl =
                ExPhil.Networks.Policy.Loss.distill_kl(logits, teacher_logits, distill_mask,
                  tau: distill_tau
                )

              Nx.add(bc, Nx.multiply(distill_weight, kl))
            end

            Nx.Defn.value_and_grad(loss_fn).(params)
          end

        probe_reg_weight > 0 and probe_trunk_fn != nil ->
          fn params, states, actions, frame_weights, probe_v ->
          states = Nx.as_type(states, precision)

          loss_fn = fn p ->
            bc = autoregressive_bc_loss(predict_fn, p, states, actions, frame_weights, loss_opts, forward_head(config), temporal)
            h = probe_trunk_fn.(Utils.ensure_model_state(p), states)
            penalty = ProbeRegularizer.alignment_penalty(h, probe_v)
            Nx.add(bc, Nx.multiply(probe_reg_weight, penalty))
          end

          Nx.Defn.value_and_grad(loss_fn).(params)
          end

        true ->
          fn params, states, actions, frame_weights ->
            # Convert states to training precision
            states = Nx.as_type(states, precision)

            # Build loss function - states/actions are already Defn.Expr from outer JIT
            loss_fn = fn p ->
              autoregressive_bc_loss(predict_fn, p, states, actions, frame_weights, loss_opts, forward_head(config), temporal)
            end

            # Compute loss and gradients
            Nx.Defn.value_and_grad(loss_fn).(params)
          end
      end

    # JIT compile the entire function - this makes states/actions flow as Defn.Expr
    # during tracing, avoiding the EXLA/Defn.Expr conflict
    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  @doc """
  Build the contiguous-BPTT loss+grad function (BPTT_LOADER_DESIGN.md
  plank D). Works with a model from `Policy.build_temporal_bptt/1`.

  Returned function signature:

      fn params, states, actions, frame_weights, initial_hidden ->
        {{loss, {final_hidden, updated_model_state}}, grads}

  - `states` `{b, t, embed}`, `actions` per-timestep target maps
    (`buttons {b, t, 8}`, categoricals `{b, t}`), `frame_weights` `{b, t}`,
    `initial_hidden` `{b, num_layers, hidden}` (caller zeroes rows whose
    chunk starts a new game — resets live OUTSIDE the graph).
  - Supervision is per-timestep: logits/targets/weights flatten
    `{b, t, *} -> {b*t, *}` and feed the existing `Policy.imitation_loss`
    unchanged.
  - `final_hidden` rides out through `value_and_grad`'s transform arg
    (one forward pass); gradients truncate at the chunk edge by
    construction — `initial_hidden` is a plain argument, never
    differentiated.
  """
  @spec build_bptt_loss_and_grad_fn(function(), map()) :: function()
  def build_bptt_loss_and_grad_fn(predict_fn, config) do
    # ONE typed loss config (INVARIANTS.md item 8)
    lc = LossConfig.from_config(config)
    precision = lc.precision
    head = lc.head
    loss_opts = LossConfig.to_loss_opts(lc)

    # the typed head, tagged with the event-head spec when configured
    event_head = case forward_head(config) do {_, events} -> {head, events}; _ -> head end
    chunk_weight = config[:chunk_weight] || 1.0

    inner_fn = fn params, states, actions, frame_weights, initial_hidden ->
      states = Nx.as_type(states, precision)

      loss_fn = fn p ->
        inputs = bptt_inputs(event_head, states, actions, initial_hidden)

        %{prediction: {prediction, final_hidden}, state: updated_state} =
          predict_fn.(Utils.ensure_model_state(p), inputs)

        loss = bptt_supervision(prediction, actions, frame_weights, loss_opts, chunk_weight)

        {loss, {final_hidden, updated_state}}
      end

      # transform selects the differentiable scalar; final_hidden rides
      # alongside: {{loss, {final_hidden, updated_model_state}}, grads}
      Nx.Defn.value_and_grad(params, loss_fn, &elem(&1, 0))
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  @doc """
  Eval-side twin of `build_bptt_loss_and_grad_fn/2`: the SAME per-timestep
  loss with dropout disabled and no gradients. Returns a jitted

      fn params, states, actions, frame_weights, initial_hidden ->
        {loss, final_hidden}

  so the caller can thread the carry across a sequential val stream
  (`Imitation.Validation.evaluate_bptt/3`) exactly as training does.
  """
  @spec build_bptt_eval_loss_fn(function(), map()) :: function()
  def build_bptt_eval_loss_fn(predict_fn, config) do
    # ONE typed loss config (INVARIANTS.md item 8)
    lc = LossConfig.from_config(config)
    precision = lc.precision
    head = lc.head
    loss_opts = LossConfig.to_loss_opts(lc)

    # the typed head, tagged with the event-head spec when configured
    event_head = case forward_head(config) do {_, events} -> {head, events}; _ -> head end

    inner_fn = fn params, states, actions, frame_weights, initial_hidden ->
      states = Nx.as_type(states, precision)
      inputs = bptt_inputs(event_head, states, actions, initial_hidden)
      {prediction, final_hidden} = predict_fn.(Utils.ensure_model_state(params), inputs)
      # val scores the main head only (chunk weight 0), like the windowed path
      loss = bptt_supervision(prediction, actions, frame_weights, loss_opts, 0.0)
      {loss, final_hidden}
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  # BPTT forward inputs: per-timestep tf_* inputs + the carry; event heads
  # (2026-10-04) get the previous input at EVERY position from the prev-action
  # slot (the sequence already carries it) and the trunk sees the slot zeroed,
  # exactly as policy_forward_inputs/4 does for the last position of a window.
  defp bptt_inputs(:independent, states, _actions, initial_hidden),
    do: %{"state_sequence" => states, "initial_hidden" => initial_hidden}

  defp bptt_inputs(:autoregressive, states, actions, initial_hidden) do
    ExPhil.Networks.Policy.Heads.tf_inputs(actions)
    |> Map.put("state_sequence", states)
    |> Map.put("initial_hidden", initial_hidden)
  end

  defp bptt_inputs({:autoregressive, {:events, ev}}, states, actions, initial_hidden) do
    offset = ev.offset
    idx = Nx.iota({Nx.axis_size(states, 2)})
    keep = Nx.logical_or(Nx.less(idx, offset), Nx.greater_equal(idx, offset + 13))
    masked = Nx.multiply(states, Nx.as_type(keep, Nx.type(states)))

    inputs = bptt_inputs(:autoregressive, masked, actions, initial_hidden)

    inputs =
      if ev.buttons,
        do: Map.put(inputs, "prev_buttons", Nx.slice_along_axis(states, offset, 8, axis: 2)),
        else: inputs

    if ev.sticks do
      buckets =
        states
        |> Nx.slice_along_axis(offset + 8, 4, axis: 2)
        |> Nx.as_type(:f32)
        |> Nx.divide(2.0)
        |> Nx.add(0.5)
        |> Nx.multiply(ev.axis_buckets)
        |> Nx.floor()
        |> Nx.clip(0, ev.axis_buckets - 1)
        |> Nx.as_type(:s64)

      Map.put(inputs, "prev_sticks", buckets)
    else
      inputs
    end
  end

  # Per-timestep supervision: logits/targets/weights flatten {b, t, *} ->
  # {b*t, *} into Policy.imitation_loss. With chunk targets the prediction is
  # {main, futures}; future head j at position i is scored against the target
  # at i + j, which a contiguous chunk already holds: shift the targets left
  # by j and mask positions that fall off the unroll (rows never cross a
  # segment boundary, and padding carries zero frame weight already).
  defp bptt_supervision({main, futures}, actions, frame_weights, loss_opts, chunk_weight)
       when is_tuple(futures) and tuple_size(main) == 6 do
    main_loss = bptt_supervision(main, actions, frame_weights, loss_opts, 0.0)
    k = tuple_size(futures)
    t = Nx.axis_size(frame_weights, 1)

    future_loss =
      for j <- 1..k, reduce: Nx.tensor(0.0) do
        acc ->
          shift = fn tensor ->
            kept = Nx.slice_along_axis(tensor, j, t - j, axis: 1)
            reps = List.duplicate(1, Nx.rank(tensor)) |> List.replace_at(1, j)
            pad = tensor |> Nx.slice_along_axis(t - 1, 1, axis: 1) |> Nx.tile(reps)
            Nx.concatenate([kept, pad], axis: 1)
          end

          targets = Map.new(actions, fn {key, v} -> {key, shift.(v)} end)
          valid = Nx.less(Nx.iota({1, t}), t - j) |> Nx.as_type(Nx.type(frame_weights))
          w = frame_weights |> Nx.multiply(shift.(frame_weights)) |> Nx.multiply(valid)
          Nx.add(acc, bptt_supervision(elem(futures, j - 1), targets, w, loss_opts, 0.0))
      end

    Nx.add(main_loss, Nx.multiply(chunk_weight, Nx.divide(future_loss, k)))
  end

  defp bptt_supervision({buttons, main_x, main_y, c_x, c_y, shoulder}, actions, frame_weights, loss_opts, _chunk_weight) do
    b = Nx.axis_size(buttons, 0)
    t = Nx.axis_size(buttons, 1)

    flat = fn tensor ->
      case Nx.rank(tensor) do
        2 -> Nx.reshape(tensor, {b * t})
        3 -> Nx.reshape(tensor, {b * t, Nx.axis_size(tensor, 2)})
      end
    end

    logits = %{
      buttons: flat.(buttons),
      main_x: flat.(main_x),
      main_y: flat.(main_y),
      c_x: flat.(c_x),
      c_y: flat.(c_y),
      shoulder: flat.(shoulder)
    }

    flat_targets = Map.new(actions, fn {key, v} -> {key, flat.(v)} end)
    Policy.imitation_loss(logits, flat_targets, loss_opts ++ [frame_weights: flat.(frame_weights)])
  end

  # The plain BC objective shared by both autoregressive loss arms
  defp autoregressive_bc_loss(predict_fn, p, states, actions, frame_weights, loss_opts, head, temporal) do
    inputs = policy_forward_inputs(head, temporal, states, actions)
    {chunk_weight, loss_opts} = Keyword.pop(loss_opts, :chunk_weight, 1.0)

    case predict_fn.(Utils.ensure_model_state(p), inputs) do
      # Chunk targets (Heads.build_future_heads/4): main head + the mean of
      # the K future heads' losses, each on the t+j target with the mask
      # (0 past the game's end) folded into the frame weights.
      {main, futures} when is_tuple(futures) and tuple_size(main) == 6 ->
        main_loss = Policy.imitation_loss(head_logits(main), actions, loss_opts ++ [frame_weights: frame_weights])
        k = tuple_size(futures)

        future_loss =
          for j <- 0..(k - 1), reduce: Nx.tensor(0.0) do
            acc ->
              at = fn t -> t |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1]) end

              targets = %{
                buttons: at.(actions.future_buttons),
                main_x: at.(actions.future_main_x),
                main_y: at.(actions.future_main_y),
                c_x: at.(actions.future_c_x),
                c_y: at.(actions.future_c_y),
                shoulder: at.(actions.future_shoulder)
              }

              w = Nx.multiply(frame_weights, at.(actions.future_mask))
              Nx.add(acc, Policy.imitation_loss(head_logits(elem(futures, j)), targets, loss_opts ++ [frame_weights: w]))
          end

        Nx.add(main_loss, Nx.multiply(chunk_weight, Nx.divide(future_loss, k)))

      main ->
        Policy.imitation_loss(head_logits(main), actions, loss_opts ++ [frame_weights: frame_weights])
    end
  end

  defp head_logits({buttons, main_x, main_y, c_x, c_y, shoulder}),
    do: %{buttons: buttons, main_x: main_x, main_y: main_y, c_x: c_x, c_y: c_y, shoulder: shoulder}

  @doc """
  The main six-head logit tuple of a policy's output. Chunk-target models
  (`--chunk-horizon K`) return `{main, futures}` in training; evals that only
  score the next-frame head unwrap it here.
  """
  def main_head({{_, _, _, _, _, _} = main, _futures}), do: main
  def main_head({_, _, _, _, _, _} = main), do: main

  @doc """
  Build the forward-pass input for a policy given the controller head type.

  `:independent` policies take the bare states tensor; `:autoregressive`
  policies take a map of states + teacher-forced target components
  (`Heads.tf_inputs/1`). `temporal` selects the state input name
  (`"state_sequence"` vs `"state"`).
  """
  @spec policy_forward_inputs(atom(), boolean(), Nx.Tensor.t(), map()) ::
          Nx.Tensor.t() | map()
  def policy_forward_inputs(:independent, _temporal, states, _actions), do: states

  def policy_forward_inputs(:autoregressive, temporal, states, actions) do
    state_key = if temporal, do: "state_sequence", else: "state"

    ExPhil.Networks.Policy.Heads.tf_inputs(actions)
    |> Map.put(state_key, states)
  end

  # Event heads (press/release buttons, hold-or-change sticks): `states`
  # {batch, window, embed} carry the prev-action slot at `offset` (8 buttons,
  # main x/y and c x/y in [-1, 1], shoulder). The heads get the LAST
  # position's previous buttons ("prev_buttons") and/or previous stick
  # buckets ("prev_sticks", the same floor(x * buckets) rule as the targets);
  # the trunk gets the whole 13-dim slot zeroed at every position, so it
  # cannot copy the previous input.
  def policy_forward_inputs({:autoregressive, {:events, ev}}, true, states, actions) do
    offset = ev.offset

    last =
      states
      |> Nx.slice_along_axis(Nx.axis_size(states, 1) - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])

    idx = Nx.iota({Nx.axis_size(states, 2)})
    keep = Nx.logical_or(Nx.less(idx, offset), Nx.greater_equal(idx, offset + 13))
    masked = Nx.multiply(states, Nx.as_type(keep, Nx.type(states)))

    inputs = ExPhil.Networks.Policy.Heads.tf_inputs(actions) |> Map.put("state_sequence", masked)

    inputs =
      if ev.buttons,
        do: Map.put(inputs, "prev_buttons", Nx.slice_along_axis(last, offset, 8, axis: 1)),
        else: inputs

    if ev.sticks do
      buckets =
        last
        |> Nx.slice_along_axis(offset + 8, 4, axis: 1)
        |> Nx.as_type(:f32)
        |> Nx.divide(2.0)
        |> Nx.add(0.5)
        |> Nx.multiply(ev.axis_buckets)
        |> Nx.floor()
        |> Nx.clip(0, ev.axis_buckets - 1)
        |> Nx.as_type(:s64)

      Map.put(inputs, "prev_sticks", buckets)
    else
      inputs
    end
  end

  @doc """
  The head tag to hand `policy_forward_inputs/4` for a trainer config:
  `{:autoregressive, {:events, %{...}}}` for the event heads
  (`button_events` / `stick_events`), otherwise the plain head atom.
  """
  def forward_head(config) do
    head = config[:head] || :independent

    if config[:button_events] || config[:stick_events] do
      {head,
       {:events,
        %{
          offset: config[:prev_action_offset],
          buttons: config[:button_events] == true,
          sticks: config[:stick_events] == true,
          axis_buckets: config[:axis_buckets] || 16
        }}}
    else
      head
    end
  end

  # Diffusion: MSE noise prediction loss

  defp build_diffusion_loss_and_grad_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision

    inner_fn = fn params, states, actions, noise, timestep ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)
      noise = Nx.as_type(noise, precision)

      loss_fn = fn p ->
        # Add noise to actions at timestep t
        noisy_actions = DiffusionPolicy.q_sample(actions, noise, timestep, p.schedule)

        # Predict noise
        predicted_noise = predict_fn.(Utils.ensure_model_state(p), %{
          "noisy_actions" => noisy_actions,
          "timestep" => timestep,
          "observations" => states
        })

        DiffusionPolicy.compute_loss(noise, predicted_noise)
      end

      Nx.Defn.value_and_grad(loss_fn).(params)
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  # ACT (Action Chunking Transformer): CVAE loss = reconstruction + KL divergence
  defp build_act_loss_and_grad_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision
    kl_weight = config[:kl_weight] || 10.0

    inner_fn = fn params, states, actions, _frame_weights ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)

      loss_fn = fn p ->
        # ACT model outputs: %{actions: predicted, mu: mu, log_var: log_var}
        outputs = predict_fn.(Utils.ensure_model_state(p), %{
          "observations" => states,
          "target_actions" => actions
        })

        ActionChunking.cvae_loss(
          actions,
          outputs.actions,
          outputs.mu,
          outputs.log_var,
          kl_weight
        )
      end

      Nx.Defn.value_and_grad(loss_fn).(params)
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  # Flow Matching: MSE velocity loss
  # Takes additional inputs: noise [batch, horizon, dim], timestep [batch]
  defp build_flow_matching_loss_and_grad_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision

    inner_fn = fn params, states, actions, noise, timestep ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)
      noise = Nx.as_type(noise, precision)

      loss_fn = fn p ->
        FlowMatching.compute_loss(p, predict_fn, states, actions, noise, timestep)
      end

      Nx.Defn.value_and_grad(loss_fn).(params)
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  @doc """
  Build a compiled loss function for evaluation (no gradients).

  Similar to `build_loss_and_grad_fn/2` but without gradient computation.
  This function is built ONCE in `Imitation.new/1` and reused for all validation batches,
  avoiding JIT recompilation overhead on each epoch.

  ## Parameters

  - `predict_fn` - The compiled forward pass function from `Axon.build/2`
  - `config` - Training configuration map with loss options

  ## Returns

  A JIT-compiled function that takes `(params, states, actions, ...)` and returns the loss tensor.
  For :diffusion and :flow_matching, additional inputs (noise, timestep) are required.
  """
  @spec build_eval_loss_fn(function(), map()) :: function()
  def build_eval_loss_fn(predict_fn, config) do
    policy_type = config[:policy_type] || :autoregressive

    case policy_type do
      :autoregressive -> build_autoregressive_eval_loss_fn(predict_fn, config)
      :diffusion -> build_diffusion_eval_loss_fn(predict_fn, config)
      :act -> build_act_eval_loss_fn(predict_fn, config)
      :flow_matching -> build_flow_matching_eval_loss_fn(predict_fn, config)
    end
  end

  defp build_autoregressive_eval_loss_fn(predict_fn, config) do
    # ONE typed loss config (INVARIANTS.md item 8)
    lc = LossConfig.from_config(config)
    precision = lc.precision
    head = lc.head
    temporal = lc.temporal

    inner_fn = fn params, states, actions ->
      # Convert states to eval precision
      states = Nx.as_type(states, precision)

      # val_loss scores the MAIN head only, so chunk-target runs (output
      # `{head, futures}`) stay comparable with plain ones
      main =
        case predict_fn.(
               Utils.ensure_model_state(params),
               policy_forward_inputs(forward_head(config), temporal, states, actions)
             ) do
          {head, futures} when is_tuple(futures) -> head
          head -> head
        end

      Policy.imitation_loss(head_logits(main), actions, LossConfig.to_loss_opts(lc))
    end

    # JIT compile for fast repeated evaluation
    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  defp build_diffusion_eval_loss_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision

    inner_fn = fn params, states, actions, noise, timestep ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)
      noise = Nx.as_type(noise, precision)

      noisy_actions = DiffusionPolicy.q_sample(actions, noise, timestep, params.schedule)

      predicted_noise = predict_fn.(Utils.ensure_model_state(params), %{
        "noisy_actions" => noisy_actions,
        "timestep" => timestep,
        "observations" => states
      })

      DiffusionPolicy.compute_loss(noise, predicted_noise)
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  defp build_act_eval_loss_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision
    kl_weight = config[:kl_weight] || 10.0

    inner_fn = fn params, states, actions ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)

      outputs = predict_fn.(Utils.ensure_model_state(params), %{
        "observations" => states,
        "target_actions" => actions
      })

      ActionChunking.cvae_loss(
        actions,
        outputs.actions,
        outputs.mu,
        outputs.log_var,
        kl_weight
      )
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  defp build_flow_matching_eval_loss_fn(predict_fn, config) do
    precision = LossConfig.from_config(config).precision

    inner_fn = fn params, states, actions, noise, timestep ->
      states = Nx.as_type(states, precision)
      actions = Nx.as_type(actions, precision)
      noise = Nx.as_type(noise, precision)

      FlowMatching.compute_loss(params, predict_fn, states, actions, noise, timestep)
    end

    Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)
  end

  # ============================================================================
  # Helper: Get policy type from config
  # ============================================================================

  @doc """
  Check if a policy type requires noise and timestep inputs for training.
  """
  @spec requires_noise_inputs?(atom()) :: boolean()
  def requires_noise_inputs?(policy_type) do
    policy_type in [:diffusion, :flow_matching]
  end

  @doc """
  Get the number of inputs required for the loss function.
  """
  @spec loss_fn_arity(atom()) :: pos_integer()
  def loss_fn_arity(policy_type) do
    case policy_type do
      :autoregressive -> 4  # params, states, actions, frame_weights
      :act -> 4             # params, states, actions, frame_weights
      :diffusion -> 5       # params, states, actions, noise, timestep
      :flow_matching -> 5   # params, states, actions, noise, timestep
    end
  end
end
