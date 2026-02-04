defmodule ExPhil.Training.Imitation.Loss do
  @moduledoc """
  Loss function builders for imitation learning.

  This module provides functions to build loss and gradient computation functions
  that are JIT-compiled once and reused for all training/validation batches.

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
    - `:stick_edge_weight` - Extra weight for stick edge values (default: nil)
  """
  @spec build_loss_fn(Axon.t(), keyword()) :: {function(), function()}
  def build_loss_fn(policy_model, opts \\ []) do
    label_smoothing = Keyword.get(opts, :label_smoothing, 0.0)
    focal_loss = Keyword.get(opts, :focal_loss, false)
    focal_gamma = Keyword.get(opts, :focal_gamma, 2.0)
    button_weight = Keyword.get(opts, :button_weight, 1.0)
    stick_edge_weight = Keyword.get(opts, :stick_edge_weight)
    {_init_fn, predict_fn} = Axon.build(policy_model)

    loss_fn = fn params, states, actions ->
      # Forward pass
      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      # Compute loss with optional label smoothing and focal loss
      Policy.imitation_loss(logits, actions,
        label_smoothing: label_smoothing,
        focal_loss: focal_loss,
        focal_gamma: focal_gamma,
        button_weight: button_weight,
        stick_edge_weight: stick_edge_weight
      )
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

  A JIT-compiled function that takes `(params, states, actions)` and returns `{loss, grads}`.

  ## Technical Notes

  The strategy here is to JIT compile a function that takes (params, states, actions) as
  explicit arguments. By using `Nx.Defn.jit` on the outer function, all tensors flow through
  as arguments and get properly traced together.

  The inner `value_and_grad` closure is fine because when the outer function is JIT compiled,
  states/actions become `Defn.Expr` during tracing (not EXLA tensors).
  """
  @spec build_loss_and_grad_fn(function(), map()) :: function()
  def build_loss_and_grad_fn(predict_fn, config) do
    # Extract config options that affect loss computation
    # These are captured once when building the function, not every batch
    label_smoothing = config[:label_smoothing] || 0.0
    focal_loss = config[:focal_loss] || false
    focal_gamma = config[:focal_gamma] || 2.0
    button_weight = config[:button_weight] || 1.0
    stick_edge_weight = config[:stick_edge_weight]
    precision = config[:precision] || :bf16

    # Build the loss+grad function using JIT compilation
    # predict_fn is captured here (once), not in train_step (every batch)
    inner_fn = fn params, states, actions ->
      # Convert states to training precision
      states = Nx.as_type(states, precision)

      # Build loss function - states/actions are already Defn.Expr from outer JIT
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

        Policy.imitation_loss(logits, actions,
          label_smoothing: label_smoothing,
          focal_loss: focal_loss,
          focal_gamma: focal_gamma,
          button_weight: button_weight,
          stick_edge_weight: stick_edge_weight
        )
      end

      # Compute loss and gradients
      Nx.Defn.value_and_grad(loss_fn).(params)
    end

    # JIT compile the entire function - this makes states/actions flow as Defn.Expr
    # during tracing, avoiding the EXLA/Defn.Expr conflict
    Nx.Defn.jit(inner_fn, compiler: EXLA)
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

  A JIT-compiled function that takes `(params, states, actions)` and returns the loss tensor.
  """
  @spec build_eval_loss_fn(function(), map()) :: function()
  def build_eval_loss_fn(predict_fn, config) do
    label_smoothing = config[:label_smoothing] || 0.0
    focal_loss = config[:focal_loss] || false
    focal_gamma = config[:focal_gamma] || 2.0
    button_weight = config[:button_weight] || 1.0
    stick_edge_weight = config[:stick_edge_weight]
    precision = config[:precision] || :bf16

    inner_fn = fn params, states, actions ->
      # Convert states to eval precision
      states = Nx.as_type(states, precision)

      {buttons, main_x, main_y, c_x, c_y, shoulder} =
        predict_fn.(Utils.ensure_model_state(params), states)

      logits = %{
        buttons: buttons,
        main_x: main_x,
        main_y: main_y,
        c_x: c_x,
        c_y: c_y,
        shoulder: shoulder
      }

      Policy.imitation_loss(logits, actions,
        label_smoothing: label_smoothing,
        focal_loss: focal_loss,
        focal_gamma: focal_gamma,
        button_weight: button_weight,
        stick_edge_weight: stick_edge_weight
      )
    end

    # JIT compile for fast repeated evaluation
    Nx.Defn.jit(inner_fn, compiler: EXLA)
  end
end
