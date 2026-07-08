defmodule ExPhil.Networks.Policy.Sampling do
  @moduledoc """
  Action sampling functions for policy networks.

  Provides sampling methods for autoregressive action generation:
  - Button sampling (independent Bernoulli)
  - Stick/shoulder sampling (categorical with temperature)
  - Confidence scoring for action predictions

  ## Usage

      # Sample from policy output
      actions = Sampling.sample(params, predict_fn, state,
        temperature: 1.0,
        deterministic: false
      )

      # Get confidence scores
      confidence = Sampling.compute_confidence(actions)

  ## Sampling Methods

  - **Buttons**: Independent Bernoulli sampling from sigmoid probabilities
  - **Sticks/Shoulder**: Categorical sampling using Gumbel-max trick
  - **Temperature**: Controls exploration (higher = more random)
  - **Deterministic**: Uses argmax instead of sampling

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Networks.Policy.Loss` - Loss computation
  """

  alias ExPhil.Training.Utils

  @doc """
  Sample actions from the policy.

  This performs autoregressive sampling, where each controller component
  is sampled conditioned on previously sampled components.

  ## Options
    - `:temperature` - Softmax temperature for exploration (default: 1.0)
    - `:deterministic` - If true, use argmax instead of sampling (default: false)
    - `:axis_buckets` - Number of stick buckets (default: 16)
    - `:shoulder_buckets` - Number of shoulder buckets (default: 4)
  """
  @spec sample(map(), function(), Nx.Tensor.t(), keyword()) :: map()
  def sample(params, predict_fn, state, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)

    # Forward pass to get all logits
    {buttons_logits, main_x_logits, main_y_logits, c_x_logits, c_y_logits, shoulder_logits} =
      logits_tuple = predict_fn.(Utils.ensure_model_state(params), state)

    # Sample ALL heads (+ confidence) in ONE compiled program. Doing this with
    # per-head eager Nx ops costs >100ms per decision (dozens of separate XLA
    # dispatches); fused it is ~1ms. See scripts/profile_agent_inference.exs.
    deterministic_buttons = Keyword.get(opts, :deterministic_buttons, false)

    {buttons, main_x, main_y, c_x, c_y, shoulder, conf} =
      if deterministic do
        jitted(:fused_det, &fused_sample_deterministic/1).(logits_tuple)
      else
        key = Nx.Random.key(:erlang.unique_integer([:positive]))

        jitted(:fused_stoch, &fused_sample_stochastic/3).(
          logits_tuple,
          key,
          Nx.tensor(temperature, type: :f32)
        )
      end

    # Mixed decode: argmax the buttons while sticks keep sampling — kills
    # stray rare-button rolls (taunts) without argmax's modal lock on
    # movement. One tiny eager op on a [1,8] tensor.
    buttons =
      if deterministic_buttons and not deterministic do
        Nx.greater(Nx.sigmoid(buttons_logits), 0.5)
      else
        buttons
      end

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder,
      # Precomputed confidence scalars (tensors) — see compute_confidence/1
      confidence_raw: conf,
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

  import Nx.Defn

  # Nx's default defn options are EMPTY, so a bare defn call runs on the
  # pure-Elixir evaluator (~120ms here). Jit explicitly with EXLA and cache
  # the compiled closure. (Same gotcha as scripts/test_fused_kernels.exs.)
  defp jitted(name, fun) do
    pt_key = {__MODULE__, name}

    case :persistent_term.get(pt_key, nil) do
      nil ->
        compiled =
          if Code.ensure_loaded?(EXLA) do
            Nx.Defn.jit(fun, compiler: EXLA)
          else
            Nx.Defn.jit(fun)
          end

        :persistent_term.put(pt_key, compiled)
        compiled

      compiled ->
        compiled
    end
  end

  # --- Fused sampling: one XLA program for all six heads + confidence ---

  defnp fused_sample_stochastic({b_l, mx_l, my_l, cx_l, cy_l, sh_l}, key, temperature) do
    probs = Nx.sigmoid(b_l)
    {u, key} = Nx.Random.uniform(key, shape: Nx.shape(probs))
    buttons = Nx.less(u, probs)

    {mx, key} = gumbel_argmax(mx_l, key, temperature)
    {my, key} = gumbel_argmax(my_l, key, temperature)
    {cx, key} = gumbel_argmax(cx_l, key, temperature)
    {cy, key} = gumbel_argmax(cy_l, key, temperature)
    {sh, _key} = gumbel_argmax(sh_l, key, temperature)

    {buttons, mx, my, cx, cy, sh, confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  defnp fused_sample_deterministic({b_l, mx_l, my_l, cx_l, cy_l, sh_l}) do
    buttons = Nx.greater(Nx.sigmoid(b_l), 0.5)
    mx = Nx.argmax(mx_l, axis: -1)
    my = Nx.argmax(my_l, axis: -1)
    cx = Nx.argmax(cx_l, axis: -1)
    cy = Nx.argmax(cy_l, axis: -1)
    sh = Nx.argmax(sh_l, axis: -1)

    {buttons, mx, my, cx, cy, sh, confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  defnp gumbel_argmax(logits, key, temperature) do
    scaled = logits / temperature
    {u, key} = Nx.Random.uniform(key, shape: Nx.shape(scaled))
    gumbel = -Nx.log(-Nx.log(u + 1.0e-10))
    {Nx.argmax(scaled + gumbel, axis: -1), key}
  end

  defnp confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l) do
    buttons = Nx.mean(Nx.abs(Nx.sigmoid(b_l) - 0.5) * 2)
    main = (max_softmax(mx_l) + max_softmax(my_l)) / 2
    c = (max_softmax(cx_l) + max_softmax(cy_l)) / 2
    shoulder = max_softmax(sh_l)
    overall = buttons * 0.4 + main * 0.3 + c * 0.15 + shoulder * 0.15
    %{buttons: buttons, main: main, c: c, shoulder: shoulder, overall: overall}
  end

  defnp max_softmax(logits) do
    Nx.exp(logits - Nx.logsumexp(logits, axes: [-1], keep_axes: true))
    |> Nx.reduce_max(axes: [-1])
    |> Nx.mean()
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

  @doc """
  Compute confidence scores from action logits.

  Returns a map with confidence scores (0-1 scale) for each component:
  - `:buttons` - Average button confidence (how far from 0.5 the sigmoid probs are)
  - `:main` - Main stick confidence (max softmax probability)
  - `:c` - C-stick confidence (max softmax probability)
  - `:shoulder` - Shoulder confidence (max softmax probability)
  - `:overall` - Weighted average of all components

  Higher values = more confident predictions.
  """
  @spec compute_confidence(map()) :: map()
  def compute_confidence(%{confidence_raw: raw}) do
    # Precomputed inside the fused sampling program — just read the scalars.
    %{
      buttons: raw.buttons |> Nx.to_number() |> Float.round(3),
      main: raw.main |> Nx.to_number() |> Float.round(3),
      c: raw.c |> Nx.to_number() |> Float.round(3),
      shoulder: raw.shoulder |> Nx.to_number() |> Float.round(3),
      overall: raw.overall |> Nx.to_number() |> Float.round(3)
    }
  end

  def compute_confidence(%{logits: logits}) do
    compute_confidence(logits)
  end

  def compute_confidence(%{
        buttons: buttons_logits,
        main_x: main_x_logits,
        main_y: main_y_logits,
        c_x: c_x_logits,
        c_y: c_y_logits,
        shoulder: shoulder_logits
      }) do
    # Button confidence: how far from 0.5 (uncertain) the probabilities are
    # Confidence = mean(|sigmoid(logit) - 0.5| * 2)
    button_probs = Nx.sigmoid(buttons_logits)

    button_confidence =
      button_probs
      |> Nx.subtract(0.5)
      |> Nx.abs()
      |> Nx.multiply(2)
      |> Nx.mean()
      |> Nx.to_number()

    # Categorical confidence: max softmax probability
    main_x_conf = max_softmax_prob(main_x_logits)
    main_y_conf = max_softmax_prob(main_y_logits)
    main_confidence = (main_x_conf + main_y_conf) / 2

    c_x_conf = max_softmax_prob(c_x_logits)
    c_y_conf = max_softmax_prob(c_y_logits)
    c_confidence = (c_x_conf + c_y_conf) / 2

    shoulder_confidence = max_softmax_prob(shoulder_logits)

    # Overall: weighted average (buttons are most important for gameplay)
    overall =
      button_confidence * 0.4 + main_confidence * 0.3 +
        c_confidence * 0.15 + shoulder_confidence * 0.15

    %{
      buttons: Float.round(button_confidence, 3),
      main: Float.round(main_confidence, 3),
      c: Float.round(c_confidence, 3),
      shoulder: Float.round(shoulder_confidence, 3),
      overall: Float.round(overall, 3)
    }
  end

  def compute_confidence(_), do: %{overall: 0.0, buttons: 0.0, main: 0.0, c: 0.0, shoulder: 0.0}

  # Helper: compute max probability from softmax of logits
  defp max_softmax_prob(logits) do
    # Softmax then max
    probs = Axon.Activations.softmax(logits, axis: -1)

    probs
    |> Nx.reduce_max(axes: [-1])
    # Average across batch if present
    |> Nx.mean()
    |> Nx.to_number()
  end
end
