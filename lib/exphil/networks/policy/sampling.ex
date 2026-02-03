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
      predict_fn.(Utils.ensure_model_state(params), state)

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
