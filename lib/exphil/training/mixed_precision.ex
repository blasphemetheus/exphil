defmodule ExPhil.Training.MixedPrecision do
  @moduledoc """
  Mixed precision training utilities for BF16/FP16 training.

  ## Overview

  Mixed precision training uses lower precision (BF16) for forward/backward passes
  while maintaining FP32 "master weights" for accurate gradient accumulation.

  This provides:
  - ~2x faster training on RTX 3000/4000 series GPUs (tensor cores)
  - ~50% memory reduction for activations
  - Maintained training stability (FP32 master weights)

  ## Usage

      # Create mixed precision state
      mp_state = MixedPrecision.init(params, precision: :bf16)

      # In training loop:
      # 1. Get BF16 params for forward pass
      bf16_params = MixedPrecision.get_compute_params(mp_state)

      # 2. Compute loss and gradients in BF16
      {loss, grads} = compute_loss_and_grad(bf16_params, batch)

      # 3. Update FP32 master weights
      mp_state = MixedPrecision.update(mp_state, grads, optimizer, optimizer_state)

  ## BF16 vs FP16

  - **BF16**: Same exponent range as FP32, no loss scaling needed. Recommended.
  - **FP16**: Smaller exponent range, requires loss scaling to prevent underflow.

  ## Numerical Stability

  Some operations should stay in FP32 for stability:
  - Softmax (intermediate exp values can overflow/underflow)
  - LayerNorm (variance computation needs precision)
  - Loss reduction (summing many small values)

  Use `stable_softmax/2` and `stable_layer_norm/3` for these.
  """

  @type precision :: :f32 | :bf16 | :f16
  @type t :: %__MODULE__{
          master_params: map(),
          precision: precision(),
          loss_scale: float()
        }

  defstruct [:master_params, :precision, :loss_scale]

  @doc """
  Initialize mixed precision state.

  ## Options

  - `:precision` - Compute precision (:bf16 or :f16). Default: :bf16
  - `:loss_scale` - Static loss scale for FP16 (not needed for BF16). Default: 1.0

  ## Examples

      iex> params = %{kernel: Nx.iota({256, 128}, type: :f32)}
      iex> mp_state = MixedPrecision.init(params, precision: :bf16)
      iex> mp_state.precision
      :bf16
  """
  @spec init(map(), keyword()) :: t()
  def init(params, opts \\ []) do
    precision = Keyword.get(opts, :precision, :bf16)
    loss_scale = Keyword.get(opts, :loss_scale, 1.0)

    # Ensure master params are FP32
    master_params = ensure_f32(params)

    %__MODULE__{
      master_params: master_params,
      precision: precision,
      loss_scale: loss_scale
    }
  end

  @doc """
  Get parameters cast to compute precision (BF16/FP16) for forward pass.

  This is a pure function - does not modify state.
  """
  @spec get_compute_params(t()) :: map()
  def get_compute_params(%__MODULE__{master_params: params, precision: precision}) do
    cast_params(params, precision)
  end

  @doc """
  Cast gradients to FP32 for accumulation.

  Gradients from BF16 forward/backward pass should be cast to FP32
  before applying to master weights to preserve small gradient updates.

  ## Arguments

  - `mp_state` - Mixed precision state (for loss_scale)
  - `grads` - Gradients (in compute precision, typically BF16)

  ## Returns

  Gradients cast to FP32 and unscaled if using loss scaling.
  """
  @spec cast_grads_to_f32(t(), map()) :: map()
  def cast_grads_to_f32(%__MODULE__{loss_scale: loss_scale}, grads) do
    grads |> ensure_f32() |> unscale_grads(loss_scale)
  end

  @doc """
  Get master weights (FP32) for checkpointing.
  """
  @spec get_master_params(t()) :: map()
  def get_master_params(%__MODULE__{master_params: params}), do: params

  @doc """
  Load master weights from checkpoint.
  """
  @spec set_master_params(t(), map()) :: t()
  def set_master_params(mp_state, params) do
    %{mp_state | master_params: ensure_f32(params)}
  end

  # ==========================================================================
  # Numerical Stability Functions
  # ==========================================================================

  @doc """
  Stable softmax that computes in FP32 regardless of input precision.

  This prevents overflow/underflow issues when computing exp() in BF16.
  Uses max-subtraction trick to prevent overflow.

  ## Options

  - `:axis` - Axis to compute softmax over. Default: -1

  ## Examples

      iex> logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :bf16)
      iex> probs = MixedPrecision.stable_softmax(logits)
      iex> Nx.type(probs)
      {:bf, 16}  # Output is in original precision
  """
  def stable_softmax(logits, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    input_type = Nx.type(logits)

    # Cast to F32 for stable computation
    logits_f32 = Nx.as_type(logits, :f32)

    # Subtract max for numerical stability
    max_logits = Nx.reduce_max(logits_f32, axes: [axis], keep_axes: true)
    shifted = Nx.subtract(logits_f32, max_logits)

    # Compute softmax in F32
    exp_shifted = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_shifted, axes: [axis], keep_axes: true)
    probs_f32 = Nx.divide(exp_shifted, sum_exp)

    # Cast back to original precision
    Nx.as_type(probs_f32, input_type)
  end

  @doc """
  Stable layer normalization that computes in FP32.

  LayerNorm involves computing variance which can lose precision in BF16.

  ## Arguments

  - `x` - Input tensor
  - `gamma` - Scale parameter
  - `beta` - Shift parameter

  ## Options

  - `:axis` - Axis to normalize. Default: -1
  - `:epsilon` - Small constant for numerical stability. Default: 1.0e-5
  """
  def stable_layer_norm(x, gamma, beta, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    epsilon = Keyword.get(opts, :epsilon, 1.0e-5)
    input_type = Nx.type(x)

    # Cast to F32 for stable computation
    x_f32 = Nx.as_type(x, :f32)

    # Compute mean and variance in F32
    mean = Nx.mean(x_f32, axes: [axis], keep_axes: true)
    variance = Nx.variance(x_f32, axes: [axis], keep_axes: true)

    # Normalize
    normalized = Nx.divide(Nx.subtract(x_f32, mean), Nx.sqrt(Nx.add(variance, epsilon)))

    # Apply scale and shift
    result_f32 = Nx.add(Nx.multiply(normalized, gamma), beta)

    # Cast back to original precision
    Nx.as_type(result_f32, input_type)
  end

  @doc """
  Stable cross-entropy loss that computes in FP32.

  Log-softmax and reduction should be done in FP32 to avoid precision loss.

  ## Arguments

  - `logits` - Model output logits [batch, classes]
  - `targets` - One-hot targets [batch, classes]

  ## Options

  - `:axis` - Class axis. Default: -1
  - `:reduction` - :mean, :sum, or :none. Default: :mean
  """
  def stable_cross_entropy(logits, targets, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    reduction = Keyword.get(opts, :reduction, :mean)

    # Cast to F32 for stable computation
    logits_f32 = Nx.as_type(logits, :f32)
    targets_f32 = Nx.as_type(targets, :f32)

    # Compute log-softmax in F32 (stable)
    max_logits = Nx.reduce_max(logits_f32, axes: [axis], keep_axes: true)
    shifted = Nx.subtract(logits_f32, max_logits)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [axis], keep_axes: true))
    log_probs = Nx.subtract(shifted, log_sum_exp)

    # Compute loss (one-hot encoded targets)
    loss_per_sample = Nx.negate(Nx.sum(Nx.multiply(targets_f32, log_probs), axes: [axis]))

    # Reduce
    case reduction do
      :mean -> Nx.mean(loss_per_sample)
      :sum -> Nx.sum(loss_per_sample)
      :none -> loss_per_sample
    end
  end

  # ==========================================================================
  # Internal Helpers
  # ==========================================================================

  # Recursively cast all tensors in a nested map to FP32
  defp ensure_f32(tensor) when is_struct(tensor, Nx.Tensor) do
    Nx.as_type(tensor, :f32)
  end

  defp ensure_f32(%Axon.ModelState{data: data} = state) do
    %{state | data: ensure_f32(data)}
  end

  defp ensure_f32(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, ensure_f32(v)} end)
  end

  defp ensure_f32(other), do: other

  # Recursively cast all tensors to compute precision
  defp cast_params(tensor, precision) when is_struct(tensor, Nx.Tensor) do
    Nx.as_type(tensor, precision)
  end

  defp cast_params(%Axon.ModelState{data: data} = state, precision) do
    %{state | data: cast_params(data, precision)}
  end

  defp cast_params(map, precision) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, cast_params(v, precision)} end)
  end

  defp cast_params(other, _precision), do: other

  # Unscale gradients (divide by loss scale)
  defp unscale_grads(grads, 1.0), do: grads

  defp unscale_grads(grads, loss_scale) do
    scale_grads(grads, 1.0 / loss_scale)
  end

  # Scale gradients (multiply by factor)
  defp scale_grads(tensor, factor) when is_struct(tensor, Nx.Tensor) do
    Nx.multiply(tensor, factor)
  end

  defp scale_grads(%Axon.ModelState{data: data} = state, factor) do
    %{state | data: scale_grads(data, factor)}
  end

  defp scale_grads(map, factor) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, scale_grads(v, factor)} end)
  end

  defp scale_grads(other, _factor), do: other
end
