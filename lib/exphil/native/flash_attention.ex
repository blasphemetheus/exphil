defmodule ExPhil.Native.FlashAttention do
  @moduledoc """
  Native FlashAttention-2 for high-performance inference and training.

  This NIF provides GPU-accelerated flash attention using CUDA kernels.

  ## API

  - `forward/4` - Fast forward pass for inference (no state saved)
  - `forward_with_states/4` - Forward pass that saves logsumexp for backward
  - `backward/7` - Backward pass computing dQ, dK, dV gradients

  ## Requirements

  For GPU acceleration:
  - CUDA 12.0+ toolkit
  - Ampere+ GPU (RTX 30xx, 40xx, A100, H100)
  - Compile with `FLASH_ATTENTION_CUDA=1`

  Without CUDA, falls back to a CPU implementation that is mathematically
  equivalent but slower.

  ## Usage

      # Inference (forward only)
      {:ok, output} = ExPhil.Native.FlashAttention.forward(q, k, v, causal: true)

      # Training (forward + backward)
      {:ok, output, logsumexp} = ExPhil.Native.FlashAttention.forward_with_states(q, k, v)
      # ... compute loss and d_out ...
      {:ok, dq, dk, dv} = ExPhil.Native.FlashAttention.backward(d_out, q, k, v, output, logsumexp)

  ## Tensor Layout

  All tensors use layout: `[batch, seq_len, num_heads, head_dim]`

  This differs from some implementations that use `[batch, num_heads, seq_len, head_dim]`.
  Ensure your tensors are in the correct layout before calling.

  ## Fallback

  When CUDA is unavailable or the NIF fails to load, use the Pure Nx
  `ExPhil.Networks.Attention.memory_efficient_attention/4` as a fallback.
  """

  # Enable CUDA feature when FLASH_ATTENTION_CUDA=1 environment variable is set
  # This requires: CUDA toolkit, nvcc, and Ampere+ GPU (RTX 30xx/40xx, A100, H100)
  @cuda_enabled System.get_env("FLASH_ATTENTION_CUDA") == "1"

  use Rustler,
    otp_app: :exphil,
    crate: "flash_attention_nif",
    path: "native/flash_attention_nif",
    features: if(@cuda_enabled, do: ["cuda"], else: [])

  @doc """
  Check if CUDA flash attention is available.

  Returns `true` if:
  - NIF compiled with CUDA support
  - CUDA runtime is available
  - GPU is Ampere or newer (compute capability >= 8.0)
  """
  @spec cuda_available?() :: boolean()
  def cuda_available?, do: cuda_available()

  @doc false
  def cuda_available, do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get information about the active backend.

  Returns one of:
  - `"cuda"` - CUDA acceleration active
  - `"cpu"` - CPU fallback (CUDA feature not compiled or GPU unavailable)
  - `"cpu (cuda feature enabled but unavailable)"` - Compiled with CUDA but GPU not found
  """
  @spec backend_info() :: String.t()
  def backend_info, do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Run FlashAttention forward pass.

  ## Arguments

  * `query` - Query tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `key` - Key tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `value` - Value tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `opts` - Options:
    * `:causal` - Apply causal masking (default: true)

  ## Returns

  `{:ok, output}` where output is `[batch, seq_len, num_heads, head_dim]`, f32

  ## Example

      # Prepare tensors (must be f32)
      query = Nx.as_type(query, :f32)
      key = Nx.as_type(key, :f32)
      value = Nx.as_type(value, :f32)

      # Run attention
      {:ok, output} = ExPhil.Native.FlashAttention.forward(query, key, value)
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def forward(query, key, value, opts \\ []) do
    causal = Keyword.get(opts, :causal, true)

    # Validate shapes match
    q_shape = Nx.shape(query)
    k_shape = Nx.shape(key)
    v_shape = Nx.shape(value)

    unless q_shape == k_shape and k_shape == v_shape do
      raise ArgumentError,
            "Shape mismatch: query=#{inspect(q_shape)}, key=#{inspect(k_shape)}, value=#{inspect(v_shape)}"
    end

    # Extract dimensions
    {batch, seq_len, num_heads, head_dim} =
      case q_shape do
        {b, s, h, d} -> {b, s, h, d}
        other -> raise ArgumentError, "Expected 4D tensor [batch, seq, heads, dim], got #{inspect(other)}"
      end

    # Convert to f32 binary on CPU
    q_bin = tensor_to_binary(query)
    k_bin = tensor_to_binary(key)
    v_bin = tensor_to_binary(value)

    # Call NIF
    case forward_f32(q_bin, k_bin, v_bin, batch, seq_len, num_heads, head_dim, causal) do
      {:ok, output_bin} ->
        output = Nx.from_binary(output_bin, :f32) |> Nx.reshape(q_shape)
        {:ok, output}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Run FlashAttention forward pass, raising on error.
  """
  @spec forward!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward!(query, key, value, opts \\ []) do
    case forward(query, key, value, opts) do
      {:ok, output} -> output
      {:error, reason} -> raise "FlashAttention failed: #{inspect(reason)}"
    end
  end

  @doc false
  def forward_f32(_q, _k, _v, _batch, _seq_len, _num_heads, _head_dim, _causal) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # ===========================================================================
  # Training API: Forward with State Saving + Backward Pass
  # ===========================================================================

  @doc """
  Run FlashAttention forward pass, saving state for backward pass.

  This saves the logsumexp (log of softmax denominator) which is needed
  to compute gradients in the backward pass without storing the full O(n²)
  attention matrix.

  ## Arguments

  * `query` - Query tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `key` - Key tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `value` - Value tensor `[batch, seq_len, num_heads, head_dim]`, f32
  * `opts` - Options:
    * `:causal` - Apply causal masking (default: true)

  ## Returns

  `{:ok, output, logsumexp}` where:
  - `output` is `[batch, seq_len, num_heads, head_dim]`, f32
  - `logsumexp` is `[batch, num_heads, seq_len]`, f32

  ## Example

      {:ok, output, logsumexp} = ExPhil.Native.FlashAttention.forward_with_states(q, k, v)
      # Save q, k, v, output, logsumexp for backward pass
  """
  @spec forward_with_states(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t(), Nx.Tensor.t()} | {:error, term()}
  def forward_with_states(query, key, value, opts \\ []) do
    causal = Keyword.get(opts, :causal, true)

    # Validate shapes match
    q_shape = Nx.shape(query)
    k_shape = Nx.shape(key)
    v_shape = Nx.shape(value)

    unless q_shape == k_shape and k_shape == v_shape do
      raise ArgumentError,
            "Shape mismatch: query=#{inspect(q_shape)}, key=#{inspect(k_shape)}, value=#{inspect(v_shape)}"
    end

    # Extract dimensions
    {batch, seq_len, num_heads, head_dim} =
      case q_shape do
        {b, s, h, d} -> {b, s, h, d}
        other -> raise ArgumentError, "Expected 4D tensor [batch, seq, heads, dim], got #{inspect(other)}"
      end

    # Convert to f32 binary on CPU
    q_bin = tensor_to_binary(query)
    k_bin = tensor_to_binary(key)
    v_bin = tensor_to_binary(value)

    # Call NIF (function name must match Rust: forward_with_logsumexp)
    case forward_with_logsumexp(q_bin, k_bin, v_bin, batch, seq_len, num_heads, head_dim, causal) do
      {:ok, output_bin, lse_bin} ->
        output = Nx.from_binary(output_bin, :f32) |> Nx.reshape(q_shape)
        lse = Nx.from_binary(lse_bin, :f32) |> Nx.reshape({batch, num_heads, seq_len})
        {:ok, output, lse}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Run FlashAttention forward pass with state saving, raising on error.
  """
  @spec forward_with_states!(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def forward_with_states!(query, key, value, opts \\ []) do
    case forward_with_states(query, key, value, opts) do
      {:ok, output, lse} -> {output, lse}
      {:error, reason} -> raise "FlashAttention forward_with_states failed: #{inspect(reason)}"
    end
  end

  # NIF stub - must match Rust function name exactly
  @doc false
  def forward_with_logsumexp(_q, _k, _v, _batch, _seq_len, _num_heads, _head_dim, _causal) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Run FlashAttention backward pass, computing gradients dQ, dK, dV.

  This recomputes attention scores in tiles using the saved logsumexp,
  avoiding O(n²) memory while computing exact gradients.

  ## Arguments

  * `d_out` - Gradient from downstream `[batch, seq_len, num_heads, head_dim]`, f32
  * `query` - Original query (saved from forward)
  * `key` - Original key (saved from forward)
  * `value` - Original value (saved from forward)
  * `output` - Forward output (saved from forward_with_states)
  * `logsumexp` - Logsumexp (saved from forward_with_states) `[batch, num_heads, seq_len]`
  * `opts` - Options:
    * `:causal` - Apply causal masking (default: true)

  ## Returns

  `{:ok, dq, dk, dv}` where each gradient matches the corresponding input shape.

  ## Example

      # Forward pass (saving state)
      {:ok, output, logsumexp} = forward_with_states(q, k, v)

      # ... compute loss and d_out from downstream ...

      # Backward pass
      {:ok, dq, dk, dv} = backward(d_out, q, k, v, output, logsumexp)
  """
  @spec backward(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: {:ok, Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()} | {:error, term()}
  def backward(d_out, query, key, value, output, logsumexp, opts \\ []) do
    causal = Keyword.get(opts, :causal, true)

    # Extract dimensions from query
    q_shape = Nx.shape(query)

    {batch, seq_len, num_heads, head_dim} =
      case q_shape do
        {b, s, h, d} -> {b, s, h, d}
        other -> raise ArgumentError, "Expected 4D tensor [batch, seq, heads, dim], got #{inspect(other)}"
      end

    # Validate shapes
    for {name, tensor, expected} <- [
          {"d_out", d_out, q_shape},
          {"key", key, q_shape},
          {"value", value, q_shape},
          {"output", output, q_shape},
          {"logsumexp", logsumexp, {batch, num_heads, seq_len}}
        ] do
      actual = Nx.shape(tensor)

      unless actual == expected do
        raise ArgumentError, "#{name} shape mismatch: expected #{inspect(expected)}, got #{inspect(actual)}"
      end
    end

    # Convert to f32 binary
    d_out_bin = tensor_to_binary(d_out)
    q_bin = tensor_to_binary(query)
    k_bin = tensor_to_binary(key)
    v_bin = tensor_to_binary(value)
    out_bin = tensor_to_binary(output)
    lse_bin = tensor_to_binary(logsumexp)

    # Call NIF (function name must match Rust: backward_f32)
    case backward_f32(
           d_out_bin,
           q_bin,
           k_bin,
           v_bin,
           out_bin,
           lse_bin,
           batch,
           seq_len,
           num_heads,
           head_dim,
           causal
         ) do
      {:ok, dq_bin, dk_bin, dv_bin} ->
        dq = Nx.from_binary(dq_bin, :f32) |> Nx.reshape(q_shape)
        dk = Nx.from_binary(dk_bin, :f32) |> Nx.reshape(q_shape)
        dv = Nx.from_binary(dv_bin, :f32) |> Nx.reshape(q_shape)
        {:ok, dq, dk, dv}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Run FlashAttention backward pass, raising on error.
  """
  @spec backward!(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def backward!(d_out, query, key, value, output, logsumexp, opts \\ []) do
    case backward(d_out, query, key, value, output, logsumexp, opts) do
      {:ok, dq, dk, dv} -> {dq, dk, dv}
      {:error, reason} -> raise "FlashAttention backward failed: #{inspect(reason)}"
    end
  end

  # NIF stub - must match Rust function name exactly
  @doc false
  def backward_f32(
        _d_out,
        _q,
        _k,
        _v,
        _output,
        _logsumexp,
        _batch,
        _seq_len,
        _num_heads,
        _head_dim,
        _causal
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Run a benchmark of the forward pass.

  ## Arguments

  * `batch` - Batch size (default: 2)
  * `seq_len` - Sequence length (default: 64)
  * `num_heads` - Number of attention heads (default: 4)
  * `head_dim` - Dimension per head (default: 64)
  * `iterations` - Number of iterations (default: 100)

  ## Returns

  `{:ok, avg_microseconds, backend}` where backend is "cuda" or "cpu"

  ## Example

      {:ok, us, backend} = ExPhil.Native.FlashAttention.benchmark(
        batch: 4, seq_len: 128, num_heads: 8, head_dim: 64, iterations: 50
      )
      IO.puts("Average: \#{us} us on \#{backend}")
  """
  @spec benchmark(keyword()) :: {:ok, float(), String.t()} | {:error, term()}
  def benchmark(opts \\ []) do
    batch = Keyword.get(opts, :batch, 2)
    seq_len = Keyword.get(opts, :seq_len, 64)
    num_heads = Keyword.get(opts, :num_heads, 4)
    head_dim = Keyword.get(opts, :head_dim, 64)
    iterations = Keyword.get(opts, :iterations, 100)

    benchmark_forward(batch, seq_len, num_heads, head_dim, iterations)
  end

  @doc false
  def benchmark_forward(_batch, _seq_len, _num_heads, _head_dim, _iterations) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # Convert Nx tensor to binary (f32, row-major)
  defp tensor_to_binary(tensor) do
    tensor
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> Nx.as_type(:f32)
    |> Nx.to_binary()
  end
end
