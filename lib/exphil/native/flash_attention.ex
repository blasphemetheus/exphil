defmodule ExPhil.Native.FlashAttention do
  @moduledoc """
  Native FlashAttention-2 forward pass for high-performance inference.

  This NIF provides GPU-accelerated flash attention using CUDA kernels.
  It is **forward-only** (no gradients) and intended for real-time inference,
  such as playing against Dolphin at 60 FPS.

  ## Requirements

  For GPU acceleration:
  - CUDA 12.0+ toolkit
  - Ampere+ GPU (RTX 30xx, 40xx, A100, H100)
  - Compile with `FLASH_ATTENTION_CUDA=1`

  Without CUDA, falls back to a CPU implementation that is mathematically
  equivalent but slower.

  ## Usage

      # Check if CUDA is available
      ExPhil.Native.FlashAttention.cuda_available?()

      # Run flash attention
      {:ok, output} = ExPhil.Native.FlashAttention.forward(query, key, value,
        causal: true
      )

  ## Tensor Layout

  All tensors use layout: `[batch, seq_len, num_heads, head_dim]`

  This differs from some implementations that use `[batch, num_heads, seq_len, head_dim]`.
  Ensure your tensors are in the correct layout before calling.

  ## Fallback

  When CUDA is unavailable or the NIF fails to load, use the Pure Nx
  `ExPhil.Networks.Attention.memory_efficient_attention/4` as a fallback.
  """

  use Rustler,
    otp_app: :exphil,
    crate: "flash_attention_nif",
    path: "native/flash_attention_nif"

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
