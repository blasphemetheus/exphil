defmodule ExPhil.Native.RustLinearScan do
  @moduledoc """
  NIF bindings to Rust-CUDA linear scan kernel (h = a*h + b).

  Uses Rustler + cudarc for CUDA GPU acceleration via Rust NIF.
  This is a benchmark comparison against CUDA C (XLA), Futhark, Julia,
  and Triton implementations of the same linear scan kernel.

  ## Building the NIF

  ```bash
  cd native/rust_linear_scan_nif
  cargo build --release --features cuda
  cp target/release/librust_linear_scan_nif.so ../../priv/native/
  ```

  ## Usage

  ```elixir
  if ExPhil.Native.RustLinearScan.available?() do
    result = ExPhil.Native.RustLinearScan.linear_scan(a, b, h0)
  end
  ```
  """

  @on_load :load_nif

  @doc false
  def load_nif do
    nif_path = Application.app_dir(:exphil, "priv/native/librust_linear_scan_nif")

    case :erlang.load_nif(String.to_charlist(nif_path), 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} ->
        require Logger; Logger.debug("Optional NIF rust_linear_scan_nif not available: #{inspect(reason)}")
        :ok
    end
  end

  @doc """
  Check if the NIF is loaded and working.
  """
  @spec available?() :: boolean()
  def available? do
    try do
      ping() == "pong from rust_linear_scan_nif"
    rescue
      _ -> false
    catch
      _ -> false
    end
  end

  @doc """
  Check if CUDA is available on this system.
  """
  @spec cuda_available?() :: boolean()
  def cuda_available? do
    cuda_available()
  rescue
    _ -> false
  end

  @doc """
  Perform linear scan on GPU: h[t] = a[t] * h[t-1] + b[t].

  ## Arguments

  * `a` - Decay coefficients `[batch, seq_len, hidden]`, f32
  * `b` - Additive terms `[batch, seq_len, hidden]`, f32
  * `h0` - Initial hidden state `[batch, hidden]`, f32

  ## Returns

  Output tensor `[batch, seq_len, hidden]`, f32
  """
  @spec linear_scan(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def linear_scan(a, b, h0) do
    {batch, seq_len, hidden} = Nx.shape(a)

    a_bin = a |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()
    b_bin = b |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()
    h0_bin = h0 |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()

    result_bin = linear_scan_nif(a_bin, b_bin, h0_bin, {batch, seq_len, hidden})

    result_bin
    |> Nx.from_binary(:f32)
    |> Nx.reshape({batch, seq_len, hidden})
  end

  # NIF stubs
  @doc false
  def ping, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def cuda_available, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def linear_scan_nif(_a, _b, _h0, _shape), do: :erlang.nif_error(:nif_not_loaded)
end
