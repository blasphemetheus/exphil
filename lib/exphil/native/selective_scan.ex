defmodule ExPhil.Native.SelectiveScan do
  @moduledoc """
  NIF bindings to CUDA-accelerated selective scan kernel.

  This module provides a fast GPU implementation of the Mamba SSM selective scan
  operation, written in Rust with CUDA via cudarc.

  ## Building the NIF

  ```bash
  cd native/selective_scan_nif
  cargo build --release --features cuda
  ```

  The compiled library will be at `native/selective_scan_nif/target/release/libselective_scan_nif.so`

  ## Usage

  ```elixir
  # Check if CUDA is available
  ExPhil.Native.SelectiveScan.cuda_available?()

  # Run selective scan
  result = ExPhil.Native.SelectiveScan.scan(x, dt, a, b, c)
  ```

  ## Fallback

  If the NIF is not available, functions will raise. Use `available?/0` to check
  before calling.
  """

  @on_load :load_nif

  @doc false
  def load_nif do
    nif_path = Application.app_dir(:exphil, "priv/native/libselective_scan_nif")

    case :erlang.load_nif(String.to_charlist(nif_path), 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} ->
        IO.warn("Failed to load selective_scan_nif: #{inspect(reason)}")
        :ok  # Don't fail app startup
    end
  end

  @doc """
  Check if the NIF is loaded and working.
  """
  def available? do
    try do
      ping() == "pong from selective_scan_nif"
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
  Get CUDA device info string.
  """
  @spec device_info() :: {:ok, String.t()} | {:error, term()}
  def device_info do
    {:ok, cuda_device_info()}
  rescue
    e -> {:error, e}
  end

  @doc """
  Perform selective scan on GPU.

  ## Arguments

  * `x` - Input tensor `[batch, seq_len, hidden]`, f32
  * `dt` - Delta/timestep tensor `[batch, seq_len, hidden]`, f32
  * `a` - State transition tensor `[hidden, state]`, f32
  * `b` - Input projection tensor `[batch, seq_len, state]`, f32
  * `c` - Output projection tensor `[batch, seq_len, state]`, f32

  ## Returns

  Output tensor `[batch, seq_len, hidden]`, f32
  """
  @spec scan(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def scan(x, dt, a, b, c) do
    # Get shapes
    {batch, seq_len, hidden} = Nx.shape(x)
    {^hidden, state} = Nx.shape(a)

    # Convert to binaries (ensure f32 and contiguous)
    x_bin = x |> Nx.as_type(:f32) |> Nx.to_binary()
    dt_bin = dt |> Nx.as_type(:f32) |> Nx.to_binary()
    a_bin = a |> Nx.as_type(:f32) |> Nx.to_binary()
    b_bin = b |> Nx.as_type(:f32) |> Nx.to_binary()
    c_bin = c |> Nx.as_type(:f32) |> Nx.to_binary()

    # Call NIF
    result_bin = selective_scan(x_bin, dt_bin, a_bin, b_bin, c_bin, {batch, seq_len, hidden, state})

    # Convert back to tensor
    Nx.from_binary(result_bin, :f32)
    |> Nx.reshape({batch, seq_len, hidden})
  end

  # NIF stubs - these get replaced when the NIF loads
  # Using :erlang.nif_error ensures proper NIF behavior

  @doc false
  def ping, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def cuda_available, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def cuda_device_info, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def selective_scan(_x, _dt, _a, _b, _c, _shape), do: :erlang.nif_error(:nif_not_loaded)
end
