defmodule ExPhil.Native.FutharkScan do
  @moduledoc """
  NIF bindings to Futhark-compiled parallel prefix scan.

  Futhark compiles a parallel prefix scan (Blelloch-style) that processes
  the linear recurrence h = a*h + b in O(log T) parallel steps instead of
  O(T) sequential steps. This is fundamentally different from the CUDA C
  kernel, which uses one thread per (batch, hidden) and loops sequentially.

  ## Building

  ```bash
  cd native/futhark_scan
  make && make install
  ```

  Requires `futhark` in PATH (add to shell.nix).

  ## Numerical Note

  Parallel scan reorders floating-point operations (associativity is not exact
  for IEEE 754), so results differ from sequential scan by ~1e-4 to 1e-3.
  Use atol < 1e-3 for correctness checks.

  ## Usage

  ```elixir
  if ExPhil.Native.FutharkScan.available?() do
    result = ExPhil.Native.FutharkScan.linear_scan(a, b, h0)
  end
  ```
  """

  @on_load :load_nif

  @doc false
  def load_nif do
    nif_path = Application.app_dir(:exphil, "priv/native/libfuthark_scan_nif")

    case :erlang.load_nif(String.to_charlist(nif_path), 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} ->
        IO.warn("Failed to load futhark_scan_nif: #{inspect(reason)}")
        :ok
    end
  end

  @doc """
  Check if the Futhark NIF is loaded and working.
  """
  @spec available?() :: boolean()
  def available? do
    try do
      ping() == ~c"pong from futhark_scan_nif"
    rescue
      _ -> false
    catch
      _ -> false
    end
  end

  @doc """
  Perform parallel prefix linear scan: h = a * h + b.

  Uses Futhark's Blelloch-style parallel scan — O(log T) depth instead of
  sequential O(T). Better for long sequences, higher constant overhead.

  ## Arguments

  * `a` - Decay coefficients `[batch, seq_len, hidden]`, f32
  * `b` - Additive terms `[batch, seq_len, hidden]`, f32
  * `h0` - Initial hidden state `[batch, hidden]`, f32

  ## Returns

  Output tensor `[batch, seq_len, hidden]`, f32

  ## Numerical Precision

  Results differ from sequential scan by ~1e-3 due to floating-point
  reordering in the parallel prefix scan. This is expected and correct.
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
  def linear_scan_nif(_a, _b, _h0, _shape), do: :erlang.nif_error(:nif_not_loaded)
end
