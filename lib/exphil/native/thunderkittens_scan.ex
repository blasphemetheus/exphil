defmodule ExPhil.Native.ThunderKittensScan do
  @moduledoc """
  NIF interface to ThunderKittens-style CUDA linear scan kernel.

  ThunderKittens (Stanford HazyResearch) is a CUDA-embedded C++ DSL for AI kernels.
  It requires sm_80+ (Ampere) for core functionality. On sm_75 (Turing/T400),
  this falls back to a standard CUDA kernel with the same algorithm.

  ## Usage

      ExPhil.Native.ThunderKittensScan.available?()
      result = ExPhil.Native.ThunderKittensScan.linear_scan(a, b, h0)

  ## Build

      cd native/thunderkittens_scan
      make && make install
  """

  @on_load :load_nif

  def load_nif do
    nif_path = Application.app_dir(:exphil, "priv/native/libthunderkittens_scan_nif")

    case :erlang.load_nif(String.to_charlist(nif_path), 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} ->
        IO.warn("Failed to load ThunderKittens NIF: #{inspect(reason)}")
        :ok
    end
  end

  @doc "Check if the ThunderKittens NIF is loaded and functional."
  def available? do
    try do
      ping() == ~c"pong from thunderkittens_scan_nif"
    rescue
      _ -> false
    catch
      _ -> false
    end
  end

  @doc "Returns device info: {:ok, name, {sm_major, sm_minor}, tk_capable}"
  def device_info do
    device_info_nif()
  end

  @doc "Check if GPU supports ThunderKittens (sm_80+)."
  def tk_capable? do
    case device_info_nif() do
      {:ok, _name, {major, _minor}, _capable} -> major >= 8
      _ -> false
    end
  end

  @doc """
  Perform fused linear scan: h[t] = a[t] * h[t-1] + b[t]

  ## Arguments

  * `a` - Decay coefficients `[batch, seq_len, hidden]`, f32
  * `b` - Additive terms `[batch, seq_len, hidden]`, f32
  * `h0` - Initial hidden state `[batch, hidden]`, f32

  ## Returns

  Nx tensor `[batch, seq_len, hidden]` with scan results.
  """
  def linear_scan(a, b, h0) do
    {batch, seq_len, hidden} = Nx.shape(a)

    a_bin = a |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()
    b_bin = b |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()
    h0_bin = h0 |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()

    case linear_scan_nif(a_bin, b_bin, h0_bin, {batch, seq_len, hidden}) do
      {:error, reason} ->
        raise "ThunderKittens scan failed: #{inspect(reason)}"

      result_bin when is_binary(result_bin) ->
        result_bin
        |> Nx.from_binary(:f32)
        |> Nx.reshape({batch, seq_len, hidden})
    end
  end

  # NIF stubs
  def ping, do: :erlang.nif_error(:nif_not_loaded)
  def device_info_nif, do: :erlang.nif_error(:nif_not_loaded)
  def linear_scan_nif(_a, _b, _h0, _shape), do: :erlang.nif_error(:nif_not_loaded)
end
