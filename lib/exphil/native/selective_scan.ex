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

  ## Training Support

  **Important:** The NIF breaks the Nx/Axon computation graph because it uses
  `Nx.to_binary()` to transfer data. This means gradients cannot flow through
  the NIF automatically.

  ### Recommended Workflow: Train with Pure Nx, Infer with NIF

  Both `ExPhil.Networks.Mamba` (pure Nx) and `ExPhil.Networks.MambaNIF` (this NIF)
  use identical layer names, so checkpoints are interchangeable:

  ```elixir
  # Train with pure Mamba (correct gradients via autodiff)
  mix run scripts/train_from_replays.exs --temporal --backbone mamba \\
    --checkpoint model.axon

  # Infer/play with MambaNIF (5x faster, same checkpoint!)
  mix run scripts/play_dolphin_async.exs --policy model.axon \\
    --backbone mamba_nif
  ```

  ### Manual Gradient Computation

  For advanced users who need to train with the NIF (e.g., for benchmarking),
  the backward kernel is available:

  ```elixir
  # Forward pass (saves hidden states for backward)
  {output, h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)

  # After computing loss and dy (output gradient)...
  # Backward pass
  {dx, d_dt, dB, dC} = SelectiveScan.backward(dy, x, h_all, dt, a, b, c)

  # Apply gradients manually (not compatible with Axon.Loop)
  ```

  ### Why Not Nx.Defn.custom_grad?

  `Nx.Defn.custom_grad` requires the forward function to be a `defn`, but NIFs
  cannot be called from inside `defn` (they require `Nx.to_binary()` which is
  not a valid defn operation). Future XLA custom call integration could solve
  this by keeping tensors on GPU throughout.

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

  @doc """
  Forward pass that saves hidden states for backward pass (training mode).

  Returns `{output, hidden_states}` where hidden_states can be passed to `backward/7`.
  """
  @spec scan_with_states(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def scan_with_states(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    {^hidden, state} = Nx.shape(a)

    x_bin = x |> Nx.as_type(:f32) |> Nx.to_binary()
    dt_bin = dt |> Nx.as_type(:f32) |> Nx.to_binary()
    a_bin = a |> Nx.as_type(:f32) |> Nx.to_binary()
    b_bin = b |> Nx.as_type(:f32) |> Nx.to_binary()
    c_bin = c |> Nx.as_type(:f32) |> Nx.to_binary()

    # NIF returns packed binary: [out, h_all]
    packed_bin =
      selective_scan_forward_with_states(x_bin, dt_bin, a_bin, b_bin, c_bin, {batch, seq_len, hidden, state})

    # Split the packed binary
    out_bytes = batch * seq_len * hidden * 4
    <<out_bin::binary-size(out_bytes), h_all_bin::binary>> = packed_bin

    out =
      Nx.from_binary(out_bin, :f32)
      |> Nx.reshape({batch, seq_len, hidden})

    h_all =
      Nx.from_binary(h_all_bin, :f32)
      |> Nx.reshape({batch, seq_len, hidden, state})

    {out, h_all}
  end

  @doc """
  Backward pass - computes gradients given output gradient and saved states.

  ## Arguments

  * `dy` - Gradient w.r.t. output `[batch, seq_len, hidden]`
  * `x` - Saved input from forward pass
  * `h_all` - Saved hidden states from `scan_with_states/5`
  * `dt` - Saved dt from forward pass
  * `a` - State transition tensor `[hidden, state]`
  * `b` - Saved B from forward pass
  * `c` - Saved C from forward pass

  ## Returns

  Tuple of `{dx, d_dt, dB, dC}` gradients.
  """
  @spec backward(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def backward(dy, x, h_all, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    {^hidden, state} = Nx.shape(a)

    dy_bin = dy |> Nx.as_type(:f32) |> Nx.to_binary()
    x_bin = x |> Nx.as_type(:f32) |> Nx.to_binary()
    h_all_bin = h_all |> Nx.as_type(:f32) |> Nx.to_binary()
    dt_bin = dt |> Nx.as_type(:f32) |> Nx.to_binary()
    a_bin = a |> Nx.as_type(:f32) |> Nx.to_binary()
    b_bin = b |> Nx.as_type(:f32) |> Nx.to_binary()
    c_bin = c |> Nx.as_type(:f32) |> Nx.to_binary()

    # NIF returns packed binary: [dx, d_dt, dB, dC]
    packed_bin =
      selective_scan_backward(dy_bin, x_bin, h_all_bin, dt_bin, a_bin, b_bin, c_bin, {batch, seq_len, hidden, state})

    # Split the packed binary
    dx_bytes = batch * seq_len * hidden * 4
    d_dt_bytes = batch * seq_len * hidden * 4
    d_b_bytes = batch * seq_len * state * 4

    <<dx_bin::binary-size(dx_bytes),
      d_dt_bin::binary-size(d_dt_bytes),
      d_b_bin::binary-size(d_b_bytes),
      d_c_bin::binary>> = packed_bin

    dx =
      Nx.from_binary(dx_bin, :f32)
      |> Nx.reshape({batch, seq_len, hidden})

    d_dt =
      Nx.from_binary(d_dt_bin, :f32)
      |> Nx.reshape({batch, seq_len, hidden})

    d_b =
      Nx.from_binary(d_b_bin, :f32)
      |> Nx.reshape({batch, seq_len, state})

    d_c =
      Nx.from_binary(d_c_bin, :f32)
      |> Nx.reshape({batch, seq_len, state})

    {dx, d_dt, d_b, d_c}
  end

  # ==========================================================================
  # Training Helpers
  # ==========================================================================

  @doc """
  Returns guidance for the recommended training workflow.

  The NIF is optimized for inference (5x faster than pure Nx), but doesn't
  support Axon's automatic differentiation. Use this to understand the
  recommended workflow.

  ## Returns

  A map with:
  - `:train_backbone` - Backbone to use for training (`:mamba`)
  - `:infer_backbone` - Backbone to use for inference (`:mamba_nif`)
  - `:checkpoint_compatible` - Whether checkpoints are interchangeable (true)
  - `:speedup` - Approximate inference speedup from NIF ("5x")
  - `:reason` - Why this workflow is recommended
  """
  @spec training_workflow_info() :: map()
  def training_workflow_info do
    %{
      train_backbone: :mamba,
      infer_backbone: :mamba_nif,
      checkpoint_compatible: true,
      speedup: "5x",
      reason: """
      The NIF uses Nx.to_binary() which breaks the computation graph.
      Gradients cannot flow through the NIF automatically.
      Train with pure Nx Mamba (:mamba), then switch to NIF (:mamba_nif) for inference.
      Both use identical layer names, so checkpoints work with either.
      """
    }
  end

  @doc """
  Check if the current context is suitable for training.

  Returns `false` because the NIF breaks the gradient computation graph.
  Use `ExPhil.Networks.Mamba` for training instead.
  """
  @spec supports_training?() :: boolean()
  def supports_training?, do: false

  @doc """
  Validates that the user is using the correct backbone for their use case.

  Raises if using NIF for training (which won't work correctly).

  ## Arguments

  - `mode` - `:training` or `:inference`

  ## Examples

      # In training script
      SelectiveScan.validate_mode!(:training)  # Raises with guidance

      # In inference script
      SelectiveScan.validate_mode!(:inference)  # OK
  """
  @spec validate_mode!(atom()) :: :ok
  def validate_mode!(:inference), do: :ok

  def validate_mode!(:training) do
    raise ArgumentError, """
    SelectiveScan NIF does not support training mode!

    The NIF breaks the Nx/Axon computation graph, so gradients cannot
    flow through it automatically.

    Recommended workflow:
    1. Train with --backbone mamba (pure Nx, supports autodiff)
    2. Infer with --backbone mamba_nif (5x faster)

    Checkpoints are compatible between both backbones.

    If you need to benchmark training with the NIF, use:
    - scan_with_states/5 for forward pass
    - backward/7 for gradient computation
    - Apply gradients manually (not with Axon.Loop)
    """
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

  @doc false
  # Returns packed binary: [out, h_all]
  def selective_scan_forward_with_states(_x, _dt, _a, _b, _c, _shape), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  # Returns packed binary: [dx, d_dt, dB, dC]
  def selective_scan_backward(_dy, _x, _h_all, _dt, _a, _b, _c, _shape), do: :erlang.nif_error(:nif_not_loaded)
end
