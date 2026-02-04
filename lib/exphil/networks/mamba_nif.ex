defmodule ExPhil.Networks.MambaNIF do
  @moduledoc """
  Mamba with CUDA-accelerated selective scan via Rust NIF.

  This module is identical to `ExPhil.Networks.Mamba` in structure, but uses
  the fast CUDA kernel from `ExPhil.Native.SelectiveScan` instead of the
  pure Nx/XLA parallel scan.

  ## Performance

  On RTX 4090:
  - Pure Nx Mamba: ~55ms inference
  - MambaNIF: ~10.96ms inference (5x faster, 60 FPS capable)

  ## Recommended Workflow: Train with Mamba, Infer with MambaNIF

  The NIF breaks the computation graph (gradients can't flow through
  `Nx.to_binary`), so use pure Mamba for training. Both modules use
  identical layer names, so checkpoints are interchangeable:

      # Train with pure Mamba (correct gradients)
      mix run scripts/train_from_replays.exs --temporal --backbone mamba \\
        --checkpoint model.axon

      # Infer/play with MambaNIF (5x faster, same checkpoint!)
      mix run scripts/play_dolphin_async.exs --policy model.axon \\
        --backbone mamba_nif

  ## Direct Usage

      # Check if NIF is available
      MambaNIF.available?()

      # Build model (same API as Mamba)
      model = MambaNIF.build(embed_size: 287, hidden_size: 256)

  ## Fallback

  If the NIF is not available (not compiled, no CUDA), this module will
  raise at runtime. Use `available?/0` to check before building.

  ## Implementation Notes

  The NIF expects these tensor shapes:
  - x: [batch, seq_len, hidden]
  - dt: [batch, seq_len, hidden]
  - A: [hidden, state] - differs from pure Nx which uses [state]
  - B: [batch, seq_len, state]
  - C: [batch, seq_len, state]
  """

  require Axon

  alias ExPhil.Networks.Mamba.Common
  alias ExPhil.Native.SelectiveScan

  @dt_min 0.001
  @dt_max 0.1

  @doc """
  Check if the NIF is available and CUDA is working.
  """
  @spec available?() :: boolean()
  def available? do
    SelectiveScan.available?() and SelectiveScan.cuda_available?()
  end

  @doc """
  Build a Mamba model using the NIF-accelerated selective scan.

  Same options as `ExPhil.Networks.Mamba.build/1`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  @doc """
  Build a single Mamba block with NIF-accelerated selective scan.
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts \\ []) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    # Use same layer names as pure Mamba for checkpoint compatibility
    name = Keyword.get(opts, :name, "mamba_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm_nif/2)
  end

  @doc """
  Build depthwise separable 1D convolution layer.
  """
  @spec build_depthwise_conv1d(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defdelegate build_depthwise_conv1d(input, channels, kernel_size, name), to: Common

  @doc """
  Build the Selective SSM using the NIF-accelerated scan.

  This computes the same SSM as the pure Nx version but uses CUDA.
  """
  @spec build_selective_ssm_nif(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm_nif(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    # Use same default name as pure Mamba for checkpoint compatibility
    name = Keyword.get(opts, :name, "ssm")

    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    # Apply the NIF-accelerated scan
    Axon.layer(
      &nif_scan_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :nif_selective_scan
    )
  end

  # NIF scan implementation
  # Calls the Rust CUDA kernel instead of pure Nx scan
  defp nif_scan_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    hidden_size = opts[:hidden_size]

    # x: [batch, seq_len, hidden_size]
    # b: [batch, seq_len, state_size]
    # c: [batch, seq_len, state_size]
    # dt: [batch, seq_len, hidden_size]

    # Clamp dt to reasonable range
    dt = Nx.clip(dt, @dt_min, @dt_max)

    # Create A matrix: [hidden_size, state_size]
    # NIF expects 2D A matrix, not 1D vector
    # Use negative values for stability (same as pure Mamba)
    a_matrix =
      Nx.iota({hidden_size, state_size})
      |> Nx.remainder(state_size)
      |> Nx.add(1.0)
      |> Nx.negate()

    # Call NIF
    # The NIF handles discretization (A_bar = exp(dt * A)) internally
    SelectiveScan.scan(x, dt, a_matrix, b, c)
  end

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @doc """
  Get the output size of a MambaNIF model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @doc """
  Calculate approximate parameter count for a MambaNIF model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  defdelegate param_count(opts), to: Common

  @doc """
  Get recommended defaults for Melee gameplay (60fps).
  """
  @spec melee_defaults() :: keyword()
  defdelegate melee_defaults(), to: Common
end
