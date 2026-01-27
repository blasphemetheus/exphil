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

  ## Usage

      # Check if NIF is available
      MambaNIF.available?()

      # Build model (same API as Mamba)
      model = MambaNIF.build(embed_size: 287, hidden_size: 256)

      # Use via --backbone mamba_nif
      mix run scripts/train_from_replays.exs --backbone mamba_nif

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

  alias ExPhil.Native.SelectiveScan

  # Default hyperparameters (same as Mamba)
  @default_hidden_size 256
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4
  @default_num_layers 2
  @default_dropout 0.0
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
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack Mamba blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_mamba_block(acc, Keyword.merge(opts, name: "mamba_nif_block_#{layer_idx}"))

        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single Mamba block with NIF-accelerated selective scan.
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    name = Keyword.get(opts, :name, "mamba_nif_block")

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    # Input normalization
    normalized = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to 2x inner_size (for x and z branches)
    xz = Axon.dense(normalized, inner_size * 2, name: "#{name}_in_proj")

    # Split into x (SSM path) and z (gating path)
    x_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
        end,
        name: "#{name}_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
        end,
        name: "#{name}_z_split"
      )

    # X branch: Depthwise Conv1D -> SiLU -> NIF Selective Scan
    x_conv = build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    # Selective SSM with NIF
    x_ssm =
      build_selective_ssm_nif(
        x_activated,
        hidden_size: inner_size,
        state_size: state_size,
        dt_rank: dt_rank,
        name: "#{name}_ssm"
      )

    # Z branch: SiLU activation (gating)
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")

    # Multiply x_ssm * z (gated output)
    gated = Axon.multiply(x_ssm, z_activated, name: "#{name}_gated")

    # Project back to hidden_size
    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

  @doc """
  Build depthwise separable 1D convolution layer (same as Mamba).
  """
  @spec build_depthwise_conv1d(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_depthwise_conv1d(input, channels, kernel_size, name) do
    Axon.nx(
      input,
      fn x ->
        batch = Nx.axis_size(x, 0)
        ch = Nx.axis_size(x, 2)

        padding = kernel_size - 1
        pad_shape = {batch, padding, ch}
        padded = Nx.concatenate([Nx.broadcast(0.0, pad_shape), x], axis: 1)

        Nx.window_mean(
          padded,
          {1, kernel_size, 1},
          strides: [1, 1, 1],
          padding: :valid
        )
      end,
      name: "#{name}_causal"
    )
    |> Axon.dense(channels, name: "#{name}_proj", use_bias: true)
  end

  @doc """
  Build the Selective SSM using the NIF-accelerated scan.

  This computes the same SSM as the pure Nx version but uses CUDA.
  """
  @spec build_selective_ssm_nif(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm_nif(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))
    name = Keyword.get(opts, :name, "ssm_nif")

    # B and C projections: [batch, seq_len, state_size] each
    bc_proj = Axon.dense(input, state_size * 2, name: "#{name}_bc_proj")

    b_matrix =
      Axon.nx(
        bc_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, state_size, axis: 2)
        end,
        name: "#{name}_B"
      )

    c_matrix =
      Axon.nx(
        bc_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, state_size, state_size, axis: 2)
        end,
        name: "#{name}_C"
      )

    # Delta (Δ) projection through low-rank bottleneck
    dt_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_dt_rank")
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_dt_softplus")

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

  @doc """
  Get the output size of a MambaNIF model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a MambaNIF model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    # Same as Mamba - the NIF only affects the scan, not the parameters
    ExPhil.Networks.Mamba.param_count(opts)
  end

  @doc """
  Get recommended defaults for Melee gameplay (60fps).
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      num_layers: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
