defmodule ExPhil.Networks.MambaCumsum do
  @moduledoc """
  Mamba variant for experimenting with alternative scan algorithms.

  Currently uses Blelloch scan (same as regular Mamba). This module exists
  to test alternative approaches like:

  - **Hillis-Steele scan**: O(L log L) work but more parallelism per level
  - **SSD algorithm**: Mamba-2's chunked matmul approach for tensor cores
  - **Chunked scan**: Process in chunks with inter-chunk recurrence

  ## Current Status

  The cumsum-based approach (log-space reformulation) doesn't work well in XLA.
  XLA's cumulative_sum kernel is slower than Blelloch's pad/slice/multiply pattern
  for this tensor structure.

  ## Usage

      # Use via --backbone mamba_cumsum
      model = MambaCumsum.build(embed_size: 287, hidden_size: 256)
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4
  @default_num_layers 2
  @default_dropout 0.0
  @dt_min 0.001
  @dt_max 0.1

  @doc """
  Build a MambaCumsum model for sequence processing.

  Same API as Mamba.build/1.
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

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_mamba_block(acc, Keyword.merge(opts, name: "mamba_cumsum_block_#{layer_idx}"))
        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Extract last timestep
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
  Build a single Mamba block.
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    name = Keyword.get(opts, :name, "mamba_cumsum_block")

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    normalized = Axon.layer_norm(input, name: "#{name}_norm")
    xz = Axon.dense(normalized, inner_size * 2, name: "#{name}_in_proj")

    x_branch =
      Axon.nx(xz, fn tensor ->
        Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
      end, name: "#{name}_x_split")

    z_branch =
      Axon.nx(xz, fn tensor ->
        Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
      end, name: "#{name}_z_split")

    # X branch: Conv -> SiLU -> SSM
    x_conv = build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    scan_algo = Keyword.get(opts, :scan_algo, :blelloch)

    x_ssm =
      build_selective_ssm(
        x_activated,
        hidden_size: inner_size,
        state_size: state_size,
        dt_rank: dt_rank,
        scan_algo: scan_algo,
        name: "#{name}_ssm"
      )

    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")
    gated = Axon.multiply(x_ssm, z_activated, name: "#{name}_gated")

    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

  # Depthwise conv (same as regular Mamba)
  defp build_depthwise_conv1d(input, channels, kernel_size, name) do
    Axon.nx(
      input,
      fn x ->
        batch = Nx.axis_size(x, 0)
        ch = Nx.axis_size(x, 2)
        padding = kernel_size - 1
        pad_shape = {batch, padding, ch}
        padded = Nx.concatenate([Nx.broadcast(0.0, pad_shape), x], axis: 1)

        Nx.window_mean(padded, {1, kernel_size, 1}, strides: [1, 1, 1], padding: :valid)
      end,
      name: "#{name}_causal"
    )
    |> Axon.dense(channels, name: "#{name}_proj", use_bias: true)
  end

  @doc """
  Build the SSM with configurable scan algorithm.

  This is where we can swap in different scan implementations:
  - :blelloch (default) - Work-efficient O(L) work, O(log L) depth
  - :hillis_steele - O(L log L) work, but more parallelism per level
  - :ssd - Mamba-2's chunked matmul approach
  """
  @spec build_selective_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))
    name = Keyword.get(opts, :name, "ssm")
    scan_algo = Keyword.get(opts, :scan_algo, :blelloch)

    # B and C projections
    bc_proj = Axon.dense(input, state_size * 2, name: "#{name}_bc_proj")

    b_matrix =
      Axon.nx(bc_proj, fn tensor ->
        Nx.slice_along_axis(tensor, 0, state_size, axis: 2)
      end, name: "#{name}_B")

    c_matrix =
      Axon.nx(bc_proj, fn tensor ->
        Nx.slice_along_axis(tensor, state_size, state_size, axis: 2)
      end, name: "#{name}_C")

    # Delta projection
    dt_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_dt_rank")
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_dt_softplus")

    # Apply SSM with selected scan algorithm
    Axon.layer(
      &ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      scan_algo: scan_algo,
      op_name: :selective_ssm
    )
  end

  # SSM implementation - selects scan algorithm based on opts
  defp ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    scan_algo = opts[:scan_algo] || :blelloch

    # Clamp dt
    dt = Nx.clip(dt, @dt_min, @dt_max)

    # A matrix (negative diagonal for stability)
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))

    # Discretize: A_bar = exp(Δ * A)
    dt_expanded = Nx.new_axis(dt, 3)
    a_expanded = Nx.reshape(a_diag, {1, 1, 1, state_size})
    a_bar = Nx.exp(Nx.multiply(dt_expanded, a_expanded))

    # Discretize B: B_bar = Δ * B
    b_expanded = Nx.new_axis(b, 2)
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    dt_for_b = Nx.new_axis(dt_mean, 3)
    b_bar = Nx.multiply(dt_for_b, b_expanded)

    # Bx = B_bar * x
    x_expanded = Nx.new_axis(x, 3)
    bx = Nx.multiply(b_bar, x_expanded)

    # Apply selected scan algorithm
    h = case scan_algo do
      :cumsum_transposed -> cumsum_transposed_scan(a_bar, bx)
      :cumsum_logspace -> cumsum_logspace_scan(a_bar, bx)
      _ -> blelloch_scan(a_bar, bx)
    end

    # Output: y = C * h
    c_expanded = Nx.new_axis(c, 2)
    Nx.sum(Nx.multiply(c_expanded, h), axes: [3])
  end

  # ============================================================================
  # Scan Algorithms
  # ============================================================================

  # Blelloch parallel scan (work-efficient O(L) work, O(log L) depth)
  # Uses Enum.reduce - this lets XLA JIT each level efficiently
  defp blelloch_scan(a, b) do
    seq_len = Nx.axis_size(a, 1)

    if seq_len <= 32 do
      sequential_scan(a, b)
    else
      blelloch_scan_impl(a, b, seq_len)
    end
  end

  defp blelloch_scan_impl(a, b, seq_len) do
    log_len = ceil(:math.log2(seq_len))

    {_a_reduced, b_reduced} =
      Enum.reduce(0..(log_len - 1), {a, b}, fn level, {a_curr, b_curr} ->
        stride = round(:math.pow(2, level))

        if stride >= seq_len do
          {a_curr, b_curr}
        else
          a_shifted = Nx.pad(
            Nx.slice_along_axis(a_curr, 0, seq_len - stride, axis: 1),
            1.0,
            [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
          )
          b_shifted = Nx.pad(
            Nx.slice_along_axis(b_curr, 0, seq_len - stride, axis: 1),
            0.0,
            [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
          )

          a_new = Nx.multiply(a_curr, a_shifted)
          b_new = Nx.add(Nx.multiply(a_curr, b_shifted), b_curr)

          {a_new, b_new}
        end
      end)

    b_reduced
  end

  # Sequential scan for short sequences
  defp sequential_scan(a, b) do
    seq_len = Nx.axis_size(a, 1)
    h0 = Nx.slice_along_axis(b, 0, 1, axis: 1)

    {_, h_list} =
      Enum.reduce(1..(seq_len - 1), {h0, [Nx.squeeze(h0, axes: [1])]}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1)

        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [Nx.squeeze(h_t, axes: [1]) | acc]}
      end)

    h_list
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Experimental: Transposed Cumsum Scan
  # ============================================================================
  #
  # Hypothesis: XLA's cumsum may be faster when operating on trailing dimensions.
  # We transpose [batch, seq, hidden, state] → [batch, hidden, state, seq]
  # then run cumsum on axis 3, then transpose back.
  #
  # The SSM recurrence h_t = a_t * h_{t-1} + b_t can be computed via:
  #   h_t = sum_{i=0}^{t} (prod_{j=i+1}^{t} a_j) * b_i
  #
  # Using log-space: log(prod a) = cumsum(log(a))

  defp cumsum_transposed_scan(a, b) do
    # a, b: [batch, seq, hidden, state]
    # Transpose to put seq last: [batch, hidden, state, seq]
    a_t = Nx.transpose(a, axes: [0, 2, 3, 1])
    b_t = Nx.transpose(b, axes: [0, 2, 3, 1])

    # Work in log-space for numerical stability
    # log_a = log(a) which are negative since 0 < a < 1
    log_a = Nx.log(Nx.clip(a_t, 1.0e-8, 1.0))

    # Cumulative sum of log(a) gives log of cumulative product
    cum_log_a = Nx.cumulative_sum(log_a, axis: 3)

    # For each position t, we need sum_{i=0}^{t} exp(cum_log_a[t] - cum_log_a[i]) * b[i]
    # This is: exp(cum_log_a[t]) * sum_{i=0}^{t} exp(-cum_log_a[i]) * b[i]
    # The inner sum is a cumsum of exp(-cum_log_a) * b

    # exp(-cum_log_a[i]) * b[i]
    scaled_b = Nx.multiply(Nx.exp(Nx.negate(cum_log_a)), b_t)

    # Cumsum of scaled_b
    cum_scaled_b = Nx.cumulative_sum(scaled_b, axis: 3)

    # Multiply by exp(cum_log_a) to get h
    h_t = Nx.multiply(Nx.exp(cum_log_a), cum_scaled_b)

    # Transpose back to [batch, seq, hidden, state]
    Nx.transpose(h_t, axes: [0, 3, 1, 2])
  end

  # ============================================================================
  # Experimental: Log-space Cumsum (original axis ordering)
  # ============================================================================
  #
  # Same algorithm as transposed but without transposing - for comparison

  defp cumsum_logspace_scan(a, b) do
    # a, b: [batch, seq, hidden, state]
    # Work directly on axis 1

    log_a = Nx.log(Nx.clip(a, 1.0e-8, 1.0))
    cum_log_a = Nx.cumulative_sum(log_a, axis: 1)

    scaled_b = Nx.multiply(Nx.exp(Nx.negate(cum_log_a)), b)
    cum_scaled_b = Nx.cumulative_sum(scaled_b, axis: 1)

    Nx.multiply(Nx.exp(cum_log_a), cum_scaled_b)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    ExPhil.Networks.Mamba.param_count(opts)
  end

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
