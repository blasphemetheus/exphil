defmodule ExPhil.Networks.MambaCumsum do
  @moduledoc """
  Mamba variant using log-space cumulative sum for GPU-optimized training.

  This implementation reformulates the SSM scan to use XLA's highly optimized
  cumsum primitive instead of a custom parallel scan.

  ## Mathematical Insight

  The SSM recurrence `h[t] = A[t] * h[t-1] + Bx[t]` has closed form:

  ```
  h[t] = sum_{k=0}^{t} (prod_{j=k+1}^{t} A[j]) * Bx[k]
       = P[t] * sum_{k=0}^{t} (Bx[k] / P[k])
       = P[t] * cumsum(Bx / P)[t]
  ```

  where `P[k] = prod_{j=0}^{k-1} A[j]` (exclusive cumulative product).

  This converts the parallel scan into:
  1. `log_P = exclusive_cumsum(log(A))` - one cumsum
  2. `P = exp(log_P)`
  3. `scaled = Bx / P`
  4. `h = P * cumsum(scaled)` - another cumsum

  Two cumsums replace the O(log L) level parallel scan, leveraging XLA's
  highly optimized reduction primitives.

  ## Trade-offs

  - **Faster training**: Uses XLA's fused cumsum kernels
  - **Numerical precision**: May have issues for very long sequences where P→0
  - **Same expressiveness**: Mathematically equivalent to true Mamba

  ## Usage

      # Use via --backbone mamba_cumsum
      model = MambaCumsum.build(embed_size: 287, hidden_size: 256)
  """

  import Nx.Defn
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
  Build a single Mamba block using cumsum-based SSM.
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

    # X branch: Conv -> SiLU -> Cumsum SSM
    x_conv = build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    x_ssm =
      build_cumsum_ssm(
        x_activated,
        hidden_size: inner_size,
        state_size: state_size,
        dt_rank: dt_rank,
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
  Build the SSM using cumulative sum formulation.

  Uses the identity: h[t] = P[t] * cumsum(Bx / P)[t]
  where P is the exclusive cumulative product of A.
  """
  @spec build_cumsum_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_cumsum_ssm(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))
    name = Keyword.get(opts, :name, "ssm")

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

    # Apply cumsum-based SSM
    Axon.layer(
      &cumsum_ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :cumsum_ssm
    )
  end

  # Cumsum-based SSM implementation
  # This is the key optimization - uses XLA's fused cumsum instead of parallel scan
  defp cumsum_ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]

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

    # Apply cumsum-based scan
    h = cumsum_scan(a_bar, bx)

    # Output: y = C * h
    c_expanded = Nx.new_axis(c, 2)
    y = Nx.sum(Nx.multiply(c_expanded, h), axes: [3])

    y
  end

  # The cumsum-based scan: h[t] = P[t] * cumsum(Bx / P)[t]
  # where P[k] = exclusive_cumprod(A)[k]
  defn cumsum_scan(a_bar, bx) do
    # a_bar: [batch, seq_len, hidden_size, state_size]
    # bx: [batch, seq_len, hidden_size, state_size]

    # Step 1: Compute log(A_bar) for numerical stability
    # A_bar is in (0, 1), so log is negative
    # Clamp to avoid log(0)
    a_safe = Nx.max(a_bar, 1.0e-10)
    log_a = Nx.log(a_safe)

    # Step 2: Exclusive cumulative sum of log(A) to get log(P)
    # P[k] = prod_{j=0}^{k-1} A[j], so log(P[k]) = sum_{j=0}^{k-1} log(A[j])
    # Key insight: exclusive_cumsum(x) = inclusive_cumsum(x) - x
    # This avoids dynamic slicing which kills XLA performance
    log_p_inclusive = Nx.cumulative_sum(log_a, axis: 1)
    log_p = log_p_inclusive - log_a

    # Step 3: P = exp(log_P)
    p = Nx.exp(log_p)

    # Step 4: Scaled inputs: Bx / P
    # Add small epsilon to avoid division by zero
    p_safe = Nx.max(p, 1.0e-10)
    scaled_bx = bx / p_safe

    # Step 5: Cumulative sum of scaled inputs
    cumsum_scaled = Nx.cumulative_sum(scaled_bx, axis: 1)

    # Step 6: h = P * cumsum(Bx / P)
    h = p * cumsum_scaled

    h
  end

  # ============================================================================
  # Utilities (same API as Mamba)
  # ============================================================================

  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    # Same as regular Mamba
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
