defmodule ExPhil.Networks.MambaSSD do
  @moduledoc """
  Mamba variant using State Space Duality (SSD) algorithm from Mamba-2.

  ## SSD Algorithm

  The key insight: SSM computation can be decomposed into matrix multiplications
  that leverage tensor cores (10-20x faster than scalar operations).

  ### Algorithm Steps

  1. **Split into chunks**: Divide sequence into chunks of size C
  2. **Intra-chunk (matmul)**: Compute outputs within each chunk using dense matmul
     - This uses tensor cores!
     - O(C²) work per chunk, but highly parallel
  3. **Inter-chunk (scan)**: Small sequential scan over chunk boundaries
     - Only L/C elements to scan
  4. **Combine**: Merge chunk outputs with inter-chunk states

  ### Complexity

  - Intra-chunk: O(L/C × C²) = O(L × C) work, but tensor core accelerated
  - Inter-chunk: O(L/C) sequential work (tiny)
  - Total: Much faster in practice due to tensor cores

  ## Current Performance

  **WARNING:** This implementation is ~10-15x slower than Blelloch scan in XLA.

  Why it's slow:
  - Each `Enum.map` creates separate XLA computation graphs (no fusion)
  - Chunking creates many small tensor operations (high dispatch overhead)
  - XLA can't optimize the inter-chunk recurrence loop
  - The algorithm is designed for fused CUDA kernels, not XLA primitives

  SSD would be fast with:
  - A single fused Triton/CUDA kernel for the whole scan
  - Custom XLA operation that handles chunking internally
  - Tensor parallelism that XLA can fuse

  This implementation exists for algorithmic correctness testing, not performance.

  ## Usage

      model = MambaSSD.build(embed_size: 287, hidden_size: 256, chunk_size: 16)
  """

  require Axon

  @default_hidden_size 256
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4
  @default_num_layers 2
  @default_dropout 0.0
  @default_chunk_size 16
  @dt_min 0.001
  @dt_max 0.1

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
        block = build_mamba_block(acc, Keyword.merge(opts, name: "mamba_ssd_block_#{layer_idx}"))
        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

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

  defp build_mamba_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    name = Keyword.get(opts, :name, "mamba_ssd_block")

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

    x_conv = build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    x_ssm =
      build_selective_ssm(
        x_activated,
        hidden_size: inner_size,
        state_size: state_size,
        dt_rank: dt_rank,
        chunk_size: chunk_size,
        name: "#{name}_ssm"
      )

    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")
    gated = Axon.multiply(x_ssm, z_activated, name: "#{name}_gated")

    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

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

  defp build_selective_ssm(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    name = Keyword.get(opts, :name, "ssm")

    bc_proj = Axon.dense(input, state_size * 2, name: "#{name}_bc_proj")

    b_matrix =
      Axon.nx(bc_proj, fn tensor ->
        Nx.slice_along_axis(tensor, 0, state_size, axis: 2)
      end, name: "#{name}_B")

    c_matrix =
      Axon.nx(bc_proj, fn tensor ->
        Nx.slice_along_axis(tensor, state_size, state_size, axis: 2)
      end, name: "#{name}_C")

    dt_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_dt_rank")
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_dt_softplus")

    Axon.layer(
      &ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      chunk_size: chunk_size,
      op_name: :ssd_ssm
    )
  end

  defp ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    chunk_size = opts[:chunk_size] || @default_chunk_size

    dt = Nx.clip(dt, @dt_min, @dt_max)
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))

    dt_expanded = Nx.new_axis(dt, 3)
    a_expanded = Nx.reshape(a_diag, {1, 1, 1, state_size})
    a_bar = Nx.exp(Nx.multiply(dt_expanded, a_expanded))

    b_expanded = Nx.new_axis(b, 2)
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    dt_for_b = Nx.new_axis(dt_mean, 3)
    b_bar = Nx.multiply(dt_for_b, b_expanded)

    x_expanded = Nx.new_axis(x, 3)
    bx = Nx.multiply(b_bar, x_expanded)

    # Use SSD algorithm
    h = ssd_scan(a_bar, bx, chunk_size)

    c_expanded = Nx.new_axis(c, 2)
    Nx.sum(Nx.multiply(c_expanded, h), axes: [3])
  end

  # ============================================================================
  # SSD (State Space Duality) Algorithm
  # ============================================================================
  #
  # From Mamba-2: converts SSM scan into chunked matmul operations

  defp ssd_scan(a, b, chunk_size) do
    # a, b: [batch, seq_len, hidden_size, state_size]
    seq_len = Nx.axis_size(a, 1)

    # If sequence fits in one chunk, use simple scan
    if seq_len <= chunk_size do
      simple_sequential_scan(a, b)
    else
      chunked_ssd_scan(a, b, chunk_size, seq_len)
    end
  end

  defp chunked_ssd_scan(a, b, chunk_size, seq_len) do
    batch = Nx.axis_size(a, 0)
    hidden = Nx.axis_size(a, 2)
    state = Nx.axis_size(a, 3)

    # Number of full chunks
    num_chunks = div(seq_len, chunk_size)
    remainder = rem(seq_len, chunk_size)

    # Process full chunks
    chunk_outputs =
      Enum.map(0..(num_chunks - 1), fn chunk_idx ->
        start_idx = chunk_idx * chunk_size

        # Extract chunk
        a_chunk = Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
        b_chunk = Nx.slice_along_axis(b, start_idx, chunk_size, axis: 1)

        # Intra-chunk computation using matmul-style approach
        # For each position t in chunk, output depends on positions 0..t
        # This is a lower-triangular matmul pattern
        intra_chunk_scan(a_chunk, b_chunk)
      end)

    # Handle remainder if any
    chunk_outputs =
      if remainder > 0 do
        start_idx = num_chunks * chunk_size
        a_rem = Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
        b_rem = Nx.slice_along_axis(b, start_idx, remainder, axis: 1)
        chunk_outputs ++ [intra_chunk_scan(a_rem, b_rem)]
      else
        chunk_outputs
      end

    # Compute final state of each chunk for inter-chunk propagation
    chunk_final_states =
      Enum.map(chunk_outputs, fn chunk_h ->
        # Final state is last element of chunk
        chunk_len = Nx.axis_size(chunk_h, 1)
        Nx.slice_along_axis(chunk_h, chunk_len - 1, 1, axis: 1)
      end)

    # Compute cumulative products of A for inter-chunk state propagation
    # NOTE: Currently unused - the inter-chunk propagation recomputes A products inline
    # TODO: Optimize by using precomputed chunk_a_products instead of recomputing
    _chunk_a_products =
      Enum.map(0..(length(chunk_outputs) - 1), fn chunk_idx ->
        if chunk_idx == length(chunk_outputs) - 1 and remainder > 0 do
          start_idx = num_chunks * chunk_size
          a_chunk = Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
          Nx.product(a_chunk, axes: [1])
        else
          start_idx = chunk_idx * chunk_size
          a_chunk = Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
          Nx.product(a_chunk, axes: [1])
        end
      end)

    # Inter-chunk state propagation
    # h_chunk[i] = h_intra[i] + A_prod[i] * h_final[i-1] + A_prod[i] * A_prod[i-1] * h_final[i-2] + ...
    # This is a small scan over chunk boundaries

    {_, propagated_outputs} =
      Enum.reduce(
        Enum.with_index(chunk_outputs),
        {Nx.broadcast(0.0, {batch, 1, hidden, state}), []},
        fn {chunk_h, idx}, {running_state, acc} ->
          # Add contribution from previous chunks to this chunk's output
          # running_state: [batch, 1, hidden, state] - accumulated state from previous chunks

          chunk_len = Nx.axis_size(chunk_h, 1)

          # Compute A products for each position in chunk relative to chunk start
          if idx == 0 do
            # First chunk: no inter-chunk contribution
            new_running = Enum.at(chunk_final_states, idx)
            {new_running, acc ++ [chunk_h]}
          else
            # Get A for this chunk to propagate running state
            a_chunk =
              if idx == length(chunk_outputs) - 1 and remainder > 0 do
                start_idx = num_chunks * chunk_size
                Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
              else
                start_idx = idx * chunk_size
                Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
              end

            # Compute cumulative A products within chunk for state propagation
            # For position t: multiply running_state by prod(A[0..t])
            a_cumprods = compute_cumulative_products(a_chunk)

            # Propagate running state through chunk
            state_contribution = Nx.multiply(a_cumprods, running_state)

            # Add to intra-chunk output
            adjusted_chunk = Nx.add(chunk_h, state_contribution)

            # Update running state: final state of this chunk
            chunk_final = Nx.slice_along_axis(adjusted_chunk, chunk_len - 1, 1, axis: 1)

            {chunk_final, acc ++ [adjusted_chunk]}
          end
        end
      )

    # Concatenate all chunk outputs
    Nx.concatenate(propagated_outputs, axis: 1)
  end

  # Intra-chunk scan using efficient pattern
  # For small chunks, we can compute all pairwise dependencies efficiently
  defp intra_chunk_scan(a_chunk, b_chunk) do
    chunk_len = Nx.axis_size(a_chunk, 1)

    if chunk_len <= 4 do
      # Very small: just do sequential
      simple_sequential_scan(a_chunk, b_chunk)
    else
      # Use Blelloch for intra-chunk (it's efficient for small sequences)
      blelloch_scan(a_chunk, b_chunk, chunk_len)
    end
  end

  # Compute cumulative product along sequence dimension
  # Returns tensor where position t contains prod(a[0..t])
  defp compute_cumulative_products(a) do
    seq_len = Nx.axis_size(a, 1)

    {_, cumprods} =
      Enum.reduce(0..(seq_len - 1), {nil, []}, fn t, {prev_prod, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)

        new_prod =
          if prev_prod == nil do
            a_t
          else
            Nx.multiply(prev_prod, a_t)
          end

        {new_prod, acc ++ [new_prod]}
      end)

    Nx.concatenate(cumprods, axis: 1)
  end

  # Simple sequential scan for small sequences or fallback
  defp simple_sequential_scan(a, b) do
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

  # Blelloch scan for intra-chunk
  defp blelloch_scan(a, b, seq_len) do
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
      chunk_size: 16,
      dropout: 0.1
    ]
  end
end
