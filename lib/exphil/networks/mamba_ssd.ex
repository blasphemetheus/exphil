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

  alias ExPhil.Networks.Mamba.Common

  @default_chunk_size 16

  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  defp build_mamba_block(input, opts) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba_ssd_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm/2)
  end

  defp build_selective_ssm(input, opts) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    name = Keyword.get(opts, :name, "ssm")

    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

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

    # Discretize SSM parameters
    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

    # Use SSD algorithm
    h = ssd_scan(a_bar, bx, chunk_size)

    # Compute output
    Common.compute_ssm_output(h, c)
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
      Common.sequential_scan(a, b)
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
      Common.sequential_scan(a_chunk, b_chunk)
    else
      # Use Blelloch for intra-chunk (it's efficient for small sequences)
      Common.blelloch_scan(a_chunk, b_chunk)
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

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @spec param_count(keyword()) :: non_neg_integer()
  defdelegate param_count(opts), to: Common

  @doc """
  Get recommended defaults for Melee gameplay (60fps).

  Includes SSD-specific `chunk_size` option.
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    Common.melee_defaults()
    |> Keyword.put(:chunk_size, 16)
  end
end
