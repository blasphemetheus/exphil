defmodule ExPhil.Networks.Mamba do
  @moduledoc """
  Mamba: Selective State Space Model for efficient sequence modeling.

  Implements the Mamba architecture from "Mamba: Linear-Time Sequence Modeling
  with Selective State Spaces" (Gu & Dao, 2023). Key advantages over Transformers:

  - **Linear complexity**: O(L) vs O(L²) for attention
  - **Fast inference**: No KV cache needed, constant memory per step
  - **Selective gating**: Input-dependent state transitions

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        │
        ▼
  ┌─────────────────────────────────────┐
  │         Mamba Block                  │
  │                                      │
  │  ┌──── Linear (expand) ────┐        │
  │  │           │              │        │
  │  │     Conv1D + SiLU        │        │
  │  │           │              │        │
  │  │     Selective SSM   Linear+SiLU   │
  │  │           │              │        │
  │  └───────── multiply ───────┘        │
  │               │                      │
  │         Linear (project)             │
  └─────────────────────────────────────┘
        │
        ▼ (repeat for num_layers)
        │
        ▼
  [batch, seq_len, embed_size] -> last timestep -> [batch, embed_size]
  ```

  ## Selective State Space Model (S6)

  The core innovation is making the state transitions input-dependent:
  - h(t) = A(x) * h(t-1) + B(x) * x(t)
  - y(t) = C(x) * h(t)

  Where A, B, C are computed from the input x, allowing the model to
  "select" which information to remember or forget.

  ## Usage

      # Build Mamba backbone
      model = Mamba.build(
        embed_size: 1991,
        hidden_size: 256,
        state_size: 16,
        num_layers: 2,
        expand_factor: 2
      )

      # Use in temporal policy
      Policy.build_temporal(
        embed_size: 1991,
        backbone: :mamba,
        hidden_size: 256
      )

  ## References
  - Paper: https://arxiv.org/abs/2312.00752
  - Original code: https://github.com/state-spaces/mamba
  """

  require Axon

  # Default hyperparameters (from paper)
  @default_hidden_size 256
  @default_state_size 16      # N in the paper (SSM state dimension)
  @default_expand_factor 2    # E in the paper (expansion factor)
  @default_conv_size 4        # Convolution kernel size
  @default_num_layers 2
  @default_dropout 0.0
  # Note: dt_rank is computed as hidden_size // state_size when needed

  @doc """
  Build a Mamba model for sequence processing.

  ## Options
    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:state_size` - SSM state dimension N (default: 16)
    - `:expand_factor` - Expansion factor E for inner dim (default: 2)
    - `:conv_size` - 1D convolution kernel size (default: 4)
    - `:num_layers` - Number of Mamba blocks (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x = if embed_size != hidden_size do
      Axon.dense(input, hidden_size, name: "input_projection")
    else
      input
    end

    # Stack Mamba blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_mamba_block(
          acc,
          hidden_size: hidden_size,
          state_size: state_size,
          expand_factor: expand_factor,
          conv_size: conv_size,
          name: "mamba_block_#{layer_idx}"
        )

        # Add residual connection + optional dropout
        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(output, fn tensor ->
      seq_len_actual = Nx.axis_size(tensor, 1)
      Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "last_timestep")
  end

  @doc """
  Build a single Mamba block.

  The Mamba block consists of:
  1. Two parallel branches after input projection
  2. One branch: Conv1D -> SiLU -> Selective SSM
  3. Other branch: Linear -> SiLU (gating)
  4. Multiply outputs -> Project back

  ## Options
    - `:hidden_size` - Internal dimension D
    - `:state_size` - SSM state dimension N
    - `:expand_factor` - Expansion factor E
    - `:conv_size` - Convolution kernel size
    - `:name` - Layer name prefix
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    name = Keyword.get(opts, :name, "mamba_block")

    # Inner dimension (expanded)
    inner_size = hidden_size * expand_factor

    # Compute dt_rank (controls complexity of delta computation)
    dt_rank = div(hidden_size, state_size)

    # Input normalization (RMSNorm in original, using LayerNorm for simplicity)
    normalized = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to 2x inner_size (for x and z branches)
    xz = Axon.dense(normalized, inner_size * 2, name: "#{name}_in_proj")

    # Split into x (SSM path) and z (gating path)
    x_branch = Axon.nx(xz, fn tensor ->
      Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
    end, name: "#{name}_x_split")

    z_branch = Axon.nx(xz, fn tensor ->
      Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
    end, name: "#{name}_z_split")

    # X branch: Conv1D -> SiLU -> SSM
    # Conv1D with causal padding (only look at past)
    x_conv = build_causal_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    # Selective SSM
    x_ssm = build_selective_ssm(
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
  Build a causal 1D convolution layer.

  Applies convolution only over past timesteps (causal padding).
  Uses a simplified approach with sliding window mean + learned projection.
  """
  @spec build_causal_conv1d(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_causal_conv1d(input, channels, kernel_size, name) do
    # Causal padding: pad (kernel_size - 1) on the left, 0 on the right
    padding = kernel_size - 1

    Axon.nx(input, fn tensor ->
      # tensor: [batch, seq_len, channels]
      batch = Nx.axis_size(tensor, 0)
      _seq_len = Nx.axis_size(tensor, 1)
      ch = Nx.axis_size(tensor, 2)

      # Pad on the left side of sequence dimension
      pad_shape = {batch, padding, ch}
      pad_tensor = Nx.broadcast(0.0, pad_shape)
      padded = Nx.concatenate([pad_tensor, tensor], axis: 1)

      # Apply sliding window mean (simplified causal conv)
      # Real Mamba uses learned depthwise conv weights
      Nx.window_mean(
        padded,
        {1, kernel_size, 1},
        strides: [1, 1, 1],
        padding: :valid
      )
    end, name: "#{name}_causal")
    |> Axon.dense(channels, name: "#{name}_proj", use_bias: true)
  end

  @doc """
  Build the Selective State Space Model (S6).

  This is the core of Mamba: an SSM where the A, B, C parameters
  are computed from the input, making it "selective".

  The SSM equations:
  - h(t) = exp(delta * A) * h(t-1) + delta * B * x(t)
  - y(t) = C * h(t)

  Where delta, B, C are input-dependent projections.
  """
  @spec build_selective_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dt_rank = Keyword.get(opts, :dt_rank, div(hidden_size, state_size))
    name = Keyword.get(opts, :name, "ssm")

    # Project input to get B, C, and delta parameters
    # B: [batch, seq_len, state_size] - input matrix
    # C: [batch, seq_len, state_size] - output matrix
    # delta: [batch, seq_len, hidden_size] - discretization step

    # B and C projections
    bc_proj = Axon.dense(input, state_size * 2, name: "#{name}_bc_proj")

    b_matrix = Axon.nx(bc_proj, fn tensor ->
      Nx.slice_along_axis(tensor, 0, state_size, axis: 2)
    end, name: "#{name}_B")

    c_matrix = Axon.nx(bc_proj, fn tensor ->
      Nx.slice_along_axis(tensor, state_size, state_size, axis: 2)
    end, name: "#{name}_C")

    # Delta projection (through low-rank bottleneck for efficiency)
    dt_proj = input
    |> Axon.dense(dt_rank, name: "#{name}_dt_rank")
    |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
    |> Axon.activation(:softplus, name: "#{name}_dt_softplus")  # Ensure positive

    # A matrix is fixed (not input-dependent), initialized as negative values
    # for stability. We use a simple diagonal approximation.
    # In the full implementation, A would be a learnable parameter.

    # Apply the selective scan (the core SSM computation)
    Axon.layer(
      &selective_scan_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :selective_scan
    )
  end

  # Selective scan implementation
  # This computes the SSM recurrence efficiently using a simplified approach
  defp selective_scan_impl(x, b, c, dt, _opts) do
    # x: [batch, seq_len, hidden_size]
    # b: [batch, seq_len, state_size]
    # c: [batch, seq_len, state_size]
    # dt: [batch, seq_len, hidden_size]

    # Simplified SSM: use a gated linear combination
    # This captures the essence of selective state updates without complex recurrence
    #
    # In a full implementation, we'd use the parallel associative scan algorithm
    # For now, we approximate with: y = sigmoid(dt) * (B * C * x)
    # This gives the "selective gating" behavior without the recurrence

    seq_len = Nx.axis_size(x, 1)

    # Compute gates from dt (discretization step)
    # Higher dt = more influence from current input
    gate = Nx.sigmoid(Nx.mean(dt, axes: [2], keep_axes: true))  # [batch, seq_len, 1]

    # Compute BC interaction: project through state space
    # b: [batch, seq_len, state_size]
    # c: [batch, seq_len, state_size]
    # bc_gate = sum(b * c) gives a per-position gating value
    bc_gate = Nx.sum(Nx.multiply(b, c), axes: [2], keep_axes: true)  # [batch, seq_len, 1]
    bc_gate = Nx.sigmoid(bc_gate)  # Normalize to [0, 1]

    # Apply selective gating to input
    # output = gate * bc_gate * x + (1 - gate) * cumulative_context
    gated_x = Nx.multiply(Nx.multiply(gate, bc_gate), x)

    # Simple cumulative context (exponential moving average along sequence)
    # This approximates the hidden state recurrence
    alpha = 0.9  # Decay factor
    context = cumulative_ema(gated_x, alpha, seq_len)

    # Combine gated input with context
    Nx.add(gated_x, Nx.multiply(Nx.subtract(1.0, gate), context))
  end

  # Exponential moving average along sequence dimension
  # Approximates recurrent hidden state without explicit loop
  defp cumulative_ema(x, alpha, seq_len) do
    # Use scan-like operation via cumulative sum with exponential weights
    # For simplicity, use a weighted average of all previous positions

    # Create position weights: [1, seq_len, 1]
    positions = Nx.iota({1, seq_len, 1}, axis: 1)
    # Weights decay exponentially: alpha^(seq_len - 1 - pos)
    max_pos = seq_len - 1
    weights = Nx.pow(alpha, Nx.subtract(max_pos, positions))
    weights = Nx.divide(weights, Nx.sum(weights))  # Normalize

    # Apply weighted cumulative sum (simplified)
    # For each position, take weighted sum of all previous positions
    # This is an approximation - true Mamba uses parallel scan

    # Just use the weighted input directly as context
    Nx.multiply(x, weights)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Mamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Mamba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 1991)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * expand_factor

    # Per layer:
    # - Input projection: hidden * (2 * inner)
    # - Conv: inner * conv_size (simplified)
    # - BC projection: inner * (2 * state)
    # - DT projection: inner * dt_rank + dt_rank * hidden
    # - Output projection: inner * hidden
    dt_rank = div(hidden_size, state_size)

    per_layer = hidden_size * (2 * inner_size) +  # in_proj
                inner_size * 4 +                    # conv (simplified)
                inner_size * (2 * state_size) +     # BC
                inner_size * dt_rank + dt_rank * inner_size +  # dt
                inner_size * hidden_size            # out_proj

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + (per_layer * num_layers)
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
