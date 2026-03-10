defmodule ExPhil.Native.XLASelectiveScan do
  @moduledoc """
  XLA Custom Call integration for selective scan via `Nx.Shared.optional`.

  On CUDA with a patched EXLA (branch `feat/edifice-lazy-callback-allocator`
  in the local nx fork), EXLA intercepts the `:fused_selective_scan` optional
  node and emits a `stablehlo.custom_call` to the fused CUDA kernel compiled
  into libexla.so. Data stays on GPU the entire time — no CPU round-trip.

  On CPU or stock EXLA, the fallback runs the same recurrence in pure Nx
  (sequential while-loop). This is slower but numerically identical.

  ## Performance

  | Path                       | Latency (RTX 4090, B=1 T=512 H=768 S=16) |
  |----------------------------|-------------------------------------------|
  | Rust NIF (GPU→CPU→GPU)     | ~11ms                                     |
  | XLA custom call (GPU only) | ~5ms (expected)                           |
  | Pure Nx fallback           | ~50ms                                     |

  ## Requirements for custom call path

  EXLA must be built from the nx fork branch that includes:
  - `exla/c_src/exla/custom_calls/fused_selective_scan.cu` (CUDA kernel + FFI)
  - `exla/lib/exla/mlir/value.ex` — `Value.fused_selective_scan/6`
  - `exla/lib/exla/defn.ex` — `:optional` handler for `:fused_selective_scan`
  """

  import Nx.Defn

  @dt_min 0.001
  @dt_max 0.1

  @doc """
  Selective scan with automatic XLA custom call dispatch and gradient support.

  When JIT-compiled with EXLA on CUDA, the forward pass dispatches to the
  fused kernel. The backward pass uses a pure-Nx implementation (reverse-time
  sweep) via `custom_grad`.

  Gradients are computed for `x`, `dt`, `b`, and `c`. The `a` matrix is
  treated as constant (no gradient) — consistent with Mamba's log-space
  parameterization where A gradients flow through the log parameters.

  ## Parameters

  - `x` — Input tensor `[batch, seq_len, hidden]` f32
  - `dt` — Delta time `[batch, seq_len, hidden]` f32
  - `a` — State decay matrix `[hidden, state]` f32 (negative values)
  - `b` — Input projection `[batch, seq_len, state]` f32
  - `c` — Output projection `[batch, seq_len, state]` f32

  ## Returns

  Output tensor `[batch, seq_len, hidden]` f32
  """
  defn selective_scan(x, dt, a, b, c) do
    forward = selective_scan_dispatch(x, dt, a, b, c)

    Nx.Defn.Kernel.custom_grad(forward, [x, dt, b, c], fn grad_output ->
      {gx, gdt, gb, gc} = selective_scan_backward_dispatch(x, dt, a, b, c, grad_output)
      [gx, gdt, gb, gc]
    end)
  end

  @doc """
  Forward-only selective scan (no gradient). Use `selective_scan/5` for training.
  """
  defn selective_scan_forward(x, dt, a, b, c) do
    selective_scan_dispatch(x, dt, a, b, c)
  end

  deftransformp selective_scan_dispatch(x, dt, a, b, c) do
    Nx.Shared.optional(:fused_selective_scan, [x, dt, a, b, c],
      %{x | shape: Nx.shape(x), names: List.duplicate(nil, tuple_size(Nx.shape(x)))},
      fn x, dt, a, b, c ->
        selective_scan_fallback(x, dt, a, b, c)
      end)
  end

  deftransformp selective_scan_backward_dispatch(x, dt, a, b, c, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(x)
    state = elem(Nx.shape(a), 1)
    nil_names = fn n -> List.duplicate(nil, n) end

    gx_template = %{x | shape: {batch, seq_len, hidden}, names: nil_names.(3)}
    gdt_template = %{x | shape: {batch, seq_len, hidden}, names: nil_names.(3)}
    gb_template = %{x | shape: {batch, seq_len, state}, names: nil_names.(3)}
    gc_template = %{x | shape: {batch, seq_len, state}, names: nil_names.(3)}

    Nx.Shared.optional(:fused_selective_scan_backward, [x, dt, a, b, c, grad_output],
      {gx_template, gdt_template, gb_template, gc_template},
      fn x, dt, a, b, c, grad_output ->
        selective_scan_backward(x, dt, a, b, c, grad_output)
      end)
  end

  @doc """
  Pure-Nx forward selective scan matching the CUDA kernel semantics.

  Implements:
    dt_clamped = clamp(dt, 0.001, 0.1)
    h[t] = exp(dt * A) * h[t-1] + dt * B * x[t]
    y[t] = sum_s(C[t] * h[t])
  """
  defn selective_scan_fallback(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    state = Nx.axis_size(a, 1)

    dt = Nx.clip(dt, @dt_min, @dt_max)
    h = Nx.broadcast(0.0, {batch, hidden, state})
    y = Nx.broadcast(0.0, {batch, seq_len, hidden})

    {_h, y, _x, _dt, _a, _b, _c} =
      while {h, y, x, dt, a, b, c}, t <- 0..(seq_len - 1) do
        x_t = x[[.., t, ..]]
        dt_t = dt[[.., t, ..]]
        b_t = b[[.., t, ..]]
        c_t = c[[.., t, ..]]

        a_bar = Nx.exp(Nx.new_axis(dt_t, 2) * Nx.new_axis(a, 0))
        bx = Nx.new_axis(dt_t, 2) * Nx.new_axis(b_t, 1) * Nx.new_axis(x_t, 2)

        h = a_bar * h + bx
        y_t = Nx.sum(Nx.new_axis(c_t, 1) * h, axes: [2])
        y = Nx.put_slice(y, [0, t, 0], Nx.new_axis(y_t, 1))

        {h, y, x, dt, a, b, c}
      end

    y
  end

  @doc """
  Pure-Nx backward pass for selective scan.

  Recomputes forward hidden states, then sweeps backward through time
  computing gradients via the chain rule.

  Returns `{grad_x, grad_dt, grad_b, grad_c}`.
  """
  defn selective_scan_backward(x, dt, a, b, c, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(x)
    state = Nx.axis_size(a, 1)

    dt_clamped = Nx.clip(dt, @dt_min, @dt_max)

    # Forward pass saving h_prev for each timestep
    h_all = Nx.broadcast(0.0, {batch, seq_len, hidden, state})
    h = Nx.broadcast(0.0, {batch, hidden, state})

    {h_all, _h, _x, _dt_clamped, _a, _b} =
      while {h_all, h, x, dt_clamped, a, b}, t <- 0..(seq_len - 1) do
        h_all = Nx.put_slice(h_all, [0, t, 0, 0], Nx.new_axis(h, 1))

        x_t = x[[.., t, ..]]
        dt_t = dt_clamped[[.., t, ..]]
        b_t = b[[.., t, ..]]

        a_bar = Nx.exp(Nx.new_axis(dt_t, 2) * Nx.new_axis(a, 0))
        bx = Nx.new_axis(dt_t, 2) * Nx.new_axis(b_t, 1) * Nx.new_axis(x_t, 2)
        h = a_bar * h + bx

        {h_all, h, x, dt_clamped, a, b}
      end

    # Backward sweep (reverse time)
    gx = Nx.broadcast(0.0, {batch, seq_len, hidden})
    gdt = Nx.broadcast(0.0, {batch, seq_len, hidden})
    gb = Nx.broadcast(0.0, {batch, seq_len, state})
    gc = Nx.broadcast(0.0, {batch, seq_len, state})
    dh = Nx.broadcast(0.0, {batch, hidden, state})

    {gx, gdt, gb, gc, _dh, _h_all, _x, _dt, _dt_clamped, _a, _b, _c, _grad_output} =
      while {gx, gdt, gb, gc, dh, h_all, x, dt, dt_clamped, a, b, c, grad_output},
            i <- 0..(seq_len - 1) do
        t = seq_len - 1 - i

        dy = grad_output[[.., t, ..]]
        x_t = x[[.., t, ..]]
        dt_raw = dt[[.., t, ..]]
        dt_t = dt_clamped[[.., t, ..]]
        b_t = b[[.., t, ..]]
        c_t = c[[.., t, ..]]
        h_prev = h_all[[.., t, .., ..]]

        dt_exp = Nx.new_axis(dt_t, 2)
        a_exp = Nx.new_axis(a, 0)
        a_bar = Nx.exp(dt_exp * a_exp)

        # dh_total = carried dh + output gradient through C
        dh_total = dh + Nx.new_axis(dy, 2) * Nx.new_axis(c_t, 1)

        # dC = sum_h(dy * h_cur)
        h_cur = a_bar * h_prev + dt_exp * Nx.new_axis(b_t, 1) * Nx.new_axis(x_t, 2)
        gc_t = Nx.sum(Nx.new_axis(dy, 2) * h_cur, axes: [1])

        # dx = sum_s(dh * dt * B)
        gx_t = Nx.sum(dh_total * dt_exp * Nx.new_axis(b_t, 1), axes: [2])

        # ddt = sum_s(dh * h_prev * a_bar * A) + sum_s(dh * B * x)
        gdt_from_a = Nx.sum(dh_total * h_prev * a_bar * a_exp, axes: [2])
        gdt_from_b = Nx.sum(dh_total * Nx.new_axis(b_t, 1) * Nx.new_axis(x_t, 2), axes: [2])
        gdt_t = gdt_from_a + gdt_from_b
        in_range = Nx.greater_equal(dt_raw, @dt_min) and Nx.less_equal(dt_raw, @dt_max)
        gdt_t = Nx.select(in_range, gdt_t, 0.0)

        # dB = sum_h(dh * x * dt)
        gb_t = Nx.sum(dh_total * Nx.new_axis(x_t, 2) * dt_exp, axes: [1])

        # Carry dh backward
        dh = dh_total * a_bar

        gx = Nx.put_slice(gx, [0, t, 0], Nx.new_axis(gx_t, 1))
        gdt = Nx.put_slice(gdt, [0, t, 0], Nx.new_axis(gdt_t, 1))
        gb = Nx.put_slice(gb, [0, t, 0], Nx.new_axis(gb_t, 1))
        gc = Nx.put_slice(gc, [0, t, 0], Nx.new_axis(gc_t, 1))

        {gx, gdt, gb, gc, dh, h_all, x, dt, dt_clamped, a, b, c, grad_output}
      end

    {gx, gdt, gb, gc}
  end
end
