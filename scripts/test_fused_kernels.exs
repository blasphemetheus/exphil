# Test script: Verify fused CUDA kernel integration via stablehlo.custom_call
#
# Usage: EDIFICE_LOCAL_NX=1 mix run scripts/test_fused_kernels.exs
#
# Tests:
#   1. custom_call_available?() returns true
#   2. MinGRU fused kernel produces correct output vs Nx fallback
#   3. stablehlo.custom_call appears in XLA IR (optional, with XLA_FLAGS)

alias Edifice.CUDA.FusedScan

IO.puts("=== Fused CUDA Kernel Integration Test ===\n")

# ── Test 1: Availability ──────────────────────────────────────────────────────

IO.write("1. custom_call_available?() ... ")

available? =
  try do
    exla_value = Module.concat([EXLA, MLIR, Value])
    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :custom_call_fused, 4)
  rescue
    _ -> false
  end

if available? do
  IO.puts("PASS (custom_call_fused/4 exported)")
else
  IO.puts("FAIL")
  IO.puts("   EXLA.MLIR.Value.custom_call_fused/4 not found.")
  IO.puts("   Make sure you're using the local nx/exla fork (EDIFICE_LOCAL_NX=1).")
  System.halt(1)
end

# ── Test 2: Numerical correctness ────────────────────────────────────────────

IO.write("2. MinGRU numerical correctness ... ")

# Create test tensors on BinaryBackend first (deterministic, no GPU allocation noise)
batch = 2
seq_len = 4
hidden = 8

{gates_raw, key} =
  Nx.with_default_backend(Nx.BinaryBackend, fn ->
    key = Nx.Random.key(42)
    Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
  end)

{candidates, _key} =
  Nx.with_default_backend(Nx.BinaryBackend, fn ->
    Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
  end)

# Reference implementation: pure Nx scan on BinaryBackend
ref_result =
  Nx.with_default_backend(Nx.BinaryBackend, fn ->
    z = Nx.sigmoid(gates_raw)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, hidden})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end)

# Run through FusedScan.mingru inside JIT context (required for Nx.Shared.optional)
# This will trace the function, see the :optional node, and emit stablehlo.custom_call
fused_result =
  try do
    jitted = Nx.Defn.jit(fn g, c -> FusedScan.mingru(g, c) end, compiler: EXLA)
    jitted.(gates_raw, candidates)
  rescue
    e ->
      IO.puts("FAIL")
      IO.puts("   Error: #{Exception.message(e)}")
      IO.puts("   #{Exception.format(:error, e, __STACKTRACE__)}")
      System.halt(1)
  end

# Compare
ref_flat = Nx.to_flat_list(ref_result)
fused_flat = Nx.to_flat_list(fused_result)

max_diff =
  Enum.zip(ref_flat, fused_flat)
  |> Enum.map(fn {a, b} -> abs(a - b) end)
  |> Enum.max()

if max_diff < 1.0e-4 do
  IO.puts("PASS (max_diff=#{Float.round(max_diff, 8)})")
else
  IO.puts("FAIL (max_diff=#{max_diff})")
  IO.puts("   Reference: #{inspect(Enum.take(ref_flat, 8))}")
  IO.puts("   Fused:     #{inspect(Enum.take(fused_flat, 8))}")
  System.halt(1)
end

# ── Test 3: Check which dispatch path was used ────────────────────────────────

IO.write("3. Dispatch path check ... ")

dispatch =
  cond do
    available? and EXLA.Client.fetch!(:default).platform == :cuda ->
      "custom_call (stablehlo.custom_call → CUDA kernel)"
    true ->
      "fallback (pure Nx scan)"
  end

IO.puts("#{dispatch}")

# ── Test 4: Verify fused handlers in libexla.so ──────────────────────────────

IO.write("4. Registered FFI handlers ... ")

libexla_path =
  :code.priv_dir(:exla)
  |> to_string()
  |> Path.join("libexla.so")

if File.exists?(libexla_path) do
  {output, 0} = System.cmd("strings", [libexla_path])

  fused_handlers =
    output
    |> String.split("\n")
    |> Enum.filter(&String.starts_with?(&1, "exla_fused_"))
    |> length()

  IO.puts("#{fused_handlers} handlers in libexla.so")
else
  IO.puts("SKIP (libexla.so not found at #{libexla_path})")
end

# ── Test 5: Backward pass (gradient flow through fused CUDA kernel) ──────────

IO.write("5. MinGRU backward pass ... ")

# Loss function: sum of forward outputs (simple scalar loss for gradient test)
# Gradients flow: dL/d(output) = ones → backward kernel → grad_z, grad_cand → chain rule → grad_gates
loss_fn = fn gates, cand ->
  FusedScan.mingru(gates, cand) |> Nx.sum()
end

# Compute gradients through the fused CUDA path
{fused_loss, fused_grads} =
  try do
    grad_fn = Nx.Defn.jit(
      fn g, c -> Nx.Defn.value_and_grad(g, fn g -> loss_fn.(g, c) end) end,
      compiler: EXLA
    )
    grad_fn.(gates_raw, candidates)
  rescue
    e ->
      IO.puts("FAIL")
      IO.puts("   Error: #{Exception.message(e)}")
      IO.puts("   #{Exception.format(:error, e, __STACKTRACE__)}")
      System.halt(1)
  end

IO.puts("computed (loss=#{Nx.to_number(fused_loss) |> Float.round(6)})")

# ── Test 6: Gradient numerical correctness via finite differences ────────────

IO.write("6. Gradient correctness (finite diff) ... ")

epsilon = 1.0e-3

# Compute forward loss with EXLA for a given perturbation
fwd_fn = Nx.Defn.jit(fn g, c -> FusedScan.mingru(g, c) |> Nx.sum() end, compiler: EXLA)

# Pick a few random indices to spot-check gradients
grad_flat = Nx.to_flat_list(fused_grads)
gates_flat = Nx.to_flat_list(gates_raw)
total_elems = batch * seq_len * hidden

# Check 8 evenly-spaced indices
check_indices = for i <- 0..7, do: div(i * (total_elems - 1), 7)

max_grad_diff = 0.0

max_grad_diff =
  Enum.reduce(check_indices, 0.0, fn idx, max_diff ->
    # Perturb gates[idx] by +epsilon and -epsilon
    gates_plus = Nx.indexed_put(
      Nx.reshape(gates_raw, {total_elems}),
      Nx.tensor([[idx]]),
      Nx.tensor([Enum.at(gates_flat, idx) + epsilon])
    ) |> Nx.reshape({batch, seq_len, hidden})

    gates_minus = Nx.indexed_put(
      Nx.reshape(gates_raw, {total_elems}),
      Nx.tensor([[idx]]),
      Nx.tensor([Enum.at(gates_flat, idx) - epsilon])
    ) |> Nx.reshape({batch, seq_len, hidden})

    loss_plus = fwd_fn.(gates_plus, candidates) |> Nx.to_number()
    loss_minus = fwd_fn.(gates_minus, candidates) |> Nx.to_number()

    numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
    analytical_grad = Enum.at(grad_flat, idx)

    diff = abs(numerical_grad - analytical_grad)
    max(max_diff, diff)
  end)

if max_grad_diff < 1.0e-2 do
  IO.puts("PASS (max_diff=#{Float.round(max_grad_diff, 6)} across #{length(check_indices)} indices)")
else
  IO.puts("FAIL (max_diff=#{Float.round(max_grad_diff, 6)})")
  System.halt(1)
end

# ── Test 7: Gradient correctness vs Nx fallback ─────────────────────────────

IO.write("7. Gradient correctness (vs Nx fallback) ... ")

# Compute gradients using pure Nx fallback (no CUDA)
# Force fallback by using BinaryBackend inside the grad computation
ref_grad_fn = fn gates, cand ->
  z = Nx.sigmoid(gates)
  h0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(gates)), {batch, hidden})

  {_, h_list} =
    Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
      z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
      {h_t, [h_t | acc]}
    end)

  h_list |> Enum.reverse() |> Nx.stack(axis: 1) |> Nx.sum()
end

{ref_loss, ref_grads} =
  Nx.Defn.jit(
    fn g, c -> Nx.Defn.value_and_grad(g, fn g -> ref_grad_fn.(g, c) end) end,
    compiler: EXLA
  ).(gates_raw, candidates)

ref_grad_flat = Nx.to_flat_list(ref_grads)

grad_vs_ref_diff =
  Enum.zip(grad_flat, ref_grad_flat)
  |> Enum.map(fn {a, b} -> abs(a - b) end)
  |> Enum.max()

if grad_vs_ref_diff < 1.0e-4 do
  IO.puts("PASS (max_diff=#{Float.round(grad_vs_ref_diff, 8)})")
else
  IO.puts("FAIL (max_diff=#{grad_vs_ref_diff})")
  IO.puts("   Fused grad (first 8): #{inspect(Enum.take(grad_flat, 8))}")
  IO.puts("   Ref   grad (first 8): #{inspect(Enum.take(ref_grad_flat, 8))}")
  System.halt(1)
end

IO.puts("\n=== All tests passed ===")
