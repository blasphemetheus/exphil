#!/usr/bin/env elixir
# Minimal reproducer for bf16 depthwise conv gradient SIGBUS
#
# This script creates the exact conv configuration that crashes (feature_group_size=1024
# with bf16 precision) and runs gradients repeatedly until SIGBUS occurs.
#
# Expected behavior: crashes with SIGBUS after ~3000 iterations
# If it completes 5000 iterations without crashing, the bug may be fixed.
#
# Run: mix run scripts/repro_bf16_conv_sigbus.exs
# Run with f32 (should not crash): mix run scripts/repro_bf16_conv_sigbus.exs --f32

use_f32 = "--f32" in System.argv()
precision = if use_f32, do: :f32, else: :bf16

IO.puts("=== bf16 Depthwise Conv Gradient SIGBUS Reproducer ===")
IO.puts("  Precision: #{precision}")
IO.puts("  GPU: #{elem(ExPhil.Training.GPUUtils.get_memory_info(), 1).total_mb}MB")
IO.puts("")

# Build minimal model: just a depthwise conv + dense output
model =
  Axon.input("input", shape: {nil, 60, 1024})
  |> Axon.conv(1024,
    kernel_size: {4},
    padding: [{3, 0}],
    feature_group_size: 1024,
    name: "dw_conv"
  )
  |> Axon.nx(fn x ->
    # Take last timestep and reduce to scalar
    x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
  end, name: "reduce")

# Apply mixed precision if bf16
model =
  if precision == :bf16 do
    policy = Axon.MixedPrecision.create_policy(
      params: {:f, 32},
      compute: {:bf, 16},
      output: {:f, 32}
    )
    Axon.MixedPrecision.apply_policy(model, policy)
  else
    model
  end

# Initialize
{init_fn, predict_fn} = Axon.build(model, mode: :train)
input_template = Nx.template({16, 60, 1024}, precision)
params = init_fn.(input_template, Axon.ModelState.empty())

IO.puts("  Params initialized: #{inspect(Map.keys(params.data))}")

# Build grad function
grad_fn = Nx.Defn.jit(
  fn params, input ->
    loss_fn = fn p ->
      predict_fn.(p, %{"input" => input})
    end
    Nx.Defn.value_and_grad(loss_fn).(params)
  end,
  compiler: EXLA,
  on_conflict: :reuse
)

# Generate input
input = Nx.broadcast(Nx.tensor(0.1, type: precision), {16, 60, 1024})

# JIT warmup
IO.puts("  JIT compiling...")
{warmup_us, {loss, _grads}} = :timer.tc(fn -> grad_fn.(params, input) end)
IO.puts("  JIT done in #{div(warmup_us, 1000)}ms, loss=#{Nx.to_number(loss)}")

# Run iterations
IO.puts("\n  Running 5000 gradient iterations...")
IO.puts("  (SIGBUS expected around step ~3000 for bf16)")
IO.puts("")

for i <- 1..5000 do
  {_loss, _grads} = grad_fn.(params, input)

  if rem(i, 500) == 0 do
    IO.puts("  Step #{i}: OK")
  end
end

IO.puts("\n  Completed 5000 iterations without crash!")
IO.puts("  The bug may be fixed or not reproducible on this hardware/XLA version.")
