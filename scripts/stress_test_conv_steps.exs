#!/usr/bin/env elixir
# Stress test: run bf16 MixedPrecision conv gradient for many steps
# to find the step count where it crashes
# Run: mix run scripts/stress_test_conv_steps.exs

alias ExPhil.Training.Output

Output.banner("bf16 Conv Step Count Stress Test")

mp_policy = Axon.MixedPrecision.create_policy(
  params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}
)

# The Mamba-like model that might crash
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.dense(1024, name: "in_proj")
|> Axon.conv(1024, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 1024, name: "dw_conv")
|> Axon.activation(:silu)
|> Axon.dense(288, name: "out_proj")
|> Axon.nx(&Nx.mean/1, name: "reduce")
|> Axon.MixedPrecision.apply_policy(mp_policy,
  except: [:batch_norm, :layer_norm, :group_norm]
)

{init_fn, predict_fn} = Axon.build(model, mode: :inference)
params = init_fn.(Nx.template({16, 60, 288}, :f32), Axon.ModelState.empty())

grad_fn = Nx.Defn.jit(fn params, input ->
  loss_fn = fn p -> predict_fn.(p, %{"x" => input}) end
  Nx.Defn.value_and_grad(loss_fn).(params)
end, compiler: EXLA, on_conflict: :reuse)

input = Nx.broadcast(Nx.tensor(0.1, type: :f32), {16, 60, 288})

Output.puts("JIT compiling...")
{jit_us, {loss, _}} = :timer.tc(fn -> grad_fn.(params, input) end)
Output.puts("  JIT: #{div(jit_us, 1000)}ms, loss: #{Nx.to_number(loss)}")

Output.puts("\nRunning 10000 steps (crash expected ~3000 based on training)...")
Output.puts("  Batch size: 16, channels: 1024, depthwise groups: 1024")
Output.puts("")

for i <- 1..10_000 do
  {_, _} = grad_fn.(params, input)
  if rem(i, 500) == 0 do
    IO.puts("  Step #{i}: OK")
  end
end

Output.puts("\n10000 steps completed without crash!")
