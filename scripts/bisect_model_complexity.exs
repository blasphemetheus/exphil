#!/usr/bin/env elixir
# Bisect: at what model complexity does bf16 MixedPrecision crash?
# Tests progressively larger models to find the threshold.
# Run: mix run scripts/bisect_model_complexity.exs

alias ExPhil.Training.Output

Output.banner("Model Complexity Bisection")

mp_policy = Axon.MixedPrecision.create_policy(
  params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}
)

defmodule ModelTest do
  def test(label, model, shape, steps, mp_policy) do
    IO.write(:stderr, "  #{label}...")

    mp_model = Axon.MixedPrecision.apply_policy(model, mp_policy,
      except: [:batch_norm, :layer_norm, :group_norm]
    )

    {init_fn, predict_fn} = Axon.build(mp_model, mode: :inference)
    params = init_fn.(Nx.template(shape, :f32), Axon.ModelState.empty())

    grad_fn = Nx.Defn.jit(fn params, input ->
      loss_fn = fn p -> predict_fn.(p, %{"x" => input}) end
      Nx.Defn.value_and_grad(loss_fn).(params)
    end, compiler: EXLA, on_conflict: :reuse)

    input = Nx.broadcast(Nx.tensor(0.1, type: :f32), shape)

    {jit_us, {loss, _}} = :timer.tc(fn -> grad_fn.(params, input) end)
    IO.write(:stderr, " JIT #{div(jit_us, 1000)}ms, loss=#{Float.round(Nx.to_number(loss), 4)}...")

    for i <- 1..steps do
      {_, _} = grad_fn.(params, input)
      if rem(i, 100) == 0, do: IO.write(:stderr, " #{i}")
    end
    IO.puts(" OK (#{steps} steps)")
    :ok
  rescue
    e ->
      IO.puts(" ERROR: #{Exception.message(e) |> String.slice(0, 80)}")
      :error
  end
end

shape = {4, 60, 288}
steps = 500

Output.puts("\n1. Single layers with MixedPrecision (#{steps} steps each):")

# Just dense
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.dense(512, name: "dense1")
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("dense only", model, shape, steps, mp_policy)

# Just conv (standard, groups=1)
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.conv(288, kernel_size: {4}, padding: [{3, 0}], name: "conv")
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("standard conv (groups=1)", model, shape, steps, mp_policy)

# Just depthwise conv
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.conv(288, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 288, name: "dw_conv")
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("depthwise conv (groups=288)", model, shape, steps, mp_policy)

Output.puts("\n2. Increasing model depth with MixedPrecision:")

# Dense + depthwise conv
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.dense(512, name: "proj")
|> Axon.conv(512, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 512, name: "dw_conv")
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("dense → depthwise conv", model, shape, steps, mp_policy)

# Dense + depthwise conv + dense (Mamba-like)
model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.dense(1024, name: "in_proj")
|> Axon.conv(1024, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 1024, name: "dw_conv")
|> Axon.activation(:silu)
|> Axon.dense(288, name: "out_proj")
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("dense → dw_conv(1024) → silu → dense", model, shape, steps, mp_policy)

# Two layers of the above (like 2-layer Mamba)
build_layer = fn input, i ->
  input
  |> Axon.dense(1024, name: "layer#{i}_in_proj")
  |> Axon.conv(1024, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 1024, name: "layer#{i}_dw_conv")
  |> Axon.activation(:silu)
  |> Axon.dense(288, name: "layer#{i}_out_proj")
end

model = Axon.input("x", shape: {nil, 60, 288})
|> build_layer.(1)
|> build_layer.(2)
|> Axon.nx(&Nx.mean/1, name: "reduce")
ModelTest.test("2-layer Mamba-like (dense→dw_conv→silu→dense)×2", model, shape, steps, mp_policy)

Output.puts("\n3. Scale up channel dimension:")

for channels <- [512, 1024, 2048] do
  model = Axon.input("x", shape: {nil, 60, 288})
  |> Axon.dense(channels, name: "proj")
  |> Axon.conv(channels, kernel_size: {4}, padding: [{3, 0}], feature_group_size: channels, name: "dw_conv")
  |> Axon.activation(:silu)
  |> Axon.dense(288, name: "out")
  |> Axon.nx(&Nx.mean/1, name: "reduce")
  ModelTest.test("channels=#{channels}", model, shape, steps, mp_policy)
end

Output.puts("\n4. Without MixedPrecision (bf16 input, f32 model):")

model = Axon.input("x", shape: {nil, 60, 288})
|> Axon.dense(1024, name: "in_proj")
|> Axon.conv(1024, kernel_size: {4}, padding: [{3, 0}], feature_group_size: 1024, name: "dw_conv")
|> Axon.activation(:silu)
|> Axon.dense(288, name: "out_proj")
|> Axon.nx(&Nx.mean/1, name: "reduce")

{init_fn, predict_fn} = Axon.build(model, mode: :inference)
params = init_fn.(Nx.template({4, 60, 288}, :bf16), Axon.ModelState.empty())
grad_fn = Nx.Defn.jit(fn params, input ->
  loss_fn = fn p -> predict_fn.(p, %{"x" => input}) end
  Nx.Defn.value_and_grad(loss_fn).(params)
end, compiler: EXLA, on_conflict: :reuse)

input = Nx.broadcast(Nx.tensor(0.1, type: :bf16), {4, 60, 288})
IO.write(:stderr, "  bf16 input, no MP policy...")
{_, _} = grad_fn.(params, input)
for i <- 1..steps do
  {_, _} = grad_fn.(params, input)
  if rem(i, 100) == 0, do: IO.write(:stderr, " #{i}")
end
IO.puts(" OK")

Output.puts("\n5. Full Mamba model via Edifice (the actual crash scenario):")
try do
  # Build actual Mamba model
  mamba_model = Edifice.SSM.Common.build_model([
    embed_dim: 288,
    hidden_size: 512,
    state_size: 16,
    expand_factor: 2,
    conv_size: 4,
    num_layers: 2,
    dropout: 0.0,
    window_size: 60
  ])
  |> Axon.nx(&Nx.mean/1, name: "reduce_final")

  mp_mamba = Axon.MixedPrecision.apply_policy(mamba_model, mp_policy,
    except: [:batch_norm, :layer_norm, :group_norm]
  )

  ModelTest.test("Full Mamba (2-layer, hidden=512)", mp_mamba, shape, steps, mp_policy)
rescue
  e -> IO.puts("  Full Mamba build failed: #{Exception.message(e) |> String.slice(0, 100)}")
end

Output.puts("\n=== Done ===")
