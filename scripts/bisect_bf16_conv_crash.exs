#!/usr/bin/env elixir
# Bisect the bf16 depthwise conv SIGBUS crash
# Tests each variable in isolation to find the exact trigger
#
# Run: mix run scripts/bisect_bf16_conv_crash.exs

alias ExPhil.Training.Output

Output.banner("bf16 Conv SIGBUS Bisection")

defmodule ConvTest do
  def run_test(label, model, input_type, steps) do
    IO.write(:stderr, "  #{label}...")

    try do
      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(Nx.template({4, 60, 1024}, input_type), Axon.ModelState.empty())

      grad_fn = Nx.Defn.jit(fn params, input ->
        loss_fn = fn p -> predict_fn.(p, %{"input" => input}) end
        Nx.Defn.value_and_grad(loss_fn).(params)
      end, compiler: EXLA, on_conflict: :reuse)

      input = Nx.broadcast(Nx.tensor(0.1, type: input_type), {4, 60, 1024})

      # Warmup
      {_loss, _grads} = grad_fn.(params, input)

      # Run steps
      for _ <- 1..steps do
        {_loss, _grads} = grad_fn.(params, input)
      end

      IO.puts(" OK (#{steps} steps)")
      :ok
    rescue
      e ->
        IO.puts(" FAILED: #{Exception.message(e)}")
        :error
    catch
      :exit, reason ->
        IO.puts(" CRASHED: #{inspect(reason)}")
        :crash
    end
  end
end

base_model = fn ->
  Axon.input("input", shape: {nil, 60, 1024})
  |> Axon.conv(1024,
    kernel_size: {4},
    padding: [{3, 0}],
    feature_group_size: 1024,
    name: "dw_conv"
  )
  |> Axon.nx(fn x ->
    x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
  end, name: "reduce")
end

mp_policy = Axon.MixedPrecision.create_policy(
  params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}
)

Output.puts("\n1. Control tests (should pass):")
ConvTest.run_test("f32 input, no MP", base_model.(), :f32, 200)
ConvTest.run_test("bf16 input, no MP", base_model.(), :bf16, 200)

Output.puts("\n2. MixedPrecision tests:")
mp_model = Axon.MixedPrecision.apply_policy(base_model.(), mp_policy)
ConvTest.run_test("f32 input, MP on all layers", mp_model, :f32, 200)

mp_except_conv = Axon.MixedPrecision.apply_policy(base_model.(), mp_policy, except: [:conv])
ConvTest.run_test("f32 input, MP except conv", mp_except_conv, :f32, 200)

Output.puts("\n3. Feature group size variations (with MP):")
for groups <- [1, 16, 64, 256, 512, 1024] do
  model = Axon.input("input", shape: {nil, 60, 1024})
  |> Axon.conv(1024,
    kernel_size: {4},
    padding: [{3, 0}],
    feature_group_size: groups,
    name: "conv_g#{groups}"
  )
  |> Axon.nx(fn x ->
    x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
  end, name: "reduce")
  |> Axon.MixedPrecision.apply_policy(mp_policy)

  ConvTest.run_test("groups=#{groups}", model, :f32, 200)
end

Output.puts("\n4. Batch size variations (groups=1024, MP):")
for batch <- [1, 4, 8, 16, 32] do
  model = base_model.() |> Axon.MixedPrecision.apply_policy(mp_policy)
  {init_fn, predict_fn} = Axon.build(model, mode: :train)
  params = init_fn.(Nx.template({batch, 60, 1024}, :f32), Axon.ModelState.empty())

  grad_fn = Nx.Defn.jit(fn params, input ->
    loss_fn = fn p -> predict_fn.(p, %{"input" => input}) end
    Nx.Defn.value_and_grad(loss_fn).(params)
  end, compiler: EXLA, on_conflict: :reuse)

  input = Nx.broadcast(Nx.tensor(0.1, type: :f32), {batch, 60, 1024})
  IO.write(:stderr, "  batch=#{batch}...")

  try do
    {_loss, _grads} = grad_fn.(params, input)
    for _ <- 1..200, do: {_loss, _grads} = grad_fn.(params, input)
    IO.puts(" OK")
  rescue
    e -> IO.puts(" FAILED: #{Exception.message(e)}")
  end
end

Output.puts("\n5. Channel dimension variations (depthwise, MP):")
for channels <- [64, 128, 256, 512, 1024, 2048] do
  model = Axon.input("input", shape: {nil, 60, channels})
  |> Axon.conv(channels,
    kernel_size: {4},
    padding: [{3, 0}],
    feature_group_size: channels,
    name: "conv_c#{channels}"
  )
  |> Axon.nx(fn x ->
    x |> Nx.slice_along_axis(-1, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.mean()
  end, name: "reduce")
  |> Axon.MixedPrecision.apply_policy(mp_policy)

  {init_fn, predict_fn} = Axon.build(model, mode: :train)
  params = init_fn.(Nx.template({4, 60, channels}, :f32), Axon.ModelState.empty())

  grad_fn = Nx.Defn.jit(fn params, input ->
    loss_fn = fn p -> predict_fn.(p, %{"input" => input}) end
    Nx.Defn.value_and_grad(loss_fn).(params)
  end, compiler: EXLA, on_conflict: :reuse)

  input = Nx.broadcast(Nx.tensor(0.1, type: :f32), {4, 60, channels})
  IO.write(:stderr, "  channels=#{channels}...")

  try do
    {_loss, _grads} = grad_fn.(params, input)
    for _ <- 1..200, do: {_loss, _grads} = grad_fn.(params, input)
    IO.puts(" OK")
  rescue
    e -> IO.puts(" FAILED: #{Exception.message(e)}")
  end
end

Output.puts("\n=== Done ===")
