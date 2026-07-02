defmodule ExPhil.Training.Bf16ConvBisectTest do
  use ExUnit.Case
  @moduletag :slow
  @moduletag :gpu
  @moduletag :known_bug

  @doc """
  Bisection tests to isolate the bf16 depthwise conv SIGBUS.
  Each test isolates one variable to find the exact trigger.

  JAX 0.9.2 does NOT reproduce this — bug is EXLA-specific.
  """

  defp build_and_run(model, input_shape, input_type, steps) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    params = init_fn.(Nx.template(input_shape, input_type), Axon.ModelState.empty())

    grad_fn = Nx.Defn.jit(fn params, input ->
      loss_fn = fn p -> predict_fn.(p, %{"x" => input}) end
      Nx.Defn.value_and_grad(loss_fn).(params)
    end, compiler: EXLA, on_conflict: :reuse)

    input = Nx.broadcast(Nx.tensor(0.1, type: input_type), input_shape)
    {_loss, _grads} = grad_fn.(params, input)

    for _ <- 1..steps do
      {_loss, _grads} = grad_fn.(params, input)
    end

    :ok
  end

  defp conv_model(channels, groups, kernel_size \\ 4) do
    Axon.input("x", shape: {nil, 60, channels})
    |> Axon.conv(channels,
      kernel_size: {kernel_size},
      padding: [{kernel_size - 1, 0}],
      feature_group_size: groups,
      name: "conv"
    )
    |> Axon.nx(&Nx.mean/1, name: "reduce")
  end

  describe "control tests (f32)" do
    @tag timeout: 60_000
    test "f32 depthwise conv gradient runs 200 steps" do
      assert :ok == build_and_run(conv_model(1024, 1024), {4, 60, 1024}, :f32, 200)
    end

    @tag timeout: 60_000
    test "f32 standard conv gradient runs 200 steps" do
      assert :ok == build_and_run(conv_model(1024, 1), {4, 60, 1024}, :f32, 200)
    end
  end

  describe "bf16 without MixedPrecision" do
    @tag timeout: 60_000
    test "bf16 standard conv (groups=1) runs 200 steps" do
      assert :ok == build_and_run(conv_model(1024, 1), {4, 60, 1024}, :bf16, 200)
    end

    @tag timeout: 60_000
    test "bf16 depthwise conv (groups=1024) runs 200 steps" do
      assert :ok == build_and_run(conv_model(1024, 1024), {4, 60, 1024}, :bf16, 200)
    end
  end

  describe "feature_group_size scaling (bf16)" do
    for groups <- [1, 16, 64, 256, 512, 1024] do
      @tag timeout: 60_000
      test "bf16 conv groups=#{groups} runs 200 steps" do
        groups = unquote(groups)
        assert :ok == build_and_run(conv_model(1024, groups), {4, 60, 1024}, :bf16, 200)
      end
    end
  end

  describe "channel dimension scaling (bf16 depthwise)" do
    for channels <- [64, 128, 256, 512, 1024] do
      @tag timeout: 60_000
      test "bf16 depthwise channels=#{channels} runs 200 steps" do
        channels = unquote(channels)
        assert :ok == build_and_run(conv_model(channels, channels), {4, 60, channels}, :bf16, 200)
      end
    end
  end

  describe "batch size scaling (bf16 depthwise 1024)" do
    for batch <- [1, 4, 8, 16] do
      @tag timeout: 60_000
      test "bf16 depthwise batch=#{batch} runs 200 steps" do
        batch = unquote(batch)
        assert :ok == build_and_run(conv_model(1024, 1024), {batch, 60, 1024}, :bf16, 200)
      end
    end
  end
end
